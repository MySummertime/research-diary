# --- coding: utf-8 ---
# --- app/core/evaluator.py ---
import math
import networkx as nx
from typing import Dict, Any, Tuple, List
from .network import TransportNetwork, Node, Arc
from .solution import Solution

class Evaluator:
    """
    评估器 (大脑)。
    """

    def __init__(self, network: TransportNetwork, config: Dict[str, Any]):
        """
        初始化评估器。
        适配嵌套的 config 字典。
        """
        self.network = network
        self.config = config    # 存储完整的 config
        
        # 提取各自关心的配置分组，提高内聚性
        self.risk_config = self.config.get("risk_model_f1", {})
        self.cost_config = self.config.get("cost_model_f2", {})
        # (应急车辆速度在 risk_config 中)

        self.emergency_times: Dict[str | Tuple[str, str], float] = {}
        self._precompute_emergency_response_times()

    def evaluate(self, solution: Solution):
        """
        [主方法] 评估一个解 (Solution) 的所有属性。
        """
        
        # 1. 计算目标函数 f2 (期望成本)
        solution.f2_cost = self._calculate_f2_expected_cost(solution)
        
        # 2. 计算目标函数 f1 (CVaR 风险)
        # 这个函数会计算并填充 solution.eta_values
        solution.f1_risk = self._calculate_f1_cvar_risk(solution)
        
        # 3. 检查所有约束 (容量 + 模糊成本预算)
        is_feasible, violation = self._check_constraints(solution)
        solution.is_feasible = is_feasible
        solution.constraint_violation = violation

    # --- 目标函数 f2 (成本) ---

    def _calculate_f2_expected_cost(self, solution: Solution) -> float:
        """
        计算 f2: 总期望成本 (运输 + 转运 + 碳排放)。
        """
        total_cost = 0.0
        
        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue
            demand = path.task.demand   # 运量 d_v
            
            # 1. 运输成本
            for arc in path.arcs:
                t1, t2, t3 = arc.fuzzy_transport_time
                expected_time = (t1 + 2 * t2 + t3) / 4.0
                total_cost += arc.shipment_cost * expected_time * demand
            
            # 2. 碳排放成本
            for arc in path.arcs:
                total_cost += arc.carbon_cost_per_ton_km * arc.length * demand
            
            # 3. 转运成本
            for hub in path.transfer_hubs:
                s1, s2, s3, s4 = hub.fuzzy_transshipment_time
                expected_time = (s1 + s2 + s3 + s4) / 4.0
                total_cost += hub.transshipment_cost * expected_time * demand
                
        return total_cost

    # --- 目标函数 f1 (风险) ---

    def _calculate_f1_cvar_risk(self, solution: Solution) -> float:
        """
        计算 f1: 所有任务的 CVaR 风险之和。
        [逻辑]
        1. 遍历每个任务。
        2. 收集该任务路径上所有的 (p, c) 事件对。
        3. 为该任务计算最优的 eta* (即 VaR_alpha)。
        4. 使用该 eta* 计算 CVaR_alpha。
        5. 将 eta* 存回 solution.eta_values 以供日志记录。
        """
        total_risk = 0.0
        alpha = self.risk_config.get("cvar_alpha", 0.95)
        
        if alpha >= 1.0 or alpha <= 0.0:
            return float('inf') 
            
        one_minus_alpha = 1.0 - alpha
        
        # 清空旧的计算结果
        solution.eta_values = {}

        for task_id, path in solution.path_selections.items():
            
            # 1. 收集 (p, c) 事件对
            p_c_pairs: List[Tuple[float, float]] = []
            
            # 1a. 收集弧段风险
            for arc in path.arcs:
                p_ijm = arc.accident_prob_per_km * arc.length
                c_ijm = self._get_dynamic_consequence(arc)
                if p_ijm > 0 and c_ijm > 0:
                    p_c_pairs.append((p_ijm, c_ijm))
                
            # 1b. 收集枢纽风险
            for hub in path.transfer_hubs:
                p_k = hub.accident_prob
                c_k = self._get_dynamic_consequence(hub)
                if p_k > 0 and c_k > 0:
                    p_c_pairs.append((p_k, c_k))

            # 2. 计算此任务的最优 eta* 和 CVaR
            if not p_c_pairs:
                # 这条路径没有风险
                optimal_eta_v = 0.0
                task_cvar = 0.0
            else:
                optimal_eta_v, task_cvar = self._find_optimal_eta_and_cvar(p_c_pairs, alpha, one_minus_alpha)
            
            # 3. 存回 solution 作为“计算结果”
            solution.eta_values[task_id] = optimal_eta_v
            
            # 4. 累加到总风险
            total_risk += task_cvar
            
        return total_risk
    
    def _find_optimal_eta_and_cvar(self, 
                                 p_c_pairs: List[Tuple[float, float]], 
                                 alpha: float, 
                                 one_minus_alpha: float) -> Tuple[float, float]:
        """
        [辅助函数]
        为一条路径（由 (p, c) 对列表表示）计算最优的 eta* (即 VaR) 和 CVaR。
        CVaR(X) = VaR_alpha(X) + (1 / (1-alpha)) * E[ (X - VaR_alpha(X))^+ ]
        最优的 eta* 就是 VaR_alpha(X)。
        """
        
        # 步骤 1: 按后果 c 升序排序
        # (p, c) -> (c, p)
        sorted_c_p_pairs = sorted([(c, p) for p, c in p_c_pairs])
        
        # 步骤 2: 找到 VaR (最优的 eta)
        # 需要找到最小的 c_i，使得 "超过它的概率" <= (1 - alpha)
        
        total_prob = sum(p for c, p in sorted_c_p_pairs)
        if total_prob <= one_minus_alpha:
            # 极端情况：所有事故概率之和都小于 (1-alpha)
            # 这意味着 VaR(alpha) 为 0 (因为 P(X > 0) <= 1-alpha)
            optimal_eta_v = 0.0
        else:
            prob_sum = 0.0
            optimal_eta_v = sorted_c_p_pairs[-1][0] # 默认为最大后果
            
            for i in range(len(sorted_c_p_pairs)):
                c_i, p_i = sorted_c_p_pairs[i]
                
                # prob_sum 是 P(X <= c_i)
                prob_sum += p_i
                
                # P(X > c_i) = total_prob - prob_sum
                prob_exceeding = total_prob - prob_sum
                
                if prob_exceeding <= one_minus_alpha:
                    # 找到的 c_i 就是第一个满足条件的 VaR(alpha)
                    optimal_eta_v = c_i
                    break
        
        # 步骤 3: 计算 CVaR
        # CVaR = eta* + (1 / (1-alpha)) * E[ (X - eta*)^+ ]
        # E[ (X - eta*)^+ ] = Σ p_i * max(0, c_i - eta*)
        
        expected_loss_over_eta = 0.0
        for p_i, c_i in p_c_pairs:
            loss = max(0.0, c_i - optimal_eta_v)
            expected_loss_over_eta += p_i * loss
            
        task_cvar = optimal_eta_v + (expected_loss_over_eta / one_minus_alpha)
        
        return optimal_eta_v, task_cvar

    def _get_dynamic_consequence(self, entity: Arc | Node) -> float:
        """
        [辅助方法] 计算 "时间依赖的事故后果"。
        从 self.risk_config 读取。
        """
        lambda_m = self.risk_config.get("accident_lambda", 100.0) 
        gamma = self.risk_config.get("accident_gamma", 0.1)     
        q = self.risk_config.get("accident_q", 5.0)           
        
        if isinstance(entity, Arc):
            t_e = self.emergency_times.get((entity.start.id, entity.end.id), float('inf'))
        else:
            t_e = self.emergency_times.get(entity.id, float('inf'))
            
        if t_e == float('inf'):
            return float('inf')
            
        a_spread = gamma * q * t_e
        
        if isinstance(entity, Arc):
            d_ij = entity.length
            rho_ij = entity.population_density
            a_base = (2.0 * lambda_m * d_ij) + (math.pi * lambda_m**2)
            consequence = (a_base + a_spread) * rho_ij
        else:
            rho_k = entity.population_density
            a_base = math.pi * lambda_m**2
            consequence = (a_base + a_spread) * rho_k
            
        return consequence

    # --- 约束检查 (可行性) ---

    def _check_constraints(self, solution: Solution) -> Tuple[bool, float]:
        """
        检查 1) 容量约束 和 2) 模糊成本约束。
        从 self.cost_config 读取。
        """
        total_violation = 0.0
        
        # --- 1. 容量约束 (Arc & Node) ---
        arc_flow: Dict[Tuple[str, str], float] = {}
        node_flow: Dict[str, float] = {}
        
        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue
            demand = path.task.demand
            
            # 累加弧段流量
            for arc in path.arcs:
                arc_key = (arc.start.id, arc.end.id)
                arc_flow[arc_key] = arc_flow.get(arc_key, 0.0) + demand
            
            # 累加枢纽流量
            for hub in path.transfer_hubs:
                node_flow[hub.id] = node_flow.get(hub.id, 0.0) + demand
        
        # 检查弧段容量
        for arc_key, flow in arc_flow.items():
            arc = self.network._arcs_dict.get(arc_key) 
            if arc and flow > arc.capacity:
                violation = (flow - arc.capacity) / arc.capacity if arc.capacity > 0 else flow
                total_violation += violation
        
        # 检查枢纽容量
        for node_id, flow in node_flow.items():
            node = self.network._nodes_dict.get(node_id) 
            if node and flow > node.capacity:
                violation = (flow - node.capacity) / node.capacity if node.capacity > 0 else flow
                total_violation += violation
                
        # --- 2. 模糊成本可靠性约束 ---
        pessimistic_cost = 0.0

        alpha_c = self.cost_config.get("fuzzy_cost_alpha_c", 0.90) 
        bgt = self.cost_config.get("fuzzy_cost_budget", 5000000)     
        
        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue
            demand = path.task.demand
            
            # 1. 运输成本 (悲观值)
            for arc in path.arcs:
                t1, t2, t3 = arc.fuzzy_transport_time
                pess_time = self._get_triangular_pessimistic_value(t1, t2, t3, alpha_c)
                pessimistic_cost += arc.shipment_cost * pess_time * demand
            
            # 2. 碳排放成本 (确定值)
            for arc in path.arcs:
                pessimistic_cost += arc.carbon_cost_per_ton_km * arc.length * demand
            
            # 3. 转运成本 (悲观值)
            for hub in path.transfer_hubs:
                s1, s2, s3, s4 = hub.fuzzy_transshipment_time
                pess_time = self._get_trapezoidal_pessimistic_value(s1, s2, s3, s4, alpha_c)
                pessimistic_cost += hub.transshipment_cost * pess_time * demand

        # 检查成本约束
        if pessimistic_cost > bgt:
            cost_violation = (pessimistic_cost - bgt) / bgt if bgt > 0 else pessimistic_cost
            total_violation += cost_violation
            
        is_feasible = (total_violation == 0.0)
        return is_feasible, total_violation

    # --- 预计算和数学辅助方法 ---
    
    def _precompute_emergency_response_times(self):
        """
        [辅助方法] 预计算应急响应时间。
        从 self.risk_config 读取。
        """
        print("开始预计算应急响应时间...")
        
        road_graph = nx.DiGraph()
        speed_v = self.risk_config.get("emergency_vehicle_speed", 45.0)
        
        for arc in self.network.arcs:
            if arc.mode == 'road':
                travel_time = float('inf') if speed_v == 0 else (arc.length / speed_v)
                road_graph.add_edge(arc.start.id, arc.end.id, weight=travel_time)
        
        center_ids = [node.id for node in self.network.get_emergency_centers()]
        
        if not center_ids:
            print("警告: 未找到任何应急中心！所有应急响应时间将为 'inf'。")
            shortest_times = {}
        else:
            try:
                shortest_times = nx.multi_source_dijkstra_path_length(road_graph.reverse(), sources=center_ids)
            except (ImportError, nx.NetworkXError):
                print("警告: multi_source_dijkstra 不可用或图未连接，回退到慢速计算。")
                shortest_times = {}
                for node in self.network.nodes:
                    min_time = float('inf')
                    for center_id in center_ids:
                        try:
                            time = nx.dijkstra_path_length(road_graph, source=node.id, target=center_id)
                            if time < min_time:
                                min_time = time
                        except nx.NetworkXNoPath:
                            continue
                    shortest_times[node.id] = min_time

        # 4. 存储节点响应时间 t_ke
        for node in self.network.nodes:
            self.emergency_times[node.id] = shortest_times.get(node.id, float('inf'))
            
        # 5. 存储弧段响应时间 t_ije (均值法)
        for arc in self.network.arcs:
            time_i = self.emergency_times.get(arc.start.id, float('inf'))
            time_j = self.emergency_times.get(arc.end.id, float('inf'))
            
            avg_time = float('inf')
            if time_i != float('inf') and time_j != float('inf'):
                avg_time = (time_i + time_j) / 2.0
                
            self.emergency_times[(arc.start.id, arc.end.id)] = avg_time
            
        print("应急响应时间预计算完成。")
        
    def _get_triangular_pessimistic_value(self, a: float, b: float, c: float, alpha_c: float) -> float:
        """
        计算三角模糊数 ξ=(a,b,c) 的 α_c-悲观值。
        """
        if alpha_c <= 0.5:
            return (2.0 * alpha_c) * b + (1.0 - 2.0 * alpha_c) * a
        else:
            return (2.0 * alpha_c - 1.0) * c + (2.0 - 2.0 * alpha_c) * b

    def _get_trapezoidal_pessimistic_value(self, a: float, b: float, c: float, d: float, alpha_c: float) -> float:
        """
        计算梯形模糊数 ζ=(a,b,c,d) 的 α_c-悲观值。
        """
        if alpha_c <= 0.5:
            return (2.0 * alpha_c) * b + (1.0 - 2.0 * alpha_c) * a
        else:
            return (2.0 * alpha_c - 1.0) * d + (2.0 - 2.0 * alpha_c) * c