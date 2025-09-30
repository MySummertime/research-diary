# --- coding: utf-8 ---
# --- problem.py ---
import numpy as np
import math
import networkx as nx
from scipy.stats import norm    # type: ignore
from collections import defaultdict
from typing import List, Tuple
from pymoo.core.problem import Problem  # type: ignore
from core.network import Node, Arc, TransportTask, TransportNetwork

class HazmatProblem(Problem):
    def __init__(self, network: TransportNetwork):
        """
        初始化危险品运输优化问题。

        Args:
            network (TransportNetwork): 包含节点、弧段、任务等信息的网络实例。
        """
        # --- 问题的元数据 ---
        self.network = network
        self.tasks = network.tasks
        self.hubs = network.get_hubs()

        # --- 问题的参数 ---
        # 1. 定义模型参数
        self.params = {
            'psi': 0.75,    # 不确定需求的置信水平
            'alpha': 0.95,  # CVaR 置信水平
            'q_leak_rate': 0.5, # 危险品泄漏速率 (kg/s)
            'gamma_diffusion_coeff': 10.0, # 扩散系数 (m^2/kg)
            'h_structural_aversion': 1.0,   # 结构稳健性偏好系数 h
            'k_steepness': 10,   # Sigmoid 函数的陡峭系数
            't_threshold': 0.2, # Sigmoid 函数的阈值
            'lambda_m': 500,    # 危险品事故影响半径 (米)
            'pbm_road': 1.6e-8, # 公路每公里单位事故概率
            'pbm_rail': 0.5e-8, # 铁路每公里单位事故概率 (通常低于公路)
            'pk_hub': 1e-6  # 枢纽单位事故概率
        }

        # --- 预计算与辅助数据结构 ---
        # 这些是为了加速计算而准备的，属于“问题求解”的范畴
        
        # 预先计算应急中心的位置，以加速计算
        self.emergency_centers = self.network.get_emergency_centers()
        
        # 预先构建辅助数据结构
        self._build_graphs_and_dicts()

        # =========================================================================
        #   ↓↓↓ 创建 self.feasible_hubs 属性 的代码 ↓↓↓
        # =========================================================================
        print("\n--- [预计算] 正在为每个任务计算所有可行的枢纽组合... ---")
        self.feasible_hubs = defaultdict(list)
        num_hubs = len(self.hubs)

        for task in self.tasks:
            feasible_count = 0
            # 遍历所有可能的双枢纽组合 (k != l)
            for k_idx in range(num_hubs):
                for l_idx in range(num_hubs):
                    if k_idx == l_idx:
                        continue # 强制双枢纽，跳过单枢纽
                    
                    # 调用 _decode_path 检查该组合是否能形成有效路径
                    path_arcs, _ = self._decode_path(task, (k_idx, l_idx))
                    
                    # 如果返回的路径不为空，说明该枢纽组合是可行的
                    if path_arcs:
                        self.feasible_hubs[task.id].append((k_idx, l_idx))
                        feasible_count += 1
            
            # 增加一个检查，如果某个任务没有任何可行的双枢纽组合，给出警告
            if feasible_count == 0:
                print(f"  - 警告：任务 {task.id} 未找到任何可行的双枢纽组合！")
            else:
                print(f"  - 任务 {task.id}: 找到了 {feasible_count} 个可行的枢纽组合。")
        
        print("--- [预计算] 完成！---\n")
        # =========================================================================
        #   ↑↑↑ 以上是 feasible_hubs 的核心代码 ↑↑↑
        # =========================================================================

        # --- Pymoo 元数据定义 ---

        # 决策变量：每个任务选择 {弧段, 枢纽}，决策变量数 == 任务数 * 2
        n_var = len(self.tasks) * 2
        
        # 目标函数：f1 (风险) 和 f2 (成本)
        n_obj = 2
        
        # 此处只定义【显式约束】的数量
        #   注：显式约束即：每个弧段的容量约束 + 每个枢纽的容量约束
        # 注意: 其他约束 (如流量守恒、转运次数等) 由 _decode_path 解码器在结构上隐式强制执行。
        n_constr = len(network.arcs) + len(network.nodes)
        
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         xl=0,  # 枢纽索引的下界 >= 0
                         xu=len(self.hubs) - 1 if self.hubs else 0, # 枢纽索引的上界 <= num_hub - 1
                         vtype=int) # 决策变量为整数
        

    def _evaluate(self, x, out, *args, **kwargs):
        """
        风险评估函数的核心实现。
        它接收整个种群的决策变量 x，并为每个个体计算目标函数值 F 和约束违反值 G
        采用“单一约束惩罚”策略，以确保算法在惩罚不可行解的同时，
        能够维持种群多样性，避免过早收敛.
        """
        # x 是一个 (n_pop, n_var) 的 NumPy 矩阵，代表整个种群
        # n_pop 是种群大小, n_var 是决策变量数量
        
        # 准备用于存放最终结果的空矩阵
        # F 用于存放目标函数值，G 用于存放约束违反值
        objectives_F = np.zeros((x.shape[0], self.n_obj))
        constraints_G = np.zeros((x.shape[0], self.n_constr))
        base_penalty_value = 1e5    # 定义一个基础惩罚值

        # --- 遍历种群中的每一个“解决方案” ---
        for i, individual_x in enumerate(x):
            # 将一维的决策变量数组，变回 V x 2 的形状
            solution_hubs = individual_x.reshape((len(self.tasks), 2))

            # --- 初始化当前解决方案的累加器 ---
            solution_total_f1_sp_cvar = 0.0
            solution_total_f2_cost = 0.0
            invalid_task_count = 0  # 用于记录无效任务的数量
            arc_flows = defaultdict(float)
            node_flows = defaultdict(float)
            
            # --- 遍历该解决方案中的每一个运输任务 ---
            for v_idx, task in enumerate(self.tasks):
                # 1. 传递枢纽对给解码器
                chosen_hub_indices = tuple(solution_hubs[v_idx])
                path_arcs, path_nodes = self._decode_path(task, chosen_hub_indices)

                # 如果解码失败（例如路径不存在/无效），则增加计数并立即跳过此任务的后续计算
                if not path_arcs:
                    invalid_task_count += 1
                    continue    # 跳到下一个任务
                
                # 2. 计算确定性等价需求
                equivalent_demand = self._calculate_deterministic_demand(task)

                # 3. 计算风险和成本
                # 综合风险值
                sp_cvar_for_task_v, base_risk_for_path_v = self._calculate_path_sp_cvar_and_base_risk(path_arcs, path_nodes, task)
                solution_total_f1_sp_cvar += sp_cvar_for_task_v

                # 综合成本
                cost_for_task_v = self._calculate_path_cost(path_arcs, path_nodes, equivalent_demand)
                solution_total_f2_cost += cost_for_task_v

                # 4. 累加流量
                for arc in path_arcs:
                    arc_flows[(arc.start.id, arc.end.id)] += equivalent_demand
                # 使用解码器返节点列表
                for node in path_nodes:
                    node_flows[node.id] += equivalent_demand
            
            # --- 循环结束后，根据无效任务数计算最终目标值和约束 ---
            # 1. 无论解是否可行，都为其目标函数F赋“真实”计算值
            #    这保留了“部分优秀”的不可行解的潜在价值，是维持多样性的关键。
            objectives_F[i, 0] = solution_total_f1_sp_cvar
            objectives_F[i, 1] = solution_total_f2_cost

            # 2. 计算总的约束违反值 G
            #    首先计算基础的容量约束违反
            total_violation = 0.0
            violations_list = [] # 用于构建最终的G向量
            for arc in self.network.arcs:
                flow = arc_flows.get((arc.start.id, arc.end.id), 0)
                violations_list.append(max(0, flow - arc.capacity))
            for node in self.network.nodes:
                flow = node_flows.get(node.id, 0)
                violations_list.append(max(0, flow - node.capacity))
            
            total_violation = sum(violations_list)
            
            #    如果存在无效任务（解码失败），则在容量违反的基础上增加一个大的惩罚值
            #    这是“单一约束惩罚”的核心：只惩罚G，不污染F。
            if invalid_task_count > 0:
                # 这里的惩罚值可以根据目标函数的数量级进行调整
                # 1e5 是一个常用的、足够大的值
                total_violation += invalid_task_count * base_penalty_value
            
            # 3. 将约束违反值赋给 G
            #    Pymoo 会利用这个 G 值来判断解是否可行并进行支配排序
            #    可以将总违反值集中在第一个约束上，或者分散开
            
            # 方案A: 集中在第一个约束上 (更简单)
            final_violations = np.zeros(self.n_constr)
            final_violations[0] = total_violation
            constraints_G[i, :] = final_violations
            
            # 方案B: 分散到各自的约束上 (更精细，但这里意义不大，因为我们混合了惩罚)
            # constraints_G[i, :] = np.array(violations_list)
            # if invalid_task_count > 0:
            #     constraints_G[i, 0] += invalid_task_count * base_penalty_value
        
        # --- 将所有解的结果一次性赋值给 out 字典 ---
        out["F"] = objectives_F
        out["G"] = constraints_G


    def _build_graphs_and_dicts(self):
        """
        [辅助方法] 构建 networkx 图和字典，以加速路径查找和数据访问。
        """
        # 创建两个空的 networkx 图，一个用于公路，一个用于铁路
        self._road_graph = nx.DiGraph()
        self._rail_graph = nx.DiGraph()
        self._full_graph = nx.DiGraph()
        
        # 将所有节点添加到图中
        for node in self.network.nodes:
            self._road_graph.add_node(node.id)
            self._rail_graph.add_node(node.id)
            self._full_graph.add_node(node.id)
        
        # 将所有弧段添加到图中
        for arc in self.network.arcs:
            start_id = arc.start.id
            end_id = arc.end.id
            
            # 为应急响应时间计算准备
            travel_time = (arc.length / getattr(arc, 'speed_kmh', 80.0)) * 60
            self._full_graph.add_edge(start_id, end_id, weight=travel_time)
            
            # 为解码器准备
            if arc.mode == 'road':
                self._road_graph.add_edge(start_id, end_id, weight=arc.length)
            elif arc.mode == 'railway':
                self._rail_graph.add_edge(start_id, end_id, weight=arc.length)


    def _decode_path(self, task: TransportTask, chosen_hub_indices: Tuple[int, int]) -> Tuple[List[Arc], List[Node]]:
        """
        解码器：将选定的枢纽索引转换为一条【严格的双枢纽】运输路径。
        - 只有当两个枢纽索引不同 (k != l) 时，才会尝试解码。
        - 如果两个枢纽索引相同 (k == l)，则解码失败。
        
        Args:
            task (TransportTask): 正在处理的运输任务。
            chosen_hub_idx (Tuple): 算法为该任务选择的枢纽索引对。

        Returns:
            Tuple[List[Arc], List[Node]]: 包含路径上的 Arc 对象和 Node 对象的列表。
        """
        k_float, l_float = chosen_hub_indices
        
        # 使用四舍五入找到最接近的整数
        k_idx = int(round(k_float))
        l_idx = int(round(l_float))
        
        # 确保转换后的索引不会因浮点数误差而越界
        num_hubs = len(self.hubs)
        k_idx = max(0, min(k_idx, num_hubs - 1))
        l_idx = max(0, min(l_idx, num_hubs - 1))
        
        # 如果算法生成了一个选择相同枢纽的解，我们将其视为无效解
        if k_idx == l_idx:
            return [], []   # 解码失败

        # 获取起点、终点、枢纽节点
        origin_node, dest_node = task.origin, task.destination
        hub_k = self.hubs[k_idx]
        hub_l = self.hubs[l_idx]

        # 核心逻辑：使用 networkx 寻找最短路径
        # NOTE: networkx 的最短路径算法需要一个 networkx.DiGraph 对象作为输入。
        full_path_node_ids = []
        try:
            # 第1段: Origin -> Hub_k (公路)
            path1 = nx.shortest_path(self._road_graph, source=origin_node.id, target=hub_k.id, weight='weight')
            # 第2段: Hub_k -> Hub_l (铁路)
            path2 = nx.shortest_path(self._rail_graph, source=hub_k.id, target=hub_l.id, weight='weight')
            # 第3段: Hub_l -> Destination (公路)
            path3 = nx.shortest_path(self._road_graph, source=hub_l.id, target=dest_node.id, weight='weight')
            
            # 拼接三段路径，去除重复的枢纽节点
            full_path_node_ids = path1[:-1] + path2[:-1] + path3

        except nx.NetworkXNoPath:
            # 如果任何一段路径不存在，解码失败
            return [], []

        # 如果未能生成任何有效路径ID，则返回失败
        if not full_path_node_ids:
            return [], []
            
        # 将节点 ID 列表转换回 Node 和 Arc 对象列表
        path_nodes = [self.network._nodes_dict[node_id] for node_id in full_path_node_ids]
        path_arcs = []
        for i in range(len(full_path_node_ids) - 1):
            start_id, end_id = full_path_node_ids[i], full_path_node_ids[i+1]
            arc = self.network._arcs_dict.get((start_id, end_id))
            if arc is None:
                return [], []
            path_arcs.append(arc)
            
        return path_arcs, path_nodes


    def _calculate_deterministic_demand(self, task: 'TransportTask') -> float:
        """
        根据置信水平 psi，计算任务v的确定性等价需求。
        Args:
            task (TransportTask): 正在处理的运输任务。
        Returns:
            float: 该任务的确定性等价需求量。
        """
        # 假设任务的需求参数 (μ, δ, a, b) 存储在 task 对象中
        # mu: 期望值, delta: 标准差, a/b: 三角模糊数的左右边界
        mu = getattr(task, 'demand_mu', 10.0)
        delta = getattr(task, 'demand_delta', 1.0)
        a = getattr(task, 'demand_a', 2.0)
        b = getattr(task, 'demand_b', 2.0)
        
        psi = self.params['psi']

        # 计算标准正态分布累积分布函数的反函数 Φ⁻¹(ψ)
        phi_inv_psi = norm.ppf(psi)
        
        # 根据决策者的风险偏好，应用论文中的公式
        if 0.5 < psi < 1: # 风险规避型
            equivalent_demand = mu + phi_inv_psi * delta + (2 * psi - 1) * b
        else: # 0 <= psi <= 0.5, 风险偏好型
            equivalent_demand = mu + phi_inv_psi * delta + (1 - 2 * psi) * a
            
        return equivalent_demand


    def _calculate_path_sp_cvar_and_base_risk(self, path_arcs: List['Arc'], path_nodes: List['Node'], task: 'TransportTask') -> Tuple[float, float]:
        """
        计算给定路径的 SP-CVaR (用于目标f1) 和基础总风险 Rv (用于Gini后处理分析)。
        按照 c -> c(t) -> c_SA(t) -> SP-CVaR 的流程实现。

        Returns:
            Tuple[float, float]: (该路径的 SP-CVaR 值, 该路径的基础总风险 Rv)
        """
        def get_emergency_response_time(segment) -> float:
            """[辅助函数] 计算到某个路段或枢纽的最小应急响应时间。"""
            min_time = float('inf')
            target_node_id = segment.start.id if isinstance(segment, Arc) else segment.id
            if not target_node_id:
                return 999.0

            for center in self.emergency_centers:
                try:
                    time = nx.shortest_path_length(self._full_graph, source=center.id, target=target_node_id, weight='weight')
                    if time < min_time:
                        min_time = time
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            return min_time if min_time != float('inf') else 999.0
        
        # 提取模型参数
        h = self.params['h_structural_aversion']
        k = self.params['k_steepness']
        th = self.params['t_threshold']
        alpha = self.params['alpha']
        q = self.params['q_leak_rate']
        gamma = self.params['gamma_diffusion_coeff']
        lambda_m = self.params['lambda_m']

        path_segments = path_arcs + [node for node in path_nodes if node.type == 'hub']
        
        # --- 步骤A & B: 计算 c_base 和 c(t)，并得到每个环节的基础风险 R_i ---
        segment_intermediate_results = []
        path_total_base_risk: float = 0.0

        for segment in path_segments:
            c_base, P = 0.0, 0.0
            if isinstance(segment, Arc):
                P = (self.params['pbm_road'] if segment.mode == 'road' else self.params['pbm_rail']) * segment.length
                c_base = (2 * lambda_m * segment.length + math.pi * lambda_m**2) * segment.population_density
            elif isinstance(segment, Node):
                P = self.params['pk_hub']
                c_base = math.pi * lambda_m**2 * segment.population_density
            
            # 计算 c(t) - 随时间演化的动态后果
            t = get_emergency_response_time(segment)
            c_t = c_base + (gamma * q * t)
            
            # 仅考虑应急响应时间的基础风险 R_i = c(t) * P
            base_risk_i = c_t * P
            
            segment_intermediate_results.append({'segment': segment, 'c_t': c_t, 'P': P, 'base_risk_i': base_risk_i})
            path_total_base_risk += base_risk_i

        # --- 步骤 C: 计算 c_SA(t) - 结构风险调整后后果 ---
        final_outcomes = [] # 存放 (最终修正后果 c_SA(t), 概率 P)

        for res in segment_intermediate_results:
            contrib = res['base_risk_i'] / path_total_base_risk if path_total_base_risk > 0 else 0
            
            # 单条路径上的风险，施加针对结构稳健性的二次惩罚（非常敏感）
            # c_SA(t) = c(t) * (1 + h * contrib^2)
            # c_sa_t = res['c_t'] * (1 + h * contrib**2)

            # 修改为线性惩罚（更平滑）
            # c_sa_t = res['c_t'] * (1 + h * contrib)

            # 或者根号惩罚（更平滑，鼓励适度差异）
            # c_sa_t = res['c_t'] * (1 + h * math.sqrt(contrib))

            # Sigmoid函数惩罚（平滑且有界）
            sigmoid = 1 / (1 + math.exp(-k*(contrib - th)))  # Sigmoid 函数平滑因子
            c_sa_t = res['c_t'] * (1 + h * sigmoid)

            final_outcomes.append({'consequence': c_sa_t, 'probability': res['P']})

        # --- 步骤 D: 基于 c_SA(t) 计算 SP-CVaR ---
        if not final_outcomes:
            return 0.0, 0.0

        sorted_outcomes = sorted(final_outcomes, key=lambda x: x['consequence'])
        
        total_accident_prob = sum(out['probability'] for out in sorted_outcomes)
        if total_accident_prob >= 1.0:
            total_accident_prob = 0.9999    # 避免无事故概率为0或负
        
        cumulative_prob = 1.0 - total_accident_prob
        VaR_sa = 0.0

        for outcome in sorted_outcomes:
            cumulative_prob += outcome['probability']
            if cumulative_prob >= alpha:
                VaR_sa = outcome['consequence']
                break
        
        expected_loss_above_var = 0.0
        for outcome in final_outcomes:
            loss_above_var = max(0, outcome['consequence'] - VaR_sa)
            expected_loss_above_var += loss_above_var * outcome['probability']
            
        sp_cvar_value = VaR_sa + (1 / (1 - alpha)) * expected_loss_above_var if (1 - alpha) > 1e-9 else VaR_sa

        return sp_cvar_value, path_total_base_risk
    

    def _calculate_path_cost(self, path_arcs: list, path_nodes: list, equivalent_demand: float) -> float:
        """
        计算给定单条路径的综合成本 (f2)。

        Args:
            path_arcs (list): 构成路径的 Arc 对象列表。
            path_nodes (list): 构成路径的 Node 对象列表。
            equivalent_demand (float): 该任务的确定性等价需求量。

        Returns:
            float: 该路径的总成本。
        """
        total_cost = 0.0
        
        # --- 1. 计算运输成本 + 碳排放成本 ---
        for arc in path_arcs:
            # 运输成本 = 单位成本 * 距离
            transport_cost = arc.cost_per_km * arc.length
            # 碳排放成本
            carbon_cost = arc.carbon_cost_per_ton
            # 累加总成本
            total_cost += (transport_cost + carbon_cost) * equivalent_demand

        # --- 2. 计算转运成本 ---
        for node in path_nodes:
            # 转运只在枢纽节点发生，并且从节点对象直接获取成本
            if node.type == 'hub':
                total_cost += node.transshipment_cost * equivalent_demand
                
        return total_cost

    def _calculate_gini(self, path_risks: List[float]) -> float:
        """
        计算给定一组路径风险的基尼系数 (Gini Coefficient)。

        Args:
            path_risks (List[float]): 一个列表，包含了解决方案中每条路径的总风险。

        Returns:
            float: 计算出的基尼系数值。
        """
        # n 是运输路径的总数
        n = len(path_risks)

        # --- 边界条件处理 ---
        # 如果路径数小于2，不存在不公平性，基尼系数为0
        if n < 2:
            return 0.0

        # --- 步骤 1: 计算平均风险 R ---
        # 根据公式 R = (Σ Rv) / n
        mean_risk = sum(path_risks) / n
        
        # --- 另一个边界条件处理 ---
        # 如果平均风险为0（例如所有路径风险都为0），则分母为0，且不存在不公平性
        if mean_risk == 0:
            return 0.0
            
        # --- 步骤 2: 计算分子 - 所有路径风险差值的绝对值之和 ---
        # 根据公式 Σ |Rp - Rq| for all p, q 
        sum_of_absolute_differences = 0.0
        for rp in path_risks:
            for rq in path_risks:
                # 公式中要求 p ≠ q，但在计算 |Rp - Rq| 时，
                # p = q 的情况 |Rp - Rp| = 0，不影响总和，所以可以直接双重循环
                sum_of_absolute_differences += abs(rp - rq)
        
        # --- 步骤 3: 计算分母 ---
        # 根据公式 2 * n^2 * R 
        denominator = 2 * (n**2) * mean_risk
        
        # --- 步骤 4: 计算基尼系数 σ ---
        gini_coefficient = sum_of_absolute_differences / denominator
        
        return gini_coefficient