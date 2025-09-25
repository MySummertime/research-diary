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
    def __init__(self, network: TransportNetwork, G_min: float, G_max: float):
        """
        初始化危险品运输优化问题。

        Args:
            network (TransportNetwork): 包含节点、弧段、任务等信息的网络实例。
            G_min (float): 基尼系数约束的下限。
            G_max (float): 基尼系数约束的上限。
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
            'k_time_penalty': 0.5,  # 应急响应时间敏感性系数 k
            'h_risk_equity': 1.0,   # 路径风险集中性敏感性系数 h
            't_base': 18.0, # 基准应急响应时间 (分钟)
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

        # 3. 预先构建一个 networkx 图，用于计算应急响应时间
        #    这个图应该包含所有节点和弧段
        # self._full_graph = nx.DiGraph() # type: ignore
        # for arc in self.network.arcs:
        #     # 假设弧段上有平均速度属性 (km/h)
        #     travel_time_minutes = (arc.length / getattr(arc, 'speed_kmh', 80.0)) * 60
        #     self._full_graph.add_edge(arc.start.id, arc.end.id, weight=travel_time_minutes)
        
        # --- Pymoo 元数据定义 ---

        # 决策变量：每个任务选择 2 个枢纽，决策变量数 == 任务数 * 2
        n_var = len(self.tasks) * 2
        
        # 目标函数：f1 (风险) 和 f2 (成本)
        n_obj = 2
        
        # 约束数量：每个弧段容量 + 每个枢纽容量 + 2个基尼系数约束
        n_constr = len(network.arcs) + len(network.nodes) + 2
        
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         xl=0,  # 枢纽索引的下界 >= 0
                         xu=len(self.hubs) - 1 if self.hubs else 0, # 枢纽索引的上界 <= num_hub - 1
                         vtype=int) # 决策变量为整数
        
        # 保存基尼系数约束的参数，以备 _evaluate 方法使用
        self.G_min = G_min
        self.G_max = G_max

    def _evaluate(self, x, out, *args, **kwargs):
        """
        评估函数的核心实现。
        它接收整个种群的决策变量 x，并为每个个体计算目标函数值 F 和约束违反值 G
        """
        # x 是一个 (n_pop, n_var) 的 NumPy 矩阵，代表整个种群
        # n_pop 是种群大小, n_var 是决策变量数量
        
        # 准备用于存放最终结果的空矩阵
        # F 用于存放目标函数值，G 用于存放约束违反值
        objectives_F = np.zeros((x.shape[0], self.n_obj))
        constraints_G = np.zeros((x.shape[0], self.n_constr))

        # --- 遍历种群中的每一个“解决方案” ---
        for i, individual_x in enumerate(x):
            # 将一维的决策变量数组，变回 V x 2 的形状
            solution_hubs = individual_x.reshape((len(self.tasks), 2))

            # --- 初始化当前解决方案的累加器 ---
            solution_total_f1_cvar = 0.0
            solution_total_f2_cost = 0.0
            solution_all_path_risks_for_gini = []   # 用于计算基尼系数
            arc_flows = defaultdict(float)
            node_flows = defaultdict(float)
            
            is_solution_valid = True
            # --- 遍历该解决方案中的每一个运输任务 ---
            for v_idx, task in enumerate(self.tasks):
                # --- 传递枢纽对给解码器 ---
                chosen_hub_indices = tuple(solution_hubs[v_idx])
                path_arcs, path_nodes = self._decode_path(task, chosen_hub_indices)

                # 如果解码失败（例如路径不存在），则将该解标记为无效
                if not path_arcs:
                    is_solution_valid = False
                    break # 中断当前解的评估
                
                # 2. 计算确定性等价需求
                equivalent_demand = self._calculate_deterministic_demand(task)

                # 3. 计算风险和成本
                # 综合风险值
                cvar_for_task_v, total_risk_for_path_v = self._calculate_path_cvar_and_total_risk(path_arcs, path_nodes, task)
                solution_total_f1_cvar += cvar_for_task_v

                # 综合成本
                cost_for_task_v = self._calculate_path_cost(path_arcs, path_nodes, equivalent_demand)
                solution_total_f2_cost += cost_for_task_v
                
                # 收集用于计算基尼系数的路径总风险
                solution_all_path_risks_for_gini.append(total_risk_for_path_v)

                # 4. 累加流量
                for arc in path_arcs:
                    arc_flows[(arc.start.id, arc.end.id)] += equivalent_demand
                # 使用解码器返节点列表
                for node in path_nodes:
                    node_flows[node.id] += equivalent_demand
            
            # 如果解无效，则为其分配一个极差的目标函数值（惩罚）
            if not is_solution_valid:
                objectives_F[i, :] = [1e20, 1e20] # 一个非常大的数值
                constraints_G[i, :] = 1e20 # 所有约束都视为严重违反
                continue # 继续评估下一个解

            # --- 所有任务评估完毕，赋值目标函数 ---
            objectives_F[i, 0] = solution_total_f1_cvar
            objectives_F[i, 1] = solution_total_f2_cost
            
            # --- 计算并赋值约束违反 ---
            violations = []
            # a) 基尼系数约束
            gini_coefficient = self._calculate_gini(solution_all_path_risks_for_gini)
            violations.append(gini_coefficient - self.G_max)  # g1 = σ - G_max <= 0
            violations.append(self.G_min - gini_coefficient)  # g2 = G_min - σ <= 0
            
            # b) 弧段和节点容量约束
            for arc in self.network.arcs:
                flow = arc_flows.get((arc.start.id, arc.end.id), 0)
                violations.append(flow - arc.capacity)  # violation = flow - capacity <= 0
            for node in self.network.nodes:
                flow = node_flows.get(node.id, 0)
                violations.append(flow - node.capacity) # violation = flow - capacity <= 0
            
            constraints_G[i, :] = np.array(violations)
        
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
        解码器：将选定的枢纽索引转换为一条严格的双枢纽运输路径。
        
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
        
        # --- 强制要求两个枢纽必须不同 ---
        # 如果算法生成了一个选择相同枢纽的解，我们将其视为无效解
        if k_idx == l_idx:
            return [], []  # 解码失败

        # 获取起点、终点、枢纽节点
        origin_node, dest_node = task.origin, task.destination
        hub_k = self.hubs[k_idx]
        hub_l = self.hubs[l_idx]

        # 核心逻辑：使用 networkx 寻找最短路径
        # NOTE: networkx 的最短路径算法需要一个 networkx.DiGraph 对象作为输入。
        full_path_node_ids = []
        try:
            path1 = nx.shortest_path(self._road_graph, source=origin_node.id, target=hub_k.id, weight='weight')
            path2 = nx.shortest_path(self._rail_graph, source=hub_k.id, target=hub_l.id, weight='weight')
            path3 = nx.shortest_path(self._road_graph, source=hub_l.id, target=dest_node.id, weight='weight')
            
            # 拼接三段路径，去除重复的枢纽节点
            full_path_node_ids = path1[:-1] + path2[:-1] + path3

        except nx.NetworkXNoPath:
            return [], []   # 如果任何一段路径不存在，解码失败

        # 将节点 ID 列表转换回 Node 和 Arc 对象列表 (逻辑不变)
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


    def _calculate_path_cvar_and_total_risk(self, path_arcs: List['Arc'], path_nodes: List['Node'], task: 'TransportTask') -> Tuple[float, float]:
        """
        计算给定路径的 CVaR (用于目标f1) 和总风险 Rv' (用于基尼系数)。
        严格按照论文中的步骤实现。

        Args:
            path_arcs (List[Arc]): 构成路径的 Arc 对象列表。
            path_nodes (List[Node]): 构成路径的 Node 对象列表。
            task (TransportTask): 正在处理的运输任务。

        Returns:
            Tuple[float, float]: (该路径的 CVaR 值, 该路径的总风险 Rv')
        """
        def get_emergency_response_time(segment) -> float:
            """
            [辅助函数] 计算到某个路段或枢纽的最小应急响应时间。
            """
            min_time = float('inf')
            
            # 确定事故发生点 (对于弧段，我们取其中点)
            target_node_id = None
            if isinstance(segment, Arc):
                # 简化处理：假设应急响应车辆需要到达弧段的起点
                target_node_id = segment.start.id
            elif isinstance(segment, Node):
                target_node_id = segment.id

            if target_node_id is None:
                return self.params['t_base']    # 返回基准时间作为惩罚

            # 遍历所有应急中心，找到最短的响应时间
            for center in self.emergency_centers:
                try:
                    # 使用预先构建的 networkx 图计算最短时间路径
                    time = nx.shortest_path_length(
                        self._full_graph, 
                        source=center.id, 
                        target=target_node_id, 
                        weight='weight' # 权重是行驶时间（h）
                    )
                    if time < min_time:
                        min_time = time
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            # 如果没有可达的应急中心，返回一个较大的惩罚时间
            return min_time if min_time != float('inf') else self.params['t_base'] * 5
        
        # 提取模型参数
        k = self.params['k_time_penalty']
        h = self.params['h_risk_equity']
        t_base = self.params['t_base']
        lambda_m = self.params['lambda_m']
        alpha = self.params['alpha']

        path_segments = path_arcs + [node for node in path_nodes if node.type == 'hub']
        
        # --- 步骤 A & B: 计算每个路段/枢纽的 R' 和路径总风险 Rv' ---
        segment_intermediate_results = []
        Rv_prime: float = 0.0

        for segment in path_segments:
            c_base, P = 0.0, 0.0
            if isinstance(segment, Arc):    # 弧段
                if segment.mode == 'road':
                    P = self.params['pbm_road'] * segment.length
                elif segment.mode == 'railway':
                    P = self.params['pbm_rail'] * segment.length
            elif isinstance(segment, Node): # 枢纽
                P = self.params['pk_hub']
                c_base = math.pi * lambda_m**2 * segment.population_density

            t = get_emergency_response_time(segment)
            f_t = 1 + k * math.log(t / t_base if t > 0 else 1.0)
            R_prime: float = c_base * f_t * P
            
            segment_intermediate_results.append({'segment': segment, 'c_base': c_base, 'P': P, 'f_t': f_t, 'R_prime': R_prime})
            Rv_prime += R_prime

        # --- 步骤 C: 计算每个路段/枢纽的最终风险 R 和最终事故后果 c ---
        final_outcomes = [] # 存放 (最终后果c, 概率P)
        for res in segment_intermediate_results:
            contrib = res['R_prime'] / Rv_prime if Rv_prime > 0 else 0

            # 最终事故后果 c = c_base * f_t * (1 + h * contrib^2)
            # 将路径风险集中性作为惩罚施加在事故后果上
            final_c = res['c_base'] * res['f_t'] * (1 + h * contrib**2)
            final_outcomes.append({'consequence': final_c, 'probability': res['P']})

        # --- 步骤 D: 计算 CVaR ---
        if not final_outcomes:
            return 0.0, 0.0

        # 1. 找到 VaR (Value-at-Risk), 即 η_T^v
        #    首先，按事故后果从小到大排序
        sorted_outcomes = sorted(final_outcomes, key=lambda x: x['consequence'])
        
        #    计算总事故概率
        total_accident_prob = sum(out['probability'] for out in sorted_outcomes)
        
        #    计算累积概率
        cumulative_prob = 1.0 - total_accident_prob # “无事故”的概率
        VaR = 0.0

        for outcome in sorted_outcomes:
            cumulative_prob += outcome['probability']
            if cumulative_prob >= alpha:
                VaR = outcome['consequence']
                break
        
        # 2. 应用 CVaR 公式 [cite: 79]
        #    CVaR = VaR + (1/(1-alpha)) * E[(c - VaR)^+]
        expected_loss_above_var = 0.0
        for outcome in final_outcomes:
            loss_above_var = max(0, outcome['consequence'] - VaR)
            expected_loss_above_var += loss_above_var * outcome['probability']
            
        if (1 - alpha) > 1e-9:  # 避免除以零
            cvar_value = VaR + (1 / (1 - alpha)) * expected_loss_above_var
        else:
            cvar_value = VaR

        return cvar_value, Rv_prime
    

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