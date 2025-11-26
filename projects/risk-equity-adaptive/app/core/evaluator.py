# --- coding: utf-8 ---
# --- app/core/evaluator.py ---
"""
[业务层] 解决方案评估器
作为协调者，调用 RiskModel 计算风险，调用 FuzzyMath 计算成本，
并汇总计算解的适应度 (Fitness) 和约束违反度 (CV)。
"""

from typing import Dict, Any, Tuple, List
from app.core.network import TransportNetwork
from app.core.solution import Solution
from app.core.risk_model import DynamicRiskModel
from app.core.fuzzy import FuzzyMath


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
        self.config = config

        # 提取各自关心的配置分组，提高内聚性
        self.risk_config = self.config.get("risk_model_f1", {})
        self.cost_config = self.config.get("cost_model_f2", {})

        # 初始化物理风险模型
        self.risk_model = DynamicRiskModel(network, self.risk_config)

        # --- 获取确定性成本参数 ---
        
        # 1. 单位运输成本 C (yuan/t*km)
        self.unit_transport_cost = self.cost_config.get(
            "unit_transport_cost", {"road": 0.55, "railway": 0.12}
        )
        # 2. 单位碳排放成本 E (yuan/t*km)
        self.unit_carbon_cost = self.cost_config.get(
            "unit_carbon_cost", {"road": 90, "railway": 70}
        )
        # 3. 运输超时单位惩罚成本 P (yuan/t*h)
        self.unit_penalty_cost = self.cost_config.get("unit_penalty_cost", 5)
        # 4. 枢纽单位转运成本 B_k (yuan/t)
        self.unit_transshipment_cost = self.cost_config.get(
            "unit_transshipment_cost", 8
        )
        # 5. 枢纽单位设备成本 I_k (yuan/t*h)
        self.unit_transshipment_infra_cost = self.cost_config.get(
            "unit_transshipment_infra_cost", 200
        )

    # =========================================================================
    # Public Interface (公共接口)
    # =========================================================================

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

    # =========================================================================
    # Objective Function 1: Risk (CVaR)
    # =========================================================================

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
            return float("inf")

        one_minus_alpha = 1.0 - alpha

        # 清空旧的计算结果
        solution.eta_values = {}

        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue
            
            # 获取该任务的运量
            dv = path.task.demand

            # 1. 收集 (p, c) 事件对
            p_c_pairs: List[Tuple[float, float]] = []

            # 1a. 收集弧段风险
            for arc in path.arcs:
                p_ijm = arc.accident_prob_per_km * arc.length
                c_base = self.risk_model.get_consequence(arc)

                # 后果 = 基础后果 * 运量
                # 这意味着运量越大，潜在的危害权重越大
                c_ijm = c_base * dv

                if p_ijm > 0 and c_ijm > 0:
                    p_c_pairs.append((p_ijm, c_ijm))

            # 1b. 收集枢纽风险
            for hub in path.transfer_hubs:
                p_k = hub.accident_prob
                c_base = self.risk_model.get_consequence(hub)
                
                # 后果 = 基础后果 * 运量
                c_k = c_base * dv
                
                if p_k > 0 and c_k > 0:
                    p_c_pairs.append((p_k, c_k))

            # 2. 计算此任务的最优 eta* 和 CVaR
            if not p_c_pairs:
                # 这条路径没有风险
                optimal_eta_v = 0.0
                task_cvar = 0.0
            else:
                optimal_eta_v, task_cvar = self._calc_cvar_for_path(
                    p_c_pairs, alpha, one_minus_alpha
                )

            # 3. 存回 solution 作为“计算结果”
            solution.eta_values[task_id] = optimal_eta_v

            # 4. 累加到总风险
            total_risk += task_cvar

        return total_risk

    def _calc_cvar_for_path(
        self, p_c_pairs: List[Tuple[float, float]], alpha: float, one_minus_alpha: float
    ) -> Tuple[float, float]:
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
            optimal_eta_v = sorted_c_p_pairs[-1][0]  # 默认为最大后果

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

    # =========================================================================
    # Objective Function 2: Cost (Expected Value)
    # =========================================================================z

    def _calculate_f2_expected_cost(self, solution: Solution) -> float:
        """
        计算 f2：总期望成本：
        1. 运输段: C_m * d_ij + P * E[t_ij]
        2. 碳排放: E_m * d_ij
        3. 枢纽段: B_k + I_k * E[s_k]
        """
        total_cost: float = 0.0

        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue

            # d^v: 任务运量
            dv = path.task.demand

            # --- Part 1 & 2: 弧段成本 (运输 + 惩罚 + 碳排放) ---
            for arc in path.arcs:
                mode = arc.mode  # 'road' or 'railway'
                d_ij = arc.length

                # 获取该模式的参数
                C_m = self.unit_transport_cost.get(mode, 0.0)
                E_m = self.unit_carbon_cost.get(mode, 0.0)
                P = self.unit_penalty_cost

                # 计算时间的期望值 E[t_ij]
                expected_time = FuzzyMath.triangular_expected_value(
                    *arc.fuzzy_transport_time
                )

                # 累计成本: d^v * (C_m * d_ij + P * E[t] + E_m * d_ij)
                # 可以合并为: d^v * ((C_m + E_m) * d_ij + P * expected_time)
                segment_cost = dv * ((C_m + E_m) * d_ij + P * expected_time)
                total_cost += segment_cost

            # --- Part 3: 枢纽成本 (转运 + 设备占用) ---
            for hub in path.transfer_hubs:
                # 获取枢纽参数
                B_k = self.unit_transshipment_cost
                I_k = self.unit_transshipment_infra_cost

                # 计算转运时间的期望值 E[s_k]
                expected_trans_time = FuzzyMath.trapezoidal_expected_value(
                    *hub.fuzzy_transshipment_time
                )

                # 累计成本: d^v * (B_k + I_k * E[s_k])
                hub_cost = dv * (B_k + I_k * expected_trans_time)
                total_cost += hub_cost

        return total_cost

    # =========================================================================
    # Constraints Checking (约束检查)
    # =========================================================================

    def _check_constraints(self, solution: Solution) -> Tuple[bool, float]:
        """
        检查 1) 容量约束 和 2) 模糊成本约束。
        从 self.cost_config 读取参数。
        返回: (is_feasible, constraint_violation)
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
                arc_key = (arc.start.node_id, arc.end.node_id)
                arc_flow[arc_key] = arc_flow.get(arc_key, 0.0) + demand

            # 累加枢纽流量
            for hub in path.transfer_hubs:
                node_flow[hub.node_id] = node_flow.get(hub.node_id, 0.0) + demand

        # 检查弧段容量
        for (u_id, v_id), flow in arc_flow.items():
            # 使用 public method get_arc(u, v) 而非私有属性访问
            arc = self.network.get_arc(u_id, v_id)
            
            if arc and flow > arc.capacity:
                violation = (
                    (flow - arc.capacity) / arc.capacity if arc.capacity > 0 else flow
                )
                total_violation += violation

        # 检查枢纽容量
        for node_id, flow in node_flow.items():
            # 使用 public method get_node(id) 而非私有属性访问
            node = self.network.get_node(node_id)
            
            if node and flow > node.capacity:
                violation = (
                    (flow - node.capacity) / node.capacity
                    if node.capacity > 0
                    else flow
                )
                total_violation += violation

        # --- 2. 模糊成本可靠性约束 (Chance Constraint) ---
        # Cr{Cost <= Budget} >= alpha_c  <==>  Pessimistic_Value <= Budget
        pessimistic_cost = 0.0
        alpha_c = self.cost_config.get("fuzzy_cost_alpha_c", 0.90)
        bgt = self.cost_config.get("fuzzy_cost_budget", 5000000)

        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue
            dv = path.task.demand

            # 1. 弧段部分 (使用悲观时间)
            for arc in path.arcs:
                mode = arc.mode
                d_ij = arc.length
                C_m = self.unit_transport_cost.get(mode, 0.0)
                E_m = self.unit_carbon_cost.get(mode, 0.0)
                P = self.unit_penalty_cost

                # 计算 t_ij 的 α-悲观值
                pess_time = FuzzyMath.triangular_pessimistic_value(
                    *arc.fuzzy_transport_time, alpha_c
                )

                # 公式同 f2，但时间换成 pess_time
                pessimistic_cost += dv * ((C_m + E_m) * d_ij + P * pess_time)

            # 2. 枢纽部分 (使用悲观时间)
            for hub in path.transfer_hubs:
                B_k = self.unit_transshipment_cost
                I_k = self.unit_transshipment_infra_cost

                # 计算 s_k 的 α-悲观值
                pess_trans_time = FuzzyMath.trapezoidal_pessimistic_value(
                    *hub.fuzzy_transshipment_time, alpha_c
                )

                # 公式同 f2
                pessimistic_cost += dv * (B_k + I_k * pess_trans_time)

        # 检查是否超支
        if pessimistic_cost > bgt:
            cost_violation = (
                (pessimistic_cost - bgt) / bgt if bgt > 0 else pessimistic_cost
            )
            total_violation += cost_violation

        is_feasible = total_violation == 0.0
        return is_feasible, total_violation
