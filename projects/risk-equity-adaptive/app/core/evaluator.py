# --- coding: utf-8 ---
# --- app/core/evaluator.py ---
"""
[业务层] 解决方案评估器
作为协调者，调用 RiskModel 计算风险，调用 FuzzyMath 计算成本，
并汇总计算解的适应度 (Fitness) 和约束违反度 (CV)。
"""

import numpy as np
from typing import Any, Dict, List, Tuple

from app.core.fuzzy import FuzzyMath
from app.core.network import TransportNetwork
from app.core.risk_model import DynamicRiskModel
from app.core.solution import Solution


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
        self.experiment_config = self.config.get("experiment", {})

        # 初始化物理风险模型
        self.risk_model = DynamicRiskModel(network, self.risk_config)

        # 提取公平性配置 [方案一：基尼系数约束]
        self.equity_config = self.config.get("equity_config", {"gini_epsilon": 0.2})
        self.gini_epsilon = self.equity_config.get("gini_epsilon", 0.2)

        # --- 获取确定性成本参数 ---

        # 1. 单位运输成本 C_m (元/t·km)
        self.unit_transport_cost = self.cost_config.get(
            "unit_transport_cost", {"road": 0.23, "railway": 0.05}
        )

        # 2. 单位碳排放因子 omega_m (kg/t·km)
        self.transport_emission_factor = self.cost_config.get(
            "transport_emission_factor", {"road": 0.05771, "railway": 0.00820}
        )

        # 3. 碳税率 C_tax (元/kg)
        self.carbon_tax_rate = self.cost_config.get("carbon_tax_rate", 0.015)

        # 4. 单位转运成本 C_k^b (元/t)
        self.unit_transshipment_cost = self.cost_config.get(
            "unit_transshipment_cost", 3.090
        )

        # 5. 单位转运碳排放量 omega_k (kg/t)
        self.transshipment_emission_factor = self.cost_config.get(
            "transshipment_emission_factor", 0.128
        )

    # =========================================================================
    # Public Interface (公共接口)
    # =========================================================================

    def evaluate(self, solution: Solution):
        """
        [主方法] 评估一个解 (Solution) 的所有属性。
        """

        # 1. 计算成本
        solution.f2_cost = self._calculate_f2_cost(solution)

        # 2. 计算目标函数 f1 (CVaR 风险)
        # 这个函数会计算并填充 solution.eta_values
        real_risk = self._calculate_f1_cvar_risk(solution)
        solution.f1_risk = real_risk

        risk_scale = self.risk_config.get("risk_objective_scale", 1.0)
        solution.f1_risk_scaled = real_risk * risk_scale

        # 3. 计算公平性指标: 基尼系数 (由方案一要求，由结果指标变为内化约束)
        solution.gini_coefficient = self._calculate_solution_gini(solution)

        # 4. 检查所有约束 (容量 + 模糊成本预算 + 基尼系数上限)
        is_feasible, violation = self._check_constraints(solution)
        solution.is_feasible = is_feasible
        solution.constraint_violation = violation

    def calculate_cost_breakdown(self, solution: Solution) -> Dict[str, float]:
        """
        [Helper] 计算成本构成的详细拆解 (用于绘图)
        返回: {'transport': val, 'transshipment': val, 'carbon': val}
        """
        breakdown = {"transport": 0.0, "transshipment": 0.0, "carbon": 0.0}

        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue
            dv = path.task.demand

            # 1. 弧段相关成本 (运输成本 + 运输碳排放)
            for arc in path.arcs:
                mode = arc.mode
                d_ij = arc.length

                # 运输成本: C_ij^m * d^v * d_ij
                c_m = self.unit_transport_cost.get(mode, 0.0)
                breakdown["transport"] += c_m * dv * d_ij

                # 运输碳排放成本: C_tax * omega_m * d^v * d_ij
                omega_m = self.transport_emission_factor.get(mode, 0.0)
                breakdown["carbon"] += self.carbon_tax_rate * omega_m * dv * d_ij

            # 2. 枢纽相关成本 (转运成本 + 转运碳排放)
            for hub in path.transfer_hubs:
                # 转运成本: C_k^b * d^v
                c_kb = self.unit_transshipment_cost
                breakdown["transshipment"] += c_kb * dv

                # 转运碳排放成本: C_tax * omega_k * d^v
                omega_k = self.transshipment_emission_factor
                breakdown["carbon"] += self.carbon_tax_rate * omega_k * dv

        return breakdown

    def calculate_expected_risk(self, solution: Solution) -> float:
        """
        [Helper] 计算期望风险 (即 alpha=0 时的 CVaR)
        用于对比 CVaR 和 Expected Risk（绘图）
        """
        # 保存当前的配置
        original_alpha = self.risk_config.get("cvar_alpha")
        # 强制设为 0 计算期望值
        self.risk_config["cvar_alpha"] = 0.0

        # 计算
        exp_risk = self._calculate_f1_cvar_risk(solution)

        # 恢复
        if original_alpha is not None:
            self.risk_config["cvar_alpha"] = original_alpha
        else:
            self.risk_config.pop("cvar_alpha", None)

        return exp_risk

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

        # 重新从 self.config 中读取，防止 self.risk_config 引用失效
        current_risk_config = self.config.get("risk_model_f1", {})
        alpha = current_risk_config.get("cvar_alpha", 0.9996)

        # 允许 alpha = 0.0 (此时 CVaR = Expected Value)
        # 只拦截 alpha >= 1.0 (除以零) 和 alpha < 0.0 (无意义)
        if alpha >= 1.0 or alpha < 0.0:
            return float("inf")

        one_minus_alpha = 1.0 - alpha

        # 清空旧的计算结果
        solution.eta_values = {}

        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue

            # 1. 收集 (p, c) 事件对
            p_c_pairs: List[Tuple[float, float]] = []

            # 1a. 收集弧段风险
            for arc in path.arcs:
                p_ijm = arc.accident_prob_per_km * arc.length
                c_base = self.risk_model.get_consequence(arc)

                # Risk = Probability * Consequence (Impact Area * Pop Density)
                # 论文模型中风险被定义为潜在受影响人数
                c_ijm = c_base

                if p_ijm > 0 and c_ijm > 0:
                    p_c_pairs.append((p_ijm, c_ijm))

            # 1b. 收集枢纽风险
            for hub in path.transfer_hubs:
                p_k = hub.accident_prob
                c_base = self.risk_model.get_consequence(hub)

                c_k = c_base

                if p_k > 0 and c_k > 0:
                    p_c_pairs.append((p_k, c_k))

            # 2. 计算此任务的最优 eta* 和 CVaR
            if not p_c_pairs:
                # 这条路径没有风险
                optimal_eta_v = 0.0
                task_cvar = 0.0
            else:
                # 如果 alpha=0，则 optimal_eta_v=0 (因为 sum(p) << 1.0)
                # task_cvar = sum(p*c) / 1.0 = Expected Risk
                optimal_eta_v, task_cvar = self._calc_cvar_for_path(
                    p_c_pairs, alpha, one_minus_alpha
                )

            # 3. 存回 solution 作为计算结果
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

        # 增加 epsilon 防止除零 (虽然 alpha 通常 < 1.0)
        epsilon = 1e-9
        safe_denominator = one_minus_alpha + epsilon

        # 步骤 2: 找到 VaR (最优的 eta)
        # 需要找到最小的 c_i，使得 "超过它的概率" <= (1 - alpha)
        # 当 alpha=0 (one_minus_alpha=1) 时, CVaR = E[L] / 1 = E[L]
        total_prob = sum(p for c, p in sorted_c_p_pairs)

        # [Case 1] 绝大多数情况: total_prob (e.g. 0.0001) < 1-alpha (e.g. 0.05)
        # 这意味着 VaR_alpha = 0 (无事故状态)
        if total_prob <= one_minus_alpha:
            expected_loss = sum(p * c for c, p in sorted_c_p_pairs)
            # 此时 CVaR 退化为: Expected_Risk / (1 - alpha)
            # 使用 safe_denominator 确保稳定性
            return 0.0, expected_loss / safe_denominator

        # [Case 2] (极少见) 事故概率总和非常高，或者 alpha 极其接近 1.0
        prob_sum = 0.0
        opt_eta = sorted_c_p_pairs[-1][0]
        for c, p in sorted_c_p_pairs:
            prob_sum += p
            if total_prob - prob_sum <= one_minus_alpha:
                opt_eta = c
                break

        # CVaR Formula
        loss_sum = sum(p * max(0.0, c - opt_eta) for c, p in sorted_c_p_pairs)
        return opt_eta, opt_eta + loss_sum / one_minus_alpha

    # =========================================================================
    # Objective Function 2: Cost
    # =========================================================================z

    def _calculate_f2_cost(self, solution: Solution) -> float:
        bd = self.calculate_cost_breakdown(solution)
        return bd["transport"] + bd["transshipment"] + bd["carbon"]

    # =========================================================================
    # Constraints Checking (约束检查)
    # =========================================================================

    def _check_constraints(self, solution: Solution) -> Tuple[bool, float]:
        """
        检查 1) 容量约束 和 2) 运到期限约束。
        从 self.cost_config 读取参数。
        返回: (is_feasible, constraint_violation)

        - 弧段 (Arc) 约束：检单车运量是否超过限重
          物理含义：路面/桥梁的承重限制。
        - 枢纽 (Hub) 约束：保持总流量累加，检查是否超过枢纽容量。
          物理含义：中转站的吞吐/仓储能力限制。
        """
        total_violation = 0.0

        # 1. Arc Flow Accumulation (Only single task violation)
        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue
            demand = path.task.demand
            for arc in path.arcs:
                if demand > arc.capacity:
                    violation = (demand - arc.capacity) / arc.capacity
                    total_violation += violation

        # 2. Hub Capacity (Accumulated)
        node_flow = {}
        for path in solution.path_selections.values():
            demand = path.task.demand
            for hub in path.transfer_hubs:
                node_flow[hub.node_id] = node_flow.get(hub.node_id, 0.0) + demand

        for nid, flow in node_flow.items():
            node = self.network.get_node(nid)
            if node and flow > node.capacity:
                total_violation += (flow - node.capacity) / node.capacity

        # 3. Fuzzy Time
        # Σ t_inf(alpha_t) + Σ s_inf(alpha_t) <= T_max
        alpha_t = self.cost_config.get("time_confidence_level", 0.9)
        t_max_limit = self.cost_config.get("max_delivery_time", 96.0)

        for task_id, path in solution.path_selections.items():
            path_pessimistic_time = 0.0

            # 3a. 运输时间的悲观值累加
            for arc in path.arcs:
                t_pess = FuzzyMath.triangular_pessimistic_value(
                    *arc.fuzzy_transport_time, alpha_t
                )
                path_pessimistic_time += t_pess

            # 3b. 转运时间的悲观值累加
            for hub in path.transfer_hubs:
                s_pess = FuzzyMath.trapezoidal_pessimistic_value(
                    *hub.fuzzy_transshipment_time, alpha_t
                )
                path_pessimistic_time += s_pess

            # 检查时效违反
            if path_pessimistic_time > t_max_limit:
                time_violation = (path_pessimistic_time - t_max_limit) / t_max_limit
                total_violation += time_violation

        # 4. [方案一] 基尼系数公平性约束：Gini <= epsilon
        # 计算违反度: CV = max(0, Gini - epsilon)
        if solution.gini_coefficient > self.gini_epsilon:
            # 归一化违反度，便于与其他约束统一量级
            # 如果 epsilon 为 0，则无法作为分母，直接取差值 (即 Gini 值本身)
            if self.gini_epsilon > 1e-9:
                gini_violation = (solution.gini_coefficient - self.gini_epsilon) / self.gini_epsilon
            else:
                gini_violation = solution.gini_coefficient
            total_violation += gini_violation

        return total_violation == 0.0, total_violation

    # =========================================================================
    # Equity Calculation (公平性计算 - 方案一核心)
    # =========================================================================

    def _calculate_solution_gini(self, solution: Solution) -> float:
        """
        [方案一] 计算给定解决方案的风险分布基尼系数。
        逻辑说明：
        1. 统计每个节点承担的“风险暴露”。
        2. 将弧段风险平摊到其两端节点。
        3. 利用 Lorenz 曲线原理计算不均衡程度。
        """
        node_risk_map: Dict[str, float] = {}

        for path in solution.path_selections.values():
            if not path.task:
                continue

            # 1. 累计弧段风险贡献 (平摊给端点)
            for arc in path.arcs:
                # 贡献 = 概率 * 后果 (此处沿用 analyzer.py 的物理含义)
                p_ijm = arc.accident_prob_per_km * arc.length
                c_base = self.risk_model.get_consequence(arc)
                risk_contrib = p_ijm * c_base

                u_id = arc.start.node_id
                v_id = arc.end.node_id
                node_risk_map[u_id] = node_risk_map.get(u_id, 0.0) + risk_contrib / 2.0
                node_risk_map[v_id] = node_risk_map.get(v_id, 0.0) + risk_contrib / 2.0

            # 2. 累计枢纽转运风险
            for hub in path.transfer_hubs:
                p_k = hub.accident_prob
                c_base = self.risk_model.get_consequence(hub)
                risk_contrib = p_k * c_base
                node_risk_map[hub.node_id] = node_risk_map.get(hub.node_id, 0.0) + risk_contrib

        # 3. 计算基尼系数
        risk_vals = np.array(list(node_risk_map.values()), dtype=float)
        n = len(risk_vals)

        # 基础校验：如果节点少于2个或总风险为0，认为绝对公平 (Gini=0)
        if n < 2 or np.sum(risk_vals) == 0:
            return 0.0

        # 快速矩阵计算 |Ri - Rj| 之和
        diff_matrix = np.abs(risk_vals[:, None] - risk_vals[None, :])
        numerator = np.sum(diff_matrix)
        
        # 分母: 2 * n^2 * mean
        denominator = 2 * n * n * np.mean(risk_vals)
        
        if denominator == 0:
            return 0.0
            
        return float(numerator / denominator)
