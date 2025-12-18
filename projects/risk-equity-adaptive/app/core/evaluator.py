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
        self.experiment_config = self.config.get("experiment", {})

        # 初始化物理风险模型
        self.risk_model = DynamicRiskModel(network, self.risk_config)

        self.delta = self.cost_config.get("cost_uncertainty_multiplier", 1.0)

        # --- 获取确定性成本参数 ---

        # 1. 单位运输成本 C (yuan/t*km)
        self.unit_transport_cost = self.cost_config.get(
            "unit_transport_cost", {"road": 0.55, "railway": 0.12}
        )
        # 2. 单位碳排放成本 E (yuan/t*km)
        self.unit_carbon_cost = self.cost_config.get(
            "unit_carbon_cost", {"road": 0.05, "railway": 0.01}
        )
        # 3. 单位运营成本 P (yuan/h)
        self.unit_operation_cost = self.cost_config.get("unit_operation_cost", 150)
        # 4. 枢纽单位转运成本 B_k (yuan/t)
        self.unit_transshipment_cost = self.cost_config.get(
            "unit_transshipment_cost", 8
        )
        # 5. 枢纽单位运营成本 I_k (yuan/h)
        self.unit_transshipment_infra_cost = self.cost_config.get(
            "unit_transshipment_infra_cost", 100
        )

    # =========================================================================
    # Public Interface (公共接口)
    # =========================================================================

    def evaluate(self, solution: Solution):
        """
        [主方法] 评估一个解 (Solution) 的所有属性。
        """

        # 1. 计算真实期望成本
        solution.f2_cost = self._calculate_f2_expected_cost(solution, False)

        # 2. 计算缩放后的期望成本 (用于算法搜索引导和 budget 灵敏度分析)
        # 如果 delta == 1.0，则 f2_cost_scaled 等于 f2_cost
        if self.delta != 1.0:
            solution.f2_cost_scaled = self._calculate_f2_expected_cost(solution, True)
        else:
            solution.f2_cost_scaled = solution.f2_cost

        # 3. 计算目标函数 f1 (CVaR 风险)
        # 这个函数会计算并填充 solution.eta_values
        real_risk = self._calculate_f1_cvar_risk(solution)
        solution.f1_risk = real_risk

        scale = self.risk_config.get("risk_objective_scale", 1.0)
        solution.f1_risk_scaled = real_risk * scale

        # 4. 检查所有约束 (容量 + 模糊成本预算)
        is_feasible, violation = self._check_constraints(solution)
        solution.is_feasible = is_feasible
        solution.constraint_violation = violation

    def calculate_cost_breakdown(
        self, solution: Solution, use_scaling: bool = False
    ) -> Dict[str, float]:
        """
        [Helper] 计算成本构成的详细拆解 (用于绘图)
        返回: {'transport': val, 'transshipment': val, 'carbon': val}
        """
        breakdown = {"transport": 0.0, "transshipment": 0.0, "carbon": 0.0}

        for task_id, path in solution.path_selections.items():
            if not path.task:
                continue
            dv = path.task.demand

            # 1. 弧段成本
            for arc in path.arcs:
                mode = arc.mode
                d_ij = arc.length
                C_m = self.unit_transport_cost.get(mode, 0.0)
                E_m = self.unit_carbon_cost.get(mode, 0.0)
                P = self.unit_operation_cost
                expected_time = FuzzyMath.triangular_expected_value(
                    *arc.fuzzy_transport_time
                )

                # 使用动态缩放后的模糊时间
                scaled_time = self._get_scaled_fuzzy(
                    arc.fuzzy_transport_time, "triangular", use_scaling
                )
                expected_time = FuzzyMath.triangular_expected_value(*scaled_time)

                # 拆分
                breakdown["transport"] += C_m * dv * d_ij + P * expected_time
                breakdown["carbon"] += E_m * dv * d_ij

            # 2. 枢纽成本
            for hub in path.transfer_hubs:
                B_k = self.unit_transshipment_cost
                I_k = self.unit_transshipment_infra_cost
                expected_trans_time = FuzzyMath.trapezoidal_expected_value(
                    *hub.fuzzy_transshipment_time
                )

                # 使用动态缩放后的模糊时间
                scaled_trans_time = self._get_scaled_fuzzy(
                    hub.fuzzy_transshipment_time, "trapezoidal", use_scaling
                )
                expected_trans_time = FuzzyMath.trapezoidal_expected_value(
                    *scaled_trans_time
                )

                # 全部算作转运成本
                breakdown["transshipment"] += B_k * dv + I_k * expected_trans_time

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

    # --- 核心辅助函数: 不确定性动态缩放 ---
    def _get_scaled_fuzzy(
        self, fuzzy_val: Tuple, shape: str, use_scaling: bool = False
    ) -> Tuple:
        """
        根据全局配置中的 'uncertainty_multiplier' (delta) 缩放模糊数区间。
        delta = 1.0 (Default): 确定性值
        delta > 1.0: 区间变宽 (环境恶化)
        """
        delta = self.delta
        if not use_scaling or delta == 1.0:
            return fuzzy_val

        # 采用比例缩放：研究 '不确定性环境整体恶化' 的标准做法
        if shape == "triangular":
            a, b, c = fuzzy_val
            # 整体按比例向右推移并扩张
            return (a * delta, b * delta, c * delta)

        elif shape == "trapezoidal":
            a, b, c, d = fuzzy_val
            return (a * delta, b * delta, c * delta, d * delta)

        return fuzzy_val

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
        alpha = current_risk_config.get("cvar_alpha", 0.95)

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
    # Objective Function 2: Cost (Expected Value)
    # =========================================================================z

    def _calculate_f2_expected_cost(
        self, solution: Solution, use_scaling: bool = False
    ) -> float:
        bd = self.calculate_cost_breakdown(solution, use_scaling)
        return bd["transport"] + bd["transshipment"] + bd["carbon"]

    # =========================================================================
    # Constraints Checking (约束检查)
    # =========================================================================

    def _check_constraints(self, solution: Solution) -> Tuple[bool, float]:
        """
        检查 1) 容量约束 和 2) 模糊成本约束。
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

        # 3. Fuzzy Cost
        pessimistic_cost_scaled = 0.0
        alpha_c = self.cost_config.get("fuzzy_cost_alpha_c", 0.99960)
        bgt = self.cost_config.get("fuzzy_cost_budget", 1e15)

        for task_id, path in solution.path_selections.items():
            dv = path.task.demand
            for arc in path.arcs:
                mode = arc.mode
                d_ij = arc.length
                C_m = self.unit_transport_cost.get(mode, 0.0)
                E_m = self.unit_carbon_cost.get(mode, 0.0)
                P = self.unit_operation_cost

                scaled_time = self._get_scaled_fuzzy(
                    arc.fuzzy_transport_time, "triangular", True
                )
                pess_time = FuzzyMath.triangular_pessimistic_value(
                    *scaled_time, alpha_c
                )
                pessimistic_cost_scaled += (C_m + E_m) * dv * d_ij + P * pess_time
            for hub in path.transfer_hubs:
                B_k = self.unit_transshipment_cost
                I_k = self.unit_transshipment_infra_cost

                scaled_trans_time = self._get_scaled_fuzzy(
                    hub.fuzzy_transshipment_time, "trapezoidal", True
                )
                pess_trans_time = FuzzyMath.trapezoidal_pessimistic_value(
                    *scaled_trans_time, alpha_c
                )
                pessimistic_cost_scaled += B_k * dv + I_k * pess_trans_time

        solution.pessimistic_cost = pessimistic_cost_scaled

        if pessimistic_cost_scaled > bgt:
            total_violation += (pessimistic_cost_scaled - bgt) / bgt

        return total_violation == 0.0, total_violation
