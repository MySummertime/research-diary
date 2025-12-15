# --- coding: utf-8 ---
# --- app/core/solution.py ---
from typing import Dict
from app.core.path import Path


class Solution:
    """
    一个 "解" (或 "染色体") 的数据容器。

    它封装了遗传算法需要优化的所有 "基因"，
    以及评估后用于排序的 "适应度" (fitness) 结果。
    """

    def __init__(self):
        # --- 1. 基因 (Chromosome) ---
        # 遗传算法需要 *进化* 的决策变量。
        # 这是论文模型的核心。

        # 基因: 路径选择 (离散变量)
        # 字典: { "task_id_1": Path_obj_A, "task_id_2": Path_obj_B, ... }
        self.path_selections: Dict[str, Path] = {}

        # eta 是一个“评估结果”的缓存，由 Evaluator 填充
        # 字典: { "task_id_1": 25000.0, "task_id_2": 31000.0, ... }
        self.eta_values: Dict[str, float] = {}

        # --- 2. 适应度 (Fitness) ---
        self.f1_risk: float = float("inf")
        self.f1_risk_scaled: float = float("inf")
        self.f2_cost: float = float("inf")
        self.gini_coefficient: float = float("inf")

        # --- 3. 可行性与约束 ---
        self.pessimistic_cost: float = float("inf")
        self.is_feasible: bool = False
        self.constraint_violation: float = float("inf")

        # --- 4. NSGA-II 排序指标 ---
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    def clone(self) -> "Solution":
        """
        创建一个高效的克隆。
        手动复制关键数据结构，避免 copy.deepcopy 带来的巨大开销。
        """
        new_sol = Solution()

        # 1. 浅拷贝路径选择字典 (Dict[str, Path])
        # Path 对象本身是只读/共享的，不需要深拷贝
        new_sol.path_selections = self.path_selections.copy()

        # 2. 复制 eta 缓存
        new_sol.eta_values = self.eta_values.copy()

        # 3. 复制标量属性
        new_sol.f1_risk = self.f1_risk
        new_sol.f1_risk_scaled = self.f1_risk_scaled
        new_sol.f2_cost = self.f2_cost
        new_sol.gini_coefficient = self.gini_coefficient

        new_sol.pessimistic_cost = self.pessimistic_cost
        new_sol.is_feasible = self.is_feasible
        new_sol.constraint_violation = self.constraint_violation

        new_sol.rank = self.rank
        new_sol.crowding_distance = self.crowding_distance

        return new_sol

    def __repr__(self):
        """
        提供一个清晰的打印输出，用于调试。
        """
        feas_icon = "✅" if self.is_feasible else "❌"
        cost_str = f"{self.f2_cost:,.0f}" if self.f2_cost != float("inf") else "inf"
        risk_str = f"{self.f1_risk:,.0f}" if self.f1_risk != float("inf") else "inf"

        return (
            f"Solution[{feas_icon}]"
            f"Rank={self.rank}, "
            f"Risk={risk_str}, "
            f"Cost={cost_str}, "
            f"Feasible={self.is_feasible}, "
            f"CV={self.constraint_violation:.4f}"
        )
