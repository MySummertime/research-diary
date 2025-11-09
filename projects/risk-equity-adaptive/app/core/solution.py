# --- coding: utf-8 ---
# --- app/core/solution.py ---
import copy
from typing import Dict
from .path import Path

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
        # 由 Evaluator (评估器) 计算并填充。
        
        # 目标函数 f1 (风险)
        # 对应论文中的 min f1 = Σ CVaR_α(v)
        self.f1_risk: float = float('inf')
        
        # 目标函数 f2 (成本)
        # 对应论文中的 min E[f2]
        self.f2_cost: float = float('inf')

        # --- 3. 可行性与约束 ---
        # 由 Evaluator 计算并填充。
        
        # 该解是否可行 (是否违反容量约束或预算约束)
        self.is_feasible: bool = False
        
        # 约束违反度 (NSGA-II 需要)
        # 0.0 = 可行; > 0.0 = 不可行
        self.constraint_violation: float = float('inf')

        # --- 4. NSGA-II 排序指标 ---
        # 由主算法循环计算并填充。
        
        # 非支配排序的等级 (Rank 0 是最好的)
        self.rank: int = 0
        
        # 拥挤度距离
        self.crowding_distance: float = 0.0

    def clone(self) -> 'Solution':
        """
        创建一个与当前解完全一样的深拷贝 (Deep Copy)。
        这在遗传算法的交叉和变异中至关重要，
        以防止父代被子代的操作意外修改。
        """
        # 使用 deepcopy 来确保所有字典和 Path 对象都被完全复制
        return copy.deepcopy(self)

    def __repr__(self):
        """
        提供一个清晰的打印输出，用于调试。
        """
        cost_str = f"{self.f2_cost:,.0f}" if self.f2_cost != float('inf') else "Inf"
        risk_str = f"{self.f1_risk:,.0f}" if self.f1_risk != float('inf') else "Inf"
        
        return (
            f"Solution(Rank={self.rank}, "
            f"f1_Risk={risk_str}, "
            f"f2_Cost={cost_str}, "
            f"Feasible={self.is_feasible})"
        )