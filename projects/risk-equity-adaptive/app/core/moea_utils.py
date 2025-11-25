# --- coding: utf-8 ---
# --- app/core/moea_utils.py ---
"""
[算法层] 多目标优化算法工具库
包含标准的非支配排序 (Non-dominated Sort) 和拥挤度距离 (Crowding Distance) 计算。
"""

from typing import List
from app.core.solution import Solution


class MOEAUtils:
    @staticmethod
    def fast_non_dominated_sort(population: List[Solution]) -> List[List[Solution]]:
        """
        执行快速非支配排序。
        返回: fronts = [[rank0_sols], [rank1_sols], ...]
        """
        # 1. 去重 (基于目标值的表现型去重，提高效率)
        unique_map = {}
        for s in population:
            # Key: (Risk, Cost, Violation)
            key = (
                round(s.f1_risk, 6),
                round(s.f2_cost, 6),
                round(s.constraint_violation, 6),
            )
            if key not in unique_map:
                unique_map[key] = s

        unique_pop = list(unique_map.values())

        fronts = [[]]
        sol_info = {}  # 存储 n_p 和 S_p

        for p in unique_pop:
            n_p = 0
            S_p = []
            for q in unique_pop:
                if p == q:
                    continue

                if MOEAUtils.constrained_dominates(p, q):
                    S_p.append(q)
                elif MOEAUtils.constrained_dominates(q, p):
                    n_p += 1

            sol_info[id(p)] = {"n_p": n_p, "S_p": S_p}

            if n_p == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in sol_info[id(p)]["S_p"]:
                    sol_info[id(q)]["n_p"] -= 1
                    if sol_info[id(q)]["n_p"] == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    @staticmethod
    def calculate_crowding_distance(front: List[Solution]):
        """
        计算并赋值拥挤度距离。
        """
        if not front:
            return

        n = len(front)
        for s in front:
            s.crowding_distance = 0.0

        # 分离可行与不可行解
        feasible_sols = [s for s in front if s.is_feasible]
        infeasible_sols = [s for s in front if not s.is_feasible]

        # 处理不可行解：简单的基于 CV 的拥挤度
        if infeasible_sols:
            max_cv = max(s.constraint_violation for s in infeasible_sols) or 1.0
            for s in infeasible_sols:
                # CV 越小越好，所以距离设为反比
                s.crowding_distance = 1.0 - (s.constraint_violation / max_cv)

        # 处理可行解：标准拥挤度计算
        if len(feasible_sols) <= 2:
            for s in feasible_sols:
                s.crowding_distance = float("inf")
            return

        # 对每个目标函数进行计算
        for attr in ["f1_risk", "f2_cost"]:
            # 按该目标值排序
            feasible_sols.sort(key=lambda s: getattr(s, attr))

            # 边界点设为无限大
            feasible_sols[0].crowding_distance = float("inf")
            feasible_sols[-1].crowding_distance = float("inf")

            min_val = getattr(feasible_sols[0], attr)
            max_val = getattr(feasible_sols[-1], attr)
            rng = max_val - min_val
            if rng == 0:
                rng = 1.0

            for i in range(1, len(feasible_sols) - 1):
                dist = (
                    getattr(feasible_sols[i + 1], attr)
                    - getattr(feasible_sols[i - 1], attr)
                ) / rng
                if feasible_sols[i].crowding_distance != float("inf"):
                    feasible_sols[i].crowding_distance += dist

    @staticmethod
    def constrained_dominates(p: Solution, q: Solution) -> bool:
        """
        带约束的支配关系判断。
        """
        if p.is_feasible and not q.is_feasible:
            return True
        elif not p.is_feasible and q.is_feasible:
            return False
        elif not p.is_feasible and not q.is_feasible:
            return p.constraint_violation < q.constraint_violation
        else:
            # 标准 Pareto 支配
            better_in_any = False
            for attr in ["f1_risk", "f2_cost"]:
                val_p = getattr(p, attr)
                val_q = getattr(q, attr)
                if val_p > val_q:
                    return False
                if val_p < val_q:
                    better_in_any = True
            return better_in_any
