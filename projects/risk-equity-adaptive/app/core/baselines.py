# --- coding: utf-8 ---
# --- app/core/baselines.py ---
"""
[基准算法适配层]
包含:
1. Pymoo 适配器: NSGA-II, SPEA2
2. Gurobi 适配器: 精确解求解器 (基于路径选择的 MILP 模型)
"""

import numpy as np
import logging
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Any

# Pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2 as PymooNSGA2
from pymoo.algorithms.moo.spea2 import SPEA2 as PymooSPEA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.termination import get_termination

# Project
from app.core.solution import Solution
from app.core.path import Path
from app.core.network import TransportNetwork
from app.core.evaluator import Evaluator
from app.core.fuzzy import FuzzyMath


class PymooHazmatProblem(ElementwiseProblem):
    """
    Pymoo 问题包装器 (保持不变)
    """

    def __init__(
        self,
        network: TransportNetwork,
        evaluator: Evaluator,
        candidate_paths_map: Dict[str, List[Path]],
    ):
        self.network = network
        self.evaluator = evaluator
        self.candidate_paths_map = candidate_paths_map

        self.task_ids = sorted([t.task_id for t in network.tasks])
        self.n_tasks = len(self.task_ids)

        self.upper_bounds = []
        for tid in self.task_ids:
            count = len(self.candidate_paths_map.get(tid, []))
            if count == 0:
                raise ValueError(f"Task {tid} has no candidate paths!")
            self.upper_bounds.append(count - 1)

        xl = np.zeros(self.n_tasks)
        xu = np.array(self.upper_bounds)

        super().__init__(
            n_var=self.n_tasks, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu, vtype=int
        )

    def _evaluate(self, x, out, *args, **kwargs):
        sol = Solution()
        for i, path_idx in enumerate(x):
            tid = self.task_ids[i]
            idx = int(path_idx)
            selected_path = self.candidate_paths_map[tid][idx]
            sol.path_selections[tid] = selected_path

        self.evaluator.evaluate(sol)

        # 针对 NSGA-II, SPEA2 的微小扰动 (Jitter)
        # 防止所有个体成本完全相同导致除零错误
        import random

        epsilon = 1e-6

        # 处理无穷大：防止计算报错
        f1 = sol.f1_risk if sol.f1_risk != float("inf") else 1e9
        f2 = sol.f2_cost if sol.f2_cost != float("inf") else 1e9

        # 添加相对噪音：Value * random(0, epsilon)
        # max(1.0, val) 确保即使值为0也能加上微小绝对噪音
        f1_jitter = random.uniform(0, epsilon * max(1.0, abs(f1)))
        f2_jitter = random.uniform(0, epsilon * max(1.0, abs(f2)))

        f1_noisy = f1 + f1_jitter
        f2_noisy = f2 + f2_jitter

        out["F"] = [f1_noisy, f2_noisy]
        out["G"] = [sol.constraint_violation]


class PymooSolver:
    """
    Pymoo 算法工厂
    """

    def __init__(self, network, evaluator, candidate_paths_map, config):
        self.network = network
        self.evaluator = evaluator
        self.candidate_paths_map = candidate_paths_map
        self.config = config

        algo_cfg = config.get("algorithm", {})
        self.pop_size = algo_cfg.get("population_size", 100)
        self.max_gen = algo_cfg.get("max_generations", 200)
        self.seed = config.get("experiment", {}).get("seed", 42)

        self.problem = PymooHazmatProblem(network, evaluator, candidate_paths_map)

    def run_algorithm(self, algo_name: str, save_history: bool = False):
        logging.info(f"--- Starting Baseline Solver: Pymoo {algo_name} ---")

        # 1. 选择算法
        if algo_name == "NSGA-II":
            algorithm = PymooNSGA2(
                pop_size=self.pop_size,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                mutation=PM(prob=0.05, eta=20, vtype=float, repair=RoundingRepair()),
                eliminate_duplicates=True,
            )
        elif algo_name == "SPEA2":
            algorithm = PymooSPEA2(
                pop_size=self.pop_size,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                mutation=PM(prob=0.05, eta=20, vtype=float, repair=RoundingRepair()),
                eliminate_duplicates=True,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        # 2. 运行优化
        termination = get_termination("n_gen", self.max_gen)

        res = minimize(
            self.problem,
            algorithm,
            termination,
            seed=self.seed,
            verbose=False,
            save_history=save_history,
        )

        logging.info(f"Pymoo {algo_name} finished. Found {len(res.opt)} solutions.")
        return res

    @staticmethod
    def convert_to_solutions(
        pymoo_result, solver_instance: "PymooSolver"
    ) -> List[Solution]:
        """
        [Helper] 将 Pymoo 的 Result 对象转换为本项目标准的 Solution 列表。
        """
        final_solutions = []
        if pymoo_result.opt is None:
            return []

        # Pymoo 的 X 是一个矩阵，每一行是一个解的变量向量
        X_matrix = np.atleast_2d(pymoo_result.opt.get("X"))
        task_ids = solver_instance.problem.task_ids

        for x_vec in X_matrix:
            sol = Solution()
            # 解码基因型
            for i, path_idx in enumerate(x_vec):
                tid = task_ids[i]
                idx = int(round(path_idx))
                path = solver_instance.candidate_paths_map[tid][idx]
                sol.path_selections[tid] = path

            # 重新评估以填充属性 (F1, F2, Feasibility)
            solver_instance.evaluator.evaluate(sol)
            if sol.is_feasible:
                final_solutions.append(sol)
        return final_solutions

    @staticmethod
    def extract_history_F(pymoo_result) -> List[np.ndarray]:
        """
        [Helper] 从 Pymoo 历史记录中提取每一代的目标值矩阵。
        """
        history_F = []
        if not pymoo_result.history:
            return []

        for snapshot in pymoo_result.history:
            F = snapshot.pop.get("F")
            G = snapshot.pop.get("G")
            # 过滤不可行解
            if G is not None:
                feasible_mask = (G <= 0).flatten()
                if np.any(feasible_mask):
                    history_F.append(F[feasible_mask])
                else:
                    history_F.append(np.empty((0, 2)))
            else:
                history_F.append(F)
        return history_F


class GurobiSolver:
    """
    精确解求解器 (Based on Weighted Sum Method)
    将问题建模为 MILP:
    - 预计算每条候选路径的 Risk, Cost, Capacity Usage, Pessimistic Cost
    - 通过改变权重 w, 求解 min w*Risk + (1-w)*Cost
    """

    def __init__(self, network, evaluator, candidate_paths_map, config):
        self.network = network
        self.evaluator = evaluator
        self.candidate_paths_map = candidate_paths_map
        self.config = config
        self.task_ids = sorted([t.task_id for t in network.tasks])

        self.path_metrics: Dict[str, List[Dict[str, Any]]] = {}
        #
        # 记录每项任务的 min/max 路径属性 (用于全局归一化)
        self.task_min_metrics: Dict[str, Dict[str, float]] = {}
        self.task_max_metrics: Dict[str, Dict[str, float]] = {}

        # 枢纽路径索引映射: {hub_id: [(task_id, path_idx, demand), ...]}
        self.hub_path_map: Dict[str, List[Tuple[str, int, float]]] = {}

        self._precompute_metrics()

    def _precompute_metrics(self):
        """预计算指标，并建立枢纽容量占用索引"""
        logging.info("Gurobi: Pre-computing metrics for all candidate paths...")
        alpha_c = self.evaluator.cost_config.get("fuzzy_cost_alpha_c", 0.90)

        # 初始化枢纽路径索引映射
        self.hub_path_map = {h.node_id: [] for h in self.network.get_hubs()}

        for tid in self.task_ids:
            paths = self.candidate_paths_map[tid]
            self.path_metrics[tid] = []
            task = self.network.get_task(tid)
            dv = task.demand

            for p_idx, path in enumerate(paths):
                # 1. Cost (Expected)
                cost_exp = 0.0
                for arc in path.arcs:
                    t_exp = FuzzyMath.triangular_expected_value(
                        *arc.fuzzy_transport_time
                    )
                    C_m = self.evaluator.unit_transport_cost.get(arc.mode, 0)
                    E_m = self.evaluator.unit_carbon_cost.get(arc.mode, 0)
                    P = self.evaluator.unit_operation_cost
                    cost_exp += ((C_m + E_m) * dv * arc.length) + P * t_exp

                for hub in path.transfer_hubs:
                    s_exp = FuzzyMath.trapezoidal_expected_value(
                        *hub.fuzzy_transshipment_time
                    )
                    B_k = self.evaluator.unit_transshipment_cost
                    I_k = self.evaluator.unit_transshipment_infra_cost
                    cost_exp += B_k * dv + I_k * s_exp

                # 2. Risk (CVaR)
                dummy_sol = Solution()
                dummy_sol.path_selections[tid] = path
                risk_cvar = self.evaluator._calculate_f1_cvar_risk(dummy_sol)

                # 3. Pessimistic Cost
                cost_pess = 0.0
                for arc in path.arcs:
                    t_pess = FuzzyMath.triangular_pessimistic_value(
                        *arc.fuzzy_transport_time, alpha_c
                    )
                    C_m = self.evaluator.unit_transport_cost.get(arc.mode, 0)
                    E_m = self.evaluator.unit_carbon_cost.get(arc.mode, 0)
                    P = self.evaluator.unit_operation_cost
                    cost_pess += ((C_m + E_m) * dv * arc.length) + P * t_pess

                for hub in path.transfer_hubs:
                    s_pess = FuzzyMath.trapezoidal_pessimistic_value(
                        *hub.fuzzy_transshipment_time, alpha_c
                    )
                    B_k = self.evaluator.unit_transshipment_cost
                    I_k = self.evaluator.unit_transshipment_infra_cost
                    cost_pess += B_k * dv + I_k * s_pess

                # 4. 容量占用索引记录
                path_key = (tid, p_idx, dv)  # (任务ID, 路径索引, 需求量)
                for hub in path.transfer_hubs:
                    self.hub_path_map[hub.node_id].append(path_key)

                # 5. 更新单任务的极值
                if tid not in self.task_min_metrics:
                    self.task_min_metrics[tid] = {"risk": risk_cvar, "cost": cost_exp}
                    self.task_max_metrics[tid] = {"risk": risk_cvar, "cost": cost_exp}
                else:
                    self.task_min_metrics[tid]["risk"] = min(
                        self.task_min_metrics[tid]["risk"], risk_cvar
                    )
                    self.task_max_metrics[tid]["risk"] = max(
                        self.task_max_metrics[tid]["risk"], risk_cvar
                    )
                    self.task_min_metrics[tid]["cost"] = min(
                        self.task_min_metrics[tid]["cost"], cost_exp
                    )
                    self.task_max_metrics[tid]["cost"] = max(
                        self.task_max_metrics[tid]["cost"], cost_exp
                    )

                self.path_metrics[tid].append(
                    {
                        "risk": risk_cvar,
                        "cost": cost_exp,
                        "pess_cost": cost_pess,
                        "path_obj": path,
                    }
                )

    def solve_weighted_sum(self, num_points: int = 10) -> List[Solution]:
        """
        通过改变权重 w (0 -> 1)，求解多次 MILP，得到 Pareto Frontier
        """
        logging.info(f"Gurobi: Starting Weighted Sum with {num_points} points...")
        solutions = []

        # 预算约束
        budget = self.evaluator.cost_config.get("fuzzy_cost_budget", 1e9)

        # 权重列表 (从纯 Cost 到纯 Risk)
        weights = np.linspace(0, 1, num_points)

        # 1. 计算全局 Ideal 和 Nadir 点 (基于单任务极值之和)
        total_min_risk = sum(
            self.task_min_metrics[tid]["risk"] for tid in self.task_ids
        )
        total_max_risk = sum(
            self.task_max_metrics[tid]["risk"] for tid in self.task_ids
        )
        total_min_cost = sum(
            self.task_min_metrics[tid]["cost"] for tid in self.task_ids
        )
        total_max_cost = sum(
            self.task_max_metrics[tid]["cost"] for tid in self.task_ids
        )

        # 归一化分母和理想点
        range_risk = max(total_max_risk - total_min_risk, 1e-6)
        range_cost = max(total_max_cost - total_min_cost, 1e-6)

        for w in weights:
            # 建立模型
            model = gp.Model("Hazmat_Routing")
            model.setParam("OutputFlag", 0)

            # 变量 x[tid][path_idx]
            x = {}
            for tid in self.task_ids:
                for p_idx in range(len(self.candidate_paths_map[tid])):
                    x[tid, p_idx] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_{tid}_{p_idx}"
                    )

            # Const 1: 选择唯一路径
            for tid in self.task_ids:
                model.addConstr(
                    gp.quicksum(
                        x[tid, p_idx]
                        for p_idx in range(len(self.candidate_paths_map[tid]))
                    )
                    == 1
                )

            # Const 2: 预算约束
            total_pess_cost = gp.quicksum(
                self.path_metrics[tid][p_idx]["pess_cost"] * x[tid, p_idx]
                for tid in self.task_ids
                for p_idx in range(len(self.candidate_paths_map[tid]))
            )
            model.addConstr(total_pess_cost <= budget, name="Budget")

            # Const 3.1: Arc Capacity 约束（单任务超限约束）
            # 单任务超限检查在 Candidate Path Generation 阶段就已经完成

            # Const 3.2: 枢纽容量约束 (累加约束)
            for hub_id, path_keys in self.hub_path_map.items():
                hub_node = self.network.get_node(hub_id)

                # 流量累加项：sum_{v,p: k in p} (Demand_v * x_{v,p})
                flow_through_hub = gp.quicksum(
                    path_key[2] * x[path_key[0], path_key[1]] for path_key in path_keys
                )

                # 容量约束: Total Flow <= Hub Capacity
                if hub_node:
                    model.addConstr(
                        flow_through_hub <= hub_node.capacity, name=f"Hub_Cap_{hub_id}"
                    )

            # 目标函数 (带归一化)
            obj_risk_raw = gp.quicksum(
                self.path_metrics[tid][p_idx]["risk"] * x[tid, p_idx]
                for tid in self.task_ids
                for p_idx in range(len(self.candidate_paths_map[tid]))
            )
            obj_cost_raw = gp.quicksum(
                self.path_metrics[tid][p_idx]["cost"] * x[tid, p_idx]
                for tid in self.task_ids
                for p_idx in range(len(self.candidate_paths_map[tid]))
            )

            # 归一化后的目标: (val - Ideal) / Range
            norm_risk = (obj_risk_raw - total_min_risk) / range_risk
            norm_cost = (obj_cost_raw - total_min_cost) / range_cost

            model.setObjective(
                w * norm_risk + (1 - w) * norm_cost,
                GRB.MINIMIZE,
            )

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = Solution()
                for tid in self.task_ids:
                    for p_idx in range(len(self.candidate_paths_map[tid])):
                        if x[tid, p_idx].X > 0.5:
                            path = self.path_metrics[tid][p_idx]["path_obj"]
                            sol.path_selections[tid] = path
                            break
                # 重新评估以填充所有目标值和可行性状态
                self.evaluator.evaluate(sol)
                sol.rank = 0
                solutions.append(sol)

        logging.info(f"Gurobi Weighted Sum found {len(solutions)} solutions.")
        return solutions
