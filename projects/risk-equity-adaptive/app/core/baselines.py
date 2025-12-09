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
from typing import List, Dict

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

# Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB

    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

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
        out["F"] = [sol.f1_risk, sol.f2_cost]
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


class GurobiSolver:
    """
    精确解求解器 (Based on Weighted Sum Method)
    将问题建模为 MILP:
    - 预计算每条候选路径的 Risk, Cost, Capacity Usage, Pessimistic Cost
    - 通过改变权重 w, 求解 min w*Risk + (1-w)*Cost
    """

    def __init__(self, network, evaluator, candidate_paths_map, config):
        if not HAS_GUROBI:
            raise ImportError("Gurobipy not installed. Cannot use GurobiSolver.")

        self.network = network
        self.evaluator = evaluator
        self.candidate_paths_map = candidate_paths_map
        self.config = config
        self.task_ids = sorted([t.task_id for t in network.tasks])

        # 预计算数据缓存
        self.path_metrics = {}  # {tid: [{risk, cost, pess_cost, arcs:[], hubs:[]}]}
        self._precompute_metrics()

    def _precompute_metrics(self):
        """预先计算所有候选路径的指标，避免在 Gurobi 建模时重复调用 evaluate"""
        logging.info("Gurobi: Pre-computing metrics for all candidate paths...")

        alpha_c = self.evaluator.cost_config.get("fuzzy_cost_alpha_c", 0.90)

        for tid in self.task_ids:
            paths = self.candidate_paths_map[tid]
            self.path_metrics[tid] = []
            task = self.network.get_task(tid)
            dv = task.demand

            for p_idx, path in enumerate(paths):
                # 构造临时解只包含这一条路径，利用 Evaluator 的私有方法计算单任务指标
                # 为了不破坏 Evaluator 封装，我们这里手动复刻部分逻辑，或者
                # 更好的方式：利用我们在 analyzer.py 里写的 _calculate_single_task_cost
                # 这里为了独立性，我们在 GurobiSolver 内部快速算一遍

                # 1. Cost (Expected)
                cost_exp = 0.0
                for arc in path.arcs:
                    # 简化逻辑，直接调用 fuzzy math
                    t_exp = FuzzyMath.triangular_expected_value(
                        *arc.fuzzy_transport_time
                    )
                    C_m = self.evaluator.unit_transport_cost.get(arc.mode, 0)
                    E_m = self.evaluator.unit_carbon_cost.get(arc.mode, 0)
                    P = self.evaluator.unit_penalty_cost
                    cost_exp += dv * ((C_m + E_m) * arc.length + P * t_exp)

                for hub in path.transfer_hubs:
                    s_exp = FuzzyMath.trapezoidal_expected_value(
                        *hub.fuzzy_transshipment_time
                    )
                    B_k = self.evaluator.unit_transshipment_cost
                    I_k = self.evaluator.unit_transshipment_infra_cost
                    cost_exp += dv * (B_k + I_k * s_exp)

                # 2. Risk (CVaR)
                # 利用 Evaluator 的 public/protected 接口
                # 我们暂时构造一个只含该任务的 dummy solution
                dummy_sol = Solution()
                dummy_sol.path_selections[tid] = path
                risk_cvar = self.evaluator._calculate_f1_cvar_risk(dummy_sol)

                # 3. Pessimistic Cost (for Constraint)
                cost_pess = 0.0
                for arc in path.arcs:
                    t_pess = FuzzyMath.triangular_pessimistic_value(
                        *arc.fuzzy_transport_time, alpha_c
                    )
                    C_m = self.evaluator.unit_transport_cost.get(arc.mode, 0)
                    E_m = self.evaluator.unit_carbon_cost.get(arc.mode, 0)
                    P = self.evaluator.unit_penalty_cost
                    cost_pess += dv * ((C_m + E_m) * arc.length + P * t_pess)

                for hub in path.transfer_hubs:
                    s_pess = FuzzyMath.trapezoidal_pessimistic_value(
                        *hub.fuzzy_transshipment_time, alpha_c
                    )
                    B_k = self.evaluator.unit_transshipment_cost
                    I_k = self.evaluator.unit_transshipment_infra_cost
                    cost_pess += dv * (B_k + I_k * s_pess)

                # 4. Resource Usage (Arc flows)
                arc_usage = []  # List of (u, v, demand)
                for arc in path.arcs:
                    arc_usage.append((arc.start.node_id, arc.end.node_id, dv))

                hub_usage = []
                for hub in path.transfer_hubs:
                    hub_usage.append((hub.node_id, dv))

                self.path_metrics[tid].append(
                    {
                        "risk": risk_cvar,
                        "cost": cost_exp,
                        "pess_cost": cost_pess,
                        "arc_usage": arc_usage,
                        "hub_usage": hub_usage,
                        "path_obj": path,
                    }
                )

    def solve_weighted_sum(self, num_points: int = 10) -> List[Solution]:
        """
        通过改变权重 w (0 -> 1)，求解多次 MILP，得到近似 Pareto 前沿
        """
        logging.info(f"Gurobi: Starting Weighted Sum with {num_points} points...")
        solutions = []

        # 预算约束
        budget = self.evaluator.cost_config.get("fuzzy_cost_budget", 1e9)

        # 权重列表 (从纯Cost 到 纯Risk)
        weights = np.linspace(0, 1, num_points)

        for w in weights:
            # 建立模型
            model = gp.Model("Hazmat_Routing")
            model.setParam("OutputFlag", 0)  # 静默模式

            # 变量 x[tid][path_idx]
            x = {}
            for tid in self.task_ids:
                for p_idx in range(len(self.candidate_paths_map[tid])):
                    x[tid, p_idx] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_{tid}_{p_idx}"
                    )

            # 约束 1: 每个任务选且仅选一条路径
            for tid in self.task_ids:
                model.addConstr(
                    gp.quicksum(
                        x[tid, p_idx]
                        for p_idx in range(len(self.candidate_paths_map[tid]))
                    )
                    == 1
                )

            # 约束 2: 预算约束 (Pessimistic Cost <= Budget)
            total_pess_cost = gp.quicksum(
                self.path_metrics[tid][p_idx]["pess_cost"] * x[tid, p_idx]
                for tid in self.task_ids
                for p_idx in range(len(self.candidate_paths_map[tid]))
            )
            model.addConstr(total_pess_cost <= budget, name="Budget")

            # 约束 3: 容量约束 (略微复杂，需要聚合所有任务的流量)
            # 为了简化，我们暂时只对 Budget 做硬约束。
            # 如果加上 Arc Capacity，模型构建会变慢，但逻辑是一样的。
            # 鉴于 benchmark 主要是看目标函数，Capacity 可以先作为 Soft Constraint 或者
            # 如果你的算例比较紧，必须加。这里我加上 Arc Capacity。

            # 预处理所有 Arc 的容量
            arc_vars = {}  # (u, v) -> expression
            for tid in self.task_ids:
                for p_idx in range(len(self.candidate_paths_map[tid])):
                    for u, v, dv in self.path_metrics[tid][p_idx]["arc_usage"]:
                        if (u, v) not in arc_vars:
                            arc_vars[u, v] = 0
                        arc_vars[u, v] += dv * x[tid, p_idx]

            for (u, v), flow_expr in arc_vars.items():
                arc_obj = self.network.get_arc(u, v)
                if arc_obj and arc_obj.capacity < 1e9:  # 只添加有意义的容量约束
                    model.addConstr(flow_expr <= arc_obj.capacity, name=f"Cap_{u}_{v}")

            # 目标函数: min w * Risk + (1-w) * Cost
            # 注意数量级差异！Risk 可能是 1e3, Cost 可能是 1e5。
            # 最好归一化，或者让 w 偏向 Cost。
            # 这里直接线性加权
            obj_risk = gp.quicksum(
                self.path_metrics[tid][p_idx]["risk"] * x[tid, p_idx]
                for tid in self.task_ids
                for p_idx in range(len(self.candidate_paths_map[tid]))
            )

            obj_cost = gp.quicksum(
                self.path_metrics[tid][p_idx]["cost"] * x[tid, p_idx]
                for tid in self.task_ids
                for p_idx in range(len(self.candidate_paths_map[tid]))
            )

            model.setObjective(w * obj_risk + (1 - w) * obj_cost, GRB.MINIMIZE)

            # 求解
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                # 提取解
                sol = Solution()
                for tid in self.task_ids:
                    for p_idx in range(len(self.candidate_paths_map[tid])):
                        if x[tid, p_idx].X > 0.5:
                            path = self.path_metrics[tid][p_idx]["path_obj"]
                            sol.path_selections[tid] = path
                            break

                # 重新评估以填充所有属性
                self.evaluator.evaluate(sol)
                sol.rank = 0  # 假定为最优
                solutions.append(sol)

        logging.info(f"Gurobi Weighted Sum found {len(solutions)} solutions.")
        return solutions
