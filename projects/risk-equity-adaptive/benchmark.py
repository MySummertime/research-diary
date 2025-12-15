# --- coding: utf-8 ---
# --- benchmark.py ---
"""
[Experiment Benchmark]

对比对象:
1. Improved NSGA-II
2. NSGA-II (via Pymoo)
3. SPEA2 (via Pymoo)
4. Gurobi - 作为 Reference Front 和 Pareto 对比基准

输出目录:
results/experiment_timestamp/benchmark/
"""

import time
import os
import random
import numpy as np
import pandas as pd
import logging
from typing import List, Dict

# --- Project Imports ---
from app.experiment_manager import Experiment
from app.core.solution import Solution
from app.core.baselines import PymooSolver, GurobiSolver, HAS_GUROBI
from app.utils.metrics import MetricCalculator, build_reference_front
from app.utils.result_keeper import setup_logging
from app.utils.plotter import ParetoPlotter, BenchmarkPlotter

# ===========================================================
# Helper Classes (辅助类)
# ===========================================================


class HistoryLogger:
    """
    [回调函数] 用于捕获算法每一代种群的完整状态。
    """

    def __init__(self):
        # 存储历史记录：每一代是一个 Solution 列表
        # List[List[Solution]]
        self.history_pop: List[List[Solution]] = []

    def on_generation_end(self, gen: int, population: List[Solution]):
        """
        每一代结束时触发。
        """
        # 深拷贝 (Clone) 可行解，确保历史记录不可变
        snapshot = [s.clone() for s in population if s.is_feasible]
        self.history_pop.append(snapshot)


# ===========================================================
# Main Execution Flow (主流程)
# ===========================================================


def main():
    # -------------------------------------------------------
    # 1. 环境初始化 (Setup)
    # -------------------------------------------------------
    exp = Experiment(config_path="config.json")

    # 获取预计算时间 (只加给基于池的算法)
    t_pre = exp.precompute_time

    # 设置 benchmark 输出目录
    benchmark_dir = os.path.join(exp.save_dir, "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)

    # 重定向日志到 benchmark.log
    setup_logging(log_dir=benchmark_dir, log_name="benchmark.log")

    # --- 读取 Config 中的实验配置 ---
    exp_config = exp.config.get("experiment", {})

    # 读取 n_runs，默认为 1
    n_runs = exp_config.get("n_runs", 1)

    # 读取 seeds 列表，默认为空
    seeds_list = exp_config.get("seeds", [])

    logging.info("==========================================")
    logging.info("   STARTING RIGOROUS BENCHMARK")
    logging.info(f"   Total Runs: {n_runs}")
    logging.info(f"   Seeds Config: {seeds_list}")
    logging.info("==========================================")

    # 鲁棒性检查：确保种子够用
    if len(seeds_list) < n_runs:
        logging.warning(
            f"⚠️ Config Warning: Provided seeds ({len(seeds_list)}) are fewer than n_runs ({n_runs}). "
            f"Generating extra random seeds..."
        )
        # 补齐种子
        while len(seeds_list) < n_runs:
            seeds_list.append(random.randint(0, 999999))
        logging.info(f"   Final Seeds List: {seeds_list}")

    # -------------------------------------------------------
    # 2. 数据容器初始化 (Initialize Containers)
    # -------------------------------------------------------

    stats_data = {
        algo: {"HV": [], "IGD": [], "SM": [], "CPU Time": []}
        for algo in ["Improved NSGA-II", "NSGA-II", "SPEA2", "Gurobi"]
    }

    all_known_solutions_F: List[np.ndarray] = []
    last_run_finals: Dict[str, List[Solution]] = {}

    # -------------------------------------------------------
    # 3. 执行循环 (Execution Loop)
    # -------------------------------------------------------
    for run_idx in range(n_runs):
        # 获取当前种子
        current_seed = seeds_list[run_idx]

        logging.info(
            f"\n>>> Starting Run {run_idx + 1}/{n_runs} [Seed: {current_seed}]..."
        )

        # 设置随机种子
        random.seed(current_seed)
        np.random.seed(current_seed)

        # --- A. Proposed Algorithm: Improved NSGA-II ---
        logging.info("Running Improved NSGA-II...")
        logger = HistoryLogger()
        start = time.process_time()

        # 强制不传入 initial_population 以确保随机初始化
        # 此时内部的 random.choice 会受上面的 seed 影响
        proposed_pop = exp.algorithm.run(callbacks=[logger], initial_population=None)
        duration = time.process_time() - start + t_pre

        # 同时过滤可行性 AND Rank 0 (非支配解)
        feasible_prop = [s for s in proposed_pop if s.is_feasible and s.rank == 0]

        _record_run_data(
            run_idx,
            n_runs,
            "Improved NSGA-II",
            feasible_prop,
            duration,
            stats_data,
            all_known_solutions_F,
            last_run_finals,
        )

        # --- B. Baselines: NSGA-II & SPEA2 (via Pymoo) ---
        solver = PymooSolver(
            exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
        )
        for base_name in ["NSGA-II", "SPEA2"]:
            logging.info(f"Running {base_name}...")

            start = time.process_time()
            res = solver.run_algorithm(base_name, save_history=True)
            duration = time.process_time() - start + t_pre

            finals = PymooSolver.convert_to_solutions(res, solver)

            _record_run_data(
                run_idx,
                n_runs,
                base_name,
                finals,
                duration,
                stats_data,
                all_known_solutions_F,
                last_run_finals,
            )

        # --- C. Exact Baseline: Gurobi ---
        if HAS_GUROBI:
            logging.info("Running Gurobi...")
            g_solver = GurobiSolver(
                exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
            )

            start = time.process_time()
            g_sols = g_solver.solve_weighted_sum(num_points=200)
            duration = time.process_time() - start

            _record_run_data(
                run_idx,
                n_runs,
                "Gurobi",
                g_sols,
                duration,
                stats_data,
                all_known_solutions_F,
                last_run_finals,
            )
        else:
            logging.warning("Skipping Gurobi (Not installed).")

    # -------------------------------------------------------
    # 4. 全局分析与绘图 (Global Analysis & Plotting)
    # -------------------------------------------------------
    _perform_global_analysis(
        stats_data, all_known_solutions_F, last_run_finals, benchmark_dir
    )

    logging.info("Benchmark Finished Successfully! 🎉")


# ===========================================================
# Internal Helpers (内部辅助函数)
# ===========================================================


def _record_run_data(
    run_idx,
    n_total,
    algo_name,
    final_sols,
    duration,
    stats_data,
    all_F_list,
    last_run_finals,
):
    if final_sols:
        F = np.array([[s.f1_risk, s.f2_cost] for s in final_sols])
        all_F_list.append(F)

    if "_raw_data" not in stats_data[algo_name]:
        stats_data[algo_name]["_raw_data"] = []
    stats_data[algo_name]["_raw_data"].append({"finals": final_sols, "time": duration})

    if run_idx == n_total - 1:
        last_run_finals[algo_name] = final_sols


def _perform_global_analysis(stats_data, all_F_list, last_finals, save_dir):
    logging.info("\n>>> Calculating Global Bounds & Metrics...")
    if not all_F_list:
        logging.error("No feasible solutions found!")
        return

    all_F = np.vstack(all_F_list)
    ideal, nadir = np.min(all_F, axis=0), np.max(all_F, axis=0)
    ref_front = build_reference_front(all_F_list)
    calc = MetricCalculator(ideal_point=ideal, nadir_point=nadir)

    for algo, data in stats_data.items():
        raw_list = data.pop("_raw_data", [])
        for rec in raw_list:
            finals = rec["finals"]
            data["HV"].append(calc.calculate_hv(finals))
            data["IGD"].append(calc.calculate_igd(finals, ref_front))
            data["SM"].append(calc.calculate_sm(finals))
            data["CPU Time"].append(rec["time"])

    # Table Final Metrics
    logging.info("Generating Table Final Metrics...")
    rows = []
    for algo, metrics in stats_data.items():
        row = {"Algorithm": algo}
        for k in ["HV", "IGD", "SM", "CPU Time"]:
            vals = metrics[k]
            row[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if vals else "N/A"
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "final_metrics.csv"), index=False)

    plotter = BenchmarkPlotter(save_dir)

    # Violin
    logging.info("Plotting Comparison Violins...")
    plotter.plot_metrics_comparison(stats_data)

    # Pareto Frontier comparison
    logging.info("Plotting Pareto Comparison...")
    pareto_plotter = ParetoPlotter(save_dir=save_dir)
    pareto_plotter.plot_frontier_comparison_by_algo(
        frontiers=last_finals, file_name="pareto_frontier_comparison_by_algo.svg"
    )


if __name__ == "__main__":
    main()
