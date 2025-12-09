# --- coding: utf-8 ---
# --- benchmark.py ---
"""
[Experiment Benchmark Script]
用于生成论文所需的 Figure 3, 4, 5, 6 和 Table 2。

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
from typing import List, Dict, Any

# --- Project Imports ---
from app.experiment_manager import Experiment
from app.core.solution import Solution
from app.core.baselines import PymooSolver, GurobiSolver, HAS_GUROBI
from app.utils.metrics import MetricCalculator, build_reference_front
from app.utils.result_keeper import setup_logging
from app.utils.plotter import BenchmarkPlotter

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

    # 设置 benchmark 输出目录
    benchmark_dir = os.path.join(exp.save_dir, "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)

    # 重定向日志到 benchmark.log
    setup_logging(log_dir=benchmark_dir, log_name="benchmark.log")

    # --- 读取 Config 中的实验配置 ---
    exp_config = exp.config.get("experiment", {})

    # 读取 n_runs，默认为 5
    n_runs = exp_config.get("n_runs", 5)

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
        algo: {"HV": [], "IGD": [], "SM": [], "Time": []}
        for algo in ["Improved NSGA-II", "NSGA-II", "SPEA2", "Gurobi"]
    }

    all_known_solutions_F: List[np.ndarray] = []
    last_run_history: Dict[str, List[List[Solution]]] = {}
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
        duration = time.process_time() - start

        feasible_prop = [s for s in proposed_pop if s.is_feasible]

        _record_run_data(
            run_idx=run_idx,
            n_total_runs=n_runs,
            algo_name="Improved NSGA-II",
            final_sols=feasible_prop,
            duration=duration,
            history_sols=logger.history_pop,
            stats_data=stats_data,
            all_F_list=all_known_solutions_F,
            last_run_hist=last_run_history,
            last_run_fin=last_run_finals,
        )

        # --- B. Baselines: NSGA-II & SPEA2 (via Pymoo) ---
        solver = PymooSolver(
            exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
        )
        for base_name in ["NSGA-II", "SPEA2"]:
            logging.info(f"Running {base_name}...")

            start = time.process_time()
            res = solver.run_algorithm(base_name, save_history=True)
            duration = time.process_time() - start

            finals = PymooSolver.convert_to_solutions(res, solver)
            history_F_list = PymooSolver.extract_history_F(res)
            history_sols = [
                [_make_dummy_sol(f) for f in F_gen] for F_gen in history_F_list
            ]

            _record_run_data(
                run_idx=run_idx,
                n_total_runs=n_runs,
                algo_name=base_name,
                final_sols=finals,
                duration=duration,
                history_sols=history_sols,
                stats_data=stats_data,
                all_F_list=all_known_solutions_F,
                last_run_hist=last_run_history,
                last_run_fin=last_run_finals,
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
                run_idx=run_idx,
                n_total_runs=n_runs,
                algo_name="Gurobi",
                final_sols=g_sols,
                duration=duration,
                history_sols=[g_sols],
                stats_data=stats_data,
                all_F_list=all_known_solutions_F,
                last_run_hist=last_run_history,
                last_run_fin=last_run_finals,
            )
        else:
            logging.warning("Skipping Gurobi (Not installed).")

    # -------------------------------------------------------
    # 4. 全局分析与绘图 (Global Analysis & Plotting)
    # -------------------------------------------------------
    _perform_global_analysis(
        stats_data=stats_data,
        all_F_list=all_known_solutions_F,
        last_hist=last_run_history,
        last_finals=last_run_finals,
        save_dir=benchmark_dir,
    )


# ===========================================================
# Internal Helpers (内部辅助函数)
# ===========================================================


def _make_dummy_sol(f_values: np.ndarray) -> Solution:
    s = Solution()
    if len(f_values) >= 2:
        s.f1_risk, s.f2_cost = f_values[0], f_values[1]
    return s


def _record_run_data(
    run_idx: int,
    n_total_runs: int,
    algo_name: str,
    final_sols: List[Solution],
    duration: float,
    history_sols: List[List[Solution]],
    stats_data: Dict[str, Any],
    all_F_list: List[np.ndarray],
    last_run_hist: Dict[str, List[List[Solution]]],
    last_run_fin: Dict[str, List[List[Solution]]],
):
    """
    记录单次运行数据。
    """
    if final_sols:
        F = np.array([[s.f1_risk, s.f2_cost] for s in final_sols])
        all_F_list.append(F)

    if "_raw_data" not in stats_data[algo_name]:
        stats_data[algo_name]["_raw_data"] = []

    stats_data[algo_name]["_raw_data"].append(
        {"finals": final_sols, "history": history_sols, "time": duration}
    )

    # 如果是最后一次运行，更新缓存
    if run_idx == n_total_runs - 1:
        last_run_hist[algo_name] = history_sols
        last_run_fin[algo_name] = final_sols


def _perform_global_analysis(
    stats_data: Dict[str, Any],
    all_F_list: List[np.ndarray],
    last_hist: Dict[str, List[List[Solution]]],
    last_finals: Dict[str, List[Solution]],
    save_dir: str,
):
    logging.info("\n>>> Calculating Global Bounds & Metrics...")

    if not all_F_list:
        logging.error("No feasible solutions generated across all runs!")
        return

    # 1. Global Normalization Bounds
    all_F = np.vstack(all_F_list)
    ideal = np.min(all_F, axis=0)
    nadir = np.max(all_F, axis=0)

    logging.info(f"Global Ideal (Min): {ideal}")
    logging.info(f"Global Nadir (Max): {nadir}")

    # 2. Reference Front
    ref_front = build_reference_front(all_F_list)
    logging.info(f"Reference Front constructed with {len(ref_front)} points.")

    # 3. Metric Calculator
    calc = MetricCalculator(ideal_point=ideal, nadir_point=nadir)

    # 4. Compute Metrics for All Runs
    for algo, data in stats_data.items():
        raw_list = data.pop("_raw_data", [])

        for run_record in raw_list:
            finals = run_record["finals"]

            hv = calc.calculate_hv(finals)
            igd = calc.calculate_igd(finals, ref_front)
            sm = calc.calculate_sm(finals)

            data["HV"].append(hv)
            data["IGD"].append(igd)
            data["SM"].append(sm)
            data["Time"].append(run_record["time"])

    # 5. Reporting
    logging.info("Generating Table 2...")
    rows = []
    for algo, metrics in stats_data.items():
        row = {"Algorithm": algo}
        for k in ["HV", "IGD", "SM", "Time"]:
            vals = metrics[k]
            if not vals:
                vals = [0.0]
            row[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    logging.info("\n" + df.to_string(index=False))
    df.to_csv(os.path.join(save_dir, "table_2_final.csv"), index=False)

    # 6. Plotting
    plotter = BenchmarkPlotter(save_dir)

    # 6.1 Convergence Curves (Last Run)
    hv_map, igd_map, sm_map = {}, {}, {}
    for algo, hist in last_hist.items():
        h_hv, h_igd, h_sm = [], [], []
        for pop in hist:
            if not pop:
                h_hv.append(0.0)
                h_igd.append(np.inf)
                h_sm.append(0.0)
                continue
            h_hv.append(calc.calculate_hv(pop))
            h_igd.append(calc.calculate_igd(pop, ref_front))
            h_sm.append(calc.calculate_sm(pop))
        hv_map[algo] = h_hv
        igd_map[algo] = h_igd
        sm_map[algo] = h_sm

    plotter.plot_convergence_curves(hv_map, "Hypervolume (HV)", "Figure_3_HV")
    plotter.plot_convergence_curves(igd_map, "IGD Metric", "Figure_4_IGD")
    plotter.plot_convergence_curves(sm_map, "Spacing Metric (SM)", "Figure_5_SM")

    # 6.2 Performance Comparison (Violin Plots)
    logging.info("Plotting Algorithm Comparison Violins...")

    # 直接传入整个 stats_data 字典
    plotter.plot_performance_comparison(stats_data)

    logging.info("Benchmark Finished Successfully! 🎉")


if __name__ == "__main__":
    main()
