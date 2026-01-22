# --- coding: utf-8 ---
# --- benchmark.py ---
"""
[Experiment Benchmark & Ablation Study]

对比对象 (共 7 种):
1. Improved NSGA-II (Proposed): 基于全量路径池
2. NSGA-II (1): 仅使用 K-shortest Path (退化版 1)
3. NSGA-II (2): 追加策略 1 (Min Cost) (退化版 2)
4. NSGA-II (3): 追加策略 2 (Fastest Response) (退化版 3)
6. SPEA2 (via Pymoo): 基于全量路径池
7. Gurobi: 基于全量路径池
"""

import logging
import os
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd

# --- Project Imports ---
from app.core.baselines import GurobiSolver, PymooSolver
from app.core.path import PathFinder
from app.core.path_large_scale import LargeScalePathFinder
from app.core.solution import Solution
from app.experiment_manager import Experiment
from app.utils.metrics import MetricCalculator, build_reference_front
from app.utils.result_keeper import setup_logging

# ===========================================================
# Main Execution Flow
# ===========================================================


def main():
    # -------------------------------------------------------
    # 1. 环境初始化
    # -------------------------------------------------------
    # 显式指定大规模实验数据目录，与主实验隔离
    exp = Experiment(config_path="config.json")

    # 15-50 节点设为 False, 100+ 节点设为 True
    USE_LARGE_SCALE = len(exp.network.nodes) > 50

    benchmark_dir = os.path.join(exp.save_dir, "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)
    setup_logging(log_dir=benchmark_dir, log_name="benchmark.log")

    exp_config = exp.config.get("experiment", {})
    n_runs = exp_config.get("n_runs", 5)
    seeds_list = exp_config.get("seeds", [14, 44, 47, 74, 94])

    # 非对称参数矩阵配置
    # 格式: (k_val, pop_size, max_gen, ablation_level)
    ALGO_CONFIGS = {
        "NSGA-II (Imp)": (20, 300, 400, "full"),  # 完整版（含混合采样）
        "NSGA-II (1)": (3, 80, 90, 1),  # Shortest
        "NSGA-II (2)": (4, 100, 100, 2),  # Shortest + Risk
        "NSGA-II (3)": (7, 140, 170, 3),  # Shortest + Risk + Response
        "SPEA2": (20, 300, 400, "full"),
    }

    ALGO_LIST = list(ALGO_CONFIGS.keys()) + ["Gurobi"]
    logging.info(
        f"STARTING LARGE-SCALE BENCHMARK | Tasks: {len(exp.network.tasks)} | Runs: {n_runs}"
    )

    # -------------------------------------------------------
    # 2. 数据容器初始化
    # -------------------------------------------------------
    stats_data = {
        algo: {
            "HV": [],
            "IGD": [],
            "SM": [],
            "MS": [],
            "PD": [],
            "CPU Time": [],
            "_raw_data": [],
        }
        for algo in ALGO_LIST
    }
    all_known_solutions_F: List[np.ndarray] = []
    last_run_finals: Dict[str, List[Solution]] = {}

    # -------------------------------------------------------
    # 3. 执行循环 (Execution Loop)
    # -------------------------------------------------------
    for run_idx in range(n_runs):
        current_seed = seeds_list[run_idx]
        logging.info(f"\n>>> Run {run_idx + 1}/{n_runs} [Seed: {current_seed}]")
        random.seed(current_seed)
        np.random.seed(current_seed)

        # 在算法循环外，根据选项实例化对应的 Finder，该 Run 内所有算法共享缓存
        if USE_LARGE_SCALE:
            logging.info("🚀 使用 LargeScalePathFinder (并行+缓存+剪枝)")
            pf = LargeScalePathFinder(exp.network, exp.evaluator)
        else:
            logging.info("🐢 使用标准 PathFinder (单线程)")
            pf = PathFinder(exp.network, exp.evaluator)

        # 生成本轮的全量池
        full_pool = pf.find_all_candidate_paths()

        # 同步给 exp 对象，供 Gurobi 使用
        exp.candidate_paths_map = full_pool

        # --- A. 启发式与消融算法 ---
        for algo_tag, config in ALGO_CONFIGS.items():
            k_val, pop_v, gen_v, level_v = config
            logging.info(
                f"Running {algo_tag} | Seed: {current_seed} | Pop: {pop_v} | Gen: {gen_v}"
            )

            if level_v == "full":
                current_pool = full_pool
            else:
                current_pool = pf._get_ablation_paths(k_val, level_v)

            start_time = time.process_time()
            if algo_tag == "NSGA-II (Imp)":
                # 注入大规模实验参数
                exp.algorithm.pop_size = pop_v
                exp.algorithm.max_gen = gen_v
                # 注入当前生成的路径池
                exp.candidate_paths_map = current_pool
                res_pop = exp.algorithm.run(initial_population=None)
                finals = [s for s in res_pop if s.is_feasible and s.rank == 0]
                # 时间计入预计算开销
                duration = time.process_time() - start_time + exp.precompute_time
            else:
                solver = PymooSolver(
                    exp.network, exp.evaluator, current_pool, exp.config
                )
                solver.pop_size = pop_v
                solver.max_gen = gen_v
                res = solver.run_algorithm(
                    "SPEA2" if algo_tag == "SPEA2" else "NSGA-II"
                )
                finals = PymooSolver.convert_to_solutions(res, solver)
                duration = time.process_time() - start_time

            # 记录数据并加入 Jitter 以适配节点量级
            is_ablation = "NSGA-II" in algo_tag and "Imp" not in algo_tag
            _record_run_data(
                run_idx,
                n_runs,
                algo_tag,
                finals,
                duration,
                stats_data,
                all_known_solutions_F,
                last_run_finals,
                jitter=is_ablation,
            )

        # --- B. 精确解基准: Gurobi ---
        logging.info("Running Exact: Gurobi (Full Pool)...")
        # 直接使用 exp.candidate_paths_map，它已经在本轮开始时同步为 full_pool
        g_solver = GurobiSolver(
            exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
        )
        start_g = time.process_time()
        g_sols = g_solver.solve_weighted_sum(num_points=1500)
        duration_g = time.process_time() - start_g
        _record_run_data(
            run_idx,
            n_runs,
            "Gurobi",
            g_sols,
            duration_g,
            stats_data,
            all_known_solutions_F,
            last_run_finals,
            jitter=False,
        )

    # ------------------------------------------------------------------
    # 4. 全局分析与可视化 (Radar + Violin + Grouped Bar Chart + Pareto)
    # ------------------------------------------------------------------
    _perform_global_analysis(
        stats_data, all_known_solutions_F, last_run_finals, benchmark_dir
    )
    logging.info("Ablation Benchmark Finished Successfully! 🏁💎")


def _record_run_data(
    run_idx,
    n_total,
    algo_name,
    final_sols,
    duration,
    stats_data,
    all_F_list,
    last_run_finals,
    jitter=False,
):
    """记录运行数据。"""
    if final_sols:
        F = np.array([[s.f1_risk, s.f2_cost] for s in final_sols])
        all_F_list.append(F)

    stats_data[algo_name]["_raw_data"].append({"finals": final_sols, "time": duration})
    if run_idx == n_total - 1:
        last_run_finals[algo_name] = final_sols


def _perform_global_analysis(stats_data, all_F_list, last_finals, save_dir):
    """
    负责计算指标并导出包含所有 Run 的详细 CSV
    """
    logging.info("\n>>> Phase 1: Metric Calculation & CSV Export...")

    all_F = np.vstack(all_F_list)
    ideal, nadir = np.min(all_F, axis=0), np.max(all_F, axis=0)
    ref_front = build_reference_front(all_F_list)
    calc = MetricCalculator(ideal_point=ideal, nadir_point=nadir)

    # 1. 构造详细记录表 (用于后续绘图，必须保留单次 Run 的数据)
    detailed_rows = []
    for algo, storage in stats_data.items():
        raw_list = storage.get("_raw_data", [])
        for run_idx, rec in enumerate(raw_list):
            finals = rec["finals"]
            row = {
                "Algorithm": algo,
                "Run": run_idx,
                "HV": calc.calculate_hv(finals),
                "IGD": calc.calculate_igd(finals, ref_front),
                "SM": calc.calculate_sm(finals),
                "PD": calc.calculate_pd(finals),
                "MS": calc.calculate_ms(finals, ref_front),
                "CPU Time": rec["time"],
            }
            detailed_rows.append(row)

    # 导出详细 CSV
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(os.path.join(save_dir, "detailed_metrics_raw.csv"), index=False)

    # 导出汇总 CSV (均值±标准差)
    summary_rows = []
    for algo in stats_data.keys():
        algo_df = detailed_df[detailed_df["Algorithm"] == algo]
        res = {"Algorithm": algo}

        for m in ["HV", "IGD", "SM", "PD", "MS", "CPU Time"]:
            # 显式检查样本量
            mean_val = algo_df[m].mean()
            std_val = algo_df[m].std()

            # 如果 std 是 nan (单样本或无数据)，替换为 0.0000 以保持表格整洁
            if pd.isna(std_val):
                std_val = 0.0
            if pd.isna(mean_val):
                mean_val = 0.0

            res[m] = f"{mean_val:.4f} ± {std_val:.4f}"

        summary_rows.append(res)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(save_dir, "final_metrics_summary.csv"), index=False
    )

    logging.info(f"✅ CSV results saved to {save_dir}.")


if __name__ == "__main__":
    main()
