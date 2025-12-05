# --- coding: utf-8 ---
# --- benchmark.py ---
"""
[Experiment Benchmark Script]
用于生成论文所需的 Figure 3, 4, 5, 6 和 Table 2。

对比对象:
1. Proposed Improved NSGA-II
2. Standard NSGA-II (via Pymoo)
3. Standard SPEA2 (via Pymoo)
4. Gurobi (Exact) - 作为 Reference Front 和 Pareto 对比基准

输出目录:
results/experiment_timestamp/benchmark/
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
from typing import List, Dict

# --- Project Imports ---
from app.experiment_manager import Experiment
from app.core.solution import Solution
from app.core.baselines import PymooSolver, GurobiSolver, HAS_GUROBI
from app.utils.metrics import MetricCalculator
from app.utils.result_keeper import setup_logging
from app.core.moea_utils import MOEAUtils


# -----------------------------------------------------------
# Helper Classes & Functions
# -----------------------------------------------------------


class HistoryLogger:
    """
    用于捕获 Proposed Algorithm 每一代种群的回调函数。
    我们需要存下每一代的 F (目标值) 用于后续计算指标历史。
    """

    def __init__(self):
        # 存储每一代种群的目标值矩阵 List[np.ndarray]
        self.history_F = []

    def on_generation_end(self, gen: int, population: List[Solution]):
        # 提取目标值 F (保留所有解以计算指标)
        F = np.array([[s.f1_risk, s.f2_cost] for s in population])
        self.history_F.append(F)

        if gen % 10 == 0:
            logging.info(f"[Proposed] Gen {gen} captured.")


def convert_pymoo_solutions(pymoo_result, solver: PymooSolver) -> List[Solution]:
    """将 Pymoo 结果转回 Solution 对象"""
    final_solutions = []
    X_matrix = np.atleast_2d(pymoo_result.opt.get("X"))
    task_ids = solver.problem.task_ids

    for x_vec in X_matrix:
        sol = Solution()
        for i, path_idx in enumerate(x_vec):
            tid = task_ids[i]
            idx = int(round(path_idx))
            path = solver.candidate_paths_map[tid][idx]
            sol.path_selections[tid] = path

        solver.evaluator.evaluate(sol)
        if sol.is_feasible:
            final_solutions.append(sol)

    return final_solutions


def extract_history_F_from_pymoo(res) -> List[np.ndarray]:
    """从 Pymoo 历史中提取每一代的 F 矩阵"""
    history_F = []
    for snapshot in res.history:
        F = snapshot.pop.get("F")
        G = snapshot.pop.get("G")

        # 简单过滤: 如果有 G，只留可行解
        if G is not None:
            feasible = (G <= 0).flatten()
            if np.any(feasible):
                history_F.append(F[feasible])
            else:
                history_F.append(np.empty((0, 2)))
        else:
            history_F.append(F)
    return history_F


def build_reference_front(all_F_arrays: List[np.ndarray]) -> np.ndarray:
    """构建 IGD 计算所需的 Reference Front (True PF)"""
    if not all_F_arrays:
        return np.empty((0, 2))

    # 1. 合并所有点
    combined_F = np.vstack(all_F_arrays)
    # 2. 去重
    combined_F = np.unique(combined_F, axis=0)

    # 3. 非支配排序提取 Rank 0
    temp_sols = []
    for f in combined_F:
        s = Solution()
        s.f1_risk, s.f2_cost = f[0], f[1]
        temp_sols.append(s)

    fronts = MOEAUtils.fast_non_dominated_sort(temp_sols)

    if not fronts:
        return np.empty((0, 2))

    rank0_sols = fronts[0]
    ref_front = np.array([[s.f1_risk, s.f2_cost] for s in rank0_sols])
    # 按 Risk 排序，方便后续处理
    ref_front = ref_front[ref_front[:, 0].argsort()]

    return ref_front


# -----------------------------------------------------------
# Main Benchmark Logic
# -----------------------------------------------------------


def main():
    logging.info("==========================================")
    logging.info("   STARTING COMPREHENSIVE BENCHMARK       ")
    logging.info("==========================================")

    # 1. 初始化实验环境
    exp = Experiment(config_path="config.json")
    config = exp.config

    # 设置 benchmark 输出目录
    # 路径: results/experiment_timestamp/benchmark/
    benchmark_dir = os.path.join(exp.save_dir, "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)
    logging.info(f"Benchmark Results will be saved to: {benchmark_dir}")

    # 日志系统 -> benchmark.log
    # 将后续所有的 logging.info/error 输出到 benchmark/benchmark.log
    setup_logging(log_dir=benchmark_dir, log_name="benchmark.log")
    logging.info(f"Benchmark Results will be saved to: {benchmark_dir}")
    logging.info("Log redirected to benchmark.log successfully.")

    # HV 参考点
    # 注意：务必确保该点大于所有可能的最差解
    ref_point = config.get("analysis", {}).get("hv_ref_point", [300000, 2000000])
    logging.info(f"Metrics Reference Point: {ref_point}")

    calculator = MetricCalculator(ref_point)

    # 存储最终的统计数据
    results_data = {
        "Algorithm": [],
        "HV": [],
        "IGD": [],
        "SM": [],
        "CPU Time (s)": [],
    }

    # 存储过程数据用于画图 {"AlgoName": [F_gen0, F_gen1, ...]}
    raw_history_map = {}
    # 用于存储各算法的最终 Pareto 前沿，用于画 Figure 6
    final_frontiers: Dict[str, List[Solution]] = {}

    # -------------------------------------------------------
    # A. Proposed Algorithm
    # -------------------------------------------------------
    algo_name = "Improved NSGA-II (Proposed)"
    logging.info(f">>> Running {algo_name}...")

    logger = HistoryLogger()

    # [CPU Time] Start
    start_time = time.perf_counter()
    proposed_final_pop = exp.algorithm.run(callbacks=[logger])
    # [CPU Time] End
    end_time = time.perf_counter()
    duration_proposed = end_time - start_time

    raw_history_map[algo_name] = logger.history_F

    # 提取可行解中的 rank 0 作为最终前沿
    feasible_prop = [s for s in proposed_final_pop if s.is_feasible and s.rank == 0]
    final_frontiers[algo_name] = feasible_prop

    logging.info(f"{algo_name} Finished in {duration_proposed:.2f}s")

    # -------------------------------------------------------
    # B. Baselines (Pymoo)
    # -------------------------------------------------------
    solver = PymooSolver(exp.network, exp.evaluator, exp.candidate_paths_map, config)
    pymoo_finals = {}
    durations = {}

    # 这里对比 NSGA2 和 SPEA2
    for base_name in ["NSGA2", "SPEA2"]:
        label = f"Standard {base_name}"
        logging.info(f">>> Running {label}...")

        # [CPU Time] Start
        start_time = time.perf_counter()
        # 注意：开启 save_history 以获取收敛过程
        res = solver.run_algorithm(base_name, save_history=True)
        # [CPU Time] End
        end_time = time.perf_counter()
        durations[label] = end_time - start_time

        raw_history_map[label] = extract_history_F_from_pymoo(res)

        # 转换并存储最终前沿
        sols = convert_pymoo_solutions(res, solver)
        pymoo_finals[label] = sols
        final_frontiers[label] = sols

        logging.info(f"{label} Finished in {durations[label]:.2f}s")

    # -------------------------------------------------------
    # C. Exact Baseline (Gurobi)
    # -------------------------------------------------------
    gurobi_sols = []
    if HAS_GUROBI:
        label = "Gurobi (Exact)"
        logging.info(f">>> Running {label}...")
        try:
            g_solver = GurobiSolver(
                exp.network, exp.evaluator, exp.candidate_paths_map, config
            )

            # [CPU Time] Start
            start_time = time.perf_counter()
            # 跑 300 个采样点，以获得更密集的参考前沿
            gurobi_sols = g_solver.solve_weighted_sum(num_points=300)
            # [CPU Time] End
            end_time = time.perf_counter()
            durations[label] = end_time - start_time

            pymoo_finals[label] = gurobi_sols
            final_frontiers[label] = gurobi_sols

            logging.info(f"{label} Finished in {durations[label]:.2f}s")
        except Exception as e:
            logging.error(f"Gurobi failed: {e}")
            durations[label] = 0.0
    else:
        logging.warning("Skipping Gurobi (gurobipy not installed).")

    # -------------------------------------------------------
    # D. Build Reference Front & Calculate Metrics
    # -------------------------------------------------------
    logging.info(">>> Building Reference Front for IGD Calculation...")

    all_final_F_list = []

    # 将所有算法（包括 Gurobi）的解都加入 Ref Front 构建池
    # 这样能保证 Reference Front 是当前所有已知解中的最优集合
    for name, sols in final_frontiers.items():
        if sols:
            f_arr = np.array([[s.f1_risk, s.f2_cost] for s in sols])
            all_final_F_list.append(f_arr)

    reference_front = build_reference_front(all_final_F_list)
    logging.info(f"Reference Front constructed with {len(reference_front)} points.")

    # -------------------------------------------------------
    # E. Calculate Curves (HV, IGD, SM)
    # -------------------------------------------------------
    logging.info(">>> Calculating Convergence Curves (HV, IGD, SM)...")

    hv_curves = {}
    igd_curves = {}
    sm_curves = {}

    for algo, history in raw_history_map.items():
        hv_list = []
        igd_list = []
        sm_list = []

        for F_gen in history:
            if len(F_gen) == 0:
                hv_list.append(0.0)
                igd_list.append(float("inf"))
                sm_list.append(0.0)
                continue

            # 临时构造 Solution 列表给 calculator 用
            temp_sols = []
            for val in F_gen:
                s = Solution()
                s.f1_risk, s.f2_cost = val[0], val[1]
                temp_sols.append(s)

            # 计算各项指标
            hv_list.append(calculator.calculate_hv(temp_sols))
            # 这里的 calculate_igd 应该在 metrics.py 中已包含归一化逻辑
            igd_list.append(calculator.calculate_igd(temp_sols, reference_front))
            sm_list.append(calculator.calculate_sm(temp_sols))

        hv_curves[algo] = hv_list
        igd_curves[algo] = igd_list
        sm_curves[algo] = sm_list

    # -------------------------------------------------------
    # F. Generate Final Report (Table 2)
    # -------------------------------------------------------
    logging.info(">>> Generating Table 2...")

    def calc_final_stats(algo, solutions, duration):
        if not solutions:
            hv, igd, sm = 0.0, float("inf"), 0.0
        else:
            hv = calculator.calculate_hv(solutions)
            igd = calculator.calculate_igd(solutions, reference_front)
            sm = calculator.calculate_sm(solutions)

        results_data["Algorithm"].append(algo)
        results_data["HV"].append(hv)
        results_data["IGD"].append(igd)
        results_data["SM"].append(sm)
        results_data["CPU Time (s)"].append(duration)

    # 计算并收集 Proposed 的最终指标
    calc_final_stats(algo_name, final_frontiers.get(algo_name, []), duration_proposed)

    # 计算并收集 Baselines 的最终指标
    for name in ["Standard NSGA2", "Standard SPEA2"]:
        calc_final_stats(name, final_frontiers.get(name, []), durations.get(name, 0))

    # 计算并收集 Gurobi 的最终指标
    if gurobi_sols:
        calc_final_stats(
            "Gurobi (Exact)", gurobi_sols, durations.get("Gurobi (Exact)", 0)
        )

    # 输出表格
    df = pd.DataFrame(results_data)

    logging.info("\n" + "=" * 80)
    logging.info("Table 2: Algorithmic Performance Comparison")
    logging.info("=" * 80)
    # 使用 to_string 让 logging 能够整齐输出表格
    logging.info(
        "\n" + df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x))
    )
    logging.info("=" * 80 + "\n")

    # 保存 CSV
    csv_path = os.path.join(benchmark_dir, "table_2_metrics.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Table 2 saved to: {csv_path}")

    # -------------------------------------------------------
    # G. Plotting (Fig 3, 4, 5, 6)
    # -------------------------------------------------------
    logging.info(">>> Plotting Figures...")

    plot_convergence(
        hv_curves, "Hypervolume (HV)", "Figure_3_HV_Convergence", benchmark_dir
    )
    plot_convergence(
        igd_curves, "IGD Metric", "Figure_4_IGD_Convergence", benchmark_dir
    )
    plot_convergence(
        sm_curves, "Spacing Metric (SM)", "Figure_5_SM_Convergence", benchmark_dir
    )

    # 绘制 Pareto 前沿对比图
    plot_pareto_comparison(final_frontiers, benchmark_dir)


def plot_convergence(
    data: Dict[str, List[float]], ylabel: str, filename: str, save_dir: str
):
    """通用绘图函数 (Convergence Curves)"""
    plt.figure(figsize=(10, 6))

    styles = {
        "Improved NSGA-II (Proposed)": {
            "color": "#d62728",
            "marker": "o",
            "markevery": 0.1,
            "linewidth": 2.5,
            "zorder": 10,
            "label": "Improved NSGA-II (Proposed)",
        },
        "Standard NSGA2": {
            "color": "#1f77b4",
            "marker": "s",
            "markevery": 0.1,
            "linewidth": 1.5,
            "linestyle": "--",
            "label": "Standard NSGA-II",
        },
        "Standard SPEA2": {
            "color": "#2ca02c",
            "marker": "^",
            "markevery": 0.1,
            "linewidth": 1.5,
            "linestyle": "-.",
            "label": "Standard SPEA2",
        },
        "Gurobi (Exact)": {
            "color": "black",
            "linewidth": 2.0,
            "linestyle": ":",
            "label": "Gurobi (Exact)",
        },
    }

    for name, history in data.items():
        if not history:
            continue
        gens = np.arange(1, len(history) + 1)

        # 获取样式，并避免修改原字典
        style = styles.get(name, {}).copy()

        # 智能兜底 label
        if "label" not in style:
            style["label"] = name

        plt.plot(gens, history, **style)

    plt.xlabel("Generation", fontsize=12, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12, fontweight="bold")
    plt.title(f"{ylabel} Convergence Analysis", fontsize=14, pad=15)
    plt.legend(fontsize=11, frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"{filename}.svg"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)
    logging.info(f"Saved {filename} to {save_dir}")


def plot_pareto_comparison(frontiers: Dict[str, List[Solution]], save_dir: str):
    """
    绘制多算法 Pareto 前沿对比图 (Figure 6)
    """
    plt.figure(figsize=(10, 6))

    styles = {
        "Improved NSGA-II (Proposed)": {
            "c": "#d62728",
            "marker": "o",
            "s": 30,
            "label": "Improved NSGA-II",
            "edgecolors": "white",
            "zorder": 10,
            "alpha": 0.7,
        },
        "Standard NSGA2": {
            "c": "#1f77b4",
            "marker": "s",
            "s": 40,
            "label": "Standard NSGA-II",
            "alpha": 0.6,
        },
        "Standard SPEA2": {
            "c": "#2ca02c",
            "marker": "^",
            "s": 40,
            "label": "Standard SPEA2",
            "alpha": 0.6,
        },
        "Gurobi (Exact)": {
            "c": "black",
            "marker": "*",
            "s": 40,
            "label": "Gurobi (Exact)",
            "zorder": 15,
            "edgecolors": "yellow",
        },
    }

    for name, sols in frontiers.items():
        if not sols:
            continue

        # 提取坐标
        risks = [s.f1_risk for s in sols]
        costs = [s.f2_cost for s in sols]

        # 获取样式
        style = styles.get(name, {"label": name}).copy()

        plt.scatter(risks, costs, **style)

    plt.xlabel("Transportation Risk (people·t)", fontsize=12, fontweight="bold")
    plt.ylabel("Transportation Cost (yuan)", fontsize=12, fontweight="bold")
    plt.title("Pareto Frontiers Comparison", fontsize=14, pad=15)
    plt.legend(fontsize=11, frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "Figure_6_Pareto_Comparison.svg"), dpi=300)
    plt.savefig(os.path.join(save_dir, "Figure_6_Pareto_Comparison.png"), dpi=300)
    logging.info(f"Saved Figure_6_Pareto_Comparison to {save_dir}")


if __name__ == "__main__":
    main()
