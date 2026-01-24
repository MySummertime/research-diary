# --- coding: utf-8 ---
# --- value_analysis.py ---
"""
[Experiment: Value of Response-Dependent Risk Model]
对比动态风险模型与静态风险模型的决策差异。
逻辑：
1. 运行 Proposed Model (动态模型): 考虑应急响应时间对事故后果的扩散影响。
2. 运行 Static Model (静态模型): 后果仅取决于 GIS 缓冲区人口统计，不随响应时间变化。
3. 交叉评估：将静态解放入动态环境中，揭示传统模型低估的“隐形风险”。
"""

import csv
import logging
import os
from typing import Dict, List, Tuple

import numpy as np

from app.core.evaluator import Evaluator
from app.core.nsga2 import NSGA2
from app.core.solution import Solution
from app.experiment_manager import Experiment
from app.utils.analyzer import (
    _calculate_single_task_cost,
    _calculate_single_task_risk,
    calculate_solution_gini,
    find_knee_point,
)
from app.utils.plotter import ModelComparisonPlotter

# ==========================================
# 1. 核心求解逻辑模块 (Logic Layer)
# ==========================================


def solve_proposed_dynamic_model(exp: Experiment) -> List[Solution]:
    """[Proposed Model] 求解考虑动态响应时间的模型"""
    logging.info("Step 1: 正在求解 Proposed Model (动态响应感知)...")
    pop = exp.algorithm.run()
    return [s for s in pop if s.is_feasible and s.rank == 0]


def solve_traditional_static_model(exp: Experiment) -> List[Solution]:
    """[Static Model] 求解基于 GIS 人口常数的传统模型"""
    logging.info("Step 2: 正在求解 Traditional Static Model (基于GIS人口常数)...")

    # 局部创建静态评估器，注入静态逻辑
    static_eval = Evaluator(exp.network, exp.config)
    orig_func = static_eval.risk_model.get_consequence
    static_eval.risk_model.get_consequence = lambda entity: orig_func(
        entity, is_static=True
    )

    static_algo = NSGA2(exp.network, static_eval, exp.candidate_paths_map, exp.config)
    pop = static_algo.run()
    return [s for s in pop if s.is_feasible and s.rank == 0]


def perform_backtest_reevaluation(
    static_front: List[Solution], real_evaluator: Evaluator
):
    """[Backtesting] 将静态解代入动态响应系统评估真实风险"""
    logging.info("Step 3: 正在执行交叉回测：重新评估静态解在动态环境下的真实表现...")
    for s in static_front:
        real_evaluator.evaluate(s)


# ==========================================
# 2. 数据准备与可视化模块 (View Layer)
# ==========================================


def export_detailed_routing_csv(
    solutions_dict: Dict[str, Solution], evaluator: Evaluator, filename: str
):
    """
    [任务1] 将 2 种模型在 3 种策略下的路径细节总结到 CSV，并包含 Gini Coefficient 数据列
    """
    header = [
        "Opinion",
        "Task_ID",
        "Origin",
        "Destination",
        "Total_Risk",
        "Total_Cost",
        "Gini_Coefficient",
        "Transport_Cost",
        "Transshipment_Cost",
        "Carbon_Cost",
        "Transfers",
        "RoadOverRail_Ratio",
        "Transfer_Hubs",
        "Route_Path",
    ]

    rows = []
    for opinion_name, sol in solutions_dict.items():
        # 为该 Opinion 计算 Gini (使用 evaluator 确保动态回测准确)
        gini_val = calculate_solution_gini(sol, evaluator)

        for task in evaluator.network.tasks:
            tid = task.task_id
            path = sol.path_selections[tid]
            total_c, trans_c, ship_c, carb_c = _calculate_single_task_cost(
                path, evaluator
            )
            risk = _calculate_single_task_risk(path, evaluator)

            # 模式比例
            road_len = sum(a.length for a in path.arcs if a.mode == "road")
            rail_len = sum(a.length for a in path.arcs if a.mode == "railway")
            ratio_str = f"{road_len / (road_len + rail_len):.0%} / {rail_len / (road_len + rail_len):.0%}"

            rows.append(
                [
                    opinion_name,
                    tid,
                    path.task.origin.node_id,
                    path.task.destination.node_id,
                    f"{risk:.4f}",
                    f"{total_c:.2f}",
                    f"{gini_val:.6f}",
                    f"{trans_c:.2f}",
                    f"{ship_c:.2f}",
                    f"{carb_c:.2f}",
                    len(path.transfer_hubs),
                    ratio_str,
                    ", ".join([h.node_id for h in path.transfer_hubs]),
                    " -> ".join([n.node_id for n in path.nodes]),
                ]
            )

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    logging.info(f"Detailed routing CSV saved: {filename} 📄")


def export_comparison_metrics_csv(
    task_ids: List[str], metrics_data: List[Dict], filename: str
):
    """
    [任务2] 保存每个 Task 的 Risk, Cost, Decision Type, 和 Gini Coefficient 数据
    """
    header = ["Model", "Decision_Type", "Metric", "Gini"] + task_ids

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in metrics_data:
            # item 包含: model, dec_type, metric, gini, values(list)
            writer.writerow(
                [item["model"], item["dec_type"], item["metric"], f"{item['gini']:.6f}"]
                + item["values"]
            )
    logging.info(f"Task metrics CSV saved: {filename} 📊")


def prepare_value_comparison_matrices(
    dyn_sol: Solution, sta_sol: Solution, evaluator: Evaluator
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    准备任务维度的风险和成本对比矩阵。
    返回: (任务列表, 风险矩阵, 成本矩阵)
    """
    tasks = [t.task_id for t in evaluator.network.tasks]
    # 2行: Proposed, Static
    risk_matrix = np.zeros((2, len(tasks)))
    cost_matrix = np.zeros((2, len(tasks)))

    for j, tid in enumerate(tasks):
        path_dyn = dyn_sol.path_selections[tid]
        path_sta = sta_sol.path_selections[tid]

        # 风险行 (使用动态评估器计算真实暴露)
        risk_matrix[0, j] = _calculate_single_task_risk(path_dyn, evaluator)
        risk_matrix[1, j] = _calculate_single_task_risk(path_sta, evaluator)

        # 成本行 (提取期望总成本)
        cost_matrix[0, j] = _calculate_single_task_cost(path_dyn, evaluator)[0]
        cost_matrix[1, j] = _calculate_single_task_cost(path_sta, evaluator)[0]

    return tasks, risk_matrix, cost_matrix


def run_value_analysis():
    """[Controller] 价值分析实验主控流程"""
    exp = Experiment(config_path="config.json")
    val_dir = os.path.join(exp.save_dir, "value_analysis")
    os.makedirs(val_dir, exist_ok=True)

    # 1. 求解模型
    dyn_front = solve_proposed_dynamic_model(exp)
    sta_front = solve_traditional_static_model(exp)

    # 2. 多决策点提取逻辑
    # 从动态模型提取
    dyn_min_risk = min(dyn_front, key=lambda s: s.f1_risk)
    dyn_min_cost = min(dyn_front, key=lambda s: s.f2_cost)
    dyn_kp = find_knee_point(dyn_front)

    # 从静态模型提取 (静态环境下评估值)
    sta_min_risk = min(sta_front, key=lambda s: s.f1_risk)
    sta_min_cost = min(sta_front, key=lambda s: s.f2_cost)
    sta_kp = find_knee_point(sta_front)

    # 3. 执行回测 (将静态解放入动态环境中重新评估真实风险)
    perform_backtest_reevaluation([sta_min_risk, sta_min_cost, sta_kp], exp.evaluator)

    # 4. 准备数据矩阵用于热图和 CSV
    tasks = [t.task_id for t in exp.network.tasks]
    labels = ["Min Risk", "Knee", "Min Cost"]
    model_names = ["Proposed (Dynamic)", "Static (Traditional)"]

    # 构造 Opinion 字典用于导出
    opinion_sols = {
        "Proposed (Min Risk)": dyn_min_risk,
        "Proposed (Knee)": dyn_kp,
        "Proposed (Min Cost)": dyn_min_cost,
        "Static (Min Risk)": sta_min_risk,
        "Static (Knee)": sta_kp,
        "Static (Min Cost)": sta_min_cost,
    }

    # 准备热图矩阵 (6行)
    risk_matrix = np.zeros((6, len(tasks)))
    cost_matrix = np.zeros((6, len(tasks)))
    metrics_csv_data = []

    for i, (name, sol) in enumerate(opinion_sols.items()):
        m_name = model_names[0] if "Proposed" in name else model_names[1]
        d_type = (
            "Min Risk"
            if "Min Risk" in name
            else ("Min Cost" if "Min Cost" in name else "Knee")
        )

        # 计算 Gini 系数
        g_val = calculate_solution_gini(sol, exp.evaluator)

        task_risks, task_costs = [], []
        for j, tid in enumerate(tasks):
            path = sol.path_selections[tid]
            r = _calculate_single_task_risk(path, exp.evaluator)
            c = _calculate_single_task_cost(path, exp.evaluator)[0]
            risk_matrix[i, j], cost_matrix[i, j] = r, c
            task_risks.append(r)
            task_costs.append(c)

        metrics_csv_data.append(
            {
                "model": m_name,
                "dec_type": d_type,
                "metric": "Risk",
                "gini": g_val,
                "values": task_risks,
            }
        )
        metrics_csv_data.append(
            {
                "model": m_name,
                "dec_type": d_type,
                "metric": "Cost",
                "gini": g_val,
                "values": task_costs,
            }
        )

    # 5. 可视化与导出
    plotter = ModelComparisonPlotter(save_dir=val_dir)
    heatmap_labels = list(opinion_sols.keys())

    plotter.plot_comparison_heatmap(
        tasks,
        heatmap_labels,
        risk_matrix,
        "Real Risk (people)",
        "Figure_Risk_Heatmap.svg",
        "RISK_CMAP",
    )
    plotter.plot_comparison_heatmap(
        tasks,
        heatmap_labels,
        cost_matrix,
        "Cost (yuan)",
        "Figure_Cost_Heatmap.svg",
        "COST_CMAP",
    )

    export_detailed_routing_csv(
        opinion_sols,
        exp.evaluator,
        os.path.join(val_dir, "detailed_routes.csv"),
    )
    export_comparison_metrics_csv(
        tasks,
        metrics_csv_data,
        os.path.join(val_dir, "task_metrics_comparison.csv"),
    )

    # 路径图
    exp.visualizer.visualize_routes(
        opinion_sols, save_dir=val_dir, prefix="Figure_Routes_"
    )
    logging.info(f"价值分析全量实验完成！结果已存至: {val_dir} 🚀🌟")


if __name__ == "__main__":
    run_value_analysis()
