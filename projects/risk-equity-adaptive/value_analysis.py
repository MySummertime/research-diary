# --- coding: utf-8 ---
# --- value_analysis.py ---
"""
[Experiment: Value of Response-Dependent Risk Model]
对应论文 Section 5.4: 对比动态风险模型与静态风险模型的决策差异。
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
    [任务1] 将不同策略下的路径细节总结到 CSV
    """
    header = [
        "Opinion",
        "Task_ID",
        "Origin",
        "Destination",
        "Total_Risk",
        "Total_Cost",
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
        for task in evaluator.network.tasks:
            tid = task.task_id
            path = sol.path_selections[tid]

            # 计算各项成本与风险
            total_c, trans_c, ship_c, carb_c = _calculate_single_task_cost(
                path, evaluator
            )
            risk = _calculate_single_task_risk(path, evaluator)

            # 统计模式比例与路径
            road_len = sum(a.length for a in path.arcs if a.mode == "road")
            rail_len = sum(a.length for a in path.arcs if a.mode == "railway")
            total_len = road_len + rail_len
            ratio_str = f"{road_len / total_len:.0%} / {rail_len / total_len:.0%}"

            hubs_str = ", ".join([h.node_id for h in path.transfer_hubs])
            path_str = " -> ".join([n.node_id for n in path.nodes])

            rows.append(
                [
                    opinion_name,
                    tid,
                    path.task.origin.node_id,
                    path.task.destination.node_id,
                    f"{risk:.4f}",
                    f"{total_c:.2f}",
                    f"{trans_c:.2f}",
                    f"{ship_c:.2f}",
                    f"{carb_c:.2f}",
                    len(path.transfer_hubs),
                    ratio_str,
                    hubs_str,
                    path_str,
                ]
            )

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    logging.info(f"Detailed routing CSV saved: {filename} 📄")


def export_comparison_metrics_csv(
    tasks: List[str], risk_mat: np.ndarray, cost_mat: np.ndarray, filename: str
):
    """
    [任务2] 保存每个 Task 的 Risk 和 Cost 数据
    """
    header = ["Model", "Metric"] + tasks
    model_names = ["Proposed (Dynamic)", "Static (Traditional)"]

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # 写入 Risk 数据
        writer.writerow([model_names[0], "Risk"] + list(risk_mat[0]))
        writer.writerow([model_names[1], "Risk"] + list(risk_mat[1]))
        # 写入 Cost 数据
        writer.writerow([model_names[0], "Cost"] + list(cost_mat[0]))
        writer.writerow([model_names[1], "Cost"] + list(cost_mat[1]))
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

    # 1. 求解两个模型
    dyn_front = solve_proposed_dynamic_model(exp)
    sta_front = solve_traditional_static_model(exp)

    # 2. 回测评估
    perform_backtest_reevaluation(sta_front, exp.evaluator)

    # 3. 提取折衷解 (Knee Point) 用于深入对比
    dyn_kp = find_knee_point(dyn_front)
    sta_kp = find_knee_point(sta_front)

    if not (dyn_kp and sta_kp):
        logging.warning("未能找到足够的有效解进行对比。")
        return

    # 4. 准备数据
    tasks, risk_mat, cost_mat = prepare_value_comparison_matrices(
        dyn_kp, sta_kp, exp.evaluator
    )

    # 5. 调用 plotter 生成两张热图
    plotter = ModelComparisonPlotter(save_dir=val_dir)
    model_names = ["Proposed (Dynamic)", "Static (Traditional)"]

    # Risk 热图 (使用红橙暖色调)
    plotter.plot_comparison_heatmap(
        tasks,
        model_names,
        risk_mat,
        "Real Risk (people)",
        "Figure_5_4_Risk_Heatmap.svg",
        "RISK_CMAP",
    )

    # Cost 热图 (使用青绿冷色调)
    plotter.plot_comparison_heatmap(
        tasks,
        model_names,
        cost_mat,
        "Cost (yuan)",
        "Figure_5_4_Cost_Heatmap.svg",
        "COST_CMAP",
    )

    # 6. 路径可视化
    exp.visualizer.visualize_routes(
        {"Proposed_KneePoint": dyn_kp, "Static_KneePoint": sta_kp},
        save_dir=val_dir,
        prefix="Value_Map",
    )
    logging.info(f"价值分析实验圆满完成！✨ 结果已存档至: {val_dir}")

    # 准备特殊解字典用于导出
    special_sols = {
        "Opinion A (Proposed Knee)": dyn_kp,
        "Opinion B (Static Knee-Backtest)": sta_kp,
    }
    export_detailed_routing_csv(
        special_sols, exp.evaluator, os.path.join(val_dir, "detailed_routes.csv")
    )

    # 导出热图背后的原始数据
    export_comparison_metrics_csv(
        tasks, risk_mat, cost_mat, os.path.join(val_dir, "task_metrics_comparison.csv")
    )


if __name__ == "__main__":
    run_value_analysis()
