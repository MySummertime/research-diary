# --- coding: utf-8 ---
# --- sensitivity_analysis.py ---
"""
[Experiment Sensitivity Analysis]
分析模型对 Risk Aversion (Alpha) 的敏感性。
包含:
1. 极端厌恶分析 (Extreme Aversion): Pareto Shift & Cost Breakdown
2. 运到期限可靠性分析 (Time Reliability): 分析 alpha_t 对风险与成本的权衡
"""

import logging
import os
import random

import numpy as np
import pandas as pd

from app.core.evaluator import Evaluator
from app.core.nsga2 import NSGA2
from app.experiment_manager import Experiment
from app.utils.plotter import ParetoPlotter, SensitivityPlotter
from app.utils.result_keeper import setup_logging


def main():
    # 1. Setup
    exp = Experiment(config_path="config.json")
    sensitivity_dir = os.path.join(exp.save_dir, "sensitivity_analysis")
    os.makedirs(sensitivity_dir, exist_ok=True)
    setup_logging(log_dir=sensitivity_dir, log_name="sensitivity.log")

    # 2. Run Experiments

    # Risk Aversion
    perform_cvar_sensitivity(exp, sensitivity_dir)

    # Reliability
    # perform_time_sensitivity(exp, sensitivity_dir)

    # Gini Sensitivity (Equity Tolerance)
    # perform_gini_sensitivity(exp, sensitivity_dir)


def perform_cvar_sensitivity(exp: Experiment, save_dir: str):
    """
    极端风险厌恶分析 (alpha -> 1)
    """
    logging.info(">>> Starting Sensitivity Experiment: CVaR Confidence Level...")

    # 适宜精度的 Alpha 区间
    alphas = [0.9990, 0.9991, 0.9992, 0.9993, 0.9994, 0.9995, 0.9996]

    # 收集 Min Risk Solution 的所有任务的路径细节
    min_risk_routes = []

    # 收集 Min Cost Solution 的 Total Risk，用于 CSV
    min_cost_risks = []

    # 收集 Min Cost Solution 的 Total Cost，用于双轴折线图
    min_cost_total_cost = []

    # 收集 Min Cost Solution 的成本构成，用于 Stacked Bar
    min_cost_breakdown = {"transport": [], "transshipment": []}

    # 收集 Min Risk Solution 的 Total Risk，用于双轴折线图和 Stacked Bar 右轴
    min_risk_total_risk = []

    # 收集 Min Risk Solution 的 Total Cost，用于 CSV
    min_risk_costs = []

    # 收集 Min Risk Solution 的成本构成，用于 CSV
    min_risk_breakdown = {"transport": [], "transshipment": []}

    pareto_fronts = {}
    x_labels = []

    # 确保获取正确的配置 Key
    target_key = "risk_model_f1"
    original_alpha = exp.config.get(target_key, {}).get("cvar_alpha", 0.9996)

    for alpha in alphas:
        alpha_val = float(alpha)
        # 格式化 label，保留足够的小数位
        label_str = f"{alpha_val:.4f}"
        logging.info(f"   Running for alpha = {label_str}")

        # 在每个 Alpha 循环开始时强制重置随机种子
        # 这样确保无论实验顺序如何，同一个 alpha 下的 GA 运行轨迹完全一致
        random.seed(49)
        np.random.seed(49)

        # 1. 动态修改配置
        if target_key not in exp.config:
            exp.config[target_key] = {}
        exp.config[target_key]["cvar_alpha"] = alpha_val

        # 2. 热重载组件 (Evaluator 会读取新的 exp.config["risk_model_f1"])
        exp.evaluator = Evaluator(exp.network, exp.config)
        # NSGA2 需要新的 Evaluator
        exp.algorithm = NSGA2(
            exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
        )

        # 3. 运行算法
        # 不传 initial_population 以确保随机初始化，避免陷入局部最优
        final_pop = exp.algorithm.run(callbacks=[], initial_population=None)
        feasible = [s for s in final_pop if s.is_feasible and s.rank == 0]

        if feasible:
            # A. 保存 Pareto Front sorted by risk and then by cost
            feasible.sort(key=lambda s: (s.f1_risk, s.f2_cost))
            pareto_fronts[label_str] = feasible

            # B. 提取两个关键解：min cost solution 和 min risk solution

            # Min Cost Solution: 成本最小；成本相同时取风险最小的
            min_cost_sol = min(feasible, key=lambda s: (s.f2_cost, s.f1_risk))
            # Min Risk Solution: 风险最小；风险相同时取成本最低的 (方案A)
            min_risk_sol = min(feasible, key=lambda s: (s.f1_risk, s.f2_cost))

            # --- ✨ 记录所有任务的路由细节 ---
            route_details_list = []
            # 按照任务 ID 排序，确保输出字符串的顺序固定
            for tid in sorted(min_risk_sol.path_selections.keys()):
                path_obj = min_risk_sol.path_selections[tid]
                nodes_str = "->".join([n.node_id for n in path_obj.nodes])
                route_details_list.append(f"Task_{tid}: {nodes_str}")

            # 将多个任务的路径用分号或竖线隔开，存成一个单元格
            min_risk_routes.append(" | ".join(route_details_list))
            # --------------------------------

            # C1. 提取数据 for Stacked Bar/CSV (Min Cost Solution 的成本结构)
            bd_min_cost = exp.evaluator.calculate_cost_breakdown(min_cost_sol)
            min_cost_breakdown["transport"].append(bd_min_cost["transport"])
            min_cost_breakdown["transshipment"].append(bd_min_cost["transshipment"])

            # C2. 提取数据 for CSV (Min Risk Solution 的成本结构)
            bd_min_risk = exp.evaluator.calculate_cost_breakdown(min_risk_sol)
            min_risk_breakdown["transport"].append(bd_min_risk["transport"])
            min_risk_breakdown["transshipment"].append(bd_min_risk["transshipment"])

            # D. 提取数据 CSV
            min_cost_total_cost.append(min_cost_sol.f2_cost)
            min_risk_total_risk.append(min_risk_sol.f1_risk)

            # 提取交叉数据 for Dual Line Chart/CSV (左轴: min cost of min risk, 右轴: min risk total)
            min_cost_risks.append(min_cost_sol.f1_risk)
            min_risk_costs.append(min_risk_sol.f2_cost)

            x_labels.append(label_str)
        else:
            logging.warning(f"No feasible solution found for alpha={alpha_val}")

    # 恢复配置
    exp.config[target_key]["cvar_alpha"] = original_alpha

    try:
        # 1. 创建数据字典
        data_for_csv = {
            # 已经格式化为字符串的 alpha 值
            "cvar_alpha": x_labels,
            # Min Cost Solution
            "min_cost_sol_risk": min_cost_risks,
            "min_cost_sol_cost": min_cost_total_cost,
            "transport_cost_min_cost_sol": min_cost_breakdown["transport"],
            "transshipment_cost_min_cost_sol": min_cost_breakdown["transshipment"],
            # Min Risk Solution
            "min_risk_sol_risk": min_risk_total_risk,
            "min_risk_sol_cost": min_risk_costs,
            "transport_cost_min_risk_sol": min_risk_breakdown["transport"],
            "transshipment_cost_min_risk_sol": min_risk_breakdown["transshipment"],
            "route_details_all_tasks": min_risk_routes,
        }

        # 2. 检查数据长度是否一致
        if not all(len(v) == len(x_labels) for k, v in data_for_csv.items()):
            logging.error("Data arrays for CSV export have inconsistent lengths!")
            # 可以 raise 异常或返回

        # 3. 创建 DataFrame 并写入 CSV
        df = pd.DataFrame(data_for_csv)
        csv_path = os.path.join(save_dir, "Sensitivity_Extreme_Aversion_Data.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Sensitivity data saved to: {csv_path}")

    except Exception as e:
        logging.error(f"Error during CSV export: {e}")

    # --- Plotting ---
    plotter = SensitivityPlotter(save_dir)
    pareto_plotter = ParetoPlotter(save_dir=save_dir)

    # Chart 1: Cost Structure (Stacked Bar) + Risk (Line)
    if x_labels:
        # 传递 min_risk_total_risk 作为 Risk 数据
        plotter.plot_cost_structure_dual_axis(
            x_labels,
            min_risk_breakdown,
            min_risk_total_risk,
            r"CVaR Confidence Level $\alpha$",
            "Figure_Extreme_Cost_Structure",
        )

        # Chart 2: Cost-Risk Dual Y-axis Line Chart
        plotter.plot_dual_line_chart(
            x_labels,
            min_risk_costs,  # 左轴: Min Risk Solution 的 Total Cost
            min_risk_total_risk,  # 右轴: Min Risk Solution 的 Total Risk
            xlabel=r"CVaR Confidence Level $\alpha$",
            filename="Figure_Extreme_Cost_Risk_Trend",
            legend_loc="upper right",
        )

    # Chart 3: Pareto Frontier Comparison
    if pareto_fronts:
        pareto_plotter.plot_frontier_comparison(
            pareto_fronts,
            file_name="Figure_Extreme_Pareto_Shift",
            x_prefix=r"$\alpha$=",
        )


def perform_time_sensitivity(exp: Experiment, save_dir: str):
    """
    运到期限置信水平 alpha_t 敏感性分析：
    分析当决策者对“准时到达”的要求越来越严格时，对系统风险和成本的影响。
    """
    logging.info(">>> Starting Sensitivity Experiment: Time Confidence Level...")

    # 1. 实验参数设置
    time_alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    # 容器初始化
    pareto_fronts = {}
    # 收集 Min Risk Solution 的所有任务的路径细节
    min_risk_routes = []
    min_risk_costs = []
    min_risk_total_risk = []
    min_risk_breakdown = {"transport": [], "transshipment": []}

    # 根据实际成功运行的组动态添加
    actual_x_labels = []

    # 备份原始配置，防止实验状态污染
    conf_f2 = exp.config.get("cost_model_f2", {})
    original_alpha_t = conf_f2.get("time_confidence_level", 0.9)
    original_max_time = conf_f2.get("max_delivery_time", 96.0)

    # 保留用户要求的硬编码 25 (用于测试较为严格的时效环境)
    exp.config["cost_model_f2"]["max_delivery_time"] = 25

    try:
        for alpha_t in time_alphas:
            alpha_val = float(alpha_t)
            label_str = f"{alpha_val:.1f}"
            logging.info(f"--- Testing Time Confidence Level alpha_t = {label_str} ---")

            # 每个循环开始重置种子，保证同一 alpha_t 的确定性
            random.seed(49)
            np.random.seed(49)

            # 动态修改配置
            exp.config["cost_model_f2"]["time_confidence_level"] = alpha_val

            # 热重载，确保评估器感知到新的 alpha_t
            exp.evaluator = Evaluator(exp.network, exp.config)
            optimizer = NSGA2(
                exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
            )

            # 运行算法
            final_pop = optimizer.run()

            # 筛选出 Rank 0 且可行的解
            feasible_sols = [s for s in final_pop if s.is_feasible and s.rank == 0]

            if feasible_sols:
                # 执行二级排序 (风险升序，成本次之)
                # 保证即使有相同风险的解，也能选出最经济的一个
                feasible_sols.sort(key=lambda s: (s.f1_risk, s.f2_cost))

                pareto_fronts[label_str] = feasible_sols

                # 提取最优风险解
                min_risk_sol = feasible_sols[0]

                # --- ✨ 记录所有任务的路由细节 ---
                route_details_list = []
                for tid in sorted(min_risk_sol.path_selections.keys()):
                    path_obj = min_risk_sol.path_selections[tid]
                    nodes_str = "->".join([n.node_id for n in path_obj.nodes])
                    route_details_list.append(f"Task_{tid}: {nodes_str}")

                min_risk_routes.append(" | ".join(route_details_list))
                # --------------------------------

                min_risk_costs.append(min_risk_sol.f2_cost)
                min_risk_total_risk.append(min_risk_sol.f1_risk)

                # 记录成本构成
                bd = exp.evaluator.calculate_cost_breakdown(min_risk_sol)
                min_risk_breakdown["transport"].append(bd["transport"])
                min_risk_breakdown["transshipment"].append(bd["transshipment"])

                # 只有成功找到解，才记录这个 x 轴标签
                actual_x_labels.append(label_str)
            else:
                logging.warning(
                    f"No feasible solutions found for alpha_t = {label_str} (可能时限约束过紧)"
                )

        # 2. 导出 CSV (使用动态生成的 actual_x_labels 确保长度一致)
        if actual_x_labels:
            data_for_csv = {
                "alpha_t": actual_x_labels,
                "Min_Risk_Total_Cost": min_risk_costs,
                "Min_Risk_Total_Risk": min_risk_total_risk,
                "Transport_Cost": min_risk_breakdown["transport"],
                "Transshipment_Cost": min_risk_breakdown["transshipment"],
                "Route_Details_All_Tasks": min_risk_routes,
            }
            # DataFrame 构建时所有 array 长度现在必然等于 len(actual_x_labels)
            df = pd.DataFrame(data_for_csv)
            csv_path = os.path.join(save_dir, "Sensitivity_Time_Reliability_Data.csv")
            df.to_csv(csv_path, index=False)
            logging.info(f"Sensitivity data saved to: {csv_path}")
        else:
            logging.error(
                "❌ 所有 alpha_t 测试组均未找到可行解，请检查 max_delivery_time 设置！"
            )

    finally:
        # 恢复原始配置，确保其他实验模块的数据一致性
        exp.config["cost_model_f2"]["time_confidence_level"] = original_alpha_t
        exp.config["cost_model_f2"]["max_delivery_time"] = original_max_time

    # 3. Plotting (传入 actual_x_labels)
    if actual_x_labels:
        plotter = SensitivityPlotter(save_dir)
        pareto_plotter = ParetoPlotter(save_dir=save_dir)

        plotter.plot_cost_structure_dual_axis(
            actual_x_labels,
            min_risk_breakdown,
            min_risk_total_risk,
            r"Time Confidence Level $\alpha_t$",
            "Figure_Time_Reliability_Structure",
        )

        plotter.plot_dual_line_chart(
            actual_x_labels,
            min_risk_costs,
            min_risk_total_risk,
            r"Time Confidence Level $\alpha_t$",
            "Figure_Time_Risk_Trend",
        )

        if pareto_fronts:
            pareto_plotter.plot_frontier_comparison(
                pareto_fronts,
                file_name="Figure_Time_Pareto_Shift",
                x_prefix=r"$\alpha_t$=",
            )


def perform_gini_sensitivity(exp: Experiment, save_dir: str):
    """
    公平性容忍度 (Gini Epsilon) 敏感性分析
    分析当决策者对“区域不公平”的容忍上限 epsilon 变化时，对系统总风险和总成本的影响。
    """
    logging.info(
        ">>> Starting Sensitivity Experiment: Gini Epsilon (Equity Tolerance)..."
    )

    # 1. 实验参数设置:
    epsilons = [0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.6, 0.64]

    # 容器初始化
    min_risk_routes = []
    min_cost_risks = []
    min_cost_total_cost = []
    min_cost_breakdown = {"transport": [], "transshipment": []}

    min_risk_total_risk = []
    min_risk_costs = []
    min_risk_breakdown = {"transport": [], "transshipment": []}

    pareto_fronts = {}
    x_labels = []

    # 备份与配置准备
    target_key = "equity_config"
    if target_key not in exp.config:
        exp.config[target_key] = {"gini_epsilon": 0.2}
    original_epsilon = exp.config[target_key].get("gini_epsilon", 0.2)

    try:
        for epsilon in epsilons:
            epsilon_val = float(epsilon)
            label_str = f"{epsilon_val:.2f}"
            logging.info(f"--- Testing Gini Epsilon = {label_str} ---")

            # 固定随机种子保证实验可重复性
            random.seed(49)
            np.random.seed(49)

            # 动态修改评估器配置
            exp.config[target_key]["gini_epsilon"] = epsilon_val

            # 热重载组件 (Evaluator 内部会计算 CV_gini)
            exp.evaluator = Evaluator(exp.network, exp.config)
            exp.algorithm = NSGA2(
                exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
            )

            # 运行算法
            final_pop = exp.algorithm.run(callbacks=[], initial_population=None)

            # 筛选出 Pareto 前沿 (Rank 0) 的可行解
            feasible = [s for s in final_pop if s.is_feasible and s.rank == 0]

            if feasible:
                # A. 排序保存用于前沿移动分析
                feasible.sort(key=lambda s: (s.f1_risk, s.f2_cost))
                pareto_fronts[label_str] = feasible

                # B. 节点提取：Min Cost 和 Min Risk (通常关注极端点)
                min_cost_sol = min(feasible, key=lambda s: (s.f2_cost, s.f1_risk))
                min_risk_sol = min(feasible, key=lambda s: (s.f1_risk, s.f2_cost))

                # C. 记录 Min Risk 方案的路由路径细节
                route_details_list = []
                for tid in sorted(min_risk_sol.path_selections.keys()):
                    path_obj = min_risk_sol.path_selections[tid]
                    nodes_str = "->".join([n.node_id for n in path_obj.nodes])
                    route_details_list.append(f"Task_{tid}: {nodes_str}")
                min_risk_routes.append(" | ".join(route_details_list))

                # D. 提取成本细分 (Min Cost 解)
                bd_min_cost = exp.evaluator.calculate_cost_breakdown(min_cost_sol)
                min_cost_breakdown["transport"].append(bd_min_cost["transport"])
                min_cost_breakdown["transshipment"].append(bd_min_cost["transshipment"])

                # E. 提取成本细分 (Min Risk 解)
                bd_min_risk = exp.evaluator.calculate_cost_breakdown(min_risk_sol)
                min_risk_breakdown["transport"].append(bd_min_risk["transport"])
                min_risk_breakdown["transshipment"].append(bd_min_risk["transshipment"])

                # F. 汇总主要指标
                min_cost_total_cost.append(min_cost_sol.f2_cost)
                min_risk_total_risk.append(min_risk_sol.f1_risk)
                min_cost_risks.append(min_cost_sol.f1_risk)
                min_risk_costs.append(min_risk_sol.f2_cost)

                x_labels.append(label_str)
            else:
                logging.warning(
                    f"No feasible solutions found for gini_epsilon = {label_str}"
                )

    finally:
        # 恢复原始配置防止影响后续实验
        exp.config[target_key]["gini_epsilon"] = original_epsilon

    # 4. 数据保存与绘图
    if x_labels:
        # 导出 CSV 数据
        data_for_csv = {
            "gini_epsilon": x_labels,
            "min_cost_sol_risk": min_cost_risks,
            "min_cost_sol_cost": min_cost_total_cost,
            "transport_cost_min_cost_sol": min_cost_breakdown["transport"],
            "transshipment_cost_min_cost_sol": min_cost_breakdown["transshipment"],
            "min_risk_sol_risk": min_risk_total_risk,
            "min_risk_sol_cost": min_risk_costs,
            "transport_cost_min_risk_sol": min_risk_breakdown["transport"],
            "transshipment_cost_min_risk_sol": min_risk_breakdown["transshipment"],
            "route_details_all_tasks": min_risk_routes,
        }
        df = pd.DataFrame(data_for_csv)
        csv_path = os.path.join(save_dir, "Sensitivity_Gini_Epsilon_Data.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Gini sensitivity data saved to: {csv_path}")

        # 实例化绘图器
        plotter = SensitivityPlotter(save_dir)
        pareto_plotter = ParetoPlotter(save_dir=save_dir)

        # 图表 1: 成本结构 (堆叠柱状) + 风险 (折线)
        plotter.plot_cost_structure_dual_axis(
            x_labels,
            min_risk_breakdown,
            min_risk_total_risk,
            r"Gini Epsilon Tolerance $\epsilon$",
            "Figure_Gini_Cost_Structure",
        )

        # 图表 2: 风险与成本的双轴演变趋势
        plotter.plot_dual_line_chart(
            x_labels,
            min_risk_costs,
            min_risk_total_risk,
            r"Gini Epsilon Tolerance $\epsilon$",
            "Figure_Gini_Risk_Trend",
        )

        # 图表 3: Pareto 前沿随公平性容忍度增加的平移现象
        if pareto_fronts:
            pareto_plotter.plot_frontier_comparison(
                pareto_fronts,
                file_name="Figure_Gini_Pareto_Shift",
                x_prefix=r"$\epsilon$=",
                legend_loc="lower right",
            )
    else:
        logging.error("No data collected for Gini sensitivity analysis.")


if __name__ == "__main__":
    main()
