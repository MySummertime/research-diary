# --- coding: utf-8 ---
# --- sensitivity_analysis.py ---
"""
[Experiment Sensitivity Analysis]
分析模型对 Risk Aversion (Alpha) 的敏感性。
包含:
1. 极端厌恶分析 (Extreme Aversion): Pareto Shift & Cost Breakdown
2. 风险态度转变分析 (Transition): Expected Risk vs CVaR
"""

import os
import logging
import pandas as pd
from app.experiment_manager import Experiment
from app.core.evaluator import Evaluator
from app.core.nsga2 import NSGA2
from app.utils.result_keeper import setup_logging
from app.utils.plotter import ParetoPlotter, SensitivityPlotter


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
    perform_budget_sensitivity(exp, sensitivity_dir)


def perform_cvar_sensitivity(exp: Experiment, save_dir: str):
    """
    极端风险厌恶分析 (alpha -> 1)
    """
    logging.info(">>> Starting Sensitivity Experiment: CVaR Confidence Level...")

    # 适宜精度的 Alpha 区间
    alphas = [0.99920, 0.99930, 0.99940, 0.99950, 0.99960, 0.99970]

    # 收集 Min Cost Solution 的 Total Risk，用于 CSV
    min_cost_risks = []

    # 收集 Min Cost Solution 的 Total Cost，用于双轴折线图
    min_cost_total_cost = []

    # 收集 Min Cost Solution 的成本构成，用于 Stacked Bar
    min_cost_breakdown = {"transport": [], "transshipment": [], "carbon": []}

    # 收集 Min Risk Solution 的 Total Risk，用于双轴折线图和 Stacked Bar 右轴
    min_risk_total_risk = []

    # 收集 Min Risk Solution 的 Total Cost，用于 CSV
    min_risk_costs = []

    # 收集 Min Risk Solution 的成本构成，用于 CSV
    min_risk_breakdown = {"transport": [], "transshipment": [], "carbon": []}

    pareto_fronts = {}
    x_labels = []

    # 确保获取正确的配置 Key
    target_key = "risk_model_f1"
    original_alpha = exp.config.get(target_key, {}).get("cvar_alpha", 0.99960)

    for alpha in alphas:
        alpha_val = float(alpha)
        # 格式化 label，保留足够的小数位
        label_str = f"{alpha_val:.5f}"
        logging.info(f"   Running for alpha = {label_str}")

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
            # A. 保存 Pareto Front sorted by risk
            feasible.sort(key=lambda s: s.f1_risk)
            pareto_fronts[label_str] = feasible

            # B. 提取两个关键解：min cost solution 和 min risk solution
            min_cost_sol = min(feasible, key=lambda s: s.f2_cost)
            min_risk_sol = min(feasible, key=lambda s: s.f1_risk)

            # C1. 提取数据 for Stacked Bar/CSV (Min Cost Solution 的成本结构)
            bd_min_cost = exp.evaluator.calculate_cost_breakdown(min_cost_sol)
            min_cost_breakdown["transport"].append(bd_min_cost["transport"])
            min_cost_breakdown["transshipment"].append(bd_min_cost["transshipment"])
            min_cost_breakdown["carbon"].append(bd_min_cost["carbon"])

            # C2. 提取数据 for CSV (Min Risk Solution 的成本结构)
            bd_min_risk = exp.evaluator.calculate_cost_breakdown(min_risk_sol)
            min_risk_breakdown["transport"].append(bd_min_risk["transport"])
            min_risk_breakdown["transshipment"].append(bd_min_risk["transshipment"])
            min_risk_breakdown["carbon"].append(bd_min_risk["carbon"])

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
            "carbon_cost_min_cost_sol": min_cost_breakdown["carbon"],
            # Min Risk Solution
            "min_risk_sol_risk": min_risk_total_risk,
            "min_risk_sol_cost": min_risk_costs,
            "transport_cost_min_risk_sol": min_risk_breakdown["transport"],
            "transshipment_cost_min_risk_sol": min_risk_breakdown["transshipment"],
            "carbon_cost_min_risk_sol": min_risk_breakdown["carbon"],
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
            "Figure_Extreme_Cost_Structure.svg",
        )

        # Chart 2: Cost-Risk Dual Y-axis Line Chart
        plotter.plot_dual_line_chart(
            x_labels,
            min_risk_costs,  # 左轴: Min Risk Solution 的 Total Cost
            min_risk_total_risk,  # 右轴: Min Risk Solution 的 Total Risk
            xlabel=r"CVaR Confidence Level $\alpha$",
            filename="Figure_Extreme_Cost_Risk_Trend.svg",
        )

    # Chart 3: Pareto Frontier Comparison
    if pareto_fronts:
        pareto_plotter.plot_frontier_comparison(
            pareto_fronts, file_name="Figure_Extreme_Pareto_Shift.svg"
        )


def perform_budget_sensitivity(exp: Experiment, save_dir: str):
    logging.info(">>> Starting Sensitivity Experiment: Budget Confidence Level...")

    # 1. 备份原始参数
    orig_alpha_c = exp.config.get("cost_model_f2", {}).get("fuzzy_cost_alpha_c", 0.9)

    # --- 阶段 1: Warm-up (获取预算基准) ---
    exp.config["risk_model_f1"]["cvar_alpha"] = 0.99960
    exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = 0.9
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = 1e15  # 使用超大预算

    exp.evaluator = Evaluator(exp.network, exp.config)
    exp.algorithm = NSGA2(
        exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
    )

    pop = exp.algorithm.run(callbacks=[], initial_population=None)
    feasible_rank0 = [s for s in pop if s.is_feasible and s.rank == 0]

    if not feasible_rank0:
        logging.error("Warm-up failed: No Rank 0 solution found.")
        return

    # 根据决策成本 (scaled) 寻找最优解
    min_cost_sol_warm = min(feasible_rank0, key=lambda s: s.f2_cost_scaled)
    C_Pess_Star = min_cost_sol_warm.pessimistic_cost

    epsilon = 0.08
    TIGHT_BUDGET = C_Pess_Star * (1.0 + epsilon)
    logging.info(f"Setting TIGHT BUDGET = {TIGHT_BUDGET:.2f} (Epsilon={epsilon})")

    # --- 阶段 2: 迭代 alpha_c ---
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = TIGHT_BUDGET
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]

    # 收集 Min Cost Solution 的 Total Risk，用于 CSV
    min_cost_risks = []

    # 收集 Min Cost Solution 的 Total Cost，用于双轴折线图
    min_cost_total_cost = []
    min_cost_scaled_costs = []

    # 收集 Min Cost Solution 的成本构成，用于 Stacked Bar
    min_cost_breakdown = {"transport": [], "transshipment": [], "carbon": []}

    # 收集 Min Risk Solution 的 Total Risk，用于双轴折线图和 Stacked Bar 右轴
    min_risk_total_risk = []

    # 收集 Min Risk Solution 的 Total Cost，用于 CSV
    min_risk_costs = []
    min_risk_scaled_costs = []

    # 收集 Min Risk Solution 的成本构成，用于 CSV
    min_risk_breakdown = {"transport": [], "transshipment": [], "carbon": []}

    pareto_fronts = {}
    x_labels = []

    # 确保获取正确的配置 Key
    target_key = "cost_model_f2"

    for alpha in alphas:
        alpha_val = float(alpha)
        label_str = f"{alpha_val:.2f}"
        logging.info(f"   Running for alpha_c = {label_str}")

        # 1. 动态修改配置并热重载
        exp.config[target_key]["fuzzy_cost_alpha_c"] = alpha_val
        exp.evaluator = Evaluator(exp.network, exp.config)
        exp.algorithm = NSGA2(
            exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
        )

        # 2. 运行算法
        final_pop = exp.algorithm.run(callbacks=[], initial_population=None)
        feasible = [s for s in final_pop if s.is_feasible and s.rank == 0]

        if feasible:
            # A. 保存 Pareto 前沿用于绘图
            feasible.sort(key=lambda s: s.f1_risk)
            pareto_fronts[label_str] = feasible

            # B. 提取关键解 (使用 scaled 指标作为决策依据)
            min_cost_sol = min(feasible, key=lambda s: s.f2_cost_scaled)
            min_risk_sol = min(feasible, key=lambda s: s.f1_risk_scaled)

            # C1. 收集 Min Cost 解的数据
            min_cost_total_cost.append(min_cost_sol.f2_cost)  # 记录物理真实值
            min_cost_risks.append(min_cost_sol.f1_risk)  # 记录物理真实值
            min_cost_scaled_costs.append(min_cost_sol.f2_cost_scaled)  # 记录决策值

            bd_min_cost = exp.evaluator.calculate_cost_breakdown(min_cost_sol)
            min_cost_breakdown["transport"].append(bd_min_cost["transport"])
            min_cost_breakdown["transshipment"].append(bd_min_cost["transshipment"])
            min_cost_breakdown["carbon"].append(bd_min_cost["carbon"])

            # C2. 收集 Min Risk 解的数据
            min_risk_total_risk.append(min_risk_sol.f1_risk)
            min_risk_costs.append(min_risk_sol.f2_cost)
            min_risk_scaled_costs.append(min_risk_sol.f2_cost_scaled)

            bd_min_risk = exp.evaluator.calculate_cost_breakdown(min_risk_sol)
            min_risk_breakdown["transport"].append(bd_min_risk["transport"])
            min_risk_breakdown["transshipment"].append(bd_min_risk["transshipment"])
            min_risk_breakdown["carbon"].append(bd_min_risk["carbon"])

            # 最后同步记录标签
            x_labels.append(label_str)

    # 恢复配置
    exp.config[target_key]["fuzzy_cost_alpha_c"] = orig_alpha_c

    # --- 3. Output ---
    try:
        # 1. 创建数据字典
        data_for_csv = {
            "alpha_c": x_labels,
            # Min Cost Solution Data
            "min_cost_sol_risk": min_cost_risks,
            "min_cost_sol_cost": min_cost_total_cost,
            "min_cost_sol_scaled_cost": min_cost_scaled_costs,
            "transport_cost_min_cost_sol": min_cost_breakdown["transport"],
            "transshipment_cost_min_cost_sol": min_cost_breakdown["transshipment"],
            "carbon_cost_min_cost_sol": min_cost_breakdown["carbon"],
            # Min Risk Solution Data
            "min_risk_sol_risk": min_risk_total_risk,
            "min_risk_sol_cost": min_risk_costs,
            "min_risk_sol_scaled_cost": min_risk_scaled_costs,
            "transport_cost_min_risk_sol": min_risk_breakdown["transport"],
            "transshipment_cost_min_risk_sol": min_risk_breakdown["transshipment"],
            "carbon_cost_min_risk_sol": min_risk_breakdown["carbon"],
        }

        # 2. 检查数据长度是否一致
        if not all(len(v) == len(x_labels) for k, v in data_for_csv.items()):
            logging.error("Data arrays for CSV export have inconsistent lengths!")

        # 3. 创建 DataFrame 并写入 CSV
        df = pd.DataFrame(data_for_csv)
        csv_path = os.path.join(save_dir, "Sensitivity_Budget_Aversion_Data.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Sensitivity data saved to: {csv_path}")

    except Exception as e:
        logging.error(f"Error during CSV export: {e}")

    # --- 4. Plotting ---
    plotter = SensitivityPlotter(save_dir)
    pareto_plotter = ParetoPlotter(save_dir=save_dir)

    # Chart 1: Cost Structure (Stacked Bar) + Risk (Line)
    if x_labels:
        # Cost Stacked Bar: Min Cost Solution's cost breakdown
        # Risk Line: Min Cost Solution's total risk
        plotter.plot_cost_structure_dual_axis(
            x_labels,
            min_risk_breakdown,
            min_risk_total_risk,
            r"Budget Confidence Level $\alpha_c$",
            "Figure_Budget_Cost_Structure.svg",
        )

        # Chart 2: Cost-Risk Dual Y-axis Line Chart
        plotter.plot_dual_line_chart(
            x_labels,
            min_risk_costs,  # 左轴: Min Risk Solution 的 Total Cost
            min_risk_total_risk,  # 右轴: Min Risk Solution 的 Total Risk
            xlabel=r"Budget Confidence Level $\alpha_c$",
            filename="Figure_Cost_Risk_Trend.svg",
        )

    # Chart 3: Pareto Frontier Comparison
    if pareto_fronts:
        pareto_plotter.plot_frontier_comparison(
            pareto_fronts,
            file_name="Figure_Budget_Pareto_Shift.svg",
            x_prefix=r"$\alpha_c$=",
        )



if __name__ == "__main__":
    main()
