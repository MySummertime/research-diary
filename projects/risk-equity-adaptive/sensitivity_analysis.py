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
from typing import List, Dict
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
    perform_extreme_aversion_analysis(exp, sensitivity_dir)

    # Reliability
    # perform_reliability_sensitivity(exp, sensitivity_dir)

    # Uncertainty
    # perform_uncertain_response_time_sensitivity(exp, sensitivity_dir)


def perform_extreme_aversion_analysis(exp: Experiment, save_dir: str):
    """
    极端风险厌恶分析 (alpha -> 1)
    """
    logging.info(">>> Starting Experiment: Extreme Risk Aversion Analysis...")

    # 适宜精度的 Alpha 区间
    alphas = [0.99945, 0.99948, 0.99951, 0.99954, 0.99957]

    pareto_fronts = {}
    cost_breakdown = {"transport": [], "transshipment": [], "carbon": []}
    min_risks = []
    x_labels = []

    # 确保获取正确的配置 Key
    target_key = "risk_model_f1"
    original_alpha = exp.config.get(target_key, {}).get("cvar_alpha", 0.99945)

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
            # A. 保存 Sorted Pareto Front
            feasible.sort(key=lambda s: s.f1_risk)
            pareto_fronts[label_str] = feasible

            # B. 提取最安全的解
            max_cost_sol = min(feasible, key=lambda s: s.f1_risk)

            # C. 计算成本构成
            bd = exp.evaluator.calculate_cost_breakdown(max_cost_sol)
            cost_breakdown["transport"].append(bd["transport"])
            cost_breakdown["transshipment"].append(bd["transshipment"])
            cost_breakdown["carbon"].append(bd["carbon"])

            min_risks.append(max_cost_sol.f1_risk)
            x_labels.append(label_str)
        else:
            logging.warning(f"No feasible solution found for alpha={alpha_val}")

    # 恢复配置
    exp.config[target_key]["cvar_alpha"] = original_alpha

    # --- Plotting ---
    plotter = SensitivityPlotter(save_dir)
    pareto_plotter = ParetoPlotter(save_dir=save_dir)

    # Chart 1: Cost Structure (Stacked Bar) + Risk (Line)
    if x_labels:
        plotter.plot_cost_structure_dual_axis(
            x_labels, cost_breakdown, min_risks, "Figure_Extreme_Cost_Structure.svg"
        )

    # Chart 2: Pareto Frontier Comparison
    if pareto_fronts:
        pareto_plotter.plot_frontier_comparison_by_cvar_alpha(
            pareto_fronts, file_name="Figure_Extreme_Pareto_Shift.svg"
        )


def perform_reliability_sensitivity(exp: Experiment, save_dir: str):
    logging.info(">>> Starting Experiment: Reliability (Budget Confidence)...")

    # Backup and set FIXED params
    orig_bgt = exp.config.get("cost_model_f2", {}).get("fuzzy_cost_budget", 1e12)
    orig_alpha_c = exp.config.get("cost_model_f2", {}).get("fuzzy_cost_alpha_c", 0.9)
    orig_cvar_alpha = exp.config.get("risk_model_f1", {}).get("cvar_alpha", 0.99945)

    # --- 1. Warm-up: 寻找 Min Expected Cost 解的悲观成本 ---

    # 锁定 CVaR alpha (固定高风险厌恶度)
    if "risk_model_f1" not in exp.config:
        exp.config["risk_model_f1"] = {}
    exp.config["risk_model_f1"]["cvar_alpha"] = 0.99945

    # Warm-up 条件: 假设 alpha_c = 0.9, Delta = 1.0 (标准), 预算巨大
    exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = 0.9  # 固定 Warm-up 时的 alpha_c
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = 1e12  # Huge budget

    exp.evaluator = Evaluator(exp.network, exp.config)
    exp.algorithm = NSGA2(
        exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
    )

    pop = exp.algorithm.run(callbacks=[], initial_population=None)
    # 聚焦 Rank 0 上的可行解
    feasible_rank0 = [s for s in pop if s.is_feasible and s.rank == 0]

    if not feasible_rank0:
        logging.error("Warm-up failed: No Rank 0 solution found.")
        # 恢复配置
        exp.config["cost_model_f2"]["fuzzy_cost_budget"] = orig_bgt
        exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = orig_alpha_c
        exp.config["risk_model_f1"]["cvar_alpha"] = orig_cvar_alpha
        return

    # 1. 找到 Min Expected Cost 的解 S_opt (Rank 0 上最便宜的)
    min_cost_sol = min(feasible_rank0, key=lambda s: s.f2_cost)

    # 2. 获取该解在 alpha_c=0.9 时的悲观成本 C_Pess_Star
    C_Pess_Star = min_cost_sol.pessimistic_cost  # 直接读取 Evaluator 填充的值

    # 3. 设置 Tight Budget (冗余系数 epsilon = 1.05)
    epsilon = 0.05
    TIGHT_BUDGET = C_Pess_Star * (1.0 + epsilon)

    logging.info(f"   Min Expected Cost (C_opt) = {min_cost_sol.f2_cost:.2f}")
    logging.info(f"   Calculated Pessimistic Cost (alpha_c=0.9) = {C_Pess_Star:.2f}")
    logging.info(f"   Setting TIGHT BUDGET = {TIGHT_BUDGET:.2f}")

    # --- 2. Iterate alpha_c ---
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = TIGHT_BUDGET  # 启用紧预算

    # 迭代 alpha_c
    alpha_cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    # alpha_cs = [0.1, 0.2]
    costs = []
    risks = []

    # 存储所有边界解的详细数据 (用于 CSV)
    full_boundary_data = []

    for ac in alpha_cs:
        logging.info(f"   Running for alpha_c = {ac}")

        exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = ac  # 动态修改 alpha_c
        exp.evaluator = Evaluator(exp.network, exp.config)
        exp.algorithm = NSGA2(
            exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
        )

        pop = exp.algorithm.run(callbacks=[], initial_population=None)
        feasible = [s for s in pop if s.is_feasible and s.rank == 0]

        if feasible:
            # 聚焦 Min Expected Cost (f2)
            best_cost_sol = min(feasible, key=lambda s: s.f2_cost)
            # 聚焦 Min Risk (f1)
            best_risk_sol = min(feasible, key=lambda s: s.f1_risk)

            # 追踪 Cost 边界
            costs.append(best_cost_sol.f2_cost)
            # 追踪 Risk 边界
            risks.append(best_risk_sol.f1_risk)

            # 记录 CSV 数据
            full_boundary_data.append(
                {
                    "alpha_c": ac,
                    "min_cost_sol_cost": best_cost_sol.f2_cost,
                    "min_cost_sol_risk": best_cost_sol.f1_risk,
                    "min_risk_sol_risk": best_risk_sol.f1_risk,
                    "min_risk_sol_cost": best_risk_sol.f2_cost,
                }
            )
        else:
            # 预测：在 alpha_c > 某个数 时，可能找不到可行解
            logging.warning(f"   Infeasible for alpha_c={ac}")
            costs.append(None)
            risks.append(None)
            full_boundary_data.append(
                {
                    "alpha_c": ac,
                    "min_cost_sol_cost": None,
                    "min_cost_sol_risk": None,
                    "min_risk_sol_risk": None,
                    "min_risk_sol_cost": None,
                }
            )

    # Restore
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = orig_bgt
    exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = orig_alpha_c
    exp.config["risk_model_f1"]["cvar_alpha"] = orig_cvar_alpha

    # Plot
    plotter = SensitivityPlotter(save_dir)
    plotter.plot_dual_line_chart(
        alpha_cs,
        costs,
        risks,
        r"Budget Confidence Level $\alpha_c$",
        "Figure_Reliability_Sensitivity.svg",
        x_ticks=alpha_cs,
    )

    # Output
    save_csv_report(
        save_dir,
        full_boundary_data,
        [
            "alpha_c",
            "min_cost_sol_cost",
            "min_cost_sol_risk",
            "min_risk_sol_risk",
            "min_risk_sol_cost",
        ],
        "Table_Reliability_Sensitivity_Boundary_Data.csv",
    )


def perform_uncertain_response_time_sensitivity(
    exp: Experiment, save_dir: str, mode: str = "2D_grid"
):
    logging.info(
        ">>> Starting Experiment: Emergency Response Uncertainty Sensitivity Analysis... 🎯"
    )

    # Backup original params
    orig_bgt = exp.config.get("cost_model_f2", {}).get("fuzzy_cost_budget", 1e12)
    orig_alpha_c = exp.config.get("cost_model_f2", {}).get("fuzzy_cost_alpha_c", 0.9)
    orig_cvar_alpha = exp.config.get("risk_model_f1", {}).get("cvar_alpha", 0.99945)
    orig_a_multi = exp.config.get("risk_model_f1", {}).get(
        "emergency_a_multiplier", 0.0
    )
    orig_c_multi = exp.config.get("risk_model_f1", {}).get(
        "emergency_c_multiplier", 0.0
    )

    def restore_config(
        exp: Experiment,
        orig_bgt,
        orig_alpha_c,
        orig_cvar_alpha,
        orig_opt,
        orig_pes,
    ):
        """恢复实验前配置"""
        exp.config["cost_model_f2"]["fuzzy_cost_budget"] = orig_bgt
        exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = orig_alpha_c
        exp.config["risk_model_f1"]["cvar_alpha"] = orig_cvar_alpha
        exp.config["risk_model_f1"]["emergency_a_multiplier"] = orig_opt
        exp.config["risk_model_f1"]["emergency_c_multiplier"] = orig_pes

    # 固定参数设置，确保实验可控
    if "risk_model_f1" not in exp.config:
        exp.config["risk_model_f1"] = {}
    exp.config["risk_model_f1"]["cvar_alpha"] = 0.999945  # 高风险厌恶

    if "cost_model_f2" not in exp.config:
        exp.config["cost_model_f2"] = {}
    exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = 0.9

    # --- 1. Warm-up: 在基准不确定性下找 Tight Budget ---
    logging.info(
        " Calibrating Tight Budget under baseline emergency uncertainty (δ_a=0.0, δ_c=0.0)..."
    )
    exp.config["risk_model_f1"]["emergency_a_multiplier"] = 0.0
    exp.config["risk_model_f1"]["emergency_c_multiplier"] = 0.0
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = 1e12  # 松预算

    exp.evaluator = Evaluator(exp.network, exp.config)
    exp.algorithm = NSGA2(
        exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
    )
    pop = exp.algorithm.run(callbacks=[], initial_population=None)

    feasible_rank0 = [s for s in pop if s.is_feasible and s.rank == 0]
    if not feasible_rank0:
        logging.error("Warm-up failed: no feasible solutions.")
        restore_config(
            exp,
            orig_bgt,
            orig_alpha_c,
            orig_cvar_alpha,
            orig_a_multi,
            orig_c_multi,
        )
        return

    min_cost_sol = min(feasible_rank0, key=lambda s: s.f2_cost)
    C_Pess_Star = min_cost_sol.pessimistic_cost

    # 设置 Tight Budget (冗余系数 epsilon = 1.05)
    epsilon = 0.05
    TIGHT_BUDGET = C_Pess_Star * (1.0 + epsilon)
    logging.info(
        f" Baseline Pessimistic Cost = {C_Pess_Star:.2f} → Tight Budget = {TIGHT_BUDGET:.2f} 💰"
    )

    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = TIGHT_BUDGET

    # --- 2. 选定非对称场景（用于双线图）---
    selected_scenarios = [
        (0.0, 0.0),  # 确定性响应
        (0.5, 0.5),  # 乐观小，悲观大（现实拥堵场景）
        (1.0, 1.0),  # 对称标准
        (1.5, 1.5),  # 悲观侧更不确定
        (2.0, 2.0),  # 极端非对称
        (2.5, 2.5),  # 反过来：乐观侧更不确定（少见但可讨论）
    ]

    scenario_labels = [f"a={a:.1f},c={c:.1f}" for a, c in selected_scenarios]
    scenario_costs = []
    scenario_risks = []

    # --- 3. 完整二维网格（用于热力图）---
    delta_a_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    delta_c_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    cost_grid = [[None] * len(delta_c_values) for _ in delta_a_values]
    risk_grid = [[None] * len(delta_c_values) for _ in delta_a_values]

    total = len(delta_a_values) * len(delta_c_values) + len(selected_scenarios)
    current = 0

    # 先跑选定的非对称场景（同时填网格）
    for da, dc in selected_scenarios:
        current += 1
        logging.info(f" Scenario {current}/{total} | δ_a={da:.1f}, δ_c={dc:.1f}")

        exp.config["risk_model_f1"]["emergency_a_multiplier"] = da
        exp.config["risk_model_f1"]["emergency_c_multiplier"] = dc

        exp.evaluator = Evaluator(exp.network, exp.config)
        exp.algorithm = NSGA2(
            exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
        )

        pop = exp.algorithm.run(callbacks=[], initial_population=None)
        feasible = [s for s in pop if s.is_feasible and s.rank == 0]

        if feasible:
            best = min(feasible, key=lambda s: s.f2_cost)
            cost = best.f2_cost
            risk = best.f1_risk
            scenario_costs.append(cost)
            scenario_risks.append(risk)

            # 如果这个点在网格里，也填上
            if da in delta_a_values and dc in delta_c_values:
                i = delta_a_values.index(da)
                j = delta_c_values.index(dc)
                cost_grid[i][j] = cost
                risk_grid[i][j] = risk
        else:
            scenario_costs.append(None)
            scenario_risks.append(None)

    # 再补全整个网格
    for i, da in enumerate(delta_a_values):
        for j, dc in enumerate(delta_c_values):
            if cost_grid[i][j] is not None:  # 已算过，跳过
                continue
            current += 1
            logging.info(f" Grid {current}/{total} | δ_a={da:.1f}, δ_c={dc:.1f}")

            exp.config["risk_model_f1"]["emergency_a_multiplier"] = da
            exp.config["risk_model_f1"]["emergency_c_multiplier"] = dc
            exp.evaluator = Evaluator(exp.network, exp.config)
            exp.algorithm = NSGA2(
                exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
            )
            pop = exp.algorithm.run(callbacks=[], initial_population=None)
            feasible = [s for s in pop if s.is_feasible and s.rank == 0]

            if feasible:
                best = min(feasible, key=lambda s: s.f2_cost)
                cost_grid[i][j] = best.f2_cost
                risk_val = best.f1_risk
                risk_grid[i][j] = risk_val
            # else: remain None

    # --- 4. 可视化 ---
    plotter = SensitivityPlotter(save_dir)

    # 自定义标签双线图
    x_indices = list(range(len(selected_scenarios)))
    plotter.plot_dual_line_chart_with_custom_labels(
        x_indices=x_indices,
        cost_data=scenario_costs,
        risk_data=scenario_risks,
        custom_x_labels=scenario_labels,
        xlabel="Emergency Response Time Uncertainty Scenarios",
        filename="Figure_Uncertain_Response_Time_Asymmetric_Scenarios.svg",
    )

    # 4.2 热力图（完整网格）
    plotter.plot_emergency_uncertainty_heatmap(
        delta_a_values=delta_a_values,
        delta_c_values=delta_c_values,
        cost_grid=cost_grid,
        risk_grid=risk_grid,
        prefix="Figure_Emergency_Asymmetric",
    )

    # --- 5. Restore ---
    restore_config(
        exp,
        orig_bgt,
        orig_alpha_c,
        orig_cvar_alpha,
        orig_a_multi,
        orig_c_multi,
    )

    logging.info("Ultimate Asymmetric Sensitivity Analysis Completed 🌟🔥🚀")


def save_csv_report(
    save_dir: str, data: List[Dict], fieldnames: List[str], filename: str
):
    """[Helper] Saves a list of dictionaries to a CSV file."""
    import csv

    full_path = os.path.join(save_dir, filename)

    try:
        with open(full_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"CSV report saved successfully: {full_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV report {filename}: {e}")


if __name__ == "__main__":
    main()
