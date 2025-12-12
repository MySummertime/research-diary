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
import numpy as np
import logging
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

    # Reliability & Uncertainty
    # perform_reliability_sensitivity(exp, sensitivity_dir)
    # perform_uncertainty_sensitivity(exp, sensitivity_dir)


def perform_extreme_aversion_analysis(exp: Experiment, save_dir: str):
    """
    极端风险厌恶分析 (alpha -> 1)
    """
    logging.info(">>> Starting Experiment: Extreme Risk Aversion Analysis...")

    # 极高精度的 Alpha 区间
    alphas = np.linspace(0.9999774, 0.999986, 7)

    pareto_fronts = {}
    cost_breakdown = {"transport": [], "transshipment": [], "carbon": []}
    min_risks = []
    x_labels = []

    # 确保获取正确的配置 Key
    target_key = "risk_model_f1"
    original_alpha = exp.config.get(target_key, {}).get("cvar_alpha", 0.95)

    for alpha in alphas:
        alpha_val = float(alpha)
        # 格式化 label，保留足够的小数位
        label_str = f"{alpha_val:.6f}"
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
            best_risk_sol = min(feasible, key=lambda s: s.f1_risk)

            # C. 计算成本构成
            bd = exp.evaluator.calculate_cost_breakdown(best_risk_sol)
            cost_breakdown["transport"].append(bd["transport"])
            cost_breakdown["transshipment"].append(bd["transshipment"])
            cost_breakdown["carbon"].append(bd["carbon"])

            min_risks.append(best_risk_sol.f1_risk)
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

    # 1. Warm-up: Determine Optimal Cost (C_opt)
    # Using Loose Budget & Default Alpha
    logging.info("   Finding C_opt with loose budget...")

    # Backup
    orig_bgt = exp.config.get("cost_model_f2", {}).get("fuzzy_cost_budget", 1e9)
    orig_alpha_c = exp.config.get("cost_model_f2", {}).get("fuzzy_cost_alpha_c", 0.9)
    orig_cvar_alpha = exp.config.get("risk_model_f1", {}).get("cvar_alpha", 0.95)

    # Set CVaR alpha to 0.999980 (Fixed)
    if "risk_model_f1" not in exp.config:
        exp.config["risk_model_f1"] = {}
    exp.config["risk_model_f1"]["cvar_alpha"] = 0.999980

    # Run with loose budget
    if "cost_model_f2" not in exp.config:
        exp.config["cost_model_f2"] = {}
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = 1e9  # Huge budget

    exp.evaluator = Evaluator(exp.network, exp.config)
    exp.algorithm = NSGA2(
        exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
    )

    pop = exp.algorithm.run(callbacks=[], initial_population=None)
    feasible = [s for s in pop if s.is_feasible]

    if not feasible:
        logging.error(
            "Warm-up failed: No feasible solution found even with loose budget."
        )
        return

    c_opt = min(s.f2_cost for s in feasible)
    logging.info(f"   Found C_opt = {c_opt:.2f}")

    # 2. Set Tight Budget
    tight_budget = c_opt * 1.1
    logging.info(f"   Setting Tight Budget = {tight_budget:.2f}")
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = tight_budget

    # 3. Iterate alpha_c
    alpha_cs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    min_costs = []
    min_risks = []

    for ac in alpha_cs:
        logging.info(f"   Running for alpha_c = {ac}")
        exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = ac

        exp.evaluator = Evaluator(exp.network, exp.config)
        exp.algorithm = NSGA2(
            exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
        )

        pop = exp.algorithm.run(callbacks=[], initial_population=None)
        feasible = [s for s in pop if s.is_feasible and s.rank == 0]

        if feasible:
            # Focus on Min Expected Cost (since we are testing cost reliability)
            best_cost_sol = min(feasible, key=lambda s: s.f2_cost)
            min_costs.append(best_cost_sol.f2_cost)
            min_risks.append(best_cost_sol.f1_risk)
        else:
            logging.warning(f"   Infeasible for alpha_c={ac}")
            min_costs.append(None)
            min_risks.append(None)

    # Restore
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = orig_bgt
    exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = orig_alpha_c
    exp.config["risk_model_f1"]["cvar_alpha"] = orig_cvar_alpha

    # Plot
    plotter = SensitivityPlotter(save_dir)
    plotter.plot_dual_line_chart(
        alpha_cs,
        min_costs,
        min_risks,
        r"Budget Confidence Level $\alpha_c$",
        "Figure_Reliability_Sensitivity.svg",
        x_ticks=alpha_cs,
    )


# --- Uncertainty Sensitivity ---
def perform_uncertainty_sensitivity(exp: Experiment, save_dir: str):
    logging.info(">>> Starting Experiment: Uncertainty Level (Delta)...")

    # 1. Warm-up for Budget (using Medium Budget)
    # Re-using C_opt logic or just use config default if calibrated.
    # Let's recalibrate to be safe.
    logging.info("   Calibrating Budget for Uncertainty test...")

    # Backup
    orig_bgt = exp.config.get("cost_model_f2", {}).get("fuzzy_cost_budget", 1e9)
    orig_alpha_c = exp.config.get("cost_model_f2", {}).get("fuzzy_cost_alpha_c", 0.9)
    orig_cvar_alpha = exp.config.get("risk_model_f1", {}).get("cvar_alpha", 0.95)
    orig_delta = exp.config.get("uncertainty_multiplier", 1.0)

    # Fixed params
    if "risk_model_f1" not in exp.config:
        exp.config["risk_model_f1"] = {}
    exp.config["risk_model_f1"]["cvar_alpha"] = 0.999980
    if "cost_model_f2" not in exp.config:
        exp.config["cost_model_f2"] = {}
    exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = 0.9

    # Find C_opt with Delta=1.0 and Loose Budget
    exp.config["uncertainty_multiplier"] = 1.0
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = 1e9

    exp.evaluator = Evaluator(exp.network, exp.config)
    exp.algorithm = NSGA2(
        exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
    )

    pop = exp.algorithm.run(callbacks=[], initial_population=None)
    feasible = [s for s in pop if s.is_feasible]

    if not feasible:
        logging.error("Warm-up failed.")
        return

    c_opt = min(s.f2_cost for s in feasible)
    # Medium Budget = 1.2 * C_opt
    medium_budget = c_opt * 1.2
    logging.info(f"   C_opt={c_opt:.2f}, Setting Medium Budget={medium_budget:.2f}")
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = medium_budget

    # 2. Iterate Delta
    deltas = [0.0, 0.5, 1.0, 1.5, 2.0]
    min_costs = []
    min_risks = []

    for d in deltas:
        logging.info(f"   Running for delta = {d}")
        exp.config["uncertainty_multiplier"] = d

        exp.evaluator = Evaluator(exp.network, exp.config)
        exp.algorithm = NSGA2(
            exp.network, exp.evaluator, exp.candidate_paths_map, exp.config
        )

        pop = exp.algorithm.run(callbacks=[], initial_population=None)
        feasible = [s for s in pop if s.is_feasible and s.rank == 0]

        if feasible:
            # When uncertainty increases, costs usually go up.
            # We track the cheapest feasible option to see if we are forced to switch.
            best_cost_sol = min(feasible, key=lambda s: s.f2_cost)
            min_costs.append(best_cost_sol.f2_cost)
            min_risks.append(best_cost_sol.f1_risk)
        else:
            logging.warning(f"   Infeasible for delta={d}")
            min_costs.append(None)
            min_risks.append(None)

    # Restore
    exp.config["cost_model_f2"]["fuzzy_cost_budget"] = orig_bgt
    exp.config["cost_model_f2"]["fuzzy_cost_alpha_c"] = orig_alpha_c
    exp.config["risk_model_f1"]["cvar_alpha"] = orig_cvar_alpha
    exp.config["uncertainty_multiplier"] = orig_delta

    # Plot
    plotter = SensitivityPlotter(save_dir)
    plotter.plot_dual_line_chart(
        deltas,
        min_costs,
        min_risks,
        r"Uncertainty Level $\delta$",
        "Figure_Uncertainty_Sensitivity.svg",
        x_ticks=deltas,
    )


if __name__ == "__main__":
    main()
