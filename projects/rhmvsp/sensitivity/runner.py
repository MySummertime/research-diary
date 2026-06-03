"""
Shared sensitivity analysis runner.
Creates per-value subfolders under results/sensitivity_<timestamp>/ with
log, solution.json, network.png/tiff, solution_plot.png/tiff — same as main.py.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from config import RHMVSPConfig
from src.core.network import MultimodalNetwork
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.utils.plotting import Visualizer
from src.utils.reporting import format_solution_report
from sensitivity.plot_helpers import dual_y_axis, three_subplots, COLOURS

# Dual Y-axis chart colours from colour set2.txt
COLOR_RISK = COLOURS["C1"]   # Warm Terracotta
COLOR_TIME = COLOURS["C2"]   # Celadon Green

def _plot_dual_y_axis(param_name, label, x_values, r_risk_vals, r_time_vals, cost_vals, save_path):
    """Generate a dual y-axis line chart (R_risk left, R_time right) and save as png + tiff."""
    from sensitivity.plot_helpers import set_style
    set_style()

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Format the label for LaTeX (e.g., 'alpha_max' -> r'$\alpha_{max}$')
    latex_label = label
    if "_" in label:
        parts = label.split("_", 1)
        if parts[0] in ["alpha", "beta", "gamma", "delta", "epsilon", "theta"]:
            latex_label = rf"$\{parts[0]}_{{{parts[1]}}}$"

    # Categorical x positions + adaptive decimal labels
    x_pos = np.arange(len(x_values))
    from sensitivity.plot_helpers import _fmt_values
    x_labels = _fmt_values(x_values)

    # Left y-axis: R_risk
    ax1.set_xlabel(latex_label, fontsize=14)
    ax1.set_ylabel(r'Network Risk Reliability ($R_{risk}$)', fontsize=14, color=COLOR_RISK)
    line1, = ax1.plot(x_pos, r_risk_vals, color=COLOR_RISK, marker='o', markersize=5,
                       linewidth=1.8, label=r'Risk Reliability ($R_{risk}$)', zorder=5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=11, rotation=45, ha='right')
    ax1.tick_params(axis='x', direction='in', labelsize=11)
    ax1.tick_params(axis='y', direction='in', labelcolor=COLOR_RISK, labelsize=11)
    ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))

    r_risk_min = min(r_risk_vals) if r_risk_vals else 0
    r_risk_max = max(r_risk_vals) if r_risk_vals else 1
    margin_r = max(0.01, (r_risk_max - r_risk_min) * 0.15)
    ax1.set_ylim(r_risk_min - margin_r, r_risk_max + margin_r)
    ax1.set_xlim(-0.5, len(x_values) - 0.5)
    ax1.spines['top'].set_visible(False)

    # Right y-axis: R_time
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Network Time Reliability ($R_{time}$)', fontsize=14, color=COLOR_TIME)
    line2, = ax2.plot(x_pos, r_time_vals, color=COLOR_TIME, marker='s', markersize=5,
                       linewidth=1.8, linestyle='--', label=r'Time Reliability ($R_{time}$)', zorder=4)
    ax2.tick_params(axis='y', direction='in', labelcolor=COLOR_TIME, labelsize=11)
    r_time_min = min(r_time_vals) if r_time_vals else 0
    r_time_max = max(r_time_vals) if r_time_vals else 1
    margin_t = max(0.01, (r_time_max - r_time_min) * 0.15)
    ax2.set_ylim(r_time_min - margin_t, r_time_max + margin_t)
    ax2.spines['top'].set_visible(False)

    # Combined legend (no frame)
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=11)

    plt.tight_layout()

    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches='tight', format=ext)
    plt.close(fig)
    return f"{save_path}.png", f"{save_path}.tiff"


def run_sensitivity(param_name: str, param_values: list, param_attr_path: str, 
                    instance_path: str, label: str, time_limit: int = 60,
                    output_root: str = "results", config_setter=None):
    """
    Run sensitivity analysis for one parameter.
    
    Args:
        param_name: short name used for folder naming (e.g. "alpha_max")
        param_values: list of 10 float values for x-axis
        param_attr_path: dotted path to set on config (e.g. "alpha_max")
        instance_path: path to instance JSON
        label: human-readable label for the summary
        time_limit: solver time limit per run in seconds
        output_root: root output directory
    
    Returns:
        list of result dicts for each parameter value
    """
    with open(instance_path) as f:
        instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_root, f"sensitivity_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Single experiment log — captures ALL print() + logger output
    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # Run each parameter value
    results = []
    for val in param_values:
        config = RHMVSPConfig()
        # Set the parameter
        if config_setter:
            config_setter(config, val)
        else:
            parts = param_attr_path.split(".")
            obj = config
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)
        config.solver_name = "gurobi"
        config.solver_backend = "gurobi"
        config.time_limit = time_limit

        # Per-value subfolder
        val_str = str(val).replace(".", "_")
        val_dir = os.path.join(exp_dir, f"{param_name}={val_str}")
        os.makedirs(val_dir, exist_ok=True)

        # Ensure instance uses the updated config.alpha_T
        for od_data in instance.get("od_pairs", []):
            od_data["alpha_T_od"] = config.alpha_T

        # Recompute L_od for each OD pair based on current alpha_T.
        # L_od determines the delivery deadline; R_T = M{Z~ <= L_od} is
        # only sensitive to alpha_T if L_od itself changes with alpha_T.
        from src.core.uncertainty import UncertaintyEngine as UE2, TrapezoidalFuzzy as TrapFz
        _temp_engine = UE2(config)
        _arcs = instance["network"]["arcs"]
        _hub_ids = {tn["node_id"] for tn in instance["network"].get("transfer_nodes", [])}
        for od in instance["od_pairs"]:
            od_o, od_d = od["origin"], od["destination"]
            # Filter for actual hub nodes only (transfer_nodes define hubs)
            roads_out = [a for a in _arcs if a["i"] == od_o and a["mode"] == 1 and a["j"] in _hub_ids]
            roads_in  = [a for a in _arcs if a["j"] == od_d and a["mode"] == 1 and a["i"] in _hub_ids]
            rails      = [a for a in _arcs if a["mode"] == 2]
            if roads_out and roads_in and rails:
                co = min(roads_out, key=lambda a: a["distance"])
                ci = min(roads_in,  key=lambda a: a["distance"])
                rm = [a for a in rails if a["i"] == co["j"] and a["j"] == ci["i"]]
                if rm:
                    r1 = _temp_engine.make_transport_time(co["distance"], 1)
                    tf = _temp_engine.make_transfer_time()
                    ra = _temp_engine.make_transport_time(rm[0]["distance"], 2)
                    r2 = _temp_engine.make_transport_time(ci["distance"], 1)
                    trap = TrapFz(r1.a+tf.a+ra.a+r2.a, r1.b+tf.b+ra.b+r2.b,
                                  r1.c+tf.c+ra.c+r2.c, r1.d+tf.d+ra.d+r2.d)
                    det_L = trap.deterministic_equivalent_upper(config.alpha_T)
                    od["L_od"] = float(round(od.get("E_od", 0.0) + det_L * 1.10, 1))

        # Logger (output goes through Tee → experiment.log)
        logger = Logger(level="INFO", log_file=None)
        logger.info(f"RHMVSP Sensitivity: {label} = {val}")
        logger.info(f"Config: alpha_max={config.alpha_max}, alpha_T={config.alpha_T}, "
                     f"rail_fuzzy_time_ratio={config.rail_fuzzy_time_ratio}, "
                     f"delta_T={config.delta_T}")
        # Diagnostic: confirm parameter propagation
        logger.info(f"  Derived: beta_od={config.beta_od:.4f}, beta_net={config.beta_net:.4f}")

        try:
            solver = BAPSolver(config, instance, logger)
            sol = solver.solve()

            # Save solution.json
            if sol and sol.get("status") != "infeasible":
                result_path = os.path.join(val_dir, "solution.json")
                with open(result_path, "w") as f:
                    json.dump(sol, f, indent=2, default=str)
                logger.info(f"Solution saved to: {result_path}")

                r_risk = sol.get("R_net_risk", 0)
                r_time = sol.get("R_net_time", 0)
                cost = sol.get("total_cost", 0)
                gap_pct = sol.get("gap_pct", 0.0)
                results.append({
                    "param": val, "status": "optimal",
                    "cost": cost, "R_risk": r_risk, "R_time": r_time,
                    "gap_pct": gap_pct,
                    "od_details": sol.get("od_reliability_details", {}),
                })
                logger.info(f"\nResult: {label}={val}: Cost=¥{cost:,.0f}, R_risk={r_risk:.4f}, R_time={r_time:.4f}")
            else:
                results.append({"param": val, "status": "infeasible"})
                logger.warning(f"Status: INFEASIBLE")

        except Exception as e:
            logger.error(f"ERROR: {e}")
            results.append({"param": val, "status": "error"})
            import traceback
            logger.error(traceback.format_exc())

        logger.info("Done.")

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SENSITIVITY: {label} vs R_risk, R_time")
    print(f"Output: {exp_dir}")
    print(f"{'=' * 80}")
    header = f"{label:>12} {'Status':>10} {'Cost':>14} {'R_risk':>8} {'R_time':>8}"
    # Per-OD columns
    for od_idx in range(5):
        header += f" {'Risk_'+str(od_idx+1):>8} {'Time_'+str(od_idx+1):>8}"
    print(header)
    print("-" * 80)
    for r in results:
        if r.get("status") != "optimal":
            print(f"{r['param']:>12.2f} {r['status']:>10}")
            continue
        line = f"{r['param']:>12.2f} {'OPTIMAL':>10} {r['cost']:>14,.0f} {r['R_risk']:>8.4f} {r['R_time']:>8.4f}"
        od_details = r.get("od_details", {})
        for od_idx in range(5):
            od_r = od_details.get(od_idx, {})
            line += f" {od_r.get('R_od_risk', 0):>8.4f} {od_r.get('R_od_time', 0):>8.4f}"
        print(line)
    print(f"{'=' * 80}")

    # Save summary CSV
    csv_path = os.path.join(exp_dir, f"summary_{param_name}.csv")
    with open(csv_path, "w") as f:
        f.write(f"{param_name},status,cost,R_risk,R_time,Gap_pct")
        for od_idx in range(5):
            f.write(f",R_od_{od_idx+1}_risk,R_od_{od_idx+1}_time")
        f.write("\n")
        for r in results:
            f.write(f"{r['param']},{r.get('status','?')}")
            if r.get("status") == "optimal":
                f.write(f",{r['cost']:.0f},{r['R_risk']:.4f},{r['R_time']:.4f},{r.get('gap_pct',0.0):.4f}")
                od_details = r.get("od_details", {})
                for od_idx in range(5):
                    od_r = od_details.get(od_idx, {})
                    f.write(f",{od_r.get('R_od_risk', 0):.4f},{od_r.get('R_od_time', 0):.4f}")
            f.write("\n")
    print(f"Summary CSV saved to: {csv_path}")

    # Generate dual Y-axis sensitivity chart
    x_vals = [r['param'] for r in results if r.get('status') == 'optimal']
    r_risk_vals = [r['R_risk'] for r in results if r.get('status') == 'optimal']
    r_time_vals = [r['R_time'] for r in results if r.get('status') == 'optimal']
    cost_vals = [r['cost'] for r in results if r.get('status') == 'optimal']

    if x_vals:
        chart_path = os.path.join(exp_dir, f"sensitivity_{param_name}")
        try:
            _plot_dual_y_axis(param_name, label, x_vals, r_risk_vals, r_time_vals, cost_vals, chart_path)
            print(f"Dual Y-axis chart saved to: {chart_path}.png/.tiff")
        except Exception as e:
            print(f"Chart generation failed: {e}")

    tee.close()
    return results
