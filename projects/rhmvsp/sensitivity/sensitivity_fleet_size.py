"""
Sensitivity Analysis: Fleet Size vs Total Cost, R_risk, R_time

Varies hub_fleet_size to observe how fleet availability affects:
  - Total logistics cost
  - Network risk reliability (R_risk)
  - Network time reliability (R_time)

Uses data_real_case/instance.json with per-hub fleet constraints.
"""
import sys, os, json, csv
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from sensitivity.plot_helpers import dual_y_axis_with_cost_bar, set_style
from sensitivity.utils import update_od_deadlines as _update_od_deadlines

# ── Colour constants ──────────────────────────────────────────────────────────
from sensitivity.plot_helpers import COLOURS

# ── Experiment parameters ─────────────────────────────────────────────────────
INSTANCE_PATH = "data_real_case/instance.json"

# Base fleet sizes (from instance)
BASE_FLEET = {"2": 35, "5": 15, "38": 12}

# Scale factors: multiply base fleet by these factors
# 0.5 = half fleet (very tight), 1.0 = base fleet, 2.0 = double fleet (loose)
FLEET_SCALE_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]


def _run_single(config, inst, logger):
    """Run one BAP solve and return dict with cost and sub-components."""
    solver = BAPSolver(config, inst, logger)
    import time
    t0 = time.time()
    sol = solver.solve()
    from sensitivity.utils import print_od_routing, format_routing_summary; print_od_routing(sol, inst)
    elapsed = time.time() - t0

    status = sol.get("status", "unknown")
    ok = status != "infeasible"
    cost_cols = ["deploy_cost", "hub_cost", "transport_cost", "transfer_cost",
                 "reposition_cost", "total_prep_cost", "idle_cost", "unmet_penalty"]
    result = {"cost": sol.get("total_cost") if ok else None,
              "R_risk": sol.get("R_net_risk", 0) if ok else None,
              "R_time": sol.get("R_net_time", 0) if ok else None,
              "raw_risk": sol.get("raw_net_risk", 0.0) if ok else None,
              "raw_time": sol.get("raw_net_time", 0.0) if ok else None,
              "status": status, "elapsed": round(elapsed, 2),
              "vehicles_used": sol.get("num_vehicles_used", 0),
              "repositions": sol.get("num_repositions", 0),
              "routes": format_routing_summary(sol, inst)}
    for col in cost_cols:
        result[col] = sol.get(col, 0) if ok else None
    return result


def run_fleet_size_sensitivity():
    """Run fleet size sensitivity experiment."""
    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"sensitivity_fleet_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Single experiment log
    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # Save experiment metadata
    meta = {
        "base_fleet": BASE_FLEET,
        "scale_values": FLEET_SCALE_VALUES,
        "instance": INSTANCE_PATH,
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    group_labels = [f"{s:.1f}x" for s in FLEET_SCALE_VALUES]
    cost_data = {l: [] for l in group_labels}
    r_risk_data = {l: [] for l in group_labels}
    r_time_data = {l: [] for l in group_labels}
    raw_risk_data = {l: [] for l in group_labels}
    raw_time_data = {l: [] for l in group_labels}

    total = len(FLEET_SCALE_VALUES)
    done = 0

    # CSV header
    csv_path = os.path.join(exp_dir, "experiment.csv")
    cost_cols = ["deploy_cost", "hub_cost", "transport_cost", "transfer_cost",
                 "reposition_cost", "total_prep_cost", "idle_cost", "unmet_penalty"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["group_label", "fleet_scale", "fleet_2", "fleet_5", "fleet_38",
             "model", "cost"] + cost_cols +
            ["R_risk", "R_time", "Raw_Risk", "Raw_Time",
             "vehicles_used", "repositions", "status", "elapsed_s", "Routes"]
        )

    for scale in FLEET_SCALE_VALUES:
        label = f"{scale:.1f}x"
        fleet_2 = int(BASE_FLEET["2"] * scale)
        fleet_5 = int(BASE_FLEET["5"] * scale)
        fleet_38 = int(BASE_FLEET["38"] * scale)

        print(f"\n=== Fleet Scale: {scale:.1f}x (Hub2={fleet_2}, Hub5={fleet_5}, Hub38={fleet_38}) ===")

        config = RHMVSPConfig()
        config.alpha_T = 0.8
        config.alpha_max = 0.05
        config.time_limit = 120
        config.solver_name = "gurobi"

        inst = json.loads(json.dumps(base_instance))
        inst["hub_fleet_size"] = {"2": fleet_2, "5": fleet_5, "38": fleet_38}
        _update_od_deadlines(inst, config)

        logger = Logger(level="INFO", log_file=None)
        r = _run_single(config, inst, logger)
        done += 1

        # Store results
        if r["cost"] is not None:
            cost_data[label].append(r["cost"])
            r_risk_data[label].append(r["R_risk"])
            r_time_data[label].append(r["R_time"])
            raw_risk_data[label].append(r["raw_risk"])
            raw_time_data[label].append(r["raw_time"])

        with open(csv_path, "a", newline="") as f:
            row = [label, scale, fleet_2, fleet_5, fleet_38,
                   "BAP", r["cost"]]
            for col in cost_cols:
                row.append(r.get(col) if r["cost"] is not None else None)
            row += [r.get("R_risk"), r.get("R_time"),
                    r.get("raw_risk"), r.get("raw_time"),
                    r.get("vehicles_used"), r.get("repositions"),
                    r["status"], r["elapsed"], r.get("routes", "")]
            csv.writer(f).writerow(row)

        # Progress
        c_str = f"CNY {r['cost']:,.0f}" if r["cost"] is not None else r["status"]
        v_str = str(r.get("vehicles_used", "N/A"))
        rr = r.get('R_risk')
        rt = r.get('R_time')
        rr_str = f"{rr:.4f}" if rr is not None else "N/A"
        rt_str = f"{rt:.4f}" if rt is not None else "N/A"
        print(f"  [{done}/{total}]  Cost={c_str}  R_risk={rr_str}  "
              f"R_time={rt_str}  Vehicles={v_str}  "
              f"({r['elapsed']:.1f}s)")

    print(f"CSV saved → {csv_path}")

    # ── Plot dual Y-axis chart ──────────────────────────────────────────────
    x_vals = [float(s.replace("x", "")) for s in group_labels]
    r_risk_vals = [np.mean(r_risk_data[l]) if r_risk_data[l] else 0 for l in group_labels]
    r_time_vals = [np.mean(r_time_data[l]) if r_time_data[l] else 0 for l in group_labels]
    cost_vals = [np.mean(cost_data[l]) if cost_data[l] else 0 for l in group_labels]

    dual_y_axis_with_cost_bar(
        x_label=r"Fleet Scale Factor",
        x_vals=x_vals,
        r_risk_vals=r_risk_vals,
        r_time_vals=r_time_vals,
        cost_vals=cost_vals,
        save_path=os.path.join(exp_dir, "fleet_sensitivity"),
        y1_label=r'$R_{risk}$',
        y2_label=r'$R_{time}$',
        y1_title=r'Network Reliability',
        y2_title=r'Total Cost (CNY)',
    )
    print(f"\nChart saved to: {exp_dir}/fleet_sensitivity.png/.tiff")

    # ── Plot dual Y-axis chart (Raw Metrics) ────────────────────────────────
    raw_risk_vals = [np.mean(raw_risk_data[l]) if raw_risk_data[l] else 0 for l in group_labels]
    raw_time_vals = [np.mean(raw_time_data[l]) if raw_time_data[l] else 0 for l in group_labels]

    dual_y_axis_with_cost_bar(
        x_label=r"Fleet Scale Factor",
        x_vals=x_vals,
        r_risk_vals=raw_risk_vals,
        r_time_vals=raw_time_vals,
        cost_vals=cost_vals,
        save_path=os.path.join(exp_dir, "fleet_sensitivity_raw"),
        y1_label=r'Risk',
        y2_label=r'Time (h)',
        y1_title=r'Risk / Time',
        y2_title=r'Total Cost (CNY)',
    )
    print(f"Chart saved to: {exp_dir}/fleet_sensitivity_raw.png/.tiff")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Scale':>8s} {'Fleet':>15s} {'Cost':>14s} {'R_risk':>8s} {'R_time':>8s} {'Vehicles':>10s}")
    for label in group_labels:
        if cost_data[label]:
            c = np.mean(cost_data[label])
            rr = np.mean(r_risk_data[label])
            rt = np.mean(r_time_vals[group_labels.index(label):group_labels.index(label)+1])
            fleet_str = f"{int(BASE_FLEET['2']*float(label.replace('x','')))}+{int(BASE_FLEET['5']*float(label.replace('x','')))}+{int(BASE_FLEET['38']*float(label.replace('x','')))}"
            print(f"{label:>8s} {fleet_str:>15s} {c:>14,.0f} {rr:>8.4f} {rt:>8.4f}")

    print(f"\nResults saved to {exp_dir}/")
    tee.close()
    sys.stdout = sys.__stdout__
    return exp_dir


if __name__ == "__main__":
    print("Fleet Size Sensitivity Analysis")
    print("=" * 50)
    exp_dir = run_fleet_size_sensitivity()
    print(f"\nDone!  Open {exp_dir}/fleet_sensitivity.png and fleet_sensitivity_raw.png")
