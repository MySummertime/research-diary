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

from config import RHMVSPConfig, default_empty_reposition_hub_fleet
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.utils.numeric_format import format_csv_value
from sensitivity.parameter_grids import BASELINE_ALPHA_R

# ── Experiment parameters ─────────────────────────────────────────────────────
INSTANCE_PATH = "data_real_case/instance.json"

# Fallback fleet centres used by the main experiment instance.
BASE_FLEET = default_empty_reposition_hub_fleet()

# Absolute number of vehicles available at each provincial fleet centre.
# The main experiment uses 50 vehicles at every centre.
FLEET_SIZE_VALUES = [10, 20, 30, 40, 50, 60, 70, 80]


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
    cost_cols = ["transport_cost", "transfer_cost",
                 "reposition_cost", "unmet_penalty"]
    result = {"cost": sol.get("total_cost") if ok else None,
              "network_risk_reliability": sol.get("network_risk_reliability", 0) if ok else None,
              "network_time_reliability": sol.get("network_time_reliability", 0) if ok else None,
              "network_risk": sol.get("network_risk", 0.0) if ok else None,
              "network_time": sol.get("network_time", 0.0) if ok else None,
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

    fleet_hubs = tuple(
        str(hub_id)
        for hub_id in base_instance.get("hub_fleet_size", BASE_FLEET)
    )

    # Save experiment metadata
    meta = {
        "main_experiment_fleet_size": 50,
        "fleet_hubs": fleet_hubs,
        "fleet_size_values": FLEET_SIZE_VALUES,
        "instance": INSTANCE_PATH,
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = len(FLEET_SIZE_VALUES)
    done = 0

    # CSV header
    csv_path = os.path.join(exp_dir, "experiment.csv")
    cost_cols = ["transport_cost", "transfer_cost",
                 "reposition_cost", "unmet_penalty"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
             ["group_label", "fleet_size", "fleet_1", "fleet_12", "fleet_38",
             "model", "cost"] + cost_cols +
            ["network_risk_reliability", "network_time_reliability", "Network_Risk", "Network_Time",
             "vehicles_used", "repositions", "status", "elapsed_s", "Routes"]
        )

    for fleet_size in FLEET_SIZE_VALUES:
        label = str(fleet_size)
        fleet_by_hub = {hub_id: fleet_size for hub_id in fleet_hubs}

        print(f"\n=== Fleet Size: {fleet_size} vehicles per centre ({fleet_by_hub}) ===")

        config = RHMVSPConfig()
        config.alpha_T = 0.8
        config.alpha_R = BASELINE_ALPHA_R
        config.time_limit = 3600
        config.solver_name = "gurobi"

        inst = json.loads(json.dumps(base_instance))
        inst["hub_fleet_size"] = fleet_by_hub

        logger = Logger(level="INFO", log_file=None)
        r = _run_single(config, inst, logger)
        done += 1

        with open(csv_path, "a", newline="") as f:
            row = [label, fleet_size, fleet_by_hub.get("1"),
                   fleet_by_hub.get("12"), fleet_by_hub.get("38"),
                   "BAP", format_csv_value("cost", r["cost"])]
            for col in cost_cols:
                row.append(format_csv_value(col, r.get(col)) if r["cost"] is not None else "")
            row += [format_csv_value("network_risk_reliability", r.get("network_risk_reliability")),
                    format_csv_value("network_time_reliability", r.get("network_time_reliability")),
                    format_csv_value("network_risk", r.get("network_risk")),
                    format_csv_value("network_time", r.get("network_time")),
                    r.get("vehicles_used"), r.get("repositions"),
                    r["status"], r["elapsed"], r.get("routes", "")]
            csv.writer(f).writerow(row)

        # Progress
        c_str = f"CNY {r['cost']:,.0f}" if r["cost"] is not None else r["status"]
        v_str = str(r.get("vehicles_used", "N/A"))
        rr = r.get('network_risk_reliability')
        rt = r.get('network_time_reliability')
        rr_str = format_csv_value("risk_reliability", rr) if rr is not None else "N/A"
        rt_str = format_csv_value("time_reliability", rt) if rt is not None else "N/A"
        print(f"  [{done}/{total}]  Cost={c_str}  R_risk={rr_str}  "
              f"R_time={rt_str}  Vehicles={v_str}  "
              f"({r['elapsed']:.1f}s)")

    print(f"CSV saved → {csv_path}")

    print(f"\nResults saved to {exp_dir}/")
    print(f"Plot with: python sensitivity/plot_fleet_size.py --csv {csv_path}")
    tee.close()
    sys.stdout = sys.__stdout__
    return exp_dir


if __name__ == "__main__":
    print("Fleet Size Sensitivity Analysis")
    print("=" * 50)
    exp_dir = run_fleet_size_sensitivity()
    print(f"\nDone! Data are in {exp_dir}/experiment.csv")
