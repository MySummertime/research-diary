"""Sensitivity of risk reliability threshold using the main experiment setup."""
import csv
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from sensitivity.plot_helpers import dual_y_axis_with_cost_bar
from sensitivity.utils import format_routing_summary, print_od_routing
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger, Tee


INSTANCE_PATH = "data_real_case/instance.json"
ALPHA_MAX_VALUES = [0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985]


def run_alpha_max_sensitivity():
    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"sensitivity_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    old_stdout = sys.stdout
    sys.stdout = tee
    results = []
    cost_cols = ["deploy_cost", "hub_cost", "transport_cost", "transfer_cost",
                 "reposition_cost", "total_prep_cost", "idle_cost", "unmet_penalty"]

    try:
        for alpha_max in ALPHA_MAX_VALUES:
            config = RHMVSPConfig()
            config.alpha_max = 1.0 - alpha_max
            config.alpha_T = 0.8
            config.time_limit = 3600
            config.solver_name = "gurobi"

            instance = json.loads(json.dumps(base_instance))
            for od in instance.get("od_pairs", []):
                od["alpha_T_od"] = config.alpha_T

            label = f"alpha_max={alpha_max:.3f}"
            val_dir = os.path.join(exp_dir, f"alpha_max={str(alpha_max).replace('.', '_')}")
            os.makedirs(val_dir, exist_ok=True)
            logger = Logger(level="INFO", log_file=None)
            logger.info(f"RHMVSP Sensitivity: {label}")
            logger.info(f"Config: alpha_max={config.alpha_max}, alpha_T={config.alpha_T}, T_max={config.T_max}")
            logger.info("Time windows: original instance deadlines")

            try:
                solution = BAPSolver(config, instance, logger).solve()
                print_od_routing(solution, instance, label)
                if solution.get("status") == "infeasible":
                    results.append({"param": alpha_max, "status": "infeasible"})
                    continue

                with open(os.path.join(val_dir, "solution.json"), "w") as f:
                    json.dump(solution, f, indent=2, default=str)

                result = {
                    "param": alpha_max,
                    "status": solution.get("status", "optimal"),
                    "cost": solution.get("total_cost", 0),
                    "R_risk": solution.get("R_net_risk", 0),
                    "R_time": solution.get("R_net_time", 0),
                    "raw_risk": solution.get("raw_net_risk", 0),
                    "raw_time": solution.get("raw_net_time", 0),
                    "gap_pct": solution.get("gap_pct", 0),
                    "od_details": solution.get("od_reliability_details", {}),
                    "routes": format_routing_summary(solution, instance),
                }
                result.update({k: solution.get(k, 0) for k in cost_cols})
                results.append(result)
                logger.info(f"Result: {label}: Cost=¥{result['cost']:,.2f}, R_risk={result['R_risk']:.6f}, R_time={result['R_time']:.6f}")
            except Exception as exc:
                logger.error(str(exc))
                results.append({"param": alpha_max, "status": "error"})

        csv_path = os.path.join(exp_dir, "summary_alpha_max.csv")
        fields = ["alpha_max", "status", "cost"] + cost_cols + ["R_risk", "R_time", "Raw_Risk", "Raw_Time", "Gap_pct"]
        for i in range(1, 6):
            fields += [f"R_od_{i}_risk", f"R_od_{i}_time", f"Raw_od_{i}_risk", f"Raw_od_{i}_time"]
        fields.append("Routes")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for result in results:
                row = {"alpha_max": result["param"], "status": result.get("status", "error")}
                if result.get("status") not in ("infeasible", "error"):
                    row.update({"cost": result["cost"], **{k: result.get(k, 0) for k in cost_cols},
                                "R_risk": result["R_risk"], "R_time": result["R_time"],
                                "Raw_Risk": result["raw_risk"], "Raw_Time": result["raw_time"],
                                "Gap_pct": result["gap_pct"], "Routes": result["routes"]})
                    for i in range(1, 6):
                        details = result.get("od_details", {}).get(i - 1, {})
                        row.update({f"R_od_{i}_risk": details.get("R_od_risk", 0),
                                    f"R_od_{i}_time": details.get("R_od_time", 0),
                                    f"Raw_od_{i}_risk": details.get("raw_risk", 0),
                                    f"Raw_od_{i}_time": details.get("raw_time", 0)})
                writer.writerow(row)

        valid = [r for r in results if r.get("status") not in ("infeasible", "error")]
        if valid:
            dual_y_axis_with_cost_bar(
                x_label=r"$\alpha_{max}$", x_vals=[r["param"] for r in valid],
                r_risk_vals=[r["R_risk"] for r in valid], r_time_vals=[r["R_time"] for r in valid],
                cost_vals=[r["cost"] for r in valid], save_path=os.path.join(exp_dir, "sensitivity_alpha_max"),
                y1_label="Risk reliability", y2_label="Time reliability")
            dual_y_axis_with_cost_bar(
                x_label=r"$\alpha_{max}$", x_vals=[r["param"] for r in valid],
                r_risk_vals=[r["raw_risk"] for r in valid], r_time_vals=[r["raw_time"] for r in valid],
                cost_vals=[r["cost"] for r in valid], save_path=os.path.join(exp_dir, "sensitivity_alpha_max_raw"),
                y1_label="Risk", y2_label="Time (h)", y1_title="Risk / Time")
        print(f"Results saved to {exp_dir}/")
    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    run_alpha_max_sensitivity()
