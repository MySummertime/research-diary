"""Sensitivity of time reliability threshold using the main experiment setup."""
import csv
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from sensitivity.parameter_grids import ALPHA_T_SCAN, BASELINE_ALPHA_R
from sensitivity.utils import format_routing_summary, print_od_routing
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger, Tee
from src.utils.numeric_format import format_alpha_R, format_alpha_T, format_csv_row, format_csv_value


INSTANCE_PATH = "data_real_case/instance.json"
ALPHA_T_VALUES = ALPHA_T_SCAN


def run_alpha_t_sensitivity():
    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"sensitivity_alpha_T_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    old_stdout = sys.stdout
    sys.stdout = tee
    results = []
    cost_cols = ["transport_cost", "transfer_cost",
                 "reposition_cost", "unmet_penalty"]

    try:
        for alpha_t in ALPHA_T_VALUES:
            config = RHMVSPConfig()
            config.alpha_R = BASELINE_ALPHA_R
            config.alpha_T = alpha_t
            config.time_limit = 3600
            config.solver_name = "gurobi"

            instance = json.loads(json.dumps(base_instance))
            for od in instance.get("od_pairs", []):
                od["alpha_T_od"] = config.alpha_T

            alpha_t_text = format_alpha_T(alpha_t)
            label = f"alpha_T={alpha_t_text}"
            val_dir = os.path.join(exp_dir, f"alpha_T={str(alpha_t).replace('.', '_')}")
            os.makedirs(val_dir, exist_ok=True)
            logger = Logger(level="INFO", log_file=None)
            logger.info(f"RHMVSP Sensitivity: {label}")
            logger.info(
                f"Config: alpha_R={format_alpha_R(config.alpha_R)}, "
                f"alpha_T={alpha_t_text}, T_max={config.T_max}"
            )
            logger.info("Time windows: original instance deadlines")

            try:
                solution = BAPSolver(config, instance, logger).solve()
                print_od_routing(solution, instance, label)
                if solution.get("status") == "infeasible":
                    results.append({"param": alpha_t, "status": "infeasible"})
                    continue

                with open(os.path.join(val_dir, "solution.json"), "w") as f:
                    json.dump(solution, f, indent=2, default=str)

                result = {
                    "param": alpha_t,
                    "status": solution.get("status", "optimal"),
                    "cost": solution.get("total_cost", 0),
                    "network_risk_reliability": solution.get("network_risk_reliability", 0),
                    "network_time_reliability": solution.get("network_time_reliability", 0),
                    "network_risk": solution.get("network_risk", 0),
                    "network_time": solution.get("network_time", 0),
                    "gap_pct": solution.get("gap_pct", 0),
                    "od_details": solution.get("od_reliability_details", {}),
                    "routes": format_routing_summary(solution, instance),
                }
                result.update({k: solution.get(k, 0) for k in cost_cols})
                results.append(result)
                logger.info(
                    f"Result: {label}: Cost=¥{result['cost']:,.2f}, "
                    f"R_risk={format_csv_value('risk_reliability', result['network_risk_reliability'])}, "
                    f"R_time={format_csv_value('time_reliability', result['network_time_reliability'])}"
                )
            except Exception as exc:
                logger.error(str(exc))
                results.append({"param": alpha_t, "status": "error"})

        csv_path = os.path.join(exp_dir, "summary_alpha_T.csv")
        fields = ["alpha_T", "status", "cost"] + cost_cols + ["network_risk_reliability", "network_time_reliability", "Network_Risk", "Network_Time", "Gap_pct"]
        for i in range(1, 6):
            fields += [f"OD_{i}_Risk_Reliability", f"OD_{i}_Time_Reliability",
                       f"OD_{i}_Risk", f"OD_{i}_Time"]
        fields.append("Routes")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for result in results:
                row = {
                    "alpha_T": format_alpha_T(result["param"]),
                    "status": result.get("status", "error"),
                }
                if result.get("status") not in ("infeasible", "error"):
                    row.update({"cost": result["cost"], **{k: result.get(k, 0) for k in cost_cols},
                                "network_risk_reliability": result["network_risk_reliability"], "network_time_reliability": result["network_time_reliability"],
                                "Network_Risk": result["network_risk"], "Network_Time": result["network_time"],
                                "Gap_pct": result["gap_pct"], "Routes": result["routes"]})
                    for i in range(1, 6):
                        details = result.get("od_details", {}).get(i - 1, {})
                        row.update({f"OD_{i}_Risk_Reliability": details.get("od_risk_reliability", 0),
                                    f"OD_{i}_Time_Reliability": details.get("od_time_reliability", 0),
                                    f"OD_{i}_Risk": details.get("od_risk", 0),
                                    f"OD_{i}_Time": details.get("od_time", 0)})
                writer.writerow(format_csv_row(row))

        print(f"Results saved to {exp_dir}/")
        print(f"Plot with: python sensitivity/plot_alpha_T.py --csv {csv_path}")
    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    run_alpha_t_sensitivity()
