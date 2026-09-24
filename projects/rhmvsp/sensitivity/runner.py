"""
Shared sensitivity analysis runner.
Creates per-value subfolders under results/sensitivity_<timestamp>/ with
log, solution.json, network.png/tiff, solution_plot.png/tiff — same as main.py.
"""
import json
import os
import sys
from datetime import datetime

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.utils.numeric_format import format_alpha_R, format_alpha_T, format_csv_value, format_parameter


def run_sensitivity(param_name: str, param_values: list, param_attr_path: str, 
                    instance_path: str, label: str, time_limit: int = 60,
                    output_root: str = "results", config_setter=None,
                    use_alpha_T_overhead: bool = False,
                    invert_param: bool = False):
    """
    Run sensitivity analysis for one parameter.
    
    Args:
        param_name: short name used for folder naming (e.g. "alpha_R")
        param_values: list of 10 float values for x-axis
        param_attr_path: dotted path to set on config (e.g. "alpha_R")
        instance_path: path to instance JSON
        label: human-readable label for the summary
        time_limit: solver time limit per run in seconds
        output_root: root output directory
    
    Returns:
        list of result dicts for each parameter value
    """
    with open(instance_path) as f:
        instance = json.load(f)

    def format_sensitivity_value(value) -> str:
        normalized = param_name.lower()
        if normalized == "alpha_r":
            return format_alpha_R(value)
        if normalized == "alpha_t":
            return format_alpha_T(value)
        return format_parameter(value, param_values)

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
        val_text = format_sensitivity_value(val)
        config = RHMVSPConfig()
        # Set the parameter
        if invert_param:
            solver_val = 1.0 - val  # legacy risk tolerance -> reliability threshold
        else:
            solver_val = val
        if config_setter:
            config_setter(config, solver_val)
        else:
            parts = param_attr_path.split(".")
            obj = config
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], solver_val)
        config.solver_name = "gurobi"
        config.time_limit = time_limit

        # Per-value subfolder
        val_str = str(val).replace(".", "_")
        val_dir = os.path.join(exp_dir, f"{param_name}={val_str}")
        os.makedirs(val_dir, exist_ok=True)

        # Deep copy instance to avoid mutation leaking across iterations
        inst = json.loads(json.dumps(instance))

        # Reliability thresholds vary, but every experiment retains the common
        # planning horizon specified centrally by config.T_max.
        from sensitivity.utils import update_od_deadlines
        update_od_deadlines(inst, config, alpha_T_dependent=use_alpha_T_overhead)

        # Logger (output goes through Tee → experiment.log)
        logger = Logger(level="INFO", log_file=None)
        logger.info(f"RHMVSP Sensitivity: {label} = {val_text}")
        logger.info(f"Config: alpha_R={format_alpha_R(config.alpha_R)}, alpha_T={format_alpha_T(config.alpha_T)}, "
                     f"rail_fuzzy_time_ratio={config.rail_fuzzy_time_ratio}")
        # Diagnostic: confirm parameter propagation
        logger.info(
            "  Risk reliability threshold: "
            f"alpha_R={format_alpha_R(config.alpha_R)}"
        )

        try:
            solver = BAPSolver(config, inst, logger)
            sol = solver.solve()
            from sensitivity.utils import print_od_routing; print_od_routing(sol, inst, f"{label}={val_text}")

            # Save solution.json
            if sol and sol.get("status") != "infeasible":
                result_path = os.path.join(val_dir, "solution.json")
                with open(result_path, "w") as f:
                    json.dump(sol, f, indent=2, default=str)
                logger.info(f"Solution saved to: {result_path}")

                r_risk = sol.get("network_risk_reliability", 0)
                r_time = sol.get("network_time_reliability", 0)
                network_risk = sol.get("network_risk", 0.0)
                network_time = sol.get("network_time", 0.0)
                cost = sol.get("total_cost", 0)
                gap_pct = sol.get("gap_pct", 0.0)
                from sensitivity.utils import format_routing_summary
                route_sum = format_routing_summary(sol, inst)
                results.append({
                    "param": val, "status": "optimal",
                    "cost": cost, "network_risk_reliability": r_risk, "network_time_reliability": r_time,
                    "network_risk": network_risk, "network_time": network_time,
                    "gap_pct": gap_pct,
                    "transport_cost": sol.get("transport_cost", 0),
                    "transfer_cost": sol.get("transfer_cost", 0),
                    "reposition_cost": sol.get("reposition_cost", 0),
                    "unmet_penalty": sol.get("unmet_penalty", 0),
                    "od_details": sol.get("od_reliability_details", {}),
                    "routes": route_sum,
                })
                logger.info(
                    f"\nResult: {label}={val_text}: Cost=¥{cost:,.2f}, "
                    f"R_risk={format_csv_value('risk_reliability', r_risk)}, "
                    f"R_time={format_csv_value('time_reliability', r_time)}"
                )
                logger.info(f"  Sub-Costs: Transport=¥{sol.get('transport_cost',0):,.0f}, Transfer=¥{sol.get('transfer_cost',0):,.0f}, Repo=¥{sol.get('reposition_cost',0):,.0f}, Unmet=¥{sol.get('unmet_penalty',0):,.0f}")
                logger.info(
                    "  Raw Metrics: "
                    f"NetworkRisk={format_csv_value('network_risk', network_risk)}, "
                    f"NetworkTime={format_csv_value('network_time', network_time)}h"
                )
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
    print(f"SENSITIVITY: {label} vs R_risk, R_Time")
    print(f"Output: {exp_dir}")
    print(f"{'=' * 80}")
    header = f"{label:>12} {'Status':>10} {'Cost':>14} {'network_risk_reliability':>8} {'network_time_reliability':>8}"
    # Per-OD columns
    for od_idx in range(5):
        header += f" {'Risk_'+str(od_idx+1):>8} {'Time_'+str(od_idx+1):>8}"
    print(header)
    print("-" * 80)
    for r in results:
        if r.get("status") != "optimal":
            print(f"{format_sensitivity_value(r['param']):>12} {r['status']:>10}")
            continue
        line = (
            f"{format_sensitivity_value(r['param']):>12} {'OPTIMAL':>10} "
            f"{r['cost']:>14,.2f} "
            f"{format_csv_value('risk_reliability', r['network_risk_reliability']):>10s} "
            f"{format_csv_value('time_reliability', r['network_time_reliability']):>10s}"
        )
        od_details = r.get("od_details", {})
        for od_idx in range(5):
            od_r = od_details.get(od_idx, {})
            line += (
                f" {format_csv_value('risk_reliability', od_r.get('od_risk_reliability', 0)):>10s}"
                f" {format_csv_value('time_reliability', od_r.get('od_time_reliability', 0)):>10s}"
            )
        print(line)
    print(f"{'=' * 80}")

    # Save summary CSV
    csv_path = os.path.join(exp_dir, f"summary_{param_name}.csv")
    cost_cols = ["transport_cost","transfer_cost",
                 "reposition_cost","unmet_penalty"]
    with open(csv_path, "w") as f:
        f.write(f"{param_name},status,cost,{','.join(cost_cols)},Network_Risk_Reliability,Network_Time_Reliability,Network_Risk,Network_Time,Gap_pct")
        for od_idx in range(5):
            f.write(f",OD_{od_idx+1}_Risk_Reliability,OD_{od_idx+1}_Time_Reliability,OD_{od_idx+1}_Risk,OD_{od_idx+1}_Time")
        f.write(",Routes\n")
        for r in results:
            f.write(f"{format_sensitivity_value(r['param'])},{r.get('status','?')}")
            if r.get("status") == "optimal":
                f.write(f",{format_csv_value('cost', r['cost'])}")
                for col in cost_cols:
                    f.write(f",{format_csv_value(col, r.get(col, 0))}")
                f.write(f",{format_csv_value('network_risk_reliability', r['network_risk_reliability'])},{format_csv_value('network_time_reliability', r['network_time_reliability'])},{format_csv_value('network_risk', r.get('network_risk',0.0))},{format_csv_value('network_time', r.get('network_time',0.0))},{format_csv_value('gap_pct', r.get('gap_pct',0.0))}")
                od_details = r.get("od_details", {})
                for od_idx in range(5):
                    od_r = od_details.get(od_idx, {})
                    f.write(f",{format_csv_value('od_risk_reliability', od_r.get('od_risk_reliability', 0))},{format_csv_value('od_time_reliability', od_r.get('od_time_reliability', 0))},{format_csv_value('od_risk', od_r.get('od_risk', 0))},{format_csv_value('od_time', od_r.get('od_time', 0))}")
                f.write(f",{r.get('routes','')}\n")
            else:
                f.write(",")  # empty cost
                for col in cost_cols:
                    f.write(",")
                f.write(",,")  # R_risk, R_time
                f.write(",,")  # Network_Risk, Network_Time
                f.write(",")  # Gap_pct
                for od_idx in range(5):
                    f.write(",,,,")  # 4 per-OD columns
                f.write(",")  # Routes (empty for infeasible)
            f.write("\n")
    print(f"Summary CSV saved to: {csv_path}")

    tee.close()
    sys.stdout = sys.__stdout__
    return results
