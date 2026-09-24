"""
Sensitivity Analysis: α_R × α_T Coupled Experiment.

Systematically vary both risk reliability limit (α_R) and time reliability
threshold (α_T) jointly to explore the coupled trade-off surface between:
  - Total logistics cost
  - Network risk reliability  (R_risk)
  - Network time reliability  (R_time)

α_R controls the arc accident belief threshold (risk constraint).
α_T controls the time reliability threshold (time constraint).
Both are hard constraints in the pricing problem; tightening either reduces
the feasible route set and increases cost.

Produce:
  1. A CSV table  summarising every grid point (Cost, R_risk, R_time, Gap_pct)
  2. Three bubble heatmaps: Cost, R_risk, R_time over the (α_R, α_T) grid
"""
import sys, os, json, csv
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.utils.numeric_format import format_alpha_R, format_alpha_T, format_csv_value
from sensitivity.parameter_grids import ALPHA_R_SCAN, ALPHA_T_SCAN

# ---- Experiment grids ----
ALPHA_T_VALS = ALPHA_T_SCAN
ALPHA_R_VALS = ALPHA_R_SCAN

# ======================================================================
#                       MAIN EXPERIMENT
# ======================================================================
def run_risk_time_coupled(instance_path="data_real_case/instance.json",
                          time_limit=3600,
                          output_root="results"):
    """Run the α_R × α_T coupled experiment."""
    with open(instance_path) as f:
        instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_root, f"coupled_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Single experiment log — captures ALL print() + logger output
    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # ---- Benchmark: constant model (no time-varying risk) ----
    cfg_bench = RHMVSPConfig()
    cfg_bench.enable_time_varying_risk = False
    if time_limit:
        cfg_bench.time_limit = time_limit
    cfg_bench.solver_name = "gurobi"

    inst_bench = json.loads(json.dumps(instance))
    for od in inst_bench.get("od_pairs", []):
        od["alpha_T_od"] = cfg_bench.alpha_T

    bench_sub = os.path.join(exp_dir, "benchmark_constant")
    os.makedirs(bench_sub, exist_ok=True)
    bench_logger = Logger(level="INFO", log_file=None)
    bench_logger.info("Benchmark: constant model (no time-varying risk)")

    bench_cost = bench_rrisk = bench_rtime = None
    bench_network_risk = bench_network_time = None
    try:
        solver_bench = BAPSolver(cfg_bench, inst_bench, bench_logger)
        sol_bench = solver_bench.solve()
        if sol_bench and sol_bench.get("status") != "infeasible":
            bench_cost  = sol_bench.get("total_cost", 0)
            bench_rrisk = sol_bench.get("network_risk_reliability", 0)
            bench_rtime = sol_bench.get("network_time_reliability", 0)
            bench_network_risk = sol_bench.get("network_risk", 0.0)
            bench_network_time = sol_bench.get("network_time", 0.0)
            bench_logger.info(
                f"  Benchmark Cost={bench_cost:,.2f}  "
                f"R_risk={format_csv_value('risk_reliability', bench_rrisk)}  "
                f"R_time={format_csv_value('time_reliability', bench_rtime)}  "
                f"NetworkRisk={format_csv_value('network_risk', bench_network_risk)}  "
                f"NetworkTime={format_csv_value('network_time', bench_network_time)}h"
            )
            with open(os.path.join(bench_sub, "solution.json"), "w") as sf:
                json.dump(sol_bench, sf, indent=2, default=str)
        else:
            bench_logger.warning("  Benchmark infeasible — gap columns will be blank")
    except Exception as e:
        bench_logger.error(f"  Benchmark error: {e}")
        import traceback
        bench_logger.error(traceback.format_exc())

    print(f"Benchmark: Cost={bench_cost}  R_risk={bench_rrisk}  R_time={bench_rtime}"
          f"  NetworkRisk={bench_network_risk if bench_network_risk is not None else 'N/A'}"
          f"  NetworkTime={bench_network_time if bench_network_time is not None else 'N/A'}")

    csv_path = os.path.join(exp_dir, "summary_coupled.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        cost_cols = ["transport_cost","transfer_cost",
                     "reposition_cost","unmet_penalty"]
        w.writerow(["alpha_R", "alpha_T", "status",
                     "cost"] + cost_cols +
                    ["network_risk_reliability", "network_time_reliability", "Network_Risk", "Network_Time", "Gap_pct", "iterations",
                     "cost_bench", "R_risk_bench", "R_time_bench",
                     "Network_Risk_bench", "Network_Time_bench",
                     "Cost_Gap%", "R_risk_Gap%", "R_time_Gap%",
                     "Routes"])

    results = []
    n = len(ALPHA_R_VALS) * len(ALPHA_T_VALS)
    idx = 0
    for a_R in ALPHA_R_VALS:
        for aT in ALPHA_T_VALS:
            idx += 1
            a_R_text = format_alpha_R(a_R)
            aT_text = format_alpha_T(aT)
            print(f"[{idx}/{n}]  α_R={a_R_text}  α_T={aT_text} ... ", end="", flush=True)

            cfg = RHMVSPConfig()
            cfg.alpha_T = aT          # sweep variable
            cfg.alpha_R = a_R
            if time_limit:
                cfg.time_limit = time_limit
            cfg.solver_name = "gurobi"

            inst = json.loads(json.dumps(instance))
            for od in inst.get("od_pairs", []):
                od["alpha_T_od"] = cfg.alpha_T

            sub = os.path.join(exp_dir, f"aR={a_R_text}_aT={aT_text}")
            os.makedirs(sub, exist_ok=True)
            logger = Logger(level="INFO", log_file=None)

            rec: dict = {"aT": aT, "a_R": a_R}

            try:
                solver = BAPSolver(cfg, inst, logger)
                sol = solver.solve()
                from sensitivity.utils import print_od_routing, format_routing_summary; print_od_routing(sol, inst, f"aR={a_R}_aT={aT}")
                routes_str = format_routing_summary(sol, inst)

                if sol and sol.get("status") != "infeasible":
                    rec["routes"] = routes_str
                    rec.update({
                        "status": sol.get("status", "optimal"),
                        "cost": sol.get("total_cost", 0),
                        "transport_cost": sol.get("transport_cost", 0),
                        "transfer_cost": sol.get("transfer_cost", 0),
                        "reposition_cost": sol.get("reposition_cost", 0),
                        "unmet_penalty": sol.get("unmet_penalty", 0),
                        "network_risk_reliability": sol.get("network_risk_reliability", 0),
                        "network_time_reliability": sol.get("network_time_reliability", 0),
                        "network_risk": sol.get("network_risk", 0.0),
                        "network_time": sol.get("network_time", 0.0),
                        "gap_pct": sol.get("gap_pct", 0.0),
                        "iterations": sol.get("iterations", 0),
                    })
                    with open(os.path.join(sub, "solution.json"), "w") as sf:
                        json.dump(sol, sf, indent=2, default=str)
                    print(
                        f"Cost={rec['cost']:,.2f}  "
                        f"R_risk={format_csv_value('risk_reliability', rec['network_risk_reliability'])}  "
                        f"R_time={format_csv_value('time_reliability', rec['network_time_reliability'])}  "
                        f"NetworkRisk={format_csv_value('network_risk', rec['network_risk'])}  "
                        f"NetworkTime={format_csv_value('network_time', rec['network_time'])}h  "
                        f"Gap={format_csv_value('gap_pct', rec['gap_pct'])}%"
                    )
                else:
                    rec["status"] = "infeasible"
                    print("INFEASIBLE")
            except Exception as e:
                rec["status"] = "error"
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

            results.append(rec)

            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                ok = rec.get("status") not in ("infeasible", "error")
                rc = rec.get("cost", 0)   if ok else 0.0
                rr = rec.get("network_risk_reliability", 0) if ok else 0.0
                rt = rec.get("network_time_reliability", 0) if ok else 0.0

                if ok and bench_cost and bench_cost > 0:
                    cost_g  = f"{(rc - bench_cost)  / bench_cost  * 100:+.2f}"
                    rr_g    = f"{(rr - bench_rrisk) / bench_rrisk * 100:+.2f}" if bench_rrisk else ""
                    rt_g    = f"{(rt - bench_rtime) / bench_rtime * 100:+.2f}" if bench_rtime else ""
                else:
                    cost_g = rr_g = rt_g = ""

                row = [a_R_text, aT_text,
                    rec.get("status", "?"),
                    f"{rc:.2f}"  if ok else ""]
                for col in cost_cols:
                    row.append(f"{rec.get(col, 0):.2f}" if ok else "")
                raw_r = rec.get("network_risk", 0) if ok else 0.0
                raw_t = rec.get("network_time", 0) if ok else 0.0
                row += [
                    format_csv_value("risk_reliability", rr) if ok else "",
                    format_csv_value("time_reliability", rt) if ok else "",
                    format_csv_value("network_risk", raw_r) if ok else "",
                    format_csv_value("network_time", raw_t) if ok else "",
                    format_csv_value("gap_pct", rec.get("gap_pct", 0)) if ok else "",
                    rec.get("iterations", "") if ok else "",
                    f"{bench_cost:.2f}"  if bench_cost  else "",
                    format_csv_value("risk_reliability", bench_rrisk) if bench_rrisk else "",
                    format_csv_value("time_reliability", bench_rtime) if bench_rtime else "",
                    format_csv_value("network_risk", bench_network_risk) if bench_network_risk else "",
                    format_csv_value("network_time", bench_network_time) if bench_network_time else "",
                    cost_g, rr_g, rt_g,
                    rec.get("routes", ""),
                ]
                w.writerow(row)

    print(f"\nCSV saved → {csv_path}")

    print(f"\nAll outputs saved in: {exp_dir}")
    print(f"Plot with: python sensitivity/plot_risk_time_coupled.py --csv {csv_path}")
    tee.close()
    sys.stdout = sys.__stdout__
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("RISK–TIME COUPLED SENSITIVITY ANALYSIS")
    print("α_R × α_T  →  Cost / R_risk / R_time Bubble Heatmaps")
    print("=" * 60)
    run_risk_time_coupled(time_limit=3600)
    print("Done.")
