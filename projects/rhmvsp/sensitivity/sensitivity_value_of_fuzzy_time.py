"""
Value-of-analysis: Fuzzy Travel Time (§3.2).

Vary delta_T from near-deterministic (δ_T ≈ 0) to highly uncertain (δ_T > 1.0) to
demonstrate the value of modeling travel time uncertainty.

When δ_T ≈ 0, travel times become essentially deterministic, degrading the model
to a conventional deterministic VSP. Higher δ_T widens the fuzzy support,
increasing time reliability pressure and making the routing decision more
valuable.

Output:
  - Dual Y-axis chart: R_risk (left) and R_time (right) vs delta_T
  - CSV with delta_T, status, cost, R_risk, R_time, Gap_pct
"""
import sys, os, json, csv, math
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from src.core.network import MultimodalNetwork
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.core.uncertainty import UncertaintyEngine as UE2, TrapezoidalFuzzy as TrapFz
from sensitivity.plot_helpers import dual_y_axis_with_cost_bar


def _update_od_deadlines(instance, config):
    """Recompute L_od for each OD pair based on current alpha_T and delta_T."""
    _temp_engine = UE2(config)
    _arcs = instance["network"]["arcs"]
    _hub_ids = {tn["node_id"] for tn in instance["network"].get("transfer_nodes", [])}
    for od in instance.get("od_pairs", []):
        od["alpha_T_od"] = config.alpha_T
        od_o, od_d = od["origin"], od["destination"]
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


def run_value_of_fuzzy_time(instance_path="data/small/instance.json",
                            time_limit=None,
                            output_root="results"):
    """
    Vary delta_T from 0.01 (near-deterministic) to 1.5 (high uncertainty).

    Each run uses a deep-copied instance with updated fuzzy time parameters.
    """
    with open(instance_path) as f:
        instance = json.load(f)

    timestamps = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_root, f"value_fuzzy_time_{timestamps}")
    os.makedirs(exp_dir, exist_ok=True)

    # Single experiment log — captures ALL print() + logger output
    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # delta_T values: near 0 (deterministic) -> default (0.8) -> high (1.5)
    delta_T_values = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]

    results = []
    for dT in delta_T_values:
        cfg = RHMVSPConfig()
        cfg.delta_T = dT
        cfg.alpha_max = 0.05   # Keep default
        cfg.alpha_T = 0.8      # Keep default
        if time_limit:
            cfg.time_limit = time_limit
        cfg.solver_name = "gurobi"
        cfg.solver_backend = "gurobi"

        # Deep copy instance for each run (L_od may differ)
        inst = json.loads(json.dumps(instance))
        _update_od_deadlines(inst, cfg)

        val_str = str(dT).replace(".", "_")
        val_dir = os.path.join(exp_dir, f"delta_T={val_str}")
        os.makedirs(val_dir, exist_ok=True)

        logger = Logger(level="INFO", log_file=None)
        logger.info(f"Value-of Fuzzy Time: delta_T={dT}")
        logger.info(f"  Config: alpha_max={cfg.alpha_max}, alpha_T={cfg.alpha_T}, delta_T={dT}")

        try:
            solver = BAPSolver(cfg, inst, logger)
            sol = solver.solve()

            if sol and sol.get("status") != "infeasible":
                r = {
                    "param": dT, "status": "optimal",
                    "cost": sol.get("total_cost", 0),
                    "deploy_cost": sol.get("deploy_cost", 0),
                    "hub_cost": sol.get("hub_cost", 0),
                    "transport_cost": sol.get("transport_cost", 0),
                    "transfer_cost": sol.get("transfer_cost", 0),
                    "reposition_cost": sol.get("reposition_cost", 0),
                    "prep_cost": sol.get("prep_cost", 0),
                    "idle_cost": sol.get("idle_cost", 0),
                    "opportunity_cost": sol.get("opportunity_cost", 0),
                    "R_risk": sol.get("R_net_risk", 0),
                    "R_time": sol.get("R_net_time", 0),
                    "gap_pct": sol.get("gap_pct", 0.0),
                    "iterations": sol.get("iterations", 0),
                }
                results.append(r)
                logger.info(f"  Cost={sol['total_cost']:,.0f}, R_risk={sol['R_net_risk']:.4f}, "
                             f"R_time={sol['R_net_time']:.4f}, Gap={sol.get('gap_pct',0):.4f}%")
                with open(os.path.join(val_dir, "solution.json"), "w") as f:
                    json.dump(sol, f, indent=2, default=str)
            else:
                results.append({"param": dT, "status": "infeasible"})
                logger.warning(f"  Status: INFEASIBLE")
        except Exception as e:
            results.append({"param": dT, "status": "error"})
            logger.error(f"  ERROR: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # ---- CSV summary ----
    csv_path = os.path.join(exp_dir, f"summary_delta_T.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                     "reposition_cost","prep_cost","idle_cost","opportunity_cost"]
        hdr = (["delta_T", "alpha_T", "alpha_max", "status", "iterations",
               "cost"] + cost_cols +
              ["R_risk", "R_time",
               "cost_bench", "R_risk_bench", "R_time_bench",
               "Cost_Gap%", "R_risk_Gap%", "R_time_Gap%"])
        w.writerow(hdr)

        # Benchmark = ablation row (delta_T = 0.01, first entry)
        bench = None
        for r in results:
            if r.get("status") == "optimal":
                bench = r
                break

        for r in results:
            if r.get("status") != "optimal":
                w.writerow([
                    f"{r['param']:.2f}", "", "", r.get("status", "?"),
                    "", "", "", "", "", "", "", "", "", "",
                ])
                continue

            bc = bench["cost"] if bench else 0.0
            br_risk = bench["R_risk"] if bench else 0.0
            br_time = bench["R_time"] if bench else 0.0

            if r is bench:
                cost_g, rr_g, rt_g = "—", "—", "—"
            else:
                cost_g = f"{(r['cost'] - bc) / bc * 100:+.2f}" if bc else ""
                rr_g = f"{(r['R_risk'] - br_risk) / br_risk * 100:+.2f}" if br_risk else ""
                rt_g = f"{(r['R_time'] - br_time) / br_time * 100:+.2f}" if br_time else ""

            row = [
                f"{r['param']:.2f}",
                f"{cfg.alpha_T:.2f}",
                f"{cfg.alpha_max:.2f}",
                "optimal",
                r.get("iterations", ""),
                f"{r['cost']:.0f}",
            ]
            for col in cost_cols:
                row.append(f"{r.get(col, 0):.0f}")
            row += [
                f"{r['R_risk']:.4f}",
                f"{r['R_time']:.4f}",
                f"{bc:.0f}",
                f"{br_risk:.4f}",
                f"{br_time:.4f}",
                cost_g, rr_g, rt_g,
            ]
            w.writerow(row)
    print(f"CSV saved to: {csv_path}")

    # ---- Charts ----
    x = [r["param"] for r in results if r.get("status") == "optimal"]
    r_risk_vals = [r["R_risk"] for r in results if r.get("status") == "optimal"]
    r_time_vals = [r["R_time"] for r in results if r.get("status") == "optimal"]
    cost_vals   = [r["cost"]  for r in results if r.get("status") == "optimal"]

    if x:
        # Dual Y-axis chart: R_risk/R_time lines (left) + Cost bars (right)
        dual_y_axis_with_cost_bar(x_label=r'$\delta_T$ (Time Fluctuation Factor)',
                                 x_vals=x, r_risk_vals=r_risk_vals,
                                 r_time_vals=r_time_vals, cost_vals=cost_vals,
                                 save_path=os.path.join(exp_dir, "value_fuzzy_time_dual"))

        print(f"Charts saved to: {exp_dir}")

    tee.close()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("VALUE OF FUZZY TRAVEL TIME ANALYSIS (P0)")
    print("=" * 60)
    run_value_of_fuzzy_time()
    print("Done.")
