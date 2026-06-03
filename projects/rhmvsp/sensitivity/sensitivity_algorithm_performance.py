import sys, os, json, time
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from sensitivity.plot_helpers import set_style, COLOURS, _fmt_values
from src.core.uncertainty import UncertaintyEngine as UE2, TrapezoidalFuzzy as TrapFz

# Algorithm performance chart colours (from colour set2.txt)
COLOR_TIME = COLOURS["C1"]  # Warm Terracotta — left Y-axis (Run time)
COLOR_ITER = COLOURS["C2"]  # Celadon Green   — right Y-axis (Iterations)


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
        rails     = [a for a in _arcs if a["mode"] == 2]
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

def _plot_algorithm_performance(param_name, x_label, x_values, time_vals, iter_vals, save_path):
    """Generate a dual y-axis line chart (Run time left, Iteration right) and save as png + tiff."""
    set_style()
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Use categorical evenly-spaced indices (avoids distorted axes for non-uniform x grids)
    x_idx = list(range(len(x_values)))

    # Left y-axis: Run time (Green with circles)
    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel('Run time (s)', fontsize=14, color=COLOR_TIME)
    line1, = ax1.plot(x_idx, time_vals, color=COLOR_TIME, marker='o', markersize=6,
                       linewidth=1.8, label='Run time', zorder=5)
    
    # Right y-axis: Iteration (Orange with triangles)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Iteration', fontsize=14, color=COLOR_ITER)
    line2, = ax2.plot(x_idx, iter_vals, color=COLOR_ITER, marker='s', markersize=6,
                       linewidth=1.8, label='Iteration', zorder=4)

    # Combined legend at top center
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=12)

    # Categorical x-ticks with formatted labels
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(_fmt_values(x_values), rotation=45, ha='right')

    # Styling axes
    ax1.tick_params(axis='x', direction='in', labelsize=11)
    ax1.tick_params(axis='y', direction='in', labelsize=11, colors=COLOR_TIME)
    ax2.tick_params(axis='y', direction='in', labelsize=11, colors=COLOR_ITER)
    ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)

    # Remove top spines (no outer frame)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Set proper y-axis limits to match the style
    t_min, t_max = min(time_vals), max(time_vals)
    ax1.set_ylim(max(0, t_min - (t_max-t_min)*0.2), t_max + (t_max-t_min)*0.2)
    
    i_min, i_max = min(iter_vals), max(iter_vals)
    ax2.set_ylim(max(0, i_min - (i_max-i_min)*0.2), i_max + (i_max-i_min)*0.2)

    plt.tight_layout()

    fig.savefig(save_path + ".png", dpi=600, bbox_inches='tight')
    fig.savefig(save_path + ".tiff", dpi=600, format='tiff', bbox_inches='tight')
    plt.close(fig)


def run_2d_sensitivity():
    alpha_T_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alpha_max_vals = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.10]
    
    instance_path = "data/small/instance.json"
    with open(instance_path) as f:
        instance = json.load(f)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"sensitivity_perf_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Single experiment log — captures ALL print() output
    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    csv_path = os.path.join(exp_dir, "algorithm_performance.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                     "reposition_cost","prep_cost","idle_cost","opportunity_cost"]
        w.writerow(["alpha_T", "alpha_max", "status", "Time", "Iter", "Cost"]
                    + cost_cols
                    + ["R_risk", "R_time", "Gap_pct",
                       "Time_LP", "Iter_LP", "Cost_LP"]
                    + [f"{c}_LP" for c in cost_cols]
                    + ["R_risk_LP", "R_time_LP",
                       "Time_Gap%", "Cost_Gap%", "R_risk_Gap%", "R_time_Gap%"])
        
    results = []
        
    for a_T in alpha_T_vals:
        for a_max in alpha_max_vals:
            config = RHMVSPConfig()
            config.alpha_T = a_T
            config.alpha_max = a_max
            config.solver_name = "gurobi"
            config.solver_backend = "gurobi"
            
            _update_od_deadlines(instance, config)
            
            logger = Logger(level="ERROR", log_file=None)
            
            # --- Full BAP solve ---
            t0 = time.time()
            sol = None
            try:
                solver = BAPSolver(config, json.loads(json.dumps(instance)), logger)
                sol = solver.solve()
            except Exception as e:
                print(f"a_T={a_T:.1f}, a_max={a_max:.3f} | BAP ERROR: {e}")
            elapsed_bap = time.time() - t0

            # --- LP-only solve (benchmark) ---
            t0_lp = time.time()
            sol_lp = None
            try:
                cfg_lp = RHMVSPConfig()
                cfg_lp.alpha_T = a_T
                cfg_lp.alpha_max = a_max
                cfg_lp.lp_only = True
                cfg_lp.solver_name = "gurobi"
                cfg_lp.solver_backend = "gurobi"
                logger_lp = Logger(level="ERROR", log_file=None)
                solver_lp = BAPSolver(cfg_lp, json.loads(json.dumps(instance)), logger_lp)
                sol_lp = solver_lp.solve()
            except Exception as e:
                print(f"a_T={a_T:.1f}, a_max={a_max:.1f} | LP-only ERROR: {e}")
            elapsed_lp = time.time() - t0_lp

            # --- Determine row values ---
            bap_ok = sol and sol.get("status") != "infeasible"
            lp_ok = sol_lp and sol_lp.get("status") != "infeasible"

            row = [f"{a_T:.1f}", f"{a_max:.3f}"]
            cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                         "reposition_cost","prep_cost","idle_cost","opportunity_cost"]
            if bap_ok:
                cost_bap = sol["total_cost"]
                r_risk_bap = sol.get("R_net_risk", 0)
                r_time_bap = sol.get("R_net_risk", 0)
                gap_pct_bap = sol.get("gap_pct", 0.0)
                iter_bap = sol.get("iterations", 0)
                row += ["optimal", f"{elapsed_bap:.2f}", str(iter_bap),
                        f"{cost_bap:.0f}"]
                for col in cost_cols:
                    row.append(f"{sol.get(col, 0):.0f}")
                row += [f"{r_risk_bap:.4f}", f"{r_time_bap:.4f}", f"{gap_pct_bap:.4f}"]
            else:
                row += ["infeasible", "", "", ""]
                row += [""] * len(cost_cols)
                row += ["", "", ""]

            if lp_ok:
                cost_lp = sol_lp["total_cost"]
                r_risk_lp = sol_lp.get("R_net_risk", 0)
                r_time_lp = sol_lp.get("R_net_risk", 0)
                iter_lp = sol_lp.get("iterations", 0)
                row += [f"{elapsed_lp:.2f}", str(iter_lp),
                        f"{cost_lp:.0f}"]
                for col in cost_cols:
                    row.append(f"{sol_lp.get(col, 0):.0f}")
                row += [f"{r_risk_lp:.4f}", f"{r_time_lp:.4f}"]
            else:
                row += ["", "", ""]
                row += [""] * len(cost_cols)
                row += ["", ""]

            # --- Gap columns ---
            if bap_ok and lp_ok and elapsed_lp > 0:
                time_gap  = (elapsed_bap - elapsed_lp) / elapsed_lp * 100
                cost_gap  = (cost_bap - cost_lp) / cost_lp * 100 if cost_lp else 0.0
                rr_gap    = (r_risk_bap - r_risk_lp) / r_risk_lp * 100 if r_risk_lp else 0.0
                rt_gap    = (r_time_bap - r_time_lp) / r_time_lp * 100 if r_time_lp else 0.0
                row += [f"{time_gap:+.2f}", f"{cost_gap:+.2f}",
                        f"{rr_gap:+.2f}", f"{rt_gap:+.2f}"]
            else:
                row += ["", "", "", ""]

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            # --- Print summary ---
            if bap_ok:
                msg = f"a_T={a_T:.1f}, a_max={a_max:.3f} | BAP: {elapsed_bap:.2f}s, Cost={cost_bap:,.0f}"
                if lp_ok:
                    msg += f" | LP: {elapsed_lp:.2f}s, Cost={cost_lp:,.0f}"
                print(msg)
            else:
                print(f"a_T={a_T:.1f}, a_max={a_max:.3f} | INFEASIBLE")

            results.append({
                "a_T": a_T, "a_max": a_max,
                "time": elapsed_bap if bap_ok else None,
                "iter": iter_bap if bap_ok else None,
                "cost": cost_bap if bap_ok else None,
                "R_risk": r_risk_bap if bap_ok else None,
                "R_time": r_time_bap if bap_ok else None,
                "gap_pct": gap_pct_bap if bap_ok else None,
                "time_lp": elapsed_lp if lp_ok else None,
                "cost_lp": cost_lp if lp_ok else None,
            })

    # Generate the requested plots if we have results
    if results:
        # Group by alpha_T
        time_by_aT = defaultdict(list)
        iter_by_aT = defaultdict(list)
        # Group by alpha_max
        time_by_aMax = defaultdict(list)
        iter_by_aMax = defaultdict(list)
        
        for r in results:
            time_by_aT[r['a_T']].append(r['time'])
            iter_by_aT[r['a_T']].append(r['iter'])
            
            time_by_aMax[r['a_max']].append(r['time'])
            iter_by_aMax[r['a_max']].append(r['iter'])
            
        # Average by alpha_T (filter out None values from failed runs)
        aT_sorted = sorted(time_by_aT.keys())
        aT_time_avg = [np.mean([v for v in time_by_aT[k] if v is not None]) for k in aT_sorted]
        aT_iter_avg = [np.mean([v for v in iter_by_aT[k] if v is not None]) for k in aT_sorted]
        
        # Average by alpha_max (filter out None values from failed runs)
        aMax_sorted = sorted(time_by_aMax.keys())
        aMax_time_avg = [np.mean([v for v in time_by_aMax[k] if v is not None]) for k in aMax_sorted]
        aMax_iter_avg = [np.mean([v for v in iter_by_aMax[k] if v is not None]) for k in aMax_sorted]
        
        _plot_algorithm_performance(
            param_name="alpha_T", 
            x_label=r'$\alpha_{T}$ (Time Reliability Limit)', 
            x_values=aT_sorted, 
            time_vals=aT_time_avg, 
            iter_vals=aT_iter_avg, 
            save_path=os.path.join(exp_dir, "perf_alpha_T")
        )
        
        _plot_algorithm_performance(
            param_name="alpha_max", 
            x_label=r'$\alpha_{max}$ (Risk Reliability Limit)', 
            x_values=aMax_sorted, 
            time_vals=aMax_time_avg, 
            iter_vals=aMax_iter_avg, 
            save_path=os.path.join(exp_dir, "perf_alpha_max")
        )

    # ---- delta_T sweep (additional chart) ----
    delta_T_values = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    print(f"\n{'='*60}\nDelta_T sensitivity sweep (alpha_T=0.8, alpha_max=0.05)\n{'='*60}")

    dt_results = []
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([])  # blank separator
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        dt_cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                        "reposition_cost","prep_cost","idle_cost","opportunity_cost"]
        w.writerow(["delta_T", "status", "Time", "Iter", "Cost"]
                    + dt_cost_cols
                    + ["R_risk", "R_time",
                       "Time_LP", "Iter_LP", "Cost_LP"]
                    + [f"{c}_LP" for c in dt_cost_cols])

    for dT in delta_T_values:
        config_dt = RHMVSPConfig()
        config_dt.alpha_T = 0.8
        config_dt.alpha_max = 0.05
        config_dt.delta_T = dT
        config_dt.solver_name = "gurobi"
        config_dt.solver_backend = "gurobi"

        _update_od_deadlines(instance, config_dt)

        logger = Logger(level="ERROR", log_file=None)

        # BAP
        t0 = time.time()
        sol = None
        try:
            solver = BAPSolver(config_dt, json.loads(json.dumps(instance)), logger)
            sol = solver.solve()
        except Exception as e:
            print(f"delta_T={dT:.2f} | BAP ERROR: {e}")
        elapsed_bap = time.time() - t0

        # LP-only
        cfg_lp = RHMVSPConfig()
        cfg_lp.alpha_T = 0.8
        cfg_lp.alpha_max = 0.05
        cfg_lp.delta_T = dT
        cfg_lp.lp_only = True
        cfg_lp.solver_name = "gurobi"
        cfg_lp.solver_backend = "gurobi"
        logger_lp = Logger(level="ERROR", log_file=None)
        t0_lp = time.time()
        sol_lp = None
        try:
            solver_lp = BAPSolver(cfg_lp, json.loads(json.dumps(instance)), logger_lp)
            sol_lp = solver_lp.solve()
        except Exception as e:
            print(f"delta_T={dT:.2f} | LP-only ERROR: {e}")
        elapsed_lp = time.time() - t0_lp

        bap_ok = sol and sol.get("status") != "infeasible"
        lp_ok = sol_lp and sol_lp.get("status") != "infeasible"

        dt_results.append({
            "delta_T": dT,
            "time": elapsed_bap if bap_ok else None,
            "iter": sol.get("iterations", 0) if bap_ok else None,
        })

        dt_cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                        "reposition_cost","prep_cost","idle_cost","opportunity_cost"]
        row = [f"{dT:.2f}"]
        if bap_ok:
            row += ["optimal", f"{elapsed_bap:.2f}", str(sol.get("iterations", 0)),
                    f"{sol['total_cost']:.0f}"]
            for col in dt_cost_cols:
                row.append(f"{sol.get(col, 0):.0f}")
            row += [f"{sol.get('R_net_risk',0):.4f}", f"{sol.get('R_net_time',0):.4f}"]
        else:
            row += ["infeasible", "", "", ""]
            row += [""] * len(dt_cost_cols)
            row += ["", ""]
        if lp_ok:
            row += [f"{elapsed_lp:.2f}", str(sol_lp.get("iterations", 0)),
                    f"{sol_lp['total_cost']:.0f}"]
            for col in dt_cost_cols:
                row.append(f"{sol_lp.get(col, 0):.0f}")
        else:
            row += ["", "", ""]
            row += [""] * len(dt_cost_cols)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        b_str = f"{elapsed_bap:.2f}s" if bap_ok else "ERR"
        l_str = f"{elapsed_lp:.2f}s" if lp_ok else "ERR"
        print(f"  delta_T={dT:<6.2f}  BAP={b_str:>8s}  LP={l_str:>8s}")

    if dt_results:
        dt_sorted = [r["delta_T"] for r in dt_results]
        dt_time = [r["time"]   for r in dt_results]
        dt_iter = [r["iter"]   for r in dt_results]
        # Replace None with 0 for plotting
        dt_time_plot = [v if v is not None else 0 for v in dt_time]
        dt_iter_plot = [v if v is not None else 0 for v in dt_iter]
        _plot_algorithm_performance(
            param_name="delta_T",
            x_label=r'$\delta_T$ (Time Fluctuation Factor)',
            x_values=dt_sorted,
            time_vals=dt_time_plot,
            iter_vals=dt_iter_plot,
            save_path=os.path.join(exp_dir, "perf_delta_T")
        )

    print("Performance plots generated successfully!")

    tee.close()


if __name__ == "__main__":
    print("Starting 2D Sensitivity Analysis for Algorithm Performance...")
    run_2d_sensitivity()
    print("Done!")
    # Note: tee is closed inside run_2d_sensitivity()
