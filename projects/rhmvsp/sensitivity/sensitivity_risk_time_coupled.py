"""
Sensitivity Analysis: α_max × δ_T Coupled Experiment (P0).

Systematically vary both risk reliability limit (α_max) and time fluctuation
factor (δ_T) jointly to explore the coupled trade-off surface between:
  - Total logistics cost
  - Network risk reliability  (R_risk)
  - Network time reliability  (R_time)

α_max controls the risk feasibility threshold (constraint).
δ_T controls the fuzzy travel-time spread (distributional input), affecting
R_time directly and R_risk indirectly via the time-dependent φ(t) multiplier.

Produce:
  1. A CSV table  summarising every grid point (Cost, R_risk, R_time, Gap_pct)
  2. Three bubble heatmaps: Cost, R_risk, R_time over the (α_max, δ_T) grid
"""
import sys, os, json, csv
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.core.uncertainty import UncertaintyEngine as UE2, TrapezoidalFuzzy as TrapFz
from sensitivity.plot_helpers import heatmap, discrete_cmap_5, set_style
from sensitivity.utils import update_od_deadlines as _update_od_deadlines

# ---- Experiment grids ----
DELTA_T_VALS     = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
ALPHA_MAX_VALS   = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.10]


def _plot_heatmap(delta_t_vals, alpha_max_vals, matrix, metric_name,
                  fmt_func=None, save_path=None):
    """Render a 2-D bubble scatter chart (uniform-size circles, colour only) using set2 colourmap."""
    fmt = fmt_func if fmt_func else lambda v: f'{v:.2f}'

    xlabels = [f'{v:.3f}' for v in alpha_max_vals]
    ylabels = [f'{v:.2f}' for v in delta_t_vals]

    set_style()
    cmap = discrete_cmap_5()
    fig, ax = plt.subplots(figsize=(max(6, len(xlabels) * 0.9),
                                    max(4, len(ylabels) * 0.6)))

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    if vmin == vmax:
        vmax = vmin + 1e-9
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    bubble_size = 300  # uniform for all cells (2.5× previous)

    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.scatter(j, i, s=bubble_size, c=[cmap(norm(val))],
                           edgecolors='none', zorder=2)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel(r'$\alpha_{max}$ (Risk Reliability Limit)', fontsize=13)
    ax.set_ylabel(r'$\delta_T$ (Time Fluctuation Factor)', fontsize=13)
    ax.set_xlim(-0.5, len(xlabels) - 0.5)
    ax.set_ylim(-0.5, len(ylabels) - 0.5)
    ax.tick_params(direction='in')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(metric_name, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    if save_path:
        for ext in ["png", "tiff"]:
            fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches='tight', format=ext)
    plt.close(fig)
    print(f"  Heatmap [{metric_name}] saved → {save_path}.png / .tiff")


# ======================================================================
#                       MAIN EXPERIMENT
# ======================================================================
def run_risk_time_coupled(instance_path="data_real_case/instance.json",
                          time_limit=None,
                          output_root="results"):
    """Run the α_max × δ_T coupled experiment."""
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
    _update_od_deadlines(inst_bench, cfg_bench)

    bench_sub = os.path.join(exp_dir, "benchmark_constant")
    os.makedirs(bench_sub, exist_ok=True)
    bench_logger = Logger(level="INFO", log_file=None)
    bench_logger.info("Benchmark: constant model (no time-varying risk)")

    bench_cost = bench_rrisk = bench_rtime = None
    bench_raw_risk = bench_raw_time = None
    try:
        solver_bench = BAPSolver(cfg_bench, inst_bench, bench_logger)
        sol_bench = solver_bench.solve()
        if sol_bench and sol_bench.get("status") != "infeasible":
            bench_cost  = sol_bench.get("total_cost", 0)
            bench_rrisk = sol_bench.get("R_net_risk", 0)
            bench_rtime = sol_bench.get("R_net_time", 0)
            bench_raw_risk = sol_bench.get("raw_net_risk", 0.0)
            bench_raw_time = sol_bench.get("raw_net_time", 0.0)
            bench_logger.info(f"  Benchmark Cost={bench_cost:,.0f}  R_risk={bench_rrisk:.4f}"
                              f"  R_time={bench_rtime:.4f}  RawRisk={bench_raw_risk:.2f}"
                              f"  RawTime={bench_raw_time:.2f}h")
            with open(os.path.join(bench_sub, "solution.json"), "w") as sf:
                json.dump(sol_bench, sf, indent=2, default=str)
        else:
            bench_logger.warning("  Benchmark infeasible — gap columns will be blank")
    except Exception as e:
        bench_logger.error(f"  Benchmark error: {e}")
        import traceback
        bench_logger.error(traceback.format_exc())

    print(f"Benchmark: Cost={bench_cost}  R_risk={bench_rrisk}  R_time={bench_rtime}"
          f"  RawRisk={bench_raw_risk if bench_raw_risk is not None else 'N/A'}"
          f"  RawTime={bench_raw_time if bench_raw_time is not None else 'N/A'}")

    csv_path = os.path.join(exp_dir, "summary_coupled.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                     "reposition_cost","total_prep_cost","idle_cost","unmet_penalty"]
        w.writerow(["alpha_max", "delta_T", "status",
                     "cost"] + cost_cols +
                    ["R_risk", "R_time", "Raw_Risk", "Raw_Time", "Gap_pct", "iterations",
                     "cost_bench", "R_risk_bench", "R_time_bench",
                     "Raw_Risk_bench", "Raw_Time_bench",
                     "Cost_Gap%", "R_risk_Gap%", "R_time_Gap%"])

    results = []
    n = len(ALPHA_MAX_VALS) * len(DELTA_T_VALS)
    idx = 0
    for a_max in ALPHA_MAX_VALS:
        for dT in DELTA_T_VALS:
            idx += 1
            print(f"[{idx}/{n}]  α_max={a_max:.3f}  δ_T={dT:.2f} ... ", end="", flush=True)

            cfg = RHMVSPConfig()
            cfg.alpha_T = 0.80        # fixed — not a sweep variable
            cfg.alpha_max = a_max
            cfg.delta_T = dT
            if time_limit:
                cfg.time_limit = time_limit
            cfg.solver_name = "gurobi"

            inst = json.loads(json.dumps(instance))
            _update_od_deadlines(inst, cfg)

            sub = os.path.join(exp_dir, f"aMax={a_max:.3f}_dT={dT:.2f}")
            os.makedirs(sub, exist_ok=True)
            logger = Logger(level="INFO", log_file=None)

            rec: dict = {"dT": dT, "a_max": a_max}

            try:
                solver = BAPSolver(cfg, inst, logger)
                sol = solver.solve()

                if sol and sol.get("status") != "infeasible":
                    rec.update({
                        "status": sol.get("status", "optimal"),
                        "cost": sol.get("total_cost", 0),
                        "deploy_cost": sol.get("deploy_cost", 0),
                        "hub_cost": sol.get("hub_cost", 0),
                        "transport_cost": sol.get("transport_cost", 0),
                        "transfer_cost": sol.get("transfer_cost", 0),
                        "reposition_cost": sol.get("reposition_cost", 0),
                        "total_prep_cost": sol.get("total_prep_cost", 0),
                        "idle_cost": sol.get("idle_cost", 0),
                        "unmet_penalty": sol.get("unmet_penalty", 0),
                        "R_risk": sol.get("R_net_risk", 0),
                        "R_time": sol.get("R_net_time", 0),
                        "raw_risk": sol.get("raw_net_risk", 0.0),
                        "raw_time": sol.get("raw_net_time", 0.0),
                        "gap_pct": sol.get("gap_pct", 0.0),
                        "iterations": sol.get("iterations", 0),
                    })
                    with open(os.path.join(sub, "solution.json"), "w") as sf:
                        json.dump(sol, sf, indent=2, default=str)
                    print(f"Cost={rec['cost']:,.0f}  R_risk={rec['R_risk']:.4f}"
                          f"  R_time={rec['R_time']:.4f}  RawRisk={rec['raw_risk']:.2f}"
                          f"  RawTime={rec['raw_time']:.2f}h  Gap={rec['gap_pct']:.4f}%")  # noqa: E501
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
                rr = rec.get("R_risk", 0) if ok else 0.0
                rt = rec.get("R_time", 0) if ok else 0.0

                if ok and bench_cost and bench_cost > 0:
                    cost_g  = f"{(rc - bench_cost)  / bench_cost  * 100:+.2f}"
                    rr_g    = f"{(rr - bench_rrisk) / bench_rrisk * 100:+.2f}" if bench_rrisk else ""
                    rt_g    = f"{(rt - bench_rtime) / bench_rtime * 100:+.2f}" if bench_rtime else ""
                else:
                    cost_g = rr_g = rt_g = ""

                row = [f"{a_max:.3f}", f"{dT:.2f}",
                    rec.get("status", "?"),
                    f"{rc:.0f}"  if ok else ""]
                for col in cost_cols:
                    row.append(f"{rec.get(col, 0):.0f}" if ok else "")
                raw_r = rec.get("raw_risk", 0) if ok else 0.0
                raw_t = rec.get("raw_time", 0) if ok else 0.0
                row += [
                    f"{rr:.4f}"  if ok else "",
                    f"{rt:.4f}"  if ok else "",
                    f"{raw_r:.4f}"  if ok else "",
                    f"{raw_t:.4f}"  if ok else "",
                    f"{rec.get('gap_pct', 0):.4f}" if ok else "",
                    rec.get("iterations", "") if ok else "",
                    f"{bench_cost:.0f}"  if bench_cost  else "",
                    f"{bench_rrisk:.4f}" if bench_rrisk else "",
                    f"{bench_rtime:.4f}" if bench_rtime else "",
                    f"{bench_raw_risk:.4f}" if bench_raw_risk else "",
                    f"{bench_raw_time:.4f}" if bench_raw_time else "",
                    cost_g, rr_g, rt_g,
                ]
                w.writerow(row)

    print(f"\nCSV saved → {csv_path}")

    # ========== Build matrices & heatmaps ==========
    cost_matrix  = np.full((len(DELTA_T_VALS), len(ALPHA_MAX_VALS)), np.nan)
    r_risk_matrix = np.full((len(DELTA_T_VALS), len(ALPHA_MAX_VALS)), np.nan)
    r_time_matrix = np.full((len(DELTA_T_VALS), len(ALPHA_MAX_VALS)), np.nan)
    raw_risk_matrix = np.full((len(DELTA_T_VALS), len(ALPHA_MAX_VALS)), np.nan)
    raw_time_matrix = np.full((len(DELTA_T_VALS), len(ALPHA_MAX_VALS)), np.nan)

    for r in results:
        if r.get("status") not in ("infeasible", "error"):
            i = DELTA_T_VALS.index(r["dT"])
            j = ALPHA_MAX_VALS.index(r["a_max"])
            cost_matrix[i, j] = r.get("cost", 0)
            r_risk_matrix[i, j] = r.get("R_risk", 0)
            r_time_matrix[i, j] = r.get("R_time", 0)
            raw_risk_matrix[i, j] = r.get("raw_risk", 0)
            raw_time_matrix[i, j] = r.get("raw_time", 0)

    _plot_heatmap(DELTA_T_VALS, ALPHA_MAX_VALS, cost_matrix,
                  "Total Cost (CNY)",
                  fmt_func=lambda v: f'{v/1000:.0f}k' if v >= 1000 else f'{v:.0f}',
                  save_path=os.path.join(exp_dir, "heatmap_cost"))

    _plot_heatmap(DELTA_T_VALS, ALPHA_MAX_VALS, r_risk_matrix,
                  r"Network Risk Reliability ($R_{risk}$)",
                  fmt_func=lambda v: f'{v:.3f}',
                  save_path=os.path.join(exp_dir, "heatmap_R_risk"))

    _plot_heatmap(DELTA_T_VALS, ALPHA_MAX_VALS, r_time_matrix,
                  r"Network Time Reliability ($R_{time}$)",
                  fmt_func=lambda v: f'{v:.3f}',
                  save_path=os.path.join(exp_dir, "heatmap_R_time"))

    _plot_heatmap(DELTA_T_VALS, ALPHA_MAX_VALS, raw_risk_matrix,
                  "Raw Network Risk",
                  fmt_func=lambda v: f'{v:.1f}',
                  save_path=os.path.join(exp_dir, "heatmap_raw_risk"))

    _plot_heatmap(DELTA_T_VALS, ALPHA_MAX_VALS, raw_time_matrix,
                  "Raw Network Time (h)",
                  fmt_func=lambda v: f'{v:.1f}',
                  save_path=os.path.join(exp_dir, "heatmap_raw_time"))

    print(f"\nAll outputs saved in: {exp_dir}")
    tee.close()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("RISK–TIME COUPLED SENSITIVITY ANALYSIS (P0)")
    print("α_max × δ_T  →  Cost / R_risk / R_time Bubble Heatmaps")
    print("=" * 60)
    run_risk_time_coupled(time_limit=120)
    print("Done.")
