"""
Sensitivity: Algorithm Performance Comparison (BAP vs BAP-NoStab vs Compact MIP).

Grid: (Instance × α_T × α_max) × (BAP vs BAP-NoStab vs Compact MIP)

For each combination:
  - Solve with BAP (full stabilization)                 → time, iter, cost
  - Solve with BAP-NoStab (no dual stabilization)       → time, iter, cost
  - Solve with Compact MIP (enumeration + direct MIP)   → time, iter, cost

Reference: Wang et al. (2025) §5.2 — compares C&CG vs BD algorithm across
different instance sizes and parameter settings.
"""
import sys, os, json, time, csv, copy
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MPLRect
from matplotlib.lines import Line2D as MPLLine
from matplotlib.ticker import MaxNLocator
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from sensitivity.plot_helpers import set_style, COLOURS
from sensitivity.utils import update_od_deadlines as _update_od_deadlines
from sensitivity.compact_mip_benchmark import solve_compact_mip

# ── Colour constants (used in _plot_dual_algorithm) ─────────────────────────

# ── Experiment grid ─────────────────────────────────────────────────────────
# 3 levels × 2 instances per level = 6 instances total.
# Each level has 2 different node-count instances.
# All instances share the same (α_T, α_max) grid.
INSTANCES = [
    # (tag, path, size_label)
    ("small_inst1", "data_performance/small/instance_1.json",  "small"),
    ("small_inst2", "data_performance/small/instance_2.json",  "small"),
    ("medium_inst1", "data_performance/medium/instance_1.json", "medium"),
    ("medium_inst2", "data_performance/medium/instance_2.json", "medium"),
    ("large_inst1", "data_performance/large/instance_1.json",  "large"),
    ("large_inst2", "data_performance/large/instance_2.json",  "large"),
]

# Same (α_T, α_max) grid for ALL instances
ALPHA_T_VALS   = [0.5, 0.8]
ALPHA_MAX_VALS = [0.015, 0.03, 0.05]
TIME_LIMIT = 3600  # generous for compact MIP


def _run_bap(instance, config, logger) -> dict:
    """Run BAP (with stabilization) and return solution dict."""
    t0 = time.time()
    solver = BAPSolver(config, instance, logger)
    sol = solver.solve()
    elapsed = time.time() - t0
    sol["solve_wall_time"] = elapsed
    sol["solve_cpu_time"] = elapsed
    sol["solver"] = "bap"
    return sol


def _run_bap_nostab(instance, config, logger) -> dict:
    """Run BAP without dual stabilization (standard BAP) and return solution dict."""
    cfg = copy.deepcopy(config)
    cfg.bap_stabilization = False  # disable stabilization → standard BAP
    t0 = time.time()
    solver = BAPSolver(cfg, instance, logger)
    sol = solver.solve()
    elapsed = time.time() - t0
    sol["solve_wall_time"] = elapsed
    sol["solve_cpu_time"] = elapsed
    sol["solver"] = "bap_nostab"
    return sol


def _run_compact_mip(instance, config, logger) -> dict:
    """Run Compact MIP and return solution dict with timing."""
    return solve_compact_mip(instance, config, logger, time_limit=TIME_LIMIT)


def _extract_metrics(sol: dict, solver_tag: str) -> dict:
    """Extract standard metrics from a solution dict."""
    ok = sol.get("status") not in ("infeasible", "error", None)
    return {
        "solver": solver_tag,
        "status": sol.get("status", "unknown"),
        "time": sol.get("solve_wall_time", 0.0),
        "iter": sol.get("iterations", 0),
        "cost": sol.get("total_cost") if ok else None,
        "R_risk": sol.get("R_net_risk") if ok else None,
        "R_time": sol.get("R_net_time") if ok else None,
        "raw_risk": sol.get("raw_net_risk", 0.0) if ok else None,
        "raw_time": sol.get("raw_net_time", 0.0) if ok else None,
    }


def _plot_dual_algorithm(bap_times, nostab_times, cmip_times,
                          bap_iters, nostab_iters, cmip_iters, gap_times,
                          param_labels, save_path):
    """
    Grouped bar (time) + marker (iter) chart for 3 algorithms.

    Bars: BAP / BAP-NoStab / Compact MIP side by side per (α_T, α_max) combo
    Markers: iter values overlaid
    Legend above the figure.
    """
    set_style()
    n_params = len(param_labels)
    x_idx = np.arange(n_params)
    n_algos = 3
    total_width = 0.8
    bar_w = total_width / n_algos
    offsets = np.linspace(-total_width/2 + bar_w/2, total_width/2 - bar_w/2, n_algos)

    algo_data = [
        (np.array(bap_times).ravel(),    np.array(bap_iters).ravel(),    COLOURS["C1"], "BAP (Optimized)"),
        (np.array(nostab_times).ravel(), np.array(nostab_iters).ravel(), COLOURS["C4"], "BAP (Standard)"),
        (np.array(cmip_times).ravel(),   np.array(cmip_iters).ravel(),   COLOURS["C2"], "Compact MIP"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    for i, (t_vals, i_vals, c, lbl) in enumerate(algo_data):
        ax.bar(x_idx + offsets[i], t_vals, width=bar_w,
               color=c, alpha=0.85, label=f"{lbl} (Time)", zorder=2)
        ax2.plot(x_idx + offsets[i], i_vals,
                 'o' if i == 0 else 's' if i == 1 else '^',
                 color=c, markersize=7, linewidth=0,
                 label=f"{lbl} (Iter)", zorder=5)

    ax.set_ylabel("Run Time (s)", fontsize=13)
    ax.tick_params(axis='y', direction='in', labelsize=11)
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.4)
    ax.spines['top'].set_visible(False)

    ax2.set_ylabel("Iterations", fontsize=13)
    ax2.tick_params(axis='y', direction='in', labelsize=11)
    ax2.spines['top'].set_visible(False)
    # Force integer ticks on iteration axis, range [0, max_iter+1]
    all_iters = np.concatenate([np.array(bap_iters).ravel(),
                                np.array(nostab_iters).ravel(),
                                np.array(cmip_iters).ravel()])
    max_iter = int(np.ceil(np.nanmax(all_iters))) if len(all_iters) > 0 else 5
    ax2.set_ylim(0, max_iter + 1)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    # Legend above
    handles = []
    for i, (_, _, c, lbl) in enumerate(algo_data):
        handles.append(MPLRect((0,0), 1, 1, color=c, alpha=0.85,
                                label=f"{lbl} (Time)"))
    markers = ['o', 's', '^']
    for i, (_, _, c, lbl) in enumerate(algo_data):
        handles.append(MPLLine([0], [0], marker=markers[i], color=c,
                               linestyle='None', markersize=7,
                               label=f"{lbl} (Iter)"))
    ax.legend(handles=handles, frameon=False, fontsize=7,
              loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(param_labels, fontsize=10)
    ax.set_xlim(-0.6, n_params - 0.4)
    ax.set_xlabel(r"($\alpha_T$, $\alpha_{max}$)", fontsize=12)

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches='tight', format=ext)
    plt.close(fig)


def run_algorithm_performance():
    """Main experiment: BAP vs Compact MIP across (Scale × α_T × α_max)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"alg_perf_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # ── CSV schema (Wang Table 4 + Table 5 combined) ────────────────────────
    csv_path = os.path.join(exp_dir, "algorithm_performance.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Instance", "Instance_Size", "alpha_T", "alpha_max",
            "BAP_Time", "BAP_Iter", "BAP_Cost", "BAP_Status",
            "NoStab_Time", "NoStab_Iter", "NoStab_Cost", "NoStab_Status",
            "CMIP_Time", "CMIP_Iter", "CMIP_Cost", "CMIP_Status",
            "CMIP_EnumTime", "CMIP_MIPTime",
            "Gap_Time_NoStab%", "Gap_Iter_NoStab%",
            "Gap_Time_CMIP%", "Gap_Iter_CMIP%",
            "BAP_Routes", "NoStab_Routes",
        ])

    rows = []
    x_labels = []
    bap_times, cmip_times, nostab_times = [], [], []
    bap_iters, cmip_iters, nostab_iters = [], [], []
    gap_times_ns = []

    total = len(INSTANCES) * len(ALPHA_T_VALS) * len(ALPHA_MAX_VALS)
    done = 0

    for inst_tag, instance_path, inst_size in INSTANCES:
        if not os.path.exists(instance_path):
            print(f"Skipping {instance_path} (not found)")
            continue

        with open(instance_path) as f:
            base_instance = json.load(f)

        for aT in ALPHA_T_VALS:
            for a_max in ALPHA_MAX_VALS:
                done += 1
                label = f"{inst_tag} (α_T={aT}, α_max={a_max})"
                x_labels.append(label)
                print(f"\n[{done}/{total}] {label}")

                cfg = RHMVSPConfig()
                cfg.alpha_T = aT
                cfg.alpha_max = a_max
                cfg.time_limit = 120
                cfg.solver_name = "gurobi"

                inst = json.loads(json.dumps(base_instance))
                _update_od_deadlines(inst, cfg)
                logger = Logger(level="INFO", log_file=None)

                # ── Run BAP (with stabilization) ────────────────────────
                print("  Solving with BAP (stabilized)...")
                sol_bap = _run_bap(inst, cfg, logger)
                m_bap = _extract_metrics(sol_bap, "bap")
                from sensitivity.utils import print_od_routing, format_routing_summary; print_od_routing(sol_bap, inst, f"BAP {label}")

                # ── Run BAP-NoStab (standard BAP) ───────────────────────
                print("  Solving with BAP-NoStab...")
                sol_nostab = _run_bap_nostab(inst, cfg, logger)
                m_nostab = _extract_metrics(sol_nostab, "bap_nostab")
                from sensitivity.utils import print_od_routing, format_routing_summary; print_od_routing(sol_nostab, inst, f"NoStab {label}")

                # ── Run Compact MIP ─────────────────────────────────────
                print("  Solving with Compact MIP...")
                sol_cmip = _run_compact_mip(inst, cfg, logger)
                m_cmip = _extract_metrics(sol_cmip, "compact_mip")

                # ── Compute gaps ────────────────────────────────────────
                ok_bap = m_bap["status"] == "optimal"
                ok_nostab = m_nostab["status"] == "optimal"
                ok_cmip = m_cmip["status"] == "optimal"

                def _gap_time(a, b):
                    if a is not None and b is not None and b > 0:
                        return (a - b) / b * 100
                    return None

                def _gap_iter(a, b):
                    if a is not None and b is not None and b > 0:
                        return (a - b) / b * 100
                    return None

                gap_t_ns = _gap_time(m_bap["time"], m_nostab["time"])
                gap_i_ns = _gap_iter(m_bap["iter"], m_nostab["iter"])
                gap_t_cmip = _gap_time(m_bap["time"], m_cmip["time"])
                gap_i_cmip = _gap_iter(m_bap["iter"], m_cmip["iter"])

                bap_times.append(m_bap["time"])
                nostab_times.append(m_nostab["time"])
                cmip_times.append(m_cmip["time"])
                bap_iters.append(m_bap["iter"])
                nostab_iters.append(m_nostab["iter"])
                cmip_iters.append(m_cmip["iter"])
                gap_times_ns.append(gap_t_ns if gap_t_ns is not None else 0.0)

                def _fmt(v, f):
                    return f"{v:{f}}" if v is not None else "N/A"

                bap_routes = format_routing_summary(sol_bap, inst)
                nostab_routes = format_routing_summary(sol_nostab, inst)
                row = [
                    inst_tag, inst_size, aT, a_max,
                    f"{m_bap['time']:.2f}", str(m_bap["iter"]),
                    _fmt(m_bap["cost"], ",.0f"), m_bap["status"],
                    f"{m_nostab['time']:.2f}", str(m_nostab["iter"]),
                    _fmt(m_nostab["cost"], ",.0f"), m_nostab["status"],
                    f"{m_cmip['time']:.2f}", str(m_cmip["iter"]),
                    _fmt(m_cmip["cost"], ",.0f"), m_cmip["status"],
                    f"{sol_cmip.get('enumeration_time', 0):.2f}",
                    f"{sol_cmip.get('mip_solve_time', 0):.2f}",
                    _fmt(gap_t_ns, ".2f"),
                    _fmt(gap_i_ns, ".2f"),
                    _fmt(gap_t_cmip, ".2f"),
                    _fmt(gap_i_cmip, ".2f"),
                    bap_routes, nostab_routes,
                ]
                rows.append(row)

                bap_s = f"¥{m_bap['cost']:,.0f}" if m_bap["cost"] else m_bap["status"]
                ns_s = f"¥{m_nostab['cost']:,.0f}" if m_nostab["cost"] else m_nostab["status"]
                cmip_s = f"¥{m_cmip['cost']:,.0f}" if m_cmip["cost"] else m_cmip["status"]
                print(f"  BAP:     Time={m_bap['time']:.1f}s  Iter={m_bap['iter']}  Cost={bap_s}")
                print(f"  NoStab:  Time={m_nostab['time']:.1f}s  Iter={m_nostab['iter']}  Cost={ns_s}")
                print(f"  CMIP:    Time={m_cmip['time']:.1f}s  Iter={m_cmip['iter']}  Cost={cmip_s}")
                gt_ns = f"{gap_t_ns:.1f}%" if gap_t_ns is not None else "N/A"
                gi_ns = f"{gap_i_ns:.1f}%" if gap_i_ns is not None else "N/A"
                gt_cmip = f"{gap_t_cmip:.1f}%" if gap_t_cmip is not None else "N/A"
                gi_cmip = f"{gap_i_cmip:.1f}%" if gap_i_cmip is not None else "N/A"
                print(f"  Gap(BAP-NoStab): Time={gt_ns}  Iter={gi_ns}")
                print(f"  Gap(BAP-CMIP):   Time={gt_cmip}  Iter={gi_cmip}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"CSV saved \u2192 {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if x_labels:
        instance_tags = [tag for tag, _, _ in INSTANCES]
        param_labels = [f"({t}, {m})" for t in ALPHA_T_VALS for m in ALPHA_MAX_VALS]

        _plot_dual_algorithm(
            bap_times=bap_times,
            nostab_times=nostab_times,
            cmip_times=cmip_times,
            bap_iters=bap_iters,
            nostab_iters=nostab_iters,
            cmip_iters=cmip_iters,
            gap_times=gap_times_ns,
            param_labels=param_labels,
            save_path=os.path.join(exp_dir, "algorithm_performance"),
        )
        print(f"\nPlot → {exp_dir}/algorithm_performance.png/.tiff")

    # ── Summary table (Wang-style) ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ALGORITHM PERFORMANCE: BAP vs BAP-NoStab vs Compact MIP")
    print(f"Results: {exp_dir}/")
    print("=" * 90)
    hdr = (f"{'Inst':>8s} {'α_T':>5s} {'α_max':>7s} "
           f"{'BAP_T':>7s} {'NS_T':>7s} {'CMIP_T':>8s} "
           f"{'G_NS%':>7s} {'G_CM%':>7s}")
    print(hdr)
    print("-" * 70)
    for row in rows:
        print(f"{row[0]:>8s} {row[2]:>5.1f} {row[3]:>7.3f} "
              f"{row[4]:>7s} {row[8]:>7s} {row[12]:>8s} "
              f"{row[17]:>7s} {row[19]:>7s}")

    ok_all = sum(1 for r in rows if r[7] == "optimal" and r[11] == "optimal" and r[15] == "optimal")
    print(f"\nAll optimal: {ok_all}/{len(rows)}")
    # Averages
    bap_t = np.mean([float(r[4]) for r in rows])
    ns_t = np.mean([float(r[8]) for r in rows])
    cmip_t = np.mean([float(r[12]) for r in rows])
    print(f"Avg Time: BAP={bap_t:.2f}s  NoStab={ns_t:.2f}s  CMIP={cmip_t:.2f}s")

    tee.close()
    sys.stdout = sys.__stdout__
    return exp_dir


if __name__ == "__main__":
    run_algorithm_performance()
