"""
Sensitivity: Algorithm Performance Comparison (BAP vs Branch-and-Cut vs Benders Decomposition).

Grid: (Instance × α_T × α_max) × (BAP vs Branch-and-Cut vs Benders Decomposition)

For each combination:
  - Solve with BAP (full stabilization)                 → time, iter, cost
  - Solve with Branch-and-Cut                           → time, iter, cost
  - Solve with Benders Decomposition                    → time, iter, cost

Reference: Wang et al. (2025) §5.2 — compares C&CG vs BD algorithm across
different instance sizes and parameter settings.
"""
import sys, os, json, time, csv, copy
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MPLRect
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
# NOTE: medium/large commented out — only small instances for current experiments
INSTANCES = [
    # (tag, path, size_label)
    ("small_inst1", "data_performance/small/instance_1.json",  "small"),
    ("small_inst2", "data_performance/small/instance_2.json",  "small"),
    # ("medium_inst1", "data_performance/medium/instance_1.json", "medium"),
    # ("medium_inst2", "data_performance/medium/instance_2.json", "medium"),
    # ("large_inst1", "data_performance/large/instance_1.json",  "large"),
    # ("large_inst2", "data_performance/large/instance_2.json",  "large"),
]

# Same (α_T, α_max) grid for ALL instances
ALPHA_T_VALS   = [0.5, 0.8]
ALPHA_MAX_VALS = [0.015, 0.03, 0.05]
TIME_LIMIT = 3600  # generous for Benders Decomposition


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
    sol["solver"] = "branch_and_cut"
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


def _plot_algorithm_bar_line(bap_times, bap_iters,
                              bnc_times, bnc_iters,
                              bd_times, bd_iters,
                              param_labels, inst_order,
                              save_path):
    """
    Horizontal grouped bar chart (BAP, BnC, BD)
    """
    set_style()
    n_inst = len(inst_order)
    unique_params = list(dict.fromkeys(param_labels))
    n_params = len(unique_params)

    algo_colors = {"BAP": COLOURS["C1"], "BnC": COLOURS["C4"], "BD": COLOURS["C2"]}

    fig, ax = plt.subplots(figsize=(8, max(5, n_params * 1.0)))

    bap_avgs, bnc_avgs, bd_avgs = [], [], []
    for pi in range(n_params):
        bap_slice = [bap_times[inst_idx + pi * n_inst]   for inst_idx in range(n_inst)
                     if inst_idx + pi * n_inst < len(bap_times)]
        bnc_slice = [bnc_times[inst_idx + pi * n_inst]  for inst_idx in range(n_inst)
                     if inst_idx + pi * n_inst < len(bnc_times)]
        bd_slice  = [bd_times[inst_idx + pi * n_inst]    for inst_idx in range(n_inst)
                     if inst_idx + pi * n_inst < len(bd_times)]

        bap_avgs.append(float(np.mean(bap_slice)) if bap_slice else 0.0)
        bnc_avgs.append(float(np.mean(bnc_slice)) if bnc_slice else 0.0)
        bd_avgs.append(float(np.mean(bd_slice)) if bd_slice else 0.0)

    y_idx = np.arange(n_params)
    total_width = 0.8
    bar_w = total_width / 3
    offsets = np.linspace(-total_width/2 + bar_w/2, total_width/2 - bar_w/2, 3)

    ax.barh(y_idx + offsets[0], bap_avgs, height=bar_w, color=algo_colors["BAP"], alpha=0.85, label="BAP", zorder=2)
    ax.barh(y_idx + offsets[1], bnc_avgs, height=bar_w, color=algo_colors["BnC"], alpha=0.85, label="BnC", zorder=2)
    ax.barh(y_idx + offsets[2], bd_avgs,  height=bar_w, color=algo_colors["BD"],  alpha=0.85, label="BD",  zorder=2)

    ax.set_yticks(y_idx)
    ax.set_yticklabels(unique_params, fontsize=10)
    
    ax.set_ylim(-0.6, n_params - 0.4)
    ax.invert_yaxis()

    ax.set_ylabel(r"($\alpha_T$, $\alpha_{max}$)", fontsize=12)
    ax.set_xlabel("Run Time (s)", fontsize=13)
    ax.tick_params(axis='x', direction='in', labelsize=11)
    ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(frameon=False, fontsize=10, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)

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
            "BnC_Time", "BnC_Iter", "BnC_Cost", "BnC_Status",
            "BD_Time", "BD_Iter", "BD_Cost", "BD_Status",
            "BD_EnumTime", "BD_MIPTime",
            "Gap_Time_BnC%", "Gap_Iter_BnC%",
            "Gap_Time_BD%", "Gap_Iter_BD%",
            "BAP_Routes", "BnC_Routes", "BD_Routes",
        ])

    rows = []
    x_labels = []
    bap_times, bd_times, bnc_times = [], [], []
    bap_iters, bd_iters, bnc_iters = [], [], []
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

                # ── Run Branch-and-Cut ───────────────────────────────────
                print("  Solving with Branch-and-Cut...")
                sol_bnc = _run_bap_nostab(inst, cfg, logger)
                m_bnc = _extract_metrics(sol_bnc, "branch_and_cut")
                from sensitivity.utils import print_od_routing, format_routing_summary; print_od_routing(sol_bnc, inst, f"BnC {label}")

                # ── Run Benders Decomposition (only for small_inst1) ────
                run_bd = (inst_tag == "small_inst1")
                if run_bd:
                    print("  Solving with Benders Decomposition...")
                    sol_bd = _run_compact_mip(inst, cfg, logger)
                else:
                    print("  Skipping Benders Decomposition (small_inst1 only)...")
                    sol_bd = {"status": "skipped", "total_cost": None,
                            "solve_wall_time": 0.0, "iterations": 0,
                            "enumeration_time": 0.0, "mip_solve_time": 0.0}
                m_bd = _extract_metrics(sol_bd, "compact_mip")

                # ── Compute gaps ────────────────────────────────────────
                ok_bap = m_bap["status"] == "optimal"
                ok_bnc = m_bnc["status"] == "optimal"
                ok_bd = m_bd["status"] == "optimal"

                def _gap_time(a, b):
                    if a is not None and b is not None and b > 0:
                        return (a - b) / b * 100
                    return None

                def _gap_iter(a, b):
                    if a is not None and b is not None and b > 0:
                        return (a - b) / b * 100
                    return None

                gap_t_bnc = _gap_time(m_bap["time"], m_bnc["time"])
                gap_i_bnc = _gap_iter(m_bap["iter"], m_bnc["iter"])
                gap_t_bd = _gap_time(m_bap["time"], m_bd["time"])
                gap_i_bd = _gap_iter(m_bap["iter"], m_bd["iter"])

                bap_times.append(m_bap["time"])
                bnc_times.append(m_bnc["time"])
                bd_times.append(m_bd["time"])
                bap_iters.append(m_bap["iter"])
                bnc_iters.append(m_bnc["iter"])
                bd_iters.append(m_bd["iter"])
                gap_times_ns.append(gap_t_bnc if gap_t_bnc is not None else 0.0)

                def _fmt(v, f):
                    return f"{v:{f}}" if v is not None else "N/A"

                bap_routes = format_routing_summary(sol_bap, inst)
                bnc_routes = format_routing_summary(sol_bnc, inst)
                bd_routes  = format_routing_summary(sol_bd, inst)
                row = [
                    inst_tag, inst_size, aT, a_max,
                    f"{m_bap['time']:.2f}", str(m_bap["iter"]),
                    _fmt(m_bap["cost"], ",.0f"), m_bap["status"],
                    f"{m_bnc['time']:.2f}", str(m_bnc["iter"]),
                    _fmt(m_bnc["cost"], ",.0f"), m_bnc["status"],
                    f"{m_bd['time']:.2f}", str(m_bd["iter"]),
                    _fmt(m_bd["cost"], ",.0f"), m_bd["status"],
                    f"{sol_bd.get('enumeration_time', 0):.2f}",
                    f"{sol_bd.get('mip_solve_time', 0):.2f}",
                    _fmt(gap_t_bnc, ".2f"),
                    _fmt(gap_i_bnc, ".2f"),
                    _fmt(gap_t_bd, ".2f"),
                    _fmt(gap_i_bd, ".2f"),
                    bap_routes, bnc_routes, bd_routes,
                ]
                rows.append(row)

                bap_s = f"¥{m_bap['cost']:,.0f}" if m_bap["cost"] else m_bap["status"]
                bnc_s = f"¥{m_bnc['cost']:,.0f}" if m_bnc["cost"] else m_bnc["status"]
                bd_s  = f"¥{m_bd['cost']:,.0f}" if m_bd["cost"] else m_bd["status"]
                print(f"  BAP:     Time={m_bap['time']:.1f}s  Iter={m_bap['iter']}  Cost={bap_s}")
                print(f"  BnC:     Time={m_bnc['time']:.1f}s  Iter={m_bnc['iter']}  Cost={bnc_s}")
                print(f"  BD:     Time={m_bd['time']:.1f}s  Iter={m_bd['iter']}  Cost={bd_s}")
                gt_bnc = f"{gap_t_bnc:.1f}%" if gap_t_bnc is not None else "N/A"
                gi_bnc = f"{gap_i_bnc:.1f}%" if gap_i_bnc is not None else "N/A"
                gt_bd  = f"{gap_t_bd:.1f}%" if gap_t_bd is not None else "N/A"
                gi_bd  = f"{gap_i_bd:.1f}%" if gap_i_bd is not None else "N/A"
                print(f"  Gap(BAP-BnC): Time={gt_bnc}  Iter={gi_bnc}")
                print(f"  Gap(BAP-BD):  Time={gt_bd}  Iter={gi_bd}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"CSV saved \u2192 {csv_path}")

    # ── Plot (raw data, all instances on same figure) ─────────────────
    if x_labels:
        from collections import OrderedDict
        # Detect unique instances from rows (CSV column "Instance" at index 0)
        inst_order = list(OrderedDict.fromkeys(r[0] for r in rows))
        n_inst = len(inst_order)
        # Detect unique parameter combos (alpha_T, alpha_max) at indices 2,3
        param_keys = sorted(set((r[2], r[3]) for r in rows))
        n_params = len(param_keys)
        # Build data matrix: for each param key, for each instance, collect metrics
        # Index into rows: 0=inst, 1=size, 2=aT, 3=a_max,
        #                 4=BAP_T, 5=BAP_I, 6=BAP_Cost, 7=BAP_Status,
        #                 8=BnC_T, 9=BnC_I, 10=BnC_Cost, 11=BnC_Status,
        #                 12=BD_T, 13=BD_I, 14=BD_Cost, 15=BD_Status,
        #                 16=BD_EnumT, 17=BD_MIPT,
        #                 18=Gap_T_BnC%, 19=Gap_I_BnC%,
        #                 20=Gap_T_BD%, 21=Gap_I_BD%,
        #                 22=Routes_BAP, 23=Routes_BnC, 24=Routes_BD
        lookup = {}
        for r in rows:
            lookup[(r[0], r[2], r[3])] = r

        bap_t_list, bap_i_list = [], []
        bnc_t_list, bnc_i_list = [], []
        bd_t_list, bd_i_list = [], []
        param_label_list = []

        for aT, a_max in param_keys:
            for inst in inst_order:
                r = lookup.get((inst, aT, a_max))
                if r is None:
                    continue
                param_label_list.append(f"({aT}, {a_max})")
                bap_t_list.append(float(r[4]))
                bap_i_list.append(int(r[5]))
                bnc_t_list.append(float(r[8]))
                bnc_i_list.append(int(r[9]))
                bd_t_list.append(float(r[12]) if r[12] not in ("N/A", "skipped") else 0.0)
                bd_i_list.append(int(r[13]) if r[13].isdigit() else 0)

        _plot_algorithm_bar_line(
            bap_times=bap_t_list,
            bap_iters=bap_i_list,
            bnc_times=bnc_t_list,
            bnc_iters=bnc_i_list,
            bd_times=bd_t_list,
            bd_iters=bd_i_list,
            param_labels=param_label_list,
            inst_order=inst_order,
            save_path=os.path.join(exp_dir, "algorithm_performance"),
        )
        print(f"\nPlot → {exp_dir}/algorithm_performance.png/.tiff")

    # ── Summary table (Wang-style) ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ALGORITHM PERFORMANCE: BAP vs Branch-and-Cut vs Benders Decomposition")
    print(f"Results: {exp_dir}/")
    print("=" * 90)
    hdr = (f"{'Inst':>8s} {'α_T':>5s} {'α_max':>7s} "
           f"{'BAP_T':>7s} {'BnC_T':>7s} {'BD_T':>8s} "
           f"{'G_BnC%':>7s} {'G_BD%':>7s}")
    print(hdr)
    print("-" * 70)
    for row in rows:
        print(f"{row[0]:>8s} {row[2]:>5.1f} {row[3]:>7.3f} "
              f"{row[4]:>7s} {row[8]:>7s} {row[12]:>8s} "
              f"{row[17]:>7s} {row[19]:>7s}")

    ok_all = sum(1 for r in rows if r[7] == "optimal" and r[11] == "optimal" and r[15] == "optimal")
    print(f"\nAll optimal: {ok_all}/{len(rows)}")
    # Averages
    bap_avg = np.mean([float(r[4]) for r in rows])
    bnc_avg = np.mean([float(r[8]) for r in rows])
    bd_avg = np.mean([float(r[12]) for r in rows])
    print(f"Avg Time: BAP={bap_avg:.2f}s  BnC={bnc_avg:.2f}s  BD={bd_avg:.2f}s")

    tee.close()
    sys.stdout = sys.__stdout__
    return exp_dir


if __name__ == "__main__":
    run_algorithm_performance()
