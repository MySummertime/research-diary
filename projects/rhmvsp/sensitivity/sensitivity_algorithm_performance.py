"""
Sensitivity: Algorithm Performance Comparison (BAP vs Branch-and-Cut vs Benders Decomposition).

Grid: (Instance × α_T × α_R) × (BAP vs Branch-and-Cut vs Benders Decomposition)

For each combination:
  - Solve with BAP (full stabilization)                 → time, iter, cost
  - Solve with Branch-and-Cut                           → time, iter, cost
  - Solve with Benders Decomposition                    → time, iter, cost

Reference: Wang et al. (2025) §5.2 — compares C&CG vs BD algorithm across
different instance sizes and parameter settings.
"""
import sys, os, json, time, csv, copy
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver, BAPNode
from src.utils.numeric_format import format_alpha_R, format_alpha_T
from sensitivity.parameter_grids import ALPHA_R_ALGORITHM_LEVELS, ALPHA_T_ALGORITHM_LEVELS

# ── Colour constants (used in _plot_dual_algorithm) ─────────────────────────

# ── Experiment grid ─────────────────────────────────────────────────────────
INSTANCES = [
    # (tag, path, size_label)
    ("small_inst1", "data_performance/small/instance_1.json",  "small"),
    ("small_inst2", "data_performance/small/instance_2.json",  "small"),
    ("medium_inst1", "data_performance/medium/instance_1.json", "medium"),
    ("medium_inst2", "data_performance/medium/instance_2.json", "medium"),
    ("large_inst1", "data_performance/large/instance_1.json",  "large"),
    ("large_inst2", "data_performance/large/instance_2.json",  "large"),
]

# Reliability is controlled at the main-experiment setting so that the
# independent variables are algorithm and instance size only.
ALPHA_T_VALS = ALPHA_T_ALGORITHM_LEVELS
ALPHA_R_VALS = ALPHA_R_ALGORITHM_LEVELS
TIME_LIMIT = 3600  # generous for Benders Decomposition


def _run_bap(instance, config, logger) -> dict:
    """Run BAP (with stabilization) and return solution dict."""
    t0 = time.time()
    solver = BAPSolver(config, instance, logger)
    sol = solver.solve()
    elapsed = time.time() - t0
    sol["solve_wall_Time"] = elapsed
    sol["solve_cpu_Time"] = elapsed
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
    sol["solve_wall_Time"] = elapsed
    sol["solve_cpu_Time"] = elapsed
    sol["solver"] = "branch_and_cut"
    return sol


def _run_benders(instance, config, logger) -> dict:
    """
    Run Benders Decomposition.
    Uses BAP's own infrastructure (pricing + master) but skips the dynamic
    column generation loop — all paths are enumerated upfront, then the
    master is solved as an integer MIP.
    """
    time_limit = TIME_LIMIT
    solve_t0 = time.time()

    cfg = copy.deepcopy(config)
    cfg.time_limit = time_limit
    cfg.solver_name = "gurobi"

    # ── 1. Create BAP controller (provides network, engine, pp, mp) ────────
    bap = BAPSolver(cfg, instance, logger)

    # ── 2. Enumerate ALL feasible columns per OD pair ──────────────────────
    big_m = getattr(cfg, 'big_m', 1e6)
    high_duals = {
        "od": {i: big_m for i in range(len(bap.network.od_pairs))},
        "vehicle": {},
        "trans_out": {},
        "trans_in": {},
        "hub": {},
    }

    orig_exact = cfg.max_labels_exact
    orig_heur = cfg.max_labels_heuristic
    cfg.max_labels_exact = 50000
    cfg.max_labels_heuristic = 20000

    all_columns = []
    existing_fingerprints: set = set()

    for od_idx, od in enumerate(bap.network.od_pairs):
        logger.info(f"  Enumerating paths for OD {od_idx+1} (origin={od.origin}, dest={od.destination}) ...")
        for exact_flag in [False, True]:
            phase = "Exact" if exact_flag else "Heuristic"
            labels = bap.pp.solve(
                od.origin, od.destination, od_idx, high_duals,
                od.time_window_start, od.time_window_end, od.alpha_T_od,
                branch_constraints=[], exact=exact_flag,
            )
            cols_added = 0
            for lab in labels:
                fp = tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence)
                if fp not in existing_fingerprints:
                    existing_fingerprints.add(fp)
                    col = bap._label_to_column(lab, od_idx)
                    if col is not None:
                        all_columns.append(col)
                        cols_added += 1
            logger.info(f"    {phase}: {cols_added} new columns")

    import random as _random
    _random.seed(42)
    n_extra_passes = 3
    n_od = len(bap.network.od_pairs)

    for pass_i in range(n_extra_passes):
        perturbed_duals = {
            "od": {i: big_m * _random.uniform(0.5, 1.5) for i in range(n_od)},
            "vehicle": {}, "trans_out": {}, "trans_in": {}, "hub": {},
        }
        total_added = 0
        for od_idx, od in enumerate(bap.network.od_pairs):
            labels = bap.pp.solve(
                od.origin, od.destination, od_idx, perturbed_duals,
                od.time_window_start, od.time_window_end, od.alpha_T_od,
                branch_constraints=[], exact=False,
            )
            for lab in labels:
                fp = tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence)
                if fp not in existing_fingerprints:
                    existing_fingerprints.add(fp)
                    col = bap._label_to_column(lab, od_idx)
                    if col is not None:
                        all_columns.append(col)
                        total_added += 1
        logger.info(f"    Random pass {pass_i+1}/{n_extra_passes}: {total_added} new columns")

    cfg.max_labels_exact = orig_exact
    cfg.max_labels_heuristic = orig_heur

    enum_time = time.time() - solve_t0
    logger.info(f"  Enumerated {len(all_columns)} total columns ({enum_time:.1f}s)")

    if not all_columns:
        logger.warning("  No columns enumerated — instance infeasible")
        return {"status": "infeasible"}

    # ── 3. Solve as MIP via BAP's extraction ───────────────────────────────
    mock_node = BAPNode(
        lower_bound=0.0,
        depth=0,
        node_id=0,
        columns=all_columns,
    )

    try:
        mip_t0 = time.time()
        sol = bap._extract_solution(mock_node)
        mip_time = time.time() - mip_t0
    except Exception as e:
        logger.error(f"  MIP solve / extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}

    # ── 4. Augment with timing info ────────────────────────────────────────
    total_time = time.time() - solve_t0
    sol["solve_wall_Time"] = total_time
    sol["solve_cpu_Time"] = total_time
    sol["enumeration_Time"] = enum_time
    sol["mip_solve_Time"] = mip_time
    sol["solver"] = "benders"
    sol.setdefault("gap_pct", 0.0)
    sol.setdefault("iterations", 1)
    sol.setdefault("num_repositions", 0)

    return sol


def _extract_metrics(sol: dict, solver_tag: str) -> dict:
    """Extract standard metrics from a solution dict."""
    ok = sol.get("status") not in ("infeasible", "error", None)
    return {
        "solver": solver_tag,
        "status": sol.get("status", "unknown"),
        "time": sol.get("solve_wall_Time", 0.0),
        "iter": sol.get("iterations", 0),
        "cost": sol.get("total_cost") if ok else None,
        "network_risk_reliability": sol.get("network_risk_reliability") if ok else None,
        "network_time_reliability": sol.get("network_time_reliability") if ok else None,
        "network_risk": sol.get("network_risk", 0.0) if ok else None,
        "network_time": sol.get("network_time", 0.0) if ok else None,
    }





def run_algorithm_performance():
    """Main experiment: BAP vs Compact MIP across (Scale × α_T × α_R)."""
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
            "Instance", "Instance_Size", "alpha_T", "alpha_R",
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

    total = len(INSTANCES) * len(ALPHA_T_VALS) * len(ALPHA_R_VALS)
    done = 0

    for inst_tag, instance_path, inst_size in INSTANCES:
        if not os.path.exists(instance_path):
            print(f"Skipping {instance_path} (not found)")
            continue

        with open(instance_path) as f:
            base_instance = json.load(f)

        for aT in ALPHA_T_VALS:
            for a_R in ALPHA_R_VALS:
                done += 1
                label = f"{inst_tag} (α_T={aT}, α_R={a_R})"
                x_labels.append(label)
                print(f"\n[{done}/{total}] {label}")

                cfg = RHMVSPConfig()
                cfg.alpha_T = aT
                cfg.alpha_R = a_R
                cfg.time_limit = TIME_LIMIT
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

                # ── Run Benders Decomposition ───────────────────────────
                print("  Solving with Benders Decomposition...")
                sol_bd = _run_benders(inst, cfg, logger)
                m_bd = _extract_metrics(sol_bd, "benders")

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
                    inst_tag, inst_size, aT, a_R,
                    f"{m_bap['time']:.2f}", str(m_bap["iter"]),
                    _fmt(m_bap["cost"], ",.2f"), m_bap["status"],
                    f"{m_bnc['time']:.2f}", str(m_bnc["iter"]),
                    _fmt(m_bnc["cost"], ",.2f"), m_bnc["status"],
                    f"{m_bd['time']:.2f}", str(m_bd["iter"]),
                    _fmt(m_bd["cost"], ",.2f"), m_bd["status"],
                    f"{sol_bd.get('enumeration_time', 0):.2f}",
                    f"{sol_bd.get('mip_solve_time', 0):.2f}",
                    _fmt(gap_t_bnc, ".2f"),
                    _fmt(gap_i_bnc, ".2f"),
                    _fmt(gap_t_bd, ".2f"),
                    _fmt(gap_i_bd, ".2f"),
                    bap_routes, bnc_routes, bd_routes,
                ]
                rows.append(row)

                bap_s = f"¥{m_bap['cost']:,.2f}" if m_bap["cost"] else m_bap["status"]
                bnc_s = f"¥{m_bnc['cost']:,.2f}" if m_bnc["cost"] else m_bnc["status"]
                bd_s  = f"¥{m_bd['cost']:,.2f}" if m_bd["cost"] else m_bd["status"]
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

    print(f"\nPlot with: python sensitivity/plot_algorithm_performance.py --csv {csv_path}")

    # ── Summary table (Wang-style) ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ALGORITHM PERFORMANCE: BAP vs Branch-and-Cut vs Benders Decomposition")
    print(f"Results: {exp_dir}/")
    print("=" * 90)
    hdr = (f"{'Inst':>8s} {'α_T':>5s} {'α_R':>7s} "
           f"{'BAP_T':>7s} {'BnC_T':>7s} {'BD_T':>8s} "
           f"{'G_BnC%':>7s} {'G_BD%':>7s}")
    print(hdr)
    print("-" * 70)
    for row in rows:
        print(f"{row[0]:>8s} {format_alpha_T(row[2]):>8s} {format_alpha_R(row[3]):>10s} "
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
