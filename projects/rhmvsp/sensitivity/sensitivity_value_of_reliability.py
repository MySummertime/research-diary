"""
Sensitivity: Value of Reliability (MR vs MW).

Compares the MR model (with risk + time reliability constraints) against
the MW model (without reliability constraints), quantifying the cost of
ensuring reliability.

For each (α_T, α_max) combination:
  - MR (Model with Reliability)     → total_cost, raw_risk, raw_time
  - MW (Model Without reliability)  → total_cost, raw_risk, raw_time
  - Gap(%) for each metric

Reference: Wang et al. (2025) §5.3 — compares full robust model [RM] vs
demand-only model [RM-D], reporting Gap-N and Gap-W percentages.
"""
import sys, os, json, time, csv
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from sensitivity.plot_helpers import set_style, COLOURS, dual_y_axis_with_cost_bar
from sensitivity.utils import update_od_deadlines as _update_od_deadlines

# ── Colour constants ────────────────────────────────────────────────────────
C_FULL  = COLOURS["C1"]  # Warm Terracotta — MR (Model with Reliability)
C_NOREL = COLOURS["C2"]  # Celadon Green  — MW (Model Without reliability)

# ── Experiment parameters ───────────────────────────────────────────────────
INSTANCE_PATH = "data_real_case/instance.json"

# Parameter grid: (α_T, α_max) combinations
PARAM_GRID = [
    (0.5, 0.015), (0.5, 0.03), (0.5, 0.05),
    (0.8, 0.015), (0.8, 0.03), (0.8, 0.05),
]


def _run_single(config, inst, logger) -> dict:
    """Run BAP solve and return metrics dict + routing summary."""
    t0 = time.time()
    solver = BAPSolver(config, inst, logger)
    sol = solver.solve()
    from sensitivity.utils import format_routing_summary, print_od_routing
    route_sum = format_routing_summary(sol, inst)
    print_od_routing(sol, inst)
    elapsed = time.time() - t0

    ok = sol.get("status") not in ("infeasible", "error", None)
    return {
        "status": sol.get("status", "unknown"),
        "time": round(elapsed, 2),
        "cost": sol.get("total_cost") if ok else None,
        "R_risk": sol.get("R_net_risk") if ok else None,
        "R_time": sol.get("R_net_time") if ok else None,
        "raw_risk": sol.get("raw_net_risk", 0.0) if ok else None,
        "raw_time": sol.get("raw_net_time", 0.0) if ok else None,
        "deploy_cost": sol.get("deploy_cost") if ok else None,
        "transport_cost": sol.get("transport_cost") if ok else None,
        "transfer_cost": sol.get("transfer_cost") if ok else None,
        "routes": route_sum,
        "reposition_cost": sol.get("reposition_cost") if ok else None,
        "unmet_penalty": sol.get("unmet_penalty") if ok else None,
        "iterations": sol.get("iterations", 0),
        "vehicles_used": sol.get("num_vehicles_used", 0),
    }


def run_value_of_reliability():
    """
    Main experiment: MR (Model with Reliability) vs MW (Model Without reliability).

    MW configuration:
      - skip_risk_constraint = True  (pricing skips R >= beta_od check)
      - alpha_T = 0.01              (time windows effectively disabled)
    """
    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"value_of_reliability_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # Save metadata
    meta = {
        "instance": INSTANCE_PATH,
        "param_grid": [(aT, am) for aT, am in PARAM_GRID],
        "norel_config": {
            "skip_risk_constraint": True,
            "alpha_T": 0.01,
        },
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # CSV schema (Wang Table 6 style)
    csv_path = os.path.join(exp_dir, "value_of_reliability.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "alpha_T", "alpha_max",
            "MR_Status", "MR_Time", "MR_Cost",
            "MR_R_risk", "MR_R_time", "MR_RawRisk", "MR_RawTime",
            "MW_Status", "MW_Time", "MW_Cost",
            "MW_R_risk", "MW_R_time", "MW_RawRisk", "MW_RawTime",
            "Gap_Cost(%)", "Gap_RawRisk(%)", "Gap_RawTime(%)",
            "Gap_R_risk(%)", "Gap_R_time(%)",
            "MR_Routes", "MW_Routes",
        ])

    group_labels = []
    full_metrics = []
    norel_metrics = []

    total = len(PARAM_GRID)
    done = 0

    for aT, a_max in PARAM_GRID:
        done += 1
        label = f"({aT:.1f}, {a_max:.3f})"
        group_labels.append(label)
        print(f"\n[{done}/{total}] {label}")

        # Deep-copy instance
        inst = json.loads(json.dumps(base_instance))

        # ── Full Model ─────────────────────────────────────────────────────
        cfg_full = RHMVSPConfig()
        cfg_full.alpha_T = aT
        cfg_full.alpha_max = a_max
        cfg_full.time_limit = 3600
        cfg_full.solver_name = "gurobi"

        inst_full = json.loads(json.dumps(inst))
        _update_od_deadlines(inst_full, cfg_full)
        logger = Logger(level="INFO", log_file=None)

        print("  Full model...")
        r_full = _run_single(cfg_full, inst_full, logger)
        full_metrics.append(r_full)

        # ── No-Reliability Model ──────────────────────────────────────────
        cfg_norel = RHMVSPConfig()
        cfg_norel.alpha_T = 0.01        # weak threshold, reasonable mid-point estimate
        cfg_norel.alpha_max = a_max     # still set for reference
        cfg_norel.skip_risk_constraint = True  # disable risk reliability in pricing
        cfg_norel.time_limit = 3600
        cfg_norel.solver_name = "gurobi"

        inst_norel = json.loads(json.dumps(inst))
        _update_od_deadlines(inst_norel, cfg_norel)
        logger2 = Logger(level="INFO", log_file=None)

        print("  No-reliability model...")
        r_norel = _run_single(cfg_norel, inst_norel, logger2)
        norel_metrics.append(r_norel)

        # ── Compute gaps (MR = baseline) ──────────────────────────────────
        def _gap_mr_baseline(mr_val, mw_val):
            """Gap = (MW - MR) / |MR| * 100 — positive means MW is larger than MR."""
            if mr_val is not None and mw_val is not None and mr_val != 0:
                return (mw_val - mr_val) / abs(mr_val) * 100
            return None

        # Cost: positive → MW costs more (reliability penalty of MW)
        gap_cost = _gap_mr_baseline(r_full["cost"], r_norel["cost"])
        # Raw risk/time: positive → MW has higher raw risk/time (MR reduces them)
        gap_raw_risk = _gap_mr_baseline(r_full["raw_risk"], r_norel["raw_risk"])
        gap_raw_time = _gap_mr_baseline(r_full["raw_time"], r_norel["raw_time"])
        # Reliability: positive → MW has better reliability (unexpected, but possible)
        gap_r_risk = _gap_mr_baseline(r_full["R_risk"], r_norel["R_risk"])
        gap_r_time = _gap_mr_baseline(r_full["R_time"], r_norel["R_time"])

        def _f(v, fmt):
            return f"{v:{fmt}}" if v is not None else "N/A"

        row = [
            aT, a_max,
            r_full["status"], f"{r_full['time']:.2f}", _f(r_full["cost"], ",.0f"),
            _f(r_full["R_risk"], ".4f"), _f(r_full["R_time"], ".4f"),
            _f(r_full["raw_risk"], ".2f"), _f(r_full["raw_time"], ".2f"),
            r_norel["status"], f"{r_norel['time']:.2f}", _f(r_norel["cost"], ",.0f"),
            _f(r_norel["R_risk"], ".4f"), _f(r_norel["R_time"], ".4f"),
            _f(r_norel["raw_risk"], ".2f"), _f(r_norel["raw_time"], ".2f"),
            _f(gap_cost, ".4f"), _f(gap_raw_risk, ".4f"), _f(gap_raw_time, ".4f"),
            _f(gap_r_risk, ".4f"), _f(gap_r_time, ".4f"),
            r_full.get("routes", ""), r_norel.get("routes", ""),
        ]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Print summary
        print(f"CSV saved \u2192 {csv_path}")
        f_str = f"¥{r_full['cost']:,.0f}" if r_full['cost'] else r_full['status']
        n_str = f"¥{r_norel['cost']:,.0f}" if r_norel['cost'] else r_norel['status']
        print(f"  MR:  Cost={f_str}  R_risk={_f(r_full['R_risk'],'.4f')}  R_time={_f(r_full['R_time'],'.4f')}")
        print(f"  MW: Cost={n_str}  R_risk={_f(r_norel['R_risk'],'.4f')}  R_time={_f(r_norel['R_time'],'.4f')}")
        print(f"  Raw MR:  Risk={_f(r_full['raw_risk'],'.2f')}  Time={_f(r_full['raw_time'],'.2f')}h")
        print(f"  Raw MW: Risk={_f(r_norel['raw_risk'],'.2f')}  Time={_f(r_norel['raw_time'],'.2f')}h")
        if gap_cost is not None:
            print(f"  Gap_Cost={gap_cost:.2f}%  Gap_R_risk={gap_r_risk:.2f}%  Gap_R_time={gap_r_time:.2f}%")

    # ── Plotting ─────────────────────────────────────────────────────────────
    from sensitivity.plot_value_of_reliability import plot_from_csv
    
    print("\nGenerating plots from CSV...")
    plot_from_csv(csv_path, exp_dir)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("VALUE OF RELIABILITY: MR (Model with Reliability) vs MW (Model Without)")
    print(f"Results: {exp_dir}/")
    print("=" * 90)
    hdr = (f"{'α_T':>5s} {'α_max':>7s} "
           f"{'MR_Cost':>12s} {'MW_Cost':>12s} {'Gap_Cost%':>10s} "
           f"{'MR_RawRisk':>10s} {'MW_RawRisk':>10s} "
           f"{'MR_RawTime':>10s} {'MW_RawTime':>10s}")
    print(hdr)
    print("-" * 90)
    for i, (aT, a_max) in enumerate(PARAM_GRID):
        f = full_metrics[i]
        n = norel_metrics[i]
        def _sv(m, k, fmt):
            v = m.get(k)
            return f"{v:{fmt}}" if v is not None else "N/A"
        print(f"{aT:>5.1f} {a_max:>7.3f} "
              f"{_sv(f,'cost',',.0f'):>12s} {_sv(n,'cost',',.0f'):>12s} "
              f"{_sv(f,'raw_risk','.2f'):>10s} {_sv(n,'raw_risk','.2f'):>10s} "
              f"{_sv(f,'raw_time','.2f'):>10s} {_sv(n,'raw_time','.2f'):>10s}")

    tee.close()
    sys.stdout = sys.__stdout__
    return exp_dir


if __name__ == "__main__":
    print("Value of Reliability: MR (Model with Reliability) vs MW (Model Without)")
    print(f"  Grid: {len(PARAM_GRID)} (α_T, α_max) combinations")
    print(f"  Instance: {INSTANCE_PATH}")
    print("=" * 60)
    exp_dir = run_value_of_reliability()
    print(f"\nDone! Results in {exp_dir}/")
