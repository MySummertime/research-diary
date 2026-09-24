"""
Sensitivity: Value of Reliability (MR vs MW).

Compares the MR model (with risk + time reliability constraints) against
the MW model (without reliability constraints), quantifying the cost of
ensuring reliability.

For each (α_T, α_R) combination:
  - MR (Model with Reliability)     → total_cost, network_risk, network_time
  - MW (Model Without reliability)  → total_cost, network_risk, network_time
  - Gap(%) for each metric

Reference: Wang et al. (2025) §5.3 — compares full robust model [RM] vs
demand-only model [RM-D], reporting Gap-N and Gap-W percentages.
"""
import sys, os, json, time, csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.utils.numeric_format import format_alpha_R, format_alpha_T, format_csv_value
from sensitivity.parameter_grids import VALUE_OF_RELIABILITY_PARAM_GRID
from sensitivity.utils import update_od_deadlines as _update_od_deadlines

# ── Experiment parameters ───────────────────────────────────────────────────
INSTANCE_PATH = "data_real_case/instance.json"

# Parameter grid: (alpha_T, alpha_R) combinations.
# alpha_R is the risk reliability threshold used directly as R >= alpha_R.
PARAM_GRID = VALUE_OF_RELIABILITY_PARAM_GRID


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
        "network_risk_reliability": sol.get("network_risk_reliability") if ok else None,
        "network_time_reliability": sol.get("network_time_reliability") if ok else None,
        "network_risk": sol.get("network_risk", 0.0) if ok else None,
        "network_time": sol.get("network_time", 0.0) if ok else None,
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
      - skip_risk_constraint = True  (pricing skips R >= alpha_R check)
      - alpha_T = 0.0               (time-reliability threshold disabled)
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
            "alpha_T": 0.0,
            "alpha_R": 0.0,
        },
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # CSV schema (Wang Table 6 style)
    csv_path = os.path.join(exp_dir, "value_of_reliability.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "alpha_T", "alpha_R",
            "MR_Status", "MR_Time", "MR_Cost",
            "MR_R_Risk", "MR_R_Time", "MR_NetworkRisk", "MR_NetworkTime",
            "MW_Status", "MW_Time", "MW_Cost",
            "MW_R_Risk", "MW_R_Time", "MW_NetworkRisk", "MW_NetworkTime",
            "Gap_Cost(%)", "Gap_NetworkRisk(%)", "Gap_NetworkTime(%)",
            "Gap_R_risk(%)", "Gap_R_time(%)",
            "MR_Routes", "MW_Routes",
        ])

    group_labels = []
    full_metrics = []
    norel_metrics = []

    total = len(PARAM_GRID)
    done = 0
    alpha_T_values = [item[0] for item in PARAM_GRID]
    alpha_R_values = [item[1] for item in PARAM_GRID]

    for aT, a_R in PARAM_GRID:
        done += 1
        label = (
            f"({format_alpha_T(aT)}, "
            f"{format_alpha_R(a_R)})"
        )
        group_labels.append(label)
        print(f"\n[{done}/{total}] {label}")

        # Deep-copy instance
        inst = json.loads(json.dumps(base_instance))

        # ── Full Model ─────────────────────────────────────────────────────
        cfg_full = RHMVSPConfig()
        cfg_full.alpha_T = aT
        cfg_full.alpha_R = a_R
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
        cfg_norel.alpha_T = 0.0
        cfg_norel.alpha_R = 0.0
        cfg_norel.skip_risk_constraint = True  # disable risk reliability in pricing
        cfg_norel.time_limit = 3600
        cfg_norel.solver_name = "gurobi"

        inst_norel = json.loads(json.dumps(inst))
        _update_od_deadlines(inst_norel, cfg_norel)
        logger2 = Logger(level="INFO", log_file=None)

        print("  No-reliability model...")
        r_norel = _run_single(cfg_norel, inst_norel, logger2)
        norel_metrics.append(r_norel)

        # ── Compute gaps (MW = baseline) ──────────────────────────────────
        def _gap_mw_baseline(mr_val, mw_val):
            """Gap = (MR - MW) / |MW| * 100 for every reported metric."""
            if mr_val is not None and mw_val is not None and mw_val != 0:
                return (mr_val - mw_val) / abs(mw_val) * 100
            return None

        # Every exported gap uses MW as the denominator and MR-MW for its sign.
        gap_cost = _gap_mw_baseline(r_full["cost"], r_norel["cost"])
        gap_network_risk = _gap_mw_baseline(r_full["network_risk"], r_norel["network_risk"])
        gap_network_time = _gap_mw_baseline(r_full["network_time"], r_norel["network_time"])
        gap_r_risk = _gap_mw_baseline(r_full["network_risk_reliability"], r_norel["network_risk_reliability"])
        gap_r_time = _gap_mw_baseline(r_full["network_time_reliability"], r_norel["network_time_reliability"])

        def _f(v, fmt):
            return f"{v:{fmt}}" if v is not None else "N/A"

        row = [
            format_alpha_T(aT),
            format_alpha_R(a_R),
            r_full["status"], f"{r_full['time']:.2f}", _f(r_full["cost"], ",.2f"),
            format_csv_value("risk_reliability", r_full["network_risk_reliability"]), format_csv_value("time_reliability", r_full["network_time_reliability"]),
            format_csv_value("network_risk", r_full["network_risk"]), format_csv_value("network_time", r_full["network_time"]),
            r_norel["status"], f"{r_norel['time']:.2f}", _f(r_norel["cost"], ",.2f"),
            format_csv_value("risk_reliability", r_norel["network_risk_reliability"]), format_csv_value("time_reliability", r_norel["network_time_reliability"]),
            format_csv_value("network_risk", r_norel["network_risk"]), format_csv_value("network_time", r_norel["network_time"]),
            _f(gap_cost, ".2f"), _f(gap_network_risk, ".2f"), _f(gap_network_time, ".2f"),
            _f(gap_r_risk, ".2f"), _f(gap_r_time, ".2f"),
            r_full.get("routes", ""), r_norel.get("routes", ""),
        ]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Print summary
        print(f"CSV saved \u2192 {csv_path}")
        f_str = f"¥{r_full['cost']:,.2f}" if r_full['cost'] else r_full['status']
        n_str = f"¥{r_norel['cost']:,.2f}" if r_norel['cost'] else r_norel['status']
        print(f"  MR:  Cost={f_str}  R_risk={format_csv_value('risk_reliability', r_full['network_risk_reliability'])}  R_time={format_csv_value('time_reliability', r_full['network_time_reliability'])}")
        print(f"  MW: Cost={n_str}  R_risk={format_csv_value('risk_reliability', r_norel['network_risk_reliability'])}  R_time={format_csv_value('time_reliability', r_norel['network_time_reliability'])}")
        print(f"  Raw MR:  Risk={format_csv_value('network_risk', r_full['network_risk'])}  Time={format_csv_value('network_time', r_full['network_time'])}h")
        print(f"  Raw MW: Risk={format_csv_value('network_risk', r_norel['network_risk'])}  Time={format_csv_value('network_time', r_norel['network_time'])}h")
        if gap_cost is not None:
            print(f"  Gap_Cost={gap_cost:.2f}%  Gap_R_risk={gap_r_risk:.2f}%  Gap_R_time={gap_r_time:.2f}%")

    print(f"\nPlot with: python sensitivity/plot_value_of_reliability.py --csv {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("VALUE OF RELIABILITY: MR (Model with Reliability) vs MW (Model Without)")
    print(f"Results: {exp_dir}/")
    print("=" * 90)
    hdr = (f"{'α_T':>5s} {'α_R':>7s} "
           f"{'MR_Cost':>12s} {'MW_Cost':>12s} {'Gap_Cost%':>10s} "
           f"{'MR_NetworkRisk':>10s} {'MW_NetworkRisk':>10s} "
           f"{'MR_NetworkTime':>10s} {'MW_NetworkTime':>10s}")
    print(hdr)
    print("-" * 90)
    for i, (aT, a_R) in enumerate(PARAM_GRID):
        f = full_metrics[i]
        n = norel_metrics[i]
        def _sv(m, k):
            v = m.get(k)
            return format_csv_value(k, v) if v is not None else "N/A"
        print(f"{format_alpha_T(aT):>8s} {format_alpha_R(a_R):>10s} "
              f"{_sv(f,'cost'):>12s} {_sv(n,'cost'):>12s} "
              f"{_sv(f,'network_risk'):>10s} {_sv(n,'network_risk'):>10s} "
              f"{_sv(f,'network_time'):>10s} {_sv(n,'network_time'):>10s}")

    tee.close()
    sys.stdout = sys.__stdout__
    return exp_dir


if __name__ == "__main__":
    print("Value of Reliability: MR (Model with Reliability) vs MW (Model Without)")
    print(f"  Grid: {len(PARAM_GRID)} (α_T, α_R) combinations")
    print(f"  Instance: {INSTANCE_PATH}")
    print("=" * 60)
    exp_dir = run_value_of_reliability()
    print(f"\nDone! Results in {exp_dir}/")
