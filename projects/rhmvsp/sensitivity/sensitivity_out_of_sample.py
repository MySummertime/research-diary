"""
Sensitivity Analysis: Out-of-Sample Test (MR vs MW).

Design (solve-on-original, evaluate-on-perturbed):
  1. Solve MR (Model with Reliability) on ORIGINAL instance → S_mr
  2. Solve MW (Model Without reliability) on ORIGINAL → S_mw
  3. Generate N Monte Carlo perturbed scenarios (demand + time + risk)
  4. Re-solve both models on each perturbed scenario
  5. Compare total cost distribution via violin + boxplot

Reference: Wang et al. (2025) §5.6 — out-of-sample tests comparing [RM]
(robust model) vs [DM] (deterministic model) with 200 scenarios.
"""
import sys, os, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.utils.numeric_format import format_alpha_R, format_alpha_T, format_csv_value
from src.core.scenario_sampler import sample_scenario
from sensitivity.parameter_grids import ALPHA_R_OOS_LEVELS, ALPHA_T_OOS_LEVELS
from sensitivity.utils import update_od_deadlines as _update_od_deadlines

# ── Experiment parameters ───────────────────────────────────────────────────
N_SCENARIOS = 200          # scenarios per group (Wang uses 200)
INSTANCE_PATH = "data_real_case/instance.json"

# Parameter combinations (alpha_T, alpha_R).
# alpha_R is the risk reliability threshold used directly as R >= alpha_R.
PARAM_GRID = [
    (alpha_T, alpha_R)
    for alpha_T in ALPHA_T_OOS_LEVELS
    for alpha_R in ALPHA_R_OOS_LEVELS
]


def _run_single(config, inst, model_tag):
    """Run one BAP solve and return dict with metrics."""
    logger = Logger(level="INFO", log_file=None)
    solver = BAPSolver(config, json.loads(json.dumps(inst)), logger)

    import time
    t0 = time.time()
    sol = solver.solve()
    from sensitivity.utils import print_od_routing, format_routing_summary; print_od_routing(sol, inst, model_tag)
    elapsed = time.time() - t0

    status = sol.get("status", "unknown")
    ok = status not in ("infeasible", "error", None)
    return {
        "cost": sol.get("total_cost") if ok else None,
        "network_risk_reliability": sol.get("network_risk_reliability") if ok else None,
        "network_time_reliability": sol.get("network_time_reliability") if ok else None,
        "network_risk": sol.get("network_risk", 0.0) if ok else None,
        "network_time": sol.get("network_time", 0.0) if ok else None,
        "status": status,
        "elapsed": round(elapsed, 2),
        "model": model_tag,
        "routes": format_routing_summary(sol, inst),
    }


def run_out_of_sample():
    """
    Out-of-sample robustness test: MR (Model with Reliability) vs MW (Model Without).

    For each (alpha_T, alpha_R) group:
      1. Solve MR (with reliability constraints) on original → S_mr
      2. Solve MW (skip risk + time reliability) on original → S_mw
      3. Generate N Monte Carlo perturbed scenarios (demand + time + risk)
      4. Re-solve both models on each perturbed scenario
      5. Compare total cost distribution via violin + boxplot

    The MW model ignores risk and time reliability constraints during
    optimization. Its solutions are cheaper but may violate safety/time
    requirements under uncertainty. The MR model accounts for these
    constraints, providing more robust solutions.
    """

    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"out_of_sample_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # Save experiment metadata
    meta = {
        "n_scenarios": N_SCENARIOS,
        "param_grid": [(aT, am) for aT, am in PARAM_GRID],
        "instance": INSTANCE_PATH,
        "mw_config": {
            "skip_risk_constraint": True,
            "alpha_T": 0.0,
            "alpha_R": 0.0,
        },
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    alpha_T_values = [item[0] for item in PARAM_GRID]
    alpha_R_values = [item[1] for item in PARAM_GRID]
    group_labels = [
        f"({format_alpha_T(aT)}, {format_alpha_R(aM)})"
        for aT, aM in PARAM_GRID
    ]
    total = len(PARAM_GRID) * N_SCENARIOS * 2
    done = 0

    # CSV header
    csv_path = os.path.join(exp_dir, "experiment.csv")
    cost_cols = ["transport_cost", "transfer_cost",
                 "reposition_cost", "unmet_penalty"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["group_label", "alpha_T", "alpha_R", "scenario_id",
             "model", "cost"] + cost_cols +
            ["network_risk_reliability", "network_time_reliability", "Network_Risk", "Network_Time", "status", "elapsed_s", "Routes"]
        )

    for aT, a_R in PARAM_GRID:
        label = f"({format_alpha_T(aT)}, {format_alpha_R(a_R)})"
        print(
            f"\n=== α_T={format_alpha_T(aT)},  "
            f"α_R={format_alpha_R(a_R)} ==="
        )

        for s_id in range(N_SCENARIOS):
            print(f"  [Scenario {s_id+1}/{N_SCENARIOS}] ...", end=" ", flush=True)
            seed = 1000 * int(aT * 100) + int(a_R * 1000) + s_id
            rng = np.random.default_rng(seed=seed)

            # ── One shared perturbed scenario (paired design) ────────────────
            scen = sample_scenario(base_instance, rng)

            # ── MR (Model with Reliability) ─────────────────────────────────
            cfg_mr = RHMVSPConfig()
            cfg_mr.alpha_T = aT
            cfg_mr.alpha_R = a_R
            cfg_mr.time_limit = 120
            cfg_mr.solver_name = "gurobi"

            scen_mr = json.loads(json.dumps(scen))
            _update_od_deadlines(scen_mr, cfg_mr)
            r_mr = _run_single(cfg_mr, scen_mr, "mr")
            done += 1

            with open(csv_path, "a", newline="") as f:
                row = [label, format_csv_value("alpha_T", aT), format_csv_value("alpha_R", a_R), s_id, "MR", format_csv_value("cost", r_mr["cost"])]
                for col in cost_cols:
                    row.append(format_csv_value(col, r_mr.get(col)) if r_mr["cost"] is not None else "")
                row += [format_csv_value("network_risk_reliability", r_mr.get("network_risk_reliability")),
                        format_csv_value("network_time_reliability", r_mr.get("network_time_reliability")),
                        format_csv_value("network_risk", r_mr.get("network_risk")),
                        format_csv_value("network_time", r_mr.get("network_time")),
                        r_mr["status"], format_csv_value("elapsed_s", r_mr["elapsed"]), r_mr.get("routes", "")]
                csv.writer(f).writerow(row)

            # ── MW (Model Without reliability) ──────────────────────────────
            cfg_mw = RHMVSPConfig()
            cfg_mw.alpha_T = 0.0
            cfg_mw.alpha_R = 0.0
            cfg_mw.skip_risk_constraint = True  # disable risk reliability
            cfg_mw.time_limit = 120
            cfg_mw.solver_name = "gurobi"

            scen_mw = json.loads(json.dumps(scen))
            _update_od_deadlines(scen_mw, cfg_mw)
            r_mw = _run_single(cfg_mw, scen_mw, "mw")
            done += 1

            with open(csv_path, "a", newline="") as f:
                row = [label, format_csv_value("alpha_T", aT), format_csv_value("alpha_R", a_R), s_id, "MW", format_csv_value("cost", r_mw["cost"])]
                for col in cost_cols:
                    row.append(format_csv_value(col, r_mw.get(col)) if r_mw["cost"] is not None else "")
                row += [format_csv_value("network_risk_reliability", r_mw.get("network_risk_reliability")),
                        format_csv_value("network_time_reliability", r_mw.get("network_time_reliability")),
                        format_csv_value("network_risk", r_mw.get("network_risk")),
                        format_csv_value("network_time", r_mw.get("network_time")),
                        r_mw["status"], format_csv_value("elapsed_s", r_mw["elapsed"]), r_mw.get("routes", "")]
                csv.writer(f).writerow(row)

            # Progress
            def _f(v, fmt, default="N/A"):
                return f"{v:{fmt}}" if v is not None else default
            mr_str = f"CNY {r_mr['cost']:,.0f}" if r_mr["cost"] is not None else r_mr["status"]
            mw_str = f"CNY {r_mw['cost']:,.0f}" if r_mw["cost"] is not None else r_mw["status"]
            print(f"  [{done:3d}/{total}]  MR={mr_str:>14s}  MW={mw_str:>14s}  "
                  f"({r_mr['elapsed']:.1f}s / {r_mw['elapsed']:.1f}s)")

    print(f"CSV saved → {csv_path}")

    print(f"\nResults saved to {exp_dir}/")
    print(f"Plot with: python sensitivity/plot_out_of_sample.py --csv {csv_path}")
    tee.close()
    sys.stdout = sys.__stdout__
    return exp_dir


if __name__ == "__main__":
    print("Out-of-Sample Test: MR (Model with Reliability) vs MW (Model Without)")
    print(f"  N={N_SCENARIOS} scenarios × {len(PARAM_GRID)} groups")
    print(f"  Instance: {INSTANCE_PATH}")
    print("=" * 65)
    exp_dir = run_out_of_sample()
    print(f"\nDone! Data are in {exp_dir}/experiment.csv")
