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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.core.scenario_sampler import sample_scenario
from sensitivity.plot_helpers import set_style, COLOURS, enhanced_boxplot
from sensitivity.utils import update_od_deadlines as _update_od_deadlines

# ── Colour constants ────────────────────────────────────────────────────────
C_MR  = COLOURS["C1"]  # Warm Terracotta — MR (Model with Reliability)
C_MW = COLOURS["C2"]  # Celadon Green  — MW (Model Without reliability)

# ── Experiment parameters ───────────────────────────────────────────────────
N_SCENARIOS = 100          # scenarios per group (Wang uses 200)
INSTANCE_PATH = "data_real_case/instance.json"

# Parameter combinations (α_T, α_max)
PARAM_GRID = [
    (0.5, 0.015), (0.5, 0.03), (0.5, 0.05),
    (0.8, 0.015), (0.8, 0.03), (0.8, 0.05),
]


def _run_single(config, inst, model_tag):
    """Run one BAP solve and return dict with metrics."""
    logger = Logger(level="INFO", log_file=None)
    solver = BAPSolver(config, json.loads(json.dumps(inst)), logger)

    import time
    t0 = time.time()
    sol = solver.solve()
    elapsed = time.time() - t0

    status = sol.get("status", "unknown")
    ok = status not in ("infeasible", "error", None)
    return {
        "cost": sol.get("total_cost") if ok else None,
        "R_risk": sol.get("R_net_risk") if ok else None,
        "R_time": sol.get("R_net_time") if ok else None,
        "raw_risk": sol.get("raw_net_risk", 0.0) if ok else None,
        "raw_time": sol.get("raw_net_time", 0.0) if ok else None,
        "status": status,
        "elapsed": round(elapsed, 2),
        "model": model_tag,
    }


def run_out_of_sample():
    """
    Out-of-sample robustness test: MR (Model with Reliability) vs MW (Model Without).

    For each (alpha_T, alpha_max) group:
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
            "alpha_T": 0.01,
        },
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    group_labels = [f"({aT:.1f}, {aM:.3f})" for aT, aM in PARAM_GRID]
    mr_costs = defaultdict(list)
    mr_raw_risk = defaultdict(list)
    mr_raw_time = defaultdict(list)
    mr_R_risk = defaultdict(list)
    mr_R_time = defaultdict(list)

    mw_costs = defaultdict(list)
    mw_raw_risk = defaultdict(list)
    mw_raw_time = defaultdict(list)
    mw_R_risk = defaultdict(list)
    mw_R_time = defaultdict(list)

    total = len(PARAM_GRID) * N_SCENARIOS * 2
    done = 0

    # CSV header
    csv_path = os.path.join(exp_dir, "experiment.csv")
    cost_cols = ["deploy_cost", "hub_cost", "transport_cost", "transfer_cost",
                 "reposition_cost", "total_prep_cost", "idle_cost", "unmet_penalty"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["group_label", "alpha_T", "alpha_max", "scenario_id",
             "model", "cost"] + cost_cols +
            ["R_risk", "R_time", "Raw_Risk", "Raw_Time", "status", "elapsed_s"]
        )

    for aT, a_max in PARAM_GRID:
        label = f"({aT:.1f}, {a_max:.3f})"
        print(f"\n=== α_T={aT:.1f},  α_max={a_max:.3f} ===")

        for s_id in range(N_SCENARIOS):
            print(f"  [Scenario {s_id+1}/{N_SCENARIOS}] ...", end=" ", flush=True)
            seed = 1000 * int(aT * 100) + int(a_max * 1000) + s_id
            rng = np.random.default_rng(seed=seed)

            # ── One shared perturbed scenario (paired design) ────────────────
            scen = sample_scenario(base_instance, rng)

            # ── MR (Model with Reliability) ─────────────────────────────────
            cfg_mr = RHMVSPConfig()
            cfg_mr.alpha_T = aT
            cfg_mr.alpha_max = a_max
            cfg_mr.time_limit = 120
            cfg_mr.solver_name = "gurobi"

            scen_mr = json.loads(json.dumps(scen))
            _update_od_deadlines(scen_mr, cfg_mr)
            r_mr = _run_single(cfg_mr, scen_mr, "mr")
            done += 1

            with open(csv_path, "a", newline="") as f:
                row = [label, aT, a_max, s_id, "MR", r_mr["cost"]]
                for col in cost_cols:
                    row.append(r_mr.get(col) if r_mr["cost"] is not None else None)
                row += [r_mr.get("R_risk"), r_mr.get("R_time"),
                        r_mr.get("raw_risk"), r_mr.get("raw_time"),
                        r_mr["status"], r_mr["elapsed"]]
                csv.writer(f).writerow(row)

            # ── MW (Model Without reliability) ──────────────────────────────
            cfg_mw = RHMVSPConfig()
            cfg_mw.alpha_T = 0.01        # weak threshold, reasonable mid-point estimate
            cfg_mw.alpha_max = a_max     # keep for reference
            cfg_mw.skip_risk_constraint = True  # disable risk reliability
            cfg_mw.time_limit = 120
            cfg_mw.solver_name = "gurobi"

            scen_mw = json.loads(json.dumps(scen))
            _update_od_deadlines(scen_mw, cfg_mw)
            r_mw = _run_single(cfg_mw, scen_mw, "mw")
            done += 1

            # Store aligned — NaN for failed runs
            mr_costs[label].append(r_mr["cost"] if r_mr["cost"] is not None else float("nan"))
            mr_raw_risk[label].append(r_mr.get("raw_risk") if r_mr.get("raw_risk") is not None else float("nan"))
            mr_raw_time[label].append(r_mr.get("raw_time") if r_mr.get("raw_time") is not None else float("nan"))
            mr_R_risk[label].append(r_mr.get("R_risk") if r_mr.get("R_risk") is not None else float("nan"))
            mr_R_time[label].append(r_mr.get("R_time") if r_mr.get("R_time") is not None else float("nan"))

            mw_costs[label].append(r_mw["cost"] if r_mw["cost"] is not None else float("nan"))
            mw_raw_risk[label].append(r_mw.get("raw_risk") if r_mw.get("raw_risk") is not None else float("nan"))
            mw_raw_time[label].append(r_mw.get("raw_time") if r_mw.get("raw_time") is not None else float("nan"))
            mw_R_risk[label].append(r_mw.get("R_risk") if r_mw.get("R_risk") is not None else float("nan"))
            mw_R_time[label].append(r_mw.get("R_time") if r_mw.get("R_time") is not None else float("nan"))

            with open(csv_path, "a", newline="") as f:
                row = [label, aT, a_max, s_id, "MW", r_mw["cost"]]
                for col in cost_cols:
                    row.append(r_mw.get(col) if r_mw["cost"] is not None else None)
                row += [r_mw.get("R_risk"), r_mw.get("R_time"),
                        r_mw.get("raw_risk"), r_mw.get("raw_time"),
                        r_mw["status"], r_mw["elapsed"]]
                csv.writer(f).writerow(row)

            # Progress
            def _f(v, fmt, default="N/A"):
                return f"{v:{fmt}}" if v is not None else default
            mr_str = f"CNY {r_mr['cost']:,.0f}" if r_mr["cost"] is not None else r_mr["status"]
            mw_str = f"CNY {r_mw['cost']:,.0f}" if r_mw["cost"] is not None else r_mw["status"]
            print(f"  [{done:3d}/{total}]  MR={mr_str:>14s}  MW={mw_str:>14s}  "
                  f"({r_mr['elapsed']:.1f}s / {r_mw['elapsed']:.1f}s)")

    # ── Generate plots: primary = Total Cost violin+boxplot ─────────────────
    from scipy.stats import wilcoxon as _wilcoxon

    # Primary: Total Cost (Wang §5.6 Fig. 8 equivalent)
    mr_cost_list = [mr_costs[l] for l in group_labels]
    mw_cost_list = [mw_costs[l] for l in group_labels]

    enhanced_boxplot(
        group_labels=group_labels,
        bap_costs=mr_cost_list,
        bench_costs=mw_cost_list,
        save_path=os.path.join(exp_dir, "oos_total_cost"),
        y_label="Total Cost (CNY)",
        x_label=r"$(\alpha_T,\;\alpha_{max})$",
        model_bap_label="MR",
        model_bench_label="MW",
        c_bap=C_MR,
        c_bench=C_MW,
    )
    print(f"  Saved oos_total_cost.png/.tiff")

    # Supplementary: 4 reliability metrics
    supp_metrics = [
        ("raw_risk",  mr_raw_risk, mw_raw_risk,
         r"Raw Risk", "oos_raw_risk"),
        ("raw_time",  mr_raw_time, mw_raw_time,
         r"Raw Time (h)", "oos_raw_time"),
        ("R_risk",    mr_R_risk,   mw_R_risk,
         r"Risk Reliability ($R_{risk}$)", "oos_R_risk"),
        ("R_time",    mr_R_time,   mw_R_time,
         r"Time Reliability ($R_{time}$)", "oos_R_time"),
    ]

    for metric_name, mr_data, mw_data, y_label, filename in supp_metrics:
        mr_list = [mr_data[l] for l in group_labels]
        mw_list = [mw_data[l] for l in group_labels]

        enhanced_boxplot(
            group_labels=group_labels,
            bap_costs=mr_list,
            bench_costs=mw_list,
            save_path=os.path.join(exp_dir, filename),
            y_label=y_label,
            x_label=r"$(\alpha_T,\;\alpha_{max})$",
            model_bap_label="MR",
            model_bench_label="MW",
            c_bap=C_MR,
            c_bench=C_MW,
        )
        print(f"  Saved {filename}.png/.tiff")

    # ── Summary statistics ──────────────────────────────────────────────────
    print("\n=== Summary ===")
    all_metrics = supp_metrics + [("total_cost", mr_costs, mw_costs,
                                    "Total Cost (CNY)", "")]
    for metric_name, mr_data, mw_data, y_label, _ in all_metrics:
        print(f"\n--- {y_label} ---")
        is_cost = metric_name == "total_cost"
        fmt = ",.0f" if is_cost else ".4f"
        for lbl in group_labels:
            m = np.array(mr_data[lbl], dtype=float)
            w = np.array(mw_data[lbl], dtype=float)
            valid = ~(np.isnan(m) | np.isnan(w))
            mp_arr, wp_arr = m[valid], w[valid]
            m_med = np.nanmedian(m)
            w_med = np.nanmedian(w)
            cnt = len(mp_arr)
            if cnt >= 3 and not np.all(mp_arr == wp_arr):
                _stat, _p = _wilcoxon(mp_arr, wp_arr, alternative='two-sided')
                if   _p < 0.001: sig = "***"
                elif _p < 0.01:  sig = "**"
                elif _p < 0.05:  sig = "*"
                else:            sig = "ns"
                print(f"  {lbl:>10s}  MR={m_med:>14{fmt}}  MW={w_med:>14{fmt}}  "
                      f"n={cnt}  p={_p:.4f} {sig}")
            elif cnt > 0:
                print(f"  {lbl:>10s}  MR={m_med:>14{fmt}}  MW={w_med:>14{fmt}}  (n={cnt})")
            else:
                print(f"  {lbl:>10s}  No data")

    print(f"\nResults saved to {exp_dir}/")
    tee.close()
    sys.stdout = sys.__stdout__
    return exp_dir


if __name__ == "__main__":
    print("Out-of-Sample Test: MR (Model with Reliability) vs MW (Model Without)")
    print(f"  N={N_SCENARIOS} scenarios × {len(PARAM_GRID)} groups")
    print(f"  Instance: {INSTANCE_PATH}")
    print("=" * 65)
    exp_dir = run_out_of_sample()
    print(f"\nDone!  Open {exp_dir}/ for violin+boxplots.")
