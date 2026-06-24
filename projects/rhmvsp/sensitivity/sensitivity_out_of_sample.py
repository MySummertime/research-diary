"""
Sensitivity Analysis: Out-of-Sample Box-Plot (BAP vs LP Relaxation).

Design (solve-on-original, evaluate-on-perturbed):
  1. Solve BAP (Branch-and-Price, integer) on ORIGINAL instance → S_bap
  2. Solve LP Relaxation (root-node CG, no branching) on ORIGINAL → S_lp
  3. Generate N Monte Carlo perturbed scenarios (demand + time + risk)
  4. Re-solve both models on each perturbed scenario
  5. Compare 4 reliability metrics: raw_risk, raw_time, R_risk, R_time

Output: 4 enhanced box-plots comparing reliability metric distributions.
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
from src.core.uncertainty import UncertaintyEngine as UE2, TrapezoidalFuzzy as TrapFz
from src.core.scenario_sampler import sample_scenario
from sensitivity.plot_helpers import set_style, COLOURS, enhanced_boxplot
from sensitivity.utils import update_od_deadlines as _update_od_deadlines

# ── Colour constants (colour set2.txt) ────────────────────────────────────────
C_BAP = COLOURS["C1"]  # Warm Terracotta — BAP (our method)
C_LP  = COLOURS["C2"]  # Celadon Green   — LP Relaxation (benchmark)

# ── Experiment parameters ─────────────────────────────────────────────────────
N_SCENARIOS = 10          # scenarios per group (reduced for faster testing)
INSTANCE_PATH = "data_real_case/instance.json"

# 4 parameter combinations for quick test
PARAM_GRID = [
    (0.5, 0.03), (0.5, 0.05),
    (0.8, 0.03), (0.8, 0.05),
]


def _run_single(config, inst, exp_dir, model_tag, aT, a_max, seed):
    """Run one BAP/LP solve and return dict with cost and sub-components."""
    logger = Logger(level="INFO", log_file=None)
    
    if model_tag == "lp":
        config.lp_only = True  # LP relaxation: stop after root-node CG
    
    solver = BAPSolver(config, json.loads(json.dumps(inst)), logger)
    
    import time
    t0 = time.time()
    sol = solver.solve()
    elapsed = time.time() - t0

    status  = sol.get("status", "unknown")
    ok = status != "infeasible"
    cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                 "reposition_cost","total_prep_cost","idle_cost","unmet_penalty"]
    result = {"cost": sol.get("total_cost") if ok else None,
              "R_risk": sol.get("R_net_risk", 0) if ok else None,
              "R_time": sol.get("R_net_time", 0) if ok else None,
              "raw_risk": sol.get("raw_net_risk", 0.0) if ok else None,
              "raw_time": sol.get("raw_net_time", 0.0) if ok else None,
              "status": status, "elapsed": round(elapsed, 2)}
    for col in cost_cols:
        result[col] = sol.get(col, 0) if ok else None
    return result


def run_out_of_sample():
    """
    Out-of-sample robustness experiment: BAP vs LP Relaxation.

    For each (alpha_T, alpha_max) group:
      1. Solve BAP (integer) on original instance → S_bap
      2. Solve LP Relaxation (root-node CG, no branching) → S_lp
      3. Generate N Monte Carlo perturbed scenarios (demand + time + risk)
      4. Re-solve both models on each perturbed scenario
      5. Compare 4 reliability metrics: raw_risk, raw_time, R_risk, R_time

    The LP relaxation gives a lower bound on cost but relaxes integrality,
    leading to optimistic (overly good) reliability estimates that don't
    hold in practice. BAP provides integer-feasible solutions with tighter
    reliability guarantees.
    """

    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir   = os.path.join("results", f"out_of_sample_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Single experiment log — captures ALL print() output
    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # Save experiment metadata
    meta = {
        "n_scenarios": N_SCENARIOS,
        "param_grid": [(aT, am) for aT, am in PARAM_GRID],
        "instance": INSTANCE_PATH,
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    group_labels = [f"({aT:.1f}, {aM:.3f})" for aT, aM in PARAM_GRID]
    bap_costs    = defaultdict(list)
    bap_raw_risk = defaultdict(list)
    bap_raw_time = defaultdict(list)
    bap_R_risk   = defaultdict(list)
    bap_R_time   = defaultdict(list)

    lp_costs     = defaultdict(list)
    lp_raw_risk  = defaultdict(list)
    lp_raw_time  = defaultdict(list)
    lp_R_risk    = defaultdict(list)
    lp_R_time    = defaultdict(list)

    total = len(PARAM_GRID) * N_SCENARIOS * 2  # BAP + LP per scenario
    done  = 0

    # CSV header
    csv_path = os.path.join(exp_dir, "experiment.csv")
    cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                 "reposition_cost","total_prep_cost","idle_cost","unmet_penalty"]
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
            print(f"  [Scenario {s_id+1}/{N_SCENARIOS}] Generating and solving...")
            seed = 1000 * int(aT * 100) + int(a_max * 1000) + s_id
            rng = np.random.default_rng(seed=seed)

            # ── One shared perturbed scenario (paired design) ────────────────
            scen = sample_scenario(base_instance, rng)

            # ── BAP (Branch-and-Price, integer) ──────────────────────────────
            cfg_bap = RHMVSPConfig()
            cfg_bap.alpha_T = aT
            cfg_bap.alpha_max = a_max
            cfg_bap.time_limit = 120
            cfg_bap.solver_name = "gurobi"
            scen_bap = json.loads(json.dumps(scen))
            _update_od_deadlines(scen_bap, cfg_bap)

            r_bap = _run_single(cfg_bap, scen_bap, exp_dir, "bap", aT, a_max, s_id)
            done += 1

            with open(csv_path, "a", newline="") as f:
                row = [label, aT, a_max, s_id, "BAP", r_bap["cost"]]
                for col in cost_cols:
                    row.append(r_bap.get(col) if r_bap["cost"] is not None else None)
                row += [r_bap.get("R_risk"), r_bap.get("R_time"),
                        r_bap.get("raw_risk"), r_bap.get("raw_time"),
                        r_bap["status"], r_bap["elapsed"]]
                csv.writer(f).writerow(row)

            # ── LP Relaxation (benchmark, root-node CG only) ─────────────────
            cfg_lp = RHMVSPConfig()
            cfg_lp.alpha_T = aT
            cfg_lp.alpha_max = a_max
            cfg_lp.time_limit = 120
            cfg_lp.solver_name = "gurobi"
            scen_lp = json.loads(json.dumps(scen))
            _update_od_deadlines(scen_lp, cfg_lp)

            r_lp = _run_single(cfg_lp, scen_lp, exp_dir, "lp", aT, a_max, s_id)
            done += 1

            # Store aligned — NaN for failed runs keeps index correspondence
            bap_costs[label].append(r_bap["cost"] if r_bap["cost"] is not None else float("nan"))
            bap_raw_risk[label].append(r_bap.get("raw_risk", float("nan")))
            bap_raw_time[label].append(r_bap.get("raw_time", float("nan")))
            bap_R_risk[label].append(r_bap.get("R_risk", float("nan")))
            bap_R_time[label].append(r_bap.get("R_time", float("nan")))

            lp_costs[label].append(r_lp["cost"] if r_lp["cost"] is not None else float("nan"))
            lp_raw_risk[label].append(r_lp.get("raw_risk", float("nan")))
            lp_raw_time[label].append(r_lp.get("raw_time", float("nan")))
            lp_R_risk[label].append(r_lp.get("R_risk", float("nan")))
            lp_R_time[label].append(r_lp.get("R_time", float("nan")))

            with open(csv_path, "a", newline="") as f:
                row = [label, aT, a_max, s_id, "LP", r_lp["cost"]]
                for col in cost_cols:
                    row.append(r_lp.get(col) if r_lp["cost"] is not None else None)
                row += [r_lp.get("R_risk"), r_lp.get("R_time"),
                        r_lp.get("raw_risk"), r_lp.get("raw_time"),
                        r_lp["status"], r_lp["elapsed"]]
                csv.writer(f).writerow(row)

            # Progress
            b_str = f"CNY {r_bap['cost']:,.0f}" if r_bap["cost"] is not None else r_bap["status"]
            n_str = f"CNY {r_lp['cost']:,.0f}" if r_lp["cost"] is not None else r_lp["status"]
            def _f(v, fmt, default="N/A"):
                return f"{v:{fmt}}" if v is not None else default
            print(f"  [{done:3d}/{total}]  s={s_id:2d}  BAP={b_str:>14s}  LP={n_str:>14s}  "
                  f"R_risk={_f(r_bap.get('R_risk'),'.4f')}/{_f(r_lp.get('R_risk'),'.4f')}  "
                  f"R_time={_f(r_bap.get('R_time'),'.4f')}/{_f(r_lp.get('R_time'),'.4f')}  "
                  f"RawRisk={_f(r_bap.get('raw_risk'),'.2f')}/{_f(r_lp.get('raw_risk'),'.2f')}  "
                  f"RawTime={_f(r_bap.get('raw_time'),'.2f')}/{_f(r_lp.get('raw_time'),'.2f')}h  "
                  f"({r_bap['elapsed']:.1f}s / {r_lp['elapsed']:.1f}s)")

    # ── Generate 4 box-plots (one per metric) ───────────────────────────────
    from scipy.stats import wilcoxon as _wilcoxon

    metrics = [
        ("raw_risk",  bap_raw_risk,  lp_raw_risk,  r"Raw Risk ($\sum (1-R_{ij})$)", "raw_risk_boxplot"),
        ("raw_time",  bap_raw_time,  lp_raw_time,  r"Raw Time (h)",                 "raw_time_boxplot"),
        ("R_risk",    bap_R_risk,    lp_R_risk,    r"Risk Reliability ($R_{risk}$)",  "R_risk_boxplot"),
        ("R_time",    bap_R_time,    lp_R_time,    r"Time Reliability ($R_{time}$)",  "R_time_boxplot"),
    ]

    for metric_name, bap_data, lp_data, y_label, filename in metrics:
        bap_list = [bap_data[l] for l in group_labels]
        lp_list  = [lp_data[l]  for l in group_labels]

        enhanced_boxplot(
            group_labels=group_labels,
            bap_costs=bap_list,
            bench_costs=lp_list,
            save_path=os.path.join(exp_dir, filename),
            y_label=y_label,
            x_label=r"$(\alpha_T,\;\alpha_{max})$",
            model_bap_label="BAP (Integer)",
            model_bench_label="LP Relaxation",
            c_bap=C_BAP,
            c_bench=C_LP,
        )
        print(f"  Saved {filename}.png/.tiff")

    # ── Summary statistics ──────────────────────────────────────────────────
    print("\n=== Summary ===")
    for metric_name, bap_data, lp_data, y_label, _ in metrics:
        print(f"\n--- {y_label} ---")
        for lbl in group_labels:
            b = np.array(bap_data[lbl], dtype=float)
            l = np.array(lp_data[lbl], dtype=float)
            valid = ~(np.isnan(b) | np.isnan(l))
            bp, lp = b[valid], l[valid]
            b_med = np.nanmedian(b)
            l_med = np.nanmedian(l)
            n = len(bp)
            if n >= 3 and not np.all(bp == lp):
                _stat, _p = _wilcoxon(bp, lp, alternative='two-sided')
                if   _p < 0.001: sig = "***"
                elif _p < 0.01:  sig = "**"
                elif _p < 0.05:  sig = "*"
                else:            sig = "ns"
                print(f"  {lbl:>10s}  BAP={b_med:>10.4f}  LP={l_med:>10.4f}  "
                      f"n={n}  p={_p:.4f} {sig}")
            elif n > 0:
                print(f"  {lbl:>10s}  BAP={b_med:>10.4f}  LP={l_med:>10.4f}  (n={n})")
            else:
                print(f"  {lbl:>10s}  No data")

    print(f"\nResults saved to {exp_dir}/")
    tee.close()
    return exp_dir


if __name__ == "__main__":
    print(f"Out-of-Sample Sensitivity Analysis: BAP vs LP Relaxation")
    print(f"  N={N_SCENARIOS} scenarios × {len(PARAM_GRID)} groups")
    print(f"  Metrics: raw_risk, raw_time, R_risk, R_time")
    print("=" * 65)
    exp_dir = run_out_of_sample()
    print(f"\nDone!  Open {exp_dir}/ for 4 boxplots.")
