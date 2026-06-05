"""
Sensitivity Analysis: Out-of-Sample Box-Plot (BAP vs LP-only Benchmark).

Generates N Monte Carlo scenarios for 6 representative (delta_T, alpha_max)
combinations, solves each with BAP (reliable model) and LP-only (benchmark),
and plots enhanced box-plots with strip-scatter overlay.

Reference style: Wang (2025) Fig. 8 — paired box-plots per parameter group.
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

# ── Colour constants (colour set2.txt) ────────────────────────────────────────
C_BAP   = COLOURS["C1"]  # Warm Terracotta — BAP (reliable model)
C_BENCH = COLOURS["C2"]  # Celadon Green   — LP-only (benchmark)

# ── Experiment parameters ─────────────────────────────────────────────────────
N_SCENARIOS = 200          # Dry-run: 50 → final: 100~200 per group
INSTANCE_PATH = "data/small/instance.json"

# 6 parameter combinations: (delta_T, alpha_max)
PARAM_GRID = [
    (0.5, 0.02),
    (0.5, 0.05),
    (0.6, 0.02),
    (0.6, 0.05),
    (0.7, 0.02),
    (0.7, 0.05),
    (0.8, 0.02),
    (0.8, 0.05),
]


def _update_od_deadlines(instance, config):
    """Recompute L_od for each OD pair based on current alpha_T and delta_T.

    alpha_T is fixed at 0.8; delta_T varies the fuzzy time distributions
    via UncertaintyEngine, which directly affects the cumulative trapezoid.
    """
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
                trap = TrapFz(r1.a + tf.a + ra.a + r2.a,
                              r1.b + tf.b + ra.b + r2.b,
                              r1.c + tf.c + ra.c + r2.c,
                              r1.d + tf.d + ra.d + r2.d)
                det_L = trap.deterministic_equivalent_upper(config.alpha_T)
                od["L_od"] = float(round(od.get("E_od", 0.0) + det_L * 1.10, 1))


def _run_single(config, inst, exp_dir, model_tag, dT, a_max, seed):
    """Run one BAP/LP-only solve and return dict with cost and sub-components."""
    logger = Logger(level="ERROR", log_file=None)  # output via stdout → Tee
    solver = BAPSolver(config, json.loads(json.dumps(inst)), logger)
    import time
    t0 = time.time()
    sol = solver.solve()
    elapsed = time.time() - t0

    status  = sol.get("status", "unknown")
    ok = status != "infeasible"
    cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                 "reposition_cost","prep_cost","idle_cost","opportunity_cost"]
    result = {"cost": sol.get("total_cost") if ok else None,
              "status": status, "elapsed": round(elapsed, 2)}
    for col in cost_cols:
        result[col] = sol.get(col, 0) if ok else None
    return result


def run_out_of_sample():
    """Main: 6 groups × N scenarios × 2 models = 12N solves."""

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
        "param_grid": [(dT, am) for dT, am in PARAM_GRID],
        "instance": INSTANCE_PATH,
        "alpha_T_fixed": 0.8,
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    group_labels = [f"{dT}-{aM:.2f}" for dT, aM in PARAM_GRID]
    bap_costs   = defaultdict(list)    # group_label → [cost, ...]
    bench_costs = defaultdict(list)

    total = len(PARAM_GRID) * N_SCENARIOS * 2
    done  = 0

    # CSV header
    csv_path = os.path.join(exp_dir, "experiment.csv")
    cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                 "reposition_cost","prep_cost","idle_cost","opportunity_cost"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["group_label", "delta_T", "alpha_max", "scenario_id",
             "model", "cost"] + cost_cols + ["status", "elapsed_s"]
        )

    for dT, a_max in PARAM_GRID:
        label = f"{dT}-{a_max:.2f}"
        print(f"\n=== \u03b4_T={dT:.1f},  \u03b1_max={a_max:.3f} ===")

        for s_id in range(N_SCENARIOS):
            seed = 1000 * int(dT * 100) + int(a_max * 1000) + s_id
            rng = np.random.default_rng(seed=seed)

            # ── One shared scenario (paired design) ─────────────────────────
            scen = sample_scenario(base_instance, rng)

            # ── BAP (reliable model) ─────────────────────────────────────────
            cfg_bap  = RHMVSPConfig()
            # alpha_T stays at default 0.8; delta_T is the sweep variable
            cfg_bap.delta_T   = dT
            cfg_bap.alpha_max = a_max
            cfg_bap.solver_name    = "gurobi"
            cfg_bap.solver_backend = "gurobi"
            scen_bap = json.loads(json.dumps(scen))  # deep copy for isolation
            _update_od_deadlines(scen_bap, cfg_bap)

            r_bap = _run_single(
                cfg_bap, scen_bap, exp_dir, "bap", dT, a_max, s_id
            )
            done += 1

            with open(csv_path, "a", newline="") as f:
                row = [label, dT, a_max, s_id, "BAP", r_bap["cost"]]
                for col in cost_cols:
                    row.append(r_bap.get(col) if r_bap["cost"] is not None else None)
                row += [r_bap["status"], r_bap["elapsed"]]
                csv.writer(f).writerow(row)

            # ── LP-only (benchmark) — same scenario, different config ─────────
            cfg_lp  = RHMVSPConfig()
            # alpha_T stays at default 0.8; delta_T is the sweep variable
            cfg_lp.delta_T   = dT
            cfg_lp.alpha_max = a_max
            cfg_lp.lp_only   = True
            cfg_lp.solver_name    = "gurobi"
            cfg_lp.solver_backend = "gurobi"
            scen_lp = json.loads(json.dumps(scen))  # deep copy of same scenario
            _update_od_deadlines(scen_lp, cfg_lp)

            r_lp = _run_single(
                cfg_lp, scen_lp, exp_dir, "lp", dT, a_max, s_id
            )
            done += 1

            # Store aligned — NaN for failed runs keeps index correspondence
            bap_costs[label].append(r_bap["cost"] if r_bap["cost"] is not None else float("nan"))
            bench_costs[label].append(r_lp["cost"] if r_lp["cost"] is not None else float("nan"))

            with open(csv_path, "a", newline="") as f:
                row = [label, dT, a_max, s_id, "LP", r_lp["cost"]]
                for col in cost_cols:
                    row.append(r_lp.get(col) if r_lp["cost"] is not None else None)
                row += [r_lp["status"], r_lp["elapsed"]]
                csv.writer(f).writerow(row)

            # Progress
            b_str = f"CNY {r_bap['cost']:,.0f}" if r_bap["cost"] is not None else r_bap["status"]
            l_str = f"CNY {r_lp['cost']:,.0f}" if r_lp["cost"] is not None else r_lp["status"]
            print(f"  [{done:3d}/{total}]  s={s_id:2d}  BAP={b_str:>14s}  LP={l_str:>14s}  "
                  f"({r_bap['elapsed']:.1f}s / {r_lp['elapsed']:.1f}s)")

    # ── Plot enhanced box-plots ───────────────────────────────────────────────
    bap_list   = [bap_costs[l]   for l in group_labels]
    bench_list = [bench_costs[l] for l in group_labels]

    enhanced_boxplot(
        group_labels=group_labels,
        bap_costs=bap_list,
        bench_costs=bench_list,
        save_path=os.path.join(exp_dir, "out_of_sample_boxplot"),
        y_label="Total Cost (CNY)",
        model_bap_label="BAP (Reliable)",
        model_bench_label="LP Relaxation (Benchmark)",
        c_bap=C_BAP,
        c_bench=C_BENCH,
    )

    # Summary statistics + cost gap + paired Wilcoxon signed-rank test
    from scipy.stats import wilcoxon as _wilcoxon
    print("\n=== Summary ===")
    for lbl in group_labels:
        b = np.array(bap_costs[lbl], dtype=float)
        l = np.array(bench_costs[lbl], dtype=float)
        valid = ~(np.isnan(b) | np.isnan(l))
        bp, lp = b[valid], l[valid]
        b_med = np.nanmedian(b)
        l_med = np.nanmedian(l)
        n = len(bp)
        if n > 0 and l_med != 0:
            gap_pct = (b_med - l_med) / l_med * 100
            gap_str = f"gap={gap_pct:+.1f}%"
            if n >= 3 and not np.all(bp == lp):
                _stat, _p = _wilcoxon(bp, lp, alternative='two-sided')
                if   _p < 0.001: sig = "***"
                elif _p < 0.01:  sig = "**"
                elif _p < 0.05:  sig = "*"
                else:            sig = "ns"
                print(f"  {lbl:>10s}  BAP={b_med:>12,.0f}  LP={l_med:>12,.0f}  "
                      f"{gap_str:>10s}  n={n}  p={_p:.4f} {sig}")
            else:
                print(f"  {lbl:>10s}  BAP={b_med:>12,.0f}  LP={l_med:>12,.0f}  "
                      f"{gap_str:>10s}  (n={n})")
        elif n > 0:
            print(f"  {lbl:>10s}  BAP={b_med:>12,.0f}  LP={l_med:>12,.0f}  (n={n})")
        else:
            print(f"  {lbl:>10s}  No data")

    print(f"\nResults saved to {exp_dir}/")
    tee.close()
    return exp_dir


if __name__ == "__main__":
    print(f"Out-of-Sample Sensitivity Analysis  (N={N_SCENARIOS} scenarios × {len(PARAM_GRID)} groups)")
    print("=" * 65)
    exp_dir = run_out_of_sample()
    print(f"\nDone!  Open {exp_dir}/out_of_sample_boxplot.png")
