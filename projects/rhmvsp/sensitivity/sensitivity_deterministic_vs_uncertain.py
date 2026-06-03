"""
Sensitivity Analysis: Deterministic vs Uncertainty-Aware Model (§3.2).

Head-to-head comparison showing the value of FULLY modeling uncertainty:
  - Deterministic model:  δ_T ≈ 0  AND  α_max = 1.0
      (crisp travel times; no risk-reliability constraint;
       treats all routes as acceptable regardless of accident belief)
  - Uncertainty-aware:    δ_T > 0  AND  α_max = 0.05
      (fuzzy travel-time distributions; risk-reliability constrained;
       routes must satisfy both time and risk reliability targets)

Both models use the SAME Branch-and-Price solver; the difference is whether
the solver accounts for fuzzy time uncertainty AND risk reliability.

Design: N shared scenarios (same `sample_scenario` seeds) across ALL groups.
  Each x-axis point represents a different δ_T value used by the
  uncertainty-aware model; the deterministic model always uses δ_T ≈ 0
  and α_max = 1.0 (no risk constraint).  Cost differences reflect the
  combined value of BOTH uncertainty dimensions.

Outputs:
  - Enhanced box-plots (paired violin per δ_T group)
  - CSV + experiment log + summary statistics
  - Wilcoxon signed-rank test + cost-gap annotation
"""
import sys, os, json, csv, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.core.uncertainty import UncertaintyEngine as UE2, TrapezoidalFuzzy as TrapFz
from src.core.scenario_sampler import sample_scenario
from sensitivity.plot_helpers import set_style, COLOURS, enhanced_boxplot

# ── Colour constants ──────────────────────────────────────────────────────────
C_UNC = COLOURS["C1"]  # Warm Terracotta — Uncertainty-aware (our model)
C_DET = COLOURS["C2"]  # Celadon Green   — Deterministic model

# ── Experiment parameters ─────────────────────────────────────────────────────
N_SCENARIOS = 50          # Per group.  Dry-run: 10 → final: 50
INSTANCE_PATH = "data/small/instance.json"

# Deterministic model: no uncertainty at all
DELTA_T_DET   = 0.001   # ≈ 0 → no fuzzy spread (crisp travel times)
ALPHA_MAX_DET = 1.0      # No risk-reliability constraint (all routes accepted)

# Uncertainty-aware model: full uncertainty modeling
ALPHA_T     = 0.8         # Fixed time-reliability threshold (§3.1)
ALPHA_MAX   = 0.05        # Risk-reliability threshold

# X-axis: environment uncertainty level δ_T.
# At each level the uncertainty-aware solver models travel times with this δ_T,
# while the deterministic solver always uses δ_T ≈ 0 and α_max = 1.0.
DELTA_T_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

# Base RNG seed (shared across all groups → same physical scenarios)
BASE_SEED = 42


def _update_od_deadlines(instance, config):
    """Recompute L_od for each OD pair based on current alpha_T and delta_T."""
    _temp_engine = UE2(config)
    _arcs = instance["network"]["arcs"]
    _hub_ids = {tn["node_id"] for tn in instance["network"].get("transfer_nodes", [])}
    for od in instance.get("od_pairs", []):
        od["alpha_T_od"] = config.alpha_T
        od_o, od_d = od["origin"], od["destination"]
        roads_out = [a for a in _arcs if a["i"] == od_o and a["mode"] == 1
                     and a["j"] in _hub_ids]
        roads_in  = [a for a in _arcs if a["j"] == od_d and a["mode"] == 1
                     and a["i"] in _hub_ids]
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
                trap = TrapFz(r1.a+tf.a+ra.a+r2.a, r1.b+tf.b+ra.b+r2.b,
                              r1.c+tf.c+ra.c+r2.c, r1.d+tf.d+ra.d+r2.d)
                det_L = trap.deterministic_equivalent_upper(config.alpha_T)
                od["L_od"] = float(round(od.get("E_od", 0.0) + det_L * 1.10, 1))


def _run_single(config, inst, logger):
    """Run one BAP solve → dict with cost and sub-components."""
    solver = BAPSolver(config, json.loads(json.dumps(inst)), logger)
    t0 = time.time()
    sol = solver.solve()
    elapsed = time.time() - t0
    status = sol.get("status", "unknown")
    ok = status != "infeasible"
    cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                 "reposition_cost","prep_cost","idle_cost","opportunity_cost"]
    result = {"cost": sol.get("total_cost") if ok else None,
              "status": status, "elapsed": round(elapsed, 2)}
    for col in cost_cols:
        result[col] = sol.get(col, 0) if ok else None
    return result


def run_deterministic_vs_uncertain():
    """Main: len(DELTA_T_ENV) groups × N scenarios × 2 models."""

    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir   = os.path.join("results", f"det_vs_unc_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Tee: capture all stdout → experiment.log
    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # Metadata
    meta = {
        "n_scenarios": N_SCENARIOS,
        "delta_T_det": DELTA_T_DET,
        "alpha_max_det": ALPHA_MAX_DET,
        "delta_T_grid": DELTA_T_GRID,
        "base_seed": BASE_SEED,
        "alpha_T": ALPHA_T,
        "alpha_max_unc": ALPHA_MAX,
        "instance": INSTANCE_PATH,
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Group labels
    group_labels = [f"{dT:.1f}" for dT in DELTA_T_GRID]

    # Storage: aligned NaN arrays for paired testing
    det_costs = defaultdict(list)   # label → [cost_0, cost_1, ...]
    unc_costs = defaultdict(list)   # label → [cost_0, cost_1, ...]

    csv_path = os.path.join(exp_dir, "experiment.csv")
    total = len(DELTA_T_GRID) * N_SCENARIOS * 2
    done  = 0

    cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                 "reposition_cost","prep_cost","idle_cost","opportunity_cost"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["model", "delta_T_group", "scenario_id",
            "cost"] + cost_cols + ["status", "elapsed_s"])

    # ── Pre-generate scenarios (shared across all groups for fairness) ────────
    # Same physical realizations; only the solver config differs per group.
    print(f"Pre-generating {N_SCENARIOS} shared scenarios...")
    scenarios = []
    for s_id in range(N_SCENARIOS):
        rng = np.random.default_rng(seed=BASE_SEED + s_id)
        scenarios.append(sample_scenario(base_instance, rng))
    print(f"  Done.")

    for dT in DELTA_T_GRID:
        label = f"{dT:.1f}"
        print(f"\n=== δ_T={dT:.1f}"
              f"  (uncertainty-aware: δ_T={dT:.1f}, α_max={ALPHA_MAX}"
              f"  |  deterministic: δ_T={DELTA_T_DET}, α_max={ALPHA_MAX_DET}) ===")

        for s_id in range(N_SCENARIOS):
            scen = scenarios[s_id]  # shared realization

            # ── Deterministic model (no uncertainty at all) ──────────────────
            cfg_det = RHMVSPConfig()
            cfg_det.delta_T   = DELTA_T_DET          # no fuzzy time
            cfg_det.alpha_T   = ALPHA_T
            cfg_det.alpha_max = ALPHA_MAX_DET         # no risk constraint
            cfg_det.solver_name    = "gurobi"
            cfg_det.solver_backend = "gurobi"
            scen_det = json.loads(json.dumps(scen))
            _update_od_deadlines(scen_det, cfg_det)

            r_det = _run_single(
                cfg_det, scen_det, Logger(level="ERROR", log_file=None)
            )
            done += 1

            with open(csv_path, "a", newline="") as f:
                row = ["DET", dT, s_id, r_det["cost"]]
                for col in cost_cols:
                    row.append(r_det.get(col) if r_det["cost"] is not None else None)
                row += [r_det["status"], r_det["elapsed"]]
                csv.writer(f).writerow(row)

            # ── Uncertainty-aware model (full uncertainty) ──────────────────
            cfg_unc = RHMVSPConfig()
            cfg_unc.delta_T   = dT                   # fuzzy time spread
            cfg_unc.alpha_T   = ALPHA_T
            cfg_unc.alpha_max = ALPHA_MAX             # risk-reliability limit
            cfg_unc.solver_name    = "gurobi"
            cfg_unc.solver_backend = "gurobi"
            scen_unc = json.loads(json.dumps(scen))
            _update_od_deadlines(scen_unc, cfg_unc)

            r_unc = _run_single(
                cfg_unc, scen_unc, Logger(level="ERROR", log_file=None)
            )
            done += 1

            with open(csv_path, "a", newline="") as f:
                row = ["UNC", dT, s_id, r_unc["cost"]]
                for col in cost_cols:
                    row.append(r_unc.get(col) if r_unc["cost"] is not None else None)
                row += [r_unc["status"], r_unc["elapsed"]]
                csv.writer(f).writerow(row)

            # Store aligned (NaN keeps index correspondence)
            det_costs[label].append(r_det["cost"] if r_det["cost"] is not None else float("nan"))
            unc_costs[label].append(r_unc["cost"] if r_unc["cost"] is not None else float("nan"))

            d_str = f"CNY {cost_det:,.0f}" if cost_det is not None else status_det
            u_str = f"CNY {cost_unc:,.0f}" if cost_unc is not None else status_unc
            print(f"  [{done:3d}/{total}]  s={s_id:2d}  DET={d_str:>14s}  UNC={u_str:>14s}"
                  f"  ({elapsed_det:.1f}s / {elapsed_unc:.1f}s)")

    # ── Plot enhanced box-plots ───────────────────────────────────────────────
    det_list = [det_costs[l] for l in group_labels]
    unc_list = [unc_costs[l] for l in group_labels]

    enhanced_boxplot(
        group_labels=group_labels,
        bap_costs=unc_list,       # Left violin  ← Uncertainty-Aware
        bench_costs=det_list,     # Right violin ← Deterministic
        save_path=os.path.join(exp_dir, "det_vs_unc_boxplot"),
        y_label="Total Cost (CNY)",
        x_label=r"$\delta_T$ (Environment Uncertainty Level)",
        model_bap_label="Uncertainty-Aware",   # (δ_T>0, α_max=0.05)
        model_bench_label="Deterministic",     # (δ_T≈0, α_max=1.0)
        c_bap=C_UNC,
        c_bench=C_DET,
    )

    # ── Summary statistics ────────────────────────────────────────────────────
    from scipy.stats import wilcoxon as _wilcoxon
    print("\n=== Summary (Deterministic vs Uncertainty-Aware) ===")
    print(f"  {'δ_T':>6s}  {'DET median':>14s}  {'UNC median':>14s}"
          f"  {'gap%':>10s}  n   {'p-value':>10s}  sig")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*10}  ---  {'-'*10}  ---")

    for lbl in group_labels:
        d = np.array(det_costs[lbl], dtype=float)
        u = np.array(unc_costs[lbl], dtype=float)
        valid = ~(np.isnan(d) | np.isnan(u))
        dp, up = d[valid], u[valid]
        d_med = np.nanmedian(d)
        u_med = np.nanmedian(u)
        n = len(dp)

        if n > 0 and u_med != 0:
            # gap: (DET − UNC) / UNC  →  positive means DET is more expensive
            gap_pct = (d_med - u_med) / u_med * 100
            gap_str = f"{gap_pct:+.1f}%"
            if n >= 3 and not np.all(dp == up):
                _stat, _p = _wilcoxon(dp, up, alternative='two-sided')
                if   _p < 0.001: sig = "***"
                elif _p < 0.01:  sig = "**"
                elif _p < 0.05:  sig = "*"
                else:            sig = "ns"
                print(f"  {lbl:>6s}  {d_med:>14,.0f}  {u_med:>14,.0f}"
                      f"  {gap_str:>10s}  {n:2d}  {_p:>10.4f}  {sig}")
            else:
                print(f"  {lbl:>6s}  {d_med:>14,.0f}  {u_med:>14,.0f}"
                      f"  {gap_str:>10s}  {n:2d}  (insufficient)")
        elif n > 0:
            print(f"  {lbl:>6s}  {d_med:>14,.0f}  {u_med:>14,.0f}"
                  f"  {'n/a':>10s}  {n:2d}")
        else:
            print(f"  {lbl:>6s}  No data")

    print(f"\nFiles saved to {exp_dir}/")
    tee.close()


if __name__ == "__main__":
    print(f"Deterministic vs Uncertainty-Aware Sensitivity"
          f"  (N={N_SCENARIOS} scenarios × {len(DELTA_T_GRID)} groups)")
    print("=" * 65)
    run_deterministic_vs_uncertain()
    print("\nDone!  Open results/det_vs_unc_*/det_vs_unc_boxplot.png")
