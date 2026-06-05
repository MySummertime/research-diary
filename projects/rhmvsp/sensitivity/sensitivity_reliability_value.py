"""
Sensitivity Analysis: Value of Reliability Constraints.

Core question: What is the cost of reliability, and what does it buy?

Design:
  - Fix δ_T=0.001 (deterministic time) to isolate reliability effects
  - Sweep α_max × α_T in a 3×3 grid
  - For each configuration, solve BAP and evaluate with Monte Carlo scenarios
  - Compare: cost, R_risk, R_time, actual failure rate

The key insight: reliability constraints increase cost but reduce actual failures
when the solution is deployed in uncertain environments.

Outputs:
  - CSV with cost breakdown, R_risk, R_time, failure rate per scenario
  - Dual Y-axis plots (R_risk + R_time vs α_max, with cost bars)
  - Summary statistics
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

# ── Colour constants (matching value_of_fuzzy_time) ──────────────────────────
C_COST   = COLOURS["C5"]  # Pale Lime  — Cost bars
C_RRISK  = COLOURS["C1"]  # Terracotta — R_risk line
C_RTIME  = COLOURS["C2"]  # Celadon    — R_time line

# ── Experiment parameters ─────────────────────────────────────────────────────
N_SCENARIOS = 10          # Monte Carlo scenarios per configuration
INSTANCE_PATH = "data/small/instance.json"
FIXED_DELTA_T  = 0.001  # Deterministic time — isolate reliability effects

# Two configurations: No-Reliability vs Full-Reliability
CONFIGS = [
    {"label": "No-Reliability",  "a_max": 1.0,  "a_T": 0.5},
    {"label": "Full-Reliability", "a_max": 0.05, "a_T": 0.8},
]


def _update_od_deadlines(instance, config):
    """Recompute L_od for each OD pair based on current alpha_T."""
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
                trap = TrapFz(r1.a+tf.a+ra.a+r2.a, r1.b+tf.b+ra.b+r2.b,
                              r1.c+tf.c+ra.c+r2.c, r1.d+tf.d+ra.d+r2.d)
                det_L = trap.deterministic_equivalent_upper(config.alpha_T)
                od["L_od"] = float(round(od.get("E_od", 0.0) + det_L * 1.10, 1))


def _make_config(a_max, a_T):
    """Create config with specified α_max and α_T, deterministic time."""
    cfg = RHMVSPConfig()
    cfg.delta_T = FIXED_DELTA_T
    cfg.alpha_max = a_max
    cfg.alpha_T = a_T
    cfg.enable_time_varying_risk = True
    cfg.enable_time_varying_consequence = True
    cfg.solver_name = "gurobi"
    cfg.solver_backend = "gurobi"
    return cfg


def _solve_model(config, instance, logger):
    """Solve one model instance, return dict with cost, reliability, and status."""
    solver = BAPSolver(config, json.loads(json.dumps(instance)), logger)
    t0 = time.time()
    sol = solver.solve()
    elapsed = time.time() - t0

    status = sol.get("status", "unknown")
    ok = status != "infeasible"
    cost_cols = ["deploy_cost", "hub_cost", "transport_cost", "transfer_cost",
                 "reposition_cost", "prep_cost", "idle_cost", "opportunity_cost"]
    result = {"cost": sol.get("total_cost") if ok else None,
              "R_risk": sol.get("R_net_risk", 0) if ok else None,
              "R_time": sol.get("R_net_time", 0) if ok else None,
              "status": status, "elapsed": round(elapsed, 2)}
    for col in cost_cols:
        result[col] = sol.get(col, 0) if ok else None
    return result


def run_reliability_value():
    """Main: Compare No-Reliability vs Full-Reliability with Monte Carlo evaluation."""

    with open(INSTANCE_PATH) as f:
        base_instance = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"reliability_value_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Tee: capture all stdout → experiment.log
    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    # Metadata
    meta = {
        "n_scenarios": N_SCENARIOS,
        "configs": CONFIGS,
        "fixed_delta_T": FIXED_DELTA_T,
        "instance": INSTANCE_PATH,
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # CSV setup
    csv_path = os.path.join(exp_dir, "reliability_value.csv")
    cost_cols = ["deploy_cost", "hub_cost", "transport_cost", "transfer_cost",
                 "reposition_cost", "prep_cost", "idle_cost", "opportunity_cost"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "alpha_max", "alpha_T", "scenario_id",
                     "cost", "R_risk", "R_time"] + cost_cols + ["status", "elapsed_s"])

    # Results storage
    results = {}  # label → {costs, rrisks, rtimes}

    total = len(CONFIGS) * N_SCENARIOS
    done = 0

    for cfg_info in CONFIGS:
        label = cfg_info["label"]
        a_max = cfg_info["a_max"]
        a_T = cfg_info["a_T"]

        print(f"\n{'='*60}")
        print(f"Configuration: {label}")
        print(f"  α_max={a_max:.3f}, α_T={a_T:.2f}, δ_T={FIXED_DELTA_T}")
        print(f"{'='*60}")

        # Solve on base instance
        cfg = _make_config(a_max, a_T)
        inst = json.loads(json.dumps(base_instance))
        _update_od_deadlines(inst, cfg)
        r_base = _solve_model(cfg, inst, Logger(level="ERROR", log_file=None))
        print(f"  Base cost: {r_base['cost']:>12,.0f}" if r_base['cost'] else "  Base: INFEASIBLE")

        # Monte Carlo evaluation
        results[label] = {"costs": [], "rrisks": [], "rtimes": [], "breakdowns": []}
        for s_id in range(N_SCENARIOS):
            seed = hash(label) % 10000 + s_id
            rng = np.random.default_rng(seed=seed)
            scen = sample_scenario(base_instance, rng)

            cfg_s = _make_config(a_max, a_T)
            scen_s = json.loads(json.dumps(scen))
            _update_od_deadlines(scen_s, cfg_s)
            r = _solve_model(cfg_s, scen_s, Logger(level="ERROR", log_file=None))
            done += 1

            with open(csv_path, "a", newline="") as f:
                row = [label, a_max, a_T, s_id, r["cost"], r.get("R_risk", ""), r.get("R_time", "")]
                for col in cost_cols:
                    row.append(r.get(col) if r["cost"] is not None else None)
                row += [r["status"], r["elapsed"]]
                csv.writer(f).writerow(row)

            # Store for analysis
            results[label]["costs"].append(r["cost"] if r["cost"] is not None else float("nan"))
            results[label]["rrisks"].append(r.get("R_risk") if r.get("R_risk") is not None else float("nan"))
            results[label]["rtimes"].append(r.get("R_time") if r.get("R_time") is not None else float("nan"))
            results[label]["breakdowns"].append(r)

            # Progress
            c_str = f"¥{r['cost']:,.0f}" if r["cost"] is not None else r["status"]
            print(f"  [{done:4d}/{total}]  s={s_id:2d}  cost={c_str:>12s}  R_risk={r.get('R_risk',0):.4f}  R_time={r.get('R_time',0):.4f}")
            if r["cost"] is not None:
                print(f"    components: deploy={r.get('deploy_cost',0):,.0f}  hub={r.get('hub_cost',0):,.0f}  transport={r.get('transport_cost',0):,.0f}  transfer={r.get('transfer_cost',0):,.0f}  reposition={r.get('reposition_cost',0):,.0f}  opp={r.get('opportunity_cost',0):,.0f}")

    # ── Summary statistics ───────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("RELIABILITY VALUE SUMMARY")
    print(f"{'='*90}")
    print(f"{'Label':>18s}  {'Cost med':>10s}  {'R_risk med':>10s}  {'R_time med':>10s}  {'n':>4s}")
    print("-" * 90)

    for cfg_info in CONFIGS:
        label = cfg_info["label"]
        costs = np.array(results[label]["costs"], dtype=float)
        rrisks = np.array(results[label]["rrisks"], dtype=float)
        rtimes = np.array(results[label]["rtimes"], dtype=float)
        valid = ~(np.isnan(costs) | np.isnan(rrisks) | np.isnan(rtimes))
        n = valid.sum()
        if n > 0:
            c_med = np.nanmedian(costs)
            r_med = np.nanmedian(rrisks)
            t_med = np.nanmedian(rtimes)
            print(f"  {label:>16s}  {c_med:>10,.0f}  {r_med:>10.4f}  {t_med:>10.4f}  {n:4d}")
        else:
            print(f"  {label:>16s}  {'No data':>10s}")

    # Cost gap
    if len(CONFIGS) == 2:
        c0 = np.nanmedian(results[CONFIGS[0]["label"]]["costs"])
        c1 = np.nanmedian(results[CONFIGS[1]["label"]]["costs"])
        r0 = np.nanmedian(results[CONFIGS[0]["label"]]["rrisks"])
        r1 = np.nanmedian(results[CONFIGS[1]["label"]]["rrisks"])
        t0 = np.nanmedian(results[CONFIGS[0]["label"]]["rtimes"])
        t1 = np.nanmedian(results[CONFIGS[1]["label"]]["rtimes"])
        print("-" * 90)
        print(f"  {'Cost Gap':>16s}  {c1-c0:>+10,.0f}  {r1-r0:>+10.4f}  {t1-t0:>+10.4f}")
        if c0 > 0:
            print(f"  {'Cost Gap %':>16s}  {(c1-c0)/c0*100:>+10.1f}%")

    print(f"{'='*90}")

    # ── Cost breakdown comparison chart ──────────────────────────────────
    if len(CONFIGS) == 2:
        import matplotlib.pyplot as plt
        set_style()

        cost_labels = ["Vehicle deploy", "Hub setup", "Transport", "Transfer",
                       "Reposition", "Opportunity"]
        cost_keys   = ["deploy_cost", "hub_cost", "transport_cost", "transfer_cost",
                       "reposition_cost", "opportunity_cost"]
        cost_colors = [COLOURS["C1"], COLOURS["C2"], COLOURS["C3"], COLOURS["C4"],
                       COLOURS["C5"], COLOURS["C1"]]

        # Compute median for each config
        medians = {}
        for cfg_info in CONFIGS:
            label = cfg_info["label"]
            medians[label] = []
            for key in cost_keys:
                vals = [r.get(key, 0) for r in results[label]["breakdowns"]]
                medians[label].append(np.nanmedian(vals))

        fig, ax = plt.subplots(figsize=(8, 5))
        y = np.arange(len(cost_labels))
        bar_h = 0.35

        vals0 = medians[CONFIGS[0]["label"]]
        vals1 = medians[CONFIGS[1]["label"]]

        ax.barh(np.array(y) + bar_h/2, vals1, height=bar_h, color=COLOURS["C1"], alpha=0.8,
                label=CONFIGS[1]["label"], edgecolor='none')
        ax.barh(np.array(y) - bar_h/2, vals0, height=bar_h, color=COLOURS["C2"], alpha=0.8,
                label=CONFIGS[0]["label"], edgecolor='none')

        ax.set_yticks(y)
        ax.set_yticklabels(cost_labels, fontsize=11)
        ax.set_xlabel("Cost (CNY)", fontsize=14)
        ax.legend(fontsize=11, frameon=False)
        ax.tick_params(axis='x', direction='in', labelsize=11)
        ax.tick_params(axis='y', direction='in', labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        chart_path = os.path.join(exp_dir, "cost_breakdown_comparison")
        for ext in ["png", "tiff"]:
            fig.savefig(f"{chart_path}.{ext}", dpi=600, bbox_inches='tight',
                        format=ext, transparent=True)
        plt.close(fig)
        print(f"\nCost breakdown chart saved → {chart_path}.png / .tiff")

    print(f"\nResults saved to {exp_dir}/")
    tee.close()
    return exp_dir


if __name__ == "__main__":
    print(f"Reliability Value Experiment ({len(CONFIGS)} configs × {N_SCENARIOS} scenarios)")
    print("=" * 60)
    exp_dir = run_reliability_value()
    print(f"\nDone!  Open {exp_dir}/")
