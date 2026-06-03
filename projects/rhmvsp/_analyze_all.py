"""
Comprehensive analysis of all experiment results for framework.md update.
Produces percentage-based summaries only.
"""
import csv, statistics, os, json

BASE = "results"

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def median_of(rows, col):
    vals = [float(r[col]) for r in rows if r[col] and r[col] != '']
    return statistics.median(vals) if vals else None

def mean_of(rows, col):
    vals = [float(r[col]) for r in rows if r[col] and r[col] != '']
    return statistics.mean(vals) if vals else None

# ============================================================
# BENCHMARK
# ============================================================
print("=" * 80)
print("BENCHMARK (constant model)")
print("=" * 80)
bench = read_csv(os.path.join(BASE, "coupled_20260602_100925/benchmark.csv"))
# Find benchmark where dT=0.8 alpha_max=0.05
for r in bench:
    if '0.05' in r.get('alpha_max','') and '0.80' in r.get('dT',''):
        print(f"  Cost={float(r['cost']):,.0f}, R_risk={float(r['risk_reliability']):.4f}, R_time={float(r['time_reliability']):.4f}")
        BENCH_COST = float(r['cost'])
        BENCH_RRISK = float(r['risk_reliability'])
        BENCH_RTIME = float(r['time_reliability'])
        break

# ============================================================
# 1. alpha_max sensitivity
# ============================================================
print("\n" + "=" * 80)
print("1. ALPHA_MAX SENSITIVITY (alpha_T=0.8, delta_T=0.8)")
print("=" * 80)
am_data = read_csv(os.path.join(BASE, "coupled_20260602_100925/experiment.csv"))
# Filter for dT=0.8, alpha_max varies
am_rows = [r for r in am_data if r.get('dT_group','') == '0.80']
am_rows_sorted = sorted(am_rows, key=lambda r: float(r['alpha_max']))
print(f"  {'alpha_max':>10} | {'Cost':>12} | {'ΔCost%':>8} | {'R_risk':>8} | {'R_time':>8} | {'ΔR_risk%':>10} | {'ΔR_time%':>10}")
for r in am_rows_sorted:
    am = float(r['alpha_max'])
    cost = float(r['total_cost'])
    rr = float(r['risk_reliability'])
    rt = float(r['time_reliability'])
    dc = (cost - BENCH_COST) / BENCH_COST * 100
    dr = (rr - BENCH_RRISK) / BENCH_RRISK * 100
    dt = (rt - BENCH_RTIME) / BENCH_RTIME * 100
    print(f"  {am:>10.3f} | {cost:>12,.0f} | {dc:>+7.1f}% | {rr:>8.4f} | {rt:>8.4f} | {dr:>+9.1f}% | {dt:>+9.1f}%")

# ============================================================
# 2. alpha_T sensitivity
# ============================================================
print("\n" + "=" * 80)
print("2. ALPHA_T SENSITIVITY (alpha_max=0.05, delta_T=0.8)")
print("=" * 80)
at_data = read_csv(os.path.join(BASE, "sensitivity_alpha_T_20260602_094416/experiment.csv"))
at_rows_sorted = sorted(at_data, key=lambda r: float(r['alpha_T']))
print(f"  {'alpha_T':>10} | {'Cost':>12} | {'ΔCost%':>8} | {'R_risk':>8} | {'R_time':>8} | {'ΔR_time%':>10}")
for r in at_rows_sorted:
    at = float(r['alpha_T'])
    cost = float(r['total_cost'])
    rr = float(r['risk_reliability'])
    rt = float(r['time_reliability'])
    dc = (cost - BENCH_COST) / BENCH_COST * 100
    dt = (rt - BENCH_RTIME) / BENCH_RTIME * 100
    print(f"  {at:>10.2f} | {cost:>12,.0f} | {dc:>+7.1f}% | {rr:>8.4f} | {rt:>8.4f} | {dt:>+9.1f}%")

# ============================================================
# 3. Value of fuzzy time (delta_T sweep)
# ============================================================
print("\n" + "=" * 80)
print("3. VALUE OF FUZZY TIME (delta_T sweep, alpha_max=0.05, alpha_T=0.8)")
print("=" * 80)
dt_dir = None
for d in os.listdir(BASE):
    if d.startswith("value_fuzzy_time_"):
        dt_dir = os.path.join(BASE, d)
        break
if dt_dir:
    dt_data = read_csv(os.path.join(dt_dir, "experiment.csv"))
    # Filter for alpha_max=0.05 only
    dt_rows = [r for r in dt_data if r.get('alpha_max','') == '0.05']
    dt_rows_sorted = sorted(dt_rows, key=lambda r: float(r['delta_T']))
    print(f"  {'delta_T':>10} | {'Cost':>12} | {'ΔCost%':>8} | {'R_risk':>8} | {'R_time':>8} | {'ΔR_time%':>10}")
    for r in dt_rows_sorted:
        dT = float(r['delta_T'])
        cost = float(r['total_cost'])
        rr = float(r['risk_reliability'])
        rt = float(r['time_reliability'])
        dc = (cost - BENCH_COST) / BENCH_COST * 100
        dt_val = (rt - BENCH_RTIME) / BENCH_RTIME * 100
        print(f"  {dT:>10.2f} | {cost:>12,.0f} | {dc:>+7.1f}% | {rr:>8.4f} | {rt:>8.4f} | {dt_val:>+9.1f}%")

# ============================================================
# 4. Coupled alpha_max × delta_T grid
# ============================================================
print("\n" + "=" * 80)
print("4. COUPLED alpha_max × delta_T (key observations)")
print("=" * 80)
coupled = read_csv(os.path.join(BASE, "coupled_20260602_100925/experiment.csv"))
# Summary: for each alpha_max, show range of costs across dT
am_vals = sorted(set(float(r['alpha_max']) for r in coupled))
print(f"  {'alpha_max':>10} | {'# groups':>8} | {'Min Cost':>12} | {'Max Cost':>12} | {'Min Δ%':>8} | {'Max Δ%':>8}")
for am in am_vals:
    rows = [r for r in coupled if float(r['alpha_max']) == am]
    costs = [float(r['total_cost']) for r in rows]
    dCs = [(c - BENCH_COST) / BENCH_COST * 100 for c in costs]
    print(f"  {am:>10.3f} | {len(rows):>8} | {min(costs):>12,.0f} | {max(costs):>12,.0f} | {min(dCs):>+7.1f}% | {max(dCs):>+7.1f}%")

# ============================================================
# 5. Out-of-sample validation
# ============================================================
print("\n" + "=" * 80)
print("5. OUT-OF-SAMPLE VALIDATION")
print("=" * 80)
oos_dir = None
for d in os.listdir(BASE):
    if d.startswith("out_of_sample_"):
        oos_dir = os.path.join(BASE, d)
        break
if oos_dir:
    oos_data = read_csv(os.path.join(oos_dir, "experiment.csv"))
    groups = {}
    for r in oos_data:
        key = (r.get('delta_T_group', ''), r.get('alpha_max', ''))
        model = r.get('model', '')
        groups.setdefault(key, {}).setdefault(model, []).append(float(r['cost']) if r['cost'] else None)
    
    print(f"  {'(dT, aMax)':>14} | {'BAP Cost':>12} | {'LP Cost':>12} | {'Gap%':>8} | {'BAP R_time':>12} | {'LP R_time':>12}")
    for key in sorted(groups.keys()):
        dT, am = key
        bap_costs = [v for v in groups[key].get('BAP', []) if v]
        lp_costs = [v for v in groups[key].get('LP', []) if v]
        bap_rt = [v for v in groups[key].get('BAP', []) if v]  # placeholder
        if bap_costs and lp_costs:
            bap_med = statistics.median(bap_costs)
            lp_med = statistics.median(lp_costs)
            gap = (bap_med - lp_med) / lp_med * 100
            print(f"  ({dT},{am:>5}) | {bap_med:>12,.0f} | {lp_med:>12,.0f} | {gap:>+7.1f}% |")

# ============================================================
# 6. Deterministic vs Uncertainty-Aware
# ============================================================
print("\n" + "=" * 80)
print("6. DETERMINISTIC vs UNCERTAINTY-AWARE")
print("=" * 80)
du_dir = None
for d in os.listdir(BASE):
    if d.startswith("det_vs_unc_"):
        du_dir = os.path.join(BASE, d)
        break
if du_dir:
    du_data = read_csv(os.path.join(du_dir, "experiment.csv"))
    groups = {}
    for r in du_data:
        dT = r.get('delta_T_group', '')
        model = r.get('model', '')
        groups.setdefault(dT, {}).setdefault(model, []).append(float(r['cost']) if r['cost'] else None)
    
    for dT in sorted(groups.keys()):
        det_costs = [v for v in groups[dT].get('DET', []) if v]
        unc_costs = [v for v in groups[dT].get('UNC', []) if v]
        if det_costs and unc_costs:
            det_med = statistics.median(det_costs)
            unc_med = statistics.median(unc_costs)
            gap = (unc_med - det_med) / det_med * 100
            print(f"  dT={dT}: DET median={det_med:>12,.0f}, UNC median={unc_med:>12,.0f}, gap={gap:+.2f}%")

# ============================================================
# 7. Algorithm performance
# ============================================================
print("\n" + "=" * 80)
print("7. ALGORITHM PERFORMANCE")
print("=" * 80)
ap_dir = None
for d in os.listdir(BASE):
    if d.startswith("sensitivity_perf_") or d.startswith("algorithm_perf"):
        ap_dir = os.path.join(BASE, d)
        break
if ap_dir:
    ap_data = read_csv(os.path.join(ap_dir, "experiment.csv"))
    # Group by (alpha_T, alpha_max) for BAP and LP
    for r in ap_data[:5]:
        print(f"  Keys: {list(r.keys())}")
    
    bap_rows = [r for r in ap_data if r.get('model','') == 'BAP']
    lp_rows = [r for r in ap_data if r.get('model','') == 'LP']
    
    if bap_rows:
        bap_costs = [float(r.get('total_cost',0)) for r in bap_rows if r.get('total_cost')]
        lp_costs = [float(r.get('total_cost',0)) for r in lp_rows if r.get('total_cost')]
        bap_times = [float(r.get('runtime',0)) for r in bap_rows if r.get('runtime')]
        lp_times = [float(r.get('runtime',0)) for r in lp_rows if r.get('runtime')]
        iters = [int(r.get('iterations',0)) for r in bap_rows if r.get('iterations')]
        
        print(f"  BAP: {len(bap_rows)} configs")
        print(f"    Cost range: {min(bap_costs):,.0f} - {max(bap_costs):,.0f}")
        print(f"    Time range: {min(bap_times):.2f}s - {max(bap_times):.2f}s")
        print(f"    Iter range: {min(iters)} - {max(iters)}")
        
        if lp_costs:
            print(f"  LP: {len(lp_rows)} configs")
            print(f"    Cost range: {min(lp_costs):,.0f} - {max(lp_costs):,.0f}")
            print(f"    Time range: {min(lp_times):.2f}s - {max(lp_times):.2f}s")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
