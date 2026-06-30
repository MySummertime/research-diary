"""
Standalone plotting script for Algorithm Performance experiment.

Reads an existing algorithm_performance.csv and generates per-instance-size
charts (BAP vs Branch-and-Cut vs Benders Decomposition).

Usage:
    conda run -n rhmvsp python sensitivity/plot_algorithm_performance.py
    conda run -n rhmvsp python sensitivity/plot_algorithm_performance.py --csv results/algorithm_performance/algorithm_performance.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MPLRect
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensitivity.plot_helpers import set_style, COLOURS


def _parse_cost(val: str):
    """Parse a cost string like '5,175,941' or 'N/A' to float or None."""
    if val is None or val.strip().upper() == "N/A":
        return None
    return float(val.replace(",", ""))


def _derive_size(row: dict) -> str:
    """Derive instance size from Instance_Size column (preferred) or instance name."""
    if "Instance_Size" in row and row["Instance_Size"].strip():
        return row["Instance_Size"].strip().lower()
    name = row["Instance"].strip().lower()
    if name.startswith("small"):
        return "small"
    if name.startswith("medium"):
        return "medium"
    if name.startswith("large"):
        return "large"
    return "unknown"


def _plot_size_group(size_rows, size_label, save_path):
    """
    Horizontal grouped bar chart (BAP, BnC, BD) per (α_T, α_max) combo.
    """
    set_style()
    from collections import OrderedDict
    param_keys = sorted(set((r["aT"], r["a_max"]) for r in size_rows))
    n_params = len(param_keys)

    fig, ax = plt.subplots(figsize=(8, max(5, n_params * 1.0)))
    algo_colors = {"BAP": COLOURS["C1"], "BnC": COLOURS["C4"], "BD": COLOURS["C2"]}

    bap_avgs, bnc_avgs, bd_avgs = [], [], []
    for pi, (aT, a_max) in enumerate(param_keys):
        bap_vals = [r["bap_t"] for r in size_rows if r["aT"] == aT and r["a_max"] == a_max]
        bnc_vals = [r["bnc_t"] for r in size_rows if r["aT"] == aT and r["a_max"] == a_max]
        bd_vals  = [r["bd_t"]  for r in size_rows if r["aT"] == aT and r["a_max"] == a_max]
        
        bap_avgs.append(float(np.mean(bap_vals)) if bap_vals else 0.0)
        bnc_avgs.append(float(np.mean(bnc_vals)) if bnc_vals else 0.0)
        bd_avgs.append(float(np.mean(bd_vals)) if bd_vals else 0.0)

    y_idx = np.arange(n_params)
    total_width = 0.8
    bar_w = total_width / 3
    offsets = np.linspace(-total_width/2 + bar_w/2, total_width/2 - bar_w/2, 3)

    ax.barh(y_idx + offsets[0], bap_avgs, height=bar_w, color=algo_colors["BAP"], alpha=0.85, label="BAP", zorder=2)
    ax.barh(y_idx + offsets[1], bnc_avgs, height=bar_w, color=algo_colors["BnC"], alpha=0.85, label="BnC", zorder=2)
    ax.barh(y_idx + offsets[2], bd_avgs,  height=bar_w, color=algo_colors["BD"],  alpha=0.85, label="BD",  zorder=2)

    ax.set_yticks(y_idx)
    ax.set_yticklabels([f"({t:.1f}, {m:.3f})" for t, m in param_keys], fontsize=10)
    
    ax.set_ylim(-0.6, n_params - 0.4)
    ax.invert_yaxis()

    ax.set_ylabel(r"($\alpha_T$, $\alpha_{max}$)", fontsize=12)
    ax.set_xlabel("Run Time (s)", fontsize=13)
    ax.tick_params(axis='x', direction='in', labelsize=11)
    ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(frameon=False, fontsize=10, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches='tight', format=ext)
    plt.close(fig)
    print(f"  Saved → {save_path}.png/.tiff")


def plot_from_csv(csv_path: str, output_dir: str = None):
    """
    Read algorithm_performance.csv and generate per-size plots.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    output_dir : str, optional
        Output directory for plots. If None, uses the CSV's parent directory.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── Read CSV ─────────────────────────────────────────────────────────
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows from {csv_path}")

    # ── Group by instance size ───────────────────────────────────────────
    groups = defaultdict(list)
    for row in rows:
        if not row.get("alpha_T") or not str(row["alpha_T"]).strip():
            continue
        size = _derive_size(row)
        groups[size].append({
            "inst": row["Instance"],
            "aT": float(row["alpha_T"]),
            "a_max": float(row["alpha_max"]),
            "bap_t": float(row["BAP_Time"]),
            "bap_i": int(row["BAP_Iter"]),
            "bnc_t": float(row["BnC_Time"]),
            "bnc_i": int(row["BnC_Iter"]),
            "bd_t": float(row["BD_Time"]),
            "bd_i": int(row["BD_Iter"]),
        })

    # ── Plot per size (all instances combined) ─────────────────────────
    size_order = ["small", "medium", "large"]
    for size in size_order:
        if size not in groups:
            print(f"  No {size} instances found, skipping.")
            continue
        grp = groups[size]
        print(f"\n{size.capitalize()}: {len(grp)} data points")
        save_path = os.path.join(output_dir, f"algorithm_performance_{size}")
        _plot_size_group(grp, size, save_path)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Per-size average times")
    print("=" * 70)
    print(f"{'Size':>8s} {'Count':>6s} {'BAP_T':>8s} {'BnC_T':>8s} {'BD_T':>8s}")
    print("-" * 46)
    for size in size_order:
        if size not in groups:
            continue
        grp = groups[size]
        n = len(grp)
        bap_avg = np.mean([r["bap_t"] for r in grp])
        bnc_avg = np.mean([r["bnc_t"] for r in grp])
        bd_avg  = np.mean([r["bd_t"]  for r in grp])
        print(f"{size:>8s} {n:>6d} {bap_avg:>8.2f}s {bnc_avg:>8.2f}s {bd_avg:>8.2f}s")

    print(f"\nAll plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Plot algorithm performance from existing CSV."
    )
    parser.add_argument( 
        "--csv",
        type=str,
        default="results/alg_performance/algorithm_performance.csv",
        help="Path to algorithm_performance.csv (default: results/alg_performance/algorithm_performance.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: same as CSV directory)",
    )
    args = parser.parse_args()
    plot_from_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
