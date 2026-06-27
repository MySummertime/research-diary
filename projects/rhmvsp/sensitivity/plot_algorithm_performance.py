"""
Standalone plotting script for Algorithm Performance experiment.

Reads an existing algorithm_performance.csv and generates per-instance-size
grouped bar charts (BAP vs BAP-NoStab vs Compact MIP).

Usage:
    conda run -n rhmvsp python sensitivity/plot_algorithm_performance.py
    conda run -n rhmvsp python sensitivity/plot_algorithm_performance.py --csv results/alg_perf_20260626_125254/algorithm_performance.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MPLRect
from matplotlib.lines import Line2D as MPLLine
from matplotlib.ticker import MaxNLocator
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
    Create a grouped bar chart for one instance size group.

    Parameters
    ----------
    size_rows : list of dict
        Each dict has keys: inst, aT, a_max, bap_t, bap_i, nostab_t,
        nostab_i, cmip_t, cmip_i
    size_label : str
        Label for this size group (e.g. "Small", "Medium", "Large")
    save_path : str
        Output path (without extension) for .png and .tiff
    """
    set_style()
    n = len(size_rows)
    x_idx = np.arange(n)
    n_algos = 3
    total_width = 0.8
    bar_w = total_width / n_algos
    offsets = np.linspace(-total_width / 2 + bar_w / 2,
                          total_width / 2 - bar_w / 2, n_algos)

    bap_t = np.array([r["bap_t"] for r in size_rows])
    bap_i = np.array([r["bap_i"] for r in size_rows])
    ns_t = np.array([r["nostab_t"] for r in size_rows])
    ns_i = np.array([r["nostab_i"] for r in size_rows])
    cmip_t = np.array([r["cmip_t"] for r in size_rows])
    cmip_i = np.array([r["cmip_i"] for r in size_rows])

    algo_data = [
        (bap_t, bap_i, COLOURS["C1"], "BAP (Optimized)"),
        (ns_t, ns_i, COLOURS["C4"], "BAP (Standard)"),
        (cmip_t, cmip_i, COLOURS["C2"], "Compact MIP"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    for i, (t_vals, i_vals, c, lbl) in enumerate(algo_data):
        ax.bar(x_idx + offsets[i], t_vals, width=bar_w,
               color=c, alpha=0.85, label=f"{lbl} (Time)", zorder=2)
        marker = 'o' if i == 0 else 's' if i == 1 else '^'
        ax2.plot(x_idx + offsets[i], i_vals, marker,
                 color=c, markersize=7, linewidth=0,
                 label=f"{lbl} (Iter)", zorder=5)

    ax.set_ylabel("Run Time (s)", fontsize=13)
    ax.tick_params(axis='y', direction='in', labelsize=11)
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.4)
    ax.spines['top'].set_visible(False)

    ax2.set_ylabel("Iterations", fontsize=13)
    ax2.tick_params(axis='y', direction='in', labelsize=11)
    ax2.spines['top'].set_visible(False)
    # Integer ticks on iteration axis
    all_iters = np.concatenate([bap_i, ns_i, cmip_i])
    all_iters = all_iters[~np.isnan(all_iters)]
    max_iter = int(np.ceil(np.nanmax(all_iters))) if len(all_iters) > 0 else 5
    ax2.set_ylim(0, max_iter + 1)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    # Legend above figure
    handles = []
    markers = ['o', 's', '^']
    for i, (_, _, c, lbl) in enumerate(algo_data):
        handles.append(MPLRect((0, 0), 1, 1, color=c, alpha=0.85,
                               label=f"{lbl} (Time)"))
    for i, (_, _, c, lbl) in enumerate(algo_data):
        handles.append(MPLLine([0], [0], marker=markers[i], color=c,
                               linestyle='None', markersize=7,
                               label=f"{lbl} (Iter)"))
    ax.legend(handles=handles, frameon=False, fontsize=7,
              loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)

    # X-axis labels: "Instance (α_T, α_max)"
    param_labels = [
        f"{r['inst']}\n({r['aT']}, {r['a_max']})" for r in size_rows
    ]
    ax.set_xticks(x_idx)
    ax.set_xticklabels(param_labels, fontsize=9)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xlabel(f"Instance Size: {size_label.capitalize()}", fontsize=12)

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
        size = _derive_size(row)
        groups[size].append({
            "inst": inst,
            "aT": float(row["alpha_T"]),
            "a_max": float(row["alpha_max"]),
            "bap_t": float(row["BAP_Time"]),
            "bap_i": int(row["BAP_Iter"]),
            "nostab_t": float(row["NoStab_Time"]),
            "nostab_i": int(row["NoStab_Iter"]),
            "cmip_t": float(row["CMIP_Time"]),
            "cmip_i": int(row["CMIP_Iter"]),
        })

    # ── Plot per size ────────────────────────────────────────────────────
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
    print(f"{'Size':>8s} {'Count':>6s} {'BAP_T':>8s} {'NS_T':>8s} {'CMIP_T':>8s}")
    print("-" * 46)
    for size in size_order:
        if size not in groups:
            continue
        grp = groups[size]
        n = len(grp)
        bap_avg = np.mean([r["bap_t"] for r in grp])
        ns_avg = np.mean([r["nostab_t"] for r in grp])
        cmip_avg = np.mean([r["cmip_t"] for r in grp])
        print(f"{size:>8s} {n:>6d} {bap_avg:>8.2f}s {ns_avg:>8.2f}s {cmip_avg:>8.2f}s")

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
