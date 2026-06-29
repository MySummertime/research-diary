"""
Standalone plotting script for Value of Reliability experiment.

Reads an existing value_of_reliability.csv and generates:
  1. A 3-panel grouped bar chart: MR vs MW for Cost, Raw Risk, Raw Time
  2. A lollipop gap chart: stem + circle markers, size = |gap|, color = sign

Usage:
    conda run -n rhmvsp python sensitivity/plot_value_of_reliability.py
    conda run -n rhmvsp python sensitivity/plot_value_of_reliability.py --csv results/value_of_reliability/value_of_reliability.csv
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensitivity.plot_helpers import set_style, COLOURS

# ── Colour constants ────────────────────────────────────────────────────────
C_MR  = COLOURS["C1"]  # Warm Terracotta — Model with Reliability
C_MW  = COLOURS["C2"]  # Celadon Green   — Model Without reliability


def _parse_cost(val: str):
    """Parse a cost string like '5,175,941' or 'N/A' to float or None."""
    if val is None or val.strip().upper() == "N/A":
        return None
    return float(val.replace(",", ""))


def plot_from_csv(csv_path: str, output_dir: str = None):
    """
    Read value_of_reliability.csv and generate plots.

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

    # ── Parse ────────────────────────────────────────────────────────────
    x_labels = [
        f"({r['alpha_T'].strip()}, {r['alpha_max'].strip()})" for r in rows
    ]
    x_idx = np.arange(len(x_labels))

    mr_costs   = [_parse_cost(r["MR_Cost"]) or 0 for r in rows]
    mw_costs   = [_parse_cost(r["MW_Cost"]) or 0 for r in rows]
    mr_raw_risk = [float(r["MR_RawRisk"]) for r in rows]
    mw_raw_risk = [float(r["MW_RawRisk"]) for r in rows]
    mr_raw_time = [float(r["MR_RawTime"]) for r in rows]
    mw_raw_time = [float(r["MW_RawTime"]) for r in rows]

    # Recompute gaps from raw data (MR = baseline), regardless of CSV column
    def _gap(mr, mw):
        if mr is not None and mw is not None and mr != 0:
            return (mw - mr) / abs(mr) * 100
        return 0.0
    gap_cost     = [_gap(mr_costs[i], mw_costs[i]) for i in range(len(x_labels))]
    gap_raw_risk = [_gap(mr_raw_risk[i], mw_raw_risk[i]) for i in range(len(x_labels))]
    gap_raw_time = [_gap(mr_raw_time[i], mw_raw_time[i]) for i in range(len(x_labels))]
    gap_r_risk   = [_gap(float(r["MR_R_risk"] or 0), float(r["MW_R_risk"] or 0)) for r in rows]
    gap_r_time   = [_gap(float(r["MR_R_time"] or 0), float(r["MW_R_time"] or 0)) for r in rows]

    # ── Plot 1: 3-panel grouped bar (Cost, RawRisk, RawTime) ────────────
    set_style()
    bar_width = 0.3

    fig, axes = plt.subplots(3, 1, figsize=(8, 10),
                              gridspec_kw={'hspace': 0.45},
                              constrained_layout=True)

    panels = [
        (mr_costs,  mw_costs,  "Total Cost (CNY)"),
        (mr_raw_risk, mw_raw_risk,  r"Raw Risk"),
        (mr_raw_time, mw_raw_time,  r"Raw Time (h)"),
    ]

    for idx, (mr_data, mw_data, ylabel) in enumerate(panels):
        ax = axes[idx]
        ax.bar(x_idx - bar_width / 2, mr_data, width=bar_width,
               color=C_MR, alpha=0.8, label="MR")
        ax.bar(x_idx + bar_width / 2, mw_data, width=bar_width,
               color=C_MW, alpha=0.8, label="MW")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels if idx == 2 else [], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='y', direction='in', labelsize=10)
        ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc='lower center',
                  bbox_to_anchor=(0.5, 1.02), ncol=2)

    axes[2].set_xlabel(r"($\alpha_T$, $\alpha_{max}$)", fontsize=12)

    base_name = os.path.join(output_dir, "value_of_reliability")
    for ext in ["png", "tiff"]:
        fig.savefig(f"{base_name}.{ext}", dpi=600, bbox_inches='tight', format=ext)
    plt.close(fig)
    print(f"  Saved → {base_name}.png/.tiff")

    # ── Plot 2: Gap lollipop chart ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    n_gaps = len(x_labels)

    # One metric per lollipop "track", separated vertically by metric.
    # Cost, Raw Risk, Raw Time, R_risk, R_time
    gap_names  = ["Cost", "Raw Risk", "Raw Time", "$R_{risk}$", "$R_{time}$"]
    gap_all    = [gap_cost, gap_raw_risk, gap_raw_time, gap_r_risk, gap_r_time]

    # Build a 2D grid: rows = metrics, columns = (α_T, α_max)
    n_metrics = len(gap_names)
    gap_arr = np.array(gap_all)  # shape (n_metrics, n_gaps)

    # ── Plot 2: Single lollipop chart (5 metrics per (α_T, α_max) cluster) ─
    gap_names  = ["Cost", "Raw Risk", "Raw Time", "$R_{risk}$", "$R_{time}$"]
    gap_all    = [gap_cost, gap_raw_risk, gap_raw_time, gap_r_risk, gap_r_time]
    gap_arr    = np.array(gap_all)  # shape (5, 6)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    n_metrics = len(gap_names)
    # Colours from palette, one per metric
    metric_colors = [COLOURS["C1"], COLOURS["C2"], COLOURS["C3"],
                     COLOURS["C4"], COLOURS["C5"]]

    # Cluster layout: 6 clusters, each with 5 lollipops
    cluster_w = 0.7
    bar_w2    = cluster_w / n_metrics
    offsets2  = np.linspace(-cluster_w/2 + bar_w2/2,
                             cluster_w/2 - bar_w2/2, n_metrics)

    for mi in range(n_metrics):
        vals = gap_arr[mi]
        x_pos = np.arange(n_gaps) + offsets2[mi]
        # Stem lines
        for gi in range(n_gaps):
            ax2.plot([x_pos[gi], x_pos[gi]], [0, vals[gi]],
                     color=metric_colors[mi], linewidth=0.8, alpha=0.4,
                     solid_capstyle='round')
        # Lollipop heads (size ∝ |gap|, min s=60)
        sizes = np.maximum(60, np.abs(vals) * 10)
        ax2.scatter(x_pos, vals, s=sizes, c=metric_colors[mi],
                    alpha=0.85, edgecolors='white', linewidth=0.3,
                    label=gap_names[mi], zorder=3)

    ax2.axhline(0, color='black', linewidth=0.7)
    ax2.set_xticks(np.arange(n_gaps))
    ax2.set_xticklabels(x_labels, fontsize=9)
    ax2.set_ylabel("Gap (%)  (MW - MR) / |MR|", fontsize=12)
    ax2.tick_params(axis='y', direction='in', labelsize=10)
    ax2.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel(r"($\alpha_T$, $\alpha_{max}$)", fontsize=11)
    ax2.legend(frameon=False, fontsize=8, loc='lower center',
               bbox_to_anchor=(0.5, 1.02), ncol=5)

    gaps_name = os.path.join(output_dir, "value_of_reliability_gaps")
    for ext in ["png", "tiff"]:
        fig2.savefig(f"{gaps_name}.{ext}", dpi=600, bbox_inches='tight', format=ext)
    plt.close(fig2)
    print(f"  Saved → {gaps_name}.png/.tiff")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALUE OF RELIABILITY: Average metrics across (α_T, α_max) grid")
    print("=" * 70)
    print(f"{'Metric':>15s} {'MR (avg)':>12s} {'MW (avg)':>12s} {'Gap (avg)':>10s}")
    print("-" * 55)
    avg_mr_cost = np.mean(mr_costs)
    avg_mw_cost = np.mean(mw_costs)
    avg_gap_cost = np.mean(gap_cost)
    print(f"{'Cost (CNY)':>15s} {avg_mr_cost:>12,.0f} {avg_mw_cost:>12,.0f} {avg_gap_cost:>10.2f}%")
    avg_mr_rr = np.mean(mr_raw_risk)
    avg_mw_rr = np.mean(mw_raw_risk)
    avg_gap_rr = np.mean(gap_raw_risk)
    print(f"{'Raw Risk':>15s} {avg_mr_rr:>12.2f} {avg_mw_rr:>12.2f} {avg_gap_rr:>10.2f}%")
    avg_mr_rt = np.mean(mr_raw_time)
    avg_mw_rt = np.mean(mw_raw_time)
    avg_gap_rt = np.mean(gap_raw_time)
    print(f"{'Raw Time (h)':>15s} {avg_mr_rt:>12.2f} {avg_mw_rt:>12.2f} {avg_gap_rt:>10.2f}%")

    print(f"\nAll plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Plot value of reliability from existing CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/value_of_reliability/value_of_reliability.csv",
        help="Path to value_of_reliability.csv",
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
