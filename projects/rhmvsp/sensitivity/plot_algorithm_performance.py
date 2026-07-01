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

# Set font to Times New Roman globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Strict colour palette from docs/colour set2.txt
COLORS = {
    "terracotta": "#C26B48",
    "celadon": "#94C1B7",
    "sage": "#ABBCAA",
    "sand": "#DDBF91",
    "lime": "#DDE4B6"
}

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensitivity.plot_helpers import set_style, COLOURS





def plot_radar(csv_path: str, output_dir: str = None):
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read Data
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("alpha_T") or not str(row["alpha_T"]).strip():
                continue
            rows.append(row)

    # Group by instance size to maintain order (Small -> Medium -> Large)
    size_order = ["small", "medium", "large"]
    grouped = {s: [] for s in size_order}
    for row in rows:
        size = row["Instance_Size"].strip().lower() if "Instance_Size" in row and row["Instance_Size"].strip() else row["Instance"].strip().lower().split("_")[0]
        if size in grouped:
            grouped[size].append(row)
        else:
            grouped["small"].append(row) # Fallback

    for size in size_order:
        size_rows = grouped[size]
        
        if not size_rows:
            continue
            
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            
        n_inst = len(size_rows)
        angles = np.linspace(0, 2 * np.pi, n_inst, endpoint=False)
        angles_closed = np.concatenate((angles, [angles[0]]))

        bap_ratios = []
        bnc_ratios = []
        bd_ratios = []
        labels = []

        for row in size_rows:
            inst_num = row['Instance'].split('_')[1].replace('inst', '')
            labels.append(f"i{inst_num}({float(row['alpha_T']):.1f}, {float(row['alpha_max']):.3f})")
            
            bap_t = float(row["BAP_Time"]) if row["BAP_Time"] != "N/A" else np.nan
            bnc_t = float(row["BnC_Time"]) if row["BnC_Time"] != "N/A" else np.nan
            bd_t = float(row["BD_Time"]) if row["BD_Time"] not in ("N/A", "skipped") else np.nan

            if np.isnan(bap_t) or bap_t <= 0:
                bap_ratios.append(np.nan)
                bnc_ratios.append(np.nan)
                bd_ratios.append(np.nan)
            else:
                bap_ratios.append(1.0)
                bnc_ratios.append(bnc_t / bap_t if not np.isnan(bnc_t) else np.nan)
                bd_ratios.append(bd_t / bap_t if not np.isnan(bd_t) else np.nan)

        bap_ratios_c = np.concatenate((bap_ratios, [bap_ratios[0]]))
        bnc_ratios_c = np.concatenate((bnc_ratios, [bnc_ratios[0]]))
        bd_ratios_c = np.concatenate((bd_ratios, [bd_ratios[0]]))

        # Log scale
        ax.set_yscale('log')
        all_vals = [v for v in np.concatenate((bap_ratios_c, bnc_ratios_c, bd_ratios_c)) if not np.isnan(v) and v > 0]
        min_v = np.min(all_vals)
        max_v = np.max(all_vals)
        ax.set_ylim(bottom=min_v * 0.4, top=max_v * 2.5)
        
        # Select appropriate ticks
        candidate_ticks = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        ticks = [t for t in candidate_ticks if min_v * 0.4 <= t <= max_v * 2.5]
        if 1 not in ticks:
            ticks.append(1)
            ticks = sorted(ticks)
            
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t}x" for t in ticks], color="gray", size=9)

        # Draw lines
        ax.plot(angles_closed, bap_ratios_c, color=COLORS["terracotta"], linewidth=2, linestyle='-', label='BAP')
        ax.plot(angles_closed, bnc_ratios_c, color=COLORS["celadon"], linewidth=2, marker='s', markersize=6, label='BnC')
        ax.plot(angles_closed, bd_ratios_c, color=COLORS["sage"], linewidth=2, marker='^', markersize=6, label='BD')

        # Angles for ticks
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=10)
        
        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
        ax.spines['polar'].set_visible(False)

        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=12)

        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"algorithm_performance_radar_{size}")
        for ext in ["png", "tiff"]:
            fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches='tight', format=ext)
        plt.close(fig)
        print(f"Saved radar chart to {save_path}.png")


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
    plot_radar(args.csv, args.output)


if __name__ == "__main__":
    main()
