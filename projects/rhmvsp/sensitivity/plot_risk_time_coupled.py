"""Create coupled alpha_R/alpha_T heatmaps from summary_coupled.csv."""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensitivity.plot_helpers import set_style

COLOURS = ["#19171A", "#455073", "#B6E1ED", "#F1DDB2", "#F2BB77"]


def _plot_heatmap(x_values, y_values, matrix, metric_name, save_path):
    set_style()
    cmap = LinearSegmentedColormap.from_list("set3_heatmap", COLOURS, N=256)
    finite = matrix[np.isfinite(matrix)]
    if not finite.size:
        return
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:
        vmax = vmin + 1e-9
    norm = PowerNorm(gamma=0.55, vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(max(6, len(x_values) * 0.9), max(4, len(y_values) * 0.6)))
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            if np.isfinite(matrix[i, j]):
                ax.scatter(j, i, s=420, c=[cmap(norm(matrix[i, j]))], edgecolors="none")
    ax.set_xticks(range(len(x_values)), [f"{v:.2f}" for v in x_values])
    ax.set_yticks(range(len(y_values)), [f"{v:g}" for v in y_values])
    ax.set_xlabel(r"$\alpha_R$")
    ax.set_ylabel(r"$\alpha_T$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, shrink=0.85, pad=0.02, label=metric_name)
    fig.tight_layout()
    for ext in ("png", "tiff"):
        fig.savefig(f"{save_path}.{ext}", dpi=600, transparent=True, bbox_inches="tight")
    plt.close(fig)


def plot_from_csv(csv_path: str, output_dir: str | None = None) -> None:
    output_dir = output_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") not in {"infeasible", "error"}]
    risk_key = "alpha_R" if rows and rows[0].get("alpha_R") is not None else "alpha_max"
    alpha_r = sorted({float(r[risk_key]) for r in rows})
    alpha_t = sorted({float(r["alpha_T"]) for r in rows})
    if not alpha_r or not alpha_t:
        raise ValueError(f"No plottable rows in {csv_path}")
    specs = [
        (("cost",), "Cost", "heatmap_cost"),
        (("network_risk_reliability", "R_risk"), r"Risk reliability ($R_{risk}$)", "heatmap_R_Risk"),
        (("network_time_reliability", "R_time"), r"Time reliability ($R_{time}$)", "heatmap_R_Time"),
        (("Network_Risk", "Raw_Risk"), "Risk", "heatmap_network_risk"),
        (("Network_Time", "Raw_Time"), "Time (h)", "heatmap_raw_Time"),
    ]
    for fields, label, filename in specs:
        matrix = np.full((len(alpha_t), len(alpha_r)), np.nan)
        for row in rows:
            value = next((row.get(field) for field in fields if row.get(field) not in {"", None}), None)
            if value is not None:
                matrix[alpha_t.index(float(row["alpha_T"])), alpha_r.index(float(row[risk_key]))] = float(value.replace(",", ""))
        _plot_heatmap(alpha_r, alpha_t, matrix, label, os.path.join(output_dir, filename))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to summary_coupled.csv")
    parser.add_argument("--output", help="Output directory; defaults to the CSV directory")
    args = parser.parse_args()
    plot_from_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
