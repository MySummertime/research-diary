"""Create coupled alpha_R/alpha_T heatmaps from summary_coupled.csv."""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensitivity.plot_helpers import set_style

# High-contrast extension of the paper palette: deep celadon -> warm ivory ->
# terracotta.  Interior stops keep the pale midpoint narrow so cells remain
# legible at final print size.
HEATMAP_COLOURS = [
    (0.00, "#527F76"),
    (0.28, "#94C1B7"),
    (0.47, "#D4E2E0"),
    (0.50, "#F3EFE6"),
    (0.53, "#E8D7C9"),
    (0.76, "#D58A74"),
    (1.00, "#C26B48"),
]


def _plot_heatmap(x_values, y_values, matrix, metric_name, save_path):
    set_style()
    cmap = LinearSegmentedColormap.from_list(
        "paper_bluecyan_ivory_terracotta", HEATMAP_COLOURS, N=256
    )
    finite = matrix[np.isfinite(matrix)]
    if not finite.size:
        return
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:
        vmax = vmin + 1e-9
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(max(6, len(x_values) * 0.9), max(4, len(y_values) * 0.6)))
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            if np.isfinite(matrix[i, j]):
                cell = FancyBboxPatch(
                    (j - 0.42, i - 0.42), 0.84, 0.84,
                    boxstyle="round,pad=0,rounding_size=0.14",
                    facecolor=cmap(norm(matrix[i, j])),
                    edgecolor="none",
                    linewidth=0,
                    alpha=1.0,
                )
                ax.add_patch(cell)
    ax.set_xlim(-0.55, len(x_values) - 0.45)
    ax.set_ylim(-0.55, len(y_values) - 0.45)
    # Preserve one display unit in x for one display unit in y so the
    # equal-width/equal-height patches render as true rounded squares.
    ax.set_aspect("equal", adjustable="box")
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
        (("network_risk_reliability", "R_risk"), "Risk reliability", "heatmap_R_Risk"),
        (("network_time_reliability", "R_time"), "Time reliability", "heatmap_R_Time"),
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
