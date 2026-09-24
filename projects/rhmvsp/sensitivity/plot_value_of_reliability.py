"""
Standalone plotting script for Value of Reliability experiment.

Reads an existing value_of_reliability.csv and generates three independent
horizontal dumbbell plots comparing MR and MW for cost, risk, and time.
Each connecting line is annotated with the percentage difference of MR
relative to MW.

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


def _float_column(row: dict, *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"Missing any of columns: {', '.join(names)}")


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
        f"({r['alpha_T'].strip()}, {r['alpha_R'].strip()})"
        for r in rows
    ]
    mr_costs   = [_parse_cost(r["MR_Cost"]) or 0 for r in rows]
    mw_costs   = [_parse_cost(r["MW_Cost"]) or 0 for r in rows]
    mr_network_risk = [_float_column(r, "MR_NetworkRisk", "MR_RawRisk") for r in rows]
    mw_network_risk = [_float_column(r, "MW_NetworkRisk", "MW_RawRisk") for r in rows]
    mr_network_time = [_float_column(r, "MR_NetworkTime", "MR_RawTime") for r in rows]
    mw_network_time = [_float_column(r, "MW_NetworkTime", "MW_RawTime") for r in rows]

    # Every metric uses MW as denominator and MR-MW for the sign.
    def _gap(mr, mw):
        if mr is not None and mw is not None and mw != 0:
            return (mr - mw) / abs(mw) * 100
        return 0.0
    gap_cost = [_gap(mr_costs[i], mw_costs[i]) for i in range(len(x_labels))]
    gap_network_risk = [
        _gap(mr_network_risk[i], mw_network_risk[i])
        for i in range(len(x_labels))
    ]
    gap_network_time = [
        _gap(mr_network_time[i], mw_network_time[i])
        for i in range(len(x_labels))
    ]
    # ── Three independent horizontal dumbbell plots ─────────────────────
    def _dumbbell(mr_data, mw_data, gaps, x_label, filename, scientific=False):
        """Compare MR designs with one shared no-reliability MW baseline."""
        set_style()
        mr = np.asarray(mr_data, dtype=float)
        mw = np.asarray(mw_data, dtype=float)
        y_pos = np.arange(len(x_labels))

        mw_baseline = float(mw[0])
        if not np.allclose(mw, mw_baseline, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"MW must be a common baseline for {filename}; observed values: {mw.tolist()}"
            )

        fig, ax = plt.subplots(figsize=(7.4, 4.8))

        ax.axvline(mw_baseline, color=C_MW, linestyle="--", linewidth=1.5,
                   label="MW", zorder=1)
        for y, mr_value, gap in zip(y_pos, mr, gaps):
            ax.plot([mr_value, mw_baseline], [y, y], color="#8A8A8A",
                    linewidth=1.25, zorder=2)
            ax.text(
                (mr_value + mw_baseline) / 2.0, y - 0.16, f"{gap:+.2f}%",
                ha="center", va="bottom", fontsize=8.5,
                color="#4A4A4A", zorder=4,
            )
        ax.scatter(mr, y_pos, color=C_MR, marker="o", s=52,
                   edgecolor="white", linewidth=0.6, label="MR", zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(x_labels, fontsize=10)
        ax.set_ylim(len(x_labels) - 0.5, -0.5)
        ax.set_ylabel(r"$(\alpha_T,\;\alpha_R)$", fontsize=13)
        ax.set_xlabel(x_label, fontsize=13)
        ax.tick_params(axis="both", direction="in", labelsize=10)
        ax.xaxis.grid(True, linestyle="--", color="gray", alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc="lower center",
                  bbox_to_anchor=(0.5, 1.02), ncol=2)

        # Tight range that preserves the absolute scale while exposing pairs.
        bounds = np.concatenate([mr, mw])
        span = float(bounds.max() - bounds.min())
        margin = span * 0.12 if span > 0 else max(abs(float(bounds[0])) * 0.05, 1.0)
        ax.set_xlim(float(bounds.min() - margin), float(bounds.max() + margin))
        if scientific:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),
                                useMathText=False)

        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        for ext in ["png", "tiff"]:
            fig.savefig(f"{save_path}.{ext}", dpi=600, transparent=True,
                        bbox_inches="tight", format=ext)
        plt.close(fig)
        print(f"  Saved → {save_path}.png/.tiff")

    _dumbbell(mr_costs, mw_costs, gap_cost, "Cost", "value_of_reliability_cost",
              scientific=True)
    _dumbbell(mr_network_risk, mw_network_risk, gap_network_risk, "Risk",
              "value_of_reliability_risk")
    _dumbbell(mr_network_time, mw_network_time, gap_network_time, "Time (h)",
              "value_of_reliability_time")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALUE OF RELIABILITY: Average metrics across (α_T, α_R) grid")
    print("=" * 70)
    print(f"{'Metric':>15s} {'MR (avg)':>12s} {'MW (avg)':>12s} {'Gap (avg)':>10s}")
    print("-" * 55)
    avg_mr_cost = np.mean(mr_costs)
    avg_mw_cost = np.mean(mw_costs)
    avg_gap_cost = np.mean(gap_cost)
    print(f"{'Cost (CNY)':>15s} {avg_mr_cost:>12,.2f} {avg_mw_cost:>12,.2f} {avg_gap_cost:>10.2f}%")
    avg_mr_rr = np.mean(mr_network_risk)
    avg_mw_rr = np.mean(mw_network_risk)
    avg_gap_rr = np.mean(gap_network_risk)
    print(f"{'Accident Belief':>15s} {avg_mr_rr:>12.8f} {avg_mw_rr:>12.8f} {avg_gap_rr:>10.2f}%")
    avg_mr_rt = np.mean(mr_network_time)
    avg_mw_rt = np.mean(mw_network_time)
    avg_gap_rt = np.mean(gap_network_time)
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
