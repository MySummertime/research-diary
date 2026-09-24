"""Publication plots and paired statistics for fixed-plan OOS results."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensitivity.plot_helpers import COLOURS, set_style

METRICS = [
    (("cost",), "Cost (CNY)", "oos_cost", "premium"),
    (("Network_Risk", "network_risk"), "Risk", "oos_risk", "reduction"),
    (("Network_Time", "network_time"), "Time (h)", "oos_time", "reduction"),
    (("network_risk_reliability",), "Risk reliability", "oos_risk_reliability", "improvement"),
    (("network_time_reliability",), "Time reliability", "oos_time_reliability", "improvement"),
]
MR_COLOUR, MW_COLOUR = COLOURS["C2"], COLOURS["C1"]
MR_EDGE, MW_EDGE = "#496C65", "#74402D"


def _number(row, fields):
    for field in fields:
        if row.get(field) not in (None, ""):
            return float(row[field].replace(",", ""))
    return None


def _save(fig, path):
    for extension in ("png", "tiff"):
        fig.savefig(f"{path}.{extension}", dpi=600, transparent=True,
                    bbox_inches="tight", format=extension)
    plt.close(fig)


def _scenario_values(rows, model, fields, label=None):
    result = {}
    for row in rows:
        if row.get("model") != model or (label is not None and row.get("group_label") != label):
            continue
        value = _number(row, fields)
        if value is not None and np.isfinite(value):
            result[row["scenario_id"]] = value
    return result


def _violin_span_at_y(body, y_value, center, fallback_half_width=0.18):
    """Return the violin outline intersections at a given height."""
    vertices = body.get_paths()[0].vertices
    crossings = []
    closed = np.vstack((vertices, vertices[0]))
    for (x1, y1), (x2, y2) in zip(closed[:-1], closed[1:]):
        if np.isclose(y1, y2):
            if np.isclose(y_value, y1):
                crossings.extend((x1, x2))
        elif min(y1, y2) <= y_value <= max(y1, y2):
            fraction = (y_value - y1) / (y2 - y1)
            crossings.append(x1 + fraction * (x2 - x1))
    if len(crossings) >= 2:
        return min(crossings), max(crossings)
    return center - fallback_half_width, center + fallback_half_width


def _absolute_violin(rows, labels, fields, y_label, path):
    datasets = [_scenario_values(rows, "MW", fields)]
    datasets.extend(_scenario_values(rows, "MR", fields, label) for label in labels)
    arrays = [np.asarray([values[k] for k in sorted(values, key=int)], dtype=float)
              for values in datasets]
    displayed = [(label, values, colour, edge)
                 for label, values, colour, edge
                 in zip(["MW", *labels], arrays,
                        [MW_COLOUR] + [MR_COLOUR] * len(labels),
                        [MW_EDGE] + [MR_EDGE] * len(labels))
                 if len(values) >= 2]
    if not displayed:
        raise ValueError(f"No valid observations for {y_label}")

    set_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    displayed_labels, arrays, colours, edges = map(list, zip(*displayed))
    positions = np.r_[0.0, 1.35 + np.arange(len(arrays) - 1)] if displayed_labels[0] == "MW" else np.arange(len(arrays))
    violin = ax.violinplot(arrays, positions=positions, widths=0.72,
                           showmeans=False, showmedians=False, showextrema=False)
    for body, colour, edge in zip(violin["bodies"], colours, edges):
        body.set_facecolor(colour)
        body.set_edgecolor(edge)
        body.set_linewidth(1.0)
        body.set_alpha(0.95)
    rng = np.random.default_rng(7)
    for position, values, edge, body in zip(positions, arrays, edges, violin["bodies"]):
        ax.scatter(position + rng.uniform(-0.20, 0.20, len(values)), values,
                   s=6, color=edge, alpha=0.55, linewidths=0, zorder=3)
        median = float(np.median(values))
        median_left, median_right = _violin_span_at_y(body, median, position)
        ax.plot([median_left, median_right], [median, median],
                color="black", linewidth=0.9, zorder=4)
    ax.set_xticks(positions, displayed_labels)
    ax.set_xlabel(r"$(\alpha_T,\;\alpha_R)$")
    ax.set_ylabel(y_label)
    ax.legend(handles=[Patch(facecolor=MR_COLOUR, edgecolor=MR_EDGE, label="MR"),
                       Patch(facecolor=MW_COLOUR, edgecolor=MW_EDGE, label="MW")], frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    _save(fig, path)


def _paired_rank_biserial(mr, mw):
    differences = np.asarray(mr) - np.asarray(mw)
    differences = differences[~np.isclose(differences, 0.0)]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive, negative = ranks[differences > 0].sum(), ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def _summary_rows(rows, labels):
    output = []
    for fields, _, filename, change_type in METRICS:
        mw = _scenario_values(rows, "MW", fields)
        for label in labels:
            mr = _scenario_values(rows, "MR", fields, label)
            common = sorted(set(mr) & set(mw), key=int)
            parameter_row = next(row for row in rows if row.get("model") == "MR"
                                 and row.get("group_label") == label)
            if not common:
                output.append({
                    "metric": filename.removeprefix("oos_"),
                    "alpha_T": parameter_row.get("alpha_T", ""),
                    "alpha_R": parameter_row.get("alpha_R", ""),
                    "status": "No feasible transport plan", "n_pairs": 0,
                    "mr_median": "", "mw_median": "", "mr_q1": "",
                    "mr_q3": "", "mr_iqr": "", "mw_q1": "", "mw_q3": "",
                    "mw_iqr": "", "median_paired_gap": "",
                    "relative_change_pct": "", "change_type": change_type,
                    "wilcoxon_p_value": "", "rank_biserial_effect_size": "",
                })
                continue
            mr_values = np.asarray([mr[key] for key in common])
            mw_values = np.asarray([mw[key] for key in common])
            mr_q1, mr_median, mr_q3 = np.percentile(mr_values, [25, 50, 75])
            mw_q1, mw_median, mw_q3 = np.percentile(mw_values, [25, 50, 75])
            numerator = mw_median - mr_median if change_type == "reduction" else mr_median - mw_median
            relative = 100.0 * numerator / abs(mw_median) if mw_median else np.nan
            differences = mr_values - mw_values
            try:
                p_value = (1.0 if np.allclose(differences, 0.0)
                           else float(wilcoxon(mr_values, mw_values).pvalue))
            except ValueError:
                p_value = 1.0
            output.append({
                "metric": filename.removeprefix("oos_"),
                "alpha_T": parameter_row.get("alpha_T", ""),
                "alpha_R": parameter_row.get("alpha_R", ""),
                "status": "evaluated", "n_pairs": len(common),
                "mr_median": mr_median, "mw_median": mw_median,
                "mr_q1": mr_q1, "mr_q3": mr_q3, "mr_iqr": mr_q3 - mr_q1,
                "mw_q1": mw_q1, "mw_q3": mw_q3, "mw_iqr": mw_q3 - mw_q1,
                "median_paired_gap": float(np.median(mr_values - mw_values)),
                "relative_change_pct": relative, "change_type": change_type,
                "wilcoxon_p_value": p_value,
                "rank_biserial_effect_size": _paired_rank_biserial(mr_values, mw_values),
            })
    return output


def _write_summary(rows, labels, path):
    summary = _summary_rows(rows, labels)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def plot_from_csv(csv_path, output_dir=None):
    output_dir = output_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels = list(dict.fromkeys(row["group_label"] for row in rows if row.get("model") == "MR"))
    if not labels:
        raise ValueError(f"No MR rows in {csv_path}")
    for fields, y_label, filename, _ in METRICS:
        _absolute_violin(rows, labels, fields, y_label, os.path.join(output_dir, filename))
    _write_summary(rows, labels, os.path.join(output_dir, "out_of_sample_summary.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    plot_from_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
