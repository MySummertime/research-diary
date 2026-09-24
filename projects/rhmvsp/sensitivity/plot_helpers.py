"""
Shared plotting utilities for all sensitivity experiments.

Enforces:
  - Times New Roman font family
  - No figure titles
  - Colour palette from docs/colour set2.txt (5 colours)
"""
import os
import math
from decimal import Decimal, InvalidOperation
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter, MultipleLocator

from config import ColorPalette, PlotProperties

# ── Colour palette (colour set2.txt) ──────────────────────
COLOURS = {
    "C1": "#C26B48",  # Warm Terracotta
    "C2": "#94C1B7",  # Celadon Green
    "C3": "#ABBCAA",  # Sage Green
    "C4": "#DDBF91",  # Warm Sand
    "C5": "#DDE4B6",  # Pale Lime
}

# ── Adaptive tick formatting helpers ──────────────────────


def _fmt_values(values):
    """Format every x-axis value at the finest precision used by the grid."""
    digits = max((_decimal_places(value) for value in values), default=0)
    return [f"{float(value):.{digits}f}" for value in values]


def _decimal_places(value):
    """Return the decimal precision visible in one input value."""
    try:
        exponent = Decimal(str(value)).as_tuple().exponent
    except (InvalidOperation, ValueError):
        return 0
    return max(0, -exponent)


def _nice_tick_step(raw_step):
    """Round a positive interval upward to a 1/2/5 × 10^n step."""
    exponent = math.floor(math.log10(raw_step))
    scale = 10.0 ** exponent
    fraction = raw_step / scale
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if fraction <= candidate:
            return candidate * scale
    return 10.0 * scale


def _configure_adaptive_y_axis(ax, values):
    """Configure one y-axis independently from its values and precision.

    Python floating-point artefacts (for example ``0.8175199999999999``)
    must not be interpreted as requested plotting precision.  Doing so can
    create a locator step near machine epsilon, for which Matplotlib tries to
    allocate an enormous tick array during layout.
    """
    pairs = [
        (value, float(value)) for value in values
        if value is not None and np.isfinite(float(value))
    ]
    if not pairs:
        ax.set_ylim(0.0, 1.0)
        return 0.2, 1

    numeric = np.asarray([number for _, number in pairs], dtype=float)
    # Eight decimal places covers the reliability metrics used by this project
    # while discarding numerical noise introduced by solver arithmetic.
    decimals = min(8, max(_decimal_places(value) for value, _ in pairs))
    precision_step = 10.0 ** (-decimals)
    data_min = float(numeric.min())
    data_max = float(numeric.max())
    data_range = data_max - data_min
    target_intervals = PlotProperties.SENSITIVITY_Y_TARGET_INTERVALS

    # Values that differ by less than one displayed unit are visually
    # indistinguishable.  Treat them as a constant series instead of deriving
    # a machine-precision locator interval from their tiny numerical spread.
    if data_range < precision_step:
        data_min = data_max = float(np.mean(numeric))
        data_range = 0.0

    raw_step = data_range / target_intervals if data_range > 0 else precision_step
    tick_step = max(_nice_tick_step(raw_step), precision_step)

    if data_range > 0:
        lower = math.floor(data_min / tick_step) * tick_step
        upper = math.ceil(data_max / tick_step) * tick_step
        if math.isclose(lower, data_min, rel_tol=0.0, abs_tol=precision_step / 10.0):
            lower -= tick_step
        if math.isclose(upper, data_max, rel_tol=0.0, abs_tol=precision_step / 10.0):
            upper += tick_step
    else:
        half_intervals = max(2, math.ceil(target_intervals / 2))
        lower = data_min - half_intervals * tick_step
        upper = data_max + half_intervals * tick_step

    zero_tolerance = precision_step / 2.0
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda tick, _pos: f"{0.0 if abs(tick) < zero_tolerance else tick:.{decimals}f}"
    ))
    return tick_step, decimals


# ── Style setup ───────────────────────────────────────────

def set_style():
    """Apply Times New Roman, no-title defaults for all subsequent plots."""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 0.8
    # Ensure no figure-level title
    plt.rcParams['figure.titlesize'] = 'medium'


# ── Dual Y-axis line chart ────────────────────────────────

def dual_y_axis(x_label: str, x_vals, r_risk_vals, r_time_vals,
                save_path: str,
                c1: str = ColorPalette.SENSITIVITY_RISK_LINE,
                c2: str = ColorPalette.SENSITIVITY_TIME_LINE,
                y1_label: str = 'Risk reliability',
                y2_label: str = 'Time reliability',
                y1_title: str = 'Risk reliability',
                y2_title: str = 'Time reliability'):
    """
    Dual Y-axis line chart: R_risk (left), R_time (right) vs x_vals.
    Uses categorical x-positions + adaptive decimal labels so that
    non-uniform float grids (e.g. alpha_R) render correctly.
    Saves as PNG and TIFF at 600 dpi next to save_path (without extension).
    """
    set_style()
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Categorical x positions (robust for non-uniform float grids)
    x_pos = np.arange(len(x_vals))
    x_labels = _fmt_values(x_vals)

    # Left y-axis: R_risk
    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel(y1_title, fontsize=14, color=c1)
    line1, = ax1.plot(
        x_pos, r_risk_vals, color=c1,
        marker=PlotProperties.SENSITIVITY_RISK_MARKER,
        markersize=PlotProperties.SENSITIVITY_RELIABILITY_MARKER_SIZE,
        linewidth=PlotProperties.SENSITIVITY_RELIABILITY_LINE_WIDTH,
        linestyle=PlotProperties.SENSITIVITY_RISK_LINE_STYLE,
        label=y1_label, zorder=5,
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=11)
    ax1.tick_params(axis='x', direction='in', labelsize=11)
    ax1.tick_params(axis='y', direction='in', labelcolor=c1, labelsize=11)
    ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    _configure_adaptive_y_axis(ax1, r_risk_vals)
    ax1.set_xlim(-0.5, len(x_vals) - 0.5)
    ax1.spines['top'].set_visible(False)

    # Right y-axis: R_time
    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_title, fontsize=14, color=c2)
    line2, = ax2.plot(
        x_pos, r_time_vals, color=c2,
        marker=PlotProperties.SENSITIVITY_TIME_MARKER,
        markersize=PlotProperties.SENSITIVITY_RELIABILITY_MARKER_SIZE,
        linewidth=PlotProperties.SENSITIVITY_RELIABILITY_LINE_WIDTH,
        linestyle=PlotProperties.SENSITIVITY_TIME_LINE_STYLE,
        label=y2_label, zorder=4,
    )
    ax2.tick_params(axis='y', direction='in', labelcolor=c2, labelsize=11)
    ax2.spines['top'].set_visible(False)

    _configure_adaptive_y_axis(ax2, r_time_vals)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center',
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=11)

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, transparent=True, bbox_inches='tight', format=ext)
    plt.close(fig)


# ── Cost-component stacked bar chart ─────────────────────

def stacked_cost_bar(x_label: str, x_vals, cost_vals, transport_vals,
                     transfer_vals, reposition_vals, unmet_penalty_vals,
                     save_path: str):
    """Plot the objective value as a three-component stacked bar chart.

    Transfer and reposition costs are combined as ``Transfer and Reposition cost``
    so that the small objective components remain legible at manuscript size.
    Total objective values are shown with black diamond markers.  Costs use
    their original units with a scientific-notation offset on the Y axis.
    """
    set_style()

    series = [
        np.asarray(values, dtype=float)
        for values in (
            cost_vals, transport_vals, transfer_vals,
            reposition_vals, unmet_penalty_vals,
        )
    ]
    if not all(len(values) == len(x_vals) for values in series):
        raise ValueError("All cost series must have the same length as x_vals")

    total = series[0]
    transport = series[1]
    other_operational = series[2] + series[3]
    unmet_penalty = series[4]

    x_pos = np.arange(len(x_vals))
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_style = {"width": 0.68, "edgecolor": "white", "linewidth": 0.6}

    ax.bar(
        x_pos, transport, color=COLOURS["C2"],
        label="Transport cost", **bar_style,
    )
    ax.bar(
        x_pos, other_operational, bottom=transport, color=COLOURS["C4"],
        label="Transfer and Reposition cost", **bar_style,
    )
    ax.bar(
        x_pos, unmet_penalty, bottom=transport + other_operational,
        color=COLOURS["C1"], label="Unmet-demand penalty", **bar_style,
    )
    ax.scatter(
        x_pos, total, color="#333333", marker="D", s=22,
        label="Total cost", zorder=5,
    )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Cost", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_fmt_values(x_vals), fontsize=11)
    ax.tick_params(axis="both", direction="in", labelsize=11)
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(0.0, max(total.max(), (transport + other_operational + unmet_penalty).max()) * 1.12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
    ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=False, fontsize=10,
    )

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(
            f"{save_path}.{ext}", dpi=600, transparent=True,
            bbox_inches="tight", format=ext,
        )
    plt.close(fig)


# ── Three vertically stacked subplots ─────────────────────

def three_subplots(x_label: str, x_vals, cost_vals, r_time_vals,
                   r_risk_vals, save_path: str,
                   c_cost: str = COLOURS["C5"],
                   c_risk: str = COLOURS["C1"],
                   c_time: str = COLOURS["C2"]):
    """
    Three vertically stacked subplots sharing the X axis:
      Top   – Cost ($)
      Mid   – R_time
      Bot   – R_risk
    Saves as PNG and TIFF at 600 dpi next to save_path (without extension).
    """
    set_style()
    fig, (ax_cost, ax_time, ax_risk) = plt.subplots(
        3, 1, figsize=(7, 9), sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.28})

    # Top: Cost
    ax_cost.plot(x_vals, cost_vals, color=c_cost, marker='^', markersize=5,
                 linewidth=1.8, zorder=5)
    ax_cost.set_ylabel(r'Cost (CNY)', fontsize=13)
    ax_cost.tick_params(axis='y', direction='in', labelsize=11)
    ax_cost.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax_cost.spines['top'].set_visible(False)
    ax_cost.spines['right'].set_visible(False)

    # Mid: R_time
    ax_time.plot(x_vals, r_time_vals, color=c_time, marker='s', markersize=5,
                 linewidth=1.8, zorder=5)
    ax_time.set_ylabel('Time reliability', fontsize=13)
    ax_time.tick_params(axis='y', direction='in', labelsize=11)
    ax_time.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)

    # Bottom: R_risk
    ax_risk.plot(x_vals, r_risk_vals, color=c_risk, marker='o', markersize=5,
                 linewidth=1.8, zorder=5)
    ax_risk.set_ylabel('Risk reliability', fontsize=13)
    ax_risk.set_xlabel(x_label, fontsize=14)
    ax_risk.tick_params(axis='x', direction='in', labelsize=11)
    ax_risk.tick_params(axis='y', direction='in', labelsize=11)
    ax_risk.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax_risk.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_risk.spines['top'].set_visible(False)
    ax_risk.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, transparent=True, bbox_inches='tight', format=ext)
    plt.close(fig)


# ── Discrete 5-segment colour map from set2 ───────────────

def discrete_cmap_5(values_arr=None):
    """
    Return a LinearSegmentedColormap built from the 5 set2 colours.
    If values_arr is provided (for heatmaps), colours are ordered by
    values: low → C5 (Pale Lime) → C3 (Sage) → C4 (Sand) → C2 (Celadon) → C1 (Terracotta).
    """
    # Low-to-high ordering for heatmaps
    cmap_colors = [
        COLOURS["C5"],  # Pale Lime  – lowest
        COLOURS["C3"],  # Sage Green
        COLOURS["C4"],  # Warm Sand
        COLOURS["C2"],  # Celadon Green
        COLOURS["C1"],  # Warm Terracotta – highest
    ]
    cmap = LinearSegmentedColormap.from_list("set2_5", cmap_colors, N=256)
    return cmap


# ── Heatmap wrapper ───────────────────────────────────────

def heatmap(data_2d, xlabels, ylabels, xlabel, ylabel, save_path,
            cmap=None, fmt=".2f", annotate=True):
    """
    Generic heatmap. data_2d is numpy array (rows=y, cols=x).
    Saves as PNG and TIFF at 600 dpi.
    """
    set_style()
    if cmap is None:
        cmap = discrete_cmap_5()

    fig, ax = plt.subplots(figsize=(max(6, len(xlabels) * 0.9), max(4, len(ylabels) * 0.6)))
    vmin = np.nanmin(data_2d)
    vmax = np.nanmax(data_2d)
    if vmin == vmax:
        vmax = vmin + 1e-9

    im = ax.imshow(data_2d, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    # Annotate cells
    if annotate:
        for i in range(len(ylabels)):
            for j in range(len(xlabels)):
                val = data_2d[i, j]
                if not np.isnan(val):
                    txt = f"{val:{fmt}}"
                    # Choose text colour for contrast
                    norm_val = (val - vmin) / (vmax - vmin)
                    text_color = 'white' if norm_val > 0.65 else 'black'
                    ax.text(j, i, txt, ha='center', va='center',
                            fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # Remove outer frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, transparent=True, bbox_inches='tight', format=ext)
    plt.close(fig)


# ── Enhanced Violin Plot (with embedded box & strip-scatter overlay) ──────────

def enhanced_boxplot(group_labels, bap_costs, bench_costs, save_path,
                     y_label="Total Cost (CNY)",
                     x_label=r"$(\alpha_T,\;\alpha_R)$",
                     model_bap_label="BAP (Reliable)",
                     model_bench_label="LP Relaxation",
                     c_bap=COLOURS["C1"], c_bench=COLOURS["C2"],
                     box_half_width=0.18):
    """
    Paired violin plots with embedded thin box-and-whisker and
    jittered black strip-scatter overlay.

    Parameters
    ----------
    group_labels : list[str]          – X-axis tick labels (one per group)
    bap_costs    : list[list[float]]  – cost values per group (BAP model)
    bench_costs  : list[list[float]]  – cost values per group (LP benchmark)
    save_path    : str                – output path without extension
    """
    import matplotlib.patches as mpatches
    from scipy.stats import wilcoxon

    set_style()

    n_groups = len(group_labels)
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * n_groups), 5.5))

    group_x = np.arange(n_groups)
    box_w   = box_half_width  # violin half-width
    offset  = box_w + 0.04   # centres for left / right violins

    # ── Colour helpers ────────────────────────────────────────────────────────
    def _alpha_hex(hex_color, alpha=0.35):
        """Convert #RRGGBB → RGBA tuple with given alpha."""
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        return (r, g, b, alpha)

    def _darken_hex(hex_color, factor=0.55):
        """Darken #RRGGBB by 'factor' (0 = black, 1 = unchanged)."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return '#{:02x}{:02x}{:02x}'.format(
            int(r * factor), int(g * factor), int(b * factor))

    bap_face   = _alpha_hex(c_bap,   0.30)
    bap_edge   = _darken_hex(c_bap,  0.50)
    bap_scatter= _darken_hex(c_bap,  0.70)  # between face and edge
    bench_face = _alpha_hex(c_bench, 0.30)
    bench_edge = _darken_hex(c_bench, 0.50)
    bench_scatter= _darken_hex(c_bench, 0.70)

    # ── Helper: clip violin Path to [data_min, data_max] ───────────────
    import matplotlib.path as mpath

    def _clip_violin_body(body, vmin, vmax):
        """Trim a violin PolyCollection body so its vertical extent
        stays within [vmin, vmax] (the actual data range)."""
        path = body.get_paths()[0]
        verts = path.vertices.copy()
        codes = path.codes.copy()
        ys = verts[:, 1]
        keep = (ys >= vmin) & (ys <= vmax)
        body.set_verts([verts[keep]])

    def _violin_span_at_y(body, y_value, center, fallback_half_width):
        """Return the violin's left/right outline intersections at ``y_value``."""
        verts = body.get_paths()[0].vertices
        if len(verts) < 2:
            return center - fallback_half_width, center + fallback_half_width

        crossings = []
        closed_verts = np.vstack((verts, verts[0]))
        for (x1, y1), (x2, y2) in zip(closed_verts[:-1], closed_verts[1:]):
            if np.isclose(y1, y2):
                if np.isclose(y_value, y1):
                    crossings.extend((x1, x2))
                continue
            if min(y1, y2) <= y_value <= max(y1, y2):
                fraction = (y_value - y1) / (y2 - y1)
                crossings.append(x1 + fraction * (x2 - x1))

        if len(crossings) >= 2:
            return min(crossings), max(crossings)
        return center - fallback_half_width, center + fallback_half_width

    # ── Draw violins + strip-scatter for each group ───────────────────────────
    y_all = []
    for i, lbl in enumerate(group_labels):
        for costs in (bap_costs[i], bench_costs[i]):
            y_all.extend([v for v in costs if v is not None and np.isfinite(v)])

    for i, lbl in enumerate(group_labels):
        gx = group_x[i]

        for costs, cx, face, edge, scatter_c, model_lbl in [
            (bap_costs[i],   gx - offset, bap_face,   bap_edge,   bap_scatter,   "BAP"),
            (bench_costs[i], gx + offset, bench_face, bench_edge, bench_scatter, "LP"),
        ]:
            vals = [v for v in costs if v is not None and np.isfinite(v)]
            if not vals:
                continue

            arr = np.array(vals)

            # ── Violin body ──────────────────────────────────────────────────
            vp = ax.violinplot(
                [arr], positions=[cx], widths=box_w * 2,
                showmeans=False, showmedians=False, showextrema=False,
            )
            # Clip violin to data range [min, max]
            vmin, vmax = arr.min(), arr.max()
            for body in vp["bodies"]:
                body.set_facecolor(face)
                body.set_edgecolor(edge)
                body.set_linewidth(1.0)
                body.set_alpha(0.85)
                _clip_violin_body(body, vmin, vmax)

            # ── Median only ─────────────────────────────────────────────────
            # Keep the central tendency marker without adding an embedded box,
            # whiskers, or caps that visually compete with the violin density.
            med = np.median(arr)
            median_left, median_right = _violin_span_at_y(
                vp["bodies"][0], med, cx, box_w * 0.55
            )
            ax.plot([median_left, median_right], [med, med],
                    color="black", linewidth=0.5, zorder=7)

            # ── Strip-scatter overlay (jittered dots) ────────────────────
            jitter = np.random.default_rng(0).uniform(
                -box_w * 0.55, box_w * 0.55, size=len(vals)
            )
            ax.scatter(
                cx + jitter, vals,
                color=scatter_c, s=2.5, alpha=0.45,
                zorder=6,
            )

    # ── Unified Y-axis (fit to data with margin) ──────────────────────────
    if y_all:
        y_min_data = min(y_all)
        y_max = max(y_all)
        margin = max(y_max * 0.02, (y_max - y_min_data) * 0.12)
        ax.set_ylim(max(0, y_min_data - margin), y_max + margin)

    # ── Paired Wilcoxon signed-rank test + cost gap annotation ─────────────
    y_range = (max(y_all) - min(y_all)) if y_all else 1.0

    for i in range(n_groups):
        # Each model can have a different number of successful scenarios.
        # Filter its observations independently; NumPy cannot pair arrays of
        # unequal length positionally, and doing so would be statistically
        # invalid when one model failed for a scenario.
        bp = np.asarray(
            [value for value in bap_costs[i] if value is not None and np.isfinite(value)],
            dtype=float,
        )
        lp = np.asarray(
            [value for value in bench_costs[i] if value is not None and np.isfinite(value)],
            dtype=float,
        )

        if len(bp) < 3 or len(lp) < 3:
            continue

        # Cost gap: (BAP − LP) / LP × 100
        lp_med = np.median(lp)
        gap_pct = (np.median(bp) - lp_med) / lp_med * 100 if lp_med != 0 else float('nan')
        gap_text = f"gap {gap_pct:+.1f}%"

        # Wilcoxon requires genuinely paired observations.  With incomplete
        # experiment rows, retain the distribution and median-gap annotation
        # but explicitly omit a significance claim.
        if len(bp) != len(lp):
            sig_text = "n/a"
        else:
            diffs = bp - lp
            if np.all(diffs == 0):
                sig_text = "ns"
            else:
                stat, p = wilcoxon(bp, lp, alternative='two-sided')
                if p < 0.001:
                    sig_text = "***"
                elif p < 0.01:
                    sig_text = "**"
                elif p < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "ns"

        # Lowest point of this group across both violins
        all_group = np.concatenate([bp, lp])
        group_lo  = float(np.min(all_group))
        text_gap = y_range * 0.04
        text_y = max(group_lo - text_gap, text_gap * 0.5)

        ax.text(group_x[i], text_y, f"{sig_text}\n{gap_text}",
                ha="center", va="top", fontsize=10, fontweight="normal",
                color="black", zorder=7, linespacing=1.4)

    # ── X-axis labels ─────────────────────────────────────────────────────────
    ax.set_xticks(group_x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)

    # ── Legend ─────────────────────────────────────────────────────────────────
    bap_patch   = mpatches.Patch(facecolor=bap_face,   edgecolor=bap_edge,
                                 linewidth=1.2, label=model_bap_label)
    bench_patch = mpatches.Patch(facecolor=bench_face, edgecolor=bench_edge,
                                 linewidth=1.2, label=model_bench_label)
    sig_patch   = mpatches.Patch(facecolor="none", edgecolor="none",
                                 label="Wilcoxon: * p<0.05, ** p<0.01, *** p<0.001")
    ax.legend(handles=[bap_patch, bench_patch, sig_patch], loc="upper left",
              frameon=False, fontsize=9, ncol=2)

    ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="in", labelsize=11)

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, transparent=True, bbox_inches="tight", format=ext)
    plt.close(fig)
