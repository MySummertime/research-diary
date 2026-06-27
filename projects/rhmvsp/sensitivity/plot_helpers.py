"""
Shared plotting utilities for all sensitivity experiments.

Enforces:
  - Times New Roman font family
  - No figure titles
  - Colour palette from docs/colour set2.txt (5 colours)
"""
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter

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
    """Format numeric values with adaptive precision for tick labels.

    Detects the maximum number of decimal places present in the input
    values and uses that for all labels, ensuring every value is
    displayed with full precision and labels never collide.
    """
    max_dp = 0
    for v in values:
        if v is None:
            continue
        s = f"{float(v):.10f}".rstrip('0')
        dp = len(s.split('.')[1]) if '.' in s else 0
        max_dp = max(max_dp, dp)
    return [f"{float(v):.{max_dp}f}" if v is not None else str(v) for v in values]


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
                c1: str = COLOURS["C1"], c2: str = COLOURS["C2"]):
    """
    Dual Y-axis line chart: R_risk (left), R_time (right) vs x_vals.
    Uses categorical x-positions + adaptive decimal labels so that
    non-uniform float grids (e.g. alpha_max) render correctly.
    Saves as PNG and TIFF at 600 dpi next to save_path (without extension).
    """
    set_style()
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Categorical x positions (robust for non-uniform float grids)
    x_pos = np.arange(len(x_vals))
    x_labels = _fmt_values(x_vals)

    # Left y-axis: R_risk
    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel(r'Network Risk Reliability ($R_{risk}$)',
                   fontsize=14, color=c1)
    line1, = ax1.plot(x_pos, r_risk_vals, color=c1, marker='o',
                      markersize=5, linewidth=1.8,
                      label=r'Risk Reliability ($R_{risk}$)', zorder=5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=11)
    ax1.tick_params(axis='x', direction='in', labelsize=11)
    ax1.tick_params(axis='y', direction='in', labelcolor=c1, labelsize=11)
    ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))

    rr_min, rr_max = min(r_risk_vals), max(r_risk_vals)
    mr = max(0.01, (rr_max - rr_min) * 0.15)
    ax1.set_ylim(rr_min - mr, rr_max + mr)
    ax1.set_xlim(-0.5, len(x_vals) - 0.5)
    ax1.spines['top'].set_visible(False)

    # Right y-axis: R_time
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Network Time Reliability ($R_{time}$)',
                   fontsize=14, color=c2)
    line2, = ax2.plot(x_pos, r_time_vals, color=c2, marker='s',
                      markersize=5, linewidth=1.8, linestyle='--',
                      label=r'Time Reliability ($R_{time}$)', zorder=4)
    ax2.tick_params(axis='y', direction='in', labelcolor=c2, labelsize=11)
    ax2.spines['top'].set_visible(False)

    rt_min, rt_max = min(r_time_vals), max(r_time_vals)
    mt = max(0.01, (rt_max - rt_min) * 0.15)
    ax2.set_ylim(rt_min - mt, rt_max + mt)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center',
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=11)

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches='tight', format=ext)
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
    ax_time.set_ylabel(r'$R_{time}$', fontsize=13)
    ax_time.tick_params(axis='y', direction='in', labelsize=11)
    ax_time.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)

    # Bottom: R_risk
    ax_risk.plot(x_vals, r_risk_vals, color=c_risk, marker='o', markersize=5,
                 linewidth=1.8, zorder=5)
    ax_risk.set_ylabel(r'$R_{risk}$', fontsize=13)
    ax_risk.set_xlabel(x_label, fontsize=14)
    ax_risk.tick_params(axis='x', direction='in', labelsize=11)
    ax_risk.tick_params(axis='y', direction='in', labelsize=11)
    ax_risk.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax_risk.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_risk.spines['top'].set_visible(False)
    ax_risk.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches='tight', format=ext)
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


# ── Dual Y-axis: lines (left) + cost bars (right) ────────

def dual_y_axis_with_cost_bar(x_label: str, x_vals, r_risk_vals, r_time_vals,
                               cost_vals, save_path: str,
                               c1: str = COLOURS["C1"], c2: str = COLOURS["C2"],
                               c_cost: str = COLOURS["C3"],
                               y1_label: str = r'$R_{risk}$',
                               y2_label: str = r'$R_{time}$',
                               y1_title: str = r'Network Reliability',
                               y2_title: str = r'Total Cost (CNY)'):
    """
    Dual Y-axis chart:
      Left axis  – R_risk (solid circle) and R_time (dashed square), line plots.
      Right axis – Cost shown as a Sage Green bar chart.
    Saves as PNG and TIFF at 600 dpi next to save_path (without extension).
    """
    set_style()
    fig, ax1 = plt.subplots(figsize=(7, 5))

    x_pos = np.arange(len(x_vals))
    bar_width = max(0.4, min(0.7, 3.0 / len(x_vals)))

    # Right y-axis: Cost bars (Sage Green)
    ax2 = ax1.twinx()
    ax2.bar(x_pos, cost_vals, width=bar_width, color=c_cost,
            edgecolor='none', alpha=0.6, label='Total Cost', zorder=1)
    ax2.set_ylabel(y2_title, fontsize=14, color=c_cost)
    ax2.tick_params(axis='y', direction='in', labelcolor=c_cost, labelsize=11)
    ax2.spines['top'].set_visible(False)

    c_min, c_max = min(cost_vals), max(cost_vals)
    mc = max(1.0, (c_max - c_min) * 0.20)
    ax2.set_ylim(c_min - mc, c_max + mc)

    # Left y-axis: R_risk and R_time lines
    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel(y1_title, fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(_fmt_values(x_vals), fontsize=11)

    line1, = ax1.plot(x_pos, r_risk_vals, color=c1, marker='o',
                      markersize=5, linewidth=1.8,
                      label=y1_label, zorder=5)
    line2, = ax1.plot(x_pos, r_time_vals, color=c2, marker='s',
                      markersize=5, linewidth=1.8, linestyle='--',
                      label=y2_label, zorder=4)

    ax1.tick_params(axis='x', direction='in', labelsize=11)
    ax1.tick_params(axis='y', direction='in', labelsize=11)
    ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax1.spines['top'].set_visible(False)

    rr_min, rr_max = min(r_risk_vals), max(r_risk_vals)
    mr = max(0.01, (rr_max - rr_min) * 0.15)
    rt_min, rt_max = min(r_time_vals), max(r_time_vals)
    mt = max(0.01, (rt_max - rt_min) * 0.15)
    lo = min(rr_min - mr, rt_min - mt)
    hi = max(rr_max + mr, rt_max + mt)
    ax1.set_xlim(-0.5, len(x_vals) - 0.5)
    ax1.set_ylim(lo, hi)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines + [plt.Rectangle((0, 0), 1, 1, fc=c_cost, alpha=0.6, ec='none')],
               labels + ['Total Cost'],
               loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3,
               frameon=False, fontsize=11)

    plt.tight_layout()
    for ext in ["png", "tiff"]:
        fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches='tight', format=ext)
    plt.close(fig)
    return f"{save_path}.png", f"{save_path}.tiff"


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
        fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches='tight', format=ext)
    plt.close(fig)


# ── Enhanced Violin Plot (with embedded box & strip-scatter overlay) ──────────

def enhanced_boxplot(group_labels, bap_costs, bench_costs, save_path,
                     y_label="Total Cost (CNY)",
                     x_label=r"$(\alpha_T,\;\alpha_{max})$",
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

    # ── Helper: clip violin Path to [data_min, data_max] ─────────────────
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

            # ── Embedded mini box-and-whisker (thin) ─────────────────────────
            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            iqr = q3 - q1
            whisker_lo = max(arr.min(), q1 - 1.5 * iqr)
            whisker_hi = min(arr.max(), q3 + 1.5 * iqr)
            bw = box_w * 0.35  # thin box width

            # Whiskers (vertical lines)
            ax.plot([cx, cx], [whisker_lo, q1], color=edge, linewidth=0.8, zorder=4)
            ax.plot([cx, cx], [q3, whisker_hi], color=edge, linewidth=0.8, zorder=4)
            # Caps (horizontal ticks)
            cap_w = bw * 0.6
            ax.plot([cx - cap_w, cx + cap_w], [whisker_lo, whisker_lo],
                    color=edge, linewidth=0.8, zorder=4)
            ax.plot([cx - cap_w, cx + cap_w], [whisker_hi, whisker_hi],
                    color=edge, linewidth=0.8, zorder=4)
            # Box (Q1–Q3)
            ax.bar(cx, q3 - q1, width=bw * 2, bottom=q1,
                   color="white", edgecolor=edge, linewidth=1.2, zorder=4)
            # Median line
            ax.plot([cx - bw, cx + bw], [med, med],
                    color=edge, linewidth=0.8, zorder=5)

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
        bap_arr = np.array(bap_costs[i], dtype=float)
        lp_arr  = np.array(bench_costs[i], dtype=float)
        # Keep only paired entries where both models succeeded
        valid = ~(np.isnan(bap_arr) | np.isnan(lp_arr))
        bp = bap_arr[valid]
        lp = lp_arr[valid]

        if len(bp) < 3:
            continue

        # Cost gap: (BAP − LP) / LP × 100
        lp_med = np.median(lp)
        gap_pct = (np.median(bp) - lp_med) / lp_med * 100 if lp_med != 0 else float('nan')
        gap_text = f"gap {gap_pct:+.1f}%"

        # Paired Wilcoxon signed-rank test (two-sided)
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
        fig.savefig(f"{save_path}.{ext}", dpi=600, bbox_inches="tight", format=ext)
    plt.close(fig)
