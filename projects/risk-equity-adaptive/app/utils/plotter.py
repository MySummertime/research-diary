# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import PchipInterpolator

from app.core.evaluator import Evaluator
from app.core.solution import Solution
from app.utils.visual_style import ColorPalette, get_color_by_key

# --- 全局 Matplotlib Formatter ---
# 仅对超出 10^-2 到 10^3 的范围的数值使用科学计数法
# 设置 useMathText=True 则为 1.xx * 10^x 样式，否则为 1.xx 1ex 样式
FMT = ScalarFormatter(useMathText=True)
FMT.set_powerlimits((-2, 3))


# --- 全局绘图风格设置 ---
def apply_academic_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "cm",  # Latex 风格数学字体
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 2,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# 应用风格
apply_academic_style()


class BasePlotter:
    """
    所有 Plotter 的抽象基类，用于统一管理全局设置和辅助方法。
    """

    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Color set
        self.default_colors = ColorPalette.DEFAULT_COLOR
        self.pareto_colors: List[Dict] = ColorPalette.PARETO
        self.pareto_colors_by_algo: List[Dict] = ColorPalette.PARETO_BY_ALGO
        self.pareto_colors_loop: List[str] = ColorPalette.PARETO_LOOP
        self.violin_colors: List[str] = ColorPalette.VIOLIN_LOOP
        self.grouped_bar_colors: List[str] = ColorPalette.GROUPED_BAR
        self.stacked_bar_chart_colors = ColorPalette.STACKED_BAR
        self.dual_line_chart_colors = ColorPalette.DUAL_LINE
        self.heatmap_colors = ColorPalette.HEATMAP

    def _format_axes(self, ax):
        """
        [辅助] 统一格式化轴刻度，使用全局 FMT 科学计数法规则
        """
        ax.xaxis.set_major_formatter(FMT)
        ax.yaxis.set_major_formatter(FMT)
        ax.grid(True, linestyle="--", alpha=0.5)

        # L 形边框：保留 bottom + left，去掉 top + right
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 可选
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)

        # 可选：tick 朝内 + 长度适中
        ax.tick_params(direction="in", length=4, width=0.8)

    def _set_dynamic_xlim(self, ax, data, margin=0.05):
        """辅助：设置动态 X 轴范围 (基于边距)"""
        if not data:
            return
        x_min, x_max = min(data), max(data)
        # 简单计算边距并设置 X 轴范围
        dx = (x_max - x_min) * margin if x_max != x_min else x_min * margin
        ax.set_xlim(x_min - dx, x_max + dx)

    def _set_dynamic_ylim(
        self,
        ax,
        data,
        margin_ratio: float = 0.1,
        is_bar: bool = False,
        lower_multiplier: Optional[float] = None,
        upper_multiplier: Optional[float] = None,
    ):
        """
        [辅助] 动态设置 Y 轴范围 (支持截断和聚焦模式)

        该函数根据数据波动率自动决定是否启用"聚焦模式":
        - 波动率 < 5% (0.05)：启用聚焦，以放大微小波动。
        - 波动率 > 5%：采用常规的 10% 边距。

        参数:
            ax (Axes): Matplotlib Axes 对象。
            data (List[float]): 需要绘制的数据列表。
            margin_ratio (float): 正常波动时的边距比例 (默认为 0.1, 即 10%)。
            is_bar (bool): 是否为柱状图。若为 True，则强制 Y 轴从 0 开始。
            lower_multiplier (Optional[float]): 聚焦模式下，Ymin 边界的边距乘数。
                                                (边距 = (Max-Min) * 乘数)。
                                                例如，设置为 0.5，则 Ymin = Min - (波动 * 0.5)。
                                                设置为 1.0，则 Ymin = Min - 波动。
                                                如果为 None (默认)，则使用 1.0。
            upper_multiplier (Optional[float]): 聚焦模式下，Ymax 边界的边距乘数。
                                                如果为 None (默认)，则使用 1.0。
        """
        if len(data) == 0:
            return
        ymin, ymax = min(data), max(data)

        # 1. 柱状图强制从 0 开始
        if is_bar:
            ax.set_ylim(0, ymax * 1.1)
            return

        # 2. 计算波动率
        if ymax > 0:
            diff = ymax - ymin
            variation = diff / ymax

            # 3. 启用聚焦模式 (波动率 < 5%)
            if variation < 0.05:
                # 聚焦微小变化
                margin = diff if diff > 0 else ymax * 0.01

                # 默认值: 使用 1.0 (即边距 = 波动范围)
                LOWER_MULTIPLIER = (
                    lower_multiplier if lower_multiplier is not None else 1.0
                )
                UPPER_MULTIPLIER = (
                    upper_multiplier if upper_multiplier is not None else 1.0
                )

                # 如果乘数设置为 0，则直接贴合数据点
                lower = max(0, ymin - margin * LOWER_MULTIPLIER)
                upper = ymax + margin * UPPER_MULTIPLIER

                ax.set_ylim(lower, upper)
            else:
                # 4. 常规模式 (波动率 >= 5%)
                margin = diff * margin_ratio
                ax.set_ylim(max(0, ymin - margin), ymax + margin)


class ParetoPlotter(BasePlotter):
    """
    [View Layer] 通用 Pareto Frontier 绘图器
    职责：绘制目标空间 (Objective Space) 的散点图。
    """

    def __init__(self, title: str = "", save_dir: str = "results"):
        super().__init__(save_dir)

        self.title = title
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot(
        self,
        solutions: List[Solution],
        file_name: str = "pareto_frontier",
        xlabel: str = r"Total Risk (people)",
        ylabel: str = r"Total Cost (yuan)",
        special_solutions: Optional[Dict[str, Solution]] = None,
    ):
        """
        绘制单个的 2-D Pareto Frontier，并高亮特殊解。
            不进行平滑，以保持离散解的真实感。
            X=Risk, Y=Cost
        Args:
            solutions: pareto frontier 的接列表
            file_name: 保存的文件名
            xlabel: x 轴的 label 名
            ylabel: y 轴的 label 名
            special_solutions: 一个字典 {'Label': Solution}，用于在图上高亮标记特殊解
        """
        # 1. 预处理：筛选可行解并按 Rank 分组
        feasible_sols = [s for s in solutions if s.is_feasible]
        if not feasible_sols:
            return

        solutions_by_rank: Dict[int, List[Solution]] = {}
        for s in feasible_sols:
            if s.rank not in solutions_by_rank:
                solutions_by_rank[s.rank] = []
            solutions_by_rank[s.rank].append(s)

        # 创建画布和坐标轴 (面向对象风格)
        fig, ax = plt.subplots(figsize=(10, 8))

        # 2. 绘制次优层级 (Rank 3 -> Rank 2 -> Rank 1)

        # 配置 Rank 1-3 的样式 (颜色逐级变浅，形状不同)
        # Rank 1: 深灰, 方块
        # Rank 2: 中灰, 三角
        # Rank 3: 浅灰, 菱形
        # 配置 Rank 1-3 的样式
        rank_styles = {
            1: {
                "color": get_color_by_key(
                    self.pareto_colors, "RANK1_COLOR"
                ),  # DARK_GRAY
                "marker": "s",
                "label": "Rank 1",
                "s": 40,
                "alpha": 0.7,
            },
            2: {
                "color": get_color_by_key(self.pareto_colors, "RANK2_COLOR"),  # GRAY
                "marker": "^",
                "label": "Rank 2",
                "s": 35,
                "alpha": 0.6,
            },
            3: {
                "color": get_color_by_key(
                    self.pareto_colors, "RANK3_COLOR"
                ),  # LIGHT_GRAY
                "marker": "D",
                "label": "Rank 3",
                "s": 30,
                "alpha": 0.5,
            },
        }

        # 倒序绘制 3, 2, 1，保证 Rank 1 在最上层
        for r in [3, 2, 1]:
            if r in solutions_by_rank:
                sols = solutions_by_rank[r]
                x_vals = [s.f1_risk for s in sols]
                y_vals = [s.f2_cost for s in sols]

                style = rank_styles[r]
                ax.scatter(
                    x_vals,
                    y_vals,
                    c=style["color"],
                    marker=style["marker"],
                    s=style["s"],
                    alpha=style["alpha"],
                    label=style["label"],
                    edgecolors="none",  # 次优解通常不需要边框，保持柔和
                    zorder=1,  # 放在底层
                )

        # 3. 绘制 Rank 0 (Pareto Front)
        if 0 in solutions_by_rank:
            rank0_sols = solutions_by_rank[0]
            # 排序以便画连线
            rank0_sols.sort(key=lambda s: s.f1_risk)

            x_r0 = [s.f1_risk for s in rank0_sols]
            y_r0 = [s.f2_cost for s in rank0_sols]

            # 画连线 (虚线)
            ax.plot(
                x_r0,
                y_r0,
                color=get_color_by_key(self.pareto_colors, "PARETO_FRONT_LINE"),
                linestyle="--",
                alpha=0.6,
                zorder=2,
            )

            # 画散点 (空心圆形)
            ax.scatter(
                x_r0,
                y_r0,
                facecolors=get_color_by_key(self.default_colors, "WHITE"),
                edgecolors=get_color_by_key(self.pareto_colors, "PARETO_POINT_EDGE"),
                linewidths=1.5,
                marker="o",
                s=60,
                label="Pareto Optimal",
                zorder=3,
            )

            # 标记 Rank 0 的特殊点
            if special_solutions:
                self._highlight_special_solutions(
                    get_color_by_key(self.pareto_colors, "SPECIAL_POINT"),
                    special_solutions,
                    ax,
                )

        # 4. 格式化图表
        # axis formatting
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        self._format_axes(ax)  # 统一格式化

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Ensure consistent order
        order = ["Pareto Optimal", "Rank 1", "Rank 2", "Rank 3"]
        sorted_keys = [k for k in order if k in by_label]
        sorted_handles = [by_label[k] for k in sorted_keys]
        ax.legend(
            sorted_handles,
            sorted_keys,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            fancybox=True,
        )

        # 5. 保存
        full_path = os.path.join(self.save_dir, f"{file_name}.tiff")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )

        full_path = os.path.join(self.save_dir, f"{file_name}.png")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close(fig)
        logging.info(f"Pareto plot saved to: {full_path}")

    def plot_frontier_comparison_by_algo(
        self,
        frontiers: Dict[str, List[Solution]],
        file_name: str = "pareto_comparison_by_algo",
        xlabel: str = r"Total Risk (people)",
        ylabel: str = r"Total Cost (yuan)",
    ):
        """
        绘制不同算法下的 Pareto Frontier 对比图。
        使用插值函数绘制平滑曲线。
        X=Risk, Y=Cost

        Args:
            frontiers: 字典 { "Algorithm Name": [Solution List], ... }
            file_name: 保存文件名
        """

        # 空解集保护逻辑
        valid_algos = {
            algo: sols for algo, sols in frontiers.items() if sols and len(sols) > 0
        }
        if not valid_algos:
            logging.warning(
                f"Plotter: No valid solutions found in any algorithm for {file_name}. Skipping plot."
            )
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # 定义 7 种算法样式表，确保色彩具有连贯性
        styles = {
            "Gurobi": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "EXACT"),
                "marker": "*",
                "s": 50,
                "label": "Gurobi",
                "zorder": 10,  # 最高层
            },
            "NSGA-II (Imp)": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "PROPOSED"),
                "marker": "o",
                "s": 110,
                "label": "NSGA-II (Imp)",
                "zorder": 4,
            },
            # --- 消融实验 Alation experiments ---
            # NSGA-II_Basic
            "NSGA-II (1)": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "ABLATION_1"),
                "marker": "^",
                "s": 50,
                "label": "Ablation 0",
                "zorder": 8,
            },
            "NSGA-II (2)": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "ABLATION_2"),
                "marker": "v",
                "s": 65,
                "label": "Ablation 1",
                "zorder": 7,
                "alpha": 0.9,
            },
            "NSGA-II (3)": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "ABLATION_3"),
                "marker": "<",
                "s": 80,
                "label": "Ablation 2",
                "zorder": 6,
                "alpha": 0.8,
            },
            "SPEA2": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "BASELINE_1"),
                "marker": "D",
                "s": 110,
                "label": "SPEA2",
                "zorder": 4,
            },
        }

        # 遍历绘制
        for algo_name, solutions in frontiers.items():
            if not solutions:
                continue

            # 提取坐标
            x_vals = [s.f1_risk for s in solutions]
            y_vals = [s.f2_cost for s in solutions]

            # 获取样式 (如果没有定义，使用默认灰)
            style = styles.get(
                algo_name,
                {
                    "c": ColorPalette.DEFAULT_GRAY,
                    "marker": "x",
                    "s": 30,
                    "label": algo_name,
                    "zorder": 1,
                },
            )

            ax.scatter(
                x_vals,
                y_vals,
                c=style["c"],
                marker=style["marker"],
                s=style["s"],
                label=style["label"],
                zorder=style.get("zorder", 1),
                alpha=style.get("alpha", 0.9),
                linewidths=style.get("linewidths", 0),
            )

        # 格式化
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")

        self._format_axes(ax)

        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fancybox=True)

        # 保存
        full_path = os.path.join(self.save_dir, f"{file_name}.tiff")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )

        full_path = os.path.join(self.save_dir, f"{file_name}.png")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close(fig)
        logging.info(f"Comparison plot saved to: {full_path}")

    def plot_frontier_comparison(
        self,
        frontiers: Dict[str, List[Solution]],
        file_name: str = "pareto_comparison_by_cvar_alpha",
        x_prefix: str = "",
        legend_loc: str = "upper right",
    ):
        """
        绘制不同 CVaR alpha 下的 Pareto Frontier 对比图。
        X=Risk, Y=Cost
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors: List[str] = self.pareto_colors_loop
        markers: List[str] = ["o", "s", "D", "p", "+", "^", "v", "<", ">"]

        all_x, all_y = [], []

        for idx, (label, solutions) in enumerate(frontiers.items()):
            if not solutions:
                continue

            # 按 Risk 排序
            solutions.sort(key=lambda s: s.f1_risk)

            x_vals = [s.f1_risk for s in solutions]
            y_vals = [s.f2_cost for s in solutions]
            all_x.extend(x_vals)
            all_y.extend(y_vals)

            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # [Straight Line] 直线连接
            ax.plot(
                x_vals, y_vals, color=color, linestyle="--", alpha=0.5, linewidth=1.5
            )
            ax.scatter(
                x_vals,
                y_vals,
                label=f"{x_prefix}{label}",
                color=color,
                marker=marker,
                s=50,
                alpha=0.9,
                zorder=3,
            )

        ax.set_xlabel(r"Total Risk (people)", fontweight="bold")
        ax.set_ylabel(r"Total Cost (yuan)", fontweight="bold")

        self._format_axes(ax)
        self._set_dynamic_xlim(ax, all_x)
        self._set_dynamic_ylim(ax, all_y)

        ax.legend(loc=legend_loc, frameon=True, fancybox=True)

        full_path = os.path.join(self.save_dir, f"{file_name}.tiff")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )

        full_path = os.path.join(self.save_dir, f"{file_name}.png")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close(fig)
        logging.info(f"Pareto comparison saved: {file_name}")

    def plot_value_comparison(
        self,
        dyn_front: List[Solution],
        static_reevaluated: List[Solution],
        file_name: str,
    ):
        """
        动态模型价值对比图。
        展示 Proposed (线+点) 与 Static (散点) 在真实动态环境下的差异。
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # 1. 绘制 Proposed 前沿 (蓝色实线 + 空心圆)
        dyn_front.sort(key=lambda s: s.f1_risk)
        dyn_risk = [s.f1_risk for s in dyn_front]
        dyn_cost = [s.f2_cost for s in dyn_front]

        ax.plot(
            dyn_risk,
            dyn_cost,
            color=get_color_by_key(self.pareto_colors_by_algo, "PROPOSED"),
            linestyle="-",
            marker="o",
            markerfacecolor="white",
            markersize=7,
            label="Proposed Model (Uncertain Consequence)",
            zorder=3,
        )

        # 2. 绘制静态解回测点 (红色叉号)
        static_risk = [s.f1_risk for s in static_reevaluated]
        static_cost = [s.f2_cost for s in static_reevaluated]

        ax.scatter(
            static_risk,
            static_cost,
            color=get_color_by_key(self.pareto_colors_by_algo, "EXACT"),
            marker="x",
            s=80,
            label="Static Model (Static Consequence)",
            zorder=2,
        )

        ax.set_xlabel(r"Total Risk $f_1$ (people)", fontweight="bold")
        ax.set_ylabel(r"Total Cost $f_2$ (yuan)", fontweight="bold")
        self._format_axes(ax)

        # 自动聚焦 Y 轴范围
        self._set_dynamic_ylim(ax, dyn_cost + static_cost, margin_ratio=0.15)

        ax.legend(loc="upper right", frameon=True, shadow=True)

        full_path = os.path.join(self.save_dir, f"{file_name}.tiff")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )

        full_path = os.path.join(self.save_dir, f"{file_name}.png")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close(fig)
        logging.info(f"Value comparison Pareto chart saved: {file_name} 📈")

    def plot_gini_tradeoff(
        self,
        solutions_with_gini: List[Tuple[Solution, float]],
        file_name: str = "gini_tradeoff",
        special_solutions=None,
    ):
        """
        绘制总风险 (Total Risk) 与基尼系数 (Gini Coefficient) 的权衡图。
        使用颜色映射 (CMAP) 嵌入第三个维度 (Cost f2)。
        """
        if not solutions_with_gini:
            logging.warning("No solutions provided for Gini Trade-off plot.")
            return

        # 1. 提取 Rank 0 数据并排序
        rank_0_data = []
        all_feasible_data = []

        for sol, gini in solutions_with_gini:
            f1_risk = sol.f1_risk
            f2_cost = sol.f2_cost

            # 提取所有可行解数据（用于背景散点）
            all_feasible_data.append((f1_risk, gini))

            if sol.rank == 0 and sol.is_feasible:
                rank_0_data.append((f1_risk, gini, f2_cost))  # (Risk, Gini, Cost)

        # 2. 按 f1 (Risk) 升序排序
        rank_0_data.sort(key=lambda x: x[0])

        r0_risk = [r for r, g, c in rank_0_data]
        r0_gini = [g for r, g, c in rank_0_data]
        r0_cost = [c for r, g, c in rank_0_data]

        # 3. 准备颜色映射 (CMAP)
        if not r0_cost:
            return
        min_cost, max_cost = min(r0_cost), max(r0_cost)

        # 使用 Cost CMAP (GnBu): 暖色代表高风险，冷色代表低风险
        cmap_name = get_color_by_key(ColorPalette.HEATMAP, "COST_CMAP")
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=min_cost, vmax=max_cost)

        # 4. 绘制图表
        fig, ax = plt.subplots(figsize=(10, 8))

        # 4.1 绘制所有可行解 (背景散点)
        ax.scatter(
            [r for r, g in all_feasible_data],
            [g for r, g in all_feasible_data],
            s=30,
            alpha=0.5,
            color=get_color_by_key(self.pareto_colors, "RANK1_COLOR"),
            label="Feasible Solutions (All Ranks)",
            zorder=1,
        )

        # 4.2 绘制 Rank 0 前沿 (连线和 CMAP 散点)
        line_color = get_color_by_key(self.pareto_colors, "PARETO_POINT_EDGE")

        # 绘制连线
        ax.plot(
            r0_risk,
            r0_gini,
            color=line_color,
            linestyle="-",
            alpha=0.6,
            linewidth=2.0,
            zorder=2,
        )

        # 绘制散点 (使用 Cost 进行颜色映射)
        scatter = ax.scatter(
            r0_risk,
            r0_gini,
            c=r0_cost,  # 使用 Cost 映射颜色
            cmap=cmap,
            norm=norm,  # 归一化
            edgecolors=get_color_by_key(self.pareto_colors, "PARETO_POINT_EDGE"),
            linewidths=1.0,
            marker="o",
            s=100,
            label="Pareto Optimal (Cost)",
            zorder=3,
        )

        # 高亮特殊解 (Opinion A, B, C)
        if special_solutions:
            highlight_color = get_color_by_key(self.pareto_colors, "SPECIAL_POINT")
            for label, sol in special_solutions.items():
                if not sol:
                    continue
                # 此时 y 轴是 Gini 系数，需从 sol 属性或 solutions_with_gini 中获取
                g_val = getattr(sol, "gini_coefficient", 0.0)
                ax.scatter(
                    [sol.f1_risk],
                    [g_val],
                    facecolors="none",
                    edgecolors=highlight_color,
                    s=220,  # 外扩尺寸
                    linestyle=":",  # 虚线风格
                    linewidths=2,  # 线宽
                    zorder=5,  # 置顶显示
                )

        # 5. 添加颜色条 (Color Bar)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label(r"Total Cost $f_2$ (yuan)", fontweight="bold")
        cbar.ax.yaxis.set_major_formatter(FMT)  # 颜色条也格式化

        # 6. 格式化
        ax.set_xlabel(r"Total Risk $f_1$ (people)", fontweight="bold")
        ax.set_ylabel("Gini Coefficient (Social Equity)", fontweight="bold")

        self._format_axes(ax)
        self._set_dynamic_xlim(ax, r0_risk)
        self._set_dynamic_ylim(ax, r0_gini)

        ax.legend(loc="upper right", frameon=True, fancybox=True)

        full_path = os.path.join(self.save_dir, f"{file_name}.tiff")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )

        full_path = os.path.join(self.save_dir, f"{file_name}.png")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close(fig)
        logging.info(f"Gini Trade-off plot saved with Cost CMAP: {file_name} 🌟")

    def plot_parallel_coordinates(
        self,
        rank0_solutions: List[Solution],
        gini_calculator: callable,
        evaluator: Evaluator,
        file_name: str = "parallel_coordinates_tradeoff",
    ):
        """
        绘制平行坐标图，展示整个 Pareto 前沿 (Rank 0) 的三目标联动关系。
        1. 使用保界平滑曲线 (PCHIP)
        2. 线条颜色映射到 Total Risk
        3. 线宽映射到拥挤距离 (CD)
        4. 刻度原始值使用科学计数法格式化
        """
        # 1. 收集和归一化数据 (Risk, Cost, Gini, CD)
        data_list = []
        cd_list = []
        for sol in rank0_solutions:
            # 确保只包含 Rank 0 可行解
            if sol.rank == 0 and sol.is_feasible:
                # Gini 值计算或提取
                gini_val = (
                    sol.gini_coefficient
                    if sol.gini_coefficient != float("inf")
                    else gini_calculator(sol, evaluator)
                )

                # 检查 CD 是否存在，否则给一个默认值
                cd_val = getattr(sol, "crowding_distance", 1.0)

                data_list.append((sol.f1_risk, sol.f2_cost, gini_val))
                cd_list.append(cd_val)

        if not data_list:
            logging.warning("No Rank 0 data found for Parallel Coordinates Plot.")
            return

        # A. 归一化 (Risk, Cost, Gini)
        normalized_data, min_vals, max_vals = self._normalize_metrics_for_comparison(
            data_list
        )

        # B. 归一化拥挤距离 (CD)
        cd_array = np.array(cd_list)

        if len(cd_array) > 1:
            min_cd, max_cd = np.min(cd_array), np.max(cd_array)
            range_cd = max_cd - min_cd

            if range_cd < 1e-9:
                # 拥挤距离相同，设置基础中等线宽
                normalized_cd = np.full_like(cd_array, 2.0)
            else:
                # --- 增大视觉对比度 ---

                # 1. 对 CD 进行归一化
                normalized_cd_raw = (cd_array - min_cd) / range_cd

                # 2. 应用对数缩放 (Lognormal Scaling)
                # 避免 log(0)
                epsilon = 1e-6
                log_scaled_cd = np.log(normalized_cd_raw + epsilon)

                # 对 log 后的值再进行一次归一化，确保在 [0, 1] 范围内
                min_log, max_log = np.min(log_scaled_cd), np.max(log_scaled_cd)
                range_log = max_log - min_log

                # 3. 映射到线宽范围 (基础线宽 0.8，最大到 4.5)
                # 增大线宽范围：从 1.0 -> 3.0 增大到 0.8 -> 4.5
                normalized_cd = 0.8 + 3.7 * ((log_scaled_cd - min_log) / range_log)

        else:
            # 单个解，默认中等线宽
            normalized_cd = np.array([2.0])

        df = pd.DataFrame(normalized_data, columns=["Risk", "Cost", "Gini"])

        # 2. 设置图表
        fig, ax = plt.subplots(figsize=(10, 8))

        axes = np.array([0, 1, 2])
        labels = [
            r"Total Risk $f_1$ (people)",
            r"Total Cost $f_2$ (yuan)",
            "Gini Coefficient",
        ]

        # 3. 设置颜色映射 (CMAP)
        # 使用 Risk CMAP，颜色映射到原始 Risk 值
        cmap = plt.get_cmap(get_color_by_key(ColorPalette.HEATMAP, "RISK_CMAP"))
        norm = mcolors.Normalize(vmin=min_vals[0], vmax=max_vals[0])

        # 4. 绘制线条 (平滑曲线 + 厚度编码)
        for i, row in df.iterrows():
            # 获取颜色 (基于原始 Risk 值) 和 线宽 (基于 CD)
            color = cmap(norm(data_list[i][0]))
            linewidth = normalized_cd[i]

            # 离散点 (axes=0, 1, 2)
            y_points = row.values

            # --- 平滑处理 (PCHIP 保界性插值) ---
            # X轴插值范围，增加平滑点数 (100个点)
            x_interp = np.linspace(axes.min(), axes.max(), 100)

            # 进行 3 个点到 100 个点的平滑插值
            pchip_interp = PchipInterpolator(axes, y_points)
            y_interp = pchip_interp(x_interp)

            # 绘制平滑曲线
            ax.plot(
                x_interp,
                y_interp,
                color=color,
                linewidth=linewidth,
                zorder=1,
            )

        # 5. 添加轴标签和刻度
        ax.set_xticks(axes)
        ax.set_xticklabels(labels, fontweight="bold", fontsize=14)
        ax.set_xlim(axes.min(), axes.max())
        ax.set_ylim(0, 1)

        # 6. 添加原始值刻度 (增强可读性)
        # --- 使用 FMT.pprint_val 进行科学计数法格式化 ---
        for i, (label, min_val, max_val) in enumerate(zip(labels, min_vals, max_vals)):
            ax.axvline(x=axes[i], color="k", linestyle="--", linewidth=1, zorder=0)

            # 底部刻度 (0.0)
            ax.text(
                axes[i] - 0.05,
                0.0,
                f"{min_val:.2e}",
                fontsize=10,
                ha="right",
                va="center",
            )
            # 顶部刻度 (1.0)
            ax.text(
                axes[i] - 0.05,
                1.0,
                f"{max_val:.2e}",
                fontsize=10,
                ha="right",
                va="center",
            )

        # 7. 颜色条 (Color Bar)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(normalized_data[:, 0])  # 映射到 Risk
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label(r"Total Risk $f_1$ (people)", fontweight="bold")
        cbar.ax.yaxis.set_major_formatter(FMT)  # 颜色条也格式化

        # 统一格式化
        ax.grid(True, linestyle="--", alpha=0.5)

        # 移除 ax.yaxis.set_major_formatter(FMT) 因为它会作用于归一化后的 [0, 1] 轴

        full_path = os.path.join(self.save_dir, f"{file_name}.tiff")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )

        full_path = os.path.join(self.save_dir, f"{file_name}.png")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close(fig)
        logging.info(f"Gini Parallel chart saved: {file_name} 🌟")

    # --- Helper function ---

    def _highlight_special_solutions(
        self, color: str, solutions_map: Dict[str, Solution], ax
    ):
        """[辅助] 在当前图上标记特殊点"""
        for label, sol in solutions_map.items():
            if not sol:
                continue
            # 绘制特殊点的虚线外框
            ax.scatter(
                [sol.f1_risk],
                [sol.f2_cost],
                facecolors="none",
                edgecolors=color,
                s=220,
                linestyle=":",
                linewidths=2,
                zorder=4,
            )

    def _normalize_metrics_for_comparison(
        self, solutions_data: List[Tuple[float, float, float]]
    ):
        """
        Normalize 3 metrics (Risk, Cost, Gini) to [0, 1] range when plotting Radar.
        """
        data = np.array(solutions_data)

        # Risk, Cost, Gini 都是最小化目标
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)

        ranges = max_vals - min_vals
        # 避免除以零
        ranges[ranges < 1e-9] = 1.0

        normalized = (data - min_vals) / ranges
        return normalized, min_vals, max_vals


class BenchmarkPlotter(BasePlotter):
    """
    [View Layer] 专门负责 Benchmark 实验的绘图
    """

    def __init__(self, save_dir: str):
        super().__init__(save_dir)

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_metrics_comparison(self, stats_data: Dict[str, Dict[str, List[float]]]):
        """
        绘制多算法对比的小提琴图。
        生成独立的图: HV_Comparison, IGD_Comparison, ...
        每张图中 X 轴是算法，Y 轴是指标分布。
        """
        metrics = ["HV", "IGD", "SM", "CPU Time"]

        # 定义优先级排序数组，体现从 Proposed 到退化到基准的递进关系
        PRIORITY = [
            "NSGA-II (Imp)",
            "NSGA-II (1)",
            "NSGA-II (2)",
            "NSGA-II (3)",
            "SPEA2",
            "Gurobi",
        ]

        # 过滤当前数据中存在的算法
        algo_names = [a for a in PRIORITY if a in stats_data.keys()]
        # 处理可能出现的未定义算法
        for a in stats_data.keys():
            if a not in algo_names:
                algo_names.append(a)

        for metric in metrics:
            self._plot_single_metric_violin(metric, algo_names, stats_data)

    # --- Helper function ---

    def _plot_single_metric_violin(
        self, metric: str, algo_names: List[str], stats_data: Dict
    ):
        """[Helper] 绘制单个指标的小提琴图"""
        fig, ax = plt.subplots(figsize=(10, 8))

        data_to_plot = []
        labels = []
        colors: List[str] = self.violin_colors

        for algo in algo_names:
            raw_data = stats_data[algo].get(metric, [])
            clean = [
                x
                for x in raw_data
                if x is not None and not np.isinf(x) and not np.isnan(x)
            ]
            data_to_plot.append(clean if clean else [])
            labels.append(algo)

        # 处理全0方差 (Gurobi) 以防 matplotlib 报错
        processed_data = []
        for d in data_to_plot:
            if len(d) > 0 and np.var(d) < 1e-9:
                d = np.array(d) + np.random.normal(0, 1e-6, size=len(d))
            processed_data.append(d)

        # -----------------------------------------------------
        # 1. Violin (分布形状)
        # -----------------------------------------------------
        # 只为有数据的列绘制 violin
        valid_data_for_violin = []
        valid_positions = []

        for i, d in enumerate(processed_data):
            if len(d) > 0:
                valid_data_for_violin.append(d)
                # i + 1 是因为 matplotlib 的 plot 索引通常从 1 开始
                valid_positions.append(i + 1)

        if valid_data_for_violin:
            try:
                parts = ax.violinplot(
                    valid_data_for_violin,
                    positions=valid_positions,
                    showextrema=False,
                    widths=0.7,
                )
                # 正确匹配颜色: valid_positions[k] 对应的原始索引是 valid_positions[k]-1
                for k, pc in enumerate(parts["bodies"]):
                    original_idx = valid_positions[k] - 1
                    pc.set_facecolor(colors[original_idx % len(colors)])
                    pc.set_edgecolor(get_color_by_key(self.default_colors, "BLACK"))
                    pc.set_alpha(0.6)
            except Exception as e:
                logging.warning(f"Violin plot failed for {metric}: {e}")
        else:
            logging.warning(f"No valid data to plot violin for {metric}")

        # -----------------------------------------------------
        # 2. Boxplot (统计矩)
        # -----------------------------------------------------
        # Boxplot 可以接受包含空列表的列表，它会自动跳过
        try:
            ax.boxplot(
                processed_data,
                widths=0.15,
                patch_artist=True,
                boxprops=dict(
                    facecolor=get_color_by_key(self.default_colors, "WHITE"),
                    alpha=0.9,
                    edgecolor=get_color_by_key(self.default_colors, "BLACK"),
                ),
                medianprops=dict(
                    color=get_color_by_key(self.default_colors, "BLACK"),
                    linewidth=1.5,
                ),
                whiskerprops=dict(color=get_color_by_key(self.default_colors, "BLACK")),
                capprops=dict(color=get_color_by_key(self.default_colors, "BLACK")),
                showfliers=False,
            )
        except Exception as e:
            logging.warning(f"Boxplot failed for {metric}: {e}")

        # -----------------------------------------------------
        # 3. Jitter Scatter (原始数据点)
        # -----------------------------------------------------
        for i, d in enumerate(data_to_plot):
            if len(d) == 0:
                continue
            y = d
            x = np.random.normal(i + 1, 0.1, size=len(y))
            ax.scatter(
                x,
                y,
                alpha=0.6,
                color=get_color_by_key(self.default_colors, "BLACK"),
                s=15,
                zorder=10,
            )

        # 装饰
        ax.set_xticks(range(1, len(algo_names) + 1))
        ax.set_xticklabels(
            labels,
            fontweight="bold",
            rotation=45,  # 倾斜45度
            ha="right",  # 水平对齐方式设为右侧，防止标签中心对准刻度导致偏离
        )

        # 应用科学计数法格式
        ax.yaxis.set_major_formatter(FMT)

        save_path = os.path.join(self.save_dir, f"comparison_{metric}.tiff")
        plt.tight_layout()
        plt.savefig(
            save_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close()

        save_path = os.path.join(self.save_dir, f"comparison_{metric}.png")
        plt.tight_layout()
        plt.savefig(
            save_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close()
        logging.info(f"Generated comparison plot: {save_path}")

    def plot_normalized_metrics_bar(
        self, stats_data, file_name: str = "normalized_bar_comparison"
    ):
        """
        归一化综合对比柱状图。
        将所有指标统一映射到 [0.2, 1.0] 空间，方便在同一个坐标系下比较。
        """
        metrics = ["HV", "IGD", "SM", "CPU Time"]
        algo_names = list(stats_data.keys())
        n_algos = len(algo_names)
        n_metrics = len(metrics)

        # 1. 提取均值并进行极性反转归一化
        mean_matrix = np.array(
            [[np.mean(stats_data[algo][m]) for m in metrics] for algo in algo_names]
        )
        norm_matrix = np.zeros_like(mean_matrix)

        for i, m in enumerate(metrics):
            vals = mean_matrix[:, i]
            v_min, v_max = np.min(vals), np.max(vals)
            if v_max == v_min:
                norm_matrix[:, i] = 1.0
            else:
                if m in ["HV", "PD", "SM"]:  # 越大越好
                    norm_matrix[:, i] = (vals - v_min) / (v_max - v_min)
                else:  # 越小越好 (IGD, SM, CPU Time) 进行反转
                    norm_matrix[:, i] = (v_max - vals) / (v_max - v_min)

        # 加上 0.2 的基础高度保护，防止表现最差的指标“柱子消失”
        norm_matrix = 0.2 + 0.8 * norm_matrix

        # 2. 绘图
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(n_algos)
        width = 0.13  # 每个柱子的宽度

        # 为不同指标分配特定颜色
        metric_colors = self.grouped_bar_colors

        for i in range(n_metrics):
            # 计算每个指标柱子的偏移位置
            offset = (i - n_metrics / 2 + 0.5) * width
            ax.bar(
                x + offset,
                norm_matrix[:, i],
                width,
                label=metrics[i],
                color=metric_colors[i],
                edgecolor=get_color_by_key(self.default_colors, "WHITE"),
                linewidth=0.5,
                alpha=0.9,
            )

        # 装饰
        ax.set_ylabel("Performance Index", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, fontsize=11)
        ax.set_ylim(0, 1.25)

        # 水平参考线
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # 图例放置在上方
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=n_metrics,  # 动态设置列数，确保只有一行
            frameon=False,
            fontsize=10,
        )

        save_path = os.path.join(self.save_dir, f"{file_name}.tiff")
        plt.tight_layout()
        plt.savefig(
            save_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close()

        save_path = os.path.join(self.save_dir, f"{file_name}.png")
        plt.tight_layout()
        plt.savefig(
            save_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close()
        logging.info(f"Normalized bar chart saved: {save_path}")

    def plot_metric_radar(
        self, stats_data, file_name: str = "algorithm_radar_comparison"
    ):
        """
        雷达图（花瓣图）：
        1. 严格控制边界：确保数值不溢出 1.0。
        2. 保留尖点：指标点处保持锐利。
        3. 向内凹陷：尖点之间使用二次曲线或中间缩进点实现弧形。
        4. 视觉精修：降低填充 alpha，精细化线条。
        """
        metrics = ["HV", "IGD", "SM", "CPU Time"]
        algo_names = list(stats_data.keys())
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()

        # 1. 提取均值与归一化逻辑
        mean_data = {
            algo: [np.mean(stats_data[algo][m]) for m in metrics] for algo in algo_names
        }
        data_matrix = np.array([mean_data[algo] for algo in algo_names])
        norm_matrix = np.zeros_like(data_matrix)
        MIN_SHAPE_VAL = 0.2

        for i, m in enumerate(metrics):
            vals = data_matrix[:, i]
            v_min, v_max = np.min(vals), np.max(vals)
            if m in ["IGD", "SM", "CPU Time"]:
                log_vals = np.log10(vals + 1e-9)
                lv_min, lv_max = np.min(log_vals), np.max(log_vals)
                norm_matrix[:, i] = (
                    MIN_SHAPE_VAL
                    + (1 - MIN_SHAPE_VAL) * (lv_max - log_vals) / (lv_max - lv_min)
                    if lv_max != lv_min
                    else 1.0
                )
            else:
                norm_matrix[:, i] = (
                    MIN_SHAPE_VAL
                    + (1 - MIN_SHAPE_VAL) * (vals - v_min) / (v_max - v_min)
                    if v_max != v_min
                    else 1.0
                )

        # 强制截断，确保不溢出
        norm_matrix = np.clip(norm_matrix, 0, 1.0)

        # 2. 绘图准备
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        colors = self.violin_colors

        for idx, algo in enumerate(algo_names):
            values = norm_matrix[idx].tolist()

            # 构造花瓣形状 (保留尖点 + 向内凹陷)
            # 在每两个指标点中间插入一个“向内塌陷”的控制点
            smooth_values = []
            smooth_angles = []

            for i in range(n_metrics):
                # 1. 获取当前尖点 (P0) 和 下一个尖点 (P2)
                p0_v, p0_a = values[i], angles[i]
                p2_v, p2_a = values[(i + 1) % n_metrics], angles[(i + 1) % n_metrics]

                # 处理角度跨越 2π 的闭合情况
                if p2_a <= p0_a:
                    p2_a += 2 * np.pi

                # 2. 定义控制点 (P1): 位于角度中点，但半径向内大幅塌陷
                # mid_factor 决定了凹陷深度，系数越小，花瓣中间缩进越深，看起来越“瘦”：
                #   0.8 丰满、圆润，适合指标差异较小时，增加视觉面积
                #   0.6 干练、优雅，呈现明显的“四叶草”或“花瓣”状，尖点非常突出
                #   0.4 极瘦，花瓣之间会有很深的凹陷，适合强调各个指标之间的独立性
                mid_factor = 0.18 if (p0_v + p2_v) / 2 > 0.3 else 0.12  # 低值区更狠地凹
                mid_v = ((p0_v + p2_v) / 2) * mid_factor

                # 2. 在尖点前加一个极短的“冲刺凸出”（只在高值尖点明显）
                t_steps = np.linspace(0, 1, 40, endpoint=False)  # 增加插值密度，更平滑

                # 3. 使用二次贝塞尔曲线插值
                # B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
                for t in t_steps:
                    # 在尖点附近（t很小）稍微向外凸出一点
                    extra_spike = 0.0
                    if t < 0.1 and p0_v > 0.7:  # 只在高性能尖点加“针尖”效果
                        extra_spike = 0.08 * (1 - t / 0.1)  # 最大凸出8%，很快收回去
                    # 半径 R 的二次插值
                    r = (
                        (1 - t) ** 2 * p0_v
                        + 2 * (1 - t) * t * mid_v
                        + t**2 * p2_v
                        + extra_spike * p0_v
                    )
                    # 角度 Alpha 的线性插值 (确保匀速旋转)
                    a = (1 - t) * p0_a + t * p2_a

                    smooth_values.append(r)
                    smooth_angles.append(a)

            # 闭合曲线
            smooth_values.append(smooth_values[0])
            smooth_angles.append(smooth_angles[0])

            # 4. 绘图执行
            color = colors[idx % len(colors)]
            ax.plot(
                smooth_angles,
                smooth_values,
                color=color,
                linewidth=0.7,
                label=algo,
                zorder=3,
            )
            ax.fill(smooth_angles, smooth_values, color=color, alpha=0.3, zorder=2)

        # 3. 装饰精修
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics, fontweight="bold", fontsize=12)
        ax.set_ylim(0, 1.18)  # 严格限制坐标轴，留足缓冲
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.set_yticklabels([])  # 隐藏圈圈数值

        # 图例放到右外侧
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
            labelspacing=0.4,
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.3,
        )

        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])  # 四周留3%安全区
        save_path = os.path.join(self.save_dir, file_name)
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()


class SensitivityPlotter(BasePlotter):
    """
    [View Layer] 负责 Sensitivity Analysis 的绘图 (支持动态缩放)
    """

    def __init__(self, save_dir: str):
        super().__init__(save_dir)

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_cost_structure_dual_axis(
        self,
        x_labels: List[str],
        cost_data: Dict[str, List[float]],
        risk_data: List[float],
        xlabel: str,
        filename: str,
    ):
        """
        双 Y 轴组合图
        """
        import matplotlib.colors as mcolors
        from matplotlib.ticker import ScalarFormatter

        fig, ax1 = plt.subplots(figsize=(11, 8))

        x = np.arange(len(x_labels))
        width = 0.45  # 柱体宽度

        # 获取颜色配置
        colors = self.stacked_bar_chart_colors
        c_transport = get_color_by_key(colors, "TRANSPORT_COLOR")
        c_transship = get_color_by_key(colors, "TRANSSHIPMENT_COLOR")
        c_risk = get_color_by_key(colors, "RISK_COLOR")
        c_trend = get_color_by_key(colors, "TREND")

        trans = np.array(cost_data["transport"])
        ship = np.array(cost_data["transshipment"])
        total_costs = trans + ship

        # --- A. 视觉增强辅助函数 ---
        def get_light_fill_color(base_color, sat_factor=0.4, alpha=0.25):
            rgb = mcolors.to_rgb(base_color)
            hsv = mcolors.rgb_to_hsv(rgb)
            return (*mcolors.hsv_to_rgb((hsv[0], hsv[1] * sat_factor, hsv[2])), alpha)

        # 构造边缘对齐坐标
        x_fill = []
        for xi in x:
            x_fill.extend([xi - width / 2, xi + width / 2])
        x_fill = np.array(x_fill)

        # y_fill 对应变成: [v0, v0, v1, v1, ...]
        def get_edge_aligned_values(values):
            edge_vals = []
            for vi in values:
                edge_vals.extend([vi, vi])
            return np.array(edge_vals)

        # --- B. 绘制左轴 (ax1): 柱体与对齐填充 ---
        # 1. 绘制堆叠柱状图
        ax1.bar(
            x,
            trans,
            width,
            label="Transport Cost",
            color=c_transport,
            alpha=0.95,
            zorder=3,
        )
        ax1.bar(
            x,
            ship,
            width,
            bottom=trans,
            label="Transshipment Cost",
            color=c_transship,
            alpha=0.95,
            zorder=3,
        )

        # 2. 绘制边缘对齐的填充带 (在柱子之间产生斜向流动效果)
        v_trans = get_edge_aligned_values(trans)
        v_ship = get_edge_aligned_values(trans + ship)

        ax1.fill_between(
            x_fill,
            0,
            v_trans,
            color=get_light_fill_color(c_transport),
            linewidth=0,
            zorder=2,
        )
        ax1.fill_between(
            x_fill,
            v_trans,
            v_ship,
            color=get_light_fill_color(c_transship),
            linewidth=0,
            zorder=2,
        )

        # 3. 总趋势线
        ax1.plot(
            x,
            total_costs,
            color=c_trend,
            linestyle="--",
            linewidth=2,
            label="Total Cost Trend",
            zorder=6,
        )

        # 4. 百分比标注 (仅显示占比 > 3% 的部分)
        for i in range(len(x_labels)):
            if total_costs[i] > 1e-9:
                annot_cfg = {
                    "ha": "center",
                    "va": "center",
                    "fontsize": 10,
                    "color": "white",
                    "fontweight": "bold",
                    "zorder": 5,
                }
                if (trans[i] / total_costs[i]) > 0.03:
                    ax1.text(
                        x[i],
                        trans[i] / 2,
                        f"{(trans[i] / total_costs[i] * 100):.1f}%",
                        **annot_cfg,
                    )
                if (ship[i] / total_costs[i]) > 0.03:
                    ax1.text(
                        x[i],
                        trans[i] + ship[i] / 2,
                        f"{(ship[i] / total_costs[i] * 100):.1f}%",
                        **annot_cfg,
                    )

        # --- C. 绘制右轴 (ax2): 风险曲线 ---
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            risk_data,
            color=c_risk,
            marker="o",
            markersize=8,
            linewidth=2.5,
            label="Total Risk",
            zorder=10,
        )

        # --- D. 科学计数法独立对齐 ---
        fmt_left = ScalarFormatter(useMathText=True)
        fmt_left.set_powerlimits((-2, 3))
        fmt_right = ScalarFormatter(useMathText=True)
        fmt_right.set_powerlimits((-2, 3))

        ax1.yaxis.set_major_formatter(fmt_left)
        ax2.yaxis.set_major_formatter(fmt_right)

        fig.canvas.draw()

        # --- E. 细节修饰 ---
        ax1.set_xlabel(xlabel, fontweight="bold")
        ax1.set_ylabel("Total Cost (yuan)", fontweight="bold", color=c_trend)
        ax1.tick_params(axis="y", labelcolor=c_trend)
        ax2.set_ylabel("Total Risk (people)", fontweight="bold", color=c_risk)
        ax2.tick_params(axis="y", labelcolor=c_risk)

        self._set_dynamic_ylim(ax1, total_costs, is_bar=True)
        self._set_dynamic_ylim(ax2, risk_data)

        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=25, ha="right")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(
            h1 + h2,
            l1 + l2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=5,
            frameon=False,
        )

        ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

        # ── 设置 L 形/U 形边框 ──
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["bottom"].set_linewidth(0.8)
        ax1.spines["left"].set_linewidth(0.8)

        # 由于 ax2 是由于 twinx 产生的，默认会绘制右边框
        # 确保其右侧边框可见，并隐藏其他无用边框
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_linewidth(0.8)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        full_path = os.path.join(self.save_dir, f"{filename}.tiff")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )

        full_path = os.path.join(self.save_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close()
        logging.info(f"✅ Edge-aligned cost structure chart saved: {filename}")

    def plot_dual_line_chart(
        self,
        x_vals: List[str],
        cost_data: List[float],
        risk_data: List[float],
        xlabel: str,
        filename: str,
        legend_loc: str = "upper left",
    ):
        """
        左轴: Cost (Teal Line), 右轴: Risk (Orange Line)
        """
        valid_indices = [
            i
            for i, (c, r) in enumerate(zip(cost_data, risk_data))
            if c is not None and r is not None
        ]
        if not valid_indices:
            return

        xs = [x_vals[i] for i in valid_indices]
        ys_cost = [cost_data[i] for i in valid_indices]
        ys_risk = [risk_data[i] for i in valid_indices]

        fig, ax1 = plt.subplots(figsize=(10, 8))

        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        colors = self.dual_line_chart_colors
        c_cost = get_color_by_key(colors, "COST_LINE")
        c_risk = get_color_by_key(colors, "RISK_LINE")

        # --- 为每个轴创建独立的 Formatter 实例 ---
        sci_fmt_left = ScalarFormatter(useMathText=True)
        sci_fmt_left.set_powerlimits((-2, 3))

        sci_fmt_right = ScalarFormatter(useMathText=True)
        sci_fmt_right.set_powerlimits((-2, 3))

        # --- 左轴 Cost ---
        line1 = ax1.plot(
            xs,
            ys_cost,
            color=c_cost,
            marker="o",
            markersize=6,
            linewidth=2,
            label="Total Cost",
            zorder=3,
        )
        ax1.set_ylabel(r"Total Cost (yuan)", color=c_cost, fontweight="bold")
        ax1.yaxis.set_major_formatter(sci_fmt_left)  # 使用独立的实例
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
        self._set_dynamic_ylim(ax1, ys_cost)

        # --- 右轴 Risk ---
        ax2 = ax1.twinx()
        line2 = ax2.plot(
            xs,
            ys_risk,
            color=c_risk,
            marker="D",
            markersize=6,
            linewidth=2,
            linestyle="--",
            label="Total Risk",
            zorder=3,
        )
        ax2.set_ylabel(r"Total Risk (people)", color=c_risk, fontweight="bold")
        ax2.yaxis.set_major_formatter(sci_fmt_right)  # 使用独立的实例
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
        self._set_dynamic_ylim(ax2, ys_risk)

        # --- 视觉强化设置 ---
        fig.canvas.draw()

        # 强制显示各自的 offset text
        ax1.yaxis.get_offset_text().set_visible(True)
        ax2.yaxis.get_offset_text().set_visible(True)

        # 统一颜色
        ax1.tick_params(axis="y", colors=c_cost, labelsize=11)
        ax2.tick_params(axis="y", colors=c_risk, labelsize=11)

        # 合并图例 (左轴 + 右轴)
        lns = line1 + line2
        labs = [ln.get_label() for ln in lns]
        ax1.legend(lns, labs, loc=legend_loc, frameon=True, fancybox=True)

        # 网格只显示在左轴，避免双重网格叠加造成视觉混乱
        ax1.grid(True, linestyle="--", alpha=0.3)

        # ── 设置 L 形/U 形边框 ──
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["bottom"].set_linewidth(0.8)
        ax1.spines["left"].set_linewidth(0.8)

        # 确保 ax2 的右边框可见
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_linewidth(0.8)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        full_path = os.path.join(self.save_dir, f"{filename}.tiff")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )

        full_path = os.path.join(self.save_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close()
        logging.info(f"Dual line chart (Fixed) saved: {filename}")


class ModelComparisonPlotter(BasePlotter):
    """
    [View Layer] 负责不同模型（如 Proposed vs Static）的多维度对比绘图。
    支持自定义坐标轴标签，并强制执行全局科学计数法格式。
    """

    def __init__(self, save_dir: str):
        super().__init__(save_dir)

    def plot_comparison_heatmap(
        self,
        task_ids: List[str],
        model_labels: List[str],
        data: np.ndarray,
        cbar_label: str,
        filename: str,
        cmap_key: str,
        xlabel: str = "Transport Task IDs",
        ylabel: str = "Model Strategy",
    ):
        """
        绘制对比热图。
        支持多行数据绘制（如 6 行），并动态调整 figsize 以保持视觉清晰度。
        """
        df = pd.DataFrame(data, index=model_labels, columns=task_ids)

        # 根据行数动态调整高度 (每行约 1.2 inch)
        height = max(8, len(model_labels) * 1.2)
        fig, ax = plt.subplots(figsize=(10, height))

        # 1. 对所有量级强制开启科学计数法（复用全局 FMT 逻辑）
        FMT.set_powerlimits((0, 0))
        annot_data = np.vectorize(lambda x: f"${FMT.format_data(x)}$")(data)

        sns.heatmap(
            df,
            annot=annot_data,
            fmt="",
            cmap=get_color_by_key(self.heatmap_colors, cmap_key),
            linewidths=1.5,
            cbar_kws={"label": cbar_label},
            ax=ax,
        )

        # 3. 将全局 FMT 应用到 Colorbar 的刻度上
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(FMT)

        plt.xlabel(xlabel, fontweight="bold")
        plt.ylabel(ylabel, fontweight="bold")

        full_path = os.path.join(self.save_dir, f"{filename}.tiff")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="tiff",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )

        full_path = os.path.join(self.save_dir, f"{filename}.png")
        plt.tight_layout()
        plt.savefig(
            full_path,
            format="png",
            dpi=600,  # 600 dpi 更保险，很多 Top 期刊接受甚至要求
            bbox_inches="tight",
            transparent=True,  # 可选：如果需要去背景
        )
        plt.close(fig)
        logging.info(f"Comparison heatmap saved: {filename} 🌡️")
