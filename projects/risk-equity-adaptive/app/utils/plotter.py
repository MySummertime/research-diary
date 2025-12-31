# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Tuple
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from scipy.interpolate import PchipInterpolator
from app.core.solution import Solution
from app.core.evaluator import Evaluator
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
        self.stacked_bar_chart_colors = ColorPalette.STACKED_BAR
        self.dual_line_chart_colors = ColorPalette.DUAL_LINE
        self.heapmap_colors = ColorPalette.HEATMAP

    def _format_axes(self, ax):
        """
        [辅助] 统一格式化轴刻度，使用全局 FMT 科学计数法规则
        """
        ax.xaxis.set_major_formatter(FMT)
        ax.yaxis.set_major_formatter(FMT)
        ax.grid(True, linestyle="--", alpha=0.5)

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
        file_name: str = "pareto_frontier.svg",
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
        full_path = os.path.join(self.save_dir, file_name)
        plt.tight_layout()
        plt.savefig(full_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Pareto plot saved to: {full_path}")

    def plot_frontier_comparison_by_algo(
        self,
        frontiers: Dict[str, List[Solution]],
        file_name: str = "pareto_comparison_by_algo.svg",
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
        fig, ax = plt.subplots(figsize=(10, 8))

        # 定义各算法的样式
        styles = {
            "Gurobi": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "EXACT"),
                "marker": "*",
                "s": 20,
                "label": "Gurobi",
                "zorder": 5,
                "edgecolors": get_color_by_key(self.pareto_colors_by_algo, "BLACK"),
                "linewidths": 1.5,
            },
            "Improved NSGA-II": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "PROPOSED"),
                "marker": "o",
                "s": 30,
                "label": "Improved NSGA-II",
                "zorder": 4,
            },
            "NSGA-II": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "BASELINE1"),
                "marker": "^",
                "s": 50,
                "label": "NSGA-II",
                "zorder": 3,
            },
            "SPEA2": {
                "c": get_color_by_key(self.pareto_colors_by_algo, "BASELINE2"),
                "marker": "s",
                "s": 60,
                "label": "SPEA2",
                "zorder": 2,
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
        full_path = os.path.join(self.save_dir, file_name)
        plt.tight_layout()
        plt.savefig(full_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Comparison plot saved to: {full_path}")

    def plot_frontier_comparison(
        self,
        frontiers: Dict[str, List[Solution]],
        file_name: str = "pareto_comparison_by_cvar_alpha.svg",
        x_prefix: str = "",
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

        ax.legend(loc="upper right", frameon=True, fancybox=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, file_name))
        plt.close(fig)
        logging.info(f"Pareto comparison saved: {file_name}")

    def plot_gini_tradeoff(
        self,
        solutions_with_gini: List[Tuple[Solution, float]],
        file_name: str = "Figure_Gini_Risk_Tradeoff.svg",
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

        # 5. 添加颜色条 (Color Bar)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label(r"Total Cost $f_2$ (yuan)", fontweight="bold")
        cbar.ax.yaxis.set_major_formatter(FMT)  # 颜色条也格式化

        # 6. 格式化
        ax.set_xlabel(r"Total Risk $f_1$ (people)", fontweight="bold")
        ax.set_ylabel("Gini Coefficient (Social Equity)", fontweight="bold")
        ax.set_title("Trade-off between Total Risk and Risk Equity", fontsize=15)

        self._format_axes(ax)
        self._set_dynamic_xlim(ax, r0_risk)
        self._set_dynamic_ylim(ax, r0_gini)

        ax.legend(loc="upper right", frameon=True, fancybox=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, file_name), format="svg", bbox_inches="tight"
        )
        plt.close(fig)
        logging.info(f"Gini Trade-off plot saved with Cost CMAP: {file_name} 🌟")

    def plot_parallel_coordinates(
        self,
        rank0_solutions: List[Solution],
        gini_calculator: callable,
        evaluator: Evaluator,
        file_name: str = "parallel_coordinates_tradeoff.svg",
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

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, file_name), format="svg", bbox_inches="tight"
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
        生成 4 张独立的图: HV_Comparison.svg, IGD_Comparison.svg, ...
        每张图中 X 轴是算法，Y 轴是指标分布。
        """
        metrics = ["HV", "IGD", "SM", "CPU Time"]
        algo_names = list(stats_data.keys())

        # 确保 Proposed 排在第一个 (为了好看)
        if "Improved NSGA-II" in algo_names:
            algo_names.remove("Improved NSGA-II")
            algo_names.insert(0, "Improved NSGA-II")

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
                    color=get_color_by_key(self.default_colors, "BLACK"), linewidth=1.5
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
            x = np.random.normal(i + 1, 0.04, size=len(y))
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
        ax.set_xticklabels(labels, fontweight="bold")

        # 应用科学计数法格式
        ax.yaxis.set_major_formatter(FMT)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"comparison_{metric}.svg")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Generated comparison plot: {save_path}")


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
        双 Y 轴组合图：
            左轴 (ax1): Cost (Stacked Bar + Trend Lines)
            右轴 (ax2): Risk (Line)
            x轴: CVaR Confidence level
        """
        fig, ax1 = plt.subplots(figsize=(10, 8))

        x = np.arange(len(x_labels))
        width = 0.4  # 柱状图宽度

        colors = self.stacked_bar_chart_colors
        c_transport = get_color_by_key(colors, "TRANSPORT_COLOR")
        c_transship = get_color_by_key(colors, "TRANSSHIPMENT_COLOR")
        c_carbon = get_color_by_key(colors, "CARBON_COLOR")
        c_risk = get_color_by_key(colors, "RISK_COLOR")
        c_trend = get_color_by_key(colors, "TREND")

        trans = np.array(cost_data["transport"])
        ship = np.array(cost_data["transshipment"])
        carb = np.array(cost_data["carbon"])
        total_costs = trans + ship + carb

        # --- 左轴: Cost (Stacked Bar + Trend Lines) ---
        # 淡化函数（饱和度降低 + 透明）
        def get_light_fill_color(
            base_color: str, sat_factor=0.5, light_factor=1.1, alpha=0.7
        ):
            rgb = mcolors.to_rgb(base_color)
            hsv = mcolors.rgb_to_hsv(rgb)
            hsv = (
                hsv[0],
                hsv[1] * sat_factor,
                hsv[2] * light_factor,
            )  # 稍微提亮，让它更柔和
            light_rgb = mcolors.hsv_to_rgb(hsv)
            light_rgb = tuple(min(max(c, 0.0), 1.0) for c in light_rgb)
            return (*light_rgb, alpha)

        # 生成每层的填充颜色（超级柔和）
        fill_transport = get_light_fill_color(c_transport, sat_factor=0.5, alpha=0.5)
        fill_transship = get_light_fill_color(c_transship, sat_factor=0.5, alpha=0.5)
        fill_carbon = get_light_fill_color(c_carbon, sat_factor=0.5, alpha=0.5)

        # --- 1. 绘制 Stacked Bar Chart ---
        ax1.bar(
            x,
            trans,
            width,
            label="Transport Cost",
            color=c_transport,
            alpha=0.9,
            edgecolor=get_color_by_key(self.default_colors, "WHITE"),
            zorder=1,
        )
        ax1.bar(
            x,
            ship,
            width,
            bottom=trans,
            label="Transshipment Cost",
            color=c_transship,
            alpha=0.9,
            edgecolor=get_color_by_key(self.default_colors, "WHITE"),
            zorder=1,
        )
        ax1.bar(
            x,
            carb,
            width,
            bottom=trans + ship,
            label="Carbon Cost",
            color=c_carbon,
            alpha=0.9,
            edgecolor=get_color_by_key(self.default_colors, "WHITE"),
            zorder=1,
        )

        # --- 2. 实现“从右侧到左侧连续的层间填充带” ---
        # 构造扩展的 x 坐标：每个 bar 的左边缘 -> 右边缘 -> 下一个 bar 的左边缘（平滑连接）
        x_extended = np.repeat(x, 2)  # 每个点重复两次
        x_fill = np.concatenate(
            [[x[0] - width / 2], x_extended, [x[-1] + width / 2]]
        )  # 首尾额外延伸一点，美观

        # 对每一层的高度也做对应的“step”扩展（右边缘保持值，左边缘用下一个值）
        def step_post(values):
            extended = np.repeat(values, 2)
            return np.concatenate([[values[0]], extended, [values[-1]]])

        # Transport 层填充：从0到trans（底部大块）
        ax1.fill_between(
            x_fill, 0, step_post(trans), color=fill_transport, zorder=2, linewidth=0
        )

        # Transshipment 层填充：从trans到trans+ship
        ax1.fill_between(
            x_fill,
            step_post(trans),
            step_post(trans + ship),
            color=fill_transship,
            zorder=2,
            linewidth=0,
        )

        # Carbon 层填充：从trans+ship到total（顶部小帽子）
        ax1.fill_between(
            x_fill,
            step_post(trans + ship),
            step_post(total_costs),
            color=fill_carbon,
            zorder=2,
            linewidth=0,
        )

        # 虚线总趋势
        ax1.plot(
            x,
            total_costs,
            color=c_trend,
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            label="Total Cost Trend",
            zorder=6,
        )

        # --- 3. 添加百分比标签 ---

        # 1. 计算所有成本占总成本的百分比
        total_costs_np = np.array(total_costs)
        # 避免除以零或无穷大
        valid_totals = np.where(total_costs_np > 1e-9, total_costs_np, 1.0)

        trans_perc = (trans / valid_totals) * 100
        ship_perc = (ship / valid_totals) * 100
        carb_perc = (carb / valid_totals) * 100

        # 2. 准备标签位置 (每段的中心点)
        trans_y = trans / 2.0
        ship_y = trans + ship / 2.0
        carb_y = trans + ship + carb / 2.0

        # 3. 循环添加文本注释
        cost_components = [
            (trans_perc, trans_y, "Transport Cost"),
            (ship_perc, ship_y, "Transshipment Cost"),
            (carb_perc, carb_y, "Carbon Cost"),
        ]

        for i in range(len(x_labels)):
            if total_costs_np[i] < 1e-9:
                continue

            annot_style = dict(
                ha="center",
                va="center",
                fontsize=12,
                color=get_color_by_key(self.default_colors, "BLACK"),
            )

            for j, (perc_array, center_y_array, label) in enumerate(cost_components):
                percentage = perc_array[i]
                center_y = center_y_array[i]

                # 只在百分比大于 2.0% 时显示标签
                if percentage > 2.0:
                    text_label = f"{percentage:.1f}%"
                    ax1.text(x[i], center_y, text_label, **annot_style)

        ax1.set_ylabel("Min Cost (yuan)", fontweight="bold", color=c_trend)
        ax1.tick_params(axis="y", labelcolor=c_trend)

        # [Auto-Scale Cost: Broken Axis Effect]
        self._set_dynamic_ylim(ax1, total_costs, is_bar=True)

        # --- 右轴: Risk (Line) ---
        ax2 = ax1.twinx()
        # [Straight Line] 直线连接，清晰展示 Trend
        ax2.plot(
            x,
            risk_data,
            color=c_risk,
            marker="o",
            markersize=8,
            linewidth=2.5,
            linestyle="-",
            label="Min Risk",
            zorder=10,
        )
        ax2.set_ylabel(r"Min Risk (people)", fontweight="bold", color=c_risk)
        ax2.tick_params(axis="y", labelcolor=c_risk)

        # [Auto-Scale Risk]
        self._set_dynamic_ylim(ax2, risk_data)

        # Format (应用全局 FMT)
        self._format_axes(ax1)  # Formats x-axis and ax1 y-axis
        ax2.yaxis.set_major_formatter(FMT)

        # 局部强制覆盖：确保 ax1 和 ax2 使用科学计数法，并使用 10^xx 格式
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)

        # 强制显示左轴的科学计数法乘数
        ax1.yaxis.get_offset_text().set_visible(True)

        # X轴刻度设置必须在所有格式化之后
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=25, ha="right")
        ax1.set_xlabel(xlabel, fontweight="bold")

        # Legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()

        # 新增了 Total Cost Trend，需要确保 Legend 能够容纳
        ax1.legend(
            h1 + h2,
            l1 + l2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=5,  # 增加列数来适应新增的 Total Cost Trend
            frameon=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        logging.info(f"Dual axis chart saved: {filename}")

    def plot_dual_line_chart(
        self,
        x_vals: List[str],
        cost_data: List[float],
        risk_data: List[float],
        xlabel: str,
        filename: str,
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

        colors = self.dual_line_chart_colors
        c_cost = get_color_by_key(colors, "COST_LINE")
        c_risk = get_color_by_key(colors, "RISK_LINE")

        # --- 左轴 Cost ---
        line1 = ax1.plot(
            xs,
            ys_cost,
            color=c_cost,
            marker="o",
            markersize=5,
            linewidth=2,
            label="Min Cost",
        )
        ax1.set_ylabel(r"Min Cost (yuan)", color=c_cost, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=c_cost)

        # Auto-Scale Cost
        self._set_dynamic_ylim(ax1, ys_cost)

        # --- 右轴 Risk ---
        ax2 = ax1.twinx()
        line2 = ax2.plot(
            xs,
            ys_risk,
            color=c_risk,
            marker="D",
            markersize=5,
            linewidth=2,
            linestyle="--",
            label="Min Risk",
        )
        ax2.set_ylabel(r"Min Risk (people)", color=c_risk, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=c_risk)

        # Auto-Scale Risk
        self._set_dynamic_ylim(ax2, ys_risk)

        # 1. 应用全局格式
        self._format_axes(ax1)  # 保持全局格式化 X 轴和部分通用设置

        # 2. 局部强制覆盖：确保 ax1 和 ax2 使用科学计数法，并使用 10^xx 格式
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)

        # 3. 强制刻度密度 (MultipleLocator 必须在 ticklabel_format 之后)

        # Cost (ax1) 的刻度间隔 (5000)
        ax1.yaxis.set_major_locator(MultipleLocator(5000))
        # Risk (ax2) 的刻度间隔 (1000，以保证密度)
        ax2.yaxis.set_major_locator(MultipleLocator(1000))

        # 设置x轴刻度的位置
        ax1.set_xticks(xs)
        # 设置x轴每个刻度的标签
        ax1.set_xticklabels(xs, rotation=25, ha="right")
        ax1.set_xlabel(xlabel, fontweight="bold")

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", frameon=True, fancybox=True)
        ax1.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        logging.info(f"Dual line chart saved: {filename}")

    def plot_emergency_uncertainty_heatmap(
        self,
        delta_a_values: List[float],
        delta_c_values: List[float],
        cost_grid: List[List[float]],
        risk_grid: List[List[float]],
        prefix: str = "Figure_Emergency_Asymmetric",
    ):
        """
        绘制两张热力图: Min Cost 和 Min Risk（在 δ_a * δ_c 网格上）。
        """
        # 1. 转换为 DataFrame
        cost_df = pd.DataFrame(cost_grid, index=delta_a_values, columns=delta_c_values)
        risk_df = pd.DataFrame(risk_grid, index=delta_a_values, columns=delta_c_values)

        # 2. 获取颜色配置
        colors = self.heapmap_colors
        # 尝试获取配置好的 CMAP，如果失败则使用学术常用的 GnBu 和 OrRd
        cost_cmap = get_color_by_key(colors, "COST_CMAP") if colors else "GnBu"
        risk_cmap = get_color_by_key(colors, "RISK_CMAP") if colors else "OrRd"

        # 3. 处理无效值 (None 或 np.nan 在热力图中显示为空白)
        cost_df = cost_df.fillna(np.nan)
        risk_df = risk_df.fillna(np.nan)

        # --- 图1: Min Cost Heatmap ---
        fig, ax = plt.subplots(figsize=(11, 8.5))

        # 使用标准 f-string 科学计数法格式化标注，保留 2 位小数
        cost_annot_fmt = lambda x: f"{x:.2e}" if not np.isnan(x) else ""
        cost_annot_values = cost_df.applymap(cost_annot_fmt)

        sns.heatmap(
            cost_df,
            annot=cost_annot_values,
            fmt="s",  # 声明标注是字符串类型
            cmap=cost_cmap,
            linewidths=0.5,
            cbar_kws={"label": "Min Cost (yuan)", "shrink": 0.8},
            mask=cost_df.isna(),
            annot_kws={"size": 10},
            ax=ax,
        )

        # 格式化颜色条 (Colorbar) 使用我们定义的科学计数法 FMT
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(FMT)

        plt.xlabel(r"Pessimistic Multiplier $\delta_c$", fontweight="bold", fontsize=12)
        plt.ylabel(r"Optimistic Multiplier $\delta_a$", fontweight="bold", fontsize=12)
        plt.title(
            "System Cost Sensitivity Under Asymmetric Uncertainty", fontsize=15, pad=20
        )

        plt.tight_layout()
        cost_filename = f"{prefix}_Cost_Heatmap.svg"
        plt.savefig(os.path.join(self.save_dir, cost_filename), dpi=300)
        plt.close(fig)
        logging.info(f"Fixed Cost heatmap saved: {cost_filename} 🌡️✨")

        # --- 图2: Min Risk Heatmap ---
        fig, ax = plt.subplots(figsize=(11, 8.5))

        # 风险热力图使用科学计数法标注
        risk_annot_fmt = lambda x: f"{x:.2e}" if not np.isnan(x) else ""
        risk_annot_values = risk_df.applymap(risk_annot_fmt)

        sns.heatmap(
            risk_df,
            annot=risk_annot_values,
            fmt="s",
            cmap=risk_cmap,
            linewidths=0.5,
            cbar_kws={"label": "Min Risk (people)", "shrink": 0.8},
            mask=risk_df.isna(),
            annot_kws={"size": 10},
            ax=ax,
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(FMT)

        plt.xlabel(r"Pessimistic Multiplier $\delta_c$", fontweight="bold", fontsize=12)
        plt.ylabel(r"Optimistic Multiplier $\delta_a$", fontweight="bold", fontsize=12)
        plt.title(
            "System Risk Sensitivity Under Asymmetric Uncertainty", fontsize=15, pad=20
        )

        plt.tight_layout()
        risk_filename = f"{prefix}_Risk_Heatmap.svg"
        plt.savefig(os.path.join(self.save_dir, risk_filename), dpi=300)
        plt.close(fig)
        logging.info(f"Fixed Risk heatmap saved: {risk_filename} 🔥✨")
