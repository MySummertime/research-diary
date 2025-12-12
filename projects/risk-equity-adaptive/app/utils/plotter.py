# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import os
import logging
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from app.core.solution import Solution


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


class ParetoPlotter:
    """
    [View Layer] 通用 Pareto Frontier 绘图器
    职责：绘制目标空间 (Objective Space) 的散点图。
    """

    def __init__(self, title: str = "", save_dir: str = "results"):
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
        rank_styles = {
            1: {
                "color": "#666666",
                "marker": "s",
                "label": "Rank 1",
                "s": 40,
                "alpha": 0.7,
            },
            2: {
                "color": "#999999",
                "marker": "^",
                "label": "Rank 2",
                "s": 35,
                "alpha": 0.6,
            },
            3: {
                "color": "#CCCCCC",
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
            ax.plot(x_r0, y_r0, color="#3498DB", linestyle="--", alpha=0.6, zorder=2)

            # 画散点 (蓝边白底圆形)
            ax.scatter(
                x_r0,
                y_r0,
                facecolors="white",
                edgecolors="#2980B9",
                linewidths=1.5,
                marker="o",
                s=60,
                label="Pareto Optimal",
                zorder=3,
            )

            # 标记 Rank 0 的特殊点
            if special_solutions:
                self._highlight_special_solutions(special_solutions, ax)

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
                "c": "#EED960",  # 暖金黄
                "marker": "*",  # 星号
                "s": 20,
                "label": "Gurobi",
                "zorder": 4,
                "edgecolors": "yellow",
            },
            "Improved NSGA-II": {
                "c": "#D62728",
                "marker": "o",
                "s": 30,
                "label": "Improved NSGA-II",
                "zorder": 3,
                "edgecolors": "red",
            },
            "NSGA-II": {
                "c": "#2CA02C",  # 绿
                "marker": "^",  # 三角
                "s": 50,
                "label": "SPEA2",
                "zorder": 2,
            },
            "SPEA2": {
                "c": "#1F77B4",  # 蓝
                "marker": "s",  # 方块
                "s": 60,
                "label": "NSGA-II",
                "zorder": 1,
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
                {"c": "gray", "marker": "x", "s": 30, "label": algo_name, "zorder": 1},
            )

            ax.scatter(
                x_vals,
                y_vals,
                c=style["c"],
                marker=style["marker"],
                s=style["s"],
                label=style["label"],
                zorder=style.get("zorder", 1),
                alpha=style.get("alpha", 1.0),
                edgecolors=style.get("edgecolors", "none"),
                linewidths=style.get("linewidths", 0),
            )

        # 格式化
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fancybox=True)

        # 保存
        full_path = os.path.join(self.save_dir, file_name)
        plt.tight_layout()
        plt.savefig(full_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Comparison plot saved to: {full_path}")

    def plot_frontier_comparison_by_cvar_alpha(
        self,
        frontiers: Dict[str, List[Solution]],
        file_name: str = "pareto_comparison_by_cvar_alpha.svg",
    ):
        """
        绘制不同 CVaR alpha 下的 Pareto Frontier 对比图。
        X=Risk, Y=Cost
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        keys = list(frontiers.keys())
        cmap = plt.get_cmap("viridis_r")
        colors = [cmap(i) for i in np.linspace(0, 1, len(keys))]
        markers = ["o", "s", "D", "^", "v", "<", ">"]

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
                label=r"$\alpha$={label}",
                color=color,
                marker=marker,
                s=80,
                edgecolors="white",
                alpha=0.9,
                zorder=3,
            )

        ax.set_xlabel(r"Total Risk (people)", fontweight="bold")
        ax.set_ylabel(r"Total Cost (yuan)", fontweight="bold")

        self._format_axes(ax)
        self._set_dynamic_limits(ax, all_x, all_y)

        ax.legend(loc="upper right", frameon=True, fancybox=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, file_name))
        plt.close(fig)
        logging.info(f"Pareto comparison saved: {file_name}")

    # --- Helper function ---

    def _highlight_special_solutions(self, solutions_map: Dict[str, Solution], ax):
        """
        [辅助] 在当前图上标记特殊点
        """
        for label, sol in solutions_map.items():
            if not sol:
                continue

            # 绘制红色虚线外框
            ax.scatter(
                [sol.f1_risk],
                [sol.f2_cost],
                facecolors="none",
                edgecolors="#D62728",
                s=220,
                linestyle=":",
                linewidths=2,
                zorder=4,
            )

    def _format_axes(self, ax):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle="--", alpha=0.5)

    def _set_dynamic_limits(self, ax, x_data, y_data, margin=0.05):
        """辅助：设置动态坐标轴范围"""
        if not x_data or not y_data:
            return
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)

        dx = (x_max - x_min) * margin if x_max != x_min else x_min * margin
        dy = (y_max - y_min) * margin if y_max != y_min else y_min * margin

        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)


class BenchmarkPlotter:
    """
    [View Layer] 专门负责 Benchmark 实验的绘图
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 定义算法颜色方案 (Highlight Proposed)
        self.colors = {
            "Improved NSGA-II": "#D62728",  # Proposed: Red
            "NSGA-II": "#1F77B4",  # Baseline 1: Blue
            "SPEA2": "#2CA02C",  # Baseline 2: Green
            "Gurobi": "#7F7F7F",  # Benchmark: Gray
        }

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
        colors = []

        for algo in algo_names:
            raw_data = stats_data[algo].get(metric, [])
            clean = [
                x
                for x in raw_data
                if x is not None and not np.isinf(x) and not np.isnan(x)
            ]
            data_to_plot.append(clean if clean else [])
            labels.append(algo)
            colors.append(self.colors.get(algo, "gray"))

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
                    pc.set_facecolor(colors[original_idx])
                    pc.set_edgecolor("black")
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
                boxprops=dict(facecolor="white", alpha=0.9, edgecolor="black"),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
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
            ax.scatter(x, y, alpha=0.6, color="black", s=15, zorder=10)

        # 装饰
        ax.set_xticks(range(1, len(algo_names) + 1))
        ax.set_xticklabels(labels, fontweight="bold")

        # 高亮 Proposed 算法的标签
        for tick_label in ax.get_xticklabels():
            if "Improved" in tick_label.get_text():
                tick_label.set_color("#D62728")
                tick_label.set_fontweight("bold")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"comparison_{metric}.svg")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Generated comparison plot: {save_path}")


class SensitivityPlotter:
    """
    [View Layer] 负责 Sensitivity Analysis 的绘图 (支持动态缩放)
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_cost_structure_dual_axis(
        self,
        x_labels: List[str],
        cost_data: Dict[str, List[float]],
        risk_data: List[float],
        filename: str,
    ):
        """
        双 Y 轴组合图：
            左轴 (ax1): Cost (Stacked Bar)
            右轴 (ax2): Risk (Line)
            x轴: CVaR Confidence level
        """
        fig, ax1 = plt.subplots(figsize=(12, 7))
        x = np.arange(len(x_labels))
        width = 0.4

        trans = np.array(cost_data["transport"])
        ship = np.array(cost_data["transshipment"])
        carb = np.array(cost_data["carbon"])
        total_costs = trans + ship + carb

        # --- 左轴: Cost (Bar) ---
        c1, c2, c3 = "#3498db", "#95a5a6", "#2ecc71"
        ax1.bar(
            x,
            trans,
            width,
            label="Transport Cost",
            color=c1,
            alpha=0.7,
            edgecolor="white",
            zorder=1,
        )
        ax1.bar(
            x,
            ship,
            width,
            bottom=trans,
            label="Transshipment Cost",
            color=c2,
            alpha=0.7,
            edgecolor="white",
            zorder=1,
        )
        ax1.bar(
            x,
            carb,
            width,
            bottom=trans + ship,
            label="Carbon Cost",
            color=c3,
            alpha=0.7,
            edgecolor="white",
            zorder=1,
        )

        ax1.set_ylabel("Total Cost (yuan)", fontweight="bold", color="#2c3e50")
        ax1.tick_params(axis="y", labelcolor="#2c3e50")
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=25, ha="right")
        ax1.set_xlabel(r"CVaR Confidence Level $\alpha$", fontweight="bold")

        # [Auto-Scale Cost: Broken Axis Effect]
        self._set_dynamic_ylim(ax1, total_costs, is_bar=True)

        # --- 右轴: Risk (Line) ---
        ax2 = ax1.twinx()
        c_risk = "#e74c3c"
        # [Straight Line] 直线连接，清晰展示 Trend
        ax2.plot(
            x,
            risk_data,
            color=c_risk,
            marker="D",
            markersize=8,
            linewidth=1.5,
            linestyle="-",
            label="Min Risk",
            zorder=10,
        )
        ax2.set_ylabel(r"Min Risk (people)", fontweight="bold", color=c_risk)
        ax2.tick_params(axis="y", labelcolor=c_risk)

        # [Auto-Scale Risk]
        self._set_dynamic_ylim(ax2, risk_data)

        # Format
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 3))
        ax1.yaxis.set_major_formatter(fmt)
        ax2.yaxis.set_major_formatter(fmt)

        # Legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(
            h1 + h2,
            l1 + l2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=4,
            frameon=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        logging.info(f"Dual axis chart saved: {filename}")

    def plot_dual_line_chart(
        self,
        x_vals: List[float],
        cost_data: List[float],
        risk_data: List[float],
        xlabel: str,
        filename: str,
        x_ticks: Optional[List[float]] = None,
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

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # --- 左轴 Cost ---
        c_cost = "#16a085"  # Teal
        line1 = ax1.plot(
            xs,
            ys_cost,
            color=c_cost,
            marker="o",
            markersize=8,
            linewidth=2.5,
            label="Min Expected Cost",
        )
        ax1.set_ylabel(r"Expected Cost (yuan)", color=c_cost, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=c_cost)
        ax1.set_xlabel(xlabel, fontweight="bold")

        if x_ticks:
            ax1.set_xticks(x_ticks)
        else:
            ax1.set_xticks(xs)

        # Auto-Scale Cost
        self._set_dynamic_ylim(ax1, ys_cost)

        # --- 右轴 Risk ---
        ax2 = ax1.twinx()
        c_risk = "#f39c12"  # Orange
        line2 = ax2.plot(
            xs,
            ys_risk,
            color=c_risk,
            marker="D",
            markersize=8,
            linewidth=2.5,
            linestyle="--",
            label="Min CVaR Risk",
        )
        ax2.set_ylabel(r"CVaR Risk (people)", color=c_risk, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=c_risk)

        # Auto-Scale Risk
        self._set_dynamic_ylim(ax2, ys_risk)

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", frameon=True, fancybox=True)
        ax1.grid(True, linestyle="--", alpha=0.5)

        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 3))
        ax1.yaxis.set_major_formatter(fmt)
        ax2.yaxis.set_major_formatter(fmt)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        logging.info(f"Dual line chart saved: {filename}")

    # --- Helper function ---

    def _set_dynamic_ylim(self, ax, data, margin_ratio=0.1, is_bar=False):
        """
        [辅助] 动态设置 Y 轴范围 (支持截断模式)
        """
        if len(data) == 0:
            return
        ymin, ymax = min(data), max(data)

        # 柱状图强制从 0 开始
        if is_bar:
            ax.set_ylim(0, ymax * 1.1)
            return

        # 其他图（折线图）使用动态缩放
        if ymax > 0:
            diff = ymax - ymin
            variation = diff / ymax

            # 如果变化率极小 (<5%)，启用聚焦模式
            if variation < 0.05:
                # 聚焦微小变化
                margin = diff if diff > 0 else ymax * 0.01
                lower = max(0, ymin - margin * 2)
                upper = ymax + margin * 2
                ax.set_ylim(lower, upper)
            else:
                # 正常波动，预留 10% 边距
                margin = diff * 0.1
                ax.set_ylim(max(0, ymin - margin), ymax + margin)
