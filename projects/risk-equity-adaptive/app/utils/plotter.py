# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import List, Dict, Optional
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
    [View Layer] 帕累托前沿绘图器
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
        绘制二维帕累托前沿图，并高亮特殊解。

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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Scientific Notation
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

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

    def plot_frontier_comparison(
        self,
        frontiers: Dict[str, List[Solution]],
        file_name: str = "pareto_comparison.svg",
        xlabel: str = r"Total Risk (people)",
        ylabel: str = r"Total Cost (yuan)",
    ):
        """
        绘制多算法 Pareto 前沿对比图。

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


class BenchmarkPlotter:
    """
    [View Layer] 专门负责 Benchmark 实验的绘图（Violin & Dual-Axis Sensitivity）
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

    def plot_convergence_curves(
        self, data_map: Dict[str, List[float]], metric_name: str, filename: str
    ):
        """绘制收敛曲线 (HV, IGD, SM)"""
        plt.figure(figsize=(10, 6))
        for name, history in data_map.items():
            if not history:
                continue

            # 样式处理
            color = self.colors.get(name, "gray")
            lw = 1.5
            ls = "-" if "Improved" in name else "--"
            if "Gurobi" in name:
                pass

            plt.plot(history, label=name, color=color, linewidth=lw, linestyle=ls)

        plt.xlabel("Generation", fontsize=12, fontweight="bold")
        plt.ylabel(metric_name, fontsize=12, fontweight="bold")

        plt.legend(fontsize=11, frameon=True, fancybox=True, framealpha=0.9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plt.savefig(os.path.join(self.save_dir, f"{filename}.svg"))
        plt.close()

    def plot_performance_comparison(
        self, stats_data: Dict[str, Dict[str, List[float]]]
    ):
        """
        绘制多算法对比的小提琴图。
        生成 4 张独立的图：HV_Comparison.svg, IGD_Comparison.svg, ...
        每张图中 X 轴是算法，Y 轴是指标分布。
        """
        metrics = ["HV", "IGD", "SM", "CPU Time"]
        titles = {
            "HV": "Hypervolume (Higher is Better)",
            "IGD": "IGD (Lower is Better)",
            "SM": "Spacing Metric (Lower is Better)",
            "CPU Time": "CPU Time (s) (Lower is Better)",
        }

        # 获取所有算法名称
        algo_names = list(stats_data.keys())
        # 确保 Proposed 排在第一个 (为了好看)
        if "Improved NSGA-II" in algo_names:
            algo_names.remove("Improved NSGA-II")
            algo_names.insert(0, "Improved NSGA-II")

        for metric in metrics:
            self._plot_single_metric_violin(
                metric, titles[metric], algo_names, stats_data
            )

    def _plot_single_metric_violin(
        self, metric: str, title: str, algo_names: List[str], stats_data: Dict
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

    def plot_dual_sensitivity_curve(
        self,
        x_vals: List[float],
        cost_vals: List[float],
        risk_vals: List[float],
        xlabel: str,
        filename: str,
    ):
        """
        双 Y 轴灵敏度分析图
        左轴 Cost (Teal/Green), 右轴 Risk (Orange), 空心标记
        """
        # 过滤无效数据 (None)
        valid_indices = [
            i
            for i, (c, r) in enumerate(zip(cost_vals, risk_vals))
            if c is not None and r is not None
        ]
        if not valid_indices:
            logging.warning(f"No valid data for {filename}")
            return

        xs = [x_vals[i] for i in valid_indices]
        ys_cost = [cost_vals[i] for i in valid_indices]
        ys_risk = [risk_vals[i] for i in valid_indices]

        fig, ax1 = plt.subplots(figsize=(10, 8))

        # --- 设置风格颜色 ---
        color_cost = "#72AACF"
        color_risk = "#FDB96B"

        # --- 左轴: Cost ---
        ax1.set_xlabel(xlabel, fontweight="bold")
        ax1.set_ylabel(r"Min Cost (yuan)", color=color_cost, fontweight="bold")

        # 强制设置 X 轴刻度，使其与输入 xs 完全一致
        ax1.set_xticks(xs) 
        # 可选：如果 tick label 太密，可以旋转
        # ax1.set_xticklabels(xs, rotation=45)

        # 绘制 Cost 曲线
        line1 = ax1.plot(
            xs,
            ys_cost,
            color=color_cost,
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            linewidth=2,
            label="Cost Objective",
        )
        ax1.tick_params(axis="y", labelcolor=color_cost)

        # Cost 轴科学计数法
        formatter1 = ScalarFormatter(useMathText=True)
        formatter1.set_powerlimits((-2, 3))
        ax1.yaxis.set_major_formatter(formatter1)

        # --- 右轴: Risk ---
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"Min Risk (people)", color=color_risk, fontweight="bold")

        # 绘制 Risk 曲线
        line2 = ax2.plot(
            xs,
            ys_risk,
            color=color_risk,
            marker="D",
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            linewidth=2,
            label="Risk Objective",
        )
        ax2.tick_params(axis="y", labelcolor=color_risk)

        # Risk 轴科学计数法
        formatter2 = ScalarFormatter(useMathText=True)
        formatter2.set_powerlimits((-2, 3))
        ax2.yaxis.set_major_formatter(formatter2)

        # --- 合并图例 ---
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(
            lines, labels, loc="upper left", frameon=True, fancybox=True, framealpha=0.9
        )

        # 标题和网格
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 保存
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Dual sensitivity plot saved to: {save_path}")
