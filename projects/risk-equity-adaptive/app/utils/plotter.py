# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from typing import List, Dict, Optional
from app.core.solution import Solution

# --- 全局绘图风格设置 ---
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


class ParetoPlotter:
    """
    [View Layer] 帕累托前沿绘图器
    职责：绘制目标空间 (Objective Space) 的散点图。
    """

    def __init__(self, title: str = "", save_dir: str = "results"):
        self.title = title
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # 设置通用的绘图风格 (可选，提升颜值)
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot(
        self,
        solutions: List[Solution],
        file_name: str = "pareto_frontier.svg",
        xlabel: str = r"Transportation Risk (people)",
        ylabel: str = r"Transportation Cost (yuan)",
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
            # self._mark_special_points(ax, x_r0, y_r0)
            if special_solutions:
                self._highlight_special_solutions(special_solutions, ax)

        # 4. 格式化图表
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.6)

        # 设置坐标轴格式
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        # 图例
        handles, labels = ax.get_legend_handles_labels()

        by_label = dict(zip(labels, handles))
        sorted_keys = [
            k for k in ["Pareto Optimal", "Rank 1", "Rank 2", "Rank 3"] if k in by_label
        ]
        sorted_handles = [by_label[k] for k in sorted_keys]

        # 添加特殊点的图例 (如果存在)
        for k, v in by_label.items():
            if k not in sorted_keys:
                sorted_keys.append(k)
                sorted_handles.append(v)

        ax.legend(sorted_handles, sorted_keys, loc="upper right", frameon=True)

        # 5. 保存
        full_path = os.path.join(self.save_dir, file_name)
        plt.tight_layout()
        plt.savefig(full_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Pareto plot saved to: {full_path}")

    def plot_frontier_comparison(
        self,
        frontiers: Dict[str, List[Solution]],
        file_name: str = "Figure_6_Pareto_Comparison.svg",
        xlabel: str = r"Transportation Risk (people)",
        ylabel: str = r"Transportation Cost (yuan)",
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
                "c": "yellow",
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
        ax.grid(True, linestyle="--", alpha=0.5)

        # 坐标轴格式化 (科学计数法或普通浮点)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fancybox=True)
        ax.set_title("Pareto Frontiers Comparison", fontsize=15, pad=15)

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
                edgecolors="red",
                s=220,
                linestyle=":",
                linewidths=2,
                zorder=4,
            )


class BenchmarkPlotter:
    """
    [View Layer] 专门负责 Benchmark 实验的绘图。
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
        # plt.title(f"{metric_name} Convergence", fontsize=13)
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
        metrics = ["HV", "IGD", "SM", "Time"]
        titles = {
            "HV": "Hypervolume (Higher is Better)",
            "IGD": "IGD (Lower is Better)",
            "SM": "Spacing Metric (Lower is Better)",
            "Time": "CPU Time (s) (Lower is Better)",
        }

        # 获取所有算法名称
        algo_names = list(stats_data.keys())
        # 确保 Proposed 排在第一个 (为了好看)
        if "Improved NSGA-II" in algo_names:
            algo_names.remove("Improved NSGA-II")
            algo_names.insert(0, "Improved NSGA-II")

        for metric in metrics:
            self._plot_single_metric_comparison(
                metric, titles[metric], algo_names, stats_data
            )

    def _plot_single_metric_comparison(
        self, metric: str, title: str, algo_names: List[str], stats_data: Dict
    ):
        """[Helper] 绘制单个指标的对比图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 准备数据列表
        data_to_plot = []
        labels = []
        colors = []

        for algo in algo_names:
            raw_data = stats_data[algo].get(metric, [])
            # 清洗 None, Inf, NaN
            clean = [
                x
                for x in raw_data
                if x is not None and not np.isinf(x) and not np.isnan(x)
            ]

            # 如果数据为空，放一个空列表占位
            data_to_plot.append(clean if clean else [])
            labels.append(algo)
            colors.append(self.colors.get(algo, "gray"))

        # 1. 绘制小提琴 (Violin) - 密度
        # 注意：如果有空数据或方差为0的数据，violinplot 需要特殊处理
        # 这里我们手动处理方差为0的情况（Gurobi）：给它加极小的抖动

        processed_data = []
        for d in data_to_plot:
            if len(d) > 0 and np.var(d) < 1e-9:
                # 方差为0 (Gurobi)，加微小抖动防止 violinplot 报错
                d = np.array(d) + np.random.normal(0, 1e-6, size=len(d))
            processed_data.append(d)

        try:
            parts = ax.violinplot(processed_data, showextrema=False, widths=0.7)

            # 设置颜色
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i])
                pc.set_edgecolor("black")
                pc.set_alpha(0.6)
        except Exception as e:
            print(f"Violin plot warning for {metric}: {e}. Skipping violin body.")

        # 2. 绘制箱线图 (Box) - 统计区间 (嵌在小提琴内部)
        ax.boxplot(
            processed_data,
            widths=0.15,
            patch_artist=True,
            boxprops=dict(facecolor="white", alpha=0.9, edgecolor="black"),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            showfliers=False,
        )  # 不显示离群点，交由 jitter 显示

        # 3. 绘制抖动散点 (Jitter Scatter) - 原始数据
        for i, d in enumerate(data_to_plot):
            y = d
            # x 坐标添加随机抖动: i+1 是因为 boxplot 索引从1开始
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, color="black", s=15, zorder=10)

        # 装饰
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(range(1, len(algo_names) + 1))
        ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

        # 微调 Gurobi 的标签颜色 (可选)
        for tick_label in ax.get_xticklabels():
            if "Improved" in tick_label.get_text():
                tick_label.set_color("#D62728")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"Comparison_{metric}.svg")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   Generated comparison plot: {save_path}")
