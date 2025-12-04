# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import os
import logging
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
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot(
        self,
        solutions: List[Solution],
        file_name: str = "pareto_frontier.svg",
        xlabel: str = r"Transportation Risk $10^5$ (people $\cdot$ t)",
        ylabel: str = r"Transportation Cost $10^4$ (yuan)",
        special_solutions: Optional[Dict[str, Solution]] = None
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
                facecolors='none',
                edgecolors='red',
                s=220,
                linestyle=":",
                linewidths=2,
                zorder=4,
            )