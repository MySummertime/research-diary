# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from typing import List
from app.core.solution import Solution


# --- 全局绘图风格设置 (Publication Quality) ---

# 1. 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
# 2. 确保数学公式使用衬线体
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
# 3. 设置刻度方向向内
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


# 科学计数法 + 保留1位小数
# 定义一个科学计数法格式化器，用于 X 和 Y 轴
class MathTextSciFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%.1e"  # 强制科学计数法，保留1位小数


class ParetoPlotter:
    """
    一个专门用于可视化帕累托前沿的类
    支持分层显示 (Rank 0, Rank 1, Rank 2...)
    """

    def __init__(self, title: str = "", save_dir: str = "results"):
        self.title = title  # 依然接收title参数，但默认不绘制
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot(
        self,
        solutions: List[Solution],
        file_name: str = "pareto_frontier", # 默认不带后缀
        xlabel: str = r"Transportation Risk $\times 10^5$ (people $\cdot$ t)",
        ylabel: str = r"Transportation Cost $\times 10^4$ (yuan)",
    ):
        """
        绘制二维帕累托前沿图。
        """
        if not solutions:
            logging.warning("Plotter: 传入的解列表为空，无法绘图。")
            return

        # --- 1. 数据提取与分类 ---
        feasible_sols = [
            s for s in solutions if s.is_feasible and s.f1_risk != float("inf")
        ]

        if not feasible_sols:
            logging.warning("Plotter: 没有可行解，跳过绘图。")
            return

        # 按 Rank 分组 (只保留 Rank 0, 1, 2, 3)
        ranks_data = {}
        target_ranks = [0, 1, 2, 3]

        for s in feasible_sols:
            r = s.rank
            if r in target_ranks:
                if r not in ranks_data:
                    ranks_data[r] = []
                ranks_data[r].append(s)

        # --- 2. 开始绘图 ---
        fig, ax = plt.subplots(figsize=(10, 8))  # 稍微方正一点的比例

        # 定义每一层的样式 (除了 Rank 0)
        # 颜色使用灰度渐变或者柔和的冷色调，形状交替使用
        # markers = ["s", "^", "D", "v", "<", ">", "p", "*"]  # 方形, 三角, 菱形...
        markers = ["s", "^", "D", "v"]  # 对应 Rank 1, 2, 3
        # 灰色系，Rank 越高颜色越浅
        base_gray = 0.4  # Rank 1 的灰度

        # A. 绘制被支配的层级 (Rank 1, Rank 2, ...)
        # 从高 Rank 画到低 Rank，这样 Rank 0 会在最上面
        for r in sorted(ranks_data.keys(), reverse=True):
            if r == 0:
                continue  # Rank 0 单独处理

            sols = ranks_data[r]
            # x_vals = [s.f1_risk for s in sols]
            # y_vals = [s.f2_cost for s in sols]
            x_vals = [s.f1_risk / 100000.0 for s in sols]
            y_vals = [s.f2_cost / 10000.0 for s in sols]

            # 动态计算样式
            marker = markers[(r - 1) % len(markers)]
            # 灰度计算: Rank 1 -> 0.4, Rank 5 -> 0.8 (更浅)
            gray_val = min(0.9, base_gray + (r - 1) * 0.1)
            color = str(gray_val)

            ax.scatter(
                x_vals,
                y_vals,
                c=color,
                marker=marker,
                s=40,
                label=f"Rank {r}",
                edgecolors="none",
                alpha=0.75,
                zorder=1,
            )

        # B. 处理 Rank 0 (Pareto Front)
        if 0 in ranks_data:
            rank0_sols = sorted(
                ranks_data[0], key=lambda s: s.f2_cost
            )   # 初始按 Cost 排序

            # x = Risk (f1), y = Cost (f2)
            # x_rank0 = [s.f1_risk for s in rank0_sols]
            # y_rank0 = [s.f2_cost for s in rank0_sols]
            x_rank0 = [s.f1_risk / 100000.0 for s in rank0_sols]
            y_rank0 = [s.f2_cost / 10000.0 for s in rank0_sols]

            # --- 绘制 Rank 0 的点 (保持你要的样式) ---
            # 白色填充，深蓝边框，圆形
            ax.scatter(
                x_rank0,
                y_rank0,
                facecolors="white",
                edgecolors="#2e86c1",  # 深蓝色边框
                linewidths=2,
                marker="o",
                s=90,
                label="Pareto Optimal",
                zorder=3,
            )

            # --- 连线与特殊点标记 ---
            if rank0_sols:
                # 1. 连线 (按 Risk x 从小到大排序)
                sorted_pairs = sorted(zip(x_rank0, y_rank0), key=lambda p: p[0])
                x_sorted = [p[0] for p in sorted_pairs]
                y_sorted = [p[1] for p in sorted_pairs]

                if len(rank0_sols) > 1:
                    ax.plot(
                        x_sorted,
                        y_sorted,
                        color="#5dade2",
                        linestyle="--",
                        linewidth=2.5,
                        zorder=2,
                    )

                # 2. 寻找特殊点
                sol_min_cost = min(zip(x_rank0, y_rank0), key=lambda p: p[1])
                sol_min_risk = min(zip(x_rank0, y_rank0), key=lambda p: p[0])

                # Knee Point (折衷解)
                best_knee = None
                min_x, max_x = min(x_rank0), max(x_rank0)
                min_y, max_y = min(y_rank0), max(y_rank0)
                min_dist = float("inf")

                x_range = max_x - min_x if max_x != min_x else 1.0
                y_range = max_y - min_y if max_y != min_y else 1.0

                for px, py in zip(x_rank0, y_rank0):
                    norm_x = (px - min_x) / x_range
                    norm_y = (py - min_y) / y_range
                    dist = norm_x**2 + norm_y**2
                    if dist < min_dist:
                        min_dist = dist
                        best_knee = (px, py)

                # 3. 绘制特殊光晕 (红色虚线圈)
                special_points = [sol_min_cost, sol_min_risk]
                if best_knee:
                    special_points.append(best_knee)
                special_points = list(set(special_points))

                for sx, sy in special_points:
                    ax.scatter(
                        sx,
                        sy,
                        s=220,
                        facecolors="none",
                        edgecolors="red",
                        linestyle=":",
                        linewidth=2,
                        zorder=4,
                    )

        # --- 3. 美化坐标轴 ---
        ax.set_xlabel(xlabel, fontweight="normal")
        ax.set_ylabel(ylabel, fontweight="normal")

        # 只保留 Y 轴上的横向虚线
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        # ax.grid(True, axis='x', linestyle='--', alpha=0.7) # 明确关闭 X 轴网格
        ax.set_ylabel(ylabel, fontweight="normal")

        # 应用格式化器
        # formatter = MathTextSciFormatter(useMathText=True)
        # formatter.set_powerlimits((-1, 1))  # 强制对较大或较小的数使用科学计数法
        # ax.xaxis.set_major_formatter(formatter)
        # ax.yaxis.set_major_formatter(formatter)

        # 坐标轴格式化: 不用科学计数法，保留1位小数
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        # 调整图例
        handles, labels = ax.get_legend_handles_labels()

        # 自定义排序: Rank 0 -> Rank 1 -> Rank 2 ...
        def get_rank_order(label):
            if "Pareto" in label:
                return 0
            try:
                return int(label.split()[-1])
            except:
                return 99

        sorted_handles_labels = sorted(
            zip(handles, labels), key=lambda t: get_rank_order(t[1])
        )

        final_handles = [h for h, l in sorted_handles_labels]
        final_labels = [l for h, l in sorted_handles_labels]

        ax.legend(
            final_handles,
            final_labels,
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="black",
        )

        # 不设置 title
        # ax.set_title(...)

        plt.tight_layout()

        # --- 4. 保存多格式 ---
        # 1. SVG (矢量图)
        full_path_svg = os.path.join(self.save_dir, f"{file_name}.svg")
        plt.savefig(full_path_svg, format="svg", dpi=300, bbox_inches="tight")
        logging.info(f"Pareto Front (SVG) 已保存至: {full_path_svg}")

        # # 2. PNG (高分位图，用于预览)
        # full_path_png = os.path.join(self.save_dir, f"{file_name}.png")
        # plt.savefig(full_path_png, format='png', dpi=300, bbox_inches='tight')
        # logging.info(f"Pareto Front (PNG) 已保存至: {full_path_png}")

        # plt.close(fig)


def plot_parallel_coordinates(rank_0_solutions: List[Solution], save_dir: str):
    """
    为 Rank 0 解绘制平行坐标图 (Parallel Coordinate Plot)。
    """
    pass
