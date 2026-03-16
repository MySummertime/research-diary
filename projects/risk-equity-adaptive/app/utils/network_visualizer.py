# --- coding: utf-8 ---
# --- app/utils/network_visualizer.py ---
import math
import os
from typing import Dict, List, Tuple

import contextily as ctx  # type: ignore
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from app.core.network import TransportNetwork
from app.core.solution import Solution
from app.utils.visual_style import ColorPalette, get_color_by_key

HAS_CTX = True


class NetworkVisualizer:
    """
    [View Layer] 网络可视化器
    职责：负责将 Network 对象和 Solution 对象转换为 SVG 图像。
    特性：自动投影、动态曲线避让、严格的形状/线型区分、节点标签显示。
    """

    def __init__(self, network: TransportNetwork):
        # ==================== 全局设置 Times New Roman 为默认字体 ====================
        plt.rcParams["font.family"] = "Times New Roman"  # 主要设置：所有文字
        plt.rcParams["font.sans-serif"] = ["Times New Roman"]  # 防止 fallback
        plt.rcParams["font.serif"] = ["Times New Roman"]  # 明确指定 serif 族
        plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示为方块

        self.network = network
        self.raw_pos = self._extract_raw_positions()

        # Color set
        self.default_colors: List[Dict] = ColorPalette.DEFAULT_COLOR
        self.network_colors: List[Dict] = ColorPalette.NETWORK_TOPOLOGY
        self.task_colors: List[str] = ColorPalette.TASK_LOOP

        self.styles = {
            "figure_size": (14, 12),
            "node": {
                "hub": {
                    "color": get_color_by_key(self.network_colors, "HUB"),
                    "shape": "^",
                    "size": 1200,
                    "alpha": 1.0,
                },
                "non-hub": {
                    "color": get_color_by_key(self.network_colors, "NON_HUB"),
                    "shape": "o",
                    "size": 1000,
                    "alpha": 1.0,
                },
                "emergency_overlay": {
                    "edgecolor": get_color_by_key(self.network_colors, "EMERGENCY"),
                    "linewidth": 4.0,
                    "linestyle": "--",
                    "facecolor": "none",
                },
            },
            "edge": {
                # Road 实线, Railway 点划线
                "road": {
                    "color": get_color_by_key(self.network_colors, "ROAD"),
                    "width": 3.0,
                    "style": "-",
                    "alpha": 0.7,
                },
                "railway": {
                    "color": get_color_by_key(self.network_colors, "RAILWAY"),
                    "width": 4.0,
                    "style": "--",
                    "alpha": 0.85,
                },
            },
            "font": {
                "size": 20,
                "color": get_color_by_key(self.default_colors, "BLACK"),
                "weight": "bold",
            },
        }

    def visualize_topology(
        self,
        save_dir: str,
        filename: str = "network_topology",
        add_basemap: bool = True,
    ):
        """
        [Topology] 绘制基础拓扑结构。
        """
        use_projection = HAS_CTX and add_basemap and self._is_geo_coords()
        plot_pos = self._get_plot_positions(use_projection)

        # 使用有向图来区分 A->B 和 B->A
        G = self.network.graph

        fig, ax = plt.subplots(figsize=self.styles["figure_size"])

        # 1. 绘制边 (动态计算曲率以避免重叠)
        self._draw_topology_edges_no_overlap(G, plot_pos, ax)

        # 2. 绘制节点 和 标签
        self._draw_nodes(G, plot_pos, ax)
        self._draw_labels(G, plot_pos, ax)

        # 3. 添加底图
        if use_projection:
            self._add_basemap(ax)
            pass

        # 4. 图例与保存
        self._add_topology_legend(ax)
        self._save_plot(save_dir, filename)

    def visualize_routes(
        self,
        solutions_map: Dict[str, Solution],
        save_dir: str,
        prefix: str = "route",
    ):
        """
        [Visualizer] 批量生成特殊解的路线地图 (SVG)
        使用 visual_style.py 中的 TASK_CYCLE 为所有任务生成统一的配色。
        Args:
            solutions_map: 字典 {"Opinion A": sol_a, "Opinion B": sol_b, ...}
            save_dir: 保存目录
            prefix: 文件名前缀（默认 "route"）
        """
        import logging

        logging.info("Generating multiple route maps for special solutions...")

        # 使用 visual_style.py 中的统一颜色（支持无限任务循环）
        task_colors: List = self.task_colors

        # 步骤3：遍历每个特殊解，生成独立路线图
        for label, sol in solutions_map.items():
            if not sol:  # 跳过空解（防御性编程）
                logging.warning(f"Skipping empty solution for {label}")
                continue

            # 生成安全文件名：替换空格为空下划线，避免文件系统问题
            safe_label = label.replace(" ", "_").replace("/", "_")
            filename = f"{prefix}_{safe_label}"

            # 调用已有方法绘制单张路线图
            self._draw_routes(
                solution=sol,
                task_colors=task_colors,  # 传入统一配色
                save_dir=save_dir,
                filename=filename,
            )

            logging.info(f"Route map saved: {filename}")

        logging.info("Multiple route maps generation completed. 🌟🚀")

    def _draw_routes(
        self,
        solution: Solution,
        task_colors: List[str],
        save_dir: str,
        filename: str,
    ):
        """
        [核心绘制函数] 绘制特定解 (Solution) 的路线图
        特点：
        - 使用无向图视图去重双向边（避免 road/rail 重叠）
        - 动态曲率避让 + 模式区分（road 实线在上，railway 虚线在下）
        - 任务线按 task_id 着色 + 图例
        - 节点中心显示 ID，名称在下方带框标签
        - 支持地理投影底图（黑白风格）

        Args:
            solution: 当前 Opinion 的完整解对象
            task_colors: 任务颜色循环列表（从 visual_style.py 传入）
            save_dir: 保存目录
            filename: 文件名（不含后缀）

        Returns:
            None（直接保存图像文件）
        """
        # ── Step 1: 判断是否需要地理投影 + 获取绘图坐标 ──
        use_projection = HAS_CTX and self._is_geo_coords()
        plot_pos = self._get_plot_positions(use_projection)

        # ── Step 2: 准备图对象 ──
        # 背景层：无向图（淡化显示所有节点和边，作为空间参考）
        G_bg = self.network.graph.to_undirected(as_view=True)

        # 前景层：有向图（用于绘制带箭头的任务路径）
        G_raw = self.network.graph

        # 创建画布
        fig, ax = plt.subplots(figsize=self.styles["figure_size"])

        # ── Step 3: 绘制淡化背景节点（作为空间参考，不抢任务线风头） ──
        nx.draw_networkx_nodes(
            G_bg,
            plot_pos,
            ax=ax,
            node_size=300,
            node_color=get_color_by_key(self.default_colors, "GRAY"),
            alpha=0.1,  # 极淡，几乎看不见但保留位置感
        )

        # ── Step 4: 构建 task_id → color 的映射（确保相同 task_id 颜色全局一致） ──
        appeared_task_ids = set()
        for path in solution.path_selections.values():
            if path.task and path.task.task_id is not None:
                appeared_task_ids.add(path.task.task_id)

        sorted_task_ids = sorted(appeared_task_ids)  # 按 ID 排序，保持稳定性

        task_id_to_color = {}
        for idx, tid in enumerate(sorted_task_ids):
            color = task_colors[idx % len(task_colors)]  # 循环使用颜色列表
            task_id_to_color[tid] = color

        # ── Step 5: 收集所有任务边（用于曲率计算和绘制） ──
        all_task_edges = []
        for path in solution.path_selections.values():
            if not path.task:
                continue
            for arc in path.arcs:
                all_task_edges.append(
                    (arc.start.node_id, arc.end.node_id, path.task.task_id, arc.mode)
                )

        # ── Step 6: 计算动态曲率样式（传入边列表 + 模式列表） ──
        edgelist_for_style = [(u, v) for u, v, _, _ in all_task_edges]
        modes_for_style = [m for _, _, _, m in all_task_edges]
        edge_styles = self._calculate_edge_styles(
            edgelist_for_style, modes=modes_for_style
        )

        # ── Step 7: 绘制任务路线（核心绘制部分） ──
        legend_handles = []
        drawn_tasks = set()

        for i, (u, v, task_id, mode) in enumerate(all_task_edges):
            color = task_id_to_color.get(task_id, task_colors[0])
            linestyle = (
                self.styles["edge"]["road"]["style"]
                if mode == "road"
                else self.styles["edge"]["railway"]["style"]
            )

            # 绘制单条任务边（使用有向图，保留箭头）
            nx.draw_networkx_edges(
                G_raw,
                plot_pos,
                ax=ax,
                edgelist=[(u, v)],  # 只画当前这条有向边
                edge_color=color,
                width=3.5,
                alpha=0.9,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=18,
                style=linestyle,
                connectionstyle=edge_styles[i],
            )

            # 添加图例（每个 task 只加一次）
            if task_id not in drawn_tasks:
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=2.5, label=f"Task {task_id}")
                )
                drawn_tasks.add(task_id)

        # ── Step 8: 重新绘制关键节点（前景层，确保节点在最上层） ──
        self._draw_nodes(G_bg, plot_pos, ax, alpha=0.9, scale=0.8)

        # ── Step 9: 绘制节点标签（名称 + ID） ──
        self._draw_labels(G_bg, plot_pos, ax)

        # ── Step 10: 添加底图（如果启用） ──
        if use_projection:
            self._add_basemap(ax)

        # ── Step 11: 添加图例（任务线 + road/railway 图例） ──
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=get_color_by_key(self.default_colors, "GRAY"),
                lw=2,
                linestyle=self.styles["edge"]["road"]["style"],
                label="Road Path",
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=get_color_by_key(self.default_colors, "GRAY"),
                lw=2,
                linestyle=self.styles["edge"]["railway"]["style"],
                label="Rail Path",
            )
        )

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fancybox=True,
            shadow=True,
            title="Legend",
            fontsize=12,
            title_fontsize=14,
        )

        # ── Step 12: 保存图像 ──
        self._save_plot(save_dir, filename)

    # --- 核心逻辑：动态曲率计算器 ---

    def _calculate_edge_styles(
        self, edgelist: List[Tuple[str, str]], modes: List[str] = None
    ) -> List[str]:
        """
        [核心算法] 计算每条边的 connectionstyle，确保不同模式/方向不重叠。
        """
        styles = []
        # 🌟 有向边 key + 模式组合，避免 (A,B,road) 和 (B,A,rail) 重合
        edge_tracker = {}  # key: (u,v,mode) → 有向 + 模式唯一标识

        for i, (u, v) in enumerate(edgelist):
            current_mode = modes[i] if (modes and i < len(modes)) else "road"
            # 🌟 有向边 + 模式作为唯一 key
            edge_key = (u, v, current_mode)
            count = edge_tracker.get(edge_key, 0)

            mode_offset = 0.12 if current_mode == "road" else 0.28  # rail 偏移更大
            base_rad = mode_offset + (count // 2) * 0.15  # 每多一对边增 0.15
            rad = base_rad if count % 2 == 0 else -base_rad

            styles.append(f"arc3,rad={rad}")
            edge_tracker[edge_key] = count + 1

        return styles

    # --- 绘图辅助逻辑 ---
    def _draw_topology_edges_no_overlap(self, G, pos, ax):
        """绘制拓扑边：每对节点只画一条线（去重双向边）"""

        # 🌟 转为无向图视图，只保留一条边
        G_undirected = G.to_undirected(as_view=True)

        # 获取去重后的边列表
        edges_with_data = list(G_undirected.edges(data=True))
        edgelist = [(u, v) for u, v, d in edges_with_data]

        # 获取对应模式（因为是无向图，取任意一条的 mode 即可）
        modes = [d.get("mode", "road") for u, v, d in edges_with_data]

        # 计算曲率样式（传入去重后的 edgelist）
        all_styles = self._calculate_edge_styles(edgelist, modes=modes)

        # 分开绘制 railway 和 road（先铁路后公路）
        rail_edges = [
            (u, v) for i, (u, v) in enumerate(edgelist) if modes[i] == "railway"
        ]
        road_edges = [(u, v) for i, (u, v) in enumerate(edgelist) if modes[i] == "road"]

        rail_styles = [
            all_styles[i] for i, (u, v) in enumerate(edgelist) if modes[i] == "railway"
        ]
        road_styles = [
            all_styles[i] for i, (u, v) in enumerate(edgelist) if modes[i] == "road"
        ]

        # 先画铁路（底层）
        style_rail = self.styles["edge"]["railway"]
        for i, (u, v) in enumerate(rail_edges):
            nx.draw_networkx_edges(
                G_undirected,  # 使用无向图
                pos,
                ax=ax,
                edgelist=[(u, v)],
                edge_color=style_rail["color"],
                width=style_rail["width"],
                style=style_rail["style"],
                alpha=style_rail["alpha"],
                arrows=True,
                arrowstyle="-",
                connectionstyle=rail_styles[i],
            )

        # 再画公路（上层）
        style_road = self.styles["edge"]["road"]
        for i, (u, v) in enumerate(road_edges):
            nx.draw_networkx_edges(
                G_undirected,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                edge_color=style_road["color"],
                width=style_road["width"],
                style=style_road["style"],
                alpha=style_road["alpha"],
                arrows=True,
                arrowstyle="-",
                connectionstyle=road_styles[i],
            )

    def _draw_nodes(self, G, pos, ax, alpha=1.0, scale=1.0):
        hubs = [n for n, d in G.nodes(data=True) if d.get("type") == "hub"]
        non_hubs = [n for n, d in G.nodes(data=True) if d.get("type") != "hub"]
        emergency_nodes = [n for n, d in G.nodes(data=True) if d.get("is_emergency")]

        # 1. 绘制基础层
        style_hub = self.styles["node"]["hub"]
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=hubs,
            node_color=style_hub["color"],
            node_shape=style_hub["shape"],
            node_size=style_hub["size"] * scale,
            alpha=alpha,
        )

        style_non = self.styles["node"]["non-hub"]
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=non_hubs,
            node_color=style_non["color"],
            node_shape=style_non["shape"],
            node_size=style_non["size"] * scale,
            alpha=alpha,
        )

        # 2. 绘制应急叠加层
        em_hubs = [n for n in emergency_nodes if n in hubs]
        em_non_hubs = [n for n in emergency_nodes if n in non_hubs]
        style_ov = self.styles["node"]["emergency_overlay"]

        if em_hubs:
            nodes_h = nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=em_hubs,
                node_color="none",
                edgecolors=style_ov["edgecolor"],
                linewidths=style_ov["linewidth"],
                node_shape=style_hub["shape"],
                node_size=style_hub["size"] * scale * 1.5,
            )
            nodes_h.set_linestyle(style_ov["linestyle"])

        if em_non_hubs:
            nodes_n = nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=em_non_hubs,
                node_color="none",
                edgecolors=style_ov["edgecolor"],
                linewidths=style_ov["linewidth"],
                node_shape=style_non["shape"],
                node_size=style_non["size"] * scale * 1.4,
            )
            nodes_n.set_linestyle(style_ov["linestyle"])

    def _draw_labels(self, G, pos, ax):
        """
        在节点符号外绘制带文本框的节点名称（name），同时在节点中心保留 ID 编号。
        偏移量动态计算，确保标签不与节点形状重叠。
        """
        # ── 1. 计算合理的垂直向下偏移（根据图的 y 范围动态调整）
        y_coords = [y for x, y in pos.values()]
        if not y_coords:
            offset_val = 0.05
        else:
            y_range = max(y_coords) - min(y_coords)
            offset_val = y_range * 0.04 if y_range > 0 else 0.05  # 8% 的 y 范围偏移
            offset_val = max(offset_val, 0.04)  # 最小偏移 0.04

        # 标签位置：节点坐标向下偏移
        label_pos = {n: (pos[n][0], pos[n][1] - offset_val) for n in G.nodes()}

        # ── 2. 获取显示的标签（优先用 english name，没有就用 node_id）
        labels = {n: G.nodes[n].get("name_en", n) for n in G.nodes()}

        # ── 3. 文本框样式（学术风格：白色半透 + 细边框）
        bbox_style = dict(
            boxstyle="round,pad=0.5",
            fc="white",
            ec="lightgray",
            lw=0.1,
            alpha=0.1,
            mutation_scale=20,
        )

        # ── 4. 绘制节点名称标签（带背景框，下方）
        nx.draw_networkx_labels(
            G,
            label_pos,
            labels=labels,
            ax=ax,
            font_size=self.styles["font"]["size"] - 1,
            font_color=self.styles["font"]["color"],
            font_weight="normal",
            bbox=bbox_style,
            verticalalignment="top",  # 文本框顶部对齐偏移点，向下生长
            horizontalalignment="center",  # 水平居中
        )

        # ── 5. 在节点中心绘制 ID 编号（不带框、加粗）
        id_labels = {n: n for n in G.nodes()}  # 用 node_id 作为中心编号
        nx.draw_networkx_labels(
            G,
            pos,
            labels=id_labels,
            ax=ax,
            font_size=self.styles["font"]["size"],
            font_color="black",
            font_weight="bold",
        )

        # ── 6. 动态调整坐标轴范围，防止标签超出画面
        # 提取所有标签位置的 Y 坐标
        label_y_coords = [p[1] for p in label_pos.values()]
        if label_y_coords:
            current_ymin, current_ymax = ax.get_ylim()
            # 计算标签到达的最小值
            min_label_y = min(label_y_coords)

            # 如果标签的最低点超出了当前 Y 轴最小值，则向下扩展
            # 额外多减去一个 offset_val 的 50% 作为边距缓冲
            if min_label_y < current_ymin:
                ax.set_ylim(min_label_y - offset_val * 0.5, current_ymax)

    def _add_basemap(self, ax):
        if HAS_CTX:
            try:
                # 浅灰白背景，不容易产生颜色混淆
                ctx.add_basemap(
                    ax, source=ctx.providers.CartoDB.PositronNoLabels, alpha=0.9
                )
            except Exception:
                pass

    def _add_topology_legend(self, ax):
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="^",
                color=get_color_by_key(self.default_colors, "WHITE"),
                label="Hub",
                markerfacecolor=self.styles["node"]["hub"]["color"],
                markersize=14,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=get_color_by_key(self.default_colors, "WHITE"),
                label="Non-Hub",
                markerfacecolor=self.styles["node"]["non-hub"]["color"],
                markersize=14,
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color=get_color_by_key(self.default_colors, "WHITE"),
                label="Emergency Center",
                markerfacecolor="none",
                markeredgecolor=self.styles["node"]["emergency_overlay"]["edgecolor"],
                markeredgewidth=2,
                markersize=16,
                linestyle=":",
            ),
            Line2D(
                [0],
                [0],
                color=self.styles["edge"]["road"]["color"],
                lw=2.0,
                linestyle=self.styles["edge"]["road"]["style"],
                label="Road",
            ),
            Line2D(
                [0],
                [0],
                color=self.styles["edge"]["railway"]["color"],
                lw=3.0,
                linestyle=self.styles["edge"]["railway"]["style"],
                label="Railway",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    def _save_plot(self, save_dir, filename):
        plt.axis("off")
        plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, filename), format="svg", bbox_inches="tight")
        # 保存为 TIFF（最推荐）
        plt.savefig(
            os.path.join(save_dir, f"{filename}.tif"),
            format="tif",
            dpi=300,  # 混合图用 600，纯线图可以冲 1000–1200
            bbox_inches="tight",  # 自动裁剪白边，超级重要
            pad_inches=0.1,  # 给边缘留 0.1 英寸的白边
            # transparent=True,  # 可选：如果需要去背景
        )

        # 备选 PNG（文件更小，透明背景可选）
        plt.savefig(
            os.path.join(save_dir, f"{filename}.png"),
            format="png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,  # 给边缘留 0.1 英寸的白边
            # transparent=True,  # 可选：如果需要去背景
        )
        plt.close()

    def _extract_raw_positions(self):
        G = self.network.graph
        return {n: (data["x"], data["y"]) for n, data in G.nodes(data=True)}

    def _is_geo_coords(self) -> bool:
        for x, y in self.raw_pos.values():
            if not (-180 <= x <= 180 and -90 <= y <= 90):
                return False
            if x == 0 and y == 0:
                continue
        return True

    def _get_plot_positions(self, use_projection: bool) -> Dict[str, tuple]:
        return (
            self._project_to_web_mercator(self.raw_pos)
            if use_projection
            else self.raw_pos
        )

    def _project_to_web_mercator(
        self, lonlat_pos: Dict[str, tuple]
    ) -> Dict[str, tuple]:
        mercator_pos = {}
        r_major = 6378137.000
        for nid, (lon, lat) in lonlat_pos.items():
            if lon == 0 and lat == 0:
                mercator_pos[nid] = (0, 0)
                continue
            x = r_major * math.radians(lon)
            lat = max(min(lat, 89.5), -89.5)
            temp = math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)
            y = r_major * math.log(temp)
            mercator_pos[nid] = (x, y)
        return mercator_pos
