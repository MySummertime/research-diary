# --- coding: utf-8 ---
# --- app/utils/visualizer.py ---
import os
import math
import networkx as nx
import contextily as ctx  # type: ignore
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from matplotlib.lines import Line2D
from app.core.network import TransportNetwork
from app.core.solution import Solution

HAS_CTX = True


class NetworkVisualizer:
    """
    [View Layer] 网络可视化器 (最终修正版)
    职责：负责将 Network 对象和 Solution 对象转换为 SVG 图像。
    特性：自动投影、动态曲线避让、严格的形状/线型区分、节点标签显示。
    """

    def __init__(self, network: TransportNetwork):
        self.network = network
        self.raw_pos = self._extract_raw_positions()

        # --- 统一视觉风格配置 ---
        self.styles = {
            "figure_size": (14, 12),
            "node": {
                "hub": {"color": "#E74C3C", "shape": "^", "size": 500, "alpha": 1.0},
                "non-hub": {
                    "color": "#3498DB",
                    "shape": "o",
                    "size": 250,
                    "alpha": 1.0,
                },
                "emergency_overlay": {
                    "edgecolor": "#F1C40F",
                    "linewidth": 2.5,
                    "linestyle": "--",
                    "facecolor": "none",
                },
            },
            "edge": {
                # Road 实线, Railway 点划线
                "road": {"color": "#95A5A6", "width": 2.0, "style": "-", "alpha": 0.7},
                "railway": {
                    "color": "#2C3E50",
                    "width": 2.5,
                    "style": "-.",
                    "alpha": 0.8,
                },
            },
            "font": {"size": 9, "color": "#2C3E50", "weight": "bold"},
        }

    def visualize_topology(
        self,
        save_dir: str,
        filename: str = "network_topology.svg",
        add_basemap: bool = True,
        title: str = "Network Topology",
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

        # 4. 图例与保存
        self._add_topology_legend(ax)
        self._save_plot(save_dir, filename, title)

    def visualize_routes(
        self,
        solution: Solution,
        task_colors: Dict[str, str],
        save_dir: str,
        filename: str,
        title: str,
    ):
        """
        [Routes] 绘制特定解的路线图 (带箭头、曲线、线型区分)。
        """
        use_projection = HAS_CTX and self._is_geo_coords()
        plot_pos = self._get_plot_positions(use_projection)

        # 底图用无向图淡化显示
        G_bg = self.network.graph.to_undirected()
        # 前景用有向图绘制特定路线
        G_raw = self.network.graph

        fig, ax = plt.subplots(figsize=self.styles["figure_size"])

        # 1. 绘制淡化背景 (简单曲线)
        nx.draw_networkx_edges(
            G_bg,
            plot_pos,
            ax=ax,
            edge_color="#ECF0F1",
            width=1.0,
            alpha=0.4,
            arrows=True,
            arrowsize="-",
            connectionstyle="arc3,rad=0.05",
        )
        nx.draw_networkx_nodes(
            G_bg, plot_pos, ax=ax, node_size=100, node_color="#BDC3C7", alpha=0.3
        )

        # 2. 绘制前景 Task 路线 (核心逻辑)
        legend_handles = []
        drawn_tasks = set()

        # 收集所有需要绘制的任务边
        all_task_edges = []
        for path in solution.path_selections.values():
            if not path.task:
                continue
            for arc in path.arcs:
                all_task_edges.append(
                    (arc.start.node_id, arc.end.node_id, path.task.task_id, arc.mode)
                )

        # 计算动态曲率样式
        edge_styles = self._calculate_edge_styles(
            [(u, v) for u, v, _, _ in all_task_edges]
        )

        # 逐条绘制
        for i, (u, v, task_id, mode) in enumerate(all_task_edges):
            color = task_colors.get(task_id, "#333333")
            linestyle = (
                self.styles["edge"]["road"]["style"]
                if mode == "road"
                else self.styles["edge"]["railway"]["style"]
            )

            nx.draw_networkx_edges(
                G_raw,
                plot_pos,
                ax=ax,
                edgelist=[(u, v)],
                edge_color=color,
                width=2.5,
                alpha=0.9,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=18,  # 箭头
                style=linestyle,  # 线型
                connectionstyle=edge_styles[i],  # 曲率
            )

            if task_id not in drawn_tasks:
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=2.5, label=f"Task {task_id}")
                )
                drawn_tasks.add(task_id)

        # 3. 重新绘制关键节点
        self._draw_nodes(G_bg, plot_pos, ax, alpha=0.9, scale=0.8)

        # 绘制节点标签
        self._draw_labels(G_bg, plot_pos, ax)

        if use_projection:
            self._add_basemap(ax)

        # 添加图例 (路线图专有)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                lw=2,
                linestyle=self.styles["edge"]["road"]["style"],
                label="Road Path",
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
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
        )
        self._save_plot(save_dir, filename, title)

    # --- 核心逻辑：动态曲率计算器 ---

    def _calculate_edge_styles(self, edgelist: List[Tuple[str, str]]) -> List[str]:
        """
        [核心算法] 计算每条边的 connectionstyle，确保它们弯曲且不重叠。
        """
        styles = []
        pair_tracker = {}  # 记录 (u, v) 无序对出现的次数

        for u, v in edgelist:
            pair_key = tuple(sorted((u, v)))
            count = pair_tracker.get(pair_key, 0)

            # 基础半径 0.1，每多一对边增加 0.1
            base_rad = 0.1 + (count // 2) * 0.1
            # 交替正负
            rad = base_rad if count % 2 == 0 else -base_rad

            styles.append(f"arc3,rad={rad}")
            pair_tracker[pair_key] = count + 1

        return styles

    # --- 绘图辅助逻辑 ---

    def _draw_topology_edges_no_overlap(self, G, pos, ax):
        road_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("mode") == "road"
        ]
        rail_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("mode") == "railway"
        ]

        all_edges_ordered = road_edges + rail_edges
        all_styles = self._calculate_edge_styles(all_edges_ordered)

        road_styles = all_styles[: len(road_edges)]
        rail_styles = all_styles[len(road_edges) :]

        style_road = self.styles["edge"]["road"]
        for i, edge in enumerate(road_edges):
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=[edge],
                edge_color=style_road["color"],
                width=style_road["width"],
                style=style_road["style"],
                alpha=style_road["alpha"],
                arrows=True,
                arrowstyle="-",
                connectionstyle=road_styles[i],
            )

        style_rail = self.styles["edge"]["railway"]
        for i, edge in enumerate(rail_edges):
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=[edge],
                edge_color=style_rail["color"],
                width=style_rail["width"],
                style=style_rail["style"],
                alpha=style_rail["alpha"],
                arrows=True,
                arrowstyle="-",
                connectionstyle=rail_styles[i],
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
                node_size=style_hub["size"] * scale * 1.4,
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
        offset = 2000 if any(abs(y) > 1000 for x, y in pos.values()) else 0.0002
        label_pos = {k: (v[0], v[1] - offset) for k, v in pos.items()}
        nx.draw_networkx_labels(
            G,
            label_pos,
            ax=ax,
            font_size=self.styles["font"]["size"],
            font_color=self.styles["font"]["color"],
            font_weight=self.styles["font"]["weight"],
        )

    def _add_basemap(self, ax):
        if HAS_CTX:
            try:
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            except Exception:
                pass

    def _add_topology_legend(self, ax):
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                label="Hub",
                markerfacecolor=self.styles["node"]["hub"]["color"],
                markersize=12,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Non-Hub",
                markerfacecolor=self.styles["node"]["non-hub"]["color"],
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                label="Emergency Center",
                markerfacecolor="none",
                markeredgecolor=self.styles["node"]["emergency_overlay"]["edgecolor"],
                markeredgewidth=2,
                markersize=14,
                linestyle="--",
            ),
            Line2D(
                [0],
                [0],
                color=self.styles["edge"]["road"]["color"],
                lw=2,
                linestyle=self.styles["edge"]["road"]["style"],
                label="Road",
            ),
            Line2D(
                [0],
                [0],
                color=self.styles["edge"]["railway"]["color"],
                lw=2,
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

    def _save_plot(self, save_dir, filename, title):
        ax = plt.gca()
        ax.set_title(title, fontsize=16, pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), format="svg", bbox_inches="tight")
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
