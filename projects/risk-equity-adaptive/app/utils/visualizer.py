# --- coding: utf-8 ---
# --- app/utils/visualizer.py ---
import os
import logging
import math
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
from typing import List, Optional, Dict, Any
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from app.core.network import TransportNetwork

HAS_CTX = True

# --- 网络可视化功能 ---


def visualize_network(
    network: TransportNetwork,
    layout_func_name: str = "spring",  # 布局函数名词
    save_path: Optional[str] = None,
    add_basemap: bool = True,  # 是否添加真实地图底图
    export_svg: bool = True,  # 是否同时导出svg图
):
    """
    可视化网络拓扑，使用 networkx 和 matplotlib，支持有向图。
    1. 优先读取 Node.x 和 Node.y（经纬度）。
    2. 如果未提供经纬度 (即全为 0.0)，则回退到 layout_func_name 指定的算法布局。
    3. 自动添加 OpenStreetMap 底图 (如果安装了 contextily)。
    4. 导出 SVG 矢量图。
    """
    # 1. 获取样式配置和准备好的绘图数据
    style = _get_style_config()
    graph_data = _prepare_graph_data(network, style)
    G = graph_data["graph"]

    # 2. 处理坐标 (Coordinates)
    # 输入的是 x=Lon, y=Lat (WGS84)
    raw_pos = {}
    has_valid_coords = False
    for node in network.nodes:
        # 检查是否有非零坐标
        if node.x != 0.0 or node.y != 0.0:
            has_valid_coords = True
        raw_pos[node.id] = (node.x, node.y)  # (Lon, Lat)

    # 3. 投影转换 (Projection)
    # 如果要加底图，必须把 (Lon, Lat) 转成 Web Mercator (EPSG:3857)
    final_pos = {}
    use_geo_layout = has_valid_coords

    if use_geo_layout:
        print("Visualizer: 使用真实地理坐标 (Geo-Spatial Layout)。")
        # 仅当 contextily 可用且需要底图时，才进行 Mercator 投影
        if HAS_CTX and add_basemap:
            # 如果有底图库，必须投影到 EPSG:3857 墨卡托投影转换 (Web Mercator)
            final_pos = _project_to_web_mercator(raw_pos)
        else:
            # 如果没有底图库，直接画经纬度也行 (会稍微压扁，但拓扑是对的)
            final_pos = raw_pos
    else:
        print(f"Visualizer: 无有效坐标，回退到自动布局 ({layout_func_name})。")
        if layout_func_name == "spring":
            final_pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
        elif layout_func_name == "circular":
            final_pos = nx.circular_layout(G)
        else:
            # 默认 Kamada-Kawai，适合一般网络拓扑
            final_pos = nx.kamada_kawai_layout(G)

    # 4. 开始绘图
    fig, ax = plt.subplots(figsize=style["figure_size"])

    # 绘制 Road 边（虚线）
    nx.draw_networkx_edges(
        G,
        final_pos,
        edgelist=graph_data["road_edges"],
        style=style["road_style"],
        alpha=style["road_alpha"],
        edge_color=style["road_color"],
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
    )
    # 绘制 Rail 边（实线）
    nx.draw_networkx_edges(
        G,
        final_pos,
        edgelist=graph_data["rail_edges"],
        style=style["rail_style"],
        alpha=style["rail_alpha"],
        edge_color=style["rail_color"],
        width=style["rail_width"],
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
    )

    # 绘制节点
    nx.draw_networkx_nodes(
        G,
        final_pos,
        node_color=graph_data["node_colors"],
        node_size=style["node_size"],
        edgecolors=graph_data["node_border_colors"],
        linewidths=style["emergency_border_width"],
    )

    # 绘制标签 (Labels)
    # 为了防止标签和点重叠，稍微偏移一点 y
    label_pos = {k: (v[0], v[1]) for k, v in final_pos.items()}
    nx.draw_networkx_labels(
        G,
        label_pos,
        font_size=style["font_size"],
        font_color=style["font_color"],
        font_weight="bold",
        ax=ax,
    )

    # 5. 添加底图 (Basemap)
    if use_geo_layout and HAS_CTX and add_basemap:
        try:
            # add_basemap 会自动根据 ax 的 extent (数据范围) 下载对应的瓦片
            # source 可以选 ctx.providers.OpenStreetMap.Mapnik, Stamen.TonerLite 等
            # crs='EPSG:3857' 是默认值，这要求前面的 final_pos 必须是墨卡托坐标(投影)
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            print("Visualizer: 已添加地理底图 (CartoDB Positron)。")
        except Exception as e:
            print(f"Visualizer: 添加底图失败: {e}")

    # 6. 创建并显示图例和标题
    legend_elements = _create_legend(style)
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12, framealpha=0.9)

    title_suffix = ""
    if use_geo_layout:
        title_suffix = "(Geo-Spatial)" + (
            " + Basemap" if HAS_CTX and add_basemap else ""
        )
    else:
        title_suffix = f"(Auto: {layout_func_name})"
    ax.set_title(f"Transport Network Topology {title_suffix}", fontsize=18)

    # 如果有底图，通常要把坐标轴关掉，或者显示经纬度（比较麻烦）
    # 简单起见，有底图就关掉轴
    if HAS_CTX and add_basemap and use_geo_layout:
        ax.axis("off")
    else:
        ax.axis("on")
        ax.grid(True, linestyle=":", alpha=0.3)
        if use_geo_layout:
            ax.set_xlabel("Longitude / X")
            ax.set_ylabel("Latitude / Y")

    # 7. 保存 (PNG & SVG)
    if save_path:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 保存 PNG
        # plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # logging.info(f"Visualizer: PNG 保存至: {save_path}")

        # 保存 SVG (矢量图，用于投稿)
        if export_svg:
            svg_path = os.path.splitext(save_path)[0] + ".svg"
            plt.savefig(svg_path, format="svg", bbox_inches="tight")
            logging.info(f"Visualizer: SVG 保存至: {svg_path}")

    # plt.close(fig)


def _project_to_web_mercator(lonlat_pos: Dict[str, tuple]) -> Dict[str, tuple]:
    """
    [辅助] 将 (Lon, Lat) 转换为 Web Mercator (EPSG:3857) 坐标。
    用于配合 contextily 底图。
    公式：
    x = lon * 20037508.34 / 180
    y = log(tan((90 + lat) * PI / 360)) / (PI / 180) * 20037508.34 / 180
    """
    mercator_pos = {}
    r_major = 6378137.000

    for node_id, (lon, lat) in lonlat_pos.items():
        x = r_major * math.radians(lon)
        # 限制纬度范围，防止 tan(90) 爆炸
        lat = max(min(lat, 89.5), -89.5)
        temp = math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)
        y = r_major * math.log(temp)
        mercator_pos[node_id] = (x, y)

    return mercator_pos


def _prepare_graph_data(
    network: TransportNetwork, style_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    [辅助方法] 从网络数据中准备 NetworkX 绘图所需的数据（支持有向图）
    """
    G: nx.DiGraph = nx.DiGraph()

    # 确保添加所有节点，即使它们没有连接边
    node_ids = [node.id for node in network.nodes]
    G.add_nodes_from(node_ids)

    node_colors = []
    node_border_colors = []

    for node in network.nodes:
        # 填充颜色
        node_colors.append(style_config["color_map"].get(node.type, "gray"))
        # 边框颜色（应急中心高亮）
        if node.is_emergency_center:
            node_border_colors.append(style_config["emergency_border_color"])
        # 非应急中心，边框同填充色
        else:
            node_border_colors.append(style_config["color_map"].get(node.type, "gray"))

    road_edges_list = []
    rail_edges_list = []

    for arc in network.arcs:
        edge_tuple = (arc.start.id, arc.end.id)
        G.add_edge(*edge_tuple)

        if arc.mode == "road":
            road_edges_list.append(edge_tuple)
        elif arc.mode == "railway":
            rail_edges_list.append(edge_tuple)

    return {
        "graph": G,
        "node_colors": node_colors,
        "node_border_colors": node_border_colors,
        "road_edges": road_edges_list,
        "rail_edges": rail_edges_list,
    }


def _get_style_config() -> Dict[str, Any]:
    """
    [辅助方法] 返回一个包含所有绘图样式的配置字典。
    """
    return {
        "figure_size": (16, 12),
        "node_size": 300,
        "font_size": 8,
        "font_color": "black",
        "color_map": {"hub": "#ff4757", "non-hub": "#54a0ff"},
        "emergency_border_color": "#ffd700",
        "emergency_border_width": 2.5,
        "road_style": "dashed",
        "road_color": "#555555",
        "road_width": 1.5,
        "road_alpha": 0.7,
        "rail_style": "solid",
        "rail_color": "#000000",
        "rail_width": 1.9,
        "rail_alpha": 0.9,
    }


def _create_legend(style_config: Dict[str, Any]) -> List:
    """
    [辅助方法] 这个模块只创建一个清晰的图例。
    """
    return [
        Patch(
            facecolor=style_config["color_map"]["hub"], edgecolor="none", label="Hub"
        ),
        Patch(
            facecolor=style_config["color_map"]["non-hub"],
            edgecolor="none",
            label="Non-Hub",
        ),
        Patch(
            facecolor="white",
            edgecolor=style_config["emergency_border_color"],
            linewidth=2,
            label="Emergency Center",
        ),
        Line2D(
            [0],
            [0],
            color=style_config["rail_color"],
            lw=style_config["rail_width"],
            label="Railway",
        ),
        Line2D(
            [0],
            [0],
            color=style_config["road_color"],
            linestyle=style_config["road_style"],
            lw=style_config["road_width"],
            label="Road",
        ),
    ]
