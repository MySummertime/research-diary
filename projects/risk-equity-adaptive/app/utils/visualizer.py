# --- coding: utf-8 ---
# --- app/utils/visualizer.py ---
import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from app.core.network import TransportNetwork

# --- 网络可视化功能 ---

def visualize_network(
    network: TransportNetwork,
    layout_func_name: str = 'kamada-kawai',   #布局函数名词
    save_path: Optional[str] = None
):
    """
    使用 networkx 和 matplotlib 可视化网络拓扑（支持有向图）。
    """
    # 步骤 1: 获取样式配置和准备好的绘图数据
    style = _get_style_config()
    
    # 把 network 和 style 传递给辅助函数
    graph_data = _prepare_graph_data(network, style) 
    G = graph_data["graph"]

    # 步骤 2: 计算节点布局
    if layout_func_name == 'spring':
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    elif layout_func_name == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # 步骤 3: 开始绘图
    plt.figure(figsize=style['figure_size'])

    connection_style = 'arc3, rad=0.1'
    
    # 绘制 Road 边
    nx.draw_networkx_edges(G, pos, 
                           edgelist=graph_data['road_edges'], 
                           style=style['road_style'], 
                           alpha=style['road_alpha'], 
                           edge_color=style['road_color'],
                           arrows=True,
                           arrowstyle='-|>',
                           arrowsize=15,
                        #    connectionstyle=connection_style
                           )
    # 绘制 Rail 边
    nx.draw_networkx_edges(G, pos, 
                           edgelist=graph_data['rail_edges'], 
                           style=style['rail_style'], 
                           alpha=style['rail_alpha'], 
                           edge_color=style['rail_color'], 
                           width=style['rail_width'],
                           arrows=True,
                           arrowstyle='-|>',
                           arrowsize=15,
                        #    connectionstyle=connection_style
                           )
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                           node_color=graph_data['node_colors'], 
                           node_size=style['node_size'],
                           edgecolors=graph_data['node_border_colors'],
                           linewidths=style['emergency_border_width']
                           )
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=style['font_size'], font_color=style['font_color'])
    
    # 步骤 4: 创建并显示图例和标题
    legend_elements = _create_legend(style)
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.title("Topology of transport network", fontsize=20)
    plt.box(False)

    if save_path:
        # 确保保存路径的目录存在
        # os.path.dirname 可能会返回空字符串，导致os.makedirs失败
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        plt.savefig(save_path)
        import logging
        logging.info(f"网络拓扑图已保存至: {save_path}")
        
    # plt.close(fig) # 增加 plt.close() 释放内存


def _prepare_graph_data(network: TransportNetwork, style_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    [辅助方法] 从网络数据中准备 NetworkX 绘图所需的数据（支持有向图）
    """
    G: nx.DiGraph = nx.DiGraph()
    
    node_ids = [node.id for node in network.nodes]
    G.add_nodes_from(node_ids)
    
    node_colors = []
    node_border_colors = []
    
    for node in network.nodes:
        node_colors.append(style_config['color_map'].get(node.type, 'gray'))
        if node.is_emergency_center:
            node_border_colors.append(style_config['emergency_border_color'])
        else:
            node_border_colors.append(style_config['color_map'].get(node.type, 'gray'))

    road_edges_list = []
    rail_edges_list = []

    for arc in network.arcs:
        edge_tuple = (arc.start.id, arc.end.id)
        G.add_edge(*edge_tuple)
        
        if arc.mode == 'road':
            road_edges_list.append(edge_tuple)
        elif arc.mode == 'railway':
            rail_edges_list.append(edge_tuple)
    
    return {
        "graph": G,
        "node_colors": node_colors,
        "node_border_colors": node_border_colors,
        "road_edges": road_edges_list,
        "rail_edges": rail_edges_list
    }

def _get_style_config() -> Dict[str, Any]:
    """
    [辅助方法] 返回一个包含所有绘图样式的配置字典。
    """
    return {
        "figure_size": (20, 15),
        "node_size": 600,
        "font_size": 10,
        "font_color": 'black',
        "color_map": {'hub': '#ff4757', 'non-hub': '#54a0ff'},
        "emergency_border_color": '#ffd700',
        "emergency_border_width": 2.5,
        "road_style": "dashed",
        "road_color": "gray",
        "road_alpha": 0.7,
        "rail_style": "solid",
        "rail_color": "black",
        "rail_width": 2.0,
        "rail_alpha": 1.0
    }

def _create_legend(style_config: Dict[str, Any]) -> List:
    """
    [辅助方法] 这个模块只创建一个清晰的图例。
    """

    legend_elements = [
        Patch(facecolor=style_config['color_map']['hub'], edgecolor='none', label='Hub Node'),
        Patch(facecolor=style_config['color_map']['non-hub'], edgecolor='none', label='Non-Hub Node'),
        Patch(facecolor='white', edgecolor=style_config['emergency_border_color'], linewidth=2, label='Emergency Center (Border)'),
        Line2D([0], [0], color=style_config['rail_color'], lw=style_config['rail_width'], label='Railway'),
        Line2D([0], [0], color=style_config['road_color'], linestyle=style_config['road_style'], lw=1.5, label='Road')
    ]
    return legend_elements