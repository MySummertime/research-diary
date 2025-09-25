# --- coding: utf-8 ---
# --- network.py ---
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    """
    定义网络中的一个节点。
    """
    def __init__(self, node_id: str, node_type: str, is_emergency_center: bool = False, capacity: float=0, population_density: float=0):
        self.id: str = node_id
        # 节点类型: 'hub', 'non-hub'
        self.type: str = node_type
        # 节点角色：是否为应急中心 (True/False)
        self.is_emergency_center: bool = is_emergency_center
        # 节点容量 (t)
        self.capacity: float = capacity
        # 节点附近的人口密度 (p/m^2)
        self.population_density: float = population_density

    def __repr__(self):
        role = f", Role={'Emergency Center' if self.is_emergency_center else 'None'}"
        return f"Node(ID={self.id}, Type='{self.type}{role}')"

class Arc:
    """
    定义连接两个节点的弧段。
    """
    def __init__(self, start_node: Node, end_node: Node, mode: str, length: float=1, capacity: float=1000, accident_prob_per_km: float=1e-7):
        self.start: Node = start_node
        self.end: Node = end_node
        # 运输模式: 'road' or 'railway'
        self.mode: str = mode
        # 弧段长度 (km)
        self.length: float = length
        # 弧段容量 (t)
        self.capacity: float = capacity
        # 每公里的单位事故概率
        self.accident_prob_per_km: float = accident_prob_per_km
        # 总事故概率 = 单位概率 * 长度
        self.total_accident_prob: float = self.accident_prob_per_km * self.length

    def __repr__(self):
        return f"Arc({self.start.id}->{self.end.id}, Mode='{self.mode}', Len={self.length}km)"

class TransportTask:
    """
    定义一个运输任务，即一个OD对。
    """
    def __init__(self, task_id: str, origin_node: Node, destination_node: Node):
        self.id: str = task_id
        self.origin: Node = origin_node
        self.destination: Node = destination_node
        # 确保OD点都是non-hub
        if self.origin.type != 'non-hub' or self.destination.type != 'non-hub':
            raise ValueError("Origin and Destination for a task must be 'non-hub' nodes.")

    def __repr__(self):
        return f"Task(ID={self.id}, O={self.origin.id}, D={self.destination.id})"

class TransportNetwork:
    """
    一个通用的运输网络数据容器。
    它只负责存储节点、弧段和任务，不关心它们是如何被创建的。
    """
    def __init__(self):
        self.nodes = []
        self.arcs = []
        self.tasks = []
        # 为了方便查找，增加字典来存储节点对象
        self._nodes_dict = {}

    # --- 增加功能 ---

    def add_node(self, node: Node):
        """向网络中添加一个节点。"""
        if node.id not in self._nodes_dict:
            self.nodes.append(node)
            self._nodes_dict[node.id] = node

    def add_arc(self, start_node_id: str, end_node_id: str, oneway: bool = False, **kwargs):
        """
        向网络中添加一条弧段。

        Args:
            start_node_id (str): 起点节点ID。
            end_node_id (str): 终点节点ID。
            oneway (bool, optional): 是否为单向。默认为 False (即双向)。
            **kwargs: 其他传递给 Arc 构造函数的参数 (mode, length 等)。
        """
        if start_node_id not in self._nodes_dict or end_node_id not in self._nodes_dict:
            raise ValueError("弧段的起点或终点不存在于网络中。")
            
        start_node = self._nodes_dict[start_node_id]
        end_node = self._nodes_dict[end_node_id]
        
        # 添加 A -> B 这条弧段
        self.arcs.append(Arc(start_node, end_node, **kwargs))
        
        # 如果是双向弧 (默认情况)，则自动添加反向弧段 B -> A
        if not oneway:
            # 注意：反向弧段共享同样的属性 (mode, length 等)
            self.arcs.append(Arc(end_node, start_node, **kwargs))
    
    def add_task(self, task: TransportTask):
        """向网络中添加一个运输任务。"""
        self.tasks.append(task)

    # --- 查询功能 ---

    def get_hubs(self):
        """获取所有枢纽节点。"""
        return [node for node in self.nodes if node.type == 'hub']
    
    def get_non_hubs(self):
        """获取所有非枢纽节点。"""
        return [node for node in self.nodes if node.type == 'non-hub']

    def get_emergency_centers(self):
        """获取所有应急中心节点。"""
        return [node for node in self.nodes if node.is_emergency_center]

    def summary(self):
        """打印网络摘要信息。"""
        print("\n--- 网络摘要 ---")
        print(f"总节点数: {len(self.nodes)}")
        print(f"  - 枢纽: {len(self.get_hubs())}")
        print(f"  - 非枢纽: {len(self.get_non_hubs())}")
        print(f"总弧段数: {len(self.arcs)}")
        print(f"总任务数: {len(self.tasks)}")
        print("------------------")

    # --- 可视化功能 ---

    def visualize(self, layout_func_name: str = 'spring'):
        """
        使用 networkx 和 matplotlib 可视化网络拓扑.
        """
        # 步骤 1: 获取样式配置和准备好的绘图数据
        style = self._get_style_config()
        graph_data = self._prepare_graph_data()
        G = graph_data["graph"]

        # 步骤 2: 计算节点布局（使用指定的布局函数）
        if layout_func_name == 'spring':
            pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)    # 添加 seed 以保证每次布局相同
        elif layout_func_name == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # 步骤 3: 开始绘图
        plt.figure(figsize=style['figure_size'])
        
        # 绘制不同类型的边
        nx.draw_networkx_edges(G, pos, edgelist=graph_data['road_edges'], style=style['road_style'], alpha=style['road_alpha'], edge_color=style['road_color'])
        nx.draw_networkx_edges(G, pos, edgelist=graph_data['rail_edges'], style=style['rail_style'], alpha=style['rail_alpha'], edge_color=style['rail_color'], width=style['rail_width'])
        
        # 绘制节点，并加入边框效果来突出应急中心
        nx.draw_networkx_nodes(
            G, 
            pos, 
            node_color=graph_data['node_colors'], 
            node_size=style['node_size'],
            edgecolors=graph_data['node_border_colors'],    # 设置节点边框颜色
            linewidths=style['emergency_border_width']  # 设置节点边框宽度
        )
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=style['font_size'], font_color=style['font_color'])
        
        # 步骤 4: 创建并显示图例和标题
        legend_elements = self._create_legend(style)
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        plt.title("Topology of transport network", fontsize=20)
        plt.box(False)  # 移除外边框
        plt.show()

    def _prepare_graph_data(self):
        """
        [辅助方法] 从网络数据中准备 NetworkX 绘图所需的数据。
        这个模块将数据处理与绘图逻辑分离。
        """
        G = nx.Graph()
        
        # 准备节点属性
        node_ids = [node.id for node in self.nodes]
        G.add_nodes_from(node_ids)
        
        node_colors = []
        node_border_colors = []
        
        # 增加应急节点的边框颜色
        style_config = self._get_style_config()
        
        for node in self.nodes:
            node_colors.append(style_config['color_map'].get(node.type, 'gray'))
            if node.is_emergency_center:
                node_border_colors.append(style_config['emergency_border_color'])
            else:
                # 普通节点的边框色与填充色相同，使其不可见
                node_border_colors.append(style_config['color_map'].get(node.type, 'gray'))

        # 准备弧段属性
        road_edges_set = set()
        rail_edges_set = set()
        for arc in self.arcs:
            # 使用 frozenset 来处理无向图的重复边，比 G.edges() 更高效
            edge_frozenset = frozenset([arc.start.id, arc.end.id])
            if arc.mode == 'road':
                road_edges_set.add(edge_frozenset)
            elif arc.mode == 'railway':
                rail_edges_set.add(edge_frozenset)
        
        # --- 将 frozenset 的集合转换为 tuple 的列表 ---
        final_road_edges = [tuple(edge) for edge in road_edges_set]
        final_rail_edges = [tuple(edge) for edge in rail_edges_set]
        
        return {
            "graph": G,
            "node_colors": node_colors,
            "node_border_colors": node_border_colors,
            "road_edges": final_road_edges,
            "rail_edges": final_rail_edges
        }

    def _get_style_config(self):
        """
        [辅助方法] 返回一个包含所有绘图样式的配置字典。
        这个模块将样式配置与绘图逻辑分离。
        """
        return {
            "figure_size": (16, 12),
            "node_size": 600,
            "font_size": 10,
            "font_color": 'black',
            "color_map": {'hub': '#ff4757', 'non-hub': '#54a0ff'},  # 使用明确的颜色代码
            "emergency_border_color": '#ffd700',    # 金色边框
            "emergency_border_width": 2.5,
            "road_style": "dashed",
            "road_color": "gray",
            "road_alpha": 0.7,
            "rail_style": "solid",
            "rail_color": "black",
            "rail_width": 2.0,
            "rail_alpha": 1.0
        }

    def _create_legend(self, style_config):
        """
        [辅助方法] 这个模块只创建一个清晰的图例。
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_elements = [
            # 使用 Patch 来创建带边框的图例项
            Patch(facecolor=style_config['color_map']['hub'], edgecolor='none', label='Hub Node'),
            Patch(facecolor=style_config['color_map']['non-hub'], edgecolor='none', label='Non-Hub Node'),
            Patch(facecolor='white', edgecolor=style_config['emergency_border_color'], linewidth=2, label='Emergency Center (Border)'),
            Line2D([0], [0], color=style_config['rail_color'], lw=style_config['rail_width'], label='Railway'),
            Line2D([0], [0], color=style_config['road_color'], linestyle=style_config['road_style'], lw=1.5, label='Road')
        ]
        return legend_elements