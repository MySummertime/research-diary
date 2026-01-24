# --- coding: utf-8 ---
# --- app/core/network.py ---
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# ==========================================
# 1. 基础数据结构 (Data Classes)
# ==========================================


@dataclass
class Node:
    """
    [数据类] 网络节点
    职责：存储节点的物理属性和风险属性。
    """

    node_id: str
    name: str = ""
    node_type: str = "non-hub"  # hub, non-hub

    # 几何属性
    x: float = 0.0
    y: float = 0.0

    # 业务属性
    is_emergency_center: bool = False
    capacity: float = 250000.0
    population_density: float = 0.01
    accident_prob: float = 1e-7

    # 模糊属性
    # (a, b, c, d) 梯形模糊数
    fuzzy_transshipment_time: Tuple[float, float, float, float] = (
        1.00,
        1.33,
        1.67,
        2.00,
    )

    def __post_init__(self):
        if not self.name:
            self.name = self.node_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Arc:
    """
    [数据类] 网络弧段 (有向边)
    职责：存储链路的物理属性、风险属性和模糊时间。
    注意：即便是双向路，在模型中也表示为两条相反方向的 Arc。
    """

    start: Node
    end: Node
    mode: str  # 'road', 'railway'

    length: float = 150.0  # km
    capacity: float = 10000.0

    # 风险属性
    population_density: float = 0.008
    accident_prob_per_km: float = 1e-7

    # 模糊属性
    # (a, b, c) 三角模糊数
    fuzzy_transport_time: Tuple[float, float, float] = (2.5, 3.0, 3.5)

    @property
    def id(self) -> Tuple[str, str]:
        """弧段唯一标识符 (起点ID, 终点ID)"""
        return (self.start.node_id, self.end.node_id)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # 序列化时只需存储 ID
        d["start_node_id"] = self.start.node_id
        d["end_node_id"] = self.end.node_id
        del d["start"]
        del d["end"]
        return d


@dataclass
class TransportTask:
    """
    [数据类] 运输任务 (OD Pair)
    """

    task_id: str
    origin: Node
    destination: Node
    demand: float = 0.0

    def __post_init__(self):
        # 简单校验
        if (
            self.origin.node_type != "non-hub"
            or self.destination.node_type != "non-hub"
        ):
            logging.warning(
                f"Task {self.task_id}: Origin/Dest are recommended to be 'non-hub'."
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # 序列化时只需存储 ID
        d["origin_node_id"] = self.origin.node_id
        d["destination_node_id"] = self.destination.node_id
        del d["origin"]
        del d["destination"]
        return d


# ==========================================
# 2. 网络逻辑核心 (Logic Class)
# ==========================================


class TransportNetwork:
    """
    [核心类] 运输网络容器
    职责：
    1. 管理节点、弧段、任务的增删改查。
    2. 维护 NetworkX 图对象 (Graph Topology)。
    3. 提供数据的序列化与反序列化。
    """

    def __init__(self):
        # 核心数据存储
        self.nodes: List[Node] = []
        self.arcs: List[Arc] = []
        self.tasks: List[TransportTask] = []

        # 快速查找索引
        self._nodes_map: Dict[str, Node] = {}
        self._arcs_map: Dict[Tuple[str, str], Arc] = {}
        self._tasks_map: Dict[str, TransportTask] = {}

        # 拓扑缓存 (Lazy Loading)
        self._graph_cache: Optional[nx.DiGraph] = None

    # --- 核心操作：增 ---

    def add_node(self, node: Node):
        """添加节点并建立索引。"""
        if node.node_id in self._nodes_map:
            logging.warning(f"Node {node.node_id} updated/overwritten.")
        else:
            self.nodes.append(node)

        self._nodes_map[node.node_id] = node
        self._invalidate_cache()

    def add_arc(
        self,
        start_id: str,
        end_id: str,
        mode: str = "road",
        length: float = 100,
        bidirectional: bool = True,
        **kwargs,
    ):
        """
        添加弧段。

        Args:
            start_id: 起点ID
            end_id: 终点ID
            mode: 运输模式 ('road', 'railway')
            distance: 距离 (km)
            bidirectional: 是否自动创建反向弧段 (默认 True)
            **kwargs: 其他 Arc 属性 (如 fuzzy_transport_time, capacity 等)
        """
        if start_id not in self._nodes_map or end_id not in self._nodes_map:
            raise KeyError(f"Cannot add arc: Node {start_id} or {end_id} not found.")

        u_node = self._nodes_map[start_id]
        v_node = self._nodes_map[end_id]

        # 1. 创建正向弧 A -> B
        forward_arc = Arc(start=u_node, end=v_node, mode=mode, length=length, **kwargs)
        self._add_arc_internal(forward_arc)

        # 2. 如果是双向，创建反向弧 B -> A
        if bidirectional:
            backward_arc = Arc(
                start=v_node, end=u_node, mode=mode, length=length, **kwargs
            )
            self._add_arc_internal(backward_arc)

        self._invalidate_cache()

    def add_task(self, task: TransportTask):
        self.tasks.append(task)
        self._tasks_map[task.task_id] = task

    def _add_arc_internal(self, arc: Arc):
        """内部方法：将弧段加入列表和索引。"""
        self.arcs.append(arc)
        self._arcs_map[arc.id] = arc

    # --- 核心属性：图拓扑 ---

    @property
    def graph(self) -> nx.DiGraph:
        """
        [Lazy Property] 获取 NetworkX 有向图对象。
        该对象包含完整的拓扑结构和关键属性，可直接用于 visualizer 和 path finding。

        Returns:
            nx.DiGraph: 带有节点属性 (type, x, y) 和 边属性 (mode, length) 的图。
        """
        if self._graph_cache is None:
            self._graph_cache = self._build_graph()
        return self._graph_cache

    def _build_graph(self) -> nx.DiGraph:
        """构建图的实际逻辑。"""
        G = nx.DiGraph()

        # 添加节点 (附带可视化和算法所需的关键属性)
        for node in self.nodes:
            G.add_node(
                node.node_id,
                type=node.node_type,
                is_emergency=node.is_emergency_center,
                x=node.x,
                y=node.y,
            )

        # 添加边 (附带属性)
        for arc in self.arcs:
            G.add_edge(
                arc.start.node_id,
                arc.end.node_id,
                mode=arc.mode,
                length=arc.length,
                # 可以根据需要添加更多属性供 path finder 使用
                capacity=arc.capacity,
            )

        return G

    def _invalidate_cache(self):
        """当网络结构发生变化时，清空缓存。"""
        self._graph_cache = None

    # --- 查询与工具 ---

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes_map.get(node_id)

    def get_arc(self, u: str, v: str) -> Optional[Arc]:
        return self._arcs_map.get((u, v))

    def get_hubs(self) -> List[Node]:
        """获取所有枢纽节点"""
        return [n for n in self.nodes if n.node_type == "hub"]

    def get_non_hubs(self) -> List[Node]:
        """获取所有非枢纽节点"""
        return [n for n in self.nodes if n.node_type == "non-hub"]

    def get_emergency_centers(self) -> List[Node]:
        """获取所有应急中心节点"""
        return [n for n in self.nodes if n.is_emergency_center]

    def get_task(self, task_id: str) -> Optional[TransportTask]:
        """根据任务ID获取任务对象"""
        return self._tasks_map.get(task_id)

    # --- 序列化与摘要 ---

    def summary(self):
        logging.info("=== Network Summary ===")
        logging.info(
            f"Nodes: {len(self.nodes)} (Hubs: {len([n for n in self.nodes if n.node_type == 'hub'])})"
        )
        logging.info(f"Arcs : {len(self.arcs)} (Bidirectional logic applied)")
        logging.info(f"Tasks: {len(self.tasks)}")
        logging.info("=======================")

    def save_to_json(self, file_path: str):
        data = {
            "nodes": [n.to_dict() for n in self.nodes],
            "arcs": [a.to_dict() for a in self.arcs],
            "tasks": [t.to_dict() for t in self.tasks],
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"Network exported to {file_path}")
        except IOError as e:
            logging.error(f"Failed to export network: {e}")
