# --- coding: utf-8 ---
# --- app/core/network.py ---
import json
import logging
from typing import Dict


class Node:
    """
    定义网络中的一个节点。
    """

    def __init__(
        self,
        node_id: str,
        node_name: str = "",
        node_type: str = "non-hub",
        x: float = 0.0,
        y: float = 0.0,
        is_emergency_center: bool = False,
        capacity: float = 10000,
        population_density: float = 0.01,
        accident_prob: float = 1e-7,
        actual_flow: float = 0.0,
        fuzzy_transshipment_time: tuple[float, float, float, float] = (
            0.8,
            1.0,
            1.2,
            1.5,
        ),
        **kwargs,
    ):
        # 基础标识
        self.id: str = node_id
        self.name: str = node_name if node_name else self.id
        self.type: str = node_type  # hub, non-hub

        # 几何属性（用于物理模型计算）
        self.x: float = x
        self.y: float = y

        # 节点角色与属性
        self.is_emergency_center: bool = is_emergency_center
        self.capacity: float = capacity  # t

        # 风险相关
        self.population_density: float = population_density  # p/m^2
        self.accident_prob: float = accident_prob

        # 状态与不确定性
        self.actual_flow: float = actual_flow  # demand (t)
        self.fuzzy_transshipment_time: tuple[float, float, float, float] = (
            fuzzy_transshipment_time
        )

    def __repr__(self):
        role = f", Role={'Emergency Center' if self.is_emergency_center else 'None'}"
        return f"Node(ID={self.id}, Name={self.name}, Type='{self.type}{role}', Pos=({self.x},{self.y}))"

    def to_dict(self) -> Dict:
        """
        [辅助方法]将节点属性转换为字典，方便JSON序列化。
        """
        return {
            "node_id": self.id,
            "name": self.name,
            "node_type": self.type,
            "x": self.x,
            "y": self.y,
            "capacity": self.capacity,
            "population_density": self.population_density,
            "is_emergency_center": self.is_emergency_center,
            "accident_prob": self.accident_prob,
            "actual_flow": self.actual_flow,
            "fuzzy_transshipment_time": self.fuzzy_transshipment_time,
        }


class Arc:
    """
    定义连接两个节点的弧段。
    """

    def __init__(
        self,
        start_node: Node,
        end_node: Node,
        oneway: bool,
        mode: str,
        length: float = 150,
        capacity: float = 10000,
        population_density: float = 0.008,
        accident_prob_per_km: float = 1e-7,
        actual_flow: float = 0.0,
        fuzzy_transport_time: tuple[float, float, float] = (2.5, 3.0, 3.5),
    ):
        self.start: Node = start_node
        self.end: Node = end_node
        self.oneway: bool = oneway
        self.mode: str = mode  # 'road' or 'railway'
        self.length: float = length  # km

        self.capacity: float = capacity  # t
        self.population_density: float = population_density  # p/m^2
        self.accident_prob_per_km: float = accident_prob_per_km

        self.actual_flow: float = actual_flow  # demand (t)
        self.fuzzy_transport_time: tuple[float, float, float] = fuzzy_transport_time

    def __repr__(self):
        return f"Arc({self.start.id}->{self.end.id}, Mode='{self.mode}', Len={self.length}km)"

    def to_dict(self) -> Dict:
        """将弧段属性转换为字典。"""
        return {
            "start_node_id": self.start.id,
            "end_node_id": self.end.id,
            "oneway": self.oneway,
            "mode": self.mode,
            "length": self.length,
            "capacity": self.capacity,
            "population_density": self.population_density,
            "accident_prob_per_km": self.accident_prob_per_km,
            "actual_flow": self.actual_flow,
            "fuzzy_transport_time": self.fuzzy_transport_time,
        }


class TransportTask:
    """
    运输任务（OD Pair）
    """

    def __init__(
        self,
        task_id: str,
        origin_node: Node,
        destination_node: Node,
        demand: float = 0.0,
    ):
        self.task_id: str = task_id
        self.origin: Node = origin_node
        self.destination: Node = destination_node
        self.demand: float = demand

        # 确保OD点都是non-hub
        if self.origin.type != "non-hub" or self.destination.type != "non-hub":
            raise ValueError(f"Task {task_id}: Origin/Dest must be 'non-hub' nodes.")

    def __repr__(self):
        return f"Task({self.task_id}: {self.origin.id}->{self.destination.id}, d={self.demand})"

    def to_dict(self) -> Dict:
        """将任务属性转换为字典。"""
        return {
            "task_id": self.task_id,
            "origin_node_id": self.origin.id,
            "destination_node_id": self.destination.id,
            "demand": self.demand,
        }


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
        self._arcs_dict = {}

    # --- 增加功能 ---

    def add_node(self, node: Node):
        """向网络中添加一个节点。"""
        if node.id not in self._nodes_dict:
            self.nodes.append(node)
            self._nodes_dict[node.id] = node

    def add_arc(
        self, start_node_id: str, end_node_id: str, oneway: bool = False, **kwargs
    ):
        """
        向网络中添加一条弧段。

        Args:
            start_node_id (str): 起点节点ID。
            end_node_id (str): 终点节点ID。
            oneway (bool, optional): 是否为单向。默认为 true.
            **kwargs: 其他传递给 Arc 构造函数的参数 (mode, length 等)。
        """
        if start_node_id not in self._nodes_dict or end_node_id not in self._nodes_dict:
            raise ValueError(f"Unknown nodes in arc: {start_node_id}->{end_node_id}")

        start_node = self._nodes_dict[start_node_id]
        end_node = self._nodes_dict[end_node_id]

        # 创建并添加 A -> B 这条弧段
        arc_forward = Arc(start_node, end_node, oneway=oneway, **kwargs)
        self.arcs.append(arc_forward)
        self._arcs_dict[(start_node_id, end_node_id)] = arc_forward

        # 如果是双向弧，则添加反向弧段 B -> A
        if not oneway:
            arc_backward = Arc(end_node, start_node, oneway=oneway, **kwargs)
            self.arcs.append(arc_backward)
            # 同时更新查找字典
            self._arcs_dict[(end_node_id, start_node_id)] = arc_backward

    def add_task(self, task: TransportTask):
        """向网络中添加一个运输任务。"""
        self.tasks.append(task)

    # --- 查询功能 ---

    def get_hubs(self):
        """获取所有枢纽节点。"""
        return [node for node in self.nodes if node.type == "hub"]

    def get_non_hubs(self):
        """获取所有非枢纽节点。"""
        return [node for node in self.nodes if node.type == "non-hub"]

    def get_emergency_centers(self):
        """获取所有应急中心节点。"""
        return [node for node in self.nodes if node.is_emergency_center]

    def save_to_json(self, file_path: str):
        """
        Dump Network to JSON.

        Args:
            file_path (str): 保存JSON文件的完整路径。
        """
        network_data = {
            "nodes": [node.to_dict() for node in self.nodes],
            "arcs": [arc.to_dict() for arc in self.arcs],
            "tasks": [task.to_dict() for task in self.tasks],
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(network_data, f, ensure_ascii=False, indent=4)
            logging.info(f"网络数据已导出至: {file_path}")
        except Exception as e:
            logging.warning(f"网络数据到JSON导出失败: {e}")

    def summary(self):
        """打印网络摘要信息。"""
        logging.info("--- 网络摘要 ---")
        logging.info(f"总节点数: {len(self.nodes)}")
        logging.info(f"  - 枢纽: {len(self.get_hubs())}")
        logging.info(f"  - 非枢纽: {len(self.get_non_hubs())}")
        logging.info(f"总弧段数: {len(self.arcs)}")
        logging.info(f"总任务数: {len(self.tasks)}")
        logging.info("------------------")
