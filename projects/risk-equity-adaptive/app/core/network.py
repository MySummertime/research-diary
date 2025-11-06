# --- coding: utf-8 ---
# --- app/core/network.py ---
import json
from typing import Dict

class Node:
    """
    定义网络中的一个节点。
    """
    def __init__(self, node_id: str, node_type: str, is_emergency_center: bool = False, transshipment_cost: float = 0.0, capacity: float=0, population_density: float=0.0, accident_prob: float=1e-7, actual_flow: float=0.0, fuzzy_transshipment_time: tuple[float, float, float, float]=(0.0, 0.0, 0.0, 0.0)):
        # 节点编号
        self.id: str = node_id
        # 节点类型: 'hub', 'non-hub'
        self.type: str = node_type
        # 转运成本 (yuan/t·h)
        self.transshipment_cost: float = transshipment_cost
        # 节点容量 (t)
        self.capacity: float = capacity
        # 节点附近的人口密度 (p/m^2)
        self.population_density: float = population_density
        # 节点角色：是否为应急中心 (True/False)
        self.is_emergency_center: bool = is_emergency_center
        # 节点事故概率
        self.accident_prob: float = accident_prob
        # 实际运量：需求（t）
        self.actual_flow: float = actual_flow
        # 模糊转运时间：(t1, t2, t3, t4)
        self.fuzzy_transshipment_time: tuple[float, float, float, float] = fuzzy_transshipment_time

    def __repr__(self):
        role = f", Role={'Emergency Center' if self.is_emergency_center else 'None'}"
        return f"Node(ID={self.id}, Type='{self.type}{role}')"
    
    def to_dict(self) -> Dict:
        """将节点属性转换为字典，方便JSON序列化。"""
        return {
            "node_id": self.id,
            "node_type": self.type,
            "transshipment_cost": self.transshipment_cost,
            "capacity": self.capacity,
            "population_density": self.population_density,
            "is_emergency_center": self.is_emergency_center,
            "accident_prob": self.accident_prob,
            "actual_flow": self.actual_flow,
            "fuzzy_transshipment_time": self.fuzzy_transshipment_time
        }


class Arc:
    """
    定义连接两个节点的弧段。
    """
    def __init__(self, start_node: Node, end_node: Node, oneway: bool, mode: str, length: float=1, capacity: float=1000, population_density: float=0, shipment_cost: float = 0.0, carbon_cost_per_ton_km: float = 0.0, accident_prob_per_km: float=1e-7, actual_flow: float=0.0, fuzzy_transport_time: tuple[float, float, float]=(0.0, 0.0, 0.0)):
        # 起始节点
        self.start: Node = start_node
        # 终止节点
        self.end: Node = end_node
        # 是否为单项弧段
        self.oneway: bool = oneway
        # 运输模式: 'road' or 'railway'
        self.mode: str = mode
        # 弧段长度 (km)
        self.length: float = length
        # 弧段容量 (t)
        self.capacity: float = capacity
        # 弧段附近的人口密度 (p/m^2)
        self.population_density: float = population_density
        # 运输成本 (yuan/t·h)
        self.shipment_cost: float = shipment_cost
        # 每公里的碳排放成本 (yuan/t·km)
        self.carbon_cost_per_ton_km: float = carbon_cost_per_ton_km
        # 每公里的单位事故概率
        self.accident_prob_per_km: float = accident_prob_per_km
        # 实际运量：需求（t）
        self.actual_flow: float = actual_flow
        # 模糊运输时间: (t1, t2, t3)
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
            "shipment_cost": self.shipment_cost,
            "carbon_cost_per_ton_km": self.carbon_cost_per_ton_km,
            "accident_prob_per_km": self.accident_prob_per_km,
            "actual_flow": self.actual_flow,
            "fuzzy_transport_time": self.fuzzy_transport_time
        }


class TransportTask:
    """
    定义一个运输任务，即一个OD对。
    """
    def __init__(self, task_id: str, origin_node: Node, destination_node: Node, demand: float=0.0):
        self.task_id: str = task_id
        self.origin: Node = origin_node
        self.destination: Node = destination_node
        self.demand: float = demand
        # 确保OD点都是non-hub
        if self.origin.type != 'non-hub' or self.destination.type != 'non-hub':
            raise ValueError("Origin and Destination for a task must be 'non-hub' nodes.")

    def __repr__(self):
        return f"Task(ID={self.task_id}, O={self.origin.id}, D={self.destination.id})"
    
    def to_dict(self) -> Dict:
        """将任务属性转换为字典。"""
        return {
            "task_id": self.task_id,
            "origin_node_id": self.origin.id,
            "destination_node_id": self.destination.id,
            "demand": self.demand
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
    
    def save_to_json(self, file_path: str):
        """
        将当前网络的所有信息（节点、弧段、任务）保存到一个JSON文件中。

        Args:
            file_path (str): 保存JSON文件的完整路径。
        """
        network_data = {
            "nodes": [node.to_dict() for node in self.nodes],
            "arcs": [arc.to_dict() for arc in self.arcs],
            "tasks": [task.to_dict() for task in self.tasks]
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(network_data, f, ensure_ascii=False, indent=4)
            print(f"网络数据已成功保存至: {file_path}")
        except Exception as e:
            print(f"错误：保存网络数据到JSON文件时失败: {e}")

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
            oneway (bool, optional): 是否为单向。默认为 true.
            **kwargs: 其他传递给 Arc 构造函数的参数 (mode, length 等)。
        """
        if start_node_id not in self._nodes_dict or end_node_id not in self._nodes_dict:
            raise ValueError("弧段的起点或终点不存在于网络中。")
            
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
