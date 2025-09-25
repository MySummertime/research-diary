# --- coding: utf-8 ---
# --- network_generator.py ---
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from core.network import Node, TransportTask, TransportNetwork

# --- 网络生成器：抽象基类（由所有的具体网络生成器继承） ---
class AbstractNetworkGenerator(ABC):
    """
    网络生成器的抽象基类 (蓝图)。
    所有具体的网络生成器都应该继承这个类，并实现 generate 方法。
    """
    def __init__(self):
        self.network = TransportNetwork()

    @abstractmethod
    def generate(self) -> TransportNetwork:
        """
        生成并返回一个 TransportNetwork 实例。
        这是一个抽象方法，子类必须重写它。
        """
        pass


# --- 数据类：Hub-and-Spoke Network 基本参数 ---
@dataclass
class HaSConfig:
    """使用 dataclass 创建的一个配置对象，让参数管理更清晰"""
    num_nodes: int = 40
    num_hubs: int = 5
    num_emergency_nodes: int = 8
    num_tasks: int = 20
    road_connect_prob: float = 0.6


# --- 网络生成器：Hub-and-Spoke Network 具体实现 ---
class HaSNetworkGenerator(AbstractNetworkGenerator):
    """
    一个具体的网络生成器，负责创建 hub-and-spoke 类型的随机网络。
    在具体实现时，新建一个 HaSNetwork 对象，使用此生成器的方法来生成网络。
    """
    def __init__(self, config: HaSConfig):
        super().__init__()
        self.config = config

    def generate(self) -> TransportNetwork:
        """实现 generate 方法，完成网络构建。"""
        print("开始使用 Hub-and-Spoke 生成器构建网络...")
        self._create_nodes()
        self._create_arcs()
        self._create_tasks()
        print("网络构建完成！🎉")
        
        # 打印摘要信息
        print(f"创建了 {self.config.num_nodes} 个节点, "
              f"{self.config.num_hubs} 个枢纽, "
              f"{self.config.num_emergency_nodes} 个应急中心, "
              f"{self.config.num_tasks} 个任务。")
              
        return self.network

    # _create_nodes, _create_arcs, _create_tasks 等私有方法的实现
    def _create_nodes(self):
        """Create nodes"""
        node_ids = list(range(self.config.num_nodes))
        random.shuffle(node_ids)
        # 生成枢纽节点
        for _ in range(self.config.num_hubs):
            hub_id = node_ids.pop()
            self.network.add_node(Node(str(hub_id), 'hub', capacity=random.randint(1500, 2500), population_density=random.uniform(0.5, 1.0)))
        # 生成非枢纽节点
        for non_hub_id in node_ids:
            self.network.add_node(Node(str(non_hub_id), 'non-hub', capacity=random.randint(500, 1000), population_density=random.uniform(0.1, 0.5)))
        # 从所有已创建的节点中，随机指定应急中心
        if self.config.num_emergency_nodes > 0 and self.config.num_nodes >= self.config.num_emergency_nodes:
            # 随机抽取指定数量的节点
            emergency_nodes = random.sample(self.network.nodes, self.config.num_emergency_nodes)
            # 将这些节点的应急中心属性设为 True
            for node in emergency_nodes:
                node.is_emergency_center = True

    def _create_arcs(self):
        """Create arcs"""
        hubs = self.network.get_hubs()
        nodes = self.network.nodes

        # 为了避免重复创建弧段，用一个集合来记录已经存在的弧段
        existing_arcs = set()

        # 1. 创建铁路网络
        for i in range(len(hubs)):
            for j in range(i + 1, len(hubs)):
                # 确保弧段不存在再添加
                if tuple(sorted((hubs[i].id, hubs[j].id))) not in existing_arcs:
                    self.network.add_arc(hubs[i].id, hubs[j].id, mode='railway', length=random.randint(200, 500))
                    existing_arcs.add(tuple(sorted((hubs[i].id, hubs[j].id))))

        # 2. 创建公路网络
        all_node_ids = [n.id for n in nodes]
        for i in range(len(all_node_ids)):
            for j in range(i + 1, len(all_node_ids)):
                node1 = self.network._nodes_dict[all_node_ids[i]]
                node2 = self.network._nodes_dict[all_node_ids[j]]
                if node1.type == 'hub' and node2.type == 'hub':
                    continue
                if random.random() < self.config.road_connect_prob:
                    if tuple(sorted((node1.id, node2.id))) not in existing_arcs:
                        self.network.add_arc(node1.id, node2.id, mode='road', length=random.randint(20, 100))
                        existing_arcs.add(tuple(sorted((node1.id, node2.id))))
        
        # 3. 为应急中心创建专用的公路连接
        emergency_centers = self.network.get_emergency_centers()
        for center in emergency_centers:
            for node in nodes:
                # 应急中心不需要连接自己
                if center.id == node.id:
                    continue
                # 如果连接尚不存在，则创建一条公路弧段
                if tuple(sorted((center.id, node.id))) not in existing_arcs:
                    # 应急响应路径通常是比较直接的道路
                    self.network.add_arc(center.id, node.id, mode='road', length=random.randint(10, 50))
                    existing_arcs.add(tuple(sorted((center.id, node.id))))

    def _create_tasks(self):
        """Create transport tasks"""
        non_hubs = self.network.get_non_hubs()
        if len(non_hubs) < 2:
            print("警告：非枢纽节点不足，无法创建任务。")
            return
        for i in range(self.config.num_tasks):
            origin, destination = random.sample(non_hubs, 2)
            self.network.add_task(TransportTask(f"T{i+1}", origin, destination))