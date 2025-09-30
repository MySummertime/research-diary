# --- coding: utf-8 ---
# --- network_generator.py ---
import random
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
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
    predefined_tasks: Optional[List[Tuple[str, str]]] = None
    # --- 成本参数 ---
    cost_road_per_km: float = 0.5
    cost_rail_per_km: float = 0.3
    cost_hub_transshipment: float = 15.0
    carbon_road_per_ton: float = 0.1
    carbon_rail_per_ton: float = 0.05


# --- 网络生成器：Hub-and-Spoke Network 具体实现 ---
class HaSNetworkGenerator(AbstractNetworkGenerator):
    """
    一个具体的网络生成器，负责创建 hub-and-spoke 类型的随机网络。
    在具体实现时，新建一个 HaSNetwork 对象，使用此生成器的方法来生成网络。
    """
    def __init__(self, config: HaSConfig):
        super().__init__()
        self.config = config

    def generate(self, seed: int = None) -> TransportNetwork:
        """
        实现 generate 方法，完成网络构建.

        Args:
            seed (int, optional): 随机种子，用于保证网络拓扑的可复现性。
        """
        # === 在所有随机操作之前设置种子 ===
        if seed is not None:
            random.seed(seed)
            print(f"网络生成器已设置随机种子: {seed}，本次拓扑将可复现。")
        # ===================================
        
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
        # 生成枢纽节点，有转运成本
        for _ in range(self.config.num_hubs):
            hub_id = node_ids.pop()
            self.network.add_node(Node(str(hub_id), 'hub', transshipment_cost=self.config.cost_hub_transshipment, capacity=random.randint(1500, 2500), population_density=random.uniform(0.5, 1.0)))
        # 生成非枢纽节点，无转运成本
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

        # 1. 创建铁路网络，带运输成本和碳排放成本
        for i in range(len(hubs)):
            for j in range(i + 1, len(hubs)):
                # 确保弧段不存在再添加
                if tuple(sorted((hubs[i].id, hubs[j].id))) not in existing_arcs:
                    self.network.add_arc(
                        hubs[i].id, hubs[j].id, mode='railway', length=random.randint(200, 500), 
                        cost_per_km=self.config.cost_rail_per_km, 
                        carbon_cost_per_ton=self.config.carbon_rail_per_ton
                        )
                    existing_arcs.add(tuple(sorted((hubs[i].id, hubs[j].id))))

        # 2. 创建公路网络，带运输成本和碳排放成本
        all_node_ids = [n.id for n in nodes]
        for i in range(len(all_node_ids)):
            for j in range(i + 1, len(all_node_ids)):
                node1 = self.network._nodes_dict[all_node_ids[i]]
                node2 = self.network._nodes_dict[all_node_ids[j]]
                if node1.type == 'hub' and node2.type == 'hub':
                    continue
                if random.random() < self.config.road_connect_prob:
                    if tuple(sorted((node1.id, node2.id))) not in existing_arcs:
                        self.network.add_arc(
                            node1.id, node2.id, mode='road', length=random.randint(20, 100),
                            cost_per_km=self.config.cost_road_per_km,
                            carbon_cost_per_ton=self.config.carbon_road_per_ton
                            )
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
        """
        创建运输任务。
        如果配置中提供了 predefined_tasks，则使用它们；否则，随机生成。
        """
        # 情况一：用户预先指定了 OD 对
        if self.config.predefined_tasks:
            print(f"使用预定义的 {len(self.config.predefined_tasks)} 个运输任务...")
            for i, (origin_id, dest_id) in enumerate(self.config.predefined_tasks):
                # 从网络中根据 ID 查找节点对象
                origin_node = self.network._nodes_dict.get(origin_id)
                dest_node = self.network._nodes_dict.get(dest_id)
                
                # 进行有效性检查
                if origin_node and dest_node:
                    # 需要检查 OD 是否都是 non-hub
                    if origin_node.type == 'non-hub' and dest_node.type == 'non-hub':
                        self.network.add_task(TransportTask(f"T{i+1}", origin_node, dest_node))
                    else:
                        print(f"警告：预定义任务 ({origin_id}, {dest_id}) 的起点或终点不是 non-hub，已跳过。")
                else:
                    print(f"警告：预定义任务 ({origin_id}, {dest_id}) 的节点 ID 不存在，已跳过。")
        
        # 情况二：没有预定义任务，随机生成运输任务
        else:
            print(f"随机生成 {self.config.num_tasks} 个运输任务...")
        
            # 1. 从网络中获取所有已经创建的 non-hub 节点
            non_hubs = self.network.get_non_hubs()
            
            # 2. 进行有效性检查
            if len(non_hubs) < 2:
                print("警告：非枢纽节点不足 (少于2个)，无法创建任何任务。")
                return
            
            # 3. 确保请求的任务数量不超过可能的最大组合数
            #    从 N 个节点中选2个进行排列
            max_possible_tasks = len(non_hubs) * (len(non_hubs) - 1)
            if self.config.num_tasks > max_possible_tasks:
                print(f"警告：请求的任务数 ({self.config.num_tasks}) 超过了可能的最大组合数 ({max_possible_tasks})。")
                print(f"将只生成 {max_possible_tasks} 个任务。")
            
            num_to_generate = min(self.config.num_tasks, max_possible_tasks)

            # 4. 使用集合来确保随机生成的 OD 对不重复
            generated_pairs = set()
            task_count = 0
            
            # 循环直到生成了足够数量的、不重复的任务
            while task_count < num_to_generate:
                # 从 non-hubs 列表中随机抽取两个不同的节点作为起点和终点
                origin, destination = random.sample(non_hubs, 2)
                
                # 检查这个 OD 对是否已经生成过
                if (origin.id, destination.id) not in generated_pairs:
                    # 如果未生成过，则创建新任务并添加到网络中
                    self.network.add_task(TransportTask(f"T{task_count+1}", origin, destination))
                    generated_pairs.add((origin.id, destination.id))
                    task_count += 1


class HaSNetworkGeneratorDeterministic:
    """
    一个从确定的数据文件（JSON格式）创建网络的生成器。
    它不再使用任何随机化，保证了网络拓扑和参数的完全可复现性。
    """
    def __init__(self, data_file_path: str):
        """
        初始化生成器。

        Args:
            data_file_path (str): 包含网络数据的JSON文件的路径。
        """
        self.network = TransportNetwork()
        self.data_file_path = data_file_path
        self.data: Dict = {}

    def _load_data_from_json(self):
        """[辅助方法] 从JSON文件中加载网络数据。"""
        try:
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"成功从 {self.data_file_path} 加载网络数据。")
        except FileNotFoundError:
            print(f"错误：找不到网络数据文件: {self.data_file_path}")
            raise
        except json.JSONDecodeError:
            print(f"错误：网络数据文件 {self.data_file_path} 格式无效。")
            raise

    def generate(self) -> TransportNetwork:
        """
        实现 generate 方法，从加载的数据中完成网络构建。
        """
        self._load_data_from_json()
        
        print("开始使用确定性数据文件构建网络...")
        
        # 1. 根据数据创建节点
        for node_data in self.data.get("nodes", []):
            self.network.add_node(Node(**node_data))
            
        # 2. 根据数据创建弧段
        for arc_data in self.data.get("arcs", []):
            # 注意：这里的 oneway=True 是因为我们假设JSON中已定义了所有需要的单向弧段
            self.network.add_arc(
                start_node_id=arc_data.pop("start_node_id"), 
                end_node_id=arc_data.pop("end_node_id"),
                oneway=True,
                **arc_data
            )
            
        # 3. 根据数据创建运输任务
        for task_data in self.data.get("tasks", []):
            origin_node = self.network._nodes_dict.get(task_data["origin_node_id"])
            dest_node = self.network._nodes_dict.get(task_data["destination_node_id"])
            if origin_node and dest_node:
                self.network.add_task(TransportTask(
                    task_id=task_data["task_id"],
                    origin_node=origin_node,
                    destination_node=dest_node
                ))
            else:
                print(f"警告：任务 {task_data['task_id']} 的起点或终点不存在，已跳过。")
        
        print("使用确定性数据文件构建网络完成！🎉")
        return self.network