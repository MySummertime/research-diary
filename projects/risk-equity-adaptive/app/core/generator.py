# --- coding: utf-8 ---
# --- app/core/generator.py ---
import json
from abc import ABC, abstractmethod
from typing import List
from .network import Node, TransportTask, TransportNetwork

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


# --- 2. 网络生成器 ---
class JSONNetworkGenerator(AbstractNetworkGenerator):
    """
    一个从确定的数据文件 (JSON格式) 创建网络的生成器。
    它从三个独立的文件中读取节点、弧段和任务。
    """
    def __init__(self, 
                 nodes_file_path: str,
                 arcs_file_path: str,
                 tasks_file_path: str):
        """
        初始化生成器。

        Args:
            nodes_file_path (str): 包含 "nodes" 列表的JSON文件路径。
            arcs_file_path (str): 包含 "arcs" 列表的JSON文件路径。
            tasks_file_path (str): 包含 "tasks" 列表的JSON文件路径。
        """
        # 调用父类的 __init__
        super().__init__()
        
        self.nodes_file_path = nodes_file_path
        self.arcs_file_path = arcs_file_path
        self.tasks_file_path = tasks_file_path

    def _load_json_file(self, file_path: str) -> List:
        """
        [辅助方法] 从一个JSON文件中加载数据列表。
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            print(f"错误：找不到网络数据文件: {file_path}")
            raise
        except json.JSONDecodeError:
            print(f"错误：网络数据文件 {file_path} 格式无效。")
            raise

    def generate(self) -> TransportNetwork:
        """
        实现 generate 方法，从加载的数据中完成网络构建。
        """
        print("开始使用JSON数据文件构建网络...")
        
        # --- 1. 根据数据创建节点 ---
        node_data_list = self._load_json_file(self.nodes_file_path)
        for node_data in node_data_list:
            # 确保ID是字符串类型
            node_data['node_id'] = str(node_data['node_id'])
            
            # 使用 **kwargs 解包所有参数
            # 只要JSON的键与Node的参数名匹配，就能自动工作
            self.network.add_node(Node(**node_data))
            
        print(f"成功加载 {len(self.network.nodes)} 个节点。")

        # --- 2. 根据数据创建弧段 ---
        arc_data_list = self._load_json_file(self.arcs_file_path)
        for arc_data in arc_data_list:
            # 确保ID是字符串
            start_node_id = str(arc_data.pop("start_node_id"))
            end_node_id = str(arc_data.pop("end_node_id"))

            # **arc_data 会传入 mode, length, fuzzy_transport_time 等
            self.network.add_arc(
                start_node_id=start_node_id,
                end_node_id=end_node_id,
                **arc_data
            )
        
        print(f"成功加载 {len(self.network.arcs)} 条弧段。")

        # --- 3. 根据数据创建运输任务 ---
        task_data_list = self._load_json_file(self.tasks_file_path)
        for task_data in task_data_list:
            # 确保ID是字符串
            origin_node_id = str(task_data["origin_node_id"])
            dest_node_id = str(task_data["destination_node_id"])

            origin_node = self.network._nodes_dict.get(origin_node_id)
            dest_node = self.network._nodes_dict.get(dest_node_id)
            
            if origin_node and dest_node:
                self.network.add_task(TransportTask(
                    task_id=str(task_data["task_id"]),
                    origin_node=origin_node,
                    destination_node=dest_node,
                    demand=task_data.get("demand", 0.0) # demand是可选的
                ))
            else:
                print(f"警告：任务 {task_data['task_id']} 的起点或终点不存在，已跳过。")
        
        print(f"成功加载 {len(self.network.tasks)} 个任务。")
        print("使用确定性数据文件构建网络完成！🎉")
        
        return self.network