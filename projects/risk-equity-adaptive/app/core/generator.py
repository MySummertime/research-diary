# --- coding: utf-8 ---
# --- app/core/generator.py ---
import json
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.core.network import Node, TransportTask, TransportNetwork


class AbstractNetworkGenerator(ABC):
    """
    网络生成器的抽象基类 (蓝图)。
    """

    def __init__(self):
        self.network = TransportNetwork()

    @abstractmethod
    def generate(self) -> TransportNetwork:
        pass


class JSONNetworkGenerator(AbstractNetworkGenerator):
    """
    从 JSON 文件加载网络数据。
    """

    def __init__(self, nodes_file_path: str, arcs_file_path: str, tasks_file_path: str):
        super().__init__()
        self.nodes_file_path = nodes_file_path
        self.arcs_file_path = arcs_file_path
        self.tasks_file_path = tasks_file_path

    def _load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """[辅助] 安全加载 JSON 文件"""
        if not os.path.exists(file_path):
            logging.error(f"文件未找到: {file_path}")
            raise FileNotFoundError(f"Cannot find file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except json.JSONDecodeError as e:
            logging.error(f"JSON 格式错误 ({file_path}): {e}")
            raise

    def generate(self) -> TransportNetwork:
        """实现网络构建流程"""
        logging.info("正在从 JSON 文件构建网络...")

        # 1. Nodes
        node_data_list = self._load_json_file(self.nodes_file_path)
        for data in node_data_list:
            # 转换 ID 为字符串以统一格式
            data["node_id"] = str(data["node_id"])
            # 这里的 **data 会自动映射到 Node.__init__ 的参数
            self.network.add_node(Node(**data))

        logging.info(f"  - 已加载节点: {len(self.network.nodes)}")

        # 2. Arcs
        arc_data_list = self._load_json_file(self.arcs_file_path)
        for data in arc_data_list:
            start_id = str(data.pop("start_node_id"))
            end_id = str(data.pop("end_node_id"))

            try:
                self.network.add_arc(start_id, end_id, **data)
            except ValueError as e:
                logging.warning(f"    跳过无效弧段 ({start_id}->{end_id}): {e}")

        logging.info(f"  - 已加载弧段: {len(self.network.arcs)}")

        # 3. Tasks
        task_data_list = self._load_json_file(self.tasks_file_path)
        loaded_tasks = 0
        for data in task_data_list:
            t_id = str(data["task_id"])
            o_id = str(data["origin_node_id"])
            d_id = str(data["destination_node_id"])
            demand = data.get("demand", 0.0)

            origin = self.network._nodes_dict.get(o_id)
            dest = self.network._nodes_dict.get(d_id)

            if origin and dest:
                try:
                    task = TransportTask(t_id, origin, dest, demand)
                    self.network.add_task(task)
                    loaded_tasks += 1
                except ValueError as e:
                    logging.warning(f"    任务 {t_id} 无效: {e}")
            else:
                logging.warning(
                    f"    任务 {t_id} 的 OD 节点不存在 ({o_id}->{d_id})，已跳过。"
                )

        logging.info(f"  - 已加载任务: {loaded_tasks}")
        logging.info("网络构建完成。")

        return self.network
