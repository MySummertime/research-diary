# --- coding: utf-8 ---
# --- app/core/path.py ---
import logging
from typing import List, Dict, Set, Optional
from app.core.network import Node, Arc, TransportTask, TransportNetwork


class Path:
    """
    存储一条从 O 到 D 的具体路径信息，以及关键节点（转运点）。
    """

    def __init__(self, task: Optional[TransportTask] = None):
        """
        初始化路径
        Args:
            task(Optional[TransportTask]): 该路径对应的运输任务 (可选)
        """
        self.task: Optional[TransportTask] = task
        self.arcs: list[Arc] = []  # 路径上的弧段 (有序)
        self.nodes: list[Node] = []  # 路径上的节点 (有序)

        # 关键节点缓存
        # 1. 只用于转运 (road/rail 切换点)
        self.transfer_hubs: list[Node] = []
        # 2. 用于日志记录 (路径上的所有hub)
        self.all_hubs_on_path: list[Node] = []

    @staticmethod
    def from_arc_list(task: Optional[TransportTask], arcs: List[Arc]) -> "Path":
        """
        [工厂方法]从一个有序的弧段列表中构建一个完整的、经过验证的Path对象。
        """
        new_path = Path(task)
        if not arcs:
            return new_path  # 返回一个空路径

        new_path.arcs = arcs

        # 1. 完整重建节点列表
        current_node = arcs[0].start
        new_path.nodes.append(current_node)

        for arc in arcs:
            if arc.start != current_node:
                task_id = task.task_id if task else "N/A"
                raise ValueError(
                    f"[PathFactory] 任务 {task_id} 路径构建失败: 弧段 {arc.start.node_id}->{arc.end.node_id} 与前序节点 {current_node.node_id} 不匹配"
                )
            current_node = arc.end
            new_path.nodes.append(current_node)

        # 2. 智能地查找 *转运* 枢纽 (H1, H2)
        # Road -> Rail (H1) OR Rail -> Road (H2)
        unique_transfer_hubs = {}  # 使用字典确保唯一性

        for i in range(len(new_path.arcs) - 1):
            arc1 = new_path.arcs[i]
            arc2 = new_path.arcs[i + 1]

            # 检查 H1 (road -> rail)
            if arc1.mode == "road" and arc2.mode == "railway":
                if arc1.end.node_type == "hub":
                    unique_transfer_hubs[arc1.end.node_id] = arc1.end

            # 检查 H2 (rail -> road)
            elif arc1.mode == "railway" and arc2.mode == "road":
                if arc1.end.node_type == "hub":
                    unique_transfer_hubs[arc1.end.node_id] = arc1.end

        new_path.transfer_hubs = list(unique_transfer_hubs.values())

        # 3. 查找路径上 *所有* 途经的枢纽 (用于日志)
        unique_all_hubs = {}
        for node in new_path.nodes:
            if node.node_type == "hub":
                unique_all_hubs[node.node_id] = node
        new_path.all_hubs_on_path = list(unique_all_hubs.values())

        return new_path

    def __repr__(self):
        task_id = self.task.task_id if self.task else "None"
        if not self.nodes:
            return f"Path(Task={task_id}, Path=Empty)"
        path_str = " -> ".join([node.node_id for node in self.nodes])
        return f"Path(Task={task_id}, Path={path_str})"


class PathFinder:
    """
    路径搜索服务。
    为运输网络中的任务搜索符合 "road-rail-road" 结构的所有替代路径。
    """

    def __init__(self, network: TransportNetwork):
        self.network = network
        # 预处理邻接表: adj[mode][node_id] = [Arc, ...]
        self.adj: Dict[str, Dict[str, List[Arc]]] = {"road": {}, "railway": {}}
        self._build_adjacency_lists()

    def _build_adjacency_lists(self):
        """
        [辅助方法] 构建邻接表以便快速搜索。
        """
        for node in self.network.nodes:
            self.adj["road"][node.node_id] = []
            self.adj["railway"][node.node_id] = []

        for arc in self.network.arcs:
            self.adj[arc.mode][arc.start.node_id].append(arc)

    def find_all_candidate_paths(self) -> Dict[str, List[Path]]:
        """
        [主方法] 遍历网络中的所有任务，并为它们找到所有候选路径。
        """
        logging.info("开始搜索所有任务的候选路径...")
        candidate_paths_map: Dict[str, List[Path]] = {}
        for task in self.network.tasks:
            paths = self.find_paths_for_task(task)
            candidate_paths_map[task.task_id] = paths

        # 正确统计所有路径的总数
        total_paths_count = sum(len(paths) for paths in candidate_paths_map.values())

        logging.info(
            f"路径搜索完成。共为 {len(self.network.tasks)} 个任务找到了 {total_paths_count} 条候选路径。"
        )
        return candidate_paths_map

    def find_paths_for_task(self, task: TransportTask) -> List[Path]:
        """
        [主方法] 为单个任务搜索所有 "road-rail-road" 路径。
        """
        all_valid_paths: List[Path] = []

        # 步骤 1: 查找 O -> H1 (公路)
        arc_lists_o_h1 = self._find_sub_paths(
            start_node=task.origin,
            mode="road",
            end_type="hub",
            allowed_intermediate_type="non-hub",
        )

        for arcs_o_h1 in arc_lists_o_h1:
            h1_node = arcs_o_h1[-1].end  # 第一个枢纽

            # 步骤 2: 查找 H1 -> H2 (铁路)
            arc_lists_h1_h2 = self._find_sub_paths(
                start_node=h1_node,
                mode="railway",
                end_type="hub",
                allowed_intermediate_type="hub",
            )

            for arcs_h1_h2 in arc_lists_h1_h2:
                h2_node = arcs_h1_h2[-1].end  # 第二个枢纽

                # 步骤 3: 查找 H2 -> D (公路)
                arc_lists_h2_d = self._find_sub_paths(
                    start_node=h2_node,
                    mode="road",
                    end_type=task.destination,
                    allowed_intermediate_type="non-hub",
                )

                for arcs_h2_d in arc_lists_h2_d:
                    combined_arc_list = arcs_o_h1 + arcs_h1_h2 + arcs_h2_d
                    combined_path = Path.from_arc_list(task, combined_arc_list)
                    all_valid_paths.append(combined_path)

        return all_valid_paths

    def _find_sub_paths(
        self,
        start_node: Node,
        mode: str,
        end_type: str | Node,
        allowed_intermediate_type: str,
    ) -> List[List[Arc]]:
        """
        [辅助函数] 使用DFS查找所有符合特定结构的简单路径。
        """

        all_found_arc_lists: List[List[Arc]] = []

        def dfs_recursive(
            current_node: Node, current_path_arcs: List[Arc], visited_nodes: Set[str]
        ):
            visited_nodes.add(current_node.node_id)

            is_end_node = False
            if isinstance(end_type, Node):
                if current_node.node_id == end_type.node_id:
                    is_end_node = True
            elif isinstance(end_type, str):
                if current_node.node_type == end_type:
                    is_end_node = True

            if is_end_node:
                if current_path_arcs:
                    all_found_arc_lists.append(list(current_path_arcs))

                if allowed_intermediate_type != "hub":
                    visited_nodes.remove(current_node.node_id)
                    return

            for arc in self.adj[mode][current_node.node_id]:
                neighbor_node = arc.end

                if neighbor_node.node_id not in visited_nodes:
                    is_neighbor_end_node = False
                    if isinstance(end_type, Node):
                        if neighbor_node.node_id == end_type.node_id:
                            is_neighbor_end_node = True
                    elif isinstance(end_type, str):
                        if neighbor_node.node_type == end_type:
                            is_neighbor_end_node = True

                    if (
                        is_neighbor_end_node
                        or neighbor_node.node_type == allowed_intermediate_type
                    ):
                        current_path_arcs.append(arc)
                        dfs_recursive(neighbor_node, current_path_arcs, visited_nodes)
                        current_path_arcs.pop()

            visited_nodes.remove(current_node.node_id)

        dfs_recursive(start_node, [], set())

        return all_found_arc_lists
