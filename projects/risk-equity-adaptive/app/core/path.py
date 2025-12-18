# --- coding: utf-8 ---
# --- app/core/path.py ---
import logging
import random
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, TYPE_CHECKING
from app.core.network import Node, Arc, TransportTask, TransportNetwork
from app.core.fuzzy import FuzzyMath

if TYPE_CHECKING:
    from app.core.evaluator import Evaluator
    from app.core.network import TransportNetwork, Node, Arc, TransportTask


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
    路径搜索服务类。
    采用预排序、子段剪枝及哈希去重技术，高效生成 road-rail-road 结构的异质候选路径池。
    """

    def __init__(self, network: "TransportNetwork", evaluator: "Evaluator"):
        self.network = network
        self.evaluator = evaluator
        # 从配置中读取种子，确保实验可重复性
        self.seed = self.evaluator.experiment_config.get("precompute_seed", 19)
        # 预处理邻接表: self.adj[mode][node_id] = [Arc, ...]
        self.adj: Dict[str, Dict[str, List["Arc"]]] = {"road": {}, "railway": {}}
        self._build_adjacency_lists()

    def _build_adjacency_lists(self):
        """构建原始邻接表以便后续快速搜索。"""
        for node in self.network.nodes:
            self.adj["road"][node.node_id] = []
            self.adj["railway"][node.node_id] = []
        for arc in self.network.arcs:
            self.adj[arc.mode][arc.start.node_id].append(arc)

    def find_all_candidate_paths(self) -> Dict[str, List["Path"]]:
        """[主方法] 遍历所有任务，生成完整候选路径库。"""
        logging.info("开始搜索 candidate paths（预排序+哈希去重）...")
        candidate_paths_map: Dict[str, List["Path"]] = {}
        for task in self.network.tasks:
            paths = self.find_paths_for_task(task)
            candidate_paths_map[task.task_id] = paths

        total_paths = sum(len(p) for p in candidate_paths_map.values())
        logging.info(
            f"搜索完成：共为 {len(self.network.tasks)} 个任务找到 {total_paths} 条异质路径。"
        )
        return candidate_paths_map

    def find_paths_for_task(self, task: "TransportTask") -> List["Path"]:
        """
        为单个任务生成候选路径。使用哈希指纹去重，大幅提升效率。
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        all_strategy_paths: List["Path"] = []
        # 使用哈希指纹集合去重，避免高开销的字符串对比
        seen_path_fingerprints: Set[Tuple[str, ...]] = set()

        # 1. 基础物理锚点策略
        base_strategies = [
            {"name": "shortest", "road_w": "length", "railway_w": "length"},
            {"name": "low_risk", "road_w": "risk", "railway_w": "risk"},
            {"name": "fast_response", "road_w": "response", "railway_w": "response"},
        ]

        # 2. Dirichlet 混合权重采样
        hybrid_count = 20  # 针对小网络，20 组采样已能提供足够的中间权衡解
        weights_samples = np.random.dirichlet((1, 1, 1), size=hybrid_count)

        strategies = base_strategies + [
            {
                "name": f"hybrid_{i}",
                "road_w": "hybrid",
                "railway_w": "hybrid",
                "weights": tuple(w),
            }
            for i, w in enumerate(weights_samples)
        ]

        # 单策略最终保留上限
        MAX_PATHS_PER_STRATEGY = 300

        for strategy in strategies:
            # 搜索当前策略下的路径
            paths = self._find_best_path_by_strategy(task, strategy)
            for p in paths[:MAX_PATHS_PER_STRATEGY]:
                # 提取弧段 ID 元组作为哈希指纹
                fingerprint = tuple(arc.id for arc in p.arcs)
                if fingerprint not in seen_path_fingerprints:
                    all_strategy_paths.append(p)
                    seen_path_fingerprints.add(fingerprint)

        return all_strategy_paths

    def _find_best_path_by_strategy(
        self, task: "TransportTask", strategy: Dict
    ) -> List["Path"]:
        """
        核心组合逻辑：执行 Road -> Rail -> Road 的笛卡尔积。
        严格控制子段数量与组合上限，防止组合爆炸。
        """
        found_paths = []
        SUB_LIMIT = 20  # 每个子段仅返回前 20 条最优路径
        COMBINED_LIMIT = 500  # 单个策略下的组合路径总数上限

        # 预排序邻接表：在进入 DFS 前按当前权重排好序，避免递归内重复计算
        sorted_adj = self._get_sorted_adj_for_strategy(strategy)

        # 第一段：Origin -> Hub (Road)
        o_h1_lists = self._find_sub_paths_optimized(
            task.origin,
            "road",
            "hub",
            "non-hub",
            strategy,
            "road_w",
            SUB_LIMIT,
            sorted_adj,
        )

        for arcs_o_h1 in o_h1_lists:
            if len(found_paths) >= COMBINED_LIMIT:
                break
            h1 = arcs_o_h1[-1].end

            # 第二段：Hub1 -> Hub2 (Railway)
            h1_h2_lists = self._find_sub_paths_optimized(
                h1,
                "railway",
                "hub",
                "hub",
                strategy,
                "railway_w",
                SUB_LIMIT,
                sorted_adj,
            )

            for arcs_h1_h2 in h1_h2_lists:
                if len(found_paths) >= COMBINED_LIMIT:
                    break
                h2 = arcs_h1_h2[-1].end
                if h1.node_id == h2.node_id:
                    continue

                # 第三段：Hub2 -> Destination (Road)
                h2_d_lists = self._find_sub_paths_optimized(
                    h2,
                    "road",
                    task.destination,
                    "non-hub",
                    strategy,
                    "road_w",
                    SUB_LIMIT,
                    sorted_adj,
                )

                for arcs_h2_d in h2_d_lists:
                    path_obj = Path.from_arc_list(
                        task, arcs_o_h1 + arcs_h1_h2 + arcs_h2_d
                    )
                    found_paths.append(path_obj)
                    if len(found_paths) >= COMBINED_LIMIT:
                        break

        return found_paths

    def _get_sorted_adj_for_strategy(
        self, strategy: Dict
    ) -> Dict[str, Dict[str, List["Arc"]]]:
        """预排序邻接表，将 O(N log N) 的排序开销从 DFS 内部移出。"""
        sorted_adj = {"road": {}, "railway": {}}
        for mode in ["road", "railway"]:
            w_key = strategy[f"{mode}_w"]
            for node_id, neighbors in self.adj[mode].items():
                sorted_adj[mode][node_id] = sorted(
                    neighbors,
                    key=lambda x: self._calculate_arc_weight(x, w_key, strategy),
                )
        return sorted_adj

    def _find_sub_paths_optimized(
        self,
        start,
        mode,
        end_type,
        intermediate_type,
        strategy,
        w_key_name,
        limit,
        sorted_adj,
    ) -> List[List["Arc"]]:
        """带有剪枝策略的 DFS 搜索。"""
        all_results = []
        weight_key = strategy[w_key_name]
        # 针对小网络优化深度限制
        MAX_DEPTH = 6 if mode == "road" else 10

        def dfs(curr, path_arcs, visited, depth):
            # 剪枝：超过深度或已搜到足够多的子段
            if depth > MAX_DEPTH or len(all_results) > limit * 8:
                return

            visited.add(curr.node_id)
            is_end = (
                (curr.node_id == end_type.node_id)
                if isinstance(end_type, Node)
                else (curr.node_type == end_type)
            )

            if is_end and path_arcs:
                all_results.append(list(path_arcs))

            # 直接使用预排好序的邻居
            for arc in sorted_adj[mode].get(curr.node_id, []):
                if arc.end.node_id not in visited:
                    is_next_end = (
                        (arc.end.node_id == end_type.node_id)
                        if isinstance(end_type, Node)
                        else (arc.end.node_type == end_type)
                    )
                    if is_next_end or arc.end.node_type == intermediate_type:
                        path_arcs.append(arc)
                        dfs(arc.end, path_arcs, visited, depth + 1)
                        path_arcs.pop()
            visited.remove(curr.node_id)

        dfs(start, [], set(), 0)

        # 子段排序
        all_results.sort(
            key=lambda x: sum(
                self._calculate_arc_weight(a, weight_key, strategy) for a in x
            )
        )

        # 对于混合策略引入扰动采样，增加帕累托前沿的覆盖度
        if weight_key == "hybrid" and len(all_results) > limit:
            pool = all_results[: limit * 3]
            return random.sample(pool, min(len(pool), limit))

        return all_results[:limit]

    def _calculate_arc_weight(self, arc, weight_key, strategy) -> float:
        """核心物理权重计算函数。"""
        target = arc.end
        SCALE_LEN, SCALE_RISK, SCALE_RESP = 0.1, 1e-1, 5.0  # 量纲对齐

        v_len = arc.length
        # 风险：E = p_arc * c_arc + p_node * c_node
        v_risk = (
            arc.accident_prob_per_km
            * arc.length
            * self.evaluator.risk_model.get_consequence(arc)
        ) + (target.accident_prob * self.evaluator.risk_model.get_consequence(target))

        arc_resp = self._get_actual_response_weight(arc)
        node_fuzzy = self.evaluator.risk_model._response_time_cache.get(
            target.node_id, (12, 12, 12)
        )

        v_resp = arc_resp + FuzzyMath.triangular_expected_value(*node_fuzzy)

        if weight_key == "length":
            return v_len
        if weight_key == "risk":
            return v_risk
        if weight_key == "response":
            return v_resp
        if weight_key == "hybrid":
            w_len, w_risk, w_resp = strategy["weights"]
            return (
                w_len * (v_len * SCALE_LEN)
                + w_risk * (v_risk * SCALE_RISK)
                + w_resp * (v_resp * SCALE_RESP)
            )
        return v_len

    def _get_actual_response_weight(self, arc) -> float:
        fuzzy_t_e = self.evaluator.risk_model._response_time_cache.get(
            (arc.start.node_id, arc.end.node_id), (12.0, 12.0, 12.0)
        )

        return FuzzyMath.triangular_expected_value(*fuzzy_t_e)
