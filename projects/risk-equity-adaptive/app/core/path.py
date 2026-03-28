# --- coding: utf-8 ---
# --- app/core/path.py ---
import logging
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from app.core.fuzzy import FuzzyMath
from app.core.network import Arc, Node, TransportNetwork, TransportTask

if TYPE_CHECKING:
    from app.core.evaluator import Evaluator
    from app.core.network import Arc, Node, TransportNetwork, TransportTask


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
    def from_arc_list(
        task: Optional[TransportTask], arcs: List[Arc]
    ) -> Optional["Path"]:
        """
        [工厂方法] 增加全局无环校验，防止路径在全尺度层面出现“打转”现象。
        """
        if not arcs:
            return Path(task)

        # --- 全局拓扑验证 ---
        node_ids = []
        current_node = arcs[0].start
        node_ids.append(current_node.node_id)

        for arc in arcs:
            # 基础连续性检查
            if arc.start != current_node:
                return None
            current_node = arc.end
            node_ids.append(current_node.node_id)

        # 核心：如果节点总数不等于唯一节点数，说明有环（打转），直接舍弃该路径
        if len(node_ids) != len(set(node_ids)):
            return None

        new_path = Path(task)
        new_path.arcs = arcs
        new_path.nodes = []
        curr = arcs[0].start
        new_path.nodes.append(curr)
        for arc in arcs:
            curr = arc.end
            new_path.nodes.append(curr)

        # --- 步骤 3: 识别转运枢纽 (Road <-> Rail 切换点) ---
        unique_transfer_hubs = {}
        for i in range(len(new_path.arcs) - 1):
            arc1 = new_path.arcs[i]
            arc2 = new_path.arcs[i + 1]

            # 检查 Road -> Rail 或 Rail -> Road 的切换点
            if arc1.mode != arc2.mode:
                if arc1.end.node_type == "hub":
                    unique_transfer_hubs[arc1.end.node_id] = arc1.end

        new_path.transfer_hubs = list(unique_transfer_hubs.values())

        # --- 步骤 4: 识别路径上所有枢纽 ---
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
        self.emergency_times = (
            evaluator.risk_model._precompute_emergency_response_times()
        )
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
            {"name": "low_cost", "road_w": "cost", "railway_w": "cost"},
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
                "weights": tuple(w),  # 存储采样的权重比例
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
        核心组合逻辑：执行 Road -> Rail -> Road 的组合。
        通过跨段节点冲突检查，从源头上过滤“打转”路径。
        """
        found_paths = []
        SUB_LIMIT = 20  # 增加子段搜索上限以提升异质性
        COMBINED_LIMIT = 500  # 增加组合上限，为遗传算法提供更多基因

        # 预排序邻接表：提高 DFS 效率
        sorted_adj = self._get_sorted_adj_for_strategy(strategy)

        # 第一段：Origin -> Hub (公路接驳)
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

            # 记录第一段经过的所有节点
            nodes_o_h1 = {a.start.node_id for a in arcs_o_h1} | {
                arcs_o_h1[-1].end.node_id
            }
            h1 = arcs_o_h1[-1].end

            # 第二段：Hub1 -> Hub2 (铁路干线)
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

                # 冲突检查：铁路段节点不能与公路段 1 重合（排除接驳点 h1）
                nodes_h1_h2 = {a.end.node_id for a in arcs_h1_h2}
                if not nodes_h1_h2.isdisjoint(nodes_o_h1 - {h1.node_id}):
                    continue

                combined_nodes_12 = nodes_o_h1 | nodes_h1_h2

                # 第三段：Hub2 -> Destination (公路配送)
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
                    # 冲突检查：公路段 2 节点不能与前两段重合（排除接驳点 h2）
                    nodes_h2_d = {a.end.node_id for a in arcs_h2_d}
                    if not nodes_h2_d.isdisjoint(combined_nodes_12 - {h2.node_id}):
                        continue

                    # 拼接并进行最后的全局验证
                    path_obj = Path.from_arc_list(
                        task, arcs_o_h1 + arcs_h1_h2 + arcs_h2_d
                    )

                    if path_obj:
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
        # 公路接驳段通常不应超过 6 跳，铁路段不超过 1 跳（2节点）
        MAX_DEPTH = 2 if mode == "road" else 1

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

            for arc in sorted_adj[mode].get(curr.node_id, []):
                next_node = arc.end

                # 策略 A: 无环检测
                if next_node.node_id in visited:
                    continue

                # 策略 B: 枢纽排他性约束 (公路段逻辑)
                # 在公路段搜索中，如果下一个节点是枢纽，但它不是我们预设的目标终点，则跳过
                # 这样可以防止路径在到达 Hub 1 之前，路过其他不相关的 Hub
                if mode == "road":
                    is_target_node = (
                        (next_node.node_id == end_type.node_id)
                        if isinstance(end_type, Node)
                        else (next_node.node_type == end_type)
                    )
                    if next_node.node_type == "hub" and not is_target_node:
                        continue

                # 策略 C: 只有中间节点类型匹配时才继续搜索
                if (
                    next_node.node_type == intermediate_type
                    or (
                        isinstance(end_type, Node)
                        and next_node.node_id == end_type.node_id
                    )
                    or (
                        not isinstance(end_type, Node)
                        and next_node.node_type == end_type
                    )
                ):
                    path_arcs.append(arc)
                    dfs(next_node, path_arcs, visited, depth + 1)
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
        SCALE_COST, SCALE_RISK, SCALE_RESP = 0.1, 1e-1, 5.0  # 量纲对齐

        # 1. 风险：E = p_arc * c_arc + p_node * c_node
        v_risk = (
            arc.accident_prob_per_km
            * arc.length
            * self.evaluator.risk_model.get_consequence(arc)
        ) + (target.accident_prob * self.evaluator.risk_model.get_consequence(target))

        # 2. 成本
        mode = arc.mode
        c_m = self.evaluator.unit_transport_cost.get(mode, 0.0)
        v_cost = c_m * arc.length

        # 3. 响应时间
        arc_resp = self._get_actual_response_weight(arc)
        node_fuzzy = self.evaluator.risk_model._response_time_cache.get(
            target.node_id, (12, 12, 12)
        )
        v_resp = arc_resp + FuzzyMath.triangular_expected_value(*node_fuzzy)

        if weight_key == "risk":
            return v_risk
        if weight_key == "cost":
            return v_cost
        if weight_key == "response":
            return v_resp

        w_cost, w_risk, w_resp = strategy["weights"]
        return (
            w_cost * (v_cost * SCALE_COST)
            + w_risk * (v_risk * SCALE_RISK)
            + w_resp * (v_resp * SCALE_RESP)
        )

    def _get_actual_response_weight(self, arc) -> float:
        fuzzy_t_e = self.evaluator.risk_model._response_time_cache.get(
            (arc.start.node_id, arc.end.node_id), (12.0, 12.0, 12.0)
        )

        return FuzzyMath.triangular_expected_value(*fuzzy_t_e)

    def _get_ablation_paths(
        self, k_val: int, strategy_level: int
    ) -> Dict[str, List["Path"]]:
        """
        [消融实验]:
        Level 1: 仅 Lowest Cost (基础 NSGA-II)
        Level 2: Lowest Cost + Lowest Risk
        Level 3: Lowest Cost + Lowest Risk + Fastest Response
        """
        level_names = {
            1: "Level 1: Lowest Cost",
            2: "Level 2: Lowest Cost + Risk",
            3: "Level 3: Lowest Cost + Risk + Response",
        }
        logging.info(f"🧬 生成消融路径池: {level_names.get(strategy_level, 'Unknown')}")

        candidate_paths_map: Dict[str, List["Path"]] = {}

        # 1. 定义三套纯物理策略
        base_strategies = [
            {"name": "low_cost", "road_w": "cost", "railway_w": "cost"},  # 策略 (1)
            {"name": "low_risk", "road_w": "risk", "railway_w": "risk"},  # 策略 (2)
            {
                "name": "fast_response",
                "road_w": "response",
                "railway_w": "response",
            },  # 策略 (3)
        ]

        # 2. 根据 level 决定策略子集
        # Level 1 取 base[:1], Level 2 取 base[:2], Level 3 取 base[:3]
        selected_strategies = base_strategies[: strategy_level + 1]

        for task in self.network.tasks:
            combined_paths: List["Path"] = []
            seen_fingerprints = set()

            for strategy in selected_strategies:
                # 调用内部 DFS 搜索对应策略下的 K 条路径
                paths = self._find_best_path_by_strategy(task, strategy)
                for p in paths[:k_val]:
                    fp = tuple(arc.id for arc in p.arcs)
                    if fp not in seen_fingerprints:
                        combined_paths.append(p)
                        seen_fingerprints.add(fp)

            candidate_paths_map[task.task_id] = combined_paths

        return candidate_paths_map
