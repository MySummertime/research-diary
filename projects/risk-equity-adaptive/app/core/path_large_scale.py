# --- coding: utf-8 ---
# --- app/core/path_large_scale.py ---
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

from app.core.path import Path, PathFinder

# 全局变量，用于子进程内的资源句柄（避免重复序列化大数据对象）
_worker_finder = None


def _init_worker(network, evaluator):
    """[子进程初始化函数]: 确保每个进程都有自己的 Finder 实例和局部缓存"""
    global _worker_finder
    _worker_finder = LargeScalePathFinder(network, evaluator)


def _worker_task_wrapper(task):
    """[子进程任务包装器]: 调用全局句柄执行任务"""
    return _worker_finder.find_paths_for_task(task)


class LargeScalePathFinder(PathFinder):
    """
    [生产级加速版]: 解决了多进程缓存共享与序列化问题的最终方案。
    1. 进程池初始化: 采用 initializer 模式，显著降低 IPC 通讯开销。
    2. 深度/宽度控制: 自动适配 40-120 节点规模。
    3. 地理冗余保护: 引入动态阈值，防止过度剪枝导致丢解。
    """

    def __init__(self, network, evaluator):
        super().__init__(network, evaluator)

        # 进程局部缓存 (子路径记忆化)
        self._sub_path_memo: Dict[Tuple, List[List]] = {}

        # 环境感知系数
        n_nodes = len(self.network.nodes)
        if n_nodes <= 20:
            self.dist_relax_factor = 2.5  # 小路网：大幅放宽，允许“绕路”寻找极低风险路径
            self.adaptive_max_depth = {"road": 12, "railway": 18}
        elif n_nodes <= 50:
            self.dist_relax_factor = 1.8  # 中等路网
            self.adaptive_max_depth = {"road": 8, "railway": 12}
        else:
            self.dist_relax_factor = 1.3  # 大规模路网：严格限制，防止搜索爆炸
            self.adaptive_max_depth = {"road": 5, "railway": 8}

    def find_all_candidate_paths(self) -> Dict[str, List["Path"]]:
        """
        [主入口]
        """
        tasks = self.network.tasks
        logging.info(
            f"🚀 大规模并行引擎启动 | 任务数: {len(tasks)} | 进程数: {os.cpu_count()}"
        )

        # 使用 initializer 将 network 和 evaluator 预先加载到子进程内存中
        # 这样不用在每次 map 时都传输巨大的网络对象
        with ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=_init_worker,
            initargs=(self.network, self.evaluator),
        ) as executor:
            results = list(executor.map(_worker_task_wrapper, tasks))

        candidate_paths_map = {tasks[i].task_id: results[i] for i in range(len(tasks))}

        total_paths = sum(len(p) for p in candidate_paths_map.values())
        logging.info(f"⚡ 并行计算完成: 共生成 {total_paths} 条路径。")
        return candidate_paths_map

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
    ):
        """
        包含地理坐标判断与记忆化加速。
        """
        weight_key = strategy[w_key_name]
        target_id = end_type.node_id if hasattr(end_type, "node_id") else str(end_type)

        # 1. 进程内记忆化检查 (针对单任务内的重复 Hub 访问)
        memo_key = (start.node_id, target_id, mode, weight_key)
        if memo_key in self._sub_path_memo:
            return self._sub_path_memo[memo_key]

        all_results = []
        # 使用动态计算出的深度
        max_depth = self.adaptive_max_depth.get(mode, 8)

        # 2. 预计算坐标锚点
        target_coords = (end_type.x, end_type.y) if hasattr(end_type, "x") else None

        # 3. 动态距离阈值 (对于 40 节点路网，放宽至 1.5 倍以保证搜索质量)
        dist_threshold = 999999
        if target_coords:
            direct_dist = math.sqrt(
                (start.x - target_coords[0]) ** 2 + (start.y - target_coords[1]) ** 2
            )
            dist_threshold = direct_dist * self.dist_relax_factor

        def dfs(curr, path_arcs, visited, depth):
            # 剪枝: 深度、结果数、地理围栏
            if depth > max_depth or len(all_results) > limit * 3:
                return

            if target_coords:
                curr_dist = math.sqrt(
                    (curr.x - target_coords[0]) ** 2 + (curr.y - target_coords[1]) ** 2
                )
                if curr_dist > dist_threshold:
                    return

            visited.add(curr.node_id)
            is_end = (
                (curr.node_id == end_type.node_id)
                if hasattr(end_type, "node_id")
                else (curr.node_type == end_type)
            )

            if is_end and path_arcs:
                all_results.append(list(path_arcs))

            for arc in sorted_adj[mode].get(curr.node_id, []):
                if arc.end.node_id not in visited:
                    is_next_target = (
                        (arc.end.node_id == end_type.node_id)
                        if hasattr(end_type, "node_id")
                        else (arc.end.node_type == end_type)
                    )
                    if is_next_target or arc.end.node_type == intermediate_type:
                        path_arcs.append(arc)
                        dfs(arc.end, path_arcs, visited, depth + 1)
                        path_arcs.pop()

            visited.remove(curr.node_id)

        dfs(start, [], set(), 0)

        # 结果排序
        all_results.sort(
            key=lambda x: sum(
                self._calculate_arc_weight(a, weight_key, strategy) for a in x
            )
        )
        res = all_results[:limit]

        self._sub_path_memo[memo_key] = res
        return res
