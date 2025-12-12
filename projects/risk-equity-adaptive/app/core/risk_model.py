# --- coding: utf-8 ---
# --- app/core/risk_model.py ---
"""
[物理层] 动态风险评估模型
对应论文 Section 3.3: Risk modeling and assessment
负责计算物理世界的变量：应急响应时间、事故扩散范围、动态后果。
"""

import math
import logging
import networkx as nx
from typing import Dict, Tuple, Any
from app.core.network import TransportNetwork, Node, Arc


class DynamicRiskModel:
    """
    封装了所有与物理风险相关的计算逻辑。
    """

    def __init__(self, network: TransportNetwork, risk_config: Dict[str, Any]):
        self.network = network
        self.config = risk_config

        # 读取最大救援时间上限 (默认 12 小时)
        # 物理意义：如果超过这个时间还没到，事故后果达到饱和，不会无限增加。
        self.max_response_time = self.config.get("max_response_time", 12.0)

        # 缓存计算出的应急时间，避免重复计算
        # Key: node_id (str) 或 edge_tuple (u, v)
        # Value: time (hours)
        self._response_time_cache: Dict[str | Tuple[str, str], float] = {}

        # 初始化时立即进行预计算
        self._precompute_emergency_response_times()

    def get_consequence(self, entity: Arc | Node) -> float:
        """
        计算给定实体（弧段或节点）的动态事故后果。
        Formula: C(t) = (A_base + A_spread(t)) * rho
        """
        lambda_m = self.config.get("accident_lambda", 100.0)
        gamma = self.config.get("accident_gamma", 0.1)
        q = self.config.get("accident_q", 5.0)

        # 1. 获取响应时间
        if isinstance(entity, Arc):
            t_e = self._response_time_cache.get(
                (entity.start.node_id, entity.end.node_id), float("inf")
            )
        else:
            t_e = self._response_time_cache.get(entity.node_id, float("inf"))

        # Double check: 确保 t_e 不会因为缓存未命中而变成 inf
        if t_e == float("inf"):
            t_e = self.max_response_time

        # 2. 物理扩散模型: A_spread = γ * q * t
        # q: kg/s, need to be transformed to kg/h for alignment
        a_spread = gamma * q * (t_e * 3600.0)

        # 3. 计算总后果
        if isinstance(entity, Arc):
            d_ij = entity.length
            rho_ij = entity.population_density
            # 矩形面积 + 两端半圆
            a_base = (2.0 * lambda_m * d_ij) + (math.pi * lambda_m**2)
            consequence = (a_base + a_spread) * rho_ij
        else:
            rho_k = entity.population_density
            # 圆形面积
            a_base = math.pi * lambda_m**2
            consequence = (a_base + a_spread) * rho_k

        return consequence

    def _precompute_emergency_response_times(self):
        """
        [Key Point Method] 预计算全网应急响应时间。
        使用 关键点法 积分近似公式:
        t_ij = t_0 + 1/4 * (t_i + 2*t_mid + t_j)
        """
        logging.info("[RiskModel] 正在预计算动态应急响应时间 (Key Point Method)...")

        speed_v = self.config.get("emergency_vehicle_speed", 45.0)
        setup_time = self.config.get("emergency_setup_time", 0.05)  # t0

        # 1. 构建路网图 (仅公路)
        road_graph = nx.DiGraph()
        for arc in self.network.arcs:
            if arc.mode == "road":
                t_travel = float("inf") if speed_v <= 0 else (arc.length / speed_v)
                road_graph.add_edge(arc.start.node_id, arc.end.node_id, weight=t_travel)

        # 2. 识别应急中心
        centers = self.network.get_emergency_centers()
        center_ids = [n.node_id for n in centers]

        # 3. 计算节点响应时间 t_k^travel (Multi-source Dijkstra)
        node_travel_times = {}
        if not center_ids:
            logging.warning("[RiskModel] 警告: 未找到应急中心，所有响应时间将为 inf.")
        else:
            try:
                # 计算所有节点到最近应急中心的最短路时间
                node_travel_times = nx.multi_source_dijkstra_path_length(
                    road_graph.reverse(), sources=center_ids, weight="weight"
                )
            except Exception as e:
                logging.warning(f"[RiskModel] Dijkstra 计算失败: {e}")

        # 4. 存储节点最终响应时间 t_k = t0 + t_k^travel
        for node in self.network.nodes:
            t_travel = node_travel_times.get(node.node_id, float("inf"))

            # 节点时间兜底
            if t_travel == float("inf"):
                final_time = self.max_response_time
            else:
                final_time = t_travel + setup_time

            self._response_time_cache[node.node_id] = final_time

        # 5. 存储弧段响应时间 (关键点法)
        center_coords = [(n.x, n.y) for n in centers]

        # 路网非直线系数 (默认 1.3，表示实际路程通常是直线距离的 1.3 倍)
        tortuosity_factor = self.config.get("tortuosity_factor", 1.3)

        for arc in self.network.arcs:
            # 获取两端点时间 (已经过兜底处理，肯定是有限值)
            t_i = node_travel_times.get(arc.start.node_id, float("inf"))
            t_j = node_travel_times.get(arc.end.node_id, float("inf"))

            if t_i == float("inf") or t_j == float("inf") or not center_coords:
                self._response_time_cache[(arc.start.node_id, arc.end.node_id)] = float(
                    "inf"
                )
                continue

            # 计算中点坐标
            mid_x = (arc.start.x + arc.end.x) / 2.0
            mid_y = (arc.start.y + arc.end.y) / 2.0

            # 计算 t_mid (欧氏距离近似)
            min_mid_time = float("inf")
            for cx, cy in center_coords:
                # dist = math.sqrt((mid_x - cx) ** 2 + (mid_y - cy) ** 2)
                # time = dist / speed_v if speed_v > 0 else float("inf")
                # 1. 计算平面直线距离 (数学上正确)
                euclidean_dist = math.sqrt((mid_x - cx) ** 2 + (mid_y - cy) ** 2)
                # 2. 修正为估计的路网距离 (物理上更真实)
                estimated_road_dist = euclidean_dist * tortuosity_factor
                # 3. 计算时间（x,y 单位为 km, speed_v 单位为 km/h）
                time = estimated_road_dist / speed_v if speed_v > 0 else float("inf")
                if time < min_mid_time:
                    min_mid_time = time

            # 中点时间也进行兜底
            min_mid_time = min(min_mid_time, self.max_response_time)

            # 应用积分公式
            t_integrated = (t_i + 2 * min_mid_time + t_j) / 4.0

            final_arc_time = setup_time + t_integrated

            # 最终兜底
            self._response_time_cache[(arc.start.node_id, arc.end.node_id)] = min(
                final_arc_time, self.max_response_time
            )

        logging.info(
            f"[RiskModel] 应急响应时间预计算完成 (Max Cap: {self.max_response_time}h)。"
        )
