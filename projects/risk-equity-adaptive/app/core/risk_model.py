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
from app.core.fuzzy import FuzzyMath


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

        # 非对称不确定性控制
        # emergency_a_multiplier: 控制乐观侧扩展 (δ_a)，默认1.0
        self.emerg_a_multiplier = self.config.get("emergency_a_multiplier", 1.0)
        # emergency_c_multiplier: 控制悲观侧扩展 (δ_c)，默认1.0
        self.emerg_c_multiplier = self.config.get("emergency_c_multiplier", 1.0)

        # 不确定性控制
        self.emerg_b_multiplier = self.config.get("emergency_b_multiplier", 1.0)

        # 缓存计算出的应急时间 → 三角模糊数 (a, b, c)
        self._response_time_cache: Dict[
            str | Tuple[str, str], Tuple[float, float, float]
        ] = {}

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

        # 1. 获取模糊响应时间
        if isinstance(entity, Arc):
            fuzzy_t_e = self._response_time_cache.get(
                (entity.start.node_id, entity.end.node_id),
                (
                    self.max_response_time,
                    self.max_response_time,
                    self.max_response_time,
                ),
            )
        else:
            fuzzy_t_e = self._response_time_cache.get(
                entity.node_id,
                (
                    self.max_response_time,
                    self.max_response_time,
                    self.max_response_time,
                ),
            )

        # 使用可信性期望值
        t_e_expected = FuzzyMath.triangular_expected_value(*fuzzy_t_e)

        # 2. 物理扩散模型: A_spread = γ * q * t
        # q: kg/s, need to be transformed to kg/h for alignment
        a_spread = gamma * q * (t_e_expected * 3600.0)

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
        使用 关键点法 积分近似公式: t_ij = t_0 + 1/4 * (t_i + 2*t_mid + t_j)。
        使用最可能旅行时间（b 值）跑 Dijkstra 得到核心响应时间。
        通过 emergency_b_multiplier (δ_b) 独立控制期望值: b = b * δ_b
        通过 emergency_a_multiplier (δ_a) 独立控制乐观侧扩展: a = b - base * δ_a
        通过 emergency_c_multiplier (δ_c) 独立控制悲观侧扩展: c = b + base * δ_c
        """
        logging.info("[RiskModel] 正在预计算非对称模糊应急响应时间...")

        setup_time = self.config.get("emergency_setup_time", 0.05)  # t0，确定性
        tortuosity_factor = self.config.get("tortuosity_factor", 1.3)

        # 应急响应期望值控制参数
        delta_b = self.emerg_b_multiplier

        # 独立边界控制参数
        delta_a = self.emerg_a_multiplier  # 乐观侧扩展系数 δ_a
        delta_c = self.emerg_c_multiplier  # 悲观侧扩展系数 δ_c

        # 1. 构建路网图：使用每条弧的“最可能”应急旅行时间作为权重
        road_graph = nx.DiGraph()
        for arc in self.network.arcs:
            # if arc.mode == "road":
            # 使用配置中的确定性速度
            speed_v = self.config.get("emergency_vehicle_speed", 45.0)
            most_likely_time = arc.length / speed_v if speed_v > 0 else float("inf")
            most_likely_time *= delta_b
            road_graph.add_edge(
                arc.start.node_id, arc.end.node_id, weight=most_likely_time
            )

        # 2. 识别应急中心
        centers = self.network.get_emergency_centers()
        center_ids = [n.node_id for n in centers]

        # 3. 计算节点到最近应急中心的“最可能”旅行时间（Multi-source Dijkstra）
        node_most_likely_travel = {}
        if center_ids:
            try:
                node_most_likely_travel = nx.multi_source_dijkstra_path_length(
                    road_graph.reverse(),  # 反向图：从中心向外扩散
                    sources=center_ids,
                    weight="weight",
                )
            except Exception as e:
                logging.warning(f"[RiskModel] Dijkstra 计算失败: {e}")

        # 4. 计算节点的模糊响应时间
        for node in self.network.nodes:
            b_travel = node_most_likely_travel.get(node.node_id, float("inf"))
            if b_travel == float("inf"):
                # 无路径：退化为最大响应时间（确定性）
                a = b = c = self.max_response_time
            else:
                b = b_travel + setup_time  # 最可能总响应时间 (b_travel 已包含 delta_b)
                base_extend = b_travel  # 可扩展的基础宽度（只使用旅行时间）

                # 独立扩展左右边界
                left_extend = base_extend * delta_a  # 乐观侧扩展量
                right_extend = base_extend * delta_c  # 悲观侧扩展量

                a = max(b - left_extend, setup_time)  # 乐观边界（不能低于 setup_time）
                c = min(
                    b + right_extend, self.max_response_time
                )  # 悲观边界（不超过上限）
                a = min(a, b)  # 保证 a ≤ b ≤ c

            self._response_time_cache[node.node_id] = (a, b, c)

        # 5. 计算弧段的模糊响应时间（模糊关键点法 + 独立边界）
        for arc in self.network.arcs:
            # if arc.mode != "road":
            #     continue

            # 获取端点和中点的模糊响应时间（已计算）
            fuzzy_i = self._response_time_cache.get(
                arc.start.node_id, (self.max_response_time,) * 3
            )
            fuzzy_j = self._response_time_cache.get(
                arc.end.node_id, (self.max_response_time,) * 3
            )

            # 计算中点的最可能旅行时间（欧氏距离 + 迂回修正）
            mid_x = (arc.start.x + arc.end.x) / 2.0
            mid_y = (arc.start.y + arc.end.y) / 2.0
            min_mid_most_likely = float("inf")
            for center in centers:
                eucl_dist = math.sqrt((mid_x - center.x) ** 2 + (mid_y - center.y) ** 2)

                road_dist = eucl_dist * tortuosity_factor
                speed_v = self.config.get("emergency_vehicle_speed", 45.0)
                time = road_dist / speed_v if speed_v > 0 else float("inf")
                min_mid_most_likely = min(min_mid_most_likely, time)

            min_mid_most_likely_with_delta_b = float("inf")
            for center in centers:
                eucl_dist = math.sqrt((mid_x - center.x) ** 2 + (mid_y - center.y) ** 2)
                road_dist = eucl_dist * tortuosity_factor
                speed_v = self.config.get("emergency_vehicle_speed", 45.0)
                time = road_dist / speed_v if speed_v > 0 else float("inf")
                time *= delta_b  # 确保应用乘子
                min_mid_most_likely_with_delta_b = min(
                    min_mid_most_likely_with_delta_b, time
                )

            if min_mid_most_likely_with_delta_b == float("inf"):
                fuzzy_mid = (self.max_response_time,) * 3
            else:
                b_mid = min_mid_most_likely_with_delta_b + setup_time
                base_mid = min_mid_most_likely_with_delta_b  # 可扩展的基础宽度（只使用旅行时间）

                a_mid = max(b_mid - base_mid * delta_a, setup_time)
                c_mid = min(b_mid + base_mid * delta_c, self.max_response_time)
                a_mid = min(a_mid, b_mid)

                fuzzy_mid = (a_mid, b_mid, c_mid)

            # 关键点法：使用三个点的期望值计算弧段的核心响应时间 b_arc
            exp_i = FuzzyMath.triangular_expected_value(*fuzzy_i)
            exp_mid = FuzzyMath.triangular_expected_value(*fuzzy_mid)
            exp_j = FuzzyMath.triangular_expected_value(*fuzzy_j)
            b_arc = (exp_i + 2 * exp_mid + exp_j) / 4.0

            # 弧段基础可扩展宽度
            base_extend_arc = b_arc - setup_time

            # 独立扩展弧段左右边界
            left_extend_arc = base_extend_arc * delta_a
            right_extend_arc = base_extend_arc * delta_c

            a_arc = max(b_arc - left_extend_arc, setup_time)
            c_arc = min(b_arc + right_extend_arc, self.max_response_time)
            a_arc = min(a_arc, b_arc)

            self._response_time_cache[(arc.start.node_id, arc.end.node_id)] = (
                a_arc,
                b_arc,
                c_arc,
            )

        logging.info(
            f"[RiskModel] 独立左右边界模糊应急响应时间预计算完成！🎉 "
            f"(δ_a={delta_a:.2f}, δ_c={delta_c:.2f}, Max Cap={self.max_response_time}h)"
        )
