# --- coding: utf-8 ---
# --- app/utils/analyzer.py ---
import os
import csv
import logging
import numpy as np
from typing import Dict, List, Optional
from app.core.solution import Solution
from app.core.evaluator import Evaluator
from app.core.fuzzy import FuzzyMath


def find_knee_point(solutions: List[Solution]) -> Optional[Solution]:
    """
    [算法] 寻找帕累托前沿上的 Knee Point (折衷解)。
    定义：距离理想点 (Min Cost, Min Risk) 归一化欧氏距离最近的点。
    """
    if not solutions:
        return None

    # 1. 获取目标值边界
    costs = [s.f2_cost for s in solutions]
    risks = [s.f1_risk for s in solutions]

    min_c, max_c = min(costs), max(costs)
    min_r, max_r = min(risks), max(risks)

    # 避免除以零
    range_c = max_c - min_c if max_c != min_c else 1.0
    range_r = max_r - min_r if max_r != min_r else 1.0

    best_sol = None
    min_dist = float("inf")

    # 2. 计算每个点到理想点 (归一化坐标 0,0) 的距离
    for s in solutions:
        norm_c = (s.f2_cost - min_c) / range_c
        norm_r = (s.f1_risk - min_r) / range_r

        # 距离公式 d^2 = x^2 + y^2
        dist = norm_c**2 + norm_r**2

        if dist < min_dist:
            min_dist = dist
            best_sol = s

    return best_sol


def generate_routing_scheme_comparison(
    solutions_map: Dict[str, Solution], evaluator: Evaluator, save_dir: str
):
    """
    生成详细的路由方案对比表 (CSV)。

    功能：
    1. 遍历 A, B, C 三个方案。
    2. 对每个方案中的每个 Task，单独计算其 Risk (CVaR) 和 Cost (Expected)。
    3. 输出包含路径细节、模式占比、碳排放等信息的 CSV。
    """
    file_name = "routing_scheme_details.csv"
    file_path = os.path.join(save_dir, file_name)
    logging.info(f"Generating detailed routing scheme to: {file_path}")

    # 定义 CSV 表头
    headers = [
        "Opinion",  # A, B, C
        "Task_ID",  # T1, T2...
        "Origin",  # 起点
        "Destination",  # 终点
        "Total_Risk",  # 该任务的风险 (CVaR)
        "Total_Cost",  # 该任务的总成本 (Expected)
        "Transport_Cost",
        "Transshipment_Cost",
        "Carbon_Cost",
        "Transfers",  # 中转次数
        "RoadOverRail_Ratio",  # 公路/铁路 距离占比 (e.g. "30% / 70%")
        "Transfer_Hubs",  # 转运枢纽 (e.g. "H1, H3")
        "Route_Path",  # 完整路径 (e.g. "S1 -> H1 -> H3 -> D1")
    ]

    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            # 遍历每个方案 (Opinion A, B, C)
            for label, solution in solutions_map.items():
                if not solution:
                    logging.warning(f"Solution for {label} is None, skipping.")
                    continue

                # 按照 Task ID 排序，保证输出整齐
                sorted_tasks = sorted(
                    solution.path_selections.items(), key=lambda x: x[0]
                )

                for task_id, path in sorted_tasks:
                    if not path.task:
                        continue

                    # --- 1. 计算单任务指标 ---
                    task_cost, transport_cost, transship_cost, carbon_cost = (
                        _calculate_single_task_cost(path, evaluator)
                    )
                    task_risk = _calculate_single_task_risk(path, evaluator)

                    # --- 2. 统计路径特征 ---
                    road_dist = 0.0
                    rail_dist = 0.0
                    route_nodes = [n.node_id for n in path.nodes]
                    route_str = " -> ".join(route_nodes)

                    transfer_hubs_str = (
                        ", ".join([h.node_id for h in path.transfer_hubs])
                        if path.transfer_hubs
                        else "Direct"
                    )
                    transfers = len(path.transfer_hubs)

                    for arc in path.arcs:
                        if arc.mode == "road":
                            road_dist += arc.length
                        elif arc.mode == "railway":
                            rail_dist += arc.length

                    total_dist = road_dist + rail_dist
                    if total_dist > 0:
                        ratio_str = f"{road_dist / total_dist * 100:.0f}% / {rail_dist / total_dist * 100:.0f}%"
                    else:
                        ratio_str = "N/A"

                    # --- 3. 写入 CSV 行 ---
                    writer.writerow(
                        [
                            label,
                            task_id,
                            path.task.origin.node_id,
                            path.task.destination.node_id,
                            f"{task_risk:.4f}",
                            f"{task_cost:.2f}",
                            f"{transport_cost:.2f}",
                            f"{transship_cost:.2f}",
                            f"{carbon_cost:.2f}",
                            transfers,
                            ratio_str,
                            transfer_hubs_str,
                            route_str,
                        ]
                    )

        logging.info(f"Successfully saved detailed routing scheme: {file_name}")

    except Exception as e:
        logging.error(f"Failed to save routing scheme CSV: {e}")
        import traceback

        logging.error(traceback.format_exc())


def calculate_solution_gini(solution: Solution, evaluator: Evaluator) -> float:
    """
    [封装函数] 计算给定解决方案的社会公平性指标（基尼系数）。

    Args:
        solution: 要评估的解。
        evaluator: 评估器。

    Returns:
        float: 基尼系数 (0.0 - 1.0)。
    """
    # 1. 获取风险暴露分布
    risk_map = _get_node_risk_exposure(solution, evaluator)

    # 2. 提取风险值列表
    risk_exposures = list(risk_map.values())

    # 3. 计算基尼系数
    gini = _calculate_gini_coefficient(risk_exposures)

    return gini


def _calculate_gini_coefficient(risk_exposures: List[float]) -> float:
    """
    [核心算法] 计算基于风险暴露分布的基尼系数 (Gini Coefficient)。
    用于衡量风险在不同地理区域/节点间分配的公平性。
    Formula: Gini = (sum_i sum_j |R_i - R_j|) / (2 * n^2 * mean(R))

    Args:
        risk_exposures: 每个区域/节点承担的总风险暴露 R_i 列表。

    Returns:
        float: 基尼系数 (0.0 - 1.0)。
    """
    if not risk_exposures or len(risk_exposures) < 2:
        return 0.0

    # 1. 转换为 Numpy 数组并过滤零值（零风险区域不影响相对公平性，但数学上保留）
    R = np.array([r for r in risk_exposures if r >= 0])
    n = len(R)

    # 如果所有风险都是零，则 Gini=0 (完全公平)
    if n == 0 or np.sum(R) == 0:
        return 0.0

    # 2. 计算分子: sum_i sum_j |R_i - R_j|
    # R[:, None] - R[None, :] 得到 R_i - R_j 的差值矩阵
    diff_matrix = np.abs(R[:, None] - R[None, :])
    numerator = np.sum(diff_matrix)

    # 3. 计算分母: 2 * n^2 * mean(R)
    R_mean = np.mean(R)
    denominator = 2 * n * n * R_mean

    # 4. 计算 Gini
    # 避免除以零（已在 np.sum(R) == 0 处处理）
    gini = numerator / denominator

    return float(gini)


def _get_node_risk_exposure(
    solution: Solution, evaluator: Evaluator
) -> Dict[str, float]:
    """
    [Helper] 计算给定方案下，每个节点（或区域）的总风险暴露。

    总风险暴露 R_k =
      sum_{v, p: k in p} [ 风险事件发生时，该节点 k 贡献的后果 * 发生概率 ]

    Args:
        solution: 要评估的解。
        evaluator: 评估器 (用于获取风险参数)。

    Returns:
        Dict[str, float]: {node_id: total_risk_contribution, ...}
    """
    node_risk_map: Dict[str, float] = {}

    for path in solution.path_selections.values():
        if not path.task:
            continue

        # 任务运量 dv
        dv = path.task.demand

        # 1. 收集弧段风险贡献
        for arc in path.arcs:
            # 弧段事故概率 p_ijm
            p_ijm = arc.accident_prob_per_km * arc.length
            # 弧段后果 c_ijm (动态后果)
            c_base = evaluator.risk_model.get_consequence(arc)
            c_ijm_final = c_base * dv

            # 贡献风险 = 概率 * 后果
            risk_contrib = p_ijm * c_ijm_final

            # 将弧段风险贡献平均分配给弧段的两个端点作为区域风险暴露
            # 这是一个常见的简化，将弧段风险转化为节点/区域风险
            u_id = arc.start.node_id
            v_id = arc.end.node_id

            node_risk_map[u_id] = node_risk_map.get(u_id, 0.0) + risk_contrib / 2.0
            node_risk_map[v_id] = node_risk_map.get(v_id, 0.0) + risk_contrib / 2.0

        # 2. 收集枢纽风险贡献
        for hub in path.transfer_hubs:
            # 枢纽事故概率 p_k
            p_k = hub.accident_prob
            # 枢纽后果 c_k (动态后果)
            c_base = evaluator.risk_model.get_consequence(hub)
            c_k_final = c_base * dv

            # 贡献风险 = 概率 * 后果
            risk_contrib = p_k * c_k_final

            node_risk_map[hub.node_id] = (
                node_risk_map.get(hub.node_id, 0.0) + risk_contrib
            )

    return node_risk_map


def _calculate_single_task_cost(path, evaluator: Evaluator):
    """
    [Helper] 重新计算单个任务的 Expected Cost 和 Carbon Cost
    """
    total_cost = 0.0
    transport_cost = 0.0
    transship_cost = 0.0
    carbon_cost = 0.0
    dv = path.task.demand

    # 1. 弧段成本
    for arc in path.arcs:
        mode = arc.mode
        d_ij = arc.length

        C_m = evaluator.unit_transport_cost.get(mode, 0.0)
        E_m = evaluator.unit_carbon_cost.get(mode, 0.0)
        P = evaluator.unit_operation_cost

        # 时间期望 (调用 FuzzyMath)
        exp_time = FuzzyMath.triangular_expected_value(*arc.fuzzy_transport_time)

        segment_transport = C_m * dv * d_ij + P * exp_time
        segment_carbon = E_m * dv * d_ij
        segment_cost = segment_transport + segment_carbon

        total_cost += segment_cost
        transport_cost += segment_transport
        carbon_cost += segment_carbon

    # 2. 枢纽成本
    for hub in path.transfer_hubs:
        B_k = evaluator.unit_transshipment_cost
        I_k = evaluator.unit_transshipment_infra_cost

        exp_time = FuzzyMath.trapezoidal_expected_value(*hub.fuzzy_transshipment_time)

        hub_cost = B_k * dv + I_k * exp_time

        transship_cost += hub_cost
        total_cost += hub_cost

    return total_cost, transport_cost, transship_cost, carbon_cost


def _calculate_single_task_risk(path, evaluator: Evaluator) -> float:
    """
    [Helper] 重新计算单个任务的 CVaR 风险
    """
    dv = path.task.demand
    p_c_pairs = []

    # 1. 收集 Arc 风险
    for arc in path.arcs:
        c_base = evaluator.risk_model.get_consequence(arc)
        p_ij = arc.accident_prob_per_km * arc.length

        c_final = c_base * dv
        if p_ij > 0 and c_final > 0:
            p_c_pairs.append((p_ij, c_final))

    # 2. 收集 Hub 风险
    for hub in path.transfer_hubs:
        c_base = evaluator.risk_model.get_consequence(hub)
        p_k = hub.accident_prob

        c_final = c_base * dv
        if p_k > 0 and c_final > 0:
            p_c_pairs.append((p_k, c_final))

    if not p_c_pairs:
        return 0.0

    # 3. 计算 CVaR
    alpha = evaluator.risk_config.get("cvar_alpha", 0.95)
    one_minus_alpha = 1.0 - alpha
    _, task_cvar = evaluator._calc_cvar_for_path(p_c_pairs, alpha, one_minus_alpha)

    return task_cvar
