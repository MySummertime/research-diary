import json
import math
import os
import random


def generate_benchmark_data(
    seed=14,
    n_hubs=30,
    n_non_hubs=70,
    total_arcs=1500,
    n_tasks=100,
    output_dir="data_benchmark",
):
    """
    [极致加速版] 生成轴辐式路网数据。
    优化点：使用 Set 替代 List 进行路径查重，时间复杂度从 O(N^2) 降至 O(N)。
    """
    random.seed(seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 生成节点 (Nodes)
    nodes = []
    for i in range(1, n_hubs + n_non_hubs + 1):
        is_hub = i <= n_hubs
        nodes.append(
            {
                "node_id": str(i),
                "name": f"{'Hub' if is_hub else 'City'}_{i}",
                "node_type": "hub" if is_hub else "non-hub",
                "x": round(random.uniform(73.0, 135.0), 2),
                "y": round(random.uniform(18.0, 53.0), 2),
                "is_emergency_center": random.random() < (0.8 if is_hub else 0.1),
                "capacity": random.randint(10000, 20000)
                if is_hub
                else random.randint(1000, 3000),
                "population_density": round(random.uniform(0.01, 0.05), 4)
                if is_hub
                else round(random.uniform(0.001, 0.01), 4),
                "accident_prob": round(random.uniform(3e-6, 8e-6), 7)
                if is_hub
                else 1e-9,
                "fuzzy_transshipment_time": [0.8, 1.0, 1.2, 1.5]
                if is_hub
                else [0, 0, 0, 0],
            }
        )

    hubs = [n for n in nodes if n["node_type"] == "hub"]
    non_hubs = [n for n in nodes if n["node_type"] == "non-hub"]

    # -------------------------------------------------------
    # 2. 生成弧段 (Arcs)
    # -------------------------------------------------------
    arcs = []
    existing_edges = set()  # 使用集合记录已存在的 (start, end)，实现 O(1) 查找

    def get_dist(n1, n2):
        return math.sqrt((n1["x"] - n2["x"]) ** 2 + (n1["y"] - n2["y"]) ** 2) * 100

    def add_arc(n1, n2, mode, capacity):
        """[Helper] 安全添加弧段并更新索引"""
        edge_key = (n1["node_id"], n2["node_id"])
        if edge_key not in existing_edges:
            dist = get_dist(n1, n2)
            # 根据模式设定参数
            p_density = (
                round(random.uniform(0.005, 0.01), 4)
                if mode == "railway"
                else round(random.uniform(0.01, 0.03), 4)
            )
            acc_prob = 4e-7 if mode == "railway" else 1.2e-6
            # 这里的 speed 分别对应铁路(70,60,50)和公路(45,40,35)
            speeds = [70, 60, 50] if mode == "railway" else [45, 40, 35]

            arcs.append(
                {
                    "start_node_id": n1["node_id"],
                    "end_node_id": n2["node_id"],
                    "mode": mode,
                    "length": round(dist, 1),
                    "capacity": capacity,
                    "population_density": p_density,
                    "accident_prob_per_km": acc_prob,
                    "fuzzy_transport_time": [round(dist / s, 1) for s in speeds],
                }
            )
            existing_edges.add(edge_key)

    # A. 铁路网 (Hub-to-Hub) 强连通环
    for i in range(n_hubs):
        h1, h2 = hubs[i], hubs[(i + 1) % n_hubs]
        add_arc(h1, h2, "railway", 30000)
        add_arc(h2, h1, "railway", 30000)

    # B. 公路网 (Non-hub to Hub)
    for nh in non_hubs:
        # 预计算所有距离，避免在 sorted() 内部重复计算
        h_dists = [(h, get_dist(nh, h)) for h in hubs]
        h_dists.sort(key=lambda x: x[1])
        for h, d in h_dists[:6]:
            add_arc(nh, h, "road", 8000)
            add_arc(h, nh, "road", 8000)

    # C. 填充剩余配额 (优化后的 While 循环)
    max_possible_rail_edges = n_hubs * (n_hubs - 1)
    while (
        len(arcs) < total_arcs
        and len(existing_edges) < max_possible_rail_edges + len(non_hubs) * 12
    ):
        h1, h2 = random.sample(hubs, 2)
        add_arc(h1, h2, "railway", 30000)

    # 3. 生成大跨度任务 (Tasks)
    tasks = []
    task_count = 0
    # 限制尝试次数，防止因地理分布导致死循环
    attempts = 0
    while task_count < n_tasks and attempts < 10000:
        attempts += 1
        o, d = random.sample(non_hubs, 2)
        if get_dist(o, d) > 2500:
            task_count += 1
            tasks.append(
                {
                    "task_id": str(task_count),
                    "origin_node_id": o["node_id"],
                    "destination_node_id": d["node_id"],
                    "demand": random.randint(100, 300),
                }
            )

    # 保存文件
    with open(os.path.join(output_dir, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "arcs.json"), "w", encoding="utf-8") as f:
        json.dump(arcs, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "tasks.json"), "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"✨ 成功生成算例: {len(nodes)} 节点, {len(arcs)} 弧段, {len(tasks)} 任务。")


# 执行生成 (可以随意修改这里的数值)
generate_benchmark_data(seed=4, n_hubs=15, n_non_hubs=35, total_arcs=65, n_tasks=25)
