import os
import json
import argparse
from config import RHMVSPConfig
from src.core.data_generator import InstanceGenerator

"""
数据生成脚本 (Data Generator Script)

本脚本用于根据指定的种子 (seeds) 动态生成 RHMVSP 实例数据。
您可以自由指定生成的实例数量、对应的随机种子，以及要生成的规模。

使用方式示例 (Usage Examples):

1. 为所有规模 (small, medium, large) 各生成 2 个实例，使用种子 42 和 43：
   > python generate_data.py --num_instances 2 --seeds 42 43

2. 只为 small 规模生成 3 个实例，使用特定种子：
   > python generate_data.py --num_instances 3 --seeds 100 200 300 --scales small

3. 错误示例（程序会拦截）：提供的实例数和种子数不一致：
   > python generate_data.py --num_instances 2 --seeds 42
   (将报错: 提供的种子数量 (1) 必须与要生成的实例个数 (2) 相等！)
"""

def generate_instances(num_instances, seeds, scales, output_dir):
    # 检查：种子数量必须和实例个数相等
    if len(seeds) != num_instances:
        raise ValueError(f"Error: 提供的种子数量 ({len(seeds)}) 必须与要生成的实例个数 ({num_instances}) 相等！")
    
    base_sizes = {
        "small":  {"nodes": 25,  "hubs": 5,  "od_pairs": 5,  "vehicles": 30, "T_max_init": 24},
        "medium": {"nodes": 60,  "hubs": 10, "od_pairs": 15, "vehicles": 90, "T_max_init": 36},
        "large":  {"nodes": 120, "hubs": 20, "od_pairs": 30, "vehicles": 150, "T_max_init": 48},
    }

    for scale in scales:
        os.makedirs(f"{output_dir}/{scale}", exist_ok=True)
        print(f"\n--- Generating {scale} instances ---")
        
        for i in range(num_instances):
            inst_idx = i + 1
            seed = seeds[i]
            
            # 每次重新初始化 config 确保隔离
            config = RHMVSPConfig()
            gen = InstanceGenerator(config, seed=seed)
            
            params = base_sizes[scale]
            
            # 按照用户的最新要求：硬编码每类规模的两个实例的 nodes 数量
            hardcoded_nodes = {
                "small": [20, 50],
                "medium": [80, 120],
                "large": [150, 200]
            }
            
            # 获取硬编码的节点数量，如果超出2个实例则循环取值
            num_nodes = hardcoded_nodes[scale][i % len(hardcoded_nodes[scale])]
            
            # 其他参数（hubs, od_pairs, vehicles）基于实际分配的 nodes 数量对比原基准的比例进行等比缩放（自适应）
            scale_ratio = num_nodes / params["nodes"]
            num_hubs = max(2, int(round(params["hubs"] * scale_ratio)))
            num_od_pairs = max(2, int(round(params["od_pairs"] * scale_ratio)))
            num_vehicles = max(5, int(round(params["vehicles"] * scale_ratio)))

            config.T_max = max(config.T_max, params["T_max_init"])
            
            # 直接调用底层方法注入带波动的参数
            inst = gen._generate_instance(
                name=f"rhmvsp_{scale}",
                num_nodes=num_nodes,
                num_hubs=num_hubs,
                num_od_pairs=num_od_pairs,
                num_vehicles=num_vehicles,
            )

            # ---- Post-process: add origin_hub / destination_hub to OD pairs ----
            transfer_hub_ids = [tn["node_id"] for tn in inst["network"].get("transfer_nodes", [])]
            # Build a mapping from node_id to its coordinates
            node_coords = {}
            for n in inst["network"].get("nodes", []):
                node_coords[n["id"]] = (n.get("x", 0), n.get("y", 0))
            hub_coords = {h: node_coords.get(h, (0, 0)) for h in transfer_hub_ids}

            for od in inst["od_pairs"]:
                origin = od["origin"]
                destination = od["destination"]
                # Find nearest hub to origin
                if transfer_hub_ids and origin in node_coords:
                    ox, oy = node_coords[origin]
                    best_h = min(transfer_hub_ids, key=lambda h: (hub_coords[h][0]-ox)**2 + (hub_coords[h][1]-oy)**2)
                    od["origin_hub"] = best_h
                elif transfer_hub_ids:
                    od["origin_hub"] = transfer_hub_ids[0]
                else:
                    od["origin_hub"] = -1
                # Find nearest hub to destination
                if transfer_hub_ids and destination in node_coords:
                    dx, dy = node_coords[destination]
                    best_h = min(transfer_hub_ids, key=lambda h: (hub_coords[h][0]-dx)**2 + (hub_coords[h][1]-dy)**2)
                    od["destination_hub"] = best_h
                elif transfer_hub_ids:
                    od["destination_hub"] = transfer_hub_ids[-1]
                else:
                    od["destination_hub"] = -1

            # ---- Post-process: add hub_fleet_size ----
            hub_fleet = {}
            for h in transfer_hub_ids:
                hub_fleet[str(h)] = max(5, num_vehicles // max(1, len(transfer_hub_ids)))
            inst["hub_fleet_size"] = hub_fleet

            # ---- Post-process: add num_vehicles_west / num_vehicles_east ----
            half = num_vehicles // 2
            inst["num_vehicles_west"] = half
            inst["num_vehicles_east"] = num_vehicles - half

            file_path = f"{output_dir}/{scale}/instance_{inst_idx}.json"
            with open(file_path, "w") as f:
                json.dump(inst, f, indent=4)
            print(f"Generated {file_path} (seed={seed}, nodes={num_nodes}, hubs={num_hubs})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RHMVSP Instances")
    parser.add_argument("--num_instances", type=int, required=True, help="Number of instances to generate per scale (e.g. 2).")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="List of seeds to use. Must match num_instances. (e.g. --seeds 42 43)")
    parser.add_argument("--scales", type=str, nargs="+", default=["small", "medium", "large"], help="Scales to generate (e.g. --scales small medium).")
    parser.add_argument("--output_dir", type=str, default="data_performance", help="Directory to save the generated instances (default: data_performance).")
    
    args = parser.parse_args()
    
    try:
        generate_instances(args.num_instances, args.seeds, args.scales, args.output_dir)
        print("\nAll instances generated successfully!")
    except ValueError as e:
        print(e)
