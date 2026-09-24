"""
Generate Real Geographic Data for RHMVSP Case Study (Xinjiang - Shanghai)
"""

import os
import json
import csv
import copy
import argparse

from config import RHMVSPConfig
from src.core.real_data_generator import RealInstanceGenerator
from src.core.network import MultimodalNetwork
from src.utils.plotting import Visualizer
from src.utils.logging import Logger

def generate_real_instance(
    output_dir="data_real_case",
    seed=None,
):
    print(f"\n--- Generating Real Geographic Instance (Xinjiang-Shanghai) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    config = RHMVSPConfig()
    if seed is not None:
        config.seed = seed
    
    # We use a deepcopy so generation doesn't permanently mutate the global config prematurely
    gen_config = copy.deepcopy(config)
    
    generator = RealInstanceGenerator(gen_config, seed=config.seed)
    instance = generator.generate()
    
    # Save the json
    output_json = os.path.join(output_dir, "instance.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(instance, f, indent=4, default=str)
    print(f"Generated {output_json} (T_max = {instance['config']['T_max']})")
    
    # Save CSVs for easy viewing
    nodes_path = os.path.join(output_dir, "nodes.csv")
    if instance["network"]["nodes"]:
        node_rows = [
            {
                "id": node["id"],
                "x": node["x"],
                "y": node["y"],
                "name": node["name"],
                "reposition_hub_id": node.get("reposition_hub_id"),
            }
            for node in instance["network"]["nodes"]
        ]
        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(node_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(node_rows)

    if instance["network"]["arcs"]:
        with open(os.path.join(output_dir, "arcs.csv"), "w", newline="", encoding="utf-8") as f:
            fieldnames = list(instance["network"]["arcs"][0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(instance["network"]["arcs"])

    if instance["od_pairs"]:
        with open(os.path.join(output_dir, "od_pairs.csv"), "w", newline="", encoding="utf-8") as f:
            fieldnames = list(instance["od_pairs"][0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(instance["od_pairs"])

    timetable_path = os.path.join(output_dir, "railway_timetable.csv")
    if instance.get("railway_timetable"):
        with open(timetable_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(instance["railway_timetable"][0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(instance["railway_timetable"])
            
    print(f"Instance saved to directory: {output_dir}/")
    print(f"  Nodes: {instance['network']['num_nodes']}")
    print(f"  Arcs: {instance['network']['num_arcs']}")
    print(f"  O-D pairs: {len(instance['od_pairs'])}")
    print(f"  Hub fleet sizes: {instance.get('hub_fleet_size', {})}")
    print(f"  T_max: {instance['config']['T_max']} hours")

    # Try plotting
    try:
        network = MultimodalNetwork.from_instance(instance)
        # Using a dummy logger for visualizer
        logger = Logger(level="WARNING")
        viz = Visualizer(gen_config)
        net_path = os.path.join(output_dir, "network.png")
        viz.plot_network(network, net_path)
        print(f"Network topology plot saved to: {net_path}")
    except Exception as e:
        print(f"Warning: Failed to generate network topology plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RHMVSP Real Case Instance")
    parser.add_argument("--output_dir", type=str, default="data_real_case", help="Directory to save the generated instance.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override; defaults to RHMVSPConfig.seed.",
    )
    
    args = parser.parse_args()
    generate_real_instance(
        args.output_dir,
        args.seed,
    )
    print("\nReal instance generated successfully!")
