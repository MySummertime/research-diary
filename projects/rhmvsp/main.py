# RHMVSP - Reliable Hazardous Materials Multi-modal Vehicle Scheduling Problem
# Single entry point: main.py
# Usage: python main.py [--instance FILE] [--mode solve|generate|demo] [--solver pulp|gurobi]

import argparse
import copy
import csv
import json
import math
import os
import sys
from datetime import datetime

from config import RHMVSPConfig
from src.core.data_generator import InstanceGenerator
from src.core.network import MultimodalNetwork
from src.core.synthetic_beliefs import arc_belief, draw_belief
from src.solvers.branch_and_price import BranchAndPriceController as BranchAndPriceSolver
from src.utils.logging import Logger
from src.utils.plotting import Visualizer
from src.utils.reporting import format_solution_report
from src.utils.numeric_format import format_alpha_R, format_alpha_T, format_csv_value
from src.utils.timing import Timer


def synchronize_instance_parameters(instance: dict, config: RHMVSPConfig) -> dict:
	"""将旧实例中的时间、容量和费率统一到当前配置。"""
	horizon = float(config.T_max)
	for od in instance.get("od_pairs", []):
		od["time_window_start"] = 0.0
		od["time_window_end"] = horizon
		od["time_window_start_fmt"] = "Day 1 00:00"
		day_number = int(horizon // 24) + 1
		remaining_hours = horizon % 24
		od["time_window_end_fmt"] = f"Day {day_number} {int(remaining_hours):02d}:00"
	instance.setdefault("config", {})["T_max"] = horizon
	instance["config"]["alpha_R"] = float(config.alpha_R)
	instance["config"]["alpha_T"] = float(config.alpha_T)
	instance["config"]["q_road"] = float(config.q_road)
	instance["config"]["q_rail"] = float(config.q_rail)
	for service in instance.get("railway_timetable", []):
		service["capacity"] = float(config.q_rail)
	for arc in instance.get("network", {}).get("arcs", []):
		distance = float(arc["distance"])
		if arc["mode"] == 1:
			ratios = config.road_fuzzy_time_ratio
			base_time = distance / config.road_speed
			arc["unit_cost"] = round(config.road_loaded_cost(distance), 2)
		else:
			ratios = config.rail_fuzzy_time_ratio
			base_time = distance / config.rail_speed
			# Rail cost depends on the total service load and is modeled in the master problem.
			arc["unit_cost"] = 0.0
		arc["time_a"], arc["time_b"], arc["time_c"] = [round(base_time * x, 2) for x in ratios]
		arc.pop("time_d", None)
		for obsolete_key in ("capacity", "e_window", "l_window", "e_window_fmt", "l_window_fmt"):
			arc.pop(obsolete_key, None)
	for node in instance.get("network", {}).get("transfer_nodes", []):
		for key in ("transfer_time_a", "transfer_time_b", "transfer_time_c", "transfer_time_d"):
			node.pop(key, None)
		node["unit_transfer_cost"] = config.c_transfer_road_rail
	return instance


def synchronize_risk_parameters(instance: dict, config: RHMVSPConfig) -> dict:
	"""按稳定组件标识生成并固定合成事故信度。"""
	for arc in instance.get("network", {}).get("arcs", []):
		arc["alpha"] = arc_belief(
			config.seed, int(arc.get("id", 0)), int(arc["i"]), int(arc["j"]), int(arc["mode"]),
			config.road_accident_belief_bounds, config.rail_accident_belief_bounds,
		)
	for transfer in instance.get("network", {}).get("transfer_nodes", []):
		transfer.pop("gamma", None)
		hub = int(transfer["node_id"])
		transfer["gamma_rr"] = draw_belief(
			config.seed, f"transfer:{hub}:rr", config.transfer_accident_belief_rail_rail_bounds
		)
		transfer["gamma_ir"] = draw_belief(
			config.seed, f"transfer:{hub}:ir", config.transfer_accident_belief_intermodal_bounds
		)
	return instance


def parse_args():
	"""Parse command-line arguments."""
	config_default = RHMVSPConfig()

	parser = argparse.ArgumentParser(
		description="RHMVSP: Reliable Hazardous Materials Multi-modal Vehicle Scheduling Problem",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python main.py --mode demo                    # Run demo with benchmark small instance
  python main.py --mode generate --size medium   # Generate medium benchmark dataset
  python main.py --mode solve --instance data/medium/instance.json --solver gurobi # Solve specific instance with Gurobi
        """,
	)
	parser.add_argument(
		"--mode",
		choices=["solve", "generate", "demo"],
		default="demo",
		help="Execution mode: solve, generate, or demo (default: demo)",
	)
	parser.add_argument(
		"--instance",
		type=str,
		default="data/small/instance.json",
		help="Path to instance JSON file (default: data/small/instance.json)",
	)
	parser.add_argument(
		"--size",
		choices=["small", "medium", "large"],
		default="small",
		help="Instance size for generate/demo mode (default: small)",
	)
	parser.add_argument(
		"--solver",
		choices=["pulp", "gurobi"],
		default=config_default.solver_name,
		help=f"MILP solver backend: pulp or gurobi (default: {config_default.solver_name})",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="results",
		help="Output directory for results (default: results)",
	)
	parser.add_argument(
		"--time-limit",
		type=int,
		default=config_default.time_limit,
		help=f"Solver time limit in seconds (default: {config_default.time_limit})",
	)
	parser.add_argument(
		"--log-level",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		default="INFO",
		help="Logging level (default: INFO)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=config_default.seed,
		help=f"Random seed for reproducibility (default: {config_default.seed})",
	)
	return parser.parse_args()


def run_demo(config: RHMVSPConfig, size: str, logger: Logger):
	"""Run demo with a synthetic instance of specified size."""
	timer = Timer()
	timer.start()
	# S2-1: Deep copy config to avoid mutating caller's config (consistent with run_solve)
	config = copy.deepcopy(config)
	logger.info("=" * 60)
	logger.info(f"RHMVSP Demo Mode - {size.upper()} Synthetic Instance")
	logger.info("=" * 60)

	# Load or generate benchmark instance
	benchmark_path = f"data/{size}/instance.json"
	if os.path.exists(benchmark_path):
		logger.info(f"Loading fixed benchmark instance from: {benchmark_path}")
		with open(benchmark_path) as f:
			instance = json.load(f)
	else:
		logger.info(f"Generating and saving fixed benchmark {size} instance...")
		instance = run_generate(config, size, logger)
	instance = synchronize_instance_parameters(instance, config)
	instance = synchronize_risk_parameters(instance, config)

	if "config" in instance:
		if "alpha_R" in instance["config"]:
			config.alpha_R = instance["config"]["alpha_R"]
		if "alpha_T" in instance["config"]:
			config.alpha_T = instance["config"]["alpha_T"]

	logger.info(f"Instance: {instance['name']}")
	logger.info(
		f"  Nodes: {instance['network']['num_nodes']}, "
		f"Arcs: {instance['network']['num_arcs']}, "
		f"O-D pairs: {len(instance['od_pairs'])}, "
		f"Hub fleet sizes: {instance.get('hub_fleet_size', {})}"
	)

	# Display network summary
	network = MultimodalNetwork.from_instance(instance)
	logger.info("\nNetwork Summary:")
	logger.info(f"  Road arcs: {len(network.get_road_arcs())}")
	logger.info(f"  Rail arcs: {len(network.get_rail_arcs())}")
	logger.info(f"  Transfer nodes: {len(network.get_transfer_nodes())}")

	# Display uncertainty parameters
	logger.info("\nUncertainty Parameters:")
	logger.info(f"  Alpha_R (risk reliability threshold): {format_alpha_R(config.alpha_R)}")
	logger.info(f"  Alpha_T (time reliability threshold): {format_alpha_T(config.alpha_T)}")

	# Solve with Branch-and-Price
	logger.info("\n" + "=" * 60)
	logger.info("Starting Branch-and-Price Solver")
	logger.info("=" * 60)

	solver = BranchAndPriceSolver(config, instance, logger)
	solution = solver.solve()
	elapsed_wall, elapsed_cpu = timer.stop()
	solution["end_to_end_wall_time"] = elapsed_wall
	solution["end_to_end_cpu_time"] = elapsed_cpu

	# Display results
	logger.info("\n" + "=" * 60)
	logger.info("Solution Report")
	logger.info("=" * 60)
	report = format_solution_report(solution, instance, total_wall=elapsed_wall, total_cpu=elapsed_cpu)
	logger.info(report)

	return solution, instance


def run_solve(config: RHMVSPConfig, instance_path: str, logger: Logger):
	"""Solve a given instance."""
	timer = Timer()
	timer.start()
	logger.info("=" * 60)
	logger.info(f"Solving Instance: {instance_path}")
	logger.info("=" * 60)

	# Load instance
	with open(instance_path) as f:
		instance = json.load(f)

	# Deep copy to avoid leaking instance-specific overrides back to caller
	config = copy.deepcopy(config)
	instance = synchronize_instance_parameters(instance, config)
	instance = synchronize_risk_parameters(instance, config)
	
	if "config" in instance:
		if "alpha_R" in instance["config"]:
			config.alpha_R = instance["config"]["alpha_R"]
		if "alpha_T" in instance["config"]:
			config.alpha_T = instance["config"]["alpha_T"]

	logger.info(f"Loaded instance: {instance['name']}")
	logger.info(
		f"  Nodes: {instance['network']['num_nodes']}, "
		f"Arcs: {instance['network']['num_arcs']}, "
		f"O-D pairs: {len(instance['od_pairs'])}, "
		f"Hub fleet sizes: {instance.get('hub_fleet_size', {})}"
	)

	# Solve
	solver = BranchAndPriceSolver(config, instance, logger)
	solution = solver.solve()
	elapsed_wall, elapsed_cpu = timer.stop()
	solution["end_to_end_wall_time"] = elapsed_wall
	solution["end_to_end_cpu_time"] = elapsed_cpu

	# Report
	report = format_solution_report(solution, instance, total_wall=elapsed_wall, total_cpu=elapsed_cpu)
	logger.info(report)

	return solution, instance


def run_generate(config: RHMVSPConfig, size: str, logger: Logger):
	"""Generate and save an instance in JSON and CSV format within data/{size}/."""
	logger.info(f"Generating {size}-size instance...")

	# Work on a deep copy so instance generation cannot mutate caller settings.
	gen_config = copy.deepcopy(config)
	gen = InstanceGenerator(gen_config, seed=config.seed)
	instance = gen.generate(size=size)

	dir_path = f"data/{size}"
	os.makedirs(dir_path, exist_ok=True)
	
	output_json = os.path.join(dir_path, "instance.json")
	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(instance, f, indent=2, default=str)

	if instance["network"]["nodes"]:
		with open(os.path.join(dir_path, "nodes.csv"), "w", newline="", encoding="utf-8") as f:
			fieldnames = list(instance["network"]["nodes"][0].keys())
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(instance["network"]["nodes"])

	if instance["network"]["arcs"]:
		with open(os.path.join(dir_path, "arcs.csv"), "w", newline="", encoding="utf-8") as f:
			fieldnames = list(instance["network"]["arcs"][0].keys())
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(instance["network"]["arcs"])

	if instance["od_pairs"]:
		with open(os.path.join(dir_path, "od_pairs.csv"), "w", newline="", encoding="utf-8") as f:
			fieldnames = list(instance["od_pairs"][0].keys())
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(instance["od_pairs"])

	if instance.get("railway_timetable"):
		with open(os.path.join(dir_path, "railway_timetable.csv"), "w", newline="", encoding="utf-8") as f:
			fieldnames = list(instance["railway_timetable"][0].keys())
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(instance["railway_timetable"])
		
		logger.info("\n" + "=" * 60)
		logger.info("中国铁路定点货运班列初始计划时刻表 (Initial Timetable)")
		logger.info("=" * 60)
		logger.info(f"{'Train ID':<10} | {'Orig -> Dest':<14} | {'Departure':<15} | {'Arrival':<15} | {'Capacity (Tons)':<15}")
		logger.info("-" * 60)
		for t in instance["railway_timetable"][:10]:
			logger.info(f"{t['train_id']:<10} | Hub {t['origin_hub']} -> Hub {t['destination_hub']:<4} | {t['departure_time_fmt']:<15} | {t['arrival_time_fmt']:<15} | {t['capacity']:<15.1f}")
		if len(instance['railway_timetable']) > 10:
			logger.info(f"... (total {len(instance['railway_timetable'])} scheduled train services)")

	logger.info(f"Instance saved to directory: {dir_path}/ (includes instance.json, nodes.csv, arcs.csv, od_pairs.csv, railway_timetable.csv)")
	logger.info(f"  Nodes: {instance['network']['num_nodes']}")
	logger.info(f"  Arcs: {instance['network']['num_arcs']}")
	logger.info(f"  O-D pairs: {len(instance['od_pairs'])}")
	logger.info(f"  Vehicles: {instance['num_vehicles']}")

	try:
		network = MultimodalNetwork.from_instance(instance)
		viz = Visualizer(gen_config)
		net_path = os.path.join(dir_path, "network.png")
		viz.plot_network(network, net_path)
		logger.info(f"Network topology plot saved to: {net_path} (and .tiff)")
	except Exception as e:
		logger.warning(f"Failed to generate network topology plot: {e}")

	return instance


def main():
	"""Main entry point."""
	args = parse_args()

	# Initialize configuration
	config = RHMVSPConfig()
	config.solver_name = args.solver
	config.time_limit = args.time_limit
	config.seed = args.seed

	# Initialize output directory with timestamp and instance size
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	size_str = args.size if args.mode in ["demo", "generate"] else "custom"
	exp_dir = os.path.join(args.output, f"exp_{timestamp}_{size_str}")
	os.makedirs(exp_dir, exist_ok=True)

	# Save the current configuration to the experiment directory
	config.save_to_json(os.path.join(exp_dir, "config.json"))

	# Initialize logger
	logger = Logger(level=args.log_level, log_file=os.path.join(exp_dir, "rhmvsp.log"))

	# Update args.output to point to the specific experiment directory for downstream use
	args.output = exp_dir

	logger.info("RHMVSP - Reliable Hazardous Materials Multi-modal VSP")
	logger.info(f"Mode: {args.mode} | Solver: {args.solver} | Seed: {args.seed}")
	logger.info(
		f"Config: alpha_R={format_alpha_R(config.alpha_R)}, "
		f"alpha_T={format_alpha_T(config.alpha_T)}, "
		f"T_max={config.T_max}"
	)

	# Dispatch
	instance = None
	solution = None
	if args.mode == "demo":
		solution, instance = run_demo(config, args.size, logger)
	elif args.mode == "solve":
		if args.instance is None:
			logger.error("--instance is required in solve mode")
			sys.exit(1)
		solution, instance = run_solve(config, args.instance, logger)
	elif args.mode == "generate":
		instance = run_generate(config, args.size, logger)

	# Save results
	if solution is not None:
		result_path = os.path.join(args.output, "solution.json")
		with open(result_path, "w") as f:
			json.dump(solution, f, indent=2, default=str)
		logger.info(f"\nSolution saved to: {result_path}")

		# Write summary CSV with cost, sub-costs, reliability and raw values
		csv_path = os.path.join(args.output, "summary.csv")
		sol = solution
		def csv_metric(key, default="N/A"):
			value = sol.get(key)
			return default if value is None else format_csv_value(key, value)

		from sensitivity.utils import format_routing_summary
		route_sum = format_routing_summary(sol, instance)
		cost_cols = ["transport_cost", "transfer_cost", "reposition_cost", "unmet_penalty"]
		benchmark_cols = [
			"solve_wall_time", "master_wall_time", "pricing_wall_time", "other_wall_time",
			"processed_node_count", "total_columns", "root_lp_bound_bp",
			"root_pricing_termination", "root_bound_available",
			"root_gap_vs_bp_incumbent_pct", "gap_bp_pct",
		]
		with open(csv_path, "w", newline="") as csv_f:
			w = csv.writer(csv_f)
			w.writerow(["Status", "Cost"] + cost_cols +
					   ["Network_Risk_Reliability", "Network_Time_Reliability",
						"Network_Risk", "Network_Time"] + benchmark_cols + ["Routes"])
			w.writerow([
				sol.get("status", "unknown"),
				format_csv_value("total_cost", sol.get("total_cost", 0)),
			] + [format_csv_value(c, sol.get(c, 0)) for c in cost_cols] + [
				csv_metric("network_risk_reliability"),
				csv_metric("network_time_reliability"),
				csv_metric("network_risk", "0.00000000"),
				csv_metric("network_time", "0.00"),
			] + [csv_metric(c) for c in benchmark_cols] + [
				route_sum,
			])
		logger.info(f"Summary CSV saved to: {csv_path}")

		# Write one row for every timetable service used by every delivered batch.
		services_path = os.path.join(args.output, "batch_train_services.csv")
		service_fields = [
			"od_id", "batch_id", "service_order", "train_id", "origin", "destination",
			"departure_time_fmt", "arrival_time_fmt",
		]
		with open(services_path, "w", newline="", encoding="utf-8-sig") as services_f:
			writer = csv.DictWriter(services_f, fieldnames=service_fields)
			writer.writeheader()
			writer.writerows(sol.get("batch_train_services", []))
		logger.info(f"Batch train-service CSV saved to: {services_path}")

		# --- Generate Publication-Quality Plots ---
		try:
			viz = Visualizer(config)
			network = MultimodalNetwork.from_instance(instance)
			
			net_path = os.path.join(args.output, "network.png")
			viz.plot_network(network, net_path)
			logger.info(f"Complete network topology plot saved to: {net_path} (and .tiff)")

			plot_path = os.path.join(args.output, "solution_plot.png")
			viz.plot_solution(network, solution, plot_path)
			logger.info(f"High-resolution solution plot saved to: {plot_path} (and .tiff)")
		except Exception as e:
			logger.warning(f"Failed to generate plots: {e}")

	logger.info("\nDone.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
