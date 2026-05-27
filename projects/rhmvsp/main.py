# RHMVSP - Reliable Hazardous Materials Multi-modal Vehicle Scheduling Problem
# Single entry point: main.py
# Usage: python main.py [--instance FILE] [--mode solve|generate|demo] [--solver pulp|gurobi]

import argparse
import csv
import json
import os
import sys
from datetime import datetime

from config import RHMVSPConfig
from src.core.data_generator import InstanceGenerator
from src.core.network import MultimodalNetwork
from src.solvers.branch_and_price import BranchAndPriceController as BranchAndPriceSolver
from src.utils.logging import Logger
from src.utils.plotting import Visualizer
from src.utils.reporting import format_solution_report
from src.utils.timing import Timer


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
		default=config_default.solver_backend,
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

	logger.info(f"Instance: {instance['name']}")
	logger.info(
		f"  Nodes: {instance['network']['num_nodes']}, "
		f"Arcs: {instance['network']['num_arcs']}, "
		f"O-D pairs: {len(instance['od_pairs'])}, "
		f"Vehicles: {instance['num_vehicles']}"
	)

	# Display network summary
	network = MultimodalNetwork.from_instance(instance)
	logger.info("\nNetwork Summary:")
	logger.info(f"  Road arcs: {len(network.get_road_arcs())}")
	logger.info(f"  Rail arcs: {len(network.get_rail_arcs())}")
	logger.info(f"  Transfer nodes: {len(network.get_transfer_nodes())}")

	# Display uncertainty parameters
	logger.info("\nUncertainty Parameters:")
	logger.info(f"  Alpha_max (arc risk threshold): {config.alpha_max}")
	logger.info(f"  Alpha_T (time reliability threshold): {config.alpha_T}")
	logger.info(f"  Gamma_sys (comprehensive reliability): {config.gamma_sys}")

	# Solve with Branch-and-Price
	logger.info("\n" + "=" * 60)
	logger.info("Starting Branch-and-Price Solver")
	logger.info("=" * 60)

	solver = BranchAndPriceSolver(config, instance, logger)
	solution = solver.solve()
	elapsed_wall, elapsed_cpu = timer.stop()

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
	logger.info(f"Loaded instance: {instance['name']}")
	logger.info(
		f"  Nodes: {instance['network']['num_nodes']}, "
		f"Arcs: {instance['network']['num_arcs']}, "
		f"O-D pairs: {len(instance['od_pairs'])}, "
		f"Vehicles: {instance['num_vehicles']}"
	)

	# Solve
	solver = BranchAndPriceSolver(config, instance, logger)
	solution = solver.solve()
	elapsed_wall, elapsed_cpu = timer.stop()

	# Report
	report = format_solution_report(solution, instance, total_wall=elapsed_wall, total_cpu=elapsed_cpu)
	logger.info(report)

	return solution, instance


def run_generate(config: RHMVSPConfig, size: str, logger: Logger):
	"""Generate and save an instance in JSON and CSV format within data/{size}/."""
	logger.info(f"Generating {size}-size instance...")

	gen = InstanceGenerator(config, seed=config.seed)
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
		viz = Visualizer(config)
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
	config.solver_backend = args.solver
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
		f"Config: alpha_max={config.alpha_max}, alpha_T={config.alpha_T}, "
		f"K_max={config.K_max}, T_max={config.T_max}"
	)

	# Dispatch
	instance = None
	if args.mode == "demo":
		solution, instance = run_demo(config, args.size, logger)
	elif args.mode == "solve":
		if args.instance is None:
			logger.error("--instance is required in solve mode")
			sys.exit(1)
		solution, instance = run_solve(config, args.instance, logger)
	elif args.mode == "generate":
		instance = run_generate(config, args.size, logger)
		solution = None

	# Save results
	if solution is not None:
		result_path = os.path.join(args.output, "solution.json")
		with open(result_path, "w") as f:
			json.dump(solution, f, indent=2, default=str)
		logger.info(f"\nSolution saved to: {result_path}")

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
