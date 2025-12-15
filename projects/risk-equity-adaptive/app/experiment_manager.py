# --- coding: utf-8 ---
# --- app/experiment_manager.py ---
import os
import random
import logging
import json
import numpy as np
from typing import List

# --- Core Modules ---
from app.core.path import PathFinder
from app.core.evaluator import Evaluator
from app.core.nsga2 import NSGA2
from app.core.solution import Solution
from app.core.generator import JSONNetworkGenerator

# --- Utils ---
from app.utils.callback import GenerationalLogger, GenerationalFileLogger
from app.utils.plotter import ParetoPlotter
from app.utils.network_visualizer import NetworkVisualizer
from app.utils.result_keeper import (
    create_experiment_directory,
    setup_logging,
)
from app.utils.analyzer import (
    find_knee_point,
    generate_routing_scheme_comparison,
    calculate_solution_gini,
)


class Experiment:
    """
    [Controller Layer] 实验总控
    """

    def __init__(self, config_path: str = "config.json", seed: int = None):
        # 1. 加载实验配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 2. 创建实验目录
        self.base_dir = self.config.get("experiment", {}).get("base_dir", "results")
        self.save_dir = create_experiment_directory(base_dir=self.base_dir)

        # 2. 设置日志系统
        setup_logging(log_dir=self.save_dir, log_name="experiment.log")

        # 3. 设置全局随机种子
        if seed is not None:
            self.seed = seed
            logging.info(f"Global Random Seed using manual override seed: {self.seed}")
        else:
            self.seed = self.config.get("experiment", {}).get("seed", 4)
            logging.info(f"Global Random Seed using configuration seed: {self.seed}")

        # 4. 初始化业务对象
        self.network, self.evaluator, self.candidate_paths_map = self._setup_core()

        # 5. 初始化可视化器 (View)
        self.visualizer = NetworkVisualizer(self.network)
        self.plotter = ParetoPlotter(save_dir=self.save_dir)

        # 6. 立即绘制网络拓扑
        self.visualizer.visualize_topology(
            save_dir=self.save_dir,
            filename="network_topology.svg",
        )

        # 7. 初始化算法
        self.algorithm = NSGA2(
            self.network, self.evaluator, self.candidate_paths_map, self.config
        )
        self.final_front: List[Solution] = []

    def _setup_core(self):
        # 设置随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)

        # 加载数据
        cfg = self.config["experiment"]
        generator = JSONNetworkGenerator(
            nodes_file_path=os.path.join(cfg["data_dir"], cfg["nodes_file"]),
            arcs_file_path=os.path.join(cfg["data_dir"], cfg["arcs_file"]),
            tasks_file_path=os.path.join(cfg["data_dir"], cfg["tasks_file"]),
        )
        network = generator.generate()
        network.summary()

        # 引入 time 模块
        import time

        start_time = time.process_time()

        # 路径搜索
        path_finder = PathFinder(network)
        paths_map = path_finder.find_all_candidate_paths()

        end_time = time.process_time()

        # 保存预计算时间
        self.precompute_time = end_time - start_time
        logging.info(f"Path Pre-computation Time: {self.precompute_time:.4f}s")

        evaluator = Evaluator(network, self.config)
        return network, evaluator, paths_map

    def run(self):
        logging.info("--- Starting Optimization ---")
        gen_logger = GenerationalLogger()
        file_logger = GenerationalFileLogger(self.save_dir)

        try:
            self.final_front = self.algorithm.run(callbacks=[gen_logger, file_logger])
        except Exception as e:
            logging.error(f"Optimization failed: {e}", exc_info=True)

    def analyze_and_report(self):
        """
        执行运行后的任务：分析、绘图和保存
        """
        logging.info("--- Generating Report & Charts ---")
        if not self.final_front:
            logging.warning("No solutions found.")
            return

        # 0. 筛选出 Rank 0 用于保存 CSV 和生成路线图 (只需要最优解)
        rank_0 = [s for s in self.final_front if s.rank == 0 and s.is_feasible]
        if not rank_0:
            logging.warning("No feasible Rank 0 solutions.")
            return

        # 1. 筛选出所有具有 Gini value 的可行解 List of (Solution, Gini_Value)
        solutions_with_gini = []

        for sol in self.final_front:  # 对所有可行解计算，确保 Gini Trade-off 散点图完整
            if sol.is_feasible:
                # 确保 solution.py 中已添加 self.gini_coefficient 属性
                sol.gini_coefficient = calculate_solution_gini(sol, self.evaluator)

            # 仅将可行解纳入绘图数据
            if sol.is_feasible:
                solutions_with_gini.append((sol, sol.gini_coefficient))

        # 2. 准备特殊解 (A, B, C)
        sol_a = min(rank_0, key=lambda s: s.f2_cost)  # Min Cost
        sol_b = min(rank_0, key=lambda s: s.f1_risk)  # Min Risk
        sol_c = find_knee_point(rank_0)  # Knee Point

        special_solutions = {"Opinion A": sol_a, "Opinion B": sol_b, "Opinion C": sol_c}

        # 3. 绘制 Pareto 前沿 (传入 special_solutions 以便高亮)
        self.plotter.plot(
            self.final_front,
            file_name="pareto_frontier.svg",
            special_solutions=special_solutions,
        )

        # 4.1 绘制 Gini Trade-off 图
        # 需要所有可行解数据，以便绘制完整的散点图和 Rank 0 前沿
        self.plotter.plot_gini_tradeoff(
            solutions_with_gini, file_name="Figure_Gini_Risk_Cost_Tradeoff.svg"
        )

        # 4.2 绘制平行坐标图
        # 只传入 Rank 0 可行解
        rank_0_solutions = [s for s, g in solutions_with_gini if s.rank == 0]
        self.plotter.plot_parallel_coordinates(
            rank_0_solutions,
            gini_calculator=calculate_solution_gini,
            evaluator=self.evaluator,
            file_name="Figure_Parallel_Coordinates_Rank0.svg",
        )

        # 5. 生成对比表格
        generate_routing_scheme_comparison(
            special_solutions, self.evaluator, self.save_dir
        )

        # 6. 生成路线地图
        self.visualizer.visualize_routes(special_solutions, self.save_dir)

        logging.info(f"All results saved to: {self.save_dir}")
