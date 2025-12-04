# --- coding: utf-8 ---
# --- app/experiment_manager.py ---
import os
import random
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
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
    save_rank0_solutions_csv
)
from app.utils.analyzer import find_knee_point

class Experiment:
    """
    [Controller Layer] 实验总控
    """

    def __init__(self, config_path: str = "config.json", seed: int = 94):
        self.config_path = config_path
        self.seed = seed

        # 1. 创建实验目录
        self.save_dir = create_experiment_directory(base_dir="results")
        setup_logging(log_dir=self.save_dir, log_name="experiment.log")

        # 2. 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 3. 初始化业务对象
        self.network, self.evaluator, self.candidate_paths_map = self._setup_core()

        # 4. 初始化可视化器 (View)
        self.visualizer = NetworkVisualizer(self.network)
        self.plotter = ParetoPlotter(save_dir=self.save_dir)

        # 5. 立即绘制网络拓扑
        self.visualizer.visualize_topology(
            save_dir=self.save_dir,
            filename="network_topology.svg",
            title="Hub-and-Spoke Hazmat Network",
        )

        # 6. 初始化算法
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

        # 路径搜索
        path_finder = PathFinder(network)
        paths_map = path_finder.find_all_candidate_paths()

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

        # 筛选出 Rank 0 用于保存 CSV 和生成路线图 (只需要最优解)
        rank_0 = [s for s in self.final_front if s.rank == 0 and s.is_feasible]
        if not rank_0:
            logging.warning("No feasible Rank 0 solutions.")
            return

        # 保存 Rank 0 的解
        save_rank0_solutions_csv(rank_0, self.save_dir)

        # # 1. 绘制 Pareto 前沿 (传入所有解 self.final_front)
        # # 这样 Plotter 才能画出 Rank 1, Rank 2...
        # self.plotter.plot(self.final_front, file_name="pareto_frontier.svg")

        # # 2. 生成路线对比图 (Figure 2) (只画 Rank 0 的极端解)
        # self._generate_route_maps(rank_0)
        
        # logging.info(f"All results saved to: {self.save_dir}")
        
        # 1. 准备特殊解 (A, B, C)
        sol_a = min(rank_0, key=lambda s: s.f2_cost)  # Min Cost
        sol_b = min(rank_0, key=lambda s: s.f1_risk)  # Min Risk
        sol_c = find_knee_point(rank_0)               # Knee Point
        
        special_solutions = {
            "Opinion A": sol_a,
            "Opinion B": sol_b,
            "Opinion C": sol_c
        }

        # 2. 绘制 Pareto 前沿 (传入 special_solutions 以便高亮)
        self.plotter.plot(
            self.final_front, 
            file_name="pareto_frontier.svg",
            special_solutions=special_solutions
        )

        # 3. 生成对比表格 (Table 1)
        # generate_routing_scheme_comparison(special_solutions, self.evaluator)
        
        # 4. 生成路线地图
        self._generate_route_maps([sol_a, sol_b, sol_c])
        
        logging.info(f"All results saved to: {self.save_dir}")

    def _generate_route_maps(self, solutions: List[Solution]):
        """生成 Min Cost 和 Min Risk 的对比图"""
        # 找极端解
        sol_min_cost = min(solutions, key=lambda s: s.f2_cost)
        sol_min_risk = min(solutions, key=lambda s: s.f1_risk)

        # 生成一致的颜色映射 (Task ID -> Color)
        # 使用 Set1 色板，颜色区分度高
        tasks = [t.task_id for t in self.network.tasks]
        cmap = plt.get_cmap("Set1")
        task_colors = {
            tid: "#{:02x}{:02x}{:02x}".format(
                *map(lambda x: int(x * 255), cmap(i % 9)[:3])
            )
            for i, tid in enumerate(tasks)
        }

        # 绘制图A: 成本优先
        self.visualizer.visualize_routes(
            solution=sol_min_cost,
            task_colors=task_colors,
            save_dir=self.save_dir,
            filename="route_min_cost.svg",
            title=f"Min Cost Strategy (Cost: {sol_min_cost.f2_cost:.0f})",
        )

        # 绘制图B: 风险优先
        self.visualizer.visualize_routes(
            solution=sol_min_risk,
            task_colors=task_colors,
            save_dir=self.save_dir,
            filename="route_min_risk.svg",
            title=f"Min Risk Strategy (Risk: {sol_min_risk.f1_risk:.1f})",
        )
