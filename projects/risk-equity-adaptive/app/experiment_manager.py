# --- coding: utf-8 ---
# --- app/experiment_manager.py ---
import os
import random
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

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
from app.utils.analyzer import find_knee_point, generate_routing_scheme_comparison


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
            title="Hub-and-Spoke Hazmat Network",
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

        # 0. 筛选出 Rank 0 用于保存 CSV 和生成路线图 (只需要最优解)
        rank_0 = [s for s in self.final_front if s.rank == 0 and s.is_feasible]
        if not rank_0:
            logging.warning("No feasible Rank 0 solutions.")
            return

        # 1. 准备特殊解 (A, B, C)
        sol_a = min(rank_0, key=lambda s: s.f2_cost)  # Min Cost
        sol_b = min(rank_0, key=lambda s: s.f1_risk)  # Min Risk
        sol_c = find_knee_point(rank_0)  # Knee Point

        special_solutions = {"Opinion A": sol_a, "Opinion B": sol_b, "Opinion C": sol_c}

        # 2. 绘制 Pareto 前沿 (传入 special_solutions 以便高亮)
        self.plotter.plot(
            self.final_front,
            file_name="pareto_frontier.svg",
            special_solutions=special_solutions,
        )

        # 3. 生成对比表格 (Table 1)
        generate_routing_scheme_comparison(
            special_solutions, self.evaluator, self.save_dir
        )

        # 4. 生成路线地图
        self._generate_route_maps(special_solutions)

        logging.info(f"All results saved to: {self.save_dir}")

    def _generate_route_maps(self, solutions_map: Dict[str, Solution]):
        """
        [Visualizer] 批量生成特殊解的路线地图 (SVG)。
        
        Args:
            solutions_map: 字典 {"Opinion A": sol_a, "Opinion B": sol_b, ...}
        """
        logging.info("Generating route maps for special solutions...")
        
        # 1. 为每个任务生成固定的颜色 (保持视觉一致性)
        # 获取所有任务ID并排序
        task_ids = sorted([t.task_id for t in self.network.tasks])
        cmap = plt.get_cmap("Set1")  # 使用 Set1 配色方案 (颜色鲜明)
        
        task_colors = {}
        for i, tid in enumerate(task_ids):
            # 将 Matplotlib 的 RGBA 转为 Hex 颜色
            rgb = cmap(i % 9)[:3]
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
            )
            task_colors[tid] = hex_color

        # 2. 遍历字典生成地图
        for label, sol in solutions_map.items():
            if not sol:
                continue
            
            # --- 智能生成标题和文件名 ---
            # 把 "Opinion_A/B/C" 用作文件名
            safe_label = label.replace(" ", "_")
            filename = f"route_{safe_label}.svg"
            
            # 生成更有意义的地图标题
            if "A" in label:
                priority = "Cost Priority"
                stats = f"¥{sol.f2_cost:,.0f}"
            elif "B" in label:
                priority = "Risk Priority"
                stats = f"Risk {sol.f1_risk:.1f}"
            else:
                priority = "Balanced Strategy"
                stats = "Knee Point"
                
            map_title = f"{label}: {priority} ({stats})"

            # --- 调用 Visualizer ---
            self.visualizer.visualize_routes(
                solution=sol,
                task_colors=task_colors,
                save_dir=self.save_dir,
                filename=filename,
                title=map_title,
            )
            
        logging.info("Route maps generation completed.")
