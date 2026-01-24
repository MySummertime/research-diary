# --- coding: utf-8 ---
# --- app/experiment_manager.py ---
import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np

from app.core.evaluator import Evaluator
from app.core.network_generator import JSONNetworkGenerator
from app.core.nsga2 import NSGA2

# --- Core Modules ---
from app.core.path import PathFinder
from app.core.solution import Solution
from app.utils.analyzer import (
    calculate_solution_gini,
    generate_routing_scheme_comparison,
)

# --- Utils ---
from app.utils.callback import GenerationalFileLogger, GenerationalLogger
from app.utils.network_visualizer import NetworkVisualizer
from app.utils.plotter import ParetoPlotter
from app.utils.result_keeper import (
    create_experiment_directory,
    setup_logging,
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

        # 3. 设置日志系统
        setup_logging(log_dir=self.save_dir, log_name="experiment.log")

        # 4. 设置全局随机种子
        if seed is not None:
            self.seed = seed
            logging.info(f"Global Random Seed using manual override seed: {self.seed}")
        else:
            self.seed = self.config.get("experiment", {}).get("seed", 4)
            logging.info(f"Global Random Seed using configuration seed: {self.seed}")

        # 5. 初始化业务对象
        self.network, self.evaluator, self.candidate_paths_map = self._setup_core()

        # 6. 初始化可视化器 (View)
        self.visualizer = NetworkVisualizer(self.network)
        self.plotter = ParetoPlotter(save_dir=self.save_dir)

        # 7. 绘制网络拓扑
        self.visualizer.visualize_topology(
            save_dir=self.save_dir,
            filename="network_topology.svg",
        )

        # 8. 初始化算法
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

        # 初始化 Evaluator
        # PathFinder 的多准则搜索（特别是 fast_response 策略）依赖 risk_model 的预计算结果
        evaluator = Evaluator(network, self.config)

        # 引入 time 模块
        import time

        start_time = time.process_time()

        # 将 evaluator 传入 PathFinder
        path_finder = PathFinder(network, evaluator)
        paths_map = path_finder.find_all_candidate_paths()

        end_time = time.process_time()

        # 保存预计算时间
        self.precompute_time = end_time - start_time
        logging.info(f"Path Pre-computation Time: {self.precompute_time:.4f}s")

        # 返回顺序保持不变，但内部逻辑已调整
        return network, evaluator, paths_map

    def run(self):
        logging.info("🚀 Starting NSGA-II Optimization...")
        gen_logger = GenerationalLogger()
        file_logger = GenerationalFileLogger(self.save_dir)

        try:
            self.final_front = self.algorithm.run(callbacks=[gen_logger, file_logger])
        except Exception as e:
            logging.error(f"Optimization failed: {e}", exc_info=True)

    def _export_gini_details_csv(
        self,
        solutions_with_gini: List[Tuple[Solution, float]],
        special_solutions: Dict[str, Solution],
    ):
        """
        [Internal Helper] 输出 Pareto 前沿每个解的 Gini 系数到 CSV 文件。
        """
        filename = "gini_coefficient_details.csv"
        file_path = os.path.join(self.save_dir, filename)

        # 建立特殊解的反向索引 (ID -> Label) 以便标注 Opinion A/B/C
        special_map = {id(sol): label for label, sol in special_solutions.items()}

        headers = [
            "Solution_ID",
            "Total_Risk",
            "Total_Cost",
            "Gini_Coefficient",
            "Opinion_Tag",
        ]

        import csv

        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

                for sol, gini in solutions_with_gini:
                    # 只导出非支配解 (Pareto Front) 的数据
                    if sol.rank == 0:
                        tag = special_map.get(id(sol), "")
                        writer.writerow(
                            [
                                id(sol),
                                f"{sol.f1_risk:.4f}",
                                f"{sol.f2_cost:.4f}",
                                f"{gini:.6f}",
                                tag,
                            ]
                        )
            logging.info(
                f"Gini coefficient details successfully exported to: {filename} ⚖️"
            )
        except Exception as e:
            logging.error(f"Failed to export Gini CSV: {e}")

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
        sol_min_risk = min(rank_0, key=lambda s: s.f1_risk)  # Min Risk
        sol_min_cost = min(rank_0, key=lambda s: s.f2_cost)  # Min Cost
        sol_min_gini = min(rank_0, key=lambda s: s.gini_coefficient)  # Min Gini

        special_solutions = {
            "Opinion A(Min Risk)": sol_min_risk,
            "Opinion B(Min Gini)": sol_min_gini,
            "Opinion C(Min Cost)": sol_min_cost,
        }

        # 3. 绘制 Pareto 前沿 (传入 special_solutions 以便高亮)
        self.plotter.plot(
            self.final_front,
            file_name="pareto_frontier.svg",
            special_solutions=special_solutions,
        )

        # 4. 导出 Gini coefficient 细节为 csv 文件
        self._export_gini_details_csv(solutions_with_gini, special_solutions)

        # 4.1 绘制 Gini Trade-off 图
        # 需要所有可行解数据，以便绘制完整的散点图和 Rank 0 前沿
        # 传入特殊解以便高亮
        self.plotter.plot_gini_tradeoff(
            solutions_with_gini,
            file_name="Figure_Gini_Risk_Cost_Tradeoff.svg",
            special_solutions=special_solutions,
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
