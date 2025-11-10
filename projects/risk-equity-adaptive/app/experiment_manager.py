# --- coding: utf-8 ---
# --- app/experiment_manager.py ---
# 封装了所有“实验管理”逻辑
import os
import shutil
import random
import logging
import json
import numpy as np
from typing import List, Dict, Any, Tuple

# --- 核心算法导入 ---
from .core.network import TransportNetwork
from .core.path import PathFinder, Path
from .core.evaluator import Evaluator
from .core.nsga2 import NSGA2
from .core.solution import Solution
from .core.generator import JSONNetworkGenerator

# --- 工具函数导入 ---
from .utils.callback import GenerationalLogger, GenerationalFileLogger
from .utils.analyzer import (
    plot_risk_histogram,
    plot_evolution,
    analyze_operator_contribution,
    perform_local_sensitivity_analysis,
    print_solution_details
)
from .utils.plotter import (
    ParetoPlotter,
    plot_parallel_coordinates
)
from .utils.result_keeper import (
    create_experiment_directory,
    setup_logging,
    log_section,
    save_results_json,
    save_solutions_csv
)
from .utils.visualizer import visualize_network


class Experiment:
    """
    一个“实验管理”类。

    它封装了一次完整实验运行所需的所有步骤：
    1. __init__: 设置 (日志, 配置, 备份, 预计算)
    2. run: 运行核心算法
    3. analyze_and_report: 分析和保存结果
    """

    def __init__(self, config_path: str = "config.json", seed: int = 42):
        """
        初始化实验环境。
        这将自动设置日志、创建文件夹、加载配置、备份文件，
        并运行所有预计算（例如路径搜索和应急时间）。
        """
        self.config_path = config_path
        self.seed = seed
        
        # --- 步骤 1: 配置与日志 (来自 main.py) ---
        self.save_dir = create_experiment_directory(base_dir="results")
        setup_logging(log_dir=self.save_dir, log_name="experiment.log")
        
        logging.info("--- [1/5] 正在加载配置... ---")
        self.config = self.load_config(self.config_path)
        
        # 1a. 备份实验文件
        self._backup_experiment_files()

        # --- 步骤 2: 准备实验 (来自 setup_experiment) ---
        logging.info("--- [2/5] 正在准备实验环境... ---")
        self.network, self.evaluator, self.candidate_paths_map = self.setup()
        
        # (可选) 可视化网络
        self._visualize_network()

        # --- 步骤 3: 配置算法 (来自 run_and_analyze_experiment) ---
        logging.info("--- [3/5] 正在配置算法... ---")
        self.algorithm = NSGA2(self.network, self.evaluator, self.candidate_paths_map, self.config)
        
        # 准备好装载结果
        self.final_pareto_front: List[Solution] = []
        self.generational_logs: Dict[str, List[Any]] = {}

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        从 JSON 文件加载实验配置。 (来自 load_config)
        """
        if not os.path.exists(config_file):
            logging.error(f"致命错误: 找不到配置文件 {config_file}!")
            raise FileNotFoundError(f"配置文件 {config_file} 不存在。")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"已从 {config_file} 成功加载配置。")
        return config

    def _backup_experiment_files(self):
        """
        备份 config.json 和 data 文件夹。 (来自 main.py)
        """
        try:
            shutil.copy(self.config_path, os.path.join(self.save_dir, "config_snapshot.json"))
            logging.info(f"已将 {self.config_path} 备份到结果文件夹。")
        except Exception as e:
            logging.warning(f"备份 {self.config_path} 失败: {e}")
            
        try:
            data_dir = self.config.get("experiment", {}).get("data_dir", "data")
            shutil.copytree(data_dir, os.path.join(self.save_dir, "data_snapshot"))
            logging.info(f"已将 {data_dir} 文件夹备份到结果文件夹。")
        except Exception as e:
            logging.warning(f"备份 {data_dir} 文件夹失败: {e}")

    def setup(self) -> Tuple[TransportNetwork, Evaluator, Dict[str, List[Path]]]:
        """
        执行所有“运行前”的准备工作。 (来自 setup_experiment)
        """
        exp_config = self.config["experiment"]
        
        # 1. 设置随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        logging.info(f"随机种子已设置为: {self.seed}")

        # 2. 加载网络
        logging.info("--- 正在加载运输网络... ---")
        try:
            data_path = exp_config["data_dir"]
            generator = JSONNetworkGenerator(
                nodes_file_path=os.path.join(data_path, exp_config["nodes_file"]),
                arcs_file_path=os.path.join(data_path, exp_config["arcs_file"]),
                tasks_file_path=os.path.join(data_path, exp_config["tasks_file"])
            )
            network = generator.generate()
        except FileNotFoundError as e:
            logging.error(f"网络加载失败: 找不到文件 {e.filename}")
            raise
        
        network.summary()
        
        # 3. (预计算) 搜索候选路径
        logging.info("--- (预计算) 正在搜索候选路径... ---")
        path_finder = PathFinder(network)
        candidate_paths_map = path_finder.find_all_candidate_paths()
        
        # 检查路径
        has_warning = False
        for task in network.tasks:
            paths = candidate_paths_map.get(task.task_id, [])
            if not paths:
                logging.warning(f"任务 {task.task_id} ( {task.origin.id} -> {task.destination.id} ) 找不到任何合法路径!")
                has_warning = True
            else:
                logging.info(f"     - 任务 {task.task_id}: 找到了 {len(paths)} 条候选路径。")
        if has_warning:
            logging.error("一个或多个任务没有候选路径，算法无法运行。")
            raise RuntimeError("路径搜索失败，一个或多个任务没有可用路径。")

        # 4. (预计算) 初始化评估器
        # 它会自动调用 _precompute_emergency_response_times
        evaluator = Evaluator(network, self.config)
        
        return network, evaluator, candidate_paths_map

    def _visualize_network(self):
        """
        (可选) 可视化网络拓扑。 (来自 main.py)
        """
        try:
            visualize_network(self.network, save_path=os.path.join(self.save_dir, "network_topology.png"))
            logging.info("网络拓扑图已保存。")
        except Exception as e:
            logging.warning(f"网络可视化失败: {e}")

    def run(self):
        """
        执行“运行中”的任务：运行 NSGA-II 算法。
        来自 run_and_analyze_experiment 的第一部分
        """
        logging.info("--- [4/5] 核心算法开始运行... ---")

        # 设置回调
        generational_logger = GenerationalLogger()
        file_logger = GenerationalFileLogger(self.save_dir)
        
        try:
            self.final_pareto_front = self.algorithm.run(callbacks=[generational_logger, file_logger])
            self.generational_logs = generational_logger.logs   # 保存内存日志用于绘图
        except Exception as e:
            logging.error(f"\nNSGA2算法运行时发生致命错误: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.final_pareto_front = [] # 确保为空

    def analyze_and_report(self):
        """
        执行“运行后”的任务：分析、绘图和保存。
        来自 run_and_analyze_experiment 的第二部分
        """
        logging.info("--- [5/5] 正在分析并可视化结果... ---")
        
        if not self.final_pareto_front:
            logging.warning("未找到任何非支配解，无法进行后续分析。")
            return

        # --- 原始数据
        save_results_json(self.final_pareto_front, os.path.join(self.save_dir, "final_archive.json"))
        
        # --- 提取 Rank 0 解
        rank_0_solutions = [s for s in self.final_pareto_front if s.rank == 0 and s.is_feasible]
        if rank_0_solutions:
            save_solutions_csv(rank_0_solutions, self.save_dir)
        else:
            logging.warning("未找到任何 Rank 0 可行解，跳过 'PF_solutions.csv' 的生成。")

        # --- 帕累托图
        plotter = ParetoPlotter(title="Pareto Frontier", save_dir=self.save_dir)
        plotter.plot(
            solutions=self.final_pareto_front,
            file_name="pareto_frontier.png",
            xlabel="Total Risk (f1)",
            ylabel="Total Cost (f2)"
        )

        # --- 平行坐标图
        if rank_0_solutions:
            plot_parallel_coordinates(rank_0_solutions, self.save_dir)
        else:
            logging.warning("未找到任何 Rank 0 可行解，跳过“平行坐标图”的绘制。")
        
        # --- 收敛图
        try:
            plot_evolution(self.generational_logs, self.save_dir)
        except Exception as e:
            logging.error(f"绘制“进化图”失败: {e}")

        # --- (附加分析) 打印极端解 ---
        try:
            with log_section(clean=True):
                if rank_0_solutions:
                    lowest_risk_sol = min(rank_0_solutions, key=lambda s: s.f1_risk)
                    print_solution_details("Optimum (Lowest Risk)", lowest_risk_sol)
                    
                    lowest_cost_sol = min(rank_0_solutions, key=lambda s: s.f2_cost)
                    print_solution_details("Optimum (Lowest Cost)", lowest_cost_sol)
                else:
                    logging.warning("未找到可行的 Rank 0 解，无法打印解详情。")
        except Exception as e:
            logging.error(f"解读最优解时出错: {e}")

        # --- (附加分析) 运行 analyzer.py 里的其他函数 ---
        logging.info("--- 正在进行附加分析 ---")
        with log_section(clean=True):
            plot_risk_histogram(self.final_pareto_front, save_dir=self.save_dir, file_name="final_front_risk_histogram.png")
            
            analyze_operator_contribution(self.algorithm.operator_log)

            try:
                analysis_config = self.config["analysis"]
                perform_local_sensitivity_analysis(
                    final_front=self.final_pareto_front,
                    evaluator=self.evaluator,
                    candidate_paths_map=self.candidate_paths_map,
                    save_dir=self.save_dir,
                    n_solutions_to_test=analysis_config.get("sensitivity_solutions", 5),
                    n_trials_per_solution=analysis_config.get("sensitivity_trials", 20)
                )
            except Exception as e:
                logging.error(f"运行“灵敏度分析”失败: {e}")

        logging.info("--- 分析与报告完成 ---")