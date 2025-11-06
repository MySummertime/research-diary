# --- coding: utf-8 ---
# --- main.py ---
import os
import time
import random
import logging
import json 
import numpy as np
from typing import List, Dict, Any, Tuple
from app.core.network import TransportNetwork
from app.core.path import PathFinder, Path
from app.core.evaluator import Evaluator
from app.core.nsga2 import NSGA2
from app.core.solution import Solution
from app.core.generator import JSONNetworkGenerator
from app.utils.callback import GenerationalLogger, GenerationalFileLogger
from app.utils.analyzer import (
    plot_risk_histogram, 
    plot_evolution, 
    analyze_operator_contribution, 
    perform_local_sensitivity_analysis,
    print_solution_details
)
from app.utils.plotter import ParetoPlotter 
from app.utils.result_keeper import (
    create_experiment_directory, 
    setup_logging, 
    log_section, 
    save_results_json
)
from app.utils.visualizer import visualize_network

# ----------------------------------------
# 1. 加载配置
# ----------------------------------------
def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """
    从 JSON 文件加载实验配置。
    """
    if not os.path.exists(config_file):
        logging.error(f"致命错误: 找不到配置文件 {config_file}!")
        raise FileNotFoundError(f"配置文件 {config_file} 不存在。")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logging.info(f"已从 {config_file} 成功加载配置。")
    return config

# ----------------------------------------
# 2. 准备实验
# ----------------------------------------
def setup_experiment(config: Dict[str, Any]) -> Tuple[TransportNetwork, PathFinder, Evaluator, Dict[str, List[Path]]]:
    """
    执行所有“运行前”的准备工作：
    从 config["experiment"] 中读取。
    """
    
    # 从 "experiment" 分组读取
    exp_config = config["experiment"] 
    
    # 1. 设置随机种子
    random.seed(exp_config["seed"])
    np.random.seed(exp_config["seed"])
    logging.info(f"随机种子已设置为: {exp_config['seed']}")

    # 2. 加载网络
    logging.info("--- [2/5] 正在加载运输网络... ---")
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
        logging.error("请确保 'data' 目录及 'nodes.json', 'arcs.json', 'tasks.json' 文件存在。")
        raise
    except Exception as e:
        logging.error(f"网络加载失败: {e}")
        raise 
        
    network.summary()
    
    # 3. (预计算) 搜索候选路径
    logging.info("--- [3/5] (预计算) 正在搜索候选路径... ---")
    path_finder = PathFinder(network)
    candidate_paths_map = path_finder.find_all_candidate_paths()
    
    has_warning = False
    for task in network.tasks:
        paths = candidate_paths_map.get(task.task_id, [])
        if not paths:
            logging.warning(f"!!! 任务 {task.task_id} ( {task.origin.id} -> {task.destination.id} ) 找不到任何合法路径!")
            has_warning = True
        else:
            logging.info(f"     - 任务 {task.task_id}: 找到了 {len(paths)} 条候选路径。")
    if has_warning:
        logging.error("一个或多个任务没有候选路径，算法无法运行。")
        raise RuntimeError("路径搜索失败，一个或多个任务没有可用路径。")

    # 4. (预计算) 初始化评估器
    # (传递 *整个* config, Evaluator 内部将负责访问 config["risk_model_f1"] 等)
    evaluator = Evaluator(network, config) 
    
    return network, path_finder, evaluator, candidate_paths_map

# ----------------------------------------
# 3. 运行并分析
# ----------------------------------------
def run_and_analyze_experiment(
    network: TransportNetwork, 
    evaluator: Evaluator,
    candidate_paths_map: Dict[str, List[Path]],
    config: Dict[str, Any],
    save_dir: str):
    """
    执行所有“运行中”和“运行后”的任务：
    (注意: NSGA2 内部将负责访问 config["algorithm"])
    """
    logging.info("--- [4/5] 正在配置并运行 NSGA-II 算法... ---")
    algorithm = NSGA2(network, evaluator, candidate_paths_map, config)
    
    generational_logger = GenerationalLogger() 
    file_logger = GenerationalFileLogger(save_dir) 
    
    logging.info("核心算法开始运行...")
    
    try:
        final_pareto_front = algorithm.run(callbacks=[generational_logger, file_logger])
    except Exception as e:
        logging.error(f"\n !!! 算法运行时发生致命错误: {e} !!!")
        import traceback
        logging.error(traceback.format_exc())
        return

    logging.info("--- [5/5] 正在分析并可视化结果... ---")
    if not final_pareto_front:
        logging.warning("未找到任何非支配解，无法进行后续分析。")
        return
    
    # 1. 保存 JSON
    save_results_json(final_pareto_front, os.path.join(save_dir, "final_pareto_front.json"))
    
    # 2. 绘制 Pareto 图
    plotter = ParetoPlotter(title="Pareto Frontier", save_dir=save_dir)
    plotter.plot(
        solutions=final_pareto_front, 
        file_name="pareto_frontier.png", 
        xlabel="Total Risk (f1)",
        ylabel="Total Cost (f2)"
    )

    # 3. 打印极端解
    try:
        with log_section(clean=True):
            feasible_solutions = [s for s in final_pareto_front if s.is_feasible]
            if feasible_solutions:
                lowest_risk_sol = min(feasible_solutions, key=lambda s: s.f1_risk)
                print_solution_details("Optimum (Lowest Risk)", lowest_risk_sol)
                
                lowest_cost_sol = min(feasible_solutions, key=lambda s: s.f2_cost)
                print_solution_details("Optimum (Lowest Cost)", lowest_cost_sol)
            else:
                logging.warning("未找到可行的最优解，无法打印解详情。")
    except Exception as e:
        logging.error(f"解读最优解时出错: {e}")

    # 4. 运行分析
    logging.info("--- 正在进行附加分析 ---")
    with log_section(clean=True):
        run_full_analysis(
            final_solutions=final_pareto_front, 
            logger=generational_logger, 
            save_dir=save_dir,
            operator_log=algorithm.operator_log,
            evaluator=evaluator, 
            candidate_paths_map=candidate_paths_map, 
            config=config   # 传递整个 config
        )

def run_full_analysis(
    final_solutions: List[Solution], 
    logger: GenerationalLogger, 
    save_dir: str,
    operator_log: Dict[str, int],
    evaluator: Evaluator,
    candidate_paths_map: Dict[str, List[Path]], 
    config: Dict[str, Any] 
    ):
    """
    从 config["analysis"] 中读取。
    """
    
    # 从 "analysis" 分组读取
    analysis_config = config["analysis"]

    # 1. 风险直方图
    plot_risk_histogram(final_solutions, save_dir=save_dir, file_name="final_front_risk_histogram.png")

    # 2. 进化曲线 (3 张图)
    plot_evolution(logger.logs, save_dir=save_dir, file_name_prefix="full")
    
    # 3. 算子贡献
    analyze_operator_contribution(operator_log)

    # 4. 灵敏度分析
    perform_local_sensitivity_analysis(
        final_front=final_solutions,
        evaluator=evaluator,
        candidate_paths_map=candidate_paths_map, 
        save_dir=save_dir,
        # 读取分组
        n_solutions_to_test=analysis_config.get("sensitivity_solutions", 5),
        n_trials_per_solution=analysis_config.get("sensitivity_trials", 20)
    )

# ----------------------------------------
# 4. 主函数 (总控制器)
# ----------------------------------------
def main():
    """
    项目主执行函数 (V3 总控制器)。
    """
    start_time = time.time()
    
    try:
        # === 步骤 1: 配置与日志 ===
        save_dir = create_experiment_directory(base_dir="results")
        setup_logging(log_dir=save_dir, log_name="experiment_log.txt")
        
        logging.info("--- [1/5] 正在加载配置... ---")
        config = load_config("config.json") # 加载嵌套字典
        
        # === 步骤 2: 准备实验 (加载网络, 搜索路径, 预计算) ===
        network, path_finder, evaluator, candidate_paths_map = setup_experiment(config)
        
        # (可选) 可视化网络拓扑
        try:
            visualize_network(network, save_path=os.path.join(save_dir, "network_topology.png"))
            logging.info("网络拓扑图已保存。")
        except Exception as e:
            logging.warning(f"网络可视化失败: {e}")

        # === 步骤 3: 运行算法并分析结果 ===
        run_and_analyze_experiment(
            network, 
            evaluator, 
            candidate_paths_map, 
            config, # 传递嵌套字典
            save_dir
        )
        
    except Exception as e:
        logging.error("\n--- !!! 实验主流程发生致命错误 !!! ---")
        logging.error(f"错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    finally:
        # 无论成功还是失败，都报告总时间
        end_time = time.time()
        total_time = end_time - start_time
        logging.info("==========================================")
        logging.info(f"实验总耗时: {total_time:.2f} 秒。")
        logging.info("==========================================")

if __name__ == '__main__':
    main()