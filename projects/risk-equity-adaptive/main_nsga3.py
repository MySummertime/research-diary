# --- coding: utf-8 ---
# --- main_nsga3.py ---
import os
import time
import logging
import numpy as np
from contextlib import redirect_stdout

from pymoo.algorithms.moo.nsga3 import NSGA3    # type: ignore
from pymoo.operators.sampling.rnd import IntegerRandomSampling  # type: ignore
# from pymoo.operators.crossover.sbx import SBX   # type: ignore
# === 引入为整数/组合问题设计的算子 ===
from pymoo.operators.crossover.ux import UniformCrossover   # type: ignore
# ===============================================
# from pymoo.operators.mutation.pm import PM  # type: ignore
# === 引入增加变异幅度的算子，强行增加种群多样性 ===
from pymoo.operators.mutation.bitflip import BitflipMutation    # type: ignore
# ===============================================
from pymoo.util.ref_dirs import get_reference_directions    # type: ignore
from pymoo.optimize import minimize # type: ignore
from pymoo.termination import get_termination   # type: ignore

from utils.network_generator import HaSConfig, HaSNetworkGenerator
from utils.plotter import ParetoPlotter
from utils.result_keeper import create_experiment_directory, setup_logging, log_section
from core.network import TransportNetwork
from core.problem import HazmatProblem
from core.callback import GenerationalLogger
from core.analyzer import plot_evolution, plot_risk_histogram, perform_local_sensitivity_analysis, analyze_operator_contribution
from core.operators import LoggingCrossover, LoggingMutation

def print_solution_details(solution_type: str, objectives: np.ndarray, solution_x: np.ndarray, problem: HazmatProblem):
    """
    [辅助函数] 打印单个最优解的详细信息。
    """
    logging.info(f"\n--- {solution_type} 解读 ---")
    logging.info(f"目标值: SP-CVaR={objectives[0]:.2f}, Cost={objectives[1]:.2f}")
    logging.info("该解对应的枢纽选择策略为:")
    
    # 将一维的决策变量数组，变回 V x 2 的形状
    solution_hubs = solution_x.reshape((len(problem.tasks), 2))
    
    for i, task in enumerate(problem.tasks):
        hub_indices_float = solution_hubs[i]
        # --- 将浮点数安全地转换为整数索引 ---
        # 使用四舍五入找到最接近的整数，然后转换为 int 类型
        hub_indices = [int(round(idx)) for idx in hub_indices_float]
        
        # 使用 problem.hubs 来获取枢纽名称
        hub_names = [problem.hubs[idx].id for idx in hub_indices]
        
        # 打印枢纽选择策略
        if hub_names[0] == hub_names[1]:
            logging.info(f"  - 任务 {task.id} (从 {task.origin.id} 到 {task.destination.id}): 选择单枢纽 {hub_names[0]}")
        else:
            logging.info(f"  - 任务 {task.id} (从 {task.origin.id} 到 {task.destination.id}): 选择双枢纽 {hub_names[0]} -> {hub_names[1]}")
        
        # --- 调用解码器，打印详细路径 ---
        # 将原始的浮点数元组传递给 _decode_path，因为它内部会处理转换
        _, detailed_path_nodes = problem._decode_path(task, tuple(hub_indices_float))
        
        # 将节点列表转换为一个易于阅读的字符串
        if detailed_path_nodes:
            path_str = " -> ".join([node.id for node in detailed_path_nodes])
            logging.info(f"      路径详情: {path_str}")
        else:
            logging.warning("      路径详情: 未能生成有效路径")

def run_analysis(res, problem: 'HazmatProblem', logger: 'GenerationalLogger', save_dir: str):
    """
    [辅助函数] 在执行完毕后，进行所有的算法性能分析步骤。
    """
    # 1. 绘制最终种群的风险直方图
    if res.pop is not None:
        plot_risk_histogram(res.pop.get("F"), save_dir=save_dir)

    # 2. 调用分析函数绘制代际演化图
    plot_evolution(logger.logs, save_dir=save_dir, file_name_prefix="evolution")
    
    # 3. 执行局部灵敏度分析
    perform_local_sensitivity_analysis(res, problem, save_dir=save_dir)
    
    # 4. 打印算子贡献度日志
    analyze_operator_contribution()

# 所有可调参数都集中在这里，方便管理
def get_experiment_config():
    """
    [辅助函数]返回本次实验的配置参数
    """

    # 网络生成器配置
    network_config = HaSConfig(
        num_nodes=15, 
        num_hubs=4, 
        num_emergency_nodes=2, 
        road_connect_prob=0.8,  # 提高连接度以确保路径存在
        num_tasks=5
    )

    # pymoo 算法配置
    algorithm_config = {
        'pop_size': 149,    # NSGA-III 推荐种群规模，通常==参考点数量
        # 参考点 ≪ pop_size → 多个个体挤在同一个参考方向 → 多样性下降
        # 参考点 ≫ pop_size → 大部分参考点无人占有 → 利用效率低
        'n_partitions': 12, # 在M维空间的每个维度生成p个分割点，M=目标函数个数，p=n_partitions划分数
        # n_partitions 需要与 pop_size 对齐
        # 例如 n_partitions=12 在 2 目标时通常会生成 C(2+12-1, 2-1)=13 个参考点
        'crossover_prob': 0.9,  # 交叉概率
        'eta_crossover': 5,    # 交叉分布指数，值越大，子代越接近父代
        'mutation_prob': 0.1,   # 变异概率，值越大，变异概率越大
        'eta_mutation': 5  # 变异分布指数，值越大，变异幅度越小
    }
    
    # 运行配置
    run_config = {
        'termination_gen': 100,
        'seed': 1,  # 设置随机种子以保证实验可复现
        'verbose': True # 是否在运行时打印进度
    }
    
    return network_config, algorithm_config, run_config

def main():
    # === 实验开始前 ===

    # 自动创建并管理实验结果文件夹
    experiment_save_dir = create_experiment_directory(base_dir="results")

    # 配置日志系统
    setup_logging(log_dir=experiment_save_dir, log_name="experiment_log.txt")
    
    # ===================

    """主实验流程"""
    start_time = time.time()
    
    # --- 2. 设置阶段 (Setup) ---
    logging.info("--- [1/4] 正在配置实验... ---")
    net_cfg, algo_cfg, run_cfg = get_experiment_config()

    logging.info("--- [2/4] 正在生成运输网络... ---")
    # 2.1 生成网络实例
    generator = HaSNetworkGenerator(net_cfg)

    # 将随机种子传递给生成器，以保证网络拓扑可复现
    my_network: TransportNetwork = generator.generate(seed=run_cfg['seed'])

    # 在生成网络后，立刻将其保存到JSON文件
    json_save_path = os.path.join(experiment_save_dir, "network_data.json")
    my_network.save_to_json(json_save_path)
    my_network.summary()

    # （可选）可视化网络拓扑，并将其保存在本次实验的专属文件夹中
    my_network.visualize(save_path=os.path.join(experiment_save_dir, "network_topology.png"))

    # 2.2 定义优化问题
    problem = HazmatProblem(network=my_network)

    # --- 3. 执行阶段 (Execution) ---
    logging.info("--- [3/4] 正在配置并运行 NSGA-III 算法... ---")

    # 3.1 使用代理模式来包装算子
    
    # 1. 实例化原生的交叉和变异算子
    uniform_crossover = UniformCrossover(prob=algo_cfg['crossover_prob'])
    bitflip_mutation = BitflipMutation(prob=algo_cfg['mutation_prob'])
    
    # 2. 用自定义的 Logging 类来包装它们
    #   operator_log 是从 core.operators 导入的全局字典
    logging_crossover = LoggingCrossover(uniform_crossover)
    logging_mutation = LoggingMutation(bitflip_mutation)

    # 3 配置算法
    # 为 NSGA-III 准备参考点
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=algo_cfg['n_partitions'])

    # 实例化 NSGA-III
    algorithm = NSGA3(
        pop_size=algo_cfg['pop_size'],
        ref_dirs=ref_dirs,
        sampling=IntegerRandomSampling(),
        # crossover=SBX(prob=algo_cfg['crossover_prob'], eta=algo_cfg['eta_crossover']),
        crossover=logging_crossover,
        # mutation=PM(prob=algo_cfg['mutation_prob'], eta=algo_cfg['eta_mutation']),
        mutation=logging_mutation,
        eliminate_duplicates=True
    )
    
    # 3.2 设置终止条件，和用于算法性能诊断的 logger
    termination = get_termination("n_gen", run_cfg['termination_gen'])
    generational_logger = GenerationalLogger()

    logging.info("核心算法开始运行...")

    # === 捕获 pymoo 的 verbose 输出 ===
    generations_file_path = os.path.join(experiment_save_dir, "generations.txt")
    with open(generations_file_path, 'a', encoding='utf-8') as log_file:
        with redirect_stdout(log_file):
            # 3.3 运行优化
            res = minimize(
                problem,
                algorithm,
                termination,
                seed=run_cfg['seed'],
                save_history=True,
                verbose=run_cfg['verbose'],
                callback=generational_logger
            )
    # =================================

    # --- 4. 分析阶段 (Analysis) ---
    logging.info("--- [4/4] 正在分析并可视化结果... ---")
    if res.F is None or len(res.F) == 0:
        logging.warning("未找到任何非支配解，无法进行后续分析。")
        return

    # --- 4.1 可视化帕累托前沿 ---
    plotter = ParetoPlotter(title="Pareto Frontier", save_dir=experiment_save_dir)
    plotter.plot(res.F, file_name="pareto_frontier.png", xlable="Total Risk", ylable="Total Cost")

    # --- 4.2 解读两个极端最优解 ---

    # 使用上下文管理器来控制日志格式
    with log_section(clean=True):
        # 在这个 'with' 代码块内，所有 logging.info 都会是纯净的
        lowest_risk_index = np.argmin(res.F[:, 0])
        print_solution_details("Optimum (Lowest Risk - SP-CVaR)", res.F[lowest_risk_index], res.X[lowest_risk_index], problem)
        
        lowest_cost_index = np.argmin(res.F[:, 1])
        print_solution_details("Optimum (Lowest Cost)", res.F[lowest_cost_index], res.X[lowest_cost_index], problem)
        
    # 退出 'with' 代码块后，日志格式会自动恢复为带时间戳

    # 停止计时器并报告核心算法耗时
    end_time = time.time()
    print(f"实验总耗时: {end_time - start_time:.2f} 秒。")
    
    logging.info("==========================================")
    logging.info(f"实验总耗时: {end_time - start_time:.2f} 秒。")
    logging.info("==========================================")

    # === 算法性能诊断 ===

    print("--- 正在进行附加分析 ---")
    # 在计时结束后，才调用独立的分析函数
    with log_section(clean=True):
        run_analysis(res, problem, generational_logger, save_dir=experiment_save_dir)
    
    # ===================

if __name__ == '__main__':
    main()