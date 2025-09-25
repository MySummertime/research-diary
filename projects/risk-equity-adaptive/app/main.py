# --- coding: utf-8 ---
# --- main.py ---
import time
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3    # type: ignore
from pymoo.operators.sampling.rnd import IntegerRandomSampling  # type: ignore
from pymoo.operators.crossover.sbx import SBX   # type: ignore
from pymoo.operators.mutation.pm import PM  # type: ignore
from pymoo.util.ref_dirs import get_reference_directions    # type: ignore
from pymoo.optimize import minimize # type: ignore
from pymoo.termination import get_termination   # type: ignore

from core.network_generator import HaSConfig, HaSNetworkGenerator
from core.problem import HazmatProblem
from core.plotter import ParetoPlotter

def print_solution_details(solution_type: str, objectives: np.ndarray, solution_x: np.ndarray, problem: HazmatProblem):
    """
    [辅助函数] 打印单个最优解的详细信息。
    """
    print(f"\n--- {solution_type} 解读 ---")
    print(f"目标值: Risk={objectives[0]:.2f}, Cost={objectives[1]:.2f}")
    print("该解对应的枢纽选择策略为:")
    
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
            print(f"  - 任务 {task.id} (从 {task.origin.id} 到 {task.destination.id}): 选择单枢纽 {hub_names[0]}")
        else:
            print(f"  - 任务 {task.id} (从 {task.origin.id} 到 {task.destination.id}): 选择双枢纽 {hub_names[0]} -> {hub_names[1]}")
        
        # --- 调用解码器，打印详细路径 ---
        # 将原始的浮点数元组传递给 _decode_path，因为它内部会处理转换
        _, detailed_path_nodes = problem._decode_path(task, tuple(hub_indices_float))
        
        # 将节点列表转换为一个易于阅读的字符串
        if detailed_path_nodes:
            path_str = " -> ".join([node.id for node in detailed_path_nodes])
            print(f"      路径详情: {path_str}")
        else:
            print("      路径详情: 未能生成有效路径")


# --- 1. 实验配置中心 ---
# 所有可调参数都集中在这里，方便管理
def get_experiment_config():
    """返回本次实验的配置参数"""
    # 预定义的 OD 对 (起点ID, 终点ID)
    # 如果设为 None，则会随机生成 num_tasks 个任务
    # PREDEFINED_OD_PAIRS = [
        # ('0', '12'), ('1', '9'), ('2', '13'), ('3', '7'), ('6', '14')
    # ]

    # 网络生成器配置
    network_config = HaSConfig(
        num_nodes=15, 
        num_hubs=4, 
        num_emergency_nodes=2, 
        road_connect_prob=0.8,  # 提高连接度以确保路径存在
        # predefined_tasks=PREDEFINED_OD_PAIRS,
        num_tasks=5 # 仅在 predefined_tasks=None 时生效
    )

    # 问题定义配置 (约束等)
    problem_config = {
        'G_min': 0.0,   # 基尼系数下限
        'G_max': 1.0    # 基尼系数上限
    }

    # pymoo 算法配置
    algorithm_config = {
        'pop_size': 150,    # NSGA-III 推荐种群规模，通常为参考点数量的整数倍
        'n_partitions': 12, # 参考点划分数，影响参考点数量，在M维空间的每个维生成p个分割点
        'crossover_prob': 0.9,  # 交叉概率
        'eta_crossover': 30,    # 交叉分布指数，值越大，子代越接近父代
        'mutation_prob': 0.1,   # 变异概率，可以根据变量数量调整
        'eta_mutation': 40  # 变异分布指数，值越大，变异幅度越小
    }
    
    # 运行配置
    run_config = {
        'termination_gen': 100,
        'seed': 1,  # 设置随机种子以保证实验可复现
        'verbose': True # 是否在运行时打印进度
    }
    
    return network_config, problem_config, algorithm_config, run_config

def main():
    """主实验流程"""
    start_time = time.time()
    
    # --- 2. 设置阶段 (Setup) ---
    print("--- [1/4] 正在配置实验... ---")
    net_cfg, prob_cfg, algo_cfg, run_cfg = get_experiment_config()

    print("\n--- [2/4] 正在生成运输网络... ---")
    # 2.1 生成网络实例
    generator = HaSNetworkGenerator(net_cfg)
    my_network = generator.generate()
    my_network.summary()
    # (可选) 在优化前先看一下网络拓扑
    my_network.visualize()

    # 2.2 定义优化问题
    problem = HazmatProblem(
        network=my_network, 
        G_min=prob_cfg['G_min'], 
        G_max=prob_cfg['G_max']
    )

    # --- 3. 执行阶段 (Execution) ---
    print("\n--- [3/4] 正在配置并运行 NSGA-III 算法... ---")
    # 3.1 配置算法
    # 为 NSGA-III 准备参考点
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=algo_cfg['n_partitions'])

    algorithm = NSGA3(
        pop_size=algo_cfg['pop_size'],
        ref_dirs=ref_dirs,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=algo_cfg['crossover_prob'], eta=algo_cfg['eta_crossover']),
        mutation=PM(prob=algo_cfg['mutation_prob'], eta=algo_cfg['eta_mutation']),
        eliminate_duplicates=True
    )
    
    # 3.2 设置终止条件
    termination = get_termination("n_gen", run_cfg['termination_gen'])

    # 3.3 运行优化
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=run_cfg['seed'],
        save_history=True,
        verbose=run_cfg['verbose']
    )

    # --- 4. 分析阶段 (Analysis) ---
    print("\n--- [4/4] 正在分析并可视化结果... ---")
    if res.F is not None and len(res.F) > 0:
        # 4.1 可视化帕累托前沿
        plotter = ParetoPlotter(title="Pareto Frontier")
        plotter.plot(res.F, file_name="pareto_frontier.png")

        # --- 4.2 解读两个极端最优解 ---
        # a) 寻找并解读“风险最优解”（不计成本）
        #    res.F[:, 0] 代表所有解的第一个目标（风险）
        lowest_risk_index = np.argmin(res.F[:, 0])
        lowest_risk_solution_x = res.X[lowest_risk_index]
        lowest_risk_objectives = res.F[lowest_risk_index]
        # --- 打印所有找到的非支配解 ---
        print_solution_details(
            "Optimum (Lowest Risk)", 
            lowest_risk_objectives, 
            lowest_risk_solution_x, 
            problem
        )

        # b) 寻找并解读“成本最优解”（不计风险）
        #    res.F[:, 1] 代表所有解的第二个目标（成本）
        lowest_cost_index = np.argmin(res.F[:, 1])
        lowest_cost_solution_x = res.X[lowest_cost_index]
        lowest_cost_objectives = res.F[lowest_cost_index]
        # --- 打印所有找到的非支配解 ---
        print_solution_details(
            "Optimum (Lowest Cost)", 
            lowest_cost_objectives, 
            lowest_cost_solution_x, 
            problem
        )
    else:
        print("未找到任何非支配解，无法进行后续分析。")

    end_time = time.time()
    print(f"\n实验总耗时: {end_time - start_time:.2f} 秒。")

if __name__ == '__main__':
    main()