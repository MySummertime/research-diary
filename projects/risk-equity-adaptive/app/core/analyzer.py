# --- coding: utf-8 ---
# --- analyzer.py ---
import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from core.operators import operator_log # type: ignore

# --- 这个文件专门用于存放所有与“算法性能诊断”相关的函数 ---

def plot_risk_histogram(objectives: np.ndarray, save_dir: str = "results", file_name: str = "risk_histogram.png"):
    """
    绘制最终种群中所有个体风险值的分布直方图。
    """
    risks = objectives[:, 0]
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.hist(risks, bins=20, edgecolor='black')
    plt.xlabel("Risk (SP-CVaR)")
    plt.ylabel("Frequency")
    plt.title("Risk distribution in final population")
    plt.grid(axis='y', alpha=0.75)
    full_path = os.path.join(save_dir, file_name)
    plt.savefig(full_path)
    print(f"风险分布直方图已保存至: {full_path}")
    # plt.show()

def plot_evolution(logs: Dict[str, List[float]], save_dir: str = "results", file_name_prefix: str = "evolution"):
    """
    绘制风险和成本随代数变化的最小/平均/中位数轨迹。
    """
    if not logs["risk_min"]:
        logging.warning("警告：日志数据为空，无法绘制目标函数值演化图。")
        return

    gens = range(len(logs["risk_min"]))
    os.makedirs(save_dir, exist_ok=True)

    # --- 绘制风险演化图 ---
    plt.figure(figsize=(12, 6))
    plt.plot(gens, logs["risk_min"], label="Minimum Risk", color='green')
    plt.plot(gens, logs["risk_mean"], label="Mean Risk", color='blue')
    plt.plot(gens, logs["risk_median"], label="Median Risk", color='orange')
    plt.xlabel("Generation")
    plt.ylabel("Risk (SP-CVaR)")
    plt.title("Risk Evolution over Generations")
    plt.legend()
    plt.grid(True)
    risk_file_path = os.path.join(save_dir, f"{file_name_prefix}_risk.png")
    plt.savefig(risk_file_path)
    print(f"风险演化轨迹图已保存至: {risk_file_path}")
    # plt.show()

    # --- 绘制成本演化图 ---
    plt.figure(figsize=(12, 6))
    plt.plot(gens, logs["cost_min"], label="Minimum Cost", color='green')
    plt.plot(gens, logs["cost_mean"], label="Mean Cost", color='blue')
    plt.plot(gens, logs["cost_median"], label="Median Cost", color='orange')
    plt.xlabel("Generation")
    plt.ylabel("Total Cost")
    plt.title("Cost Evolution over Generations")
    plt.legend()
    plt.grid(True)
    cost_file_path = os.path.join(save_dir, f"{file_name_prefix}_cost.png")
    plt.savefig(cost_file_path)
    print(f"成本演化轨迹图已保存至: {cost_file_path}")
    # plt.show()

def plot_sensitivity_boxplot(all_deltas: List[List[float]], save_dir: str = "results", file_name: str = "sensitivity_boxplot.png"):
    """
    绘制局部灵敏度测试的结果箱线图。
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.boxplot(all_deltas)
    plt.xlabel("Solution Index")
    plt.ylabel("ΔRisk after small mutation")
    plt.title("Local Sensitivity of SP-CVaR Landscape")
    plt.grid(True)
    full_path = os.path.join(save_dir, file_name)
    plt.savefig(full_path)
    print(f"局部灵敏度箱线图已保存至: {full_path}")
    # plt.show()

def perform_local_sensitivity_analysis(res: 'Result', problem: 'HazmatProblem', save_dir: str = "results", n_solutions_to_test: int = 5, n_trials_per_solution: int = 20):
    """
    对帕累托前沿上的部分解进行局部扰动，以测试适应度地形的崎岖程度。
    """
    logging.info("--- [附加分析] 正在执行局部灵敏度测试... ---")
    
    n_solutions = len(res.X)
    if n_solutions == 0:
        logging.warning("警告：没有找到非支配解，无法进行灵敏度分析。")
        return
        
    n_to_select = min(n_solutions, n_solutions_to_test)
    selected_indices = np.random.choice(n_solutions, n_to_select, replace=False)
    
    selected_solutions_X = res.X[selected_indices]
    selected_solutions_F = res.F[selected_indices]
    
    all_deltas = []
    
    for i, x in enumerate(selected_solutions_X):
        deltas = []
        original_risk = selected_solutions_F[i, 0]
        
        for _ in range(n_trials_per_solution):
            y = x.copy()
            
            num_tasks = len(problem.tasks)
            num_hubs = len(problem.hubs)
            task_idx_to_mutate = random.randint(0, num_tasks - 1)
            hub_pos_to_mutate = random.randint(0, 1)
            gene_idx_to_mutate = task_idx_to_mutate * 2 + hub_pos_to_mutate
            
            new_hub_idx = random.randint(0, num_hubs - 1)
            while new_hub_idx == y[gene_idx_to_mutate]:
                new_hub_idx = random.randint(0, num_hubs - 1)
            y[gene_idx_to_mutate] = new_hub_idx
            
            out = {}
            problem._evaluate(np.array([y]), out)
            new_risk = out["F"][0, 0]
            
            delta = new_risk - original_risk
            deltas.append(delta)
            
        all_deltas.append(deltas)

    plot_sensitivity_boxplot(all_deltas, save_dir=save_dir)

# === 算子贡献度分析函数 ===
def analyze_operator_contribution():
    """
    打印并记录算子在整个优化过程中的调用次数。
    """
    logging.info("--- [附加分析] 算子贡献度日志 ---")
    
    total_calls = 0
    for op_name, stats in operator_log.items():
        total_calls += stats.get("calls", 0)
    
    if total_calls == 0:
        logging.warning("\n未记录到算子调用信息。")
        return

    logging.info("\n算子调用次数及占比分析:")
    for op_name, stats in operator_log.items():
        calls = stats.get("calls", 0)
        percentage = (calls / total_calls) * 100 if total_calls > 0 else 0
        logging.info(f"  - {op_name.capitalize():<12}: 调用 {calls:<6} 次, 占比 {percentage:.2f}%")
        
    # 此日志展示了各算子的工作量.
    # 要进一步分析其有效性（即产生了多少优良解），
    # 需要在算子代理类中集成更复杂的支配关系判断逻辑，
    # 这是一个重要的未来研究方向.