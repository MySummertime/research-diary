# --- coding: utf-8 ---
# --- app/utils/analyzer.py ---
import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List 
from app.core.solution import Solution
from app.core.evaluator import Evaluator
from app.core.path import Path

def plot_risk_histogram(solutions: List[Solution], save_dir: str = "results", file_name: str = "risk_histogram.png"):
    """
    绘制最终种群中所有可行解的风险值分布直方图。
    (注: 此处过滤 'is_feasible' 是合理的设计选择，因为我们关心的是 *最终可用解* 的分布)
    """
    risks = [s.f1_risk for s in solutions if s.is_feasible and s.f1_risk != float('inf')]
    
    if not risks:
        logging.warning("Analyzer: 未找到可行的解，无法绘制风险直方图。")
        return

    os.makedirs(save_dir, exist_ok=True)
    try:
        plt.figure(figsize=(10, 7))
        plt.hist(risks, bins=20, edgecolor='black', alpha=0.7, color='red')
        plt.xlabel("f1_Risk")
        plt.ylabel("Frequency")
        plt.title("Risk Distribution in Final Feasible Population")
        plt.grid(axis='y', alpha=0.75)
        
        # 添加均值和中位数线
        plt.axvline(np.mean(risks), color='blue', linestyle='dashed', linewidth=2, label=f'Mean (avg: {np.mean(risks):,.0f})')
        plt.axvline(np.median(risks), color='orange', linestyle='dashed', linewidth=2, label=f'Median (med: {np.median(risks):,.0f})')
        plt.legend()

        full_path = os.path.join(save_dir, file_name)
        plt.savefig(full_path, dpi=300)
        plt.close()
        logging.info(f"风险分布直方图已保存至: {full_path}")
    except Exception as e:
        logging.error(f"风险直方图绘制失败: {e}")

def plot_evolution(logs: Dict[str, List[float]], save_dir: str = "results", file_name_prefix: str = "evolution"):
    """
    绘制风险、成本 *和* 约束违反度(CV)的进化轨迹。
    """
    # 检查 'cv_avg'，这是我们 V2 callback 里的新数据
    if not logs.get("cv_avg"):
        logging.warning("Analyzer: 日志数据不完整 (缺少 'cv_avg')，无法绘制进化图。")
        return

    gens = range(len(logs["cv_avg"]))
    os.makedirs(save_dir, exist_ok=True)

    try:
        # --- 1. 绘制“可行性” (CV) 演化图 ---
        plt.figure(figsize=(12, 6))
        plt.plot(gens, logs["cv_avg"], label="Mean Constraint Violation (CV)", color='purple')
        plt.plot(gens, logs["cv_min"], label="Min Constraint Violation (CV)", color='pink', linestyle='--')
        plt.xlabel("Generation")
        plt.ylabel("Constraint Violation (CV)")
        plt.title("Feasibility Evolution (Constraint Violation)")
        # plt.yscale('log')   # CV 通常最好用对数尺度看
        plt.legend()
        plt.grid(True)
        cv_file_path = os.path.join(save_dir, f"{file_name_prefix}_feasibility_cv.png")
        plt.savefig(cv_file_path, dpi=300)
        plt.close()
        logging.info(f"“可行性”演化轨迹图已保存至: {cv_file_path}")

        # --- 2. 绘制风险演化图 (可行解) ---
        plt.figure(figsize=(12, 6))
        plt.plot(gens, logs["risk_min"], label="Minimum Risk (of feasible)", color='green')
        plt.plot(gens, logs["risk_mean"], label="Mean Risk (of feasible)", color='blue', linestyle='--')
        plt.plot(gens, logs["risk_median"], label="Median Risk (of feasible)", color='orange', linestyle=':')
        plt.xlabel("Generation")
        plt.ylabel("Risk (f1)")
        plt.title("Risk Evolution (Feasible Solutions Only)")
        plt.legend()
        plt.grid(True)
        risk_file_path = os.path.join(save_dir, f"{file_name_prefix}_risk.png")
        plt.savefig(risk_file_path, dpi=300)
        plt.close()
        logging.info(f"风险演化轨迹图已保存至: {risk_file_path}")

        # --- 3. 绘制成本演化图 (可行解) ---
        plt.figure(figsize=(12, 6))
        plt.plot(gens, logs["cost_min"], label="Minimum Cost (of feasible)", color='green')
        plt.plot(gens, logs["cost_mean"], label="Mean Cost (of feasible)", color='blue', linestyle='--')
        plt.plot(gens, logs["cost_median"], label="Median Cost (of feasible)", color='orange', linestyle=':')
        plt.xlabel("Generation")
        plt.ylabel("Cost (f2)")
        plt.title("Cost Evolution (Feasible Solutions Only)")
        plt.legend()
        plt.grid(True)
        cost_file_path = os.path.join(save_dir, f"{file_name_prefix}_cost.png")
        plt.savefig(cost_file_path, dpi=300)
        plt.close()
        logging.info(f"成本演化轨迹图已保存至: {cost_file_path}")
    except Exception as e:
        logging.error(f"进化曲线绘图失败: {e}")

def analyze_operator_contribution(log: Dict[str, int]):
    """
    [V2 改进] 打印并记录 *混合* 算子的调用次数。
    """
    logging.info("--- [附加分析] 算子贡献度日志 ---")
    
    crossover_calls = log.get("crossover_calls", 0)
    # 正确汇总两个变异键
    mut_path_calls = log.get("mutation_path_calls", 0)
    mut_eta_calls = log.get("mutation_eta_calls", 0)
    total_mutations = mut_path_calls + mut_eta_calls
    
    total_calls = crossover_calls + total_mutations
    
    if total_calls == 0:
        logging.warning("\n未记录到算子调用信息。")
        return

    logging.info("\n算子调用次数及占比分析:")
    crossover_pct = (crossover_calls / total_calls) * 100 if total_calls > 0 else 0
    mutation_pct = (total_mutations / total_calls) * 100 if total_calls > 0 else 0
    
    logging.info(f"  - 交叉 (Crossover) 总调用: {crossover_calls:<6} 次, 占比 {crossover_pct:.2f}%")
    logging.info(f"  - 变异 (Mutation)  总调用: {total_mutations:<6} 次, 占比 {mutation_pct:.2f}%")
    logging.info(f"      ↳ (路径变异): {mut_path_calls:<6} 次")
    logging.info(f"      ↳ (Eta 变异): {mut_eta_calls:<6} 次")

def plot_sensitivity_boxplot(all_deltas: List[List[float]], save_dir: str = "results", file_name: str = "sensitivity_boxplot.png"):
    """
    [辅助函数] 绘制局部灵敏度测试的结果箱线图。
    """
    if not all_deltas:
        logging.warning("Analyzer: 灵敏度分析数据为空，跳过箱线图绘制。")
        return

    os.makedirs(save_dir, exist_ok=True)
    try:
        plt.figure(figsize=(12, 8))
        # 过滤掉空的 deltas 列表（如果某个解的所有变异都失败了）
        valid_deltas = [d for d in all_deltas if d]
        if not valid_deltas:
            logging.warning("Analyzer: 灵敏度分析未收集到任何有效的 delta 数据。")
            return
            
        plt.boxplot(valid_deltas)
        plt.xlabel("Solution Index (Randomly Sampled from Pareto Front)")
        plt.ylabel("ΔRisk (Risk_mutated - Risk_original)")
        plt.title("Local Sensitivity Analysis (via Path Mutation)")
        plt.grid(True)
        full_path = os.path.join(save_dir, file_name)
        plt.savefig(full_path, dpi=300)
        plt.close()
        logging.info(f"局部灵敏度箱线图已保存至: {full_path}")
    except Exception as e:
        logging.error(f"灵敏度箱线图绘制失败: {e}")

def perform_local_sensitivity_analysis(
    final_front: List[Solution], 
    evaluator: Evaluator, 
    candidate_paths_map: Dict[str, List[Path]],
    save_dir: str, 
    n_solutions_to_test: int = 5, 
    n_trials_per_solution: int = 20):
    """
    对帕累托前沿上的部分解进行局部扰动。
    同时记录 'deltas' 和 'infeasible_count'。
    """
    logging.info("--- [附加分析] 正在执行局部灵敏度测试... ---")
    
    # 1. 选择解
    feasible_front = [s for s in final_front if s.is_feasible]
    if not feasible_front:
        logging.warning("Analyzer: 灵敏度分析失败，因为最终前沿没有 *可行* 解可供测试。")
        return
        
    n_solutions = len(feasible_front)
    n_to_select = min(n_solutions, n_solutions_to_test)
    
    if n_to_select == 0:
        logging.warning("Analyzer: n_solutions_to_test 为 0，跳过灵敏度分析。")
        return
        
    selected_indices = np.random.choice(n_solutions, n_to_select, replace=False)
    selected_solutions = [feasible_front[i] for i in selected_indices]
    
    all_deltas = [] # 存储 (new_risk - original_risk)
    total_infeasible_mutations = 0
    total_mutations = 0
    
    for i, solution in enumerate(selected_solutions):
        deltas = []
        original_risk = solution.f1_risk
        
        logging.info(f"  - 正在测试 (可行的) 解 {i+1}/{n_to_select} (Risk: {original_risk:,.0f})...")
        
        for _ in range(n_trials_per_solution):
            mutated_solution = solution.clone()
            
            try:
                task_id_to_mutate = random.choice(list(mutated_solution.path_selections.keys()))
                
                all_paths = candidate_paths_map[task_id_to_mutate] 
                if len(all_paths) <= 1:
                    continue    # 无法变异

                new_path = random.choice(all_paths)
                while new_path == solution.path_selections[task_id_to_mutate]:
                    new_path = random.choice(all_paths)
                
                mutated_solution.path_selections[task_id_to_mutate] = new_path
                total_mutations += 1
            
            except Exception as e:
                logging.warning(f"灵敏度分析中的变异步骤失败: {e}")
                continue 

            evaluator.evaluate(mutated_solution)

            # 不再关心变异后是否可行，我们只关心风险值
            new_risk = mutated_solution.f1_risk
            
            if new_risk != float('inf'):
                delta = new_risk - original_risk
                deltas.append(delta)
            
            # 我们 *额外* 统计它是否“掉下了悬崖”
            if not mutated_solution.is_feasible:
                total_infeasible_mutations += 1
                
        all_deltas.append(deltas)

    # 2. 绘制箱线图
    plot_sensitivity_boxplot(all_deltas, save_dir=save_dir)
    
    # 3. 报告“稳定性”
    if total_mutations > 0:
        infeasible_pct = (total_infeasible_mutations / total_mutations) * 100
        logging.info(f"  - 灵敏度总结: 在 {total_mutations} 次变异中, 有 {total_infeasible_mutations} 次 ({infeasible_pct:.2f}%) 导致了解变为“不可行”。")

def print_solution_details(solution_type: str, solution: Solution):
    """
    打印单个最优解的详细信息。
    """
    logging.info(f"\n--- {solution_type} 解读 ---")
    logging.info(f"目标值: f1_Risk={solution.f1_risk:,.2f}, f2_Cost={solution.f2_cost:,.2f}")
    logging.info(f"状态: Feasible={solution.is_feasible}, CV={solution.constraint_violation:.4f}")
    logging.info("该解对应的路径选择策略为:")
    
    for task_id, path in solution.path_selections.items():
        if path.task:
            task = path.task
            eta_v = solution.eta_values.get(task_id, 0.0)
            
            logging.info(f"  - 任务 {task.task_id} (从 {task.origin.id} 到 {task.destination.id}):")
            
            # 打印详细路径信息
            path_str = " -> ".join([node.id for node in path.nodes])
            all_hubs_str = ", ".join([h.id for h in path.all_hubs_on_path])
            transfer_hubs_str = ", ".join([h.id for h in path.transfer_hubs])
            
            logging.info(f"      完整路径: {path_str}")
            logging.info(f"      途经所有枢纽: [{all_hubs_str}]")
            logging.info(f"      转运枢纽 ({len(path.transfer_hubs)}): [{transfer_hubs_str}]")
            logging.info(f"      此任务VaR (η_v): {eta_v:,.2f}")
        else:
            logging.warning(f"  - 任务 {task_id} 的路径缺少 task 引用。")