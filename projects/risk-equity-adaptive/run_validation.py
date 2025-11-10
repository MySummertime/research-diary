# --- coding: utf-8 ---
# --- run_validation.py ---
import os
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict 
from main import load_config, setup_experiment, run_and_analyze_experiment
from app.utils.result_keeper import create_experiment_directory, setup_logging
from app.core.solution import Solution
from app.core.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ----------------------------------------
# 任务1：可视化解空间分布
# ----------------------------------------
def run_task_1(config: Dict, save_dir: str) -> List[np.ndarray]:
    """
    运行 N 次实验，保存所有 Rank 0 解，并返回它们的目标值。
    """
    logging.info("======= [开始任务 1: 多种子运行与可视化] =======")
    
    # 从 config 中读取 seeds
    SEEDS = config.get("validation", {}).get("seeds", [4, 42, 1337, 9001])
    
    all_run_fronts: List[List[Solution]] = []
    
    for i, seed in enumerate(SEEDS):
        logging.info(f"\n--- [1/{len(SEEDS)}] 正在运行 Seed: {seed} ---")
        
        seed_save_dir = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(seed_save_dir, exist_ok=True)
        
        network, pf, evaluator, cmap = setup_experiment(config, seed)
        
        full_archive = run_and_analyze_experiment(network, evaluator, cmap, config, seed_save_dir)
        
        rank_0_front = [s for s in full_archive if s.rank == 0 and s.is_feasible]
        all_run_fronts.append(rank_0_front)
        logging.info(f"--- Seed {seed} 运行完毕, 找到了 {len(rank_0_front)} 个 Rank 0 解 ---")

    # --- 任务1：开始绘图 ---
    logging.info("\n--- [任务1] 正在绘制多种子重叠图... ---")
    plt.figure(figsize=(14, 9))
    colors = ['#FF0000', '#0000FF', '#00AA00', '#FF8C00', '#9400D3'] # 红, 蓝, 绿, 橙, 紫
    markers = ['o', 's', '^', 'D', 'P'] # 圆, 方, 三角, 菱形, 加号
    
    all_objectives_list = []    # 用于任务2

    for i, front in enumerate(all_run_fronts):
        if front:
            objectives = np.array([[s.f1_risk, s.f2_cost] for s in front])
            all_objectives_list.append(objectives)
            
            plt.scatter(objectives[:, 0], objectives[:, 1], 
                        color=colors[i % len(colors)], 
                        marker=markers[i % len(markers)],
                        s=80, 
                        alpha=0.7,
                        edgecolors='black',
                        label=f'Run (Seed {SEEDS[i]}) ({len(objectives)} points)')

    plt.xlabel("Total Risk (f1)", fontsize=12)
    plt.ylabel("Total Cost (f2)", fontsize=12)
    plt.title("Multi-Seed Pareto Front Overlap", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "task_1_overlap_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"多种子重叠图已保存至: {plot_path}")
    
    return all_objectives_list

# ----------------------------------------
# 任务2：计算统计指标 (Pymoo)
# ----------------------------------------
def run_task_2(all_objectives_list: List[np.ndarray], config: Dict, save_dir: str):
    """
    使用 pymoo 计算 HV, IGD, SP 指标并保存到 'PF_indicators.csv'.
    """
    logging.info("\n======= [开始任务 2: 计算统计指标 =======")
    
    # 检查 pymoo 是否成功导入
    if not all([HV, IGD, SpacingIndicator, NonDominatedSorting]):
        logging.error("Pymoo 库未能成功导入，任务2 [计算统计指标] 被跳过。")
        return
        
    if not all_objectives_list:
        logging.warning("任务2跳过：没有可供分析的前沿 (all_objectives_list 为空)。")
        return

    # --- 步骤1：创建“理想前沿” (Reference Front) ---
    try:
        combined_objectives = np.vstack(all_objectives_list)
    except ValueError:
        logging.warning("任务2跳过：所有运行都没有找到任何 Rank 0 解。")
        return
    
    nd_sorter = NonDominatedSorting()
    nd_indices = nd_sorter.do(combined_objectives)
    ref_front = combined_objectives[nd_indices[0]]
    
    logging.info(f"--- [任务2] 已创建理想前沿 (Reference Front), 包含 {len(ref_front)} 个点 ---")

    # --- 步骤2：初始化指标 ---
    ref_point = np.max(combined_objectives, axis=0) * 1.1
    hv_indicator = HV(ref_point=ref_point)
    igd_indicator = IGD(pf=ref_front) 
    sp_indicator = SpacingIndicator()
    
    hv_scores, igd_scores, sp_scores = [], [], []

    # --- 步骤3：计算指标 ---
    logging.info("--- [任务2] 正在计算性能指标... ---")
    for front in all_objectives_list:
        if front.ndim == 1: 
            front = front.reshape(1, -1)
        if front.size == 0:
            hv_scores.append(0.0)
            igd_scores.append(float('inf'))
            sp_scores.append(float('inf'))
            continue
            
        hv_scores.append(hv_indicator(front))
        igd_scores.append(igd_indicator(front))
        sp_scores.append(sp_indicator(front))
        
    # --- 步骤4：打印日志表格 ---
    SEEDS = config.get("validation", {}).get("seeds", [42, 1337, 9001])
    header = f"    {'指标':<10} |"
    divider = "--------------------"
    for i in range(len(all_objectives_list)):
        # 如果 SEEDS 的长度小于 all_objectives_list (例如只修改了config的task 1)
        # 用 'N/A' 作为种子ID来防止崩溃
        seed_str = f"S{SEEDS[i]}" if i < len(SEEDS) else "S(N/A)"
        header += f" Run {i+1} ({seed_str}) |"
        divider += "-----------------"
    header += "   Mean    |   StdDev  "
    divider += "-------------------"
    
    logging.info(f"    参考点 (HV Ref Point): {ref_point}")
    logging.info(header)
    logging.info(divider)
    
    def log_metric(name: str, scores: List[float]):
        mean = np.mean(scores)
        std = np.std(scores)
        var_pct = (std / abs(mean)) * 100 if mean != 0 else 0
        
        line = f"    {name:<10} |"
        for score in scores:
            line += f" {score:^13.3e} |"
        line += f" {mean:^9.3e} | {var_pct:^9.2f}%"
        logging.info(line)

    log_metric("HV", hv_scores)
    log_metric("IGD", igd_scores)
    log_metric("SP", sp_scores)
    logging.info(divider)

    # 保存 'PF_indicators.csv'
    csv_path = os.path.join(save_dir, "PF_indicators.csv")
    logging.info(f"正在将统计指标保存至: {csv_path}")
    
    try:
        with open(csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            csv_headers = ["Indicator"]
            for i in range(len(all_objectives_list)):
                seed_str = f"S{SEEDS[i]}" if i < len(SEEDS) else "S(N/A)"
                csv_headers.append(f"Run_{i+1}_({seed_str})")
            csv_headers.extend(["Mean", "StdDev", "Var_Pct (%)"])
            writer.writerow(csv_headers)
            
            # 封装一个函数来写入行
            def write_metric_row(name: str, scores: List[float]):
                mean = np.mean(scores)
                std = np.std(scores)
                var_pct = (std / abs(mean)) * 100 if mean != 0 else 0
                
                row = [name]
                row.extend([f"{s:.4e}" for s in scores])    # 添加所有运行的分数
                row.extend([f"{mean:.4e}", f"{std:.4e}", f"{var_pct:.2f}"])
                writer.writerow(row)

            # 写入数据
            write_metric_row("Hypervolume (HV)", hv_scores)
            write_metric_row("Inv. Gen. Dist. (IGD)", igd_scores)
            write_metric_row("Spacing (SP)", sp_scores)
            
    except Exception as e:
        logging.error(f"保存 PF_indicators.csv 失败: {e}")

# ----------------------------------------
# 任务3：重启 + 局部扰动
# ----------------------------------------
def run_task_3(config: Dict, save_dir: str):
    """
    执行重启实验。
    (这个函数 100% 正确，保持不变)
    """
    logging.info("\n======= [开始任务 3: 重启 + 局部扰动实验] =======")
    
    # --- 步骤 1: (提醒) ---
    logging.info("--- [3/0] 提醒: 本任务假设 'nsga2.py' 的 'run' 函数已按要求修改... ---")
    
    # --- 步骤 2: 运行“原始”实验 ---
    logging.info("--- [3/1] 正在运行“原始”实验 (Run 1)... ---")
    seed = config.get("experiment", {}).get("seed", 42) # 使用 config.json 的默认种子
    
    try:
        network, pf, evaluator, cmap = setup_experiment(config, seed)
    except Exception as e:
        logging.error(f"任务3的 setup_experiment 失败: {e}")
        return
    
    nsga2_run1 = NSGA2(network, evaluator, cmap, config)
    archive_run1 = nsga2_run1.run(callbacks=[]) 
    
    rank_0_run1 = [s for s in archive_run1 if s.rank == 0 and s.is_feasible]
    logging.info(f"--- “原始”实验结束, 找到 {len(rank_0_run1)} 个 Rank 0 解 ---")
    
    if not archive_run1:
        logging.error("“原始”实验未能产生任何存档，任务3 中止。")
        return

    # --- 步骤 3: 扰动并重启 ---
    logging.info("--- [3/2] 正在扰动存档并“重启”实验 (Run 2)... ---")

    perturbed_archive: List[Solution] = []
    for sol in archive_run1:
        perturbed_sol = sol.clone()
        nsga2_run1._mutation(perturbed_sol) 
        perturbed_archive.append(perturbed_sol)
    
    logging.info("正在评估“被扰动”的种群...")
    nsga2_run1._evaluate_population(perturbed_archive)
    
    nsga2_run2 = NSGA2(network, evaluator, cmap, config)

    archive_run2 = nsga2_run2.run(callbacks=[], initial_population=perturbed_archive)
        
    rank_0_run2 = [s for s in archive_run2 if s.rank == 0 and s.is_feasible]
    logging.info(f"--- “重启”实验结束, 找到 {len(rank_0_run2)} 个 Rank 0 解 ---")
    
    # --- 步骤 4: 可视化对比 ---
    logging.info("--- [3/3] 正在绘制“重启”对比图... ---")
    plt.figure(figsize=(14, 9))
    
    if rank_0_run1:
        obj_run1 = np.array([[s.f1_risk, s.f2_cost] for s in rank_0_run1])
        plt.scatter(obj_run1[:, 0], obj_run1[:, 1], 
                    color='red', marker='o', s=100, alpha=0.8,
                    edgecolors='black', label=f'Original Run 1 (Rank 0) ({len(obj_run1)})',
                    zorder=10)

    if rank_0_run2:
        obj_run2 = np.array([[s.f1_risk, s.f2_cost] for s in rank_0_run2])
        plt.scatter(obj_run2[:, 0], obj_run2[:, 1], 
                    color='blue', marker='x', s=100, alpha=0.8,
                    label=f'Restarted Run 2 (Rank 0) ({len(obj_run2)})',
                    zorder=5) 
        
    plt.xlabel("Total Risk (f1)", fontsize=12)
    plt.ylabel("Total Cost (f2)", fontsize=12)
    plt.title("Task 3: Restart vs Original Run", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path_task3 = os.path.join(save_dir, "task_3_restart_plot.png")
    plt.savefig(plot_path_task3, dpi=300)
    plt.close()
    logging.info(f"“重启”对比图已保存至: {plot_path_task3}")

# ----------------------------------------
# 总控 MAIN
# ----------------------------------------
if __name__ == '__main__':
    base_dir = "results_validation"
    config_path = "config.json"
    experiment_log_name = "validation.log"
    
    # 1. 创建总的实验目录
    save_dir = create_experiment_directory(base_dir=base_dir)
    setup_logging(log_dir=save_dir, log_name=experiment_log_name)
    
    logging.info(f"======= [开始验证实验 @ {save_dir}] =======")
    
    config = load_config(config_file=config_path)

    # 2. 运行任务1 (绘图) 并获取任务2的数据
    objectives_list = run_task_1(config, save_dir)
    
    # 3. 运行任务2 (计算指标 + 保存 CSV)
    run_task_2(objectives_list, config, save_dir)
    
    # 4. 运行任务3 (重启实验)
    run_task_3(config, save_dir)
    
    logging.info("======= [所有验证任务已完成] =======")