# --- coding: utf-8 ---
# --- run_validation.py ---
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from main import load_config, setup_experiment, run_and_analyze_experiment
from app.utils.result_keeper import create_experiment_directory, setup_logging
from app.core.solution import Solution
from app.core.nsga2 import NSGA2 # (任务3需要)
try:
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.spacing import SpacingIndicator
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError:
    print("错误: 找不到 'pymoo' 库。请先 'pip install pymoo'")
    exit()

# ----------------------------------------
# 任务1：可视化解空间分布
# ----------------------------------------
def run_task_1_and_2(config: Dict, save_dir: str) -> List[np.ndarray]:
    """
    运行 N 次实验，保存所有 Rank 0 解，并返回它们的目标值。
    """
    logging.info("======= [开始任务 1 & 2: 多种子运行与指标计算] =======")
    
    # 定义你要跑的随机种子
    SEEDS = [42, 1337, 9001] 
    
    all_run_fronts: List[List[Solution]] = []
    
    for seed in SEEDS:
        logging.info(f"\n--- [1/3] 正在运行 Seed: {seed} ---")
        
        # 1. 为本次运行创建单独的子目录
        seed_save_dir = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(seed_save_dir, exist_ok=True)
        
        # 2. 运行完整的实验
        network, pf, evaluator, cmap = setup_experiment(config, seed)
        full_archive = run_and_analyze_experiment(network, evaluator, cmap, config, seed_save_dir)
        
        # 3. 只提取 Rank 0 的解
        rank_0_front = [s for s in full_archive if s.rank == 0 and s.is_feasible]
        all_run_fronts.append(rank_0_front)
        logging.info(f"--- Seed {seed} 运行完毕, 找到了 {len(rank_0_front)} 个 Rank 0 解 ---")

    # --- 任务1：开始绘图 ---
    logging.info("\n--- [任务1] 正在绘制多种子重叠图... ---")
    plt.figure(figsize=(14, 9))
    colors = ['#FF0000', '#0000FF', '#00AA00'] # 红, 蓝, 绿
    markers = ['o', 's', '^'] # 圆, 方, 三角
    
    all_objectives_list = [] # 用于任务2

    for i, front in enumerate(all_run_fronts):
        if front:
            objectives = np.array([[s.f1_risk, s.f2_cost] for s in front])
            all_objectives_list.append(objectives)
            
            plt.scatter(objectives[:, 0], objectives[:, 1], 
                        color=colors[i], 
                        marker=markers[i],
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
def run_task_2(all_objectives_list: List[np.ndarray], save_dir: str):
    """
    使用 pymoo 计算 HV, IGD, SP 指标。
    """
    if not all_objectives_list:
        logging.warning("任务2跳过：没有可供分析的前沿。")
        return

    # --- 步骤1：创建“理想前沿” (Reference Front) ---
    # 合并所有三次运行的 *所有* Rank 0 解
    combined_objectives = np.vstack(all_objectives_list)
    
    # 对这个“合并集”再做一次非支配排序
    nd_sorter = NonDominatedSorting()
    nd_indices = nd_sorter.do(combined_objectives)
    
    # 真正的“理想前沿”是这个“合并集”的 Rank 0
    ref_front = combined_objectives[nd_indices[0]]
    
    logging.info(f"\n--- [任务2] 已创建理想前沿 (Reference Front), 包含 {len(ref_front)} 个点 ---")

    # --- 步骤2：初始化指标 ---
    # HV 需要一个“参考点 (ref_point)”，它必须*差于*所有解
    # 我们取所有解中的 (f1_max, f2_max) 再放大 1.1 倍
    ref_point = np.max(combined_objectives, axis=0) * 1.1
    
    hv_indicator = HV(ref_point=ref_point)
    igd_indicator = IGD(pf=ref_front)   # IGD 需要“理想前沿”
    sp_indicator = SpacingIndicator()   # Spacing 只需要自己
    
    hv_scores, igd_scores, sp_scores = [], [], []

    logging.info("--- [任务2] 正在计算性能指标... ---")
    logging.info(f"    参考点 (HV Ref Point): {ref_point}")
    logging.info("    指标       |   Run 1   |   Run 2   |   Run 3   |   Mean    |   StdDev  ")
    logging.info("--------------------------------------------------------------------------------")
    
    for front in all_objectives_list:
        hv_scores.append(hv_indicator(front))
        igd_scores.append(igd_indicator(front))
        sp_scores.append(sp_indicator(front))
    
    def log_metric(name: str, scores: List[float]):
        mean = np.mean(scores)
        std = np.std(scores)
        var_pct = (std / mean) * 100 if mean != 0 else 0
        logging.info(f"    {name:<10} | {scores[0]:^9.3e} | {scores[1]:^9.3e} | {scores[2]:^9.3e} | {mean:^9.3e} | {var_pct:^9.2f}%")

    log_metric("HV", hv_scores)
    log_metric("IGD", igd_scores)
    log_metric("SP", sp_scores)
    logging.info("--------------------------------------------------------------------------------")

# ----------------------------------------
# 任务3：重启 + 局部扰动
# ----------------------------------------
def run_task_3(config: Dict, save_dir: str):
    """
    执行重启实验。
    """
    logging.info("\n======= [开始任务 3: 重启 + 局部扰动实验] =======")
    
    # --- 步骤 1: (小改造) 改造 nsga2.py ---
    # 需要让 run() 接受一个 'initial_population'
    # 打开 'app/core/nsga2.py'，在 'run' 函数定义处修改:
    
    # 原代码:
    # def run(self, callbacks: Optional[List[Callback]] = None) -> List[Solution]:
    
    # 新代码:
    # def run(self, callbacks: Optional[List[Callback]] = None, initial_population: Optional[List[Solution]] = None) -> List[Solution]:
    
    # 然后，在 run() 函数的 *最开头*，替换掉原来的初始化逻辑:
    
    # 原代码:
    # logging.info("开始初始化种群 (P_0)...")
    # population = self._initialize_population()
    # self._evaluate_population(population)
    # self.archive = self._update_archive([], population)
    # self._assign_ranks_and_crowding(self.archive)
    
    # 新代码:
    # if initial_population is None:
    #     logging.info("开始初始化种群 (P_0)...")
    #     population = self._initialize_population()
    #     self._evaluate_population(population)
    #     self.archive = self._update_archive([], population)
    # else:
    #     logging.info("--- [!!!] 正在从一个已提供的种群重启 [!!!] ---")
    #     # 我们假设传入的 population 已经被评估过了
    #     # 并且它的大小符合 self.archive_size
    #     self.archive = initial_population 
    #
    # # 不管是新是旧，都重新计算一次 ranks
    # self._assign_ranks_and_crowding(self.archive)
    
    # --- 步骤 2: 运行“原始”实验 ---
    logging.info("--- [3/1] 正在运行“原始”实验 (Run 1)... ---")
    run1_save_dir = os.path.join(save_dir, "task_3_run_1_original")
    seed = config.get("experiment", {}).get("seed", 42)
    
    network, pf, evaluator, cmap = setup_experiment(config, seed)
    
    nsga2_run1 = NSGA2(network, evaluator, cmap, config)
    archive_run1 = nsga2_run1.run(callbacks=[]) # 用空回调运行
    
    rank_0_run1 = [s for s in archive_run1 if s.rank == 0 and s.is_feasible]
    logging.info(f"--- “原始”实验结束, 找到 {len(rank_0_run1)} 个 Rank 0 解 ---")

    # --- 步骤 3: 扰动并重启 ---
    logging.info("--- [3/2] 正在扰动存档并“重启”实验 (Run 2)... ---")
    run2_save_dir = os.path.join(save_dir, "task_3_run_2_restarted")

    # 创造“被扰动”的初始种群
    # 直接使用 nsga2_run1 实例的 _mutation 方法
    perturbed_archive: List[Solution] = []
    for sol in archive_run1:
        perturbed_sol = sol.clone()
        nsga2_run1._mutation(perturbed_sol) # 调用内部变异
        perturbed_archive.append(perturbed_sol)
    
    # 评估这个“被扰动”的新种群
    logging.info("正在评估“被扰动”的种群...")
    nsga2_run1._evaluate_population(perturbed_archive)
    
    # 创建 *新* 的 NSGA2 实例并运行
    nsga2_run2 = NSGA2(network, evaluator, cmap, config)
    
    # 注入“被扰动”的种群
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
                    zorder=5) # 让它在红点下面

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
    
    # 1. 创建总的实验目录
    exp_save_dir = create_experiment_directory(base_dir="results_validation")
    setup_logging(log_dir=exp_save_dir, log_name="validation_log.txt")
    
    logging.info(f"======= [开始验证实验 @ {exp_save_dir}] =======")
    
    config = load_config("config.json")
    
    # 2. 运行任务1和任务2
    objectives_list = run_task_1_and_2(config, exp_save_dir)
    
    # 3. 运行任务2
    run_task_2(objectives_list, exp_save_dir)
    
    # 4. 运行任务3
    # (注意: 任务3 需要你先手动修改 nsga2.py)
    logging.warning("即将开始任务3... 请确保你已经按要求修改了 'nsga2.py' 的 'run' 函数！")
    # time.sleep(5) # 5 秒钟反应时间
    run_task_3(config, exp_save_dir)
    
    logging.info("======= [所有验证任务已完成] =======")