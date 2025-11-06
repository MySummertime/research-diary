# --- coding: utf-8 ---
# --- app/utils/callback.py ---
import os
import numpy as np
from typing import Dict, List
from app.core.nsga2 import Callback
from app.core.solution import Solution

# 回调 1: 用于内存日志 (供 绘图/分析 使用)
class GenerationalLogger(Callback):
    """
    一个自定义的回调类，用于在算法的每一代结束时，
    在 *内存* 中记录下关键的统计数据 (用于 plot_evolution 绘图)
    增加了对 'cv_min' 和 'cv_avg' 的跟踪，
    以便在没有可行解时也能监控算法的“可行性”收敛过程。
    """
    def __init__(self):
        super().__init__()
        self.logs: Dict[str, List[float]] = {
            "risk_min": [], "risk_mean": [], "risk_median": [],
            "cost_min": [], "cost_mean": [], "cost_median": [],
            "cv_min": [], "cv_avg": [] 
        }

    def on_generation_end(self, generation: int, population: List[Solution]):
        """
        [接口] 在每代结束时，从 Solution 列表中计算统计数据。
        """
        if not population:
            return

        # --- 步骤 1 - 立即计算约束违反度 (CV) ---
        # 无论种群是否可行，我们都跟踪CV
        violations = [s.constraint_violation for s in population if s.constraint_violation != float('inf')]
        if violations:
            self.logs["cv_min"].append(float(np.min(violations)))
            self.logs["cv_avg"].append(float(np.mean(violations)))
        else:
            # 如果没有违反（例如，所有解都可行，或者种群为空），则CV为0
            self.logs["cv_min"].append(0.0)
            self.logs["cv_avg"].append(0.0)

        # --- 步骤 2: 仅计算 *可行解* 的目标函数统计 ---
        # (这部分逻辑与 V1 相同)
        risks = [s.f1_risk for s in population if s.is_feasible and s.f1_risk != float('inf')]
        costs = [s.f2_cost for s in population if s.is_feasible and s.f2_cost != float('inf')]
        
        if risks and costs:
            # 计算并记录风险的统计数据
            self.logs["risk_min"].append(float(np.min(risks)))
            self.logs["risk_mean"].append(float(np.mean(risks)))
            self.logs["risk_median"].append(float(np.median(risks)))

            # 计算并记录成本的统计数据
            self.logs["cost_min"].append(float(np.min(costs)))
            self.logs["cost_mean"].append(float(np.mean(costs)))
            self.logs["cost_median"].append(float(np.median(costs)))
        
        else:
            # 如果本代 *没有可行解*，记录一个标记值（重复前一代的值）
            self._log_fallback_values()

    def _log_fallback_values(self):
        """[辅助] 当没有有效可行解时，重复上一代的目标函数数据。"""
        # (此函数不影响 'cv' 日志, 'cv' 日志总是有值的)
        last_risk_min = self.logs["risk_min"][-1] if self.logs["risk_min"] else 0
        last_risk_mean = self.logs["risk_mean"][-1] if self.logs["risk_mean"] else 0
        last_risk_median = self.logs["risk_median"][-1] if self.logs["risk_median"] else 0
        
        last_cost_min = self.logs["cost_min"][-1] if self.logs["cost_min"] else 0
        last_cost_mean = self.logs["cost_mean"][-1] if self.logs["cost_mean"] else 0
        last_cost_median = self.logs["cost_median"][-1] if self.logs["cost_median"] else 0
        
        self.logs["risk_min"].append(last_risk_min)
        self.logs["risk_mean"].append(last_risk_mean)
        self.logs["risk_median"].append(last_risk_median)

        self.logs["cost_min"].append(last_cost_min)
        self.logs["cost_mean"].append(last_cost_mean)
        self.logs["cost_median"].append(last_cost_median)

# 回调 2: 用于文件日志 (生成 generations.txt)
class GenerationalFileLogger(Callback):
    """
    一个自定义的回调类，用于在每一代结束时，
    将 *专业的代际统计数据* 写入 'generations.txt' 文件。
    """
    def __init__(self, save_dir: str, file_name: str = "generations.txt"):
        super().__init__()
        self.log_file_path = os.path.join(save_dir, file_name)
        self._initialize_file()

    def _initialize_file(self):
        """创建日志文件并写入表头。"""
        header = (
            "===============================================================================\n"
            " n_gen | n_eval | n_nds  |      cv_min       |       cv_avg      |   risk_min     \n"
            "===============================================================================\n"
        )
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(header)
        except Exception as e:
            print(f"警告: 无法初始化日志文件 {self.log_file_path}: {e}")

    def on_generation_end(self, generation: int, population: List[Solution]):
        """
        [接口] 在每代结束时，计算统计数据并写入文件。
        """
        if not population:
            return

        pop_size = len(population)
        # (P_0 + Q_t)
        n_eval = pop_size + generation * pop_size 
        
        # 1. 计算非支配解 (Non-Dominated Solutions) 的数量
        n_nds = sum(1 for s in population if s.rank == 0)
        
        # 2. 计算约束违反度 (Constraint Violation)
        violations = [s.constraint_violation for s in population if s.constraint_violation != float('inf')]
        if violations:
            cv_min = np.min(violations)
            cv_avg = np.mean(violations)
        else:
            cv_min = 0.0
            cv_avg = 0.0
            
        # 3. 计算 Pareto前沿 上的最小风险 (V2 加固逻辑)
        rank_0_risks = [s.f1_risk for s in population if s.rank == 0 and s.is_feasible and s.f1_risk != float('inf')]
        if rank_0_risks:
            risk_min = np.min(rank_0_risks)
        else:
            # 如果 Rank 0 上没有可行解，就从 *所有* 可行解中找
            all_feasible_risks = [s.f1_risk for s in population if s.is_feasible and s.f1_risk != float('inf')]
            if all_feasible_risks:
                risk_min = np.min(all_feasible_risks)
            else:
                risk_min = 0.0  # 或者 np.nan

        # 格式化输出
        log_line = (
            f" {generation:>5} | {n_eval:>6} | {n_nds:>6} | {cv_min:17.10E} | {cv_avg:17.10E} | {risk_min:10.2f}\n"
        )
        
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except Exception as e:
            print(f"警告: 无法写入日志行到 {self.log_file_path}: {e}")