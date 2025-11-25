# --- coding: utf-8 ---
# --- app/utils/callback.py ---
import os
import csv
import logging
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
            "risk_min": [],
            "risk_mean": [],
            "risk_median": [],
            "cost_min": [],
            "cost_mean": [],
            "cost_median": [],
            "cv_min": [],
            "cv_avg": [],
        }

    def on_generation_end(self, generation: int, population: List[Solution]):
        """
        [接口] 在每代结束时，从 Solution 列表中计算统计数据。
        """
        if not population:
            return

        # --- 步骤 1 - 立即计算约束违反度 (CV) ---
        # 无论种群是否可行，我们都跟踪CV
        violations = [
            s.constraint_violation
            for s in population
            if s.constraint_violation != float("inf")
        ]
        if violations:
            self.logs["cv_min"].append(float(np.min(violations)))
            self.logs["cv_avg"].append(float(np.mean(violations)))
        else:
            # 如果没有违反（例如，所有解都可行，或者种群为空），则CV为0
            self.logs["cv_min"].append(0.0)
            self.logs["cv_avg"].append(0.0)

        # --- 步骤 2: 仅计算 *可行解* 的目标函数统计 ---
        risks = [
            s.f1_risk for s in population if s.is_feasible and s.f1_risk != float("inf")
        ]
        costs = [
            s.f2_cost for s in population if s.is_feasible and s.f2_cost != float("inf")
        ]

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
        last_risk_median = (
            self.logs["risk_median"][-1] if self.logs["risk_median"] else 0
        )

        last_cost_min = self.logs["cost_min"][-1] if self.logs["cost_min"] else 0
        last_cost_mean = self.logs["cost_mean"][-1] if self.logs["cost_mean"] else 0
        last_cost_median = (
            self.logs["cost_median"][-1] if self.logs["cost_median"] else 0
        )

        self.logs["risk_min"].append(last_risk_min)
        self.logs["risk_mean"].append(last_risk_mean)
        self.logs["risk_median"].append(last_risk_median)

        self.logs["cost_min"].append(last_cost_min)
        self.logs["cost_mean"].append(last_cost_mean)
        self.logs["cost_median"].append(last_cost_median)


# 回调 2: 用于文件日志 (生成 generations.csv)
class GenerationalFileLogger(Callback):
    """
    一个自定义的回调类.
    将 *每一代* 的摘要统计信息实时写入 "generations.csv".
    """

    def __init__(self, save_dir: str, file_name: str = "generations.csv"):
        super().__init__()
        self.csv_file_path = os.path.join(save_dir, file_name)
        self.headers = [
            "generation",
            "n_total",
            "n_feasible",
            "n_nds",
            "cv_min",
            "cv_mean",
            "cv_median",
            "risk_min",
            "risk_mean",
            "risk_median",
            "cost_min",
            "cost_mean",
            "cost_median",
        ]
        self.csv_file = None
        self.csv_writer = None
        try:
            # 立即创建并写入表头
            with open(self.csv_file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            # 2. 重新打开文件准备“追加”
            self.csv_file = open(self.csv_file_path, "a", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            logging.info(f"代际收敛日志将保存至: {self.csv_file_path}")
        except Exception as e:
            logging.error(f"GenerationalFileLogger 无法初始化 CSV: {e}")
            # 如果打开文件出错，也要尝试关闭它
            if self.csv_file:
                self.csv_file.close()
            self.csv_writer = None

    def on_generation_end(self, generation: int, population: List[Solution]):
        if not self.csv_writer:
            return

        try:
            # 与 GenerationalLogger 相同的统计逻辑
            n_total = len(population)
            if n_total == 0:
                return

            feasible_sols = [s for s in population if s.is_feasible]
            n_feasible = len(feasible_sols)

            cvs = [s.constraint_violation for s in population if not s.is_feasible]
            cv_min = np.min(cvs) if cvs else 0.0
            cv_mean = np.mean(cvs) if cvs else 0.0
            cv_median = np.median(cvs) if cvs else 0.0

            if feasible_sols:
                risks = [s.f1_risk for s in feasible_sols]
                costs = [s.f2_cost for s in feasible_sols]
                n_nds = sum(1 for s in feasible_sols if s.rank == 0)

                risk_min, risk_mean, risk_median = (
                    np.min(risks),
                    np.mean(risks),
                    np.median(risks),
                )
                cost_min, cost_mean, cost_median = (
                    np.min(costs),
                    np.mean(costs),
                    np.median(costs),
                )
            else:
                n_nds = 0
                risk_min, risk_mean, risk_median = 0.0, 0.0, 0.0
                cost_min, cost_mean, cost_median = 0.0, 0.0, 0.0

            # 写入 CSV 行
            self.csv_writer.writerow(
                [
                    generation,
                    n_total,
                    n_feasible,
                    n_nds,
                    f"{cv_min:.4f}",
                    f"{cv_mean:.4f}",
                    f"{cv_median:.4f}",
                    f"{risk_min:.4f}",
                    f"{risk_mean:.4f}",
                    f"{risk_median:.4f}",
                    f"{cost_min:.4f}",
                    f"{cost_mean:.4f}",
                    f"{cost_median:.4f}",
                ]
            )
            # 增加 flush 确保实时写入
            self.csv_file.flush()

        except Exception as e:
            logging.warning(f"GenerationalFileLogger 写入 CSV 失败: {e}")

    def __del__(self):
        # 在对象被销毁时关闭文件
        if hasattr(self, "csv_file") and self.csv_file:
            self.csv_file.close()
