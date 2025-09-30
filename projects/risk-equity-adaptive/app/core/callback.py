# --- coding: utf-8 ---
# --- callback.py ---
import numpy as np

from pymoo.core.callback import Callback    # type: ignore
from typing import Dict, List


class GenerationalLogger(Callback):
    """
    一个自定义的回调类，用于在算法的每一代结束时，记录下关键的统计数据。
    """
    def __init__(self):
        super().__init__()
        # 初始化一个字典来存储历史数据
        self.logs: Dict[str, List[float]] = {
            "risk_min": [], "risk_mean": [], "risk_median": [],
            "cost_min": [], "cost_mean": [], "cost_median": []
        }

    def notify(self, algorithm):
        # 这个方法会在每一代结束后被 pymoo 自动调用
        objectives = algorithm.pop.get("F")
        
        # 过滤掉无效解（目标值为1e20）
        valid_objectives = objectives[objectives[:, 0] < 1e19]
        
        if len(valid_objectives) > 0:
            risks = valid_objectives[:, 0]
            costs = valid_objectives[:, 1]

            # 计算并记录风险的统计数据
            self.logs["risk_min"].append(np.min(risks))
            self.logs["risk_mean"].append(np.mean(risks))
            self.logs["risk_median"].append(np.median(risks))

            # 计算并记录成本的统计数据
            self.logs["cost_min"].append(np.min(costs))
            self.logs["cost_mean"].append(np.mean(costs))
            self.logs["cost_median"].append(np.median(costs))
        else:
            # 如果某一代全是无效解，记录一个标记值（例如重复前一代的值）
            last_risk_min = self.logs["risk_min"][-1] if self.logs["risk_min"] else 0
            last_cost_min = self.logs["cost_min"][-1] if self.logs["cost_min"] else 0
            self.logs["risk_min"].append(last_risk_min)
            self.logs["risk_mean"].append(last_risk_min)
            self.logs["risk_median"].append(last_risk_min)
            self.logs["cost_min"].append(last_cost_min)
            self.logs["cost_mean"].append(last_cost_min)
            self.logs["cost_median"].append(last_cost_min)