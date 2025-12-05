# --- coding: utf-8 ---
# --- app/utils/metrics.py ---
import numpy as np
from typing import List
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from app.core.solution import Solution


class MetricCalculator:
    """
    [核心指标计算器]
    封装 Pymoo 的指标库，适配本项目的 Solution 对象。
    """

    def __init__(self, ref_point: List[float]):
        """
        初始化指标计算器
        :param ref_point: HV 计算的参考点 [ref_f1_risk, ref_f2_cost]
                          注意：必须大于所有解的最差值 (Nadir Point)
        """
        self.ref_point = np.array(ref_point)

    def calculate_hv(self, solutions: List[Solution]) -> float:
        """
        计算超体积 (Hypervolume)
        衡量解集的收敛性和多样性。越大越好。
        """
        if not solutions:
            return 0.0

        # 提取目标值矩阵 (N, 2)
        # F1: Risk, F2: Cost
        F = np.array([[s.f1_risk, s.f2_cost] for s in solutions])

        # Pymoo 的 HV 计算器
        ind = HV(ref_point=self.ref_point)
        return ind(F)

    def calculate_igd(
        self, solutions: List[Solution], true_pareto_front: np.ndarray
    ) -> float:
        """
        计算反转世代距离 (IGD) [已归一化]
        衡量解集到真实前沿的平均距离。越小越好。
        :param true_pareto_front: 近似真实前沿的目标值矩阵 (M, 2)
        """
        if not solutions or len(true_pareto_front) == 0:
            return float("inf")

        # 1. 提取目标值
        F = np.array([[s.f1_risk, s.f2_cost] for s in solutions])
        PF = true_pareto_front

        # 2. 归一化 (Normalization)
        # 使用 Reference Front 的最大最小值作为边界
        # 这样可以消除 Cost (1e6) 和 Risk (1e3) 的量级差异
        min_vals = np.min(PF, axis=0)
        max_vals = np.max(PF, axis=0)

        # 避免分母为 0
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0

        # 归一化到 [0, 1]
        F_norm = (F - min_vals) / range_vals
        PF_norm = (PF - min_vals) / range_vals

        # 3. 调用 Pymoo 计算归一化后的 IGD
        ind = IGD(PF_norm)
        return ind(F_norm)

    def calculate_sm(self, solutions: List[Solution]) -> float:
        """
        计算间距指标 (Spacing Metric, SM) [Schott 1995]
        衡量解分布的均匀程度。越小越均匀。
        这对应论文中的 "Social Equity" 分析。
        """
        if len(solutions) < 2:
            return 0.0

        # 1. 按 f1 (risk) 排序，保证计算相邻距离是有序的
        sorted_sols = sorted(solutions, key=lambda s: s.f1_risk)
        F = np.array([[s.f1_risk, s.f2_cost] for s in sorted_sols])

        # 2. 计算相邻解的曼哈顿距离 (L1 norm)
        # d_i = |f1_i - f1_{i+1}| + |f2_i - f2_{i+1}|
        # axis=1 表示对每一行操作
        d = np.sum(np.abs(F[:-1] - F[1:]), axis=1)

        # 3. 计算标准差
        d_mean = np.mean(d)

        # SM = sqrt( (1 / (n-1)) * sum( (d_i - d_mean)^2 ) )
        sm = np.sqrt(np.sum((d - d_mean) ** 2) / (len(solutions) - 1))

        return sm
