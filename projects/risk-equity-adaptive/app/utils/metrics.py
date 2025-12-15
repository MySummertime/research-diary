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
    封装 Pymoo 的指标库，并基于全局边界强制执行归一化 (Normalization)。
    """

    def __init__(self, ideal_point: np.ndarray, nadir_point: np.ndarray):
        """
        初始化指标计算器。

        Args:
            ideal_point: 全局理想点 (Min bounds) [min_risk, min_cost]
            nadir_point: 全局最差点 (Max bounds) [max_risk, max_cost]
        """
        self.ideal = np.array(ideal_point)
        self.nadir = np.array(nadir_point)

        # 计算量程，防止分母为 0
        self.range = self.nadir - self.ideal
        self.range[self.range < 1e-6] = 1.0  # 避免除以零

    def _normalize(self, F: np.ndarray) -> np.ndarray:
        """
        将目标值矩阵 F 归一化到 [0, 1] 区间。
        Formula: F_norm = (F - Ideal) / (Nadir - Ideal)
        """
        return (F - self.ideal) / self.range

    def calculate_hv(self, solutions: List[Solution]) -> float:
        """
        计算归一化超体积 (Normalized Hypervolume)。
        衡量解集的收敛性和多样性。越大越好。
        """
        if not solutions:
            return 0.0

        # 1. 提取目标值
        F = np.array([[s.f1_risk, s.f2_cost] for s in solutions])

        # 2. 归一化! ✨
        # 归一化后，理想点变为 [0,0]，最差点变为 [1,1]
        F_norm = self._normalize(F)

        # 3. 计算 HV
        # 参考点设为 [1.1, 1.1] 以确保包含边界解
        # 因为我们已经归一化了，所以参考点固定为 1.1 即可
        ref_point = np.array([1.1, 1.1])
        ind = HV(ref_point=ref_point)

        return ind(F_norm)

    def calculate_igd(
        self, solutions: List[Solution], true_pareto_front: np.ndarray
    ) -> float:
        """
        计算反转世代距离 (IGD) [已归一化]。
        衡量解集到真实前沿的平均距离。越小越好。

        Args:
            true_pareto_front: 原始尺度的真实前沿 (M, 2)
        """
        if not solutions or len(true_pareto_front) == 0:
            return float("inf")

        # 1. 提取目标值
        F = np.array([[s.f1_risk, s.f2_cost] for s in solutions])
        PF = true_pareto_front

        # 2. 归一化! ✨
        # 必须使用与 HV 相同的全局边界进行归一化，保证指标的一致性
        F_norm = self._normalize(F)
        PF_norm = self._normalize(PF)

        # 3. 计算 IGD
        ind = IGD(PF_norm)
        return ind(F_norm)

    def calculate_sm(self, solutions: List[Solution]) -> float:
        """
        计算间距指标 (Spacing Metric, SM) [Schott 1995]。
        衡量解分布的均匀程度。越小越均匀。
        注：SM 在归一化空间计算更准确，避免某个大数值维度主导距离。
        """
        if len(solutions) < 2:
            return 0.0

        # 1. 提取并归一化
        F = np.array([[s.f1_risk, s.f2_cost] for s in solutions])
        F_norm = self._normalize(F)

        # 2. 按 f1 (risk) 排序
        # argsort 返回排序后的索引
        sorted_indices = np.argsort(F_norm[:, 0])
        F_sorted = F_norm[sorted_indices]

        # 3. 计算相邻解的曼哈顿距离 (L1 norm)
        # d_i = |f1_i - f1_{i+1}| + |f2_i - f2_{i+1}|
        d = np.sum(np.abs(F_sorted[:-1] - F_sorted[1:]), axis=1)

        # 4. 计算标准差
        d_mean = np.mean(d)

        # SM = sqrt( (1 / (n-1)) * sum( (d_i - d_mean)^2 ) )
        # 自由度为 n-1
        sm = np.sqrt(np.sum((d - d_mean) ** 2) / (len(solutions) - 1))

        return sm


def build_reference_front(all_solutions_F: List[np.ndarray]) -> np.ndarray:
    """
    [Utility] 基于所有已知解的目标值，构建近似的真实帕累托前沿 (Reference Front)。
    用于计算 IGD。
    Args:
        all_solutions_F: 包含多个 (N, 2) 矩阵的列表
    """
    if not all_solutions_F:
        return np.empty((0, 2))

    # 1. 合并所有点
    combined_F = np.vstack(all_solutions_F)
    # 2. 去重
    combined_F = np.unique(combined_F, axis=0)

    # 3. 简单的非支配排序 (Extract Rank 0)
    # 由于只有 2 个目标，可以高效过滤
    is_dominated = np.zeros(len(combined_F), dtype=bool)
    for i in range(len(combined_F)):
        if is_dominated[i]:
            continue
        # 检查是否被其他点支配
        # A dominates B if A <= B and A != B (Minimization)
        # 向量化比较优化速度
        dominators = np.all(combined_F <= combined_F[i], axis=1) & np.any(
            combined_F < combined_F[i], axis=1
        )
        if np.any(dominators):
            is_dominated[i] = True

    ref_front = combined_F[~is_dominated]
    # 按 Risk 排序，整齐
    if len(ref_front) > 0:
        ref_front = ref_front[ref_front[:, 0].argsort()]

    return ref_front
