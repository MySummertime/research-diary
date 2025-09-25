# --- coding: utf-8 ---
# --- dtlz1.py ---
import numpy as np
from core.problem import Problem

# --- 一个基准测试函数，使用Problem基类 ---
class DTLZ1(Problem):
    def __init__(self, n_vars=7, n_obj=3):
        """
        DTLZ1 测试函数。
        这是一个标准的、用于测试多目标优化算法性能的数学问题。
        该问题有 n_vars 个决策变量和 n_obj 个目标函数。
        n_vars = n_obj + k - 1, usually with k=5

        Args:
            n_vars (int): 决策空间的维度，决策空间的形状为 (n_pop, n_vars)
            ^ n_pop, n_vars = decision_vars.shape
            n_obj (int): 目标空间的维度，即目标函数的数量

        Returns:
            np.ndarray: 目标函数值矩阵，形状为 (n_pop, n_obj)
        """
        super().__init__(n_vars, n_obj, np.zeros(n_vars), np.ones(n_vars))

    def evaluate(self, x):
        """
        DTLZ1 目标函数的“绝对可靠”版实现。
        """
        n_obj = self.n_obj
        
        # DTLZ1 中，g(x) 函数依赖于最后 k 个决策变量，k = n_vars - n_obj + 1
        # 对于 3 目标、7 变量问题，k=5，这 5 个变量是 x[:, 2:]
        x_m = x[:, n_obj - 1:]
        
        # 计算 g 函数
        # 这是 Rastrigin-like 函数，最小值在所有 x_m 均为 0.5 时取到 g=0
        g = 100 * (x_m.shape[1] + np.sum((x_m - 0.5)**2 - np.cos(20 * np.pi * (x_m - 0.5)), axis=1))
        
        # 为了后续计算方便，将 g 变成 (n_pop, 1) 的列向量
        g_col = g.reshape(-1, 1)

        # --- 直接、清晰地计算每个目标，不使用循环 ---
        # 提取前两个决策变量 x0 和 x1
        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)

        # 计算 f1, f2, f3
        f1 = 0.5 * (1 + g_col) * x0 * x1
        f2 = 0.5 * (1 + g_col) * x0 * (1 - x1)
        f3 = 0.5 * (1 + g_col) * (1 - x0)
        
        # 将三个目标列合并成最终的目标矩阵
        objs = np.hstack((f1, f2, f3))
        
        return objs