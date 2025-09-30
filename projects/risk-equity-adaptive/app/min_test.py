# --- minimal_test.py ---
# 一个绝对最小化的Pymoo测试脚本

import numpy as np
from pymoo.core.crossover import Crossover

class TaskSliceCrossover(Crossover):
    def __init__(self):
        # 保证把常用参数传给父类
        super().__init__(n_parents=2, n_offsprings=2, prob=0.9)
        print("TaskSliceCrossover initialized")

    def do(self, problem, X, **kwargs):
        # 在进入任何 shape-check 前把原始 X 打印出来（便于诊断）
        print(">>> do() called, raw X.shape =", getattr(X, "shape", None))
        # 规范化为 (n_matings, n_parents, n_var)
        if X is None:
            raise AssertionError("do() received X == None")

        if isinstance(X, np.ndarray):
            if X.ndim == 3:
                X3 = X
            elif X.ndim == 2:
                # 情况 A: 这是单次交叉父代 (n_parents, n_var)
                if X.shape[0] == self.n_parents:
                    X3 = X[np.newaxis, :, :]
                # 情况 B: 可能是整个人口 (pop_size, n_var) —— 若 pop_size 可被 n_parents 整除，则 reshape
                elif X.shape[0] % self.n_parents == 0:
                    n_matings = X.shape[0] // self.n_parents
                    X3 = X.reshape(n_matings, self.n_parents, X.shape[1])
                    print(f">>> do(): interpreted 2D X as full-pop reshaped to {X3.shape}")
                else:
                    # fallback：取前 n_parents 行作为单一 mating（保守策略）
                    print(">>> do(): WARNING: 2D X with rows not divisible by n_parents; using first n_parents rows as single mating")
                    X3 = X[:self.n_parents][np.newaxis, :, :]
            else:
                raise AssertionError(f"Shape is incorrect of crossover impl. (ndim={X.ndim})")
        else:
            raise AssertionError(f"do() expects numpy.ndarray for X, got {type(X)}")

        # 最后再保证第二维等于 n_parents
        if X3.shape[1] != self.n_parents:
            raise AssertionError(f"After reshape, expected n_parents={self.n_parents}, got {X3.shape[1]}")

        # 现在直接调用 _do（跳过 super().do 的额外断言），并打印返回 shape
        Y = self._do(problem, X3, **kwargs)
        print(">>> _do returned shape:", getattr(Y, "shape", None))
        return Y

    def _do(self, problem, X, **kwargs):
        # 这里假定 X 已经是 (n_matings, n_parents, n_var)
        if X.ndim != 3:
            raise AssertionError("TaskSliceCrossover._do expects a 3D-array")

        n_matings, n_parents, n_var = X.shape
        # 你的实际交叉逻辑在这里 —— 我先放一个安全的默认实现（直接复制父代）
        Y = np.empty((n_matings, self.n_offsprings, n_var), dtype=X.dtype)
        for i in range(n_matings):
            p0 = X[i, 0, :].copy()
            p1 = X[i, 1, :].copy()
            # 简单复制（仅用于通过自检）；把这里替换为你的 task-slice 逻辑
            Y[i, 0, :] = p0
            Y[i, 1, :] = p1
        return Y