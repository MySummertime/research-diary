# --- coding: utf-8 ---
# --- app/core/fuzzy.py ---
"""
[数学层] 模糊数学工具箱
对应论文 Section 3.1 & 3.5: Fuzzy Theory & Credibility Theory
只包含纯数学计算，不包含任何业务逻辑。
"""


class FuzzyMath:
    @staticmethod
    def triangular_expected_value(a: float, b: float, c: float) -> float:
        """
        计算三角模糊数 ξ=(a,b,c) 的可信性期望。
        Formula: E[ξ] = (a + 2b + c) / 4
        """
        return (a + 2 * b + c) / 4.0

    @staticmethod
    def trapezoidal_expected_value(a: float, b: float, c: float, d: float) -> float:
        """
        计算梯形模糊数 ζ=(a,b,c,d) 的可信性期望。
        Formula: E[ζ] = (a + b + c + d) / 4
        """
        return (a + b + c + d) / 4.0

    @staticmethod
    def triangular_pessimistic_value(
        a: float, b: float, c: float, alpha_c: float
    ) -> float:
        """
        计算三角模糊数 ξ=(a,b,c) 的 α_c-悲观值 (逆可信性分布)。
        用于处理机会约束: Cr{ξ <= x} >= alpha_c  <=>  x >= inf_alpha(ξ)
        """
        if alpha_c <= 0.5:
            return (2.0 * alpha_c) * b + (1.0 - 2.0 * alpha_c) * a
        else:
            return (2.0 * alpha_c - 1.0) * c + (2.0 - 2.0 * alpha_c) * b

    @staticmethod
    def trapezoidal_pessimistic_value(
        a: float, b: float, c: float, d: float, alpha_c: float
    ) -> float:
        """
        计算梯形模糊数 ζ=(a,b,c,d) 的 α_c-悲观值。
        """
        if alpha_c <= 0.5:
            return (2.0 * alpha_c) * b + (1.0 - 2.0 * alpha_c) * a
        else:
            return (2.0 * alpha_c - 1.0) * d + (2.0 - 2.0 * alpha_c) * c
