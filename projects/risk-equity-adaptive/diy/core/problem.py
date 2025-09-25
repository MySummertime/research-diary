# --- coding: utf-8 ---
# --- problem.py ---

# --- 一个基础的问题类，可以继承和定义自己的问题 ---
class Problem:
    def __init__(self, n_vars, n_obj, xl, xu):
        """
        定义优化问题的基类

        Args:
            n_vars (int): The dimension of decision space, i.e., number of decision variables(input)
            n_obj (int): The dimension of objective space, i.e., number of objectives(output)
            xl (np.ndarray): The lower bound of decision variables
            xu (np.ndarray): The upper bound of decision variables
        """
        self.n_vars = n_vars
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu

    def evaluate(self, x):
        """
        计算给定决策变量 x 的目标函数值。
        这个方法应该在子类中被重写。
        """
        # 在这里抛出异常，强制子类实现这个方法
        raise NotImplementedError("请在子类中实现 'evaluate' 方法！")