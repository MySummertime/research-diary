# --- coding: utf-8 ---
# --- plotter.py ---

import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # type: ignore

class ParetoPlotter:
    """
    一个专门用于可视化帕累托前沿的类。
    """
    def __init__(self, title: str = "Final Pareto Front", save_dir: str = "results"):
        self.title = title
        self.save_dir = save_dir
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

    def plot(self, objectives: np.ndarray, file_name: str = "pareto_front.png"):
        """
        根据目标值的维度，自动绘制 2D 或 3D 的帕累托前沿。

        Args:
            objectives (np.ndarray): 目标函数值矩阵。
            file_name (str): 保存图片的文件名。
        """
        if objectives is None or objectives.shape[0] == 0:
            print("警告：传入的目标值为空，无法绘图。")
            return
            
        n_obj = objectives.shape[1]
        fig = plt.figure(figsize=(10, 8))

        if n_obj == 2:
            # 建立一个 1x1 的二维子图
            ax = fig.add_subplot(111)   # 使用 fig.add_sublot(1, 1, 1) 也可以
            self._plot_2d(ax, objectives)
        elif n_obj == 3:
            # 建立一个 1x1 的三维子图
            ax = fig.add_subplot(111, projection='3d')
            assert isinstance(ax, Axes3D)   # 确保 ax 是 Axes3D 类型的对象
            self._plot_3d(ax, objectives)
        else:
            print(f"可视化仅支持2或3个目标，当前目标数为: {n_obj}。")
            plt.close(fig)
            return

        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 使用配置好的路径和文件名进行保存
        full_path = os.path.join(self.save_dir, file_name)
        plt.savefig(full_path)
        print(f"帕累托前沿图像已保存至: {full_path}")
        
        plt.show()

    def _plot_2d(self, ax, objectives: np.ndarray):
        """[辅助方法] 绘制二维帕累托前沿。"""
        ax.set_title(self.title)
        ax.scatter(objectives[:, 0], objectives[:, 1], c='r', marker='o', label='Pareto Front')
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')

    def _plot_3d(self, ax, objectives: np.ndarray):
        """[辅助方法] 绘制三维帕累托前沿。"""
        ax.set_title(self.title)
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='r', marker='o', label='Pareto Front')
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.view_init(elev=30, azim=45)  # elev: 仰角, azim: 方位角