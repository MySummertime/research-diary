# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List
from app.core.solution import Solution

class ParetoPlotter:
    """
    一个专门用于可视化帕累托前沿的类。
    """
    def __init__(self, title: str = "Final Pareto Front", save_dir: str = "results"):
        self.title = title
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot(self, 
             solutions: List[Solution], # 接收 List[Solution]
             file_name: str = "pareto_front.png",
             xlabel: str="Objective 1",
             ylabel: str="Objective 2"):
        """
        绘制一个“诚实”的二维帕累托前沿。

        - 可行解 (Feasible) 将被绘制为 红色 'o'。
        - 不可行解 (Infeasible) (但仍是 Rank 0) 将被绘制为 灰色 'x'。
        """
        
        if not solutions:
            logging.warning("Plotter: 传入的解列表为空，无法绘图。")
            return

        # 1. 将解分为“可行”和“不可行”两组
        feasible_objectives = []
        infeasible_objectives = []
        
        for s in solutions:
            # 确保有合法的数值
            if s.f1_risk != float('inf') and s.f2_cost != float('inf'):
                obj_pair = [s.f1_risk, s.f2_cost]
                if s.is_feasible:
                    feasible_objectives.append(obj_pair)
                else:
                    infeasible_objectives.append(obj_pair)
        
        # 2. 检查是否有数据可画
        if not feasible_objectives and not infeasible_objectives:
            logging.warning("Plotter: 没有任何包含有效目标值的解，无法绘图。")
            return
            
        # 3. 开始绘图
        fig, ax = plt.subplots(figsize=(12, 8)) # 放大画布
        ax.set_title(self.title, fontsize=16)

        # 4. 绘制“冰山” (不可行的解)
        if infeasible_objectives:
            infeas_np = np.array(infeasible_objectives)
            ax.scatter(infeas_np[:, 0], infeas_np[:, 1], 
                       c='#AAAAAA',       # 灰色
                       marker='x',         # 'x' 标记
                       alpha=0.6,          # 半透明
                       s=50,               # 标记大小
                       label=f'Infeasible (Rank 0) ({len(infeas_np)})')

        # 5. 绘制“冰山尖” (可行的解)
        if feasible_objectives:
            feas_np = np.array(feasible_objectives)
            ax.scatter(feas_np[:, 0], feas_np[:, 1], 
                       c='red',            # 红色
                       marker='o',         # 'o' 标记
                       alpha=0.9,          # 不透明
                       s=80,               # 标记更大
                       edgecolors='black', # 加个黑边，更清晰
                       label=f'Feasible Pareto Front ({len(feas_np)})')

        # 6. 美化
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # 使用科学计数法处理大数字
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0)) 
        
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 7. 保存
        full_path = os.path.join(self.save_dir, file_name)
        try:
            plt.savefig(full_path, dpi=300) # 提高分辨率
            plt.close(fig) # 释放内存
            logging.info(f"帕累托前沿图像已保存至: {full_path}")
        except Exception as e:
            logging.error(f"Plotter: 保存图像失败: {e}")
            plt.close(fig) # 即使失败也要释放内存