# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List, Dict
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
        绘制一个分层的二维帕累托前沿。
        """
        
        if not solutions:
            logging.warning("Plotter: 传入的解列表为空，无法绘图。")
            return

        # --- 1. 数据分层处理 ---
        feasible_groups: Dict[int, List[List[float]]] = {
            0: [],  # Rank 0
            1: [],  # Rank 1
            2: [],  # Rank 2
            99: []  # Rank 3+ (用 99 代表 'other')
        }
        infeasible_objectives: List[List[float]] = []
        
        for s in solutions:
            if s.f1_risk == float('inf') or s.f2_cost == float('inf'):
                continue
            obj_pair = [s.f1_risk, s.f2_cost]
            
            if s.is_feasible:
                rank = s.rank
                if rank == 0:
                    feasible_groups[0].append(obj_pair)
                elif rank == 1:
                    feasible_groups[1].append(obj_pair)
                elif rank == 2:
                    feasible_groups[2].append(obj_pair)
                else:
                    feasible_groups[99].append(obj_pair)    # Rank 3 及以上
            else:
                infeasible_objectives.append(obj_pair)
        
        has_data = any(len(v) > 0 for v in feasible_groups.values()) or len(infeasible_objectives) > 0
        if not has_data:
            logging.warning("Plotter: 没有任何包含有效目标值的解，无法绘图。")
            return
            
        # --- 3. 开始分层绘图 ---
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(self.title, fontsize=16)

        # 使用 'legend_order' 键用于控制图例顺序
        styles = {
            'infeasible': {
                'data': infeasible_objectives, 'c': '#AAAAAA', 'marker': 'x', 
                'alpha': 0.5, 's': 40, 'label_key': 'Infeasible', 
                'zorder': 2, 'legend_order': 5
            },
            99: { # Rank 3+
                'data': feasible_groups[99], 'c': '#e0e0e0', 'marker': '.', 
                'alpha': 0.8, 's': 30, 'label_key': 'Dominated (Rank 3+)', 
                'zorder': 3, 'legend_order': 4
            },
            2: { # Rank 2
                'data': feasible_groups[2], 'c': '#adb5bd', 'marker': '+', 
                'alpha': 0.8, 's': 50, 'label_key': 'Dominated (Rank 2)', 
                'zorder': 4, 'legend_order': 3
            },
            1: { # Rank 1
                'data': feasible_groups[1], 'c': '#6c757d', 'marker': 's', 
                'alpha': 0.7, 's': 40, 'label_key': 'Dominated (Rank 1)', 
                'zorder': 5, 'legend_order': 2
            },
            0: { # Rank 0, i.e. The Star
                'data': feasible_groups[0], 'c': 'red', 'marker': 'o', 
                'alpha': 0.9, 's': 80, 'edgecolors': 'black', 
                'label_key': 'Feasible Pareto Front (Rank 0)', 
                'zorder': 10, 'legend_order': 1 # legend_order=1 (排第一)
            }
        }

        # 按 zorder 顺序绘图 (从底层到顶层)
        # zorder: infeasible -> Rank 3+ -> Rank 2 -> Rank 1 -> Rank 0
        for key in sorted(styles.keys(), key=lambda k: styles[k]['zorder']):
            style = styles[key]
            data = style['data']
            if data:
                data_np = np.array(data)
                # 1. 先准备好所有通用参数
                plot_kwargs = {
                    'c': style['c'],
                    'marker': style['marker'],
                    'alpha': style['alpha'],
                    's': style['s'],
                    'label': f'{style["label_key"]} ({len(data_np)})',
                    'zorder': style['zorder']
                }

                # 2. 只有当 'edgecolors' 被明确定义时，才把它加进去
                #  (这样 'none' 和 'None' 都不会被传递)
                if 'edgecolors' in style:
                    plot_kwargs['edgecolors'] = style['edgecolors']
                
                # 3. 用 **kwargs 把它传递进去
                ax.scatter(data_np[:, 0], data_np[:, 1], **plot_kwargs)

        # --- 4. 美化 ---
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0)) 
        
        # 使用 'legend_order' 键来排序图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # 创建一个反向查找表
            label_to_key_map = {}
            for key, style in styles.items():
                label_name = f"{style['label_key']} ({len(style['data'])})"
                label_to_key_map[label_name] = key
            
            # 创建 (handle, label, order) 元组
            sorted_items = []
            for h, l in zip(handles, labels):
                style_key = label_to_key_map.get(l) # 用完整标签匹配
                if style_key is not None:
                    order = styles[style_key]['legend_order']
                    sorted_items.append((h, l, order))
            
            # 按 legend_order 升序排列 (1, 2, 3, 4, 5)
            sorted_items.sort(key=lambda item: item[2])
            
            # 重新解包
            sorted_handles = [h for h, l, o in sorted_items]
            sorted_labels = [l for h, l, o in sorted_items]
            
            ax.legend(sorted_handles, sorted_labels, fontsize=12)

        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # --- 5. 保存 ---
        full_path = os.path.join(self.save_dir, file_name)
        try:
            plt.savefig(full_path, dpi=300)
            plt.close(fig)
            logging.info(f"分层帕累托前沿图像已保存至: {full_path}")
        except Exception as e:
            logging.error(f"Plotter: 保存图像失败: {e}")
            plt.close(fig)