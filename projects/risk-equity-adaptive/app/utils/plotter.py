# --- coding: utf-8 ---
# --- app/utils/plotter.py ---
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
             solutions: List[Solution],
             file_name: str = "pareto_front.png",
             xlabel: str="Objective 1",
             ylabel: str="Objective 2"):
        """
        绘制一个分层的二维帕累托前沿。
        """
        if not solutions:
            logging.warning("Plotter: 传入的解列表为空，无法绘图。")
            return

        # --- 1. 数据预处理 ---
        
        feasible_sols: List[Solution] = []
        infeasible_objectives: List[List[float]] = []
        
        for s in solutions:
            if s.f1_risk == float('inf') or s.f2_cost == float('inf'):
                continue
            
            if s.is_feasible:
                feasible_sols.append(s)
            else:
                infeasible_objectives.append([s.f1_risk, s.f2_cost])
        
        # --- 2. 检查是否有数据可画 ---
        if not feasible_sols and not infeasible_objectives:
            logging.warning("Plotter: 没有任何包含有效目标值的解，无法绘图。")
            return
            
        # --- 3. 计算 Baselines ---
        utopia_point, lowest_risk_point, lowest_cost_point = None, None, None
        
        if feasible_sols:
            # 1. 提取所有 f1 和 f2
            f1_vals = [s.f1_risk for s in feasible_sols]
            f2_vals = [s.f2_cost for s in feasible_sols]
            
            # 2. 找到 Utopia Point (理想点)
            utopia_point = [np.min(f1_vals), np.min(f2_vals)]
            
            # 3. 找到 Baseline 1 (最低风险解)
            lowest_risk_sol = min(feasible_sols, key=lambda s: s.f1_risk)
            lowest_risk_point = [lowest_risk_sol.f1_risk, lowest_risk_sol.f2_cost]
            
            # 4. 找到 Baseline 2 (最低成本解)
            lowest_cost_sol = min(feasible_sols, key=lambda s: s.f2_cost)
            lowest_cost_point = [lowest_cost_sol.f1_risk, lowest_cost_sol.f2_cost]
            
            # 5. 从 feasible_sols 中分类出 Ranks
            feasible_groups: Dict[int, List[List[float]]] = { 0: [], 1: [], 2: [], 99: [] }
            for s in feasible_sols:
                obj_pair = [s.f1_risk, s.f2_cost]
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
            # 如果没有可行解，feasible_groups 就是空的
            feasible_groups = { 0: [], 1: [], 2: [], 99: [] }

            
        # --- 4. 开始分层绘图---
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(self.title, fontsize=16)

        # 样式定义 (zorder 决定了谁在最上面)
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
            0: { # Rank 0 (The Star)
                'data': feasible_groups[0], 'c': 'red', 'marker': 'o', 
                'alpha': 0.9, 's': 80, 'edgecolors': 'black', 
                'label_key': 'Feasible Pareto Front (Rank 0)', 
                'zorder': 10, 'legend_order': 1 
            }
        }
        
        # 5. 绘制 Baselines (在绘制其他点之前) 
        if utopia_point:
            ax.scatter(
                utopia_point[0], utopia_point[1], 
                c='blue', marker='*', s=300, 
                edgecolors='black', label=f'Utopia Point (Ref) ({len(feasible_sols)} sols)', 
                zorder=8    # 在 Rank 0 下面，但在其他 Rank 上面
            )
        if lowest_risk_point:
            ax.scatter(
                lowest_risk_point[0], lowest_risk_point[1], 
                c='green', marker='P', s=200, 
                edgecolors='black', label=f'Lowest Risk (Baseline)', 
                zorder=9
            )
        if lowest_cost_point:
            ax.scatter(
                lowest_cost_point[0], lowest_cost_point[1], 
                c='green', marker='D', s=200, 
                edgecolors='black', label=f'Lowest Cost (Baseline)', 
                zorder=9
            )
        
        # 6. 绘制所有数据点
        for key in sorted(styles.keys(), key=lambda k: styles[k]['zorder']):
            style = styles[key]
            data = style['data']
            if data:
                data_np = np.array(data)
                plot_kwargs = {
                    'c': style['c'], 'marker': style['marker'],
                    'alpha': style['alpha'], 's': style['s'],
                    'label': f'{style["label_key"]} ({len(data_np)})',
                    'zorder': style['zorder']
                }
                if 'edgecolors' in style:
                    plot_kwargs['edgecolors'] = style['edgecolors']
                ax.scatter(data_np[:, 0], data_np[:, 1], **plot_kwargs)

        # --- 7. 美化 ---
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0)) 
        
        # 8. 图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # 图例排序
            label_to_key_map = {}
            for key, style in styles.items():
                label_name = f"{style['label_key']} ({len(style['data'])})"
                label_to_key_map[label_name] = key
            
            # 添加新的 Baselines
            legend_order_map = {
                'Utopia Point (Ref)': 1.1,
                'Lowest Risk (Baseline)': 1.2,
                'Lowest Cost (Baseline)': 1.3
            }
            
            sorted_items = []
            for h, l in zip(handles, labels):
                base_label = l.split(' (')[0]
                
                # 检查是不是 Baseline/Utopia
                if base_label in legend_order_map:
                    order = legend_order_map[base_label]
                    sorted_items.append((h, l, order))
                    continue
                
                # 检查是不是 Ranks
                style_key = None
                for lbl_key, s_key in label_to_key_map.items():
                    if lbl_key.startswith(base_label):
                        style_key = s_key
                        break
                
                if style_key is not None:
                    order = styles[style_key]['legend_order']
                    sorted_items.append((h, l, order))
            
            sorted_items.sort(key=lambda item: item[2]) # 按 order 升序
            sorted_handles = [h for h, l, o in sorted_items]
            sorted_labels = [l for h, l, o in sorted_items]
            
            ax.legend(sorted_handles, sorted_labels, fontsize=12)

        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # --- 9. 保存 ---
        full_path = os.path.join(self.save_dir, file_name)
        try:
            plt.savefig(full_path, dpi=300) 
            plt.close(fig) 
            logging.info(f"分层帕累托前沿图像 (带Baselines) 已保存至: {full_path}")
        except Exception as e:
            logging.error(f"Plotter: 保存图像失败: {e}")
            plt.close(fig)

def plot_parallel_coordinates(rank_0_solutions: List[Solution], save_dir: str):
    """
    为 Rank 0 解绘制平行坐标图 (Parallel Coordinate Plot)。
    """
    file_path = os.path.join(save_dir, "parallel_coordinate_plot.png")
    logging.info(f"正在绘制平行坐标图并保存至: {file_path}")

    if not rank_0_solutions:
        logging.warning("PCP: 没有 Rank 0 解，跳过绘图。")
        return

    try:
        data = np.array([[s.f1_risk, s.f2_cost] for s in rank_0_solutions])
        f1_min, f1_max = data[:, 0].min(), data[:, 0].max()
        f2_min, f2_max = data[:, 1].min(), data[:, 1].max()
        f1_range = f1_max - f1_min if (f1_max - f1_min) > 0 else 1.0
        f2_range = f2_max - f2_min if (f2_max - f2_min) > 0 else 1.0
        
        norm_data = np.zeros_like(data)
        norm_data[:, 0] = (data[:, 0] - f1_min) / f1_range
        norm_data[:, 1] = (data[:, 1] - f2_min) / f2_range
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(len(norm_data)):
            color = plt.get_cmap('hot')(norm_data[i, 0]) 
            ax.plot([0, 1], norm_data[i, :], marker='o', color=color, alpha=0.5)

        ax.set_title("Parallel Coordinate Plot (Rank 0 Solutions)", fontsize=16) 
        ax.set_xticks([0, 1], ["Total Risk (f1)", "Total Cost (f2)"], fontsize=12) 
        ax.set_yticks([0, 0.5, 1], 
                   [f"Min\n({f1_min:.2e}, {f2_min:.2e})", 
                    "Mean", 
                    f"Max\n({f1_max:.2e}, {f2_max:.2e})"])
        ax.grid(True, axis='y', linestyle='--', alpha=0.7) 
        ax.set_ylabel("Normalized Objective Value", fontsize=12) 
        
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap('hot'), 
            norm=Normalize(vmin=f1_min, vmax=f1_max)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Risk (f1) Level', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close(fig) 
        
    except Exception as e:
        logging.error(f"绘制平行坐标图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        else:
            plt.close() 