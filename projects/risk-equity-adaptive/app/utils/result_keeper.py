# --- coding: utf-8 ---
# --- app/utils/result_keeper.py ---
# 负责实验目录创建和日志管理。
import os
import time
import json
import logging
from typing import List
from contextlib import contextmanager
from app.core.solution import Solution

# ----------------------------------------
# 1. 目录创建器
# ----------------------------------------
def create_experiment_directory(base_dir: str = "results") -> str:
    """
    在 base_dir 下创建一个带时间戳的唯一实验目录。
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 使用更清晰的打印
    print("=================================================")
    print(f" 本次实验结果将保存至: {exp_dir}")
    print("=================================================")
    
    return exp_dir

# ----------------------------------------
# 2. 高级日志系统
# ----------------------------------------

# === 全局变量，用于存储处理器和格式化器 ===
_file_handler = None
_console_handler = None
_verbose_formatter = None
_clean_formatter = None

def setup_logging(log_dir: str, log_name: str = "experiment_log.txt"):
    """
    配置一个全局的根日志记录器，设置两种格式，并同时输出到文件和控制台。
    """
    global _file_handler, _console_handler, _verbose_formatter, _clean_formatter
    
    log_file_path = os.path.join(log_dir, log_name)
    
    # 创建两种格式化器
    _verbose_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(message)s', 
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    _clean_formatter = logging.Formatter('%(message)s')

    # --- 创建处理器 ---
    
    # 1. 文件处理器
    _file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(_verbose_formatter)  # 默认使用带时间戳的格式

    # 2. 控制台处理器
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(_verbose_formatter)   # 默认也使用带时间戳的格式

    # --- 配置根记录器 ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除在 main.py 中可能已被设置的旧 handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(_file_handler)
    root_logger.addHandler(_console_handler)


@contextmanager
def log_section(clean: bool = False):
    """
    一个上下文管理器，用于临时切换 *所有* 处理器 (文件+控制台) 的日志格式。
    
    Args:
        clean (bool, optional): 是否切换到不带时间戳的纯净格式。默认为 False。
    """
    global _file_handler, _console_handler, _verbose_formatter, _clean_formatter
    
    handlers = [_file_handler, _console_handler]
    handlers = [h for h in handlers if h is not None]   # 过滤掉 None
    
    if not handlers or _verbose_formatter is None or _clean_formatter is None:
        yield
        return

    original_formatters = [h.formatter for h in handlers]
    
    try:
        if clean:
            for h in handlers:
                h.setFormatter(_clean_formatter)
        else:
            for h in handlers:
                h.setFormatter(_verbose_formatter)
        yield
    finally:
        # 无论 'with' 代码块内部发生什么，最终都会恢复原始的格式
        for h, original_formatter in zip(handlers, original_formatters):
            if original_formatter:
                h.setFormatter(original_formatter)

# ----------------------------------------
# 3. 结果保存器
# ----------------------------------------
def save_results_json(solutions: List[Solution], output_path: str):
    """
    将最终的Pareto前沿 (f1, f2, 路径) 保存为 JSON 文件。
    """
    logging.info(f"正在保存 {len(solutions)} 个解到 {output_path}...")
    results = []
    for sol in solutions:
        # if sol.is_feasible:
        paths_str_map = {task_id: " -> ".join([n.id for n in p.nodes]) for task_id, p in sol.path_selections.items()}
        results.append({
            "is_feasible": sol.is_feasible,
            "constraint_violation": round(sol.constraint_violation, 4), 
            "rank": sol.rank,
            "f1_risk": sol.f1_risk,
            "f2_cost": sol.f2_cost,
            "paths": paths_str_map,
            "eta_values": sol.eta_values
        })
    
    # 按 f1 (Risk) 排序
    # results.sort(key=lambda x: x["f1_risk"])
    # 按可信性、约束违反度、风险排序
    results.sort(key=lambda x: (not x["is_feasible"], x["constraint_violation"], x["f1_risk"]))

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info("JSON 结果保存成功。")
    except Exception as e:
        logging.error(f"保存 JSON 结果时失败: {e}")