# --- coding: utf-8 ---
# --- result_keeper.py ---
import os
import logging
from typing import List
from contextlib import contextmanager

def create_experiment_directory(base_dir: str = "results") -> str:
    """
    在基础目录下创建一个新的、按顺序命名的实验文件夹。

    Args:
        base_dir (str, optional): 存放所有实验结果的基础目录。默认为 "results"。

    Returns:
        str: 新创建的实验文件夹的完整路径。
    """
    # 确保基础目录存在
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. 计算下一个实验的编号
    # 列出基础目录下的所有条目，并筛选出名为“实验n”的文件夹
    existing_experiments: List[str] = [
        d for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("实验")
    ]
    next_experiment_num = len(existing_experiments) + 1
    
    # 2. 创建本次实验专属的文件夹
    experiment_save_dir = os.path.join(base_dir, f"实验{next_experiment_num}")
    os.makedirs(experiment_save_dir, exist_ok=True)
    
    print(f"=================================================")
    print(f" 本次实验结果将保存至: {experiment_save_dir}")
    print(f"=================================================")
    
    return experiment_save_dir

# === 全局变量，用于存储处理器和格式化器 ===
_file_handler = None
_verbose_formatter = None
_clean_formatter = None

def setup_logging(log_dir: str, log_name: str = "experiment_log.txt"):
    """
    配置一个全局的根日志记录器，并设置两种不同的格式化器。
    """
    global _file_handler, _verbose_formatter, _clean_formatter
    
    log_file_path = os.path.join(log_dir, log_name)
    
    # 创建两种格式化器
    _verbose_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _clean_formatter = logging.Formatter('%(message)s')

    # 创建文件处理器
    _file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(_verbose_formatter) # 默认使用带时间戳的格式

    # 获取并配置根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(_file_handler)


@contextmanager
def log_section(clean: bool = False):
    """
    一个上下文管理器，用于临时切换日志格式。
    在 'with' 代码块内，日志格式将被改变；退出后，将自动恢复。
    
    Args:
        clean (bool, optional): 是否切换到不带时间戳的纯净格式。默认为 False。
    """
    global _file_handler, _verbose_formatter, _clean_formatter
    
    # 检查全局变量是否已初始化
    if _file_handler is None:
        yield
        return

    original_formatter = _file_handler.formatter
    try:
        if clean and _clean_formatter:
            _file_handler.setFormatter(_clean_formatter)
        elif not clean and _verbose_formatter:
            _file_handler.setFormatter(_verbose_formatter)
        yield
    finally:
        # 无论 'with' 代码块内部发生什么，最终都会恢复原始的格式
        if original_formatter:
            _file_handler.setFormatter(original_formatter)