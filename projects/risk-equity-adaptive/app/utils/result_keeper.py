# --- coding: utf-8 ---
# --- app/utils/result_keeper.py ---
# 负责实验目录创建和日志管理。
import os
import sys
import time
import logging
from contextlib import contextmanager


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

# 全局常量
VERBOSE_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
CLEAN_FORMATTER = logging.Formatter("%(message)s")


def setup_logging(log_dir: str, log_name: str = "experiment.log"):
    """
    配置一个全局的根日志记录器.
    根据分级，选择性输出到 文件 和/或 控制台.
    """
    log_path = os.path.join(log_dir, log_name)
    root_logger = logging.getLogger()

    # 清除任何旧的logging处理器
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. 根(Root)设置为“最低”级别
    root_logger.setLevel(logging.DEBUG)

    # 2. 文件处理器 (FileHandler) - “啰嗦”模式
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    # 使用 VERBOSE_FORMATTER
    file_handler.setFormatter(VERBOSE_FORMATTER)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # 3. 控制台处理器 (StreamHandler) - “安静”模式
    console_handler = logging.StreamHandler(sys.stdout)
    # 使用 VERBOSE_FORMATTER
    console_handler.setFormatter(VERBOSE_FORMATTER)
    console_handler.setLevel(logging.WARNING)
    root_logger.addHandler(console_handler)

    logging.info("日志系统已启动。(文件日志级别: INFO, 控制台日志级别: WARNING)")
    logging.info(f"日志将保存至: {log_path}")


@contextmanager
def log_section(clean: bool = False):
    """
    一个上下文管理器，用于临时切换 *所有* 处理器 (文件+控制台) 的日志格式。
    动态地从 root_logger 查找处理器

    Args:
        clean (bool, optional): 是否切换到不带时间戳的纯净格式。默认为 False。
    """

    root_logger = logging.getLogger()
    handlers = root_logger.handlers

    if not handlers:
        yield
        return

    original_formatters = [h.formatter for h in handlers]

    try:
        if clean:
            for h in handlers:
                h.setFormatter(CLEAN_FORMATTER)
        else:
            for h in handlers:
                h.setFormatter(VERBOSE_FORMATTER)
        yield
    finally:
        # 无论 'with' 代码块内部发生什么，最终都会恢复原始的格式
        for h, original_formatter in zip(handlers, original_formatters):
            if original_formatter:
                h.setFormatter(original_formatter)
