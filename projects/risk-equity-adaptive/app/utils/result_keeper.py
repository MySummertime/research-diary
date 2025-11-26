# --- coding: utf-8 ---
# --- app/utils/result_keeper.py ---
# 负责实验目录创建和日志管理。
import os
import sys
import time
import csv
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
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # 3. 控制台处理器 (StreamHandler) - “安静”模式
    console_handler = logging.StreamHandler(sys.stdout)
    # 使用 VERBOSE_FORMATTER
    console_handler.setFormatter(VERBOSE_FORMATTER)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    logging.info("日志系统已启动。(文件日志级别: INFO, 控制台日志级别: INFO)")
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


# ----------------------------------------
# 3. 结果保存器
# ----------------------------------------
def save_rank0_solutions_csv(rank_0_solutions: List[Solution], save_dir: str):
    """
    将所有 Rank 0 的解的详细路径保存到 'optimal_solutions.csv'。
    """
    file_path = os.path.join(save_dir, "optimal_solutions.csv")
    logging.info(f"Pareto Optimal solutions saved to: {file_path}")

    # 定义 CSV 表头
    headers = [
        "solution_index",  # 解的编号 (e.g., 1, 2, 3...)
        "task_id",  # 任务 ID (e.g., "T1")
        "origin_id",  # 任务起点
        "destination_id",  # 任务终点
        "f1_risk_total",  # (该解的) 总风险
        "f2_cost_total",  # (该解的) 总成本
        "path_nodes",  # 路径节点
        "path_arcs_mode",  # 路径弧段(模式)
    ]

    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            # 遍历传入的 Rank 0 解列表
            for i, sol in enumerate(rank_0_solutions):
                sol_index = i + 1

                # 遍历该解中的每一条路径 (每一个任务)
                for task_id, path in sol.path_selections.items():
                    if not path.task:
                        continue

                    # 提取路径信息
                    nodes_str = " -> ".join([node.node_id for node in path.nodes])
                    arcs_str = " -> ".join(
                        [
                            f"({arc.start.node_id},{arc.end.node_id})({arc.mode})"
                            for arc in path.arcs
                        ]
                    )

                    writer.writerow(
                        [
                            sol_index,
                            task_id,
                            path.task.origin.node_id,
                            path.task.destination.node_id,
                            f"{sol.f1_risk:.4f}",  # 记录该解的总目标值
                            f"{sol.f2_cost:.4f}",  # 记录该解的总目标值
                            nodes_str,
                            arcs_str,
                        ]
                    )

    except Exception as e:
        logging.error(f"保存 PF_solutions.csv 失败: {e}")
