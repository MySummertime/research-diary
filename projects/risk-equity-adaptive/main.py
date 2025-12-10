# --- coding: utf-8 ---
# --- main.py ---
import time
import logging
from app.experiment_manager import Experiment


def main():
    wall_start = time.time()  # 总壁钟时间
    algo_process_start = 0  # 算法纯CPU时间
    algo_wall_start = 0  # 算法壁钟时间，作对比用

    exp = None

    try:
        # 1. 初始化(自动设置日志、备份、加载数据、预计算)
        exp = Experiment(config_path="config.json")

        # 2. 运行
        # --- 算法开始 ---
        algo_process_start = time.process_time()
        algo_wall_start = time.perf_counter()

        exp.run()

        # --- 算法结束 ---
        algo_process_end = time.process_time()
        algo_wall_end = time.perf_counter()

        # 3. 分析
        exp.analyze_and_report()

    except Exception as e:
        logging.error("\n--- 实验主流程发生致命错误 ---")
        logging.error(f"错误: {e}")

        import traceback

        logging.error(traceback.format_exc())

    finally:
        wall_end = time.time()

        logging.info("============================================")
        logging.info(f"实验总壁钟时间: {wall_end - wall_start:.3f} 秒")

        if "algo_process_start" in locals():  # 确保算法真的跑过
            process_time = algo_process_end - algo_process_start
            wall_algo_time = algo_wall_end - algo_wall_start
            logging.info(f"算法纯CPU时间: {process_time:.4f} 秒")
            logging.info(f"算法壁钟时间: {wall_algo_time:.3f} 秒（含IO、Gurobi回调等）")
            logging.info(
                f"CPU利用率估算: {(process_time / wall_algo_time) * 100:5.1f}%"
            )

            if wall_algo_time > 0:
                logging.info(
                    f"I/O与系统开销占比: {100 * (1 - process_time / wall_algo_time):.1f}%"
                )

        if exp and exp.save_dir:
            logging.info(f"所有结果已保存至: {exp.save_dir}")
        logging.info("============================================")


if __name__ == "__main__":
    main()
