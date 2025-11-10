# --- coding: utf-8 ---
# --- main.py ---
import time
import logging
from app.experiment_manager import Experiment

def main():
    start_time = time.time()
    exp = None

    try:
        # 1. 初始化(自动设置日志、备份、加载数据、预计算)
        exp = Experiment(config_path="config.json", seed=4)

        # 2. 运行
        exp.run()

        # 3. 分析
        exp.analyze_and_report()

    except Exception as e:
        logging.error("\n--- 实验主流程发生致命错误 ---")
        logging.error(f"错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    finally:
        end_time = time.time()
        logging.info("==========================================")
        logging.info(f"实验总耗时: {end_time - start_time:.2f} 秒。")
        if exp and exp.save_dir:
            logging.info(f"所有结果已保存至: {exp.save_dir}")
        logging.info("==========================================")

if __name__ == '__main__':
    main()