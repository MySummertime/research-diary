# --- coding: utf-8 ---
# --- run_test.py ---
from utils.runner import ProjectRunner

if __name__ == "__main__":
    # 实例化启动器，告诉解释器我想测试的脚本在根目录下的哪个文件夹里
    # 如果用的是 'tests' 文件夹，就写 'test'
    runner = ProjectRunner(script_dir_name="tests")
    
    # 调用 run 方法，启动想测试的python文件
    # 这个方法会自动处理路径问题，避免 ImportError
    runner.run()