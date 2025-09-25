# --- coding: utf-8 ---
# --- test-python_interpreter.py ---

# 此文件用于打印并对比{项目路径, python解释器的搜索路径}
# ++++++++++++++ 侦察代码 ++++++++++++++
# 原因：
# Python 在寻找模块时，会遍历 sys.path 列表中的所有路径。
# Python 不会自动将当前工作目录 (cwd) 添加到 sys.path 中。
# 相反，Python 总是将被执行脚本所在的目录（在这里是 ...\tests）作为 sys.path 的第一项。
import sys
import os

print("\n" + "="*50)
print("   DIAGNOSTIC INFO FROM inside test-dtlz1.py")
print("="*50)
print(f"当前工作目录 (os.getcwd()):\n  {os.getcwd()}\n")
print("Python 模块搜索路径 (sys.path):")
for i, path in enumerate(sys.path):
    print(f"  [{i}] {path}")
print("="*50 + "\n")
# ++++++++++++++ 侦察代码到此为止 ++++++++++++++