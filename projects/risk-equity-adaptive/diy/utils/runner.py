# --- coding: utf-8 ---
# --- /utils/runner.py ---
import sys
import os
import subprocess
from typing import List, Optional

class ProjectRunner:
    """
    一个通用的项目脚本启动器。
    它负责处理路径问题、环境设置，并以交互方式运行指定目录下的脚本。
    因为：
        Python 在寻找模块时，会遍历 sys.path 列表中的所有路径。
        Python 不会自动将当前工作目录 (cwd) 添加到 sys.path 中。
        相反，Python 总是将被执行脚本所在的目录（在这里是 utils/）作为 sys.path 的第一项。
        为了避免出现自定义模块导入错误，需要将项目根目录添加到 PYTHONPATH 中.
    """
    def __init__(self, script_dir_name: str):
        """
        初始化启动器。

        Args:
            script_dir_name (str): 存放待运行脚本的目录名称。
        """
        self.project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.script_dir: str = os.path.join(self.project_root, script_dir_name)
        self.python_executable: str = sys.executable

    def _find_scripts(self) -> List[str]:
        """[辅助方法] 扫描脚本目录，找到所有可运行的 .py 脚本。"""
        try:
            all_files = os.listdir(self.script_dir)
            # 筛选并排序，保证每次菜单顺序一致
            demo_scripts = sorted([f for f in all_files if f.endswith('.py') and not f.startswith('__')])
            return demo_scripts
        except FileNotFoundError:
            print(f"❌ 错误: 找不到脚本目录 '{self.script_dir}'")
            return []

    def _get_user_choice(self, scripts: List[str]) -> Optional[str]:
        """[辅助方法] 显示菜单并获取用户的有效选择。"""
        if not scripts:
            print(f"❌ 在 '{self.script_dir}' 目录下没有找到可运行的脚本。")
            return None

        print("\n--- 请选择要运行的脚本 ---")
        for i, script_name in enumerate(scripts):
            display_name, _ = os.path.splitext(script_name)
            print(f"  [{i + 1}] {display_name}")
        print("---------------------------------")

        while True:
            try:
                raw_input = input(f"请输入选项编号 (1-{len(scripts)}): ")
                choice = int(raw_input)
                if 1 <= choice <= len(scripts):
                    return scripts[choice - 1]
                else:
                    print(f"❌ 输入无效，请输入 1 到 {len(scripts)} 之间的数字。")
            except (ValueError, KeyboardInterrupt):
                print("\n操作取消。")
                return None
    
    def run(self) -> None:
        """
        执行主流程：寻找脚本、获取用户选择、并以正确的环境运行脚本。
        """
        scripts = self._find_scripts()
        chosen_script = self._get_user_choice(scripts)

        if chosen_script:
            script_path = os.path.join(self.script_dir, chosen_script)
            
            # 创建包含正确 PYTHONPATH 的环境变量
            new_env = os.environ.copy()
            new_env['PYTHONPATH'] = f"{self.project_root}{os.pathsep}{new_env.get('PYTHONPATH', '')}"

            print(f"\n🐍 正在启动脚本: {chosen_script}")
            print(f"   工作目录: {self.project_root}")
            print(f"   PYTHONPATH: {new_env['PYTHONPATH']}\n" + "-"*50)

            result = subprocess.run(
                [self.python_executable, script_path],
                cwd=self.project_root,
                env=new_env
            )
            print("-"*50 + f"\n✅ 脚本 '{chosen_script}' 执行完毕，退出码: {result.returncode}\n")