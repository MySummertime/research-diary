module.exports = {
  '**/*.{js,jsx,ts,tsx,mjs,cjs}': ['eslint --fix', 'prettier --write'],
  '**/*.{md,json,jsonc,yml,yaml,toml,css,scss,html}': ['prettier --write'],
  '**/*.py': [
    // 以下命令需要在能够访问到 Conda 环境中的 Black/Flake8/MyPy 的环境下运行
    // 先激活 Conda 环境，再进行 lint-staged 操作
    'black --line-length 100', // 确保这个配置与 pyproject.toml 匹配
    'flake8', // Flake8 会读取 setup.cfg 或 .flake8
    'mypy', // MyPy 会读取 mypy.ini 或 pyproject.toml
    'git add', // 格式化和 Lint 后重新暂存
  ],
};
