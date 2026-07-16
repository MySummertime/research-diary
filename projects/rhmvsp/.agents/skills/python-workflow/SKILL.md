---
name: python-workflow
description: Enforces structured directory organization and rigorous Conda environment execution rules for all Python scripts and experiments within this project.
allowed-tools: Read Write Edit Bash CMD
license: MIT license
metadata:
  version: "1.0"
  skill-author: Frances Jiang
---

# Python Workflow

## Overview

本规范定义了本项目中所有 Python 开发、脚本运行、实验以及学术绘图的标准工作流。通过严格约束代码存放路径、运行环境以及图表绘制标准，确保整个科研项目在不同平台上的可复现性，并使生成的学术配图直接达到国际高水平期刊的发表要求。

---

## When to Use This Skill

在遇到以下场景时，必须严格执行本工作流规范：

- 编写、生成或者修改任何 Python 源代码文件（`.py`）
- 运行项目中的数据处理、优化算法或者可视化脚本
- 为新项目初始化开发环境
- 生成图片
- 在本地或者远程服务器上配置自动化运行脚本

---

## Core Workflow

Python 项目的生命周期管理主要分为以下3个核心阶段：

### Phase 1: Directory Structure and Code Placement

为了保持项目根目录的清爽，需要根据脚本的性质分类存放。这里的“项目根目录”特指 **VS Code 当前打开的 Workspace（工作区）根目录**。

1. **识别脚本类型**：在保存或者生成任何 Python 文件之前，必须判断该脚本的定位。
2. **触发条件——辅助性工具脚本**：**仅当**该 `.py` 文件是作为**不属于项目核心代码的辅助性工具**时（例如：阅读 PDF 并提取文字、临时数据处理、文件格式转换、数据清洗、latex内容更新等），才需要将其统一放置到 `ai_scripts/` 目录下。
3. **创建目录**：若满足上述触发条件，且当前 Workspace 根目录下不存在 `ai_scripts/` 文件夹，必须在根目录下即时创建该文件夹。
4. **代码归档**：满足触发条件的辅助性工具脚本，必须统一放置在 `ai_scripts/` 文件夹中。
5. **非触发情况**：如果生成的 `.py` 文件属于**项目正式代码**（如核心算法模块、类定义、业务逻辑、被其他模块导入的依赖等），则**不需要**创建或使用 `ai_scripts/` 目录，应直接存放于其他合乎逻辑的核心目录（`src/`）。
6. **禁止**：在项目根目录直接放置任何 `.py` 文件。

### Phase 2: Environment Verification and Execution

为了防止全局环境污染，所有代码的执行必须锁定在专用的 Conda 虚拟环境中。

1. **探查现有环境**：在运行任何代码之前，必须首先扫描系统当前已经存在的环境列表。
2. **环境激活与使用**：
   - **如果环境已存在**：直接通过激活命令切入该环境，然后再执行代码。
   - **如果环境不存在**：必须立即基于项目的特定属性，新创建一个具有明确辨识度名称的全局虚拟环境（例如 `research-diary-env`）。
3. **标准运行序列**：确保在环境激活成功之后，再行调用 Python 解释器运行相应的代码。

**Note**: 本项目rhmvsp/的conda环境是rhmvsp，所有代码必须在该环境中运行，禁止在base环境中直接安装或运行任何与本项目相关的包。

### Phase 3: Academic Plotting and Figure Generation

如果使用 Python 进行数据可视化或科学绘图，生成的图片必须强制满足以下出版级别的美化规则：

1. **图表样式控制**：
   - **无标题与外框**：所有生成的图片内部**都不包含**任何总标题（No Title），且必须去除图表的最外层边框（No Outer Border）。
   - **轴线精简绘制**：如果图表只有 X 轴和 Y 轴，则**只绘制** X 轴和 Y 轴（隐藏上方和右侧的横纵坐标轴线）。如果图表包含 X 轴和双 Y 轴，则**只绘制** X 轴和这两个对称的 Y 轴。
2. **字体与数学符号**：
   - **统一字体**：如果是中文字符，则使用 **SimSun** 字体；除此之外，图中的所有文本、数学符号和数字统一使用 **Times New Roman** 字体。
   - **数学符号渲染**：如果图中出现数学符号，则 **必须经过 LaTeX 引擎进行渲染**（例如使用 Matplotlib 的 `text.usetex` 配置或使用 `$` 符号包裹）。
3. **图例与背景规范**：
   - **图例要求**：图中**必须包含**用以区分数据的图例（Legend），且图例本身**不需要外边框**。
   - **背景透明**：整个图片背景必须设置为**透明（Transparent）**，以便于后续排版排入不同背景色系的论文中。
4. **分辨率（DPI）控制**：
   - **彩色图片**：保存时必须指定 **DPI = 600**（保障高精度色彩表现）。
   - **黑白图片**：保存时必须指定 **DPI = 1000**（保障高对比度线条的极致清晰）。

---

## Best Practices

### 1. 目录结构最佳实践

- **模块化组织**：在 `ai_scripts/` 目录内部，建议根据功能进一步细分，比如将 PDF 处理脚本放入 `ai_scripts/pdf_utils/`，将数据转换脚本放入 `ai_scripts/data_utils/`。
- **清晰命名**：辅助脚本名称应直接反映其临时用途，便于后续识别和清理。
- **核心与辅助分离**：项目正式代码放在 `src/`，临时性辅助脚本放在 `ai_scripts/`。

### 2. 环境管理最佳实践

- **依赖显式化**：在创建完新的虚拟环境后，建议及时将环境内的包导出为 `environment.yml` 文件，方便团队其他成员一键同步。
- **避免环境交叉**：严禁在 `base` 环境中直接安装或者运行与本项目相关的科研计算包。

### 3. Matplotlib 绘图配置参考（RC Parameters）

为了在脚本中自动化应用 Phase 3 的规范，推荐在绘图脚本头部加入如下初始化配置：

```python
import matplotlib.pyplot as plt

# 启用 Times New Roman 与 LaTeX 渲染
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['text.usetex'] = True

# 示例：保存透明且高分辨率的图像
# plt.savefig('output.png', dpi=600, transparent=True, bbox_inches='tight')
```

---

## Common Pitfalls to Avoid

1. **滥用 ai_scripts/ 目录**：只有临时性的、不属于项目正式代码的辅助工具才放入 `ai_scripts/`，项目核心代码不应放入该目录。
2. **辅助脚本散落根目录**：符合触发条件的辅助脚本不应堆放在项目根目录，必须统一收纳进 `ai_scripts/` 目录。
3. **直接在 base 环境中跑代码**：这会引发严重的底层依赖包版本冲突，必须使用独立的虚拟环境。
4. **创建无名环境**：尽量避免使用路径前缀（`-p`）去创建没有名字的环境，这会导致环境难以通过名字全局激活，推荐直接使用具有明确语义的名称进行全局注册。
5. **带有冗余视觉元素**：忘记去除图表外边框、图例框或者带有默认标题，都会破坏学术论文配图的严谨性。

---

## Dependencies

### Required CLI Tools

- **Conda**（Miniconda 优先，Anaconda 发行版其次）
- **Python 3.14+**（根据具体学术项目的实际需求在环境中指定）

### Environment Initialization Example

以下是首次启动本项目时的标准环境配置与运行范例：

1. 确认并切换到当前 VS Code Workspace 的根目录。
2. 检查是否存在rhmvsp这个conda环境，若不存在则创建新环境。
3. 激活conda环境rhmvsp。
4. 需要生成辅助性工具脚本（如阅读 PDF、提取文字等）时，必须确保 `ai_scripts/` 目录存在。
5. 运行相应脚本。

---

## Summary

本 Python Workflow 提供：

1. **智能的目录层级规范**：在生成临时性、非项目核心代码的辅助工具（如 PDF 阅读、文字提取等）时，强制将脚本收拢于 `ai_scripts/` 目录；所有项目正式代码必须存放于 `src/`。
2. **安全的隔离运行环境**：强制通过有名字的 Conda 虚拟环境执行代码，彻底告别环境依赖冲突。
3. **顶刊级别的绘图美化**：将高标准学术配图规范直接写入底层规则，确保每一次实验产出的图表都具备国际发表质量。
