# RHMVSP 框架定义

## 1. 科学问题与创新点

本项目实现了一个面向危险品多式联运网络的可靠性优化调度框架。核心科学问题是：

- 在具有风险不确定性和时间不确定性的多模态运输网络中，如何同时满足风险可靠性和时间可靠性约束？
- 如何将危险品道路与铁路运输、转运节点、班列时刻表、车辆重用等混合调度问题纳入统一的分支定价模型？
- 在不确定性建模和精确求解之间，如何评估可靠模型与常规 LP 放松的性能差异？

本项目的主要创新点包括：

1. 使用**不确定性理论**构造风险和时间的不确定变量，风险采用布尔不确定变量，时间采用梯形/三角形模糊变量；
2. 将**时变事故概率**和**时变后果暴露**引入危险品运输弧的风险评估，体现昼夜与高峰时段风险波动；
3. 在多式联运路网中统一定义风险可靠性和时间可靠性，并在求解过程中以**Branch-and-Price**精确求解可行路径。

## 2. 不确定性定义

### 2.1 风险不确定性：布尔不确定变量

项目中每条运输弧和转运节点的事故状态由布尔不确定变量建模：

- `BoolUncertainVariable(alpha)` 表示事故发生的信任度 $\alpha$（即该弧/节点不发生事故的可信度）；
- 其风险可靠性定义为 $R = 1 - \alpha$。

代码实现：`src/core/uncertainty.py` 中的 `BoolUncertainVariable`。

此外，项目引入了**时间依赖的风险乘数**：

- `phi(t)` 表示事故概率随时段的折算倍率；
- `psi(t)` 表示事故后果随人口暴露时段的放大倍率；
- 这两个函数分别在 `src/core/uncertainty.py` 中的 `_traffic_multiplier()` 和 `_population_exposure_multiplier()` 实现。

实际风险计算采用时间区间上的精确分段积分：

- 对运输窗口 $[t_{start}, t_{end}]$ 进行分段积分，计算平均 $\phi(t)$；
- 有效事故概率为 $\min(0.95, \alpha_{base} \times \text{平均}\phi(t))$；
- 后果成本可进一步按 $\psi(t)$ 加权。

### 2.2 时间不确定性：模糊行程时间

项目使用模糊变量表现运输和转运时间：

- `TrapezoidalFuzzy(a,b,c,d)` 为通用梯形模糊变量；
- 对于公路弧，`b=c` 时退化为等效三角形；
- 对于铁路弧，则保持平台段，反映班列运行不确定性。

时间不确定测度的形式为：

$$
\mathcal{M}\{\tilde{Z} \leqslant L\} = \begin{cases}
0, & L \leqslant a \\\\[4pt]
\dfrac{L-a}{2(b-a)}, & a < L \leqslant b \\\\[4pt]
0.5, & b < L \leqslant c \\\\[4pt]
1 - \dfrac{d-L}{2(d-c)}, & c < L \leqslant d \\\\[4pt]
1, & L \geqslant d
\end{cases}
$$

`UncertaintyEngine` 使用参数化规则生成模糊时间：

- `make_transport_time(distance, mode)` 为道路/铁路弧生成 `TrapezoidalFuzzy`；
- 道路基准时间 $base = distance / road_speed$，生成 $a=base \times a_{ratio}$，$b=c=base$，$d=base \times c_{ratio}$；
- 铁路基准时间 $base = distance / rail_speed$，使用 `rail_fuzzy_time_ratio` 计算 $(a,b,c,d)$；
- `make_transfer_time()` 为转运时间生成基准 1h 的梯形模糊变量，其扩展受 `delta_T` 控制。

代码实现：`src/core/uncertainty.py` 中的 `UncertaintyEngine`。

## 3. 可靠性定义

### 3.1 风险可靠性层次

风险可靠性采用串联系统最弱链原则：

- 弧可靠性 $R_{ij} = 1 - \alpha_{ij}$；
- 节点可靠性 $R_i = 1 - \gamma_i$；
- 路径可靠性 $R(P) = \min\{R_{arc}, R_{node}\}$；
- OD 对可靠性 $R_{od} = \min\{R(P)\}$；
- 网络风险可靠性 $R_{net}^{risk} = \min\{R_{od}\}$。

在配置中，风险阈值通过 `alpha_max` 控制：

- `beta_od = 1 - alpha_max`；
- `beta_net = beta_od`。

这些约束体现在 `ReliabilityAnalyzer.check_arc_risk()`、`check_node_risk()` 和 `compute_path_risk()` 中。

### 3.2 时间可靠性层次

时间可靠性由模糊行程时间能否满足截止时间 $L_{od}$ 衡量：

- 计算路径上所有弧与转运时间的模糊和；
- 生成路径总模糊时间 $\tilde{Z}_{total}$；
- 评估 $R_T = \mathcal{M}\{s + \tilde{Z}_{total} \leqslant L_{od}\}$。

`ReliabilityAnalyzer.compute_time_reliability_od()` 将离散式转运时间、道路/铁路运输时间、班列时刻窗口和 `alpha_T` 结合，输出：

- `time_window_feasible`：确定性等价到达时间是否不超过 $L_{od}$；
- `R_T`：实际连续时间可靠性数值；
- `arrival_time_upper` / `arrival_time_lower`：模糊路径的边界。

对于 $\alpha_T > 0.5$ 的情况，时间可靠性约束可用确定性等价表示：

$$
L_{od} \geqslant (2\alpha_T - 1) d + 2(1-\alpha_T) c
$$

其中 $(c,d)$ 来自路径总模糊时间的右尾参数。

### 3.3 综合可靠性

完整解的网络可靠性由 `ReliabilityAnalyzer.analyze_solution()` 计算：

- 对每个 OD 的所有批次路径分别计算 `R_od_risk` 和 `R_od_time`；
- OD 级别采用最弱批次 `min()` 聚合；
- 网络级可靠性取所有 OD 的最小值。

最终返回结果字典包含 `R_net_risk`、`R_net_time` 和 `od_reliability_details`。

## 4. 模型构成

### 4.1 网络结构与模式

项目采用多式联运网络建模：

- 公路弧（mode=1）和铁路弧（mode=2）；
- 转运节点用于道路与铁路之间的换装；
- OD 对由原点、目标、需求量、软时窗等信息构成；
- `MultimodalNetwork.from_instance()` 在 `src/core/network.py` 中解析实例。

### 4.2 成本构成

成本项在 `config.py` 中定义，主要包括：

- 车辆部署成本 `c_deploy_road`、`c_deploy_rail`、`c_deploy_rent`；
- 单位运输成本 `c_road`、铁路两部制 `c_rail_base_fixed`、`c_rail_base_var`、`c_rail_hazmat_markup`、`c_rail_fund`；
- 转运成本 `c_transfer` 和转运设施成本 `c_hub`；
- 空驶重定位成本 `c_empty_reposition_per_km`；
- 机遇成本 `c_opp_base`；
- 库存持有成本 `c_holding`。

这些成本在分支定价模型中一同构成 `total_cost`。

### 4.3 不确定性参数

关键参数来源于 `config.py`：

- `alpha_max`：弧级风险阈值；
- `alpha_T`：时间可靠性阈值；
- `delta_T`：模糊时间波动因子；
- `rail_fuzzy_time_ratio`：铁路模糊时间形状；
- `road_speed` / `rail_speed`：基准速度；
- `risk_scale`：时间变化风险强度；
- `lp_only`：对比实验的 LP 放松开关。

### 4.4 时变风险与时间窗设计

项目通过 `risk_scale` 开关在 `UncertaintyEngine` 中控制是否启用时变风险：

- `enable_time_varying_risk` 和 `enable_time_varying_consequence` 控制 `phi(t)` / `psi(t)` 的激活；
- `time_varying_alpha()` 和 `time_varying_consequence()` 实现窗口内风险与后果的平均化。

OD 截止时间 `L_od` 由基础期望到达时间 `E_od` 加上 `config.L_od_buffer_hours` 确定。敏感性分析中，`alpha_T` 变化会驱动 `L_od` 重算。

### 4.5 解决方案结果结构

`BranchAndPriceController.solve()` 输出包含：

- `total_cost`、`deploy_cost`、`transport_cost`、`transfer_cost`、`reposition_cost`、`holding_cost`、`opportunity_cost`；
- `R_net_risk`、`R_net_time`；
- `od_reliability_details`；
- `iterations`、`cg_history`、`total_columns`、`gap_pct`；
- `transport_routes`、`rail_consolidation_summary`、`active_hubs` 等路线和资源信息。

## 5. 算法

### 5.1 Branch-and-Price 框架

求解器的核心由 `src/solvers/branch_and_price.py` 实现。流程为：

1. 初始化主问题 `MasterProblem` 和定价问题 `PricingProblem`；
2. 生成初始可行列；
3. 对每个节点执行列生成：
   - 求解 LP 主问题得到对偶值；
   - 先执行启发式定价（`exact=False`），再执行精确定价（`exact=True`）；
   - 新列被包装成 `Column` 并加入节点；
4. 若无需分支或 `lp_only=True`，则停止；
5. 否则构建分支约束并继续 Best-First Search。

### 5.2 LP-only 基准

实验中通过 `config.lp_only = True` 启用 LP 模型作为 benchmark，这使算法在根节点列生成后停止，不进行分支求解。该基准用于收敛对比和外样本验证。

### 5.3 算法配置与稳定化

分支定价参数在 `config.py` 中定义：

- `bap_max_iterations`：列生成最大迭代；
- `bap_pricing_columns_limit`、`bap_ng_route_size`：定价策略限幅；
- `bap_stabilization`、`stabilization_mu`：对偶值稳定化；
- `max_labels_exact`、`max_labels_heuristic`：ESPPRC 标签数限制；
- `time_limit`：求解时间上限。

`BranchAndPriceController` 的 `cg_history` 记录每次列生成后的 LP 下界和当前最优上界，用于收敛性分析。

## 6. 实验与结果

### 6.1 主实验入口

主入口为 `main.py`，支持模式：

- `demo`：加载 `data/{size}/instance.json` 并直接求解；
- `solve`：求解指定实例 `--instance`；
- `generate`：生成并保存 benchmark 实例。

实验结果保存至 `results/exp_{timestamp}_{size}`，并输出：

- `config.json`、`rhmvsp.log`、`solution.json`；
- 可选 `network.png`、`solution_plot.png`；
- 日志中包含 `R_net_risk`、`R_net_time`、`total_cost` 等指标。

### 6.2 敏感性实验

敏感性实验由 `sensitivity/` 下多个脚本组织。每个实验均产生单独 `results/` 子目录、日志、`solution.json` 以及 CSV/图表。

#### 6.2.1 `alpha_max` 敏感性

- 脚本：`sensitivity/sensitivity_alpha_max.py`
- 变量：`alpha_max_values = [0.01, 0.015, ..., 0.10]`
- 目标：分析 `alpha_max` 对网络风险可靠性 `R_risk`、时间可靠性 `R_time` 和成本的影响。
- 输出：`summary_alpha_max.csv`、双 Y 轴折线图。

#### 6.2.2 `alpha_T` 敏感性

- 脚本：`sensitivity/sensitivity_alpha_T.py`
- 变量：`alpha_T_values = [0.05, 0.1, ..., 0.5]`
- 目标：分析时间可靠性阈值 `alpha_T` 对可行性和成本的影响。
- 输出：`summary_alpha_T.csv`、双 Y 轴折线图。

#### 6.2.3 收敛性比较

- 脚本：`sensitivity/sensitivity_convergence.py`
- 内容：比较 `BAP` 与 `LP-only` 的上界/下界收敛曲线；
- 目标：验证分支定价算法在给定参数组合下的收敛性与界差。
- 输出：`convergence.png` / `convergence.tiff`，`cg_history` 数据。

#### 6.2.4 风险-时间耦合实验

- 脚本：`sensitivity/sensitivity_risk_time_coupled.py`
- 变量网格：`alpha_max` 与 `delta_T` 的 10×10 组合；
- 目标：探索风险约束与时间不确定性之间的耦合贸易面；
- 输出：`summary_coupled.csv`、三张热力图：成本、`R_risk`、`R_time`。

#### 6.2.5 模糊时间价值分析

- 脚本：`sensitivity/sensitivity_value_of_fuzzy_time.py`
- 变量：`delta_T` 从非常确定（0.01）到高度不确定（1.5）；
- 目标：量化建模模糊旅行时间对可靠性和成本的影响；
- 输出：`summary_delta_T.csv`、双 Y 轴折线图及成本对比条形图。

#### 6.2.6 外样本实验

- 脚本：`sensitivity/sensitivity_out_of_sample.py`
- 目标：对 6 组典型 `(alpha_T, alpha_max)` 组合进行 Monte Carlo 方案抽样，比较 `BAP` 与 `LP-only` 的稳健性；
- 输出：`experiment.csv`、`out_of_sample_boxplot.png`、`out_of_sample_boxplot.tiff`。

#### 6.2.7 确定性模型 vs 不确定性感知模型对比

- 脚本：`sensitivity/sensitivity_deterministic_vs_uncertain.py`
- 目标：进行确定性模型（无不确定性建模）与不确定性感知模型（完整不确定性建模）的头对头对比，量化同时考虑模糊旅行时间不确定性和风险可靠性约束带来的价值。
- 实验设计：
  - **确定性模型**：`delta_T ≈ 0`（无模糊时间扩散，使用确定性等价旅行时间）且 `alpha_max = 1.0`（无风险可靠性约束，所有路径均被接受）
  - **不确定性感知模型**：`delta_T > 0`（使用模糊旅行时间分布）且 `alpha_max = 0.05`（风险可靠性约束激活）
  - 固定 `alpha_T = 0.8`，横轴为环境不确定性水平 `delta_T`，取值范围 `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]`
  - 两个模型共享相同的 Monte Carlo 方案样本（相同的随机种子和场景实现），以确保对比的公平性，唯一的差异在于求解器是否建模模糊旅行时间和风险可靠性约束
  - 每个 `delta_T` 组施加 Wilcoxon 符号秩检验（配对样本），评估两模型成本差异的统计显著性
- 输出：`experiment.csv`（每条记录包含模型类型、`delta_T` 组标识、场景序号、成本、求解状态、用时）、`det_vs_unc_boxplot.png`/`.tiff`（配对小提琴箱线图）、`meta.json`（实验元参数）、`experiment.log`。

### 6.3 实验结果指标

主要比较指标包括：

- `total_cost`：总物流成本；
- `R_net_risk`：网络风险可靠性；
- `R_net_time`：网络时间可靠性；
- `gap_pct`：分支定价求解的上界/下界差；
- `iterations`：柱生成迭代次数；
- `total_columns`：生成列数；
- `active_hubs`、`num_vehicles_used`、`transport_routes` 等调度细节。

## 7. 结论

从当前代码实现可知：

- 风险可靠性与时间可靠性是两个不同但耦合的约束体。`alpha_max` 越小，风险约束越严格，成本通常上升；而 `alpha_T` 越高，时间窗越紧，时间可靠性压力越大。
- `delta_T` 作为模糊时间离散度参数，决定了运输时间不确定性传播到可行性和成本的强度；较大的 `delta_T` 会降低时间可靠性并驱动更多保守路径。
- `Branch-and-Price` 框架能够在可行性和成本之间寻找精确折衷，`LP-only` 作为基准可以揭示可靠性约束下的松弛 gap。
- 时变风险建模通过 `phi(t)` / `psi(t)` 引入昼夜与高峰效应，使危险品调度结果更贴近实际运营风险分布。
- 以上实验体系为论文中的主实验和敏感性分析提供了完整的代码支撑，且结果可通过 `results/` 目录下的 CSV 与图表复现。

---

## 8. 代码映射速览

| 文档概念     | 代码位置                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------ |
| 风险不确定性 | `src/core/uncertainty.py` 中 `BoolUncertainVariable`、`time_varying_alpha()`               |
| 时间不确定性 | `src/core/uncertainty.py` 中 `TrapezoidalFuzzy`、`UncertaintyEngine.make_transport_time()` |
| 风险可靠性   | `src/core/reliability.py` 中 `check_arc_risk()`, `compute_path_risk()`                     |
| 时间可靠性   | `src/core/reliability.py` 中 `compute_time_reliability_od()`                               |
| 主求解器     | `src/solvers/branch_and_price.py`                                                          |
| 参数定义     | `config.py`                                                                                |
| 主实验入口   | `main.py`                                                                                  |
| 敏感性分析   | `sensitivity/*.py`                                                                         |
