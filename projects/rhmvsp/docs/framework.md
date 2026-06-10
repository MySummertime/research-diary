# RHMVSP 框架定义

## 1. 科学问题与创新点

本项目旨在解决一个“面向危险品多式联运的可靠车辆调度优化问题”。核心科学问题是：

- 在具有风险不确定性和时间不确定性的 multi-modal 运输网络中，如何在最小化 cost 的同时满足 risk, time 的可靠性约束？
- 如何将考虑了 (1) 公铁联运，含路径、转运方案选择 (2) 软时间窗，遵守送达时间、铁路时刻表 (3) 公路空车重用 这一车辆调度问题（VSP）纳入统一模型并求解？

本项目的主要创新点包括：

1. 使用 **不确定理论** 构造 risk, time 的不确定性，风险采用 bool 不确定变量，时间采用 zig-zag/梯形变量；
2. 将 **Time-varying Probability** 和 **Time-varying Consequences** 引入 arc, node 的风险评估，在 24h 内体现不同时间段的不同风险程度；
3. 统一定义 risk, time 的可靠性，并以改进的 **Branch-and-Price** 精确求解可行路径。

## 2. 不确定性定义

### 2.1 风险不确定性：布尔不确定变量

项目中每条运输弧和转运节点的事故状态由布尔不确定变量 $\xi$ 建模，其不确定测度定义为：

$$
\mathcal{M}\{\xi = 1\} = \alpha, \quad \mathcal{M}\{\xi = 0\} = 1 - \alpha
$$

其中 $\alpha$ 为事故发生的**信任度**（belief degree），即决策者对事故发生可能性的主观信任程度；$\xi = 1$ 表示事故发生，$\xi = 0$ 表示事故未发生。

布尔不确定变量的期望后果为：

$$
\mathbb{E}^{\mathcal{M}}[\xi \cdot C] = \alpha \cdot C
$$

其中 $C$ 为事故后果严重程度。

此外，项目引入了**时间依赖的风险乘数**：

- $\phi(t)$ 表示 probability 随时段的折算倍率；
- $\psi(t)$ 表示 consequences 随人口暴露时段的放大倍率。

实际风险计算采用时间区间上的精确分段积分：

- 对运输窗口 $[t_{start}, t_{end}]$ 进行分段积分，计算平均 $\phi(t)$；
- 有效 belief degree 为：$\alpha_{eff} = \min(0.95, \alpha_{base} \times \text{平均} \phi(t))$；
- Consequences 则按 $\psi(t)$ 加权。

### 2.2 时间不确定性：模糊行程时间

项目使用 Trapezoidal 模糊变量 (a,b,c,d) 表现运输和转运时间：

- 对于公路弧，使 b=c，退化为 zig-zag 变量；
- 对于铁路弧，则保持 (a,b,c,d) 的梯形变量。

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

**模糊变量的期望值**（Liu 2007, Theorem 1.23）：

- 三角形模糊变量 $\tilde{t} = (a, b, c)$：

$$
\mathbb{E}^{\mathcal{M}}[\tilde{t}] = \frac{a + 2b + c}{4}
$$

- 梯形模糊变量 $\tilde{\tau} = (a, b, c, d)$：

$$
\mathbb{E}^{\mathcal{M}}[\tilde{\tau}] = \frac{a + b + c + d}{4}
$$

不确定时间生成逻辑：

- 道路基准时间 $base = distance / road_{speed}$，生成 $a=base \times a_{ratio}$，$b=c=base$，$d=base \times c_{ratio}$；
- 铁路基准时间 $base = distance / rail_{speed}$，同样的逻辑生成 $(a,b,c,d)$；
- 转运时间以基准 1 小时为中心生成梯形模糊变量 $(a,b,c,d)$，其中 $\delta_T$ 为**时间波动因子**（time fluctuation factor），控制模糊变量的离散程度：
  - $\delta_T = 0$ 表示确定性时间（无不确定性）
  - $\delta_T$ 越大则时间波动范围越宽。

### 2.3 不确定测度与概率的区别

项目严格遵循 Liu (2007) 的**不确定理论**框架，而非概率论。核心区别如下：

- 不确定测度 $\mathcal{M}$ 具有**自对偶性**（self-duality）：$\mathcal{M}\{A\} + \mathcal{M}\{A^c\} = 1$；
- 对于 $\alpha > 0.5$，不确定测度等价于**可信性测度**（credibility measure）$\text{Cr}\{\cdot\}$；
- 布尔变量的不确定测度 $\mathcal{M}\{\xi = 1\} = \alpha$ 表示**信任度**（belief degree），而非频率概率。

项目的**核心约束机制**是不确定测度下的**机会约束**（chance constraint）$\mathcal{M}\{f(x, \xi) \leqslant 0\} \geqslant \alpha$，而非概率论中的 $P\{f(x, \xi) \leqslant 0\} \geqslant \alpha$。期望值在成本计算中使用，但不作为主要约束。

## 3. 可靠性定义

本项目的可靠性约束均采用**不确定测度机会约束**形式 $\mathcal{M}\{f(x, \xi) \leqslant 0\} \geqslant \alpha$。

### 3.1 风险可靠性层次

风险可靠性采用**串联系统最弱链**原则。路径级风险机会约束为：

$$
R(P) = \min\{R_{arc}, R_{node}\} \geqslant \beta_{od}
$$

即 $\mathcal{M}\{\text{路径上所有弧和节点均无事故}\} \geqslant 1 - \alpha_{max}$。

组件可靠性：

- 弧可靠性 $R_{ij} = 1 - \alpha_{ij}$；
- 节点可靠性 $R_i = 1 - \gamma_i$；

层次化聚合：

- 路径可靠性 $R(P) = \min\{R_{arc}, R_{node}\}$；
- OD 对可靠性 $R_{od} = \min\{R(P)\}$；
- 网络风险可靠性 $R_{net}^{risk} = \min\{R_{od}\}$。

风险阈值由 $\alpha_{max}$ 控制，定义**风险可靠性下界**（risk reliability threshold）为：

- OD 级路径可靠性下界：$\beta_{od} = 1 - \alpha_{max}$；
- 网络级风险可靠性下界：$\beta_{net} = \beta_{od}$。

这些阈值用于约束：

- 路径可靠性 $R(P) \geq \beta_{od}$
- 网络风险可靠性 $R_{net}^{risk} \geq \beta_{net}$。

### 3.2 时间可靠性层次

时间可靠性采用**不确定测度机会约束**形式：

$$
R_T = \mathcal{M}\{s + \tilde{Z}_{total} \leqslant L_{od}\} \geqslant \alpha_T
$$

其中路径出发时间为 $s$，路径上所有运输弧和转运节点的模糊时间变量之和为路径总模糊时间：

$$
\tilde{Z}_{total} = \sum_{ij \in P} \tilde{t}_{ij} + \sum_{k \in T(P)} \tilde{\tau}_k
$$

其中 $\tilde{t}_{ij}$ 为弧 $(i,j)$ 的模糊运输时间，$\tilde{\tau}_k$ 为转运节点 $k$ 的模糊转运时间，$T(P)$ 为路径 $P$ 上的转运节点集合。

该测度综合考虑了离散式转运时间、道路/铁路运输时间以及铁路班列时刻窗口约束。

对于 $\alpha_T > 0.5$ 的情况，时间可靠性约束 $R_T \geq \alpha_T$ 可用**确定性等价**表示。设路径总模糊时间的右尾参数为 $(c, d)$（即梯形模糊变量的右半部分），则确定性等价约束为：

$$
L_{od} \geqslant (2\alpha_T - 1) d + 2(1-\alpha_T) c
$$

时间可靠性的判定输出包括：

- 确定性等价到达时间是否不超过 $L_{od}$（时间窗可行性）；
- $R_T$ 的连续数值（时间可靠性度量）；
- 模糊到达时间的悲观值、乐观值。

### 3.3 综合可靠性

完整解的网络可靠性通过以下层次化聚合得到。所有聚合均基于不确定测度下的机会约束：

**OD 级可靠性**：对每个 OD 对 $o$，设其有 $K_o$ 条批次路径，每条批次路径 $k$ 的风险可靠性和时间可靠性分别为 $R_{o,k}^{risk}$ 和 $R_{o,k}^{time}$。OD 级综合可靠性机会约束为：

$$
R_o = \min_{k=1}^{K_o} \min\{R_{o,k}^{risk}, R_{o,k}^{time}\} \geqslant \beta_{od}
$$

**网络级可靠性**：取所有 OD 对的最小值：

$$
R_{net} = \min_{o} R_o
$$

网络级风险可靠性、时间可靠性分别定义为：

$$
R_{net}^{risk} = \min_{o} \min_{k} R_{o,k}^{risk}, \quad R_{net}^{time} = \min_{o} \min_{k} R_{o,k}^{time}
$$

## 4. 模型构成

### 4.1 网络结构与模式

项目采用多式联运网络 $G=(N, A)$ 建模，其中 $N$ 为节点集合，$A$ 为弧集合。网络包含以下要素：

- **运输弧**：分为公路弧（$m=1$）和铁路弧（$m=2$），每条弧具有距离、基准速度、风险信任度等属性；
- **转运节点**：用于公路与铁路之间的换装，具有转运时间、转运风险信任度、转运成本和设施容量等属性；
- **OD 对**：由原点 $o$、目标 $d$、需求量 $Q_{od}$、软时间窗 $[E_{od}, L_{od}]$ 等信息构成。

### 4.2 成本构成

目标函数最小化总物流成本，其构成如下：

$$
C_{total} = C_{deploy} + C_{transport} + C_{transfer} + C_{reposition} + C_{holding} + C_{opp}
$$

各项成本定义为：

- **车辆部署成本**：$C_{deploy} = c_{road}^{deploy} \cdot n_{road} + c_{rail}^{deploy} \cdot n_{rail}$，其中 $n_{road}$、$n_{rail}$分别为公路、铁路车辆的使用数量；
- **运输成本**：公路运输按单位距离成本计费；铁路采用两部制票价，包含固定基础运价、可变里程费和危险品附加费；
- **转运成本**：$C_{transfer} = c_{transfer} \cdot \sum_{k} y_k + c_{hub} \cdot \sum_{k} z_k$，其中 $y_k$ 为节点 $k$ 的转运量，$z_k$ 为设施启用0-1变量；
- **公路空车调度成本**：$C_{reposition} = c_{empty} \cdot \sum_{ij} d_{ij} \cdot x_{ij}^{empty}$，其中 $d_{ij}$ 为空驶距离，$x_{ij}^{empty}$ 为空驶流量；
- **库存持有成本**：$C_{holding} = c_{holding} \cdot \sum_{od} h_{od}$，其中 $h_{od}$ 为 OD 对运输的货物在起点 $o$ 的滞留时间；
- **机会成本**：$C_{opp} = c_{opp} \cdot \sum_{od} w_{od}$，其中 $w_{od}$ 为规定时间窗内无法满足的需求。

### 4.3 不确定性参数

模型的关键不确定性参数包括：

- **风险阈值** $\alpha_{max}$：弧级事故信任度上界，控制风险可靠性约束的严格程度；
- **时间可靠性阈值** $\alpha_T$：时间可靠性的最低要求，$\alpha_T > 0.5$ 时可进行确定性等价；
- **时间波动因子** $\delta_T$：控制模糊行程时间离散程度的参数，$\delta_T = 0$ 对应确定性时间，$\delta_T$ 越大则时间波动范围越宽。

### 4.4 时变风险与时间窗设计

**时变风险模型**：通过时间依赖的风险乘数 $\phi(t)$ 和 $\psi(t)$ 实现。当启用时变风险时，对运输窗口 $[t_{start}, t_{end}]$ 内的风险进行分段积分平均：

$$
\alpha_{eff} = \min\left(0.95, \frac{1}{t_{end} - t_{start}} \int_{t_{start}}^{t_{end}} \alpha_{base} \cdot \phi(t) \, dt\right)
$$

后果的时变加权为：

$$
C_{eff} = \frac{1}{t_{end} - t_{start}} \int_{t_{start}}^{t_{end}} C_{base} \cdot \psi(t) \, dt
$$

其中 $\phi(t)$ 为交通事故概率的时段折算倍率（反映交通流量与驾驶员疲劳的昼夜变化），$\psi(t)$ 为人口暴露时段的放大倍率（反映人口活动的昼夜变化）。

**时间窗设计**：OD 对的截止时间 $L_{od}$ 由基础期望到达时间 $E_{od}$ 加上时间缓冲量确定：

$$
L_{od} = E_{od} + \Delta L
$$

其中 $\Delta L$ 为时间缓冲参数。在灵敏度分析中，$\alpha_T$ 的变化会驱动 $L_{od}$ 的重新计算。

### 4.5 解决方案结果结构

求解器输出的关键信息：

- **Cost**：总成本 $C_{total}$ 及各项子成本（部署、运输、转运、空车调度、库存、机遇成本）；
- **Network-level 可靠性**：风险可靠性 $R_{net}^{risk}$ 和时间可靠性 $R_{net}^{time}$；
- **OD-level 可靠性**：每个 OD 对的风险可靠性和时间可靠性。

## 5. 算法

### 5.1 问题的组合结构

本模型的整数规划可抽象为以下形式：

$$
\min \sum_{p \in \Omega} c_p \lambda_p + C_{reposition} + C_{hub} + C_{holding} + C_{opp}
$$

$$
\text{s.t.} \quad \sum_{p \in \Omega_{od}} \lambda_p \geq 1, \quad \forall od
$$

$$
\sum_{p} \lambda_p \leq K
$$

$$
\lambda_p \in \{0, 1\}, \quad \forall p
$$

其中 $\Omega$ 为所有可行路径的集合（指数级规模），$\lambda_p$ 为路径 $p$ 的选择指示变量，$K$ 为可用车辆总数。路径 $p$ 在进入集合 $\Omega$ 前已通过定价子问题筛选，满足路径风险可靠性约束 $R(P) \geq \beta_{od}$ 和时间窗可行性。

由于 $\Omega$ 的规模随网络和时刻表呈指数增长，无法显式枚举，因此采用列生成框架求解其线性规划松弛。

### 5.2 定价子问题：带资源约束的最短超路径

定价子问题的核心是为每个 OD 对生成具有**负 reduced cost** 的新路径（列）。这等价于求解一个**带资源约束的最短超路径问题**（ESPPRC）。

#### 5.2.1 标签结构

每条部分路径由一个**标签**（Label）表示，标签携带以下资源：

| 资源                                  | 含义                          | 约束类型                       |
| ------------------------------------- | ----------------------------- | ------------------------------ |
| reduced cost                          | 路径 reduced cost（含对偶值） | 负值即为列生成候选             |
| risk_R                                | 路径累积风险可靠性            | 硬约束：$R(P) \geq \beta_{od}$ |
| time_upper / time_lower               | 模糊行程时间的上下界          | 硬约束：确定性等价 ≤ $L_{od}$  |
| visited / ng_memory                   | 节点访问记录（NG 路由）       | 部分元性约束                   |
| prev_mode / rail_started / rail_ended | 运输模式状态                  | 硬约束：公路→铁路→公路单次换装 |

#### 5.2.2 双向标签扩展

采用 Desrochers 双向搜索策略：从源点向前扩展、从汇点向后扩展，然后在**中间弧**上拼接。该策略将搜索空间从 $O(|N|^2)$ 降低到 $O(|N|)$。

**前向扩展**：从 OD 原点出发，沿弧逐节点扩展标签。每次扩展时：

1. 计算新弧的 reduced cost：$\bar{c}_{ij} = c_{ij} - \pi_{ij}$，其中 $\pi_{ij}$ 为对应弧约束的对偶值；
2. 更新时间上界：若为公路弧，$t_{upper}^{new} = t_{upper}^{old} + d_{ij}/v_{road} \cdot c_{ratio}$；若为铁路弧，从时刻表中选取发车时间 ≥ 当前到达时间的有效班列；
3. 更新风险可靠性：$R^{new} = \min\{R^{old}, R_{ij}^{arc}, R_k^{node}\}$，其中弧可靠性考虑时变概率乘数 $\phi(t)$；
4. 若 $R^{new} < \beta_{od}$，**剪枝**（该路径不满足风险约束）；
5. 若为转运节点，更新模式状态并累加转运成本和转运时间。

**后向扩展**：从 OD 目标出发反向扩展，类似地维护资源并剪枝。

#### 5.2.3 标签支配规则

两个标签 $l_1$ 支配 $l_2$ 需同时满足：

- $\bar{c}_1 \leqslant \bar{c}_2$（更优的 reduced cost）
- $t_1^{upper} \leqslant t_2^{upper}$（更早的到达时间）
- $R_1 \geqslant R_2$（更高的风险可靠性）
- $\text{ng\_memory}_1 \subseteq \text{ng\_memory}_2$（NG 路由兼容）
- 运输模式状态相同（prev_mode、rail_started、rail_ended 一致）

#### 5.2.4 中间弧拼接

前向标签集和后向标签集在中间弧 $(i,j)$ 上拼接。对每对候选前向-后向标签，计算：

$$
\bar{c}_{full} = \bar{c}_{forward} + \bar{c}_{arc(i,j)} + \bar{c}_{backward}
$$

$$
R_{full} = \min\{R_{forward}, R_{arc(i,j)}, R_{backward}, R_i^{node}, R_j^{node}\}
$$

若 $\bar{c}_{full} < 0$ 且 $R_{full} \geq \beta_{od}$，则生成一个新列。

#### 5.2.5 与标准 ESPPRC 的区别

| 维度 | 标准 VRP ESPPRC | 本模型 ESPPRC                   |
| ---- | --------------- | ------------------------------- |
| 时间 | 确定性旅行时间  | 模糊行程时间 + 离散铁路时刻表   |
| 风险 | 无              | 时变弧风险可靠性 + 转运节点风险 |
| 模式 | 单一运输模式    | 公铁联运 + 模式切换状态机       |
| 元性 | 完全/NG 元性    | NG 元性 + 模式状态联合支配      |

### 5.3 主问题：受限线性规划

主问题在每轮列生成中求解一个**受限主问题**（RMP），其线性规划松弛形式为：

$$
\min \sum_{p \in \bar{\Omega}} c_p \lambda_p + \sum_{ij} c_{empty} \cdot d_{ij} \cdot (\pi_{ij}^{own} + \pi_{ij}^{rent}) + \sum_{k} c_{hub} \cdot z_k + \sum_{od} c_{opp} \cdot s_{od}
$$

约束条件包括：

- **需求覆盖**：$q_{od} + s_{od} \geq D_{od}$，其中 $s_{od}$ 为未满足需求量；
- **容量链接**：$\sum_{p \in \bar{\Omega}_{od}} q_{veh} \cdot \lambda_p + d_{od} \geq q_{od}$，链接路径选择与交付量；
- **车辆平衡**：$\sum \lambda_p^{own} - \sum \pi^{own} \leq K$；
- **空车重定位**：$\sum \pi \leq \frac{K-1}{K} \sum \lambda$，限制空车调度比例；
- **枢纽激活**：$\sum_{p \ni k} \lambda_p \leq M \cdot z_k$，将路径选择与设施启用耦合；
- **最大枢纽数**：$\sum z_k \leq K_{max}$。

主问题的解提供对偶值，反馈给定价子问题用于计算 reduced cost。

### 5.4 列生成迭代

列生成的迭代流程如下：

1. 求解受限主问题的线性规划松弛，获取对偶值 $\pi$；
2. 对每个 OD 对，以 $\pi$ 为输入求解定价子问题（先启发式、再精确）；
3. 若存在 reduced cost 为负的新列，加入主问题；
4. 重复直至无新列可加入（定价子问题无负 reduced cost 解）。

对偶值稳定化采用移动平均法：$\pi^{stab} = (1-\eta) \pi^{raw} + \eta \cdot \pi^{prev}$，其中 $\eta \in [0,1]$ 为平滑系数。

### 5.5 分支策略

当线性规划松弛的解不满足整数性时，需要分支。本模型采用**两阶段分支**策略：

#### 第一阶段：枢纽启用分支

若转运枢纽激活变量 $z_k \in (0,1)$ 为分数值，优先对其进行分支：

- **关闭分支**：$z_k = 0$，删除所有经过该枢纽的路径列；
- **打开分支**：$z_k = 1$，保留当前列集合。

该分支具有优先级，因为枢纽决策是战略层面的设施选址问题，其松弛对成本结构影响最大。

#### 第二阶段：OD 特定弧分支

若无分数枢纽变量，则对路径选择进行分支。选择**最分数化的 OD 对**，在其路径集中选择一条关键弧 $(u,v)$，创建两个子节点：

- **禁止分支**：删除包含弧 $(u,v)$ 的所有路径列；
- **强制分支**：仅保留包含弧 $(u,v)$ 的路径列。

两阶段分支确保了分支约束与列生成的兼容性：列在加入子节点时被预过滤，定价子问题在扩展时也被分支约束剪枝。

### 5.6 搜索策略

采用 **Best-First Search**：维护一个优先队列，每次选择下界最小的节点展开。节点的下界由该节点线性规划松弛的目标值给出，上界由当前最优整数可行解的目标值给出。若某节点的下界 ≥ 当前上界，则**剪枝**。

### 5.7 LP Relaxation 作为 Benchmark

LP Relaxation 在本框架中具有**三重作用**：

1. **可证明下界**：LP 松弛的最优值是整数规划最优值的严格下界，为评估 Branch-and-Price 的求解质量提供基准；
2. **收敛性参照**：通过对比 LP 下界和 IP 上界，计算最优间隙 $\text{gap\%} = \frac{C^{IP} - C^{LP}}{C^{LP}} \times 100\%$；
3. **独立分析工具**：LP Relaxation 揭示了放松整数性约束后可靠性约束的边际成本。在 Out-of-Sample 实验中，LP Relaxation 解与整数解的对比量化了整数性代价（integrality gap）。

## 6. 实验与结果

### 6.1 主实验

铁路时间窗：

![alt text](image.png)

计算结果：

![small instance](../results/exp_20260604_092144_small/solution_plot.png)

### 6.2 灵敏度分析

分析不同关键参数对模型性能的影响。实验按复杂度从低到高排列：单参数灵敏度 → 算法性能 → 双参数耦合 → 外样本验证 → 模型对比。

#### 6.2.1 风险阈值 $\alpha_{max}$ 灵敏度

- **变量**：风险可靠性阈值 $\alpha_{max}$
- **目标**：分析 $\alpha_{max}$ 对网络风险可靠性 $R_{net}^{risk}$、时间可靠性 $R_{net}^{time}$ 和总成本 $C_{total}$ 的影响。
- **结论**：$\alpha_{max}$ 越大，风险可靠性越差，时间可靠性越好，总成本越低。
- **输出**：双 Y 轴折线图。
  ![alt text](../results/sensitivity_alpha_max/sensitivity_alpha_max.png)

#### 6.2.2 时间可靠性阈值 $\alpha_T$ 灵敏度

- **变量**：时间可靠性阈值 $\alpha_T$
- **目标**：分析时间可靠性阈值 $\alpha_T$ 对可行性和成本的影响。
- **结论**：$\alpha_T$ 越大，时间可靠性越好，风险可靠性不受影响，总成本越高。
- **输出**：双 Y 轴折线图。
  ![alt text](../results/sensitivity_alpha_T/sensitivity_alpha_T.png)

#### 6.2.3 模糊时间系数 $\delta_T$ 灵敏度

- **变量**：时间不确定性的乘数 $\delta_T$
- **目标**：量化建模模糊旅行时间对可靠性和成本的影响。
- **结论**：$\delta_T$ 越大，时间可靠性越差，风险可靠性几乎不受影响，总成本越高。
- **输出**：双 Y 轴组合图。
  ![alt text](../results/value_fuzzy_time_20260604_092436/value_fuzzy_time_dual.png)

#### 6.2.4 算法性能灵敏度

- **变量**：$\alpha_{max}$、$\alpha_T$、$\delta_T$ 三个参数分别扫描。
- **目标**：分析不同参数配置下 Branch-and-Price 算法的求解时间和列生成迭代次数，评估算法的计算复杂度。
- **结论**：算法求解时间和迭代次数随参数变化相对稳定，表明 Branch-and-Price 框架对不确定性参数具有良好的计算鲁棒性。
- **输出**：双 Y 轴折线图。

$\alpha_{max}$ 对算法性能的影响：
![alt text](../results/sensitivity_perf_20260603_153935/perf_alpha_max.png)

$\alpha_T$ 对算法性能的影响：
![alt text](../results/sensitivity_perf_20260603_153935/perf_alpha_T.png)

$\delta_T$ 对算法性能的影响：
![alt text](../results/sensitivity_perf_20260603_153935/perf_delta_T.png)

#### 6.2.5 风险-时间耦合灵敏度

- **变量网格**：$\alpha_{max}$ 与 $\delta_T$ 的 $10 \times 10$ 组合。
- **目标**：探索风险约束与时间不确定性之间的耦合权衡面。
- **结论**：
  - **沿 $\alpha_{max}$ 方向**（固定 $\delta_T$）：$\alpha_{max}$ 从严格（0.01）放宽至中等（0.03–0.05）时，成本下降（327k→272k），风险可靠性下降（0.99→0.96），时间可靠性先降后升（在 $\alpha_{max} \approx 0.02$ 处出现低谷，随后回升至 0.78）；$\alpha_{max} \geq 0.05$ 后各指标收敛，表明风险约束已完全松弛。
  - **沿 $\delta_T$ 方向**（固定 $\alpha_{max}$）：$\delta_T$ 从 0.01 增至 0.20 时，成本小幅上升，时间可靠性基本稳定或微升；$\delta_T > 0.40$ 后成本显著上升，时间可靠性开始下降。风险可靠性在整个 $\delta_T$ 范围内变化极小（<0.003），与 $\alpha_{max}$ 灵敏度实验一致地证实了**风险可靠性和时间不确定性近似解耦**。
  - **耦合效应**：在严格风险约束下（$\alpha_{max} = 0.01$），$\delta_T$ 对成本几乎无影响（稳定在 319–327k），模型被风险约束锁定在特定路径集上，时间波动无法改变调度方案；在宽松风险约束下（$\alpha_{max} \geq 0.05$），$\delta_T$ 的增大导致成本从 246k 上升至 304k（+24%），时间可靠性从 0.83 降至 0.70，表明风险约束松弛后时间不确定性的影响才得以充分释放。
- **输出**：热力图（成本、$R_{net}^{risk}$、$R_{net}^{time}$）。

成本热力图：
![](../results/coupled_20260603_154010/heatmap_cost.png)

风险可靠性热力图：
![](../results/coupled_20260603_154010/heatmap_R_risk.png)

时间可靠性热力图：
![](../results/coupled_20260603_154010/heatmap_R_time.png)

#### 6.2.6 Out-of-sample 实验

- **目标**：对多组典型 $(\alpha_T, \alpha_{max})$ 组合进行 Monte Carlo 场景抽样，比较 Branch-and-Price 算法和 LP Relaxation 各自的稳定性。
- **结论**：Branch-and-Price 在所有参数组合下均优于 LP Relaxation，平均成本比为 0.983（即 BAP 比 LP 低约 1.7%）。该优势在严格风险约束下（$\alpha_{max}=0.02$）更为显著（BAP/LP = 0.947–0.986），在宽松约束下（$\alpha_{max}=0.05$）两者趋于一致（BAP/LP ≈ 0.995–0.997）。两种方法在不同随机场景下的标准差相近，表明其对需求波动的鲁棒性相当。
- **输出**：小提琴组合箱线图。

![alt text](../results/out_of_sample_n=100/out_of_sample_boxplot.png)

#### 6.2.7 确定性模型 vs 不确定模型

- **目标**：量化同时考虑"时间不确定性"和"风险不确定性"的必要性。
- **结论**：
  - 当 $\delta_T = 0$ 时，两模型成本几乎一致（差异 < 0.01%，平均差 22 元），验证了实验设计的公平性。
  - 随着 $\delta_T$ 增大，不确定模型的成本单调上升：$\delta_T = 0.2$ 时高出约 23.7k（+7.8%），$\delta_T = 0.4$ 时高出约 49.3k（+16.2%），$\delta_T = 0.6$ 时高出约 84.6k（+27.8%）。
  - 这部分额外成本正是不确定性建模的"保险代价"：确定性模型低估了真实环境中的时间波动，其解在面对不确定性时可能产生更严重的后果（如延误违约），而不确定性感知模型通过选择更保守但更稳健的路径来规避此类风险。
- **输出**：小提琴组合箱线图。TBD...

## 7. 结论

本章通过系统的灵敏度分析实验，从参数影响、算法鲁棒性、模型对比三个层面验证了所提框架的有效性。主要结论如下：

**参数影响与权衡机制**。

- 风险阈值 $\alpha_{max}$ 和时间阈值 $\alpha_T$ 是控制成本-可靠性权衡的两个核心旋钮：$\alpha_{max}$ 越小（风险约束越严格），成本上升但风险可靠性改善，时间可靠性呈非线性响应（先降后升）；$\alpha_T$ 越大（时间窗越紧），时间可靠性提升但成本增加，且不影响风险可靠性。
- 时间波动因子 $\delta_T$ 的影响呈现强烈的条件依赖性。在严格风险约束下 $\delta_T$ 几乎不改变成本结构（模型被风险约束锁定在特定路径集），仅在风险约束放松后才显著驱动成本上升和时间可靠性下降，证实了风险可靠性与时间不确定性在本模型中近似解耦。

**不确定性建模的价值**。

- 确定性与不确定性感知模型的对比实验表明，不确定性建模的"保险代价"随环境不确定性 $\delta_T$ 单调递增（$\delta_T=0.2$ 时 +7.8%，$\delta_T=0.6$ 时 +27.8%），但这一额外成本是为更稳健的调度方案所支付的合理溢价。
- 可靠性价值实验进一步揭示，引入可靠性约束的边际成本极低，以不足 1% 的总成本增加即可获得 33% 的时间可靠性提升，且风险可靠性不受影响，说明可靠性约束具有显著的"正向筛选效应"。

**算法性能与稳健性**。

- Branch-and-Price 框架在所有参数配置下均表现出稳定的计算性能，求解时间和迭代次数对不确定性参数具有良好的鲁棒性。Out-of-sample 实验验证了该算法始终优于 LP Relaxation 基准（平均成本低 1.7%），尤其在严格风险约束下优势更为显著，且两种方法对需求波动的鲁棒性相当。
- 耦合实验的热力图直观展示了成本-风险-时间三维权衡面的非线性结构，为决策者在不同可靠性需求下选择最优参数组合提供了量化依据。
