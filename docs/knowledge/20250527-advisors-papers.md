---
layout: default
title: Study of 6 papers
date: 2025-5-27
---

<small>{{ page.date | date: "%-d %B %Y" }}</small>

# 6 papers published by my advisor

## Hazmat Routing Problem 1

Source:

- 静态确定性网络中的多目标最短路径问题  
  Multi-objective Shortest Path Problem in Static Deterministic Networks.  
  [[1]代存杰,李引珍,何瑞春,等.危险品运输路径多准则优化模型及求解算法[J].交通运输系统工程与信息,2016,16(01):189-195.DOI:10.16097/j.cnki.1009-6744.2016.01.029.](https://kns.cnki.net/kcms2/article/abstract?v=mV2q5OJ_OLwNA_0nZil7FiJfgZRsuYovrj0jEA4YW5enY4TLU6pSvWmqz_wwZ0meHQ0202xITbtKtANEF-yjieoAYLzHj6qGUp89aY11loCdnhGID-o-zbsaKboKkU_WFq4IDINtm2IlNMB6Lkk1GV3fVysK9l_5feh_-JgeXKA=&uniplatform=NZKPT)

Model:

$$
\text{3 objectives}
\begin{cases}
  \min \quad R_k, \\
  \min \quad C_k, \\
  \min \quad T_k \\
\end{cases}
$$

$$
\text{3 special constraints}
\begin{cases}
  \text{risk threshold of}
  \begin{cases}
    \text{links}, \\
    \text{paths} \\
  \end{cases}, \\
  x \in \{0,1\}
\end{cases}
$$

- $\min R$ - Minimize risk:
  - Linear function
- $\min C$ - Minimize cost:
  - Linear function
- $\min T$ - Minimize time:
  - Linear function

Algorithm:

1. 双向拓扑搜索算法（精确算法 Exact Algorithm）  
   Bidirectional Topological Search Algorithm
   - 通过前向+后向拓扑搜索，删除不满足风险约束的路段（和/或）节点，以生成 k 剩余网络.
2. 改进的标号算法（Exact Algorithm）  
   Improved Labeling Algorithm, using Fibonacci-Heap to build Priority Queue, completes the solution with time complexity O(m + n logn)
   - 根据不同的权重准则生成最优路径.
   - 本质上是最短路算法，e.g., Dijkstra，的高效实现.
3. 关键路段调整策略（采用启发式算法的思想 Heuristic Algorithm）  
   Critical Section Adjustment Strategy
   - 解决多条路径因共享“关键路段”而可能违反风险约束的问题.
     > 启发式算法的思想：基于一套特定的规则对解进行迭代更新，目的是快速地找到高质量的解，而不保证得到全局最优解.

Category:

- 从数学规划的角度看:  
  From the perspective of Mathematical Programming,

  1. 顶层类别：确定性优化  
     Top-level: Deterministic Optimization
  2. 子类：离散优化  
     Second-level: Discrete Optimization
  3. 子类：多目标整数线性规划  
     Third-level: Multi-objective Integer Linear Programming

- 从网络理论的角度看:  
  From the perspective of Network Theory,
  - 多目标最短路径问题  
    Multi-objective Shortest Path Problem

Pseudocode (C++):

```c++
// 算法1: 构建 k 剩余网络
void buildResidualNetwork(Graph& g, double R_max, double r_max) {
    // 正向搜索剪枝
    map<Node*, double> forward_risk;
    for (auto& node : g.nodes) forward_risk[node] = infinity;
    forward_risk[g.source] = 0.0;

    for (auto& edge : g.edges) {
        if (forward_risk[edge.from] + edge.risk > R_max || edge.risk > r_max) {
            // 删除路段
            g.removeEdge(edge);
        }
    }

    // 反向搜索剪枝
    map<Node*, double> backward_risk;
    for (auto& node : g.nodes) backward_risk[node] = infinity;
    backward_risk[g.sink] = 0.0;

    for (auto& edge : g.edges) {
        if (backward_risk[edge.to] + edge.risk > R_max) {
            // 删除路段
            g.removeEdge(edge);
        }
    }
}

// 算法2: 关键路段调整
vector<Path> adjustCriticalSections(vector<Path>& initial_paths, Graph& g, double r_max) {
    vector<Path> final_paths = initial_paths;
    while (true) {
        map<Edge*, int> usage_count = countEdgeUsage(final_paths);
        Edge* critical_edge = findViolatingCriticalEdge(usage_count, r_max, g);

        // 没有违规的关键路段，调整结束
        if (critical_edge == nullptr) {
            break;
        }

        Path* path_to_adjust = nullptr;
        double min_penalty = infinity;

        // 找到调整代价最小的路径
        for (auto& path : final_paths) {
            if (path.uses(critical_edge)) {
                double penalty = calculateAdjustmentPenalty(path, critical_edge, g);
                if (penalty < min_penalty) {
                    min_penalty = penalty;
                    path_to_adjust = &path;
                }
            }
        }

        // 调整路径
        if (path_to_adjust != nullptr) {
            g.temporarilyDisable(critical_edge);
            Path new_path = findShortestPath(g, path_to_adjust->source, path_to_adjust->sink);
            // 更新路径
            *path_to_adjust = new_path;
            g.enable(critical_edge);
        }
    }
    // 返回调整后的路径集
    return final_paths;
}
```

## Hazmat Routing Problem 2

Source:

- 静态确定性网络中的多目标相异路径优化问题.  
  Multi-objective dissimilarly routing problem in static deterministic networks.  
  [[2]代存杰,李引珍,马昌喜,等.考虑风险分布特征的危险品运输路径优化[J].中国公路学报,2018,31(04):330-342.DOI:10.19721/j.cnki.1001-7372.2018.04.038.](https://kns.cnki.net/kcms2/article/abstract?v=mV2q5OJ_OLxWmc4LLXdx8Ks6M1NJxBylcZrNr18_OMTutfgX01pvxpKIWslVSKzfYEQalCzgcSeDbwOCsbemIUcQmaMAwqinYxzvCPsgTaIdWSC8TcxubwYszT8ocA5i6jQeaBUCcfzVRrW2fxQp6C3OeOpzEcLBQBt2GNtTWAE=&uniplatform=NZKPT)

Model:

$$
\text{3 objectives}
\begin{cases}
  \min \quad R, \\
  \min \quad C, \\
  \min \quad T \\
\end{cases}
$$

$$
\text{4 special constraints}
\begin{cases}
  \text{risk threshold of paths}, \\
  \text{dissimilarity of paths}
    \begin{cases}
      \text{physically}, \\
      \text{spacially} \\
    \end{cases}, \\
  x,y \in \{0,1\} \\
\end{cases}
$$

- $\min R$ - Minimize risk:
  - Quadratic function
- $\min C$ - Minimize cost:
  - Linear function
- $\min T$ - Minimize time:
  - Linear function

Algorithm:

- 改进的 NSGA-II 算法（元启发式算法 Metaheuristic Algorithm）  
  Improved Non-dominated Sorting Genetic Algorithm II

  - 编码方式：基于节点优先权矩阵，可以加速染色体的编/解码.
  - 选择方式：基于动态拥挤距离，可以保证种群多样性，增加 pareto frontier，并且使解的空间分布更均匀.
  - 交叉与变异方式：基于一系列基因序列交换、反转、随机插入操作，可以保证解的有效性.

  > 元启发式算法的典型特征：基于群体的，受自然（e.g. 进化论）的启发进行迭代搜索，可以高效地找到高质量的解，但不保证解的全局最优性。

Category:

- 从数学规划的角度看：  
  From the perspective of Mathematical Programming,
  1. 顶层类别：确定性优化  
     Top-level: Deterministic Optimization
  2. 子类：离散优化  
     Second-level: Discrete Optimization
  3. 子类：多目标整数非线性规划  
     Third-level: Multi-Objective Integer Nonlinear Programming
- 从网络理论的角度看：  
  From the perspective of Network Theory,
  1. 多目标最短路径问题  
     Multi-Objective Shortest Path Problem
  2. 多目标相异路径问题  
     Multi-Objective Dissimilar Path Problem

Pseudocode(C++):

```c++
// 改进的 NSGA-II 算法
vector<Solution> improvedNSGA_II(int population_size, int max_generations) {
    // 1. 初始化种群
    vector<Solution> population = initializePopulation(population_size);
    evaluate(population);

    for (int g = 0; g < max_generations; ++g) {
        // 2. 创建子代
        vector<Solution> offspring = createOffspring(population); // 交叉与变异 crossover and mutation
        evaluate(offspring);

        // 3. 合并父代与子代
        vector<Solution> combined_pop = population;
        combined_pop.insert(combined_pop.end(), offspring.begin(), offspring.end());

        // 4. 非支配排序
        vector<vector<Solution>> fronts = nonDominatedSort(combined_pop);

        // 5. 选择新一代的个体
        vector<Solution> next_population;
        int front_idx = 0;
        while (next_population.size() + fronts[front_idx].size() <= population_size) {
            next_population.insert(next_population.end(), fronts[front_idx].begin(), fronts[front_idx].end());
            front_idx++;
        }

        // 6. 处理临时种群
        if (next_population.size() < population_size) {
            vector<Solution>& last_front = fronts[front_idx];
            // 计算个体间的动态拥挤距离
            calculateDynamicCrowdingDistance(last_front);
            // 排序
            sort(last_front.begin(), last_front.end(), compareByCrowdingDistance);
            // 填充剩余位置
            int remaining_count = population_size - next_population.size();
            next_population.insert(next_population.end(), last_front.begin(), last_front.begin() + remaining_count);
        }

        population = next_population;
    }
    // 返回 Pareto 最优解集
    return nonDominatedSort(population)[0];
}
```

## Hazmat Routing Problem 3

Source:

- 时变不确定性（随机性）网络中的多目标路径优化问题.  
  Multi-objective routing problem in time-varying uncertain(stochastic) networks.  
  [[3]代存杰,李引珍,马昌喜,等.随机时间依赖路网中危险品运输路径多准则优化[J].哈尔滨工业大学学报,2018,50(03):156-164.](https://kns.cnki.net/kcms2/article/abstract?v=mV2q5OJ_OLy4_q944FjuinQaDrSNid-anu6hHBAGG81oZrHM6aaZByFMOwdG9rcyP_e_cFUXUF1HFn4f2-YJtPlXyZKzXAYle9s7XhYr7a2jsoiIgzTbCSMK91Qf2aue_1m7gFddXs-4mUmRHIV8LP4vZB1deuRUU8KTQ0lnKkw=&uniplatform=NZKPT)

Model:

$$
\text{4 objectives}
\begin{cases}
  \min \quad E[R], \\
  \min \quad \sqrt{V[R]}, \\
  \min \quad E[T], \\
  \min \quad \sqrt{V[T]} \\
\end{cases}
$$

$$
\text{2 special constraints}
\begin{cases}
  \text{arrival time window}, \\
  x \in \{0,1\}
\end{cases}
$$

- $\min E[R]$ - Expected value of risk:
  - Stochastic function with Shrifted Log-Normal Distribution.
- $\min \sqrt{V[R]}$ - Standard variance of risk:
  - Stochastc function.
- $\min E[T]$ - Expected value of time:
  - Stochastic function with Truncated Log-Normal Distribution.
- $\min \sqrt{V[T]}$ - Standard variance of time:
  - Stochastic function.

Algorithm:

- 两阶段多维标号修正算法（Heuristic Algorithm, 对精确算法进行了启发式改造）  
  Two-stage Multi-dimensional Labeling Correcting Algorithm
  - 第一阶段：生成出发时刻
    - 根据路段随机行程时间、到达时间窗、准时到达置信水平，计算出发时间窗.
  - 第二阶段：生成非支配路径集
    - 准则权重法 criterion-weight method
      - 为多个目标（如时间期望、风险方差等）赋予不同的权重，将多维的目标评价转化为单一的综合评价值，用于在阈值支配方法中判定两个解的支配关系.
    - 阈值支配法 threshold-dominate method
      - 标准帕累托支配法要求：一个解在所有目标上都不差，且至少在一个目标上严格更优.
      - 阈值支配法要求：一个解必须在综合评价值上“好出一个指定的阈值$\Delta$”才算支配另一个解，可以避免对非支配路径的完全枚举，提高计算效率.
        > 启发式算法的思想：为了在有限的时间内得到高质量的可行解，制定一套规则，允许算法丢弃一些比较“好”的解.

Category:

- 从数学规划的角度看：  
  From the perspective of Mathematical Programming,
  1. 顶层分类：不确定性优化  
     Top-level: Optimization under Uncertainty
  2. 子类：多目标随机整数非线性规划  
     Second-level: Multi-Objective stochastic Integer Nonlinear Programming
- 从网络理论的角度看：  
  From the perspective of Network Theory,
  - 带时间窗约束的多目标随机时变最短路问题  
    Multi-Objective stochastic time-varying Shortest Path Problem with time windows

Pseudocode(C++):

```c++
// 多维标号
struct Label {
    double expected_time;
    double variance_time;
    double expected_risk;
    double variance_risk;
    // ... 前驱节点信息等
};

// 检查L1是否阈值支配L2
bool isDonimatedBy(const Label& L1, const Label& L2, double delta) {
    // 如果L1在所有维度上都不差于L2，且至少在一个维度上严格优于L2+delta
    return (L1.expected_time <= L2.expected_time && L1.variance_time < L2.variance_time - delta) || ... ;
}

// 算法核心: 多维标号修正
vector<Label> multiDimLabelCorrecting(Node* source, Node* sink, double t_start, double delta) {
    map<Node*, vector<Label>> labels;
    for (auto& node : g.nodes) labels[node] = {};
    // 初始化标号
    labels[source].push_back({0, 0, 0, 0});
    // 初始化队列
    queue<Node*> q;
    q.push(source);

    while (!q.empty()) {
        Node* u = q.front();
        q.pop();

        for (auto& edge : u->outgoing_edges) {
            Node* v = edge.to;
            for (auto& u_label : labels[u]) {
                // 计算到达节点 j 的新标号
                Label v_new_label = calculateNewLabel(u_label, edge, t_start);

                // 如果新标号“阈值支配”现有标号
                if (!isDominatedBy(v_new_label, labels[v], delta)) {
                  // 将新标号加入，并移除被新标号支配的旧标号
                    removeDominatedLabels(labels[v], v_new_label, delta);
                    labels[v].push_back(v_new_label);

                    if (!v->isInQueue) {
                        q.push(v);
                    }
                }
            }
        }
    }
    // 返回终点处的非支配路径集
    return labels[sink];
}
```

## Hazmat Routing Problem 4

Source:

- 时变确定性网络中的单目标路径优化问题.  
  Single-objective routing problem in time-varying deterministic networks.  
  [[4]杨信丰,李引珍,何瑞春,等.多属性时间依赖网络的城市危险品运输路径优化[J].中国安全科学学报,2012,22(09):103-108.DOI:10.16265/j.cnki.issn1003-3033.2012.09.013.](https://kns.cnki.net/kcms2/article/abstract?v=mV2q5OJ_OLwYz0IMLfI9hXpkTizf3O8q4Jb9qZ0IqZUqnB5KugiUVjiCkuNhzg_MQ713CtumjEeQ6prei4c34ayCSgKsnLOJZpLZdi3edLduBLQxjbOP5D6oJR8eD5x4m0ifBe_wWCOast7aP9mVsFhe1xJm4uZq&uniplatform=NZKPT)

Models:

$$
\text{1 objective}
\quad \max z = \sum_{j=1}^{5} w_{i}r_{ij}
$$

标准的标号算法只能处理单一属性值，无法直接处理五个属性.  
而此模型基于信息熵理论，计算出 5 个属性各自的权重，再加权合为一个综合属性值，对多属性问题进行了降维，使得多属性问题也能使用标号算法进行求解.

- 在时刻 $s$，节点 $o$ 处共 $d_{o}$ 条可选路段的属性矩阵  
  At time $s$, the attribute matrix composed of $d_{o}$ optional links at node $o$

  $$
  \mathbf{MA}_{o}^{s} =
  \begin{bmatrix}
    \overrightarrow{t(x)}_{d_{o}\times1}^{s}, \quad
    \overrightarrow{t(x)p}_{d_{o}\times1}^{s}, \quad
    \overrightarrow{l}_{d_{o}\times1}^{s}, \quad
    \overrightarrow{i}_{d_{o}\times1}^{s}, \quad
    \overrightarrow{r}_{d_{o}\times1}^{s}\\
  \end{bmatrix}
  $$

- 规范化矩阵  
  The normalized attribute matrix

  $$
  \mathbf{R}_{ij}^{s} =
  \begin{bmatrix}
    r_{ij}^{s}
  \end{bmatrix}_{d_{o}\times5}
  $$

- 单位化矩阵  
  The unitized attribute matrix

  $$
  \mathbf{H}_{ij}^{s} =
  \begin{bmatrix}
    h_{ij}^{s}
  \end{bmatrix}_{d_{o}\times5}
  $$

- 属性 $j$ 的熵值  
  The entropy value of attribute $j$

  $$
  en_{j}^{s} = -\frac{\sum_{i=1}^{d_o}h_{ij}^{s}\ln{h_{ij}^{s}}}{\ln{d}}
  $$

- 属性 $j$ 在决策中的权重  
  The weight of attribute $j$

  $$
  w_{j}^{s} = \frac{1-en_{j}^{s}}{\sum_{k=1}^{5}(1-en_{k}^{s})}
  $$

- 路段 $i$ 的综合属性值  
  The comprehensive attribute value of link $i$
  $$
  z_{i}^{s} = \sum_{j=1}^{5}w_{j}^{s}r_{ij}^{s}, \quad i = 1,2,...,d_o
  $$

Algorithm:

- 标号算法（Exact Algorithm）  
  Labeling Algorithm
  - 一种求解最短路径的经典算法（e.g. Dijkstra, Bellman-ford）：维护节点标号和前向节点，对解进行迭代更新.
  - 只要网络中没有负环路，就能保证找到最优解.

Category:

- 从数学规划的角度看：  
  From the perspective of Mathematical Programming,
  1. 顶层分类：确定性优化  
     Top-level: Deterministic Optimization
  2. 子类：离散规划  
     Second-level: Discrete Optimization
  3. 子类：单目标整数线性规划  
     Third-level: Single-Objective Integer Linear Programming
- 从网络理论的角度看：  
  From the perspective of Network Theory,
  - 单目标最短路径问题  
    Single-Objective Shortest Path Problem

Pseudocode (C++):

```c++
// 算法1: 信息熵法计算综合属性
map<Edge*, double> synthesizeAttributes(const Matrix& M) {
    // 1. 矩阵的 Min-Max 规范化
    Matrix R = normalize(M);

    // 2. 矩阵的列归一化 (单位化)
    Matrix H = unitizeByColumn(R);

    // 3. 计算各属性权重
    vector<double> weights;
    for (int j = 0; j < M.cols(); ++j) {
        double entropy_j = calculateEntropy(H.getColumn(j));
        weights.push_back(1.0 - entropy_j);
    }
    normalizeWeights(weights);

    // 4. 算出一个综合属性值
    map<Edge*, double> z_scores;
    for (int i = 0; i < M.rows(); ++i) {
        double z_i = 0.0;
        for (int j = 0; j < M.cols(); ++j) {
            z_i += weights[j] * R(i, j);
        }
        z_scores[g.edges[i]] = z_i;
    }
    // 返回综合属性值
    return z_scores;
}

// 算法2: 求解最短路径 (最大化综合属性值)
Path findBestPath_TimeDependent(Graph& g, const map<Edge*, double>& z, Node* start, Node* end, double t_start) {
    map<Node*, double> utility;
    map<Node*, Node*> predecessor;

    // 初始化各节点的综合属性值
    for (auto& node : g.nodes) utility[node] = -infinity;
    utility[start] = 0.0;

    // 初始化优先队列
    priority_queue<pair<double, Node*>> pq;
    pq.push({0.0, start});

    while (!pq.empty()) {
        // 取出综合属性值最高的节点
        Node* u = pq.top().second;
        pq.pop();
        // 计算到达节点 j 的时刻
        double current_time = calculateCurrentTimeAt(u);

        for (auto& edge : u->outgoing_edges) {
            // 计算该时刻的综合属性值
            double z_ij = getAttributeAtTime(z, edge, current_time);
            Node* v = edge.to;
            if (utility[u] + z_ij > utility[v]) {
                utility[v] = utility[u] + z_ij;
                // 记录前驱节点
                predecessor[v] = u;
                pq.push({utility[v], v});
            }
        }
    }
    // 从终点回溯路径
    return reconstructPath(predecessor, start, end);
}
```

## Hazmat Routing-Scheduling Problem (VRP) 1

Source:

- 静态确定性网络中的多目标车辆路径优化问题.  
  [[5]柴获,何瑞春,马昌喜,等.危险品运输车辆路径问题的多目标优化[J].中国安全科学学报,2015,25(10):84-90.DOI:10.16265/j.cnki.issn1003-3033.2015.10.014.](https://kns.cnki.net/kcms2/article/abstract?v=mV2q5OJ_OLwxdurW3LguWPVZaTBD_HkPxSobZNqdnDas9qyAjj_l3brULNzmFyyF7N-3M-dCYTc6P_ycj3jgHI9zSdB0LYObpGsbSH5ods9AAgbkduvjhoNPMQUlJ5bSlJPjix_nYKreHofunmCQESew7meF6xxwxPJy_hUaTlg=&uniplatform=NZKPT)

Model:

$$
\text{3 objectives}
\begin{cases}
  \min \quad \sum d, \\
  \min \quad \sum x, \\
  \min \quad \sum l \\
\end{cases}
$$

$$
\text{3 special constraints}
\begin{cases}
  \text{vehicle capacity}, \\
  \text{time window}, \\
  x,y \in \{0,1\}
\end{cases}
$$

- $\min \sum d$ - minimize distance traveled:
  - Linear function
- $\min \sum x$ - minimize the number of vehicles used:
  - Linear function
- $\min \sum l$ - minimize risk, quantified by distance traveled in densely populated areas:
  - Linear function

Algorithm:

- 基于概率模型的多目标演化算法（Metaheuristic Algorithm）  
  Multi-objective Evolutionary Algorithm based on Probabilistic Modeling
  - 采用演化算法的框架，用“概率模型+抽样”取代了传统的“交叉与变异”，对种群进行代际演化，得到一个高质量 pareto 解集.  
    Evolve the population generationally until the maximum genetic generation is reached to obtain a high-quality pareto solution set.
  - 迭代过程  
    Iteration process
    - 初始化  
      Initialization
    - 采用非支配排序，从当前种群中筛选出最优个体  
      Selection, based on Non-dominated Sorting
    - 根据筛选出的最优个体的特征，更新概率模型，从更新后的概率模型中抽样，构造出新一代种群  
      Reproduction, based on Probabilistic Modeling

Category:

- 从数学规划的角度看：  
  From the perspective of Mathematical Programming,
  1. 顶层分类：确定性优化  
     Top-level: Deterministic Optimization
  2. 子类：离散优化  
     Second-level: Discrete Optimization
  3. 子类：多目标整数线性规划  
     Third-level: Multi-Objective Integer Linear Programming
- 从网络理论的角度看：  
  From the perspective of Network Theory,
  - 带时间窗的多目标容量约束车辆路径问题
    Multi-objective Capacitated Vehicle Routing Problem with time windows

Pseudocode (C++):

```c++
// 基于概率模型的演化算法
vector<Solution> probabilisticModelEA(int population_size, int max_generations) {
    // 1. 初始化
    vector<Solution> population = initializePopulation(population_size);
    evaluate(population);

    for (int g = 0; g < max_generations; ++g) {
        // 2. 选择最优个体
        vector<vector<Solution>> fronts = nonDominatedSort(population);
        vector<Solution>& pareto_front = fronts[0];

        // 3. 建立/更新概率模型
        ProbabilityModel model = learnFromSolutions(pareto_front);

        // 4. 从模型中抽样生成新一代
        vector<Solution> next_population;
        for (int i = 0; i < population_size; ++i) {
            Solution new_sol = generateSolutionFromModel(model);
            next_population.push_back(new_sol);
        }
        evaluate(next_population);

        population = next_population;
    }
    // 返回最优解集
    return nonDominatedSort(population)[0];
}
```

## Hazmat Routing-Scheduling Problem (VRP) 2

Source:

- 静态不确定性（随机性+模糊性）网络中的多目标车辆路径优化问题.  
  [[6]代存杰,李引珍,马昌喜,等.不确定条件下危险品配送路线多准则优化[J].吉林大学学报(工学版),2018,48(06):1694-1702.DOI:10.13229/j.cnki.jdxbgxb20170894.](https://kns.cnki.net/kcms2/article/abstract?v=mV2q5OJ_OLzHipkvsMU6neCy-Twy3EEXY8zA6hPSGN0OA16AxzMoWetxYeKbOOMXNlGTkmB5Nw7jY1UKfV9bokPBrcVPrb0eV2q_fizKeeIX3A4Qicw6NI0MQOZl-UhndyoyLFIEe4DljiC6yOoqMLrm0Mh-Eyp98s3NxTifPcQ=&uniplatform=NZKPT)

Model:

$$
\text{3 objectives}
\begin{cases}
  f_R(x), \\
  f_C(x), \\
  f_T(x) \\
\end{cases}
$$

$$
\text{2 special constraints}
\begin{cases}
  \text{vehicle capacity}, \\
  x, y \in \{0,1\}
\end{cases}
$$

- $f_R(x)$ - risk, consistent with Normal Distribution:
  - Stochastic Chance Constrained Programming
- $f_C(x)$ - cost, which is a Triangular fuzzy number:
  - Fuzzy Programming
- $f_T(x)$ - time, which is a Trapezoidal fuzzy number:
  - Expected Value Model

Algorithm:

- 基于非支配排序和变邻域搜索的模拟退火算法（混合元启发式算法 Hybrid Metaheuristic Algorithm）  
  Non-dominated Sorting Simulated Annealing (NSSA) Algorithm
  - 个体选择：借用 NSGA 的非支配排序方法，取代了标准 SA 的单目标比较，以保证解的质量
  - 解的生成：使用变邻域搜索策略 VNS，取代了标准 SA 的随机邻域搜索，以增强算法的局部搜索能力

Category:

- 从数学规划的角度看：  
  From the perspective of Mathematical Programming,
  1. 顶层分类：不确定优化  
     Top-level: Optimization under Uncertainty
  2. 子类：混合不确定性规划，含（模糊）期望值模型、随机规划、模糊规划  
     Second-level: Hybrid Uncertainty Programming, including (Fuzzy) Expected Value Model, Stochastic Programming, and Fuzzy Programming
  3. 子类：多目标整数非线性规划  
     Third-level: Multi-Objective Integer Non-linear Programming
- 从网络理论的角度看：  
  From the perspective of Network Theory,
  - 多目标容量约束车辆路径问题  
    Multi-objective Capacitated Vehicle Routing Problem

Pseudocode (C++):

```c++
// 基于非支配排序和变邻域搜索的 SA
vector<Solution> NSSA(int pop_size, double T_initial, double T_final, double cooling_rate) {
    // 1. 初始化当前解集
    vector<Solution> current_set = initializePopulation(pop_size);
    evaluate(current_set);
    double T = T_initial;

    while (T > T_final) {
        // 2. 生成新解
        vector<Solution> new_set;
        for (const auto& sol : current_set) {
            // 变邻域搜索
            Solution neighbor_sol = variableNeighborhoodSearch(sol);
            evaluate(neighbor_sol);

            // 3. 接受准则 (一个综合评价值)
            if (isBetter(neighbor_sol, sol) || acceptanceProbability(sol, neighbor_sol, T) > random_uniform(0,1)) {
                new_set.push_back(neighbor_sol);
            }
        }

        // 4. 环境选择
        vector<Solution> combined_set = current_set;
        // 合并解集
        combined_set.insert(combined_set.end(), new_set.begin(), new_set.end());

        // 使用非支配排序进行解的评估
        vector<vector<Solution>> fronts = nonDominatedSort(combined_set);
        // 从最优层级开始选择 n 个解构成新一代
        current_set = selectBest(fronts, pop_size);

        // 5. 降温
        T *= cooling_rate;
    }
    // 返回最优解集
    return current_set;
}
```
