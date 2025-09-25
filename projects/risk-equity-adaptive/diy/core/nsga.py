# --- coding: utf-8 ---
# --- nsgaiii.py ---

import random
import math
import numpy as np
from scipy.spatial.distance import cdist    # type: ignore

# --- NSGA-III 求解器类 ---
class NSGA3:
    def __init__(self, problem, n_pop=1000, pc=0.9, pm=0.1, eta_c=20, eta_m=20):
        """
        NSGA-III 求解器对象

        Args:
            problem (Problem): A instance that defined n_vars, n_obj, xl, xu, and evaluate method.
            n_pop (int): The initial suggested population size(will be adjusted based on reference points).
            pc (float): The probability of crossover.
            pm (float): The probability of mutation.
            eta_c (float): SBX crossover distribution index.
            eta_m (float): Polynomial mutation distribution index.
        """
        self.problem = problem
        self.pc = pc
        self.pm = pm
        self.eta_c = eta_c
        self.eta_m = eta_m

        # 初始化参考点和种群大小
        self.ref_points = self._generate_reference_points(n_pop)
        self.n_pop = self.ref_points.shape[0]
        print(f"在种群初始化过程中，据目标数和分割数，种群大小自动调整为: {self.n_pop}")

        # 初始化种群状态
        self.population = None
        self.objectives = None

    def _generate_reference_points(self, n_pop: int):
        """生成均匀分布的参考点"""
        # 计算参考点所需的分割数 p
        p = 0
        while math.comb(p + self.problem.n_obj - 1, self.problem.n_obj - 1) <= n_pop:
            p += 1
        
        # 使用递归方式生成参考点
        def get_points_recursive(points, n_obj, divisions, layer):
            if layer == n_obj - 1:
                points[:, layer] = divisions - np.sum(points[:, :layer], axis=1)
                return points / divisions
            else:
                new_points = []
                # 遍历当前层的可能取值
                for i in range(divisions - int(np.sum(points[:, :layer], axis=1)[0]) + 1):
                    temp_points = points.copy()
                    temp_points[:, layer] = i
                    new_points.append(get_points_recursive(temp_points, n_obj, divisions, layer + 1))
                return np.vstack(new_points)

        initial_points = np.zeros((1, self.problem.n_obj))
        return get_points_recursive(initial_points, self.problem.n_obj, p, 0)
    
    def _fast_non_dominated_sort(self, objectives):
        """
        执行快速非支配排序
        
        Args:
            objectives (np.adarray): 目标函数值矩阵，形状为 (n_pop, n_obj)
        
        Returns:
            fronts (list[list[int]]): 包含每个Pareto前沿个体索引的列表。  
            ranks (np.ndarray): 每个个体的支配等级(rank)
        """
        n_pop, n_obj = objectives.shape
        domination_counts = np.zeros(n_pop, dtype=int)
        dominated_solutions = [[] for _ in range(n_pop)]
        ranks = np.zeros(n_pop, dtype=int)
        fronts = [[]]

        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                #  检查 i 和 j 的支配关系
                dom_i_j = 0 # i 支配 j 的数目
                dom_j_i = 0 # j 支配 i 的数目
                for k in range(n_obj):
                    if objectives[i, k] < objectives[j, k]:
                        dom_i_j += 1
                    elif objectives[j, k] < objectives[i, k]:
                        dom_j_i += 1
                
                # i 完全支配 j
                if dom_i_j == n_obj:
                    dominated_solutions[i].append(j)
                    domination_counts[j] += 1
                # j 完全支配 i
                elif dom_j_i == n_obj:
                    dominated_solutions[j].append(i)
                    domination_counts[i] += 1

        for i in range(n_pop):
            # 初始化第 1 个 pareto frontier
            if domination_counts[i] == 0:
                fronts[0].append(i)
                ranks[i] = 0

        # 逐层构建后续 pareto frontier
        front_index = 0
        while front_index < len(fronts):
            next_front = []
            for i in fronts[front_index]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
                        ranks[j] = front_index + 1
            front_index += 1
            if next_front:
                fronts.append(next_front)

        return fronts, ranks

    # ... (其他辅助方法如 _selection, _crossover, _mutation, _environmental_selection)
    def _binary_tournament_selection(self, population, ranks):
        """
        通过二元锦标赛选择法构建交配池。

        Args:
            population (np.ndarray): 当前种群。
            ranks (np.ndarray): 种群中每个个体的等级。
            pool_size (int): 需要选择的个体数量，通常为种群大小。

        Returns:
            np.ndarray: 构建好的交配池。
        """
        n_pop = population.shape[0]
        mating_pool = np.zeros_like(population)
        
        for i in range(n_pop):
            # 随机选择两个不同的个体进行比赛
            p1, p2 = np.random.choice(n_pop, 2, replace=False)
            # 选择等级更优（rank值更小）的个体
            winner_idx = p1 if ranks[p1] <= ranks[p2] else p2
            mating_pool[i] = population[winner_idx]
            
        return mating_pool

    def _simulated_binary_crossover(self, mating_pool, pc, eta_c, lb, ub):
        """
        模拟二进制交叉 (SBX)。

        Args:
            mating_pool (np.ndarray): 交配池。
            pc (float): 交叉概率。
            eta_c (float): 交叉分布指数。
            lb (np.ndarray): 决策变量下界。
            ub (np.ndarray): 决策变量上界。

        Returns:
            np.ndarray: 交叉后生成的子代。
        """
        n_off, n_vars = mating_pool.shape
        offspring = np.zeros_like(mating_pool)
        
        for i in range(0, n_off, 2):
            if i >= n_off - 1:
                break
            parent1, parent2 = mating_pool[i], mating_pool[i+1]
            
            if np.random.rand() <= pc:
                # 执行交叉
                mu = np.random.rand(n_vars)
                beta = np.empty(n_vars)
                
                # 计算 beta
                beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1.0 / (eta_c + 1.0))
                beta[mu > 0.5] = (1.0 / (2.0 * (1.0 - mu[mu > 0.5]))) ** (1.0 / (eta_c + 1.0))
                
                # 生成子代
                offspring[i] = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
                offspring[i+1] = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
            else:
                # 不交叉，直接复制父代
                offspring[i], offspring[i+1] = parent1.copy(), parent2.copy()
                
        # 使用 np.clip 保证子代在边界内，原代码的min/max方式是错误的
        offspring = np.clip(offspring, lb, ub)
        return offspring

    def _differential_evolution_crossover(self, population, F=0.5, CR=0.8):
        """
        差分进化交叉算子 (DE/rand/1/bin)。
        这是一个探索能力非常强的算子。
        
        Args:
            population (np.ndarray): 当前种群。
            F (float): 缩放因子。
            CR (float): 交叉概率。

        Returns:
            offspring (np.ndarray): 生成的子代。
        """
        n_pop, n_vars = population.shape
        offspring = np.zeros_like(population)
        
        for i in range(n_pop):
            # 1. 从种群中随机选择三个与当前个体不同的个体
            indices = list(range(n_pop))
            indices.remove(i)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)
            
            # 2. 生成变异向量 (donor vector)
            donor_vector = population[r1] + F * (population[r2] - population[r3])
            # 确保变异向量在边界内
            donor_vector = np.clip(donor_vector, self.problem.xl, self.problem.xu)
            
            # 3. 进行交叉操作 (binomial crossover)
            trial_vector = population[i].copy()
            j_rand = np.random.randint(0, n_vars) # 确保至少有一个维度被替换
            
            for j in range(n_vars):
                if np.random.rand() < CR or j == j_rand:
                    trial_vector[j] = donor_vector[j]
            
            offspring[i] = trial_vector
            
        return offspring

    
    def _polynomial_mutation(self, population, pm, eta_m, lb, ub):
        """
        多项式变异。

        Args:
            population (np.ndarray): 种群。
            pm (float): 变异概率。
            eta_m (float): 变异分布指数。
            lb (np.ndarray): 决策变量下界。
            ub (np.ndarray): 决策变量上界。

        Returns:
            np.ndarray: 变异后的种群。
        """
        n_pop, n_vars = population.shape
        mutated_pop = population.copy()
        
        for i in range(n_pop):
            if np.random.rand() <= pm:
                for j in range(n_vars):
                    mu = np.random.rand()
                    delta = 0.0
                    
                    if mu < 0.5:
                        delta = (2 * mu) ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1 - mu)) ** (1.0 / (eta_m + 1.0))
                    
                    mutated_pop[i, j] += delta * (ub[j] - lb[j])
                    
        # 使用 np.clip 保证边界
        mutated_pop = np.clip(mutated_pop, lb, ub)
        return mutated_pop
    
    def _environmental_selection(self, combined_pop, combined_objs, n_pop, ref_points):
        """
        NSGA-III 的环境选择机制。

        Args:
            combined_pop (np.ndarray): 父代和子代合并后的种群，形状为 (2*n_pop, n_vars)
            combined_objs (np.ndarray): 合并后种群的目标函数值，形状为 (2*n_pop, n_obj)
            n_pop (int): 需要选择的下一代种群大小
            ref_points (np.ndarray): 参考点，形状为 (n_pop, n_obj)

        Returns:
            tuple: (下一代种群 (np.array), 下一代目标值 (np.array), 下一代的等级 (np.ndarray))
        """
        # 步骤 1: 对合并后的种群进行快速非支配排序
        fronts, _ = self._fast_non_dominated_sort(combined_objs)

        # 初始化下一代种群列表
        next_pop = []
        next_objs = []
        front_idx = 0
        
        # --- 处理精英超额和正常情况 ---
        # Case 1: 第1个前沿（最优秀的解）的个体数已经超过种群大小 n_pop
        if len(fronts[0]) > n_pop:
            # 我们只在第 1 个前沿中进行选择，后续前沿不再考虑
            last_front_indices = fronts[0]
            # 此时，已选的个体为 empty
            current_pop_objs = np.array([]) 
            # 需要从第 1 个前沿中选出 n_pop 个个体
            K = n_pop
        # Case 2: 第 1 个前沿的个体数不超过 n_pop
        else:
            # 逐个前沿填充下一代，直到无法完整放入一个前沿
            while len(next_pop) + len(fronts[front_idx]) <= n_pop:
                # 获取当前前沿的所有个体的索引
                current_front_indices = fronts[front_idx]
                # 将这些个体直接添加到下一代
                next_pop.extend(combined_pop[current_front_indices])
                next_objs.extend(combined_objs[current_front_indices])
                # 更新索引，准备处理下一个前沿
                if front_idx + 1 < len(fronts):
                    front_idx += 1
            
            # 如果种群刚好填满，说明最后一个完整前沿恰好满足数量要求，正好返回
            if len(next_pop) == n_pop:
                # 此时无需进行小生境选择，直接返回结果
                _, final_ranks = self._fast_non_dominated_sort(np.array(next_objs))
                return np.array(next_pop), np.array(next_objs), final_ranks

            # 如果种群未满，则需要从下一个前沿（即最后一个需要部分选择的前沿）中挑选
            last_front_indices = fronts[front_idx]
            # 已经确定选入的个体目标值
            current_pop_objs = np.array(next_objs)
            # 计算还需要挑选的个体数量
            K = n_pop - len(next_pop)               
        
        # --- 步骤 2: 小生境保留操作 (Niche-Preservation) ---
        # 无论上述哪种情况，此步骤都会对 last_front_indices 中的个体进行多样性选择
        
        # 2.1 规范化目标值
        # 获取最后一个前沿个体对应的目标值
        last_front_objs = combined_objs[last_front_indices]
        
        # 根据情况一或情况二，构建用于规范化的目标值集合
        if current_pop_objs.size > 0:
            # 正常情况：合并已选个体和最后一个前沿的个体
            temp_objs = np.vstack((current_pop_objs, last_front_objs))
        else:
            # “精英超额”情况：只使用第一个前沿的个体
            temp_objs = last_front_objs

        # 找到理想点 (ideal point)
        z_min = np.min(temp_objs, axis=0)
        # 所有目标值减去理想点，进行平移
        normalized_objs = temp_objs - z_min

        # 寻找极端点，以构建参考超平面
        n_obj = combined_objs.shape[1]
        # 使用 ASF (Achievement Scalarization Function) 方法寻找极端点
        weights = np.eye(n_obj) * 1e6 + 1e-6
        # 找到每个轴上 ASF 值最小的个体索引
        extreme_points_idx = np.argmin(np.max(normalized_objs / weights[:, np.newaxis, :], axis=2), axis=0)

        # 计算超平面的截距
        try:
            # 通过求解线性方程组 Ax=b (A:极端点目标值, b:全1向量) 来找到法向量
            hyperplane = np.linalg.solve(normalized_objs[extreme_points_idx], np.ones(n_obj))
            # 截距是法向量各分量的倒数
            intercepts = 1.0 / hyperplane
            # 如果截距过小或为负（可能由共线性引起），则使用备用方法（直接使用最大值作为截距）
            if np.any(intercepts <= 1e-6):
                intercepts = np.max(normalized_objs, axis=0) - z_min
        except np.linalg.LinAlgError:
            # 如果矩阵非满秩、无法求解，则直接使用最大值作为截距
            intercepts = np.max(normalized_objs, axis=0) - z_min

        # 避免除以0
        intercepts[intercepts < 1e-6] = 1e-6
        # 使用截距对目标值进行规范化
        normalized_objs = normalized_objs / intercepts
        
        # 2.2 关联个体与参考点
        # 计算所有参与规范化的个体到每个参考点的垂直距离
        dist_matrix = cdist(normalized_objs, ref_points)
        # 每个个体都与距离自己最近的那个参考点相关联
        associations = np.argmin(dist_matrix, axis=1)
        
        # 2.3 小生境保留 Niche-preservation operation
        # 计算每个参考点已经关联的个体数量 (只计算已选入前沿的个体)
        niche_counts = np.zeros(len(ref_points), dtype=int)
        # len(next_pop) 是已确定入选的个体数，在精英超额情况下为0
        for assoc in associations[:len(next_pop)]:
            niche_counts[assoc] += 1
            
        # 获取最后一个前沿中，每个个体的关联情况和距离
        last_front_associations = associations[len(next_pop):]
        last_front_distances = np.min(dist_matrix[len(next_pop):], axis=1)

        # 构建一个候选池，包含最后一个前沿中所有待选个体的信息
        # 格式: [个体在last_front_indices中的索引, 关联的参考点索引, 与该参考点的距离]
        candidate_pool = []
        for i in range(len(last_front_indices)):
            candidate_pool.append([i, last_front_associations[i], last_front_distances[i]])

        # 循环 K 次，从候选池中选出 K 个最优且最多样化的个体
        for _ in range(K):
            # 找到当前小生境计数值最小的参考点（最不拥挤的参考点）
            min_niche_idx = np.argmin(niche_counts)
            
            # 从候选池中，找到所有关联到这个最不拥挤参考点的个体
            associating_candidates = [p for p in candidate_pool if p[1] == min_niche_idx]
            
            # 如果找到了候选个体
            if associating_candidates:
                # 如果这个最不拥挤的参考点原本是空的（计数值为0）
                if niche_counts[min_niche_idx] == 0:
                    # 那么就从候选者中挑选距离该参考点最近的那个（最“名副其实”的）
                    best_candidate = min(associating_candidates, key=lambda p: p[2])
                else:
                    # 如果这个参考点已经有关联个体了，就从候选者中随机挑选一个，以增加多样性
                    best_candidate = random.choice(associating_candidates)
                
                # 将选中的个体从候选池中移除，避免重复选择
                candidate_pool.remove(best_candidate)
                
                # 获取该个体在 last_front_indices 中的原始索引
                chosen_idx_in_last_front = best_candidate[0]
                
                # 将选中的个体正式添加到下一代种群中
                original_pop_idx = last_front_indices[chosen_idx_in_last_front]
                next_pop.append(combined_pop[original_pop_idx])
                next_objs.append(combined_objs[original_pop_idx])
                
                # 更新该参考点的小生境计数值
                niche_counts[min_niche_idx] += 1
            
            # 如果没找到候选个体（即最不拥挤的参考点也没有任何待选个体与之关联）
            else:
                # 将这个参考点的计数值设为整数最大值，相当于将其“排除”，下一轮循环不会再选中它
                max_int = np.iinfo(niche_counts.dtype).max
                niche_counts[min_niche_idx] = max_int

        # 最终，对选出的新一代种群计算一次 rank (主要用于返回信息)
        _, final_ranks = self._fast_non_dominated_sort(np.array(next_objs))
        
        return np.array(next_pop), np.array(next_objs), final_ranks
    
    def _environmental_selection_nsga2(self, combined_pop, combined_objs, n_pop):
        """
        NSGA-II 的环境选择机制（拥挤度距离）。
        这是一个对照组，用于诊断 NSGA-III 实现中的问题。
        """
        fronts, _ = self._fast_non_dominated_sort(combined_objs)
        
        next_pop = []
        front_idx = 0
        
        # 逐个前沿填充下一代，直到无法完整放入一个前沿
        while len(next_pop) + len(fronts[front_idx]) <= n_pop:
            next_pop.extend(combined_pop[fronts[front_idx]])
            front_idx += 1
            
        # 如果种群刚好填满，直接返回
        if len(next_pop) == n_pop:
            next_objs = self.problem.evaluate(np.array(next_pop))
            return np.array(next_pop), next_objs, None # NSGA-II 不返回 ranks

        # --- 拥挤度计算 ---
        # 需要部分选择的最后一个前沿
        last_front_indices = fronts[front_idx]
        K = n_pop - len(next_pop)
        
        # 获取最后一个前沿的目标值
        last_front_objs = combined_objs[last_front_indices]
        n_points, n_obj = last_front_objs.shape
        
        # 初始化拥挤度距离
        crowding_distance = np.zeros(n_points)
        
        # 对每个目标维度进行计算
        for m in range(n_obj):
            # 按当前目标值排序
            sorted_indices = np.argsort(last_front_objs[:, m])
            # 边界点的拥挤度设为无穷大，以确保它们被保留
            crowding_distance[sorted_indices[0]] = np.inf
            crowding_distance[sorted_indices[-1]] = np.inf
            
            # 如果前沿不止两个点，计算中间点的拥挤度
            if n_points > 2:
                obj_min = last_front_objs[sorted_indices[0], m]
                obj_max = last_front_objs[sorted_indices[-1], m]
                # 归一化分母，避免除以零
                denominator = obj_max - obj_min
                if denominator == 0:
                    denominator = 1e-6

                # 累加每个点的距离
                for i in range(1, n_points - 1):
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]
                    crowding_distance[sorted_indices[i]] += (last_front_objs[next_idx, m] - last_front_objs[prev_idx, m]) / denominator

        # --- 根据拥挤度选择 ---
        # 将最后一个前沿的个体与他们的拥挤度配对
        # 选择拥挤度最大的 K 个个体
        # argsort 默认升序，所以我们取最后 K 个
        chosen_indices_in_last_front = np.argsort(crowding_distance)[-K:]
        
        # 从 last_front_indices 中获取原始索引
        final_chosen_indices = [last_front_indices[i] for i in chosen_indices_in_last_front]
        next_pop.extend(combined_pop[final_chosen_indices])
        
        next_objs = self.problem.evaluate(np.array(next_pop))
        return np.array(next_pop), next_objs, None


    def run(self, max_gen):
        """
        执行算法的主函数。

        Args:
            max_gen (int): 最大迭代次数。

        Returns:
            np.ndarray: 最终Pareto前沿上的决策变量。
            np.ndarray: 最终Pareto前沿上的目标函数值。
        """
        # 步骤 1: 初始化种群
        self.population = np.random.uniform(self.problem.xl, self.problem.xu, size=(self.n_pop, self.problem.n_vars))
        self.objectives = self.problem.evaluate(self.population)

        # 步骤 2: 主循环
        for gen in range(max_gen):
            print(f"--- Generation {gen + 1}/{max_gen} ---")

            # 2.1 生成子代
            _, ranks = self._fast_non_dominated_sort(self.objectives)
            mating_pool = self._binary_tournament_selection(self.population, ranks)
            # --- 暂时切换到 DE 交叉算子，并禁用多项式变异 ---
            # offspring_pop = self._differential_evolution_crossover(mating_pool, F=0.5, CR=0.9)
            offspring_pop = self._simulated_binary_crossover(mating_pool, self.pc, self.eta_c, self.problem.xl, self.problem.xu)
            offspring_pop = self._polynomial_mutation(offspring_pop, self.pm, self.eta_m, self.problem.xl, self.problem.xu)
            offspring_objs = self.problem.evaluate(offspring_pop)
            
            # 2.2 环境选择
            combined_pop = np.vstack((self.population, offspring_pop))
            combined_objs = np.vstack((self.objectives, offspring_objs))
            # --- 暂时切换到 NSGA-II 的选择机制进行诊断 ---
            # self.population, self.objectives, _ = self._environmental_selection_nsga2(combined_pop, combined_objs, self.n_pop)
            self.population, self.objectives, _ = self._environmental_selection(combined_pop, combined_objs, self.n_pop, self.ref_points)

        # 步骤 3: 结束与输出
        final_fronts, _ = self._fast_non_dominated_sort(self.objectives)
        final_pareto_indices = final_fronts[0]
        
        print("\n优化完成！")
        return self.population[final_pareto_indices], self.objectives[final_pareto_indices]
