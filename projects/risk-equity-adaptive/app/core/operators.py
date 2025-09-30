# --- coding: utf-8 ---
# --- operators.py ---
import random
import numpy as np
from pymoo.core.repair import Repair    # type: ignore
from pymoo.core.sampling import Sampling    # type: ignore
from pymoo.core.crossover import Crossover  # type: ignore
from pymoo.core.mutation import Mutation    # type: ignore

# --- 全局日志字典，用于记录算子贡献 ---
operator_log = {
    "crossover": {"calls": 0},
    "mutation": {"calls": 0},
}

# --- 代理交叉算子 ---
class LoggingCrossover(Crossover):
    def __init__(self, crossover_op: Crossover):
        # n_parents, n_offspring 等参数从被包装的算子中继承
        super().__init__(crossover_op.n_parents, crossover_op.n_offsprings)
        self.crossover_op = crossover_op

    def _do(self, problem, X, **kwargs):
        # 1. 记录调用次数
        operator_log["crossover"]["calls"] += 1
        
        # 2. 调用真正的交叉算子执行核心逻辑
        return self.crossover_op._do(problem, X, **kwargs)

# --- 代理变异算子 ---
class LoggingMutation(Mutation):
    def __init__(self, mutation_op: Mutation):
        super().__init__()
        self.mutation_op = mutation_op

    def _do(self, problem, X, **kwargs):
        # 1. 记录调用次数
        operator_log["mutation"]["calls"] += 1

        # 2. 调用真正的变异算子执行核心逻辑
        return self.mutation_op._do(problem, X, **kwargs)

class HubSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # problem 对象就是 HazmatProblem 的一个实例
        X = np.full((n_samples, problem.n_var), -1, dtype=int)

        for i in range(n_samples):
            solution_vector = []
            for task in problem.tasks:
                valid_hub_pairs = problem.feasible_hubs.get(task.id)
                if not valid_hub_pairs:
                    raise Exception(f"任务 {task.id} 没有任何可行的枢纽组合，无法生成初始种群！")
                
                chosen_pair = random.choice(valid_hub_pairs)
                solution_vector.extend(chosen_pair)
            
            X[i, :] = np.array(solution_vector)
            
        return X

class HubCrossover(Crossover):
    """
    本交叉算子专为“可行枢纽对”整数编码问题设计，
    并严格遵循 Pymoo 的三维输入/三维输出模式。

    它对代表任务枢纽选择的整数向量进行均匀交叉，确保子代的可行性。
    
    输入 X Shape: (2, n_matings, n_var)  (n_parents, n_matings, n_var)
    输出 Y Shape: (2, n_matings, n_var)  (n_offsprings, n_matings, n_var)
    """
    def __init__(self, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)

    def _do(self, problem, X, **kwargs):
        # 1. 获取输入的维度信息
        #    X 的形状是 (n_parents, n_matings, n_var)
        n_parents, n_matings, n_var = X.shape

        # 2. 创建一个与输入X形状完全相同的输出数组Y
        #    因为 n_offsprings 和 n_parents 都是 2，所以形状可以直接复用
        Y = np.full_like(X, -1, dtype=int)

        # 3. 提取父代1和父代2的矩阵
        #    parent1 的形状是 (n_matings, n_var)
        parent1 = X[0]
        parent2 = X[1]

        # 4. 核心交叉逻辑: 对每个任务的枢纽对进行均匀交叉
        #    mask 的形状为 (n_matings, num_tasks)
        mask = np.random.rand(n_matings, len(problem.tasks)) < 0.5

        for task_idx in range(len(problem.tasks)):
            start, end = task_idx * 2, task_idx * 2 + 2
            
            # task_mask 的形状为 (n_matings,)，是一个布尔数组
            task_mask = mask[:, task_idx]

            # 根据 task_mask 将父代的基因片段分配给子代
            # 当 task_mask 为 True 时，子代1从父代1继承，子代2从父代2继承
            Y[0, task_mask, start:end] = parent1[task_mask, start:end]
            Y[1, task_mask, start:end] = parent2[task_mask, start:end]

            # 当 task_mask 为 False 时 (~) ，交换继承关系
            Y[0, ~task_mask, start:end] = parent2[~task_mask, start:end]
            Y[1, ~task_mask, start:end] = parent1[~task_mask, start:end]
        
        # 5. 返回符合 Pymoo 要求的三维数组 Y
        return Y

class LocalSearchMutation(Mutation):
    """
    一个融合了全局探索和局部寻优的混合突变算子。

    它在算法的突变阶段，根据不同概率执行两种操作：
    1. 全局探索 (大步长): 随机更换一个任务的整个枢纽对 (来自 HubSwapMutation)。
    2. 局部寻优 (小步长): 对一个任务进行单点邻域搜索 (来自 LocalSearchRepair)。
    """
    def __init__(self, problem: 'HazmatProblem', 
                 global_prob: float = 0.2, 
                 local_prob: float = 0.8):
        """
        初始化混合突变算子。
        
        Args:
            problem: HazmatProblem 实例。
            global_prob (float): 触发全局探索（大步长）的概率。
            local_prob (float): 触发局部寻优（小步长）的概率。
                                注意：总的触发概率是 global_prob + local_prob。
        """
        super().__init__()
        self.problem = problem
        self.global_prob = global_prob
        self.local_prob = local_prob

    def _do(self, problem, X, **kwargs):
        # 遍历种群中的每个个体
        for i in range(len(X)):
            
            # --- 决定执行哪种操作 ---
            trigger = random.random()
            
            if trigger < self.global_prob:
                # --- 执行全局探索 (大步长) ---
                self._perform_global_search(X, i)
                
            elif trigger < self.global_prob + self.local_prob:
                # --- 执行局部寻优 (小步长) ---
                self._perform_local_search(X, i)

        return X

    def _perform_global_search(self, X, individual_idx):
        """全局探索：随机替换一个任务的枢纽对。"""
        individual = X[individual_idx]
        num_tasks = len(self.problem.tasks)
        
        task_to_mutate_idx = random.randint(0, num_tasks - 1)
        task_id = self.problem.tasks[task_to_mutate_idx].id
        
        start_idx = task_to_mutate_idx * 2
        current_pair = tuple(individual[start_idx : start_idx + 2])
        
        valid_pairs = self.problem.feasible_hubs.get(task_id, [])
        if len(valid_pairs) <= 1:
            return

        new_pair = current_pair
        while new_pair == current_pair:
            new_pair = random.choice(valid_pairs)
        
        individual[start_idx : start_idx + 2] = new_pair

    def _perform_local_search(self, X, individual_idx):
        """局部寻优：对随机一个任务进行单点邻域搜索。"""
        current_individual = X[individual_idx].copy()
        num_tasks = len(self.problem.tasks)
        
        task_to_improve_idx = random.randint(0, num_tasks - 1)
        task_id = self.problem.tasks[task_to_improve_idx].id

        out_current = {}
        self.problem._evaluate(np.array([current_individual]), out_current)
        current_f1, current_f2 = out_current["F"][0, 0], out_current["F"][0, 1]

        start_idx = task_to_improve_idx * 2
        best_hub_pair = tuple(current_individual[start_idx : start_idx + 2])
        k1_current, l1_current = best_hub_pair
        
        feasible_pairs_for_task = self.problem.feasible_hubs.get(task_id, [])
        
        for k_new, l_new in feasible_pairs_for_task:
            is_neighbor = (k_new == k1_current and l_new != l1_current) or \
                          (k_new != k1_current and l_new == l1_current)
            
            if not is_neighbor:
                continue

            neighbor_individual = current_individual.copy()
            neighbor_individual[start_idx : start_idx + 2] = [k_new, l_new]

            out_neighbor = {}
            self.problem._evaluate(np.array([neighbor_individual]), out_neighbor)
            neighbor_f1, neighbor_f2 = out_neighbor["F"][0, 0], out_neighbor["F"][0, 1]

            if (neighbor_f1 <= current_f1 and neighbor_f2 < current_f2) or \
               (neighbor_f1 < current_f1 and neighbor_f2 <= current_f2):
                current_f1, current_f2 = neighbor_f1, neighbor_f2
                best_hub_pair = (k_new, l_new)
        
        current_individual[start_idx : start_idx + 2] = best_hub_pair
        X[individual_idx] = current_individual