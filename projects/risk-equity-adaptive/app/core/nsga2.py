# --- coding: utf-8 ---
# --- app/core/nsga2.py ---
import random
import logging
from typing import List, Dict, Any, Optional
from .network import TransportNetwork
from .path import Path
from .solution import Solution
from .evaluator import Evaluator

# --- 回调(Callback)基类 ---
class Callback:
    """
    回调基类。
    """
    def on_generation_end(self, generation: int, population: List[Solution]):
        """在每一代评估和选择 *之后* 调用。"""
        pass

class NSGA2:
    """
    一个 "改进的" NSGA-II 算法实现。
    采用了精英存档 (Elite Archive) 策略来替代标准的人口更替。
    这有助于维持选择压力，避免“早熟”和“遗传漂变”。
    """

    def __init__(self,
                 network: TransportNetwork,
                 evaluator: Evaluator,
                 candidate_paths_map: Dict[str, List[Path]],
                 config: Dict[str, Any]):
        """
        初始化NSGA-II算法。
        """
        self.network = network
        self.evaluator = evaluator
        self.candidate_paths_map = candidate_paths_map
        self.config = config

        # --- 算法超参数 ---
        self.algo_config = self.config.get("algorithm", {})
        self.op_config = self.algo_config.get("operators", {})

        # 种群大小 (Population size, N)
        self.population_size = self.algo_config.get("population_size", 100)
        # 精英存档大小 (Archive size, M)
        self.archive_size = self.algo_config.get("archive_size", 100)
        # 最大迭代次数
        self.max_generations = self.algo_config.get("max_generations", 200)
        # 交叉概率
        self.crossover_prob = self.algo_config.get("crossover_prob", 0.9)
        # 变异参数
        self.prob_mut_path = self.op_config.get("mutation_prob_path", 0.05) 
        # 存档混合概率
        self.mix_ratio = self.op_config.get("mix_ratio", 0.5)

        # 精英存档，代表当前种群，即P(t)
        self.archive: List[Solution] = []

        self.operator_log = {
            "crossover_calls": 0,
            "mutation_path_calls": 0,
        }

    # --- 1. 主运行方法 (Public) ---

    def run(self, callbacks: Optional[List[Callback]] = None, initial_population: Optional[List[Solution]] = None) -> List[Solution]:
        """
        [主方法] 运行完整的NSGA-II进化过程。
        使用精英存档 (SPEA2-like) 流程。
        """
        if callbacks is None:
            callbacks = []

        # 步骤 1: (P_0) 创建并评估初始种群
        # logging.info("开始初始化种群 (P_0)...")
        # initial_population = self._initialize_population()
        # self._evaluate_population(initial_population)
        
        # 步骤 2: (A_0) 用初始种群填充精英存档
        # self.archive = self._update_archive([], initial_population)
        
        if initial_population is None:
            logging.info("开始初始化种群 (P_0)...")
            population = self._initialize_population()
            self._evaluate_population(population)
            self.archive = self._update_archive([], population)
        else:
            logging.info("--------------------------------")
            logging.info("--- 正在从一个已提供的种群重启 ---")
            logging.info("--------------------------------")
            # 假设传入的 population 已经被评估过了
            self.archive = self._update_archive([], initial_population)
        
        # 为存档个体分配适应度等级和拥挤度
        self._assign_ranks_and_crowding(self.archive)
        
        # --- [回调] 触发第0代 (初始存档) ---
        self._trigger_callbacks(0, self.archive, callbacks)
        
        # 步骤 3: (P_t -> Q_t -> R_t -> P_{t+1}) 开始迭代
        for gen in range(self.max_generations):
            # 1. 计算需要“注入”的 *全新* 随机解的数量
            num_immigrants = int(self.mix_ratio * self.population_size)
            
            if num_immigrants > 0:
                # 2. 创建并评估这些 "随机移民" (Immigrants)
                immigrants = self._create_random_solutions(num_immigrants)
                self._evaluate_population(immigrants)
            else:
                immigrants = []

            # 3. 创建繁殖池 (Mating Pool)
            # 繁殖池 = (当前种群 A_t) + (新注入的随机移民)
            mating_pool = self.archive + immigrants

            # (Q_t) 从 *混合后* 的 Mating Pool 中选择父代来创建子代种群
            offspring = self._create_offspring_population(mating_pool)
            
            # (Q_t) 评估子代种群
            self._evaluate_population(offspring)
            
            # (A_{t+1}) 合并 (存档 + 子代)，并选出新的存档
            self.archive = self._update_archive(self.archive, offspring)

            # (A_{t+1}) 重新计算新种群的 Ranks 和 Crowding
            self._assign_ranks_and_crowding(self.archive)

            # --- [回调] 触发本代结束 ---
            # 回调函数接收的是当前最优解集 (存档)
            self._trigger_callbacks(gen + 1, self.archive, callbacks)
            
            if (gen + 1) % 10 == 0:
                logging.debug(f"--- Generation {gen + 1}/{self.max_generations} --- Archive Size: {len(self.archive)}")
            
        # 循环结束，返回最终的精英存档
        logging.info("\n--- 进化完成 ---")
        
        # 统计 Rank 0 的解
        rank_0_solutions = [s for s in self.archive if s.rank == 0 and s.is_feasible]
        logging.info(f"最终在 {len(self.archive)} 个存档解中，找到了 {len(rank_0_solutions)} 个可行的最优解 (Rank 0)。")
        
        logging.info(f"算子统计: {self.operator_log}")
        
        return self.archive

    # --- 2. 核心遗传算子 ---

    def _initialize_population(self) -> List[Solution]:
        """
        创建随机初始种群 (P_0)
        """
        population = []
        for _ in range(self.population_size):
            solution = Solution()
            
            for task in self.network.tasks:
                task_id = task.task_id
                
                if task_id in self.candidate_paths_map:
                    all_paths = self.candidate_paths_map[task_id]
                    if all_paths: 
                        chosen_path = random.choice(all_paths)
                        solution.path_selections[task_id] = chosen_path
                    else:
                        logging.warning(f"警告: 任务 {task_id} 没有任何候选路径！")
                
            population.append(solution)
            
        return population
    
    def _create_random_solutions(self, num_solutions: int) -> List[Solution]:
        """
        创建 'num_solutions' 个随机解
        """
        new_solutions = []
        for _ in range(num_solutions):
            solution = Solution()
            for task in self.network.tasks:
                task_id = task.task_id
                
                if task_id in self.candidate_paths_map:
                    all_paths = self.candidate_paths_map[task_id]
                    if all_paths: 
                        chosen_path = random.choice(all_paths)
                        solution.path_selections[task_id] = chosen_path
                    else:
                        # 第一次初始化时警告，后续 "移民" 不再警告
                        if hasattr(self, 'archive'):    # 检查是否已初始化
                            pass 
                        else:
                            logging.warning(f"警告: 任务 {task_id} 没有任何候选路径！")
                
            new_solutions.append(solution)
            
        return new_solutions

    def _create_offspring_population(self, mating_pool: List[Solution]) -> List[Solution]:
        """
        从整个父代池 mating_pool 中创建 self.population_size 个子代。
        """
        offspring_pop = []
        while len(offspring_pop) < self.population_size:
            # 从 mating pool 中选择
            parent1 = self._selection(mating_pool)
            parent2 = self._selection(mating_pool)
            
            if random.random() < self.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
                self.operator_log["crossover_calls"] += 1
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            self._mutation(child1)
            self._mutation(child2)
            
            offspring_pop.append(child1)
            if len(offspring_pop) < self.population_size:
                offspring_pop.append(child2)
            
        return offspring_pop

    def _selection(self, population: List[Solution]) -> Solution:
        """
        标准 NSGA-II 二元锦标赛，从存档(archive)中选择一个优胜者。
        （作用于 Archive，而不是 P_t)
        """
        p1 = random.choice(population)
        p2 = random.choice(population)
        
        # 使用约束支配来选择
        if self._constrained_dominates(p1, p2):
            return p1
        elif self._constrained_dominates(p2, p1):
            return p2
        
        # 如果两者互不支配 (或 rank 相同)
        if p1.crowding_distance > p2.crowding_distance:
            return p1
        elif p2.crowding_distance > p1.crowding_distance:
            return p2
            
        # 如果 rank 和 crowding 都一样，随机选一个
        return random.choice([p1, p2])

    def _crossover(self, p1: Solution, p2: Solution) -> tuple[Solution, Solution]:
        """
        混合交叉算子。
        基因 (路径) 使用均匀交叉。
        """
        c1 = p1.clone()
        c2 = p2.clone()
        
        for task in self.network.tasks:
            task_id = task.task_id 
            if task_id not in c1.path_selections or task_id not in c2.path_selections:
                continue

            # --- 基因：路径交叉 (均匀交叉) ---
            if random.random() < 0.5:
                # 交换路径
                c1.path_selections[task_id] = p2.path_selections[task_id]
                c2.path_selections[task_id] = p1.path_selections[task_id]
            
        return c1, c2

    def _mutation(self, solution: Solution):
        """
        混合变异算子，直接使用基因层面的概率
        """

        for task in self.network.tasks:
            task_id = task.task_id 

            # 基因：(路径): 使用 "随机重置变异"
            if random.random() < self.prob_mut_path:
                if task_id in self.candidate_paths_map: 
                    all_paths = self.candidate_paths_map[task_id]
                    if all_paths and len(all_paths) > 1:
                        # 确保我们真的选了一个 *新* 路径 (如果可能)
                        current_path = solution.path_selections.get(task_id)
                        new_path = random.choice(all_paths)
                        attempts = 0
                        # 尝试5次以找到一个不同的路径
                        while new_path == current_path and attempts < 5:
                            new_path = random.choice(all_paths)
                            attempts += 1
                            
                        if new_path != current_path:
                            solution.path_selections[task_id] = new_path
                            self.operator_log["mutation_path_calls"] += 1

    # --- 3. 存档更新 ---

    def _update_archive(self, archive: List[Solution], population: List[Solution]) -> List[Solution]:
        """
        更新精英存档。
        1. 合并当前存档和新种群。
        2. 选出所有 Rank 0 的解。
        3. 如果超出存档大小，则进行截断。
        """
        
        # 1. 合并并找出新的 Rank 0 前沿
        combined_pop = archive + population
        fronts = self._fast_non_dominated_sort(combined_pop)
        
        if not fronts:
            return []   # 如果没有解，返回空列表

        new_archive = []
        for front in fronts:
            # 存档未满，直接添加整个前沿
            if len(new_archive) + len(front) <= self.archive_size:
                new_archive.extend(front)
            else:
                # 存档已满，需要“截断”(Truncation)
                # 使用 NSGA-II 的拥挤度排序来保留多样性
                self._calculate_crowding_distance(front)

                # 按拥挤度从高到低排序 (最不拥挤的解排在前面)
                front.sort(key=lambda s: s.crowding_distance, reverse=True)
                
                # 截断存档，只保留 archive_size 个最优且最分散的解
                remaining = self.archive_size - len(new_archive)
                new_archive.extend(front[:remaining])
                break
        
        return new_archive

    # --- 4. 标准 NSGA-II 辅助方法 (现在也用于存档管理) ---

    def _evaluate_population(self, population: List[Solution]):
        """[辅助] 遍历种群，调用 Evaluator 评估每一个解。"""
        # logging.debug(f"正在评估 {len(population)} 个解...")
        for solution in population:
            self.evaluator.evaluate(solution)

    def _fast_non_dominated_sort(self, population: List[Solution]) -> List[List[Solution]]:
        """
        [标准] 执行快速非支配排序。
        (对 'archive + offspring' 的组合体使用)
        """
        # 增加一个去重步骤，防止同一个解被评估多次
        # 基于目标值来去重
        unique_solutions = {}
        for s in population:
            # 使用目标函数值作为 key。如果两个解的目标函数完全一样，我们只保留一个。
            # 注意：这可能会丢弃“基因型不同但表现型相同”的解，但在存档中这是可接受的。
            key = (round(s.f1_risk, 5), round(s.f2_cost, 5), round(s.constraint_violation, 5))
            if key not in unique_solutions:
                unique_solutions[key] = s
        
        unique_pop = list(unique_solutions.values())
        
        fronts = [[]]
        sol_data = {} 

        for p in unique_pop:
            n_p = 0
            S_p = []
            for q in unique_pop:
                if p == q:
                    continue
                
                if self._constrained_dominates(p, q):
                    S_p.append(q)
                elif self._constrained_dominates(q, p):
                    n_p += 1
                
            sol_data[id(p)] = {'n_p': n_p, 'S_p': S_p}
            
            if n_p == 0:
                p.rank = 0 
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in sol_data[id(p)]['S_p']:
                    sol_data[id(q)]['n_p'] -= 1
                    if sol_data[id(q)]['n_p'] == 0:
                        q.rank = i + 1 
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break   # 避免无限循环
                
        return fronts

    def _calculate_crowding_distance(self, front: List[Solution]):
        """
        归一化拥挤度计算。
        """
        if not front:
            return

        for sol in front:
            sol.crowding_distance = 0.0 # 初始化

        # 我们只对可行解计算拥挤度，不可行解的拥挤度默认为0
        feasible_sols = [s for s in front if s.is_feasible]
        infeasible_sols = [s for s in front if not s.is_feasible]

        if not feasible_sols:
            # 归一化约束违反度 (值越小越好)
            max_v = max(s.constraint_violation for s in infeasible_sols) or 1.0
            for sol in infeasible_sols:
                sol.crowding_distance = 1.0 - (sol.constraint_violation / max_v)
            return
        
        # --- 只对可行解进行计算 ---
        if len(feasible_sols) <= 2:
            for s in feasible_sols:
                s.crowding_distance = float('inf')  # 边界点
            return
        
        # 归一化 f1,f2 后计算 crowding
        f1_vals = [s.f1_risk for s in feasible_sols]
        f2_vals = [s.f2_cost for s in feasible_sols]

        f1_min, f1_max = min(f1_vals), max(f1_vals)
        f2_min, f2_max = min(f2_vals), max(f2_vals)

        # 避免除以零
        f1_range = f1_max - f1_min or 1.0
        f2_range = f2_max - f2_min or 1.0

        norm_f1 = [(v - f1_min) / f1_range for v in f1_vals]
        norm_f2 = [(v - f2_min) / f2_range for v in f2_vals]

        for obj_key, values in zip(['f1_risk', 'f2_cost'], [norm_f1, norm_f2]):
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
            feasible_sols[sorted_indices[0]].crowding_distance = float('inf')
            feasible_sols[sorted_indices[-1]].crowding_distance = float('inf')

            for i in range(1, len(values) - 1):
                prev_idx, next_idx = sorted_indices[i - 1], sorted_indices[i + 1]
                dist = values[next_idx] - values[prev_idx]
                feasible_sols[sorted_indices[i]].crowding_distance += dist

    def _assign_ranks_and_crowding(self, population: List[Solution]):
        """[辅助] 评估后，为给定种群分配rank和crowding"""
        fronts = self._fast_non_dominated_sort(population)
        for i, front in enumerate(fronts):
            for sol in front:
                sol.rank = i 
            self._calculate_crowding_distance(front)
    
    def _trigger_callbacks(self, generation: int, archive: List[Solution], callbacks: List[Callback]):
        """触发所有回调函数，传递当前的存档"""
        for callback in callbacks:
            callback.on_generation_end(generation, archive)
            
    # --- 5. 支配关系 (Dominance) 辅助方法 ---

    def _dominates(self, p: Solution, q: Solution) -> bool:
        """
        [支配] 检查 p 是否支配 q (不考虑约束)。
        """
        if p.f1_risk == float('inf') or p.f2_cost == float('inf'):
            return False 
        if q.f1_risk == float('inf') or q.f2_cost == float('inf'):
            return True 

        not_worse = (p.f1_risk <= q.f1_risk) and (p.f2_cost <= q.f2_cost)
        better_in_one = (p.f1_risk < q.f1_risk) or (p.f2_cost < q.f2_cost)
        return not_worse and better_in_one

    def _constrained_dominates(self, p: Solution, q: Solution) -> bool:
        """
        [带约束的支配] 检查 p 是否 带约束地 支配 q
        """
        if p.is_feasible and not q.is_feasible:
            return True # 可行的永远优于不可行的
        
        elif not p.is_feasible and q.is_feasible:
            return False    # 不可行的永远劣于可行的
            
        elif not p.is_feasible and not q.is_feasible:
            # 两个都不可行，约束违反度小的更优
            return p.constraint_violation < q.constraint_violation
            
        else:
            # 两个都可行，使用标准支配
            return self._dominates(p, q)
