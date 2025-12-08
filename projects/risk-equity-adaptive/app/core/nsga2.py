# --- coding: utf-8 ---
# --- app/core/nsga2.py ---
"""
[算法层] NSGA-II 主流程
只负责种群的进化循环 (Selection, Crossover, Mutation, Archive Update)。
底层排序逻辑委托给 MOEAUtils。
"""

import random
import logging
from typing import List, Dict, Any
from app.core.network import TransportNetwork
from app.core.path import Path
from app.core.solution import Solution
from app.core.evaluator import Evaluator
from app.core.moea_utils import MOEAUtils


class Callback:
    def on_generation_end(self, generation: int, population: List[Solution]):
        pass


class NSGA2:
    def __init__(
        self,
        network: TransportNetwork,
        evaluator: Evaluator,
        candidate_paths_map: Dict[str, List[Path]],
        config: Dict[str, Any],
    ):
        self.network = network
        self.evaluator = evaluator
        self.candidate_paths_map = candidate_paths_map

        algo = config.get("algorithm", {})
        self.pop_size = algo.get("population_size", 100)
        self.archive_size = algo.get("archive_size", 100)
        self.max_gen = algo.get("max_generations", 200)
        self.cx_prob = algo.get("crossover_prob", 0.9)
        self.mut_prob = algo.get("operators", {}).get("mutation_prob_path", 0.05)
        self.mix_ratio = algo.get("operators", {}).get("mix_ratio", 0.5)

        self.archive: List[Solution] = []
        self.op_log = {"crossover": 0, "mutation": 0}

    def run(
        self,
        callbacks: List[Callback] = None,
        initial_population: List[Solution] = None,
    ) -> List[Solution]:
        if not callbacks:
            callbacks = []

        # P0 -> A0
        if not initial_population:
            pop = self._init_pop()
            self._evaluate(pop)
            self.archive = self._update_archive([], pop)
        else:
            self.archive = self._update_archive([], initial_population)

        self._notify(0, callbacks)

        for gen in range(self.max_gen):
            # Immigrants
            n_new = int(self.mix_ratio * self.pop_size)
            immigrants = self._init_pop(n_new)
            self._evaluate(immigrants)

            # Mating Pool
            pool = self.archive + immigrants

            # Offspring
            offspring = self._create_offspring(pool)
            self._evaluate(offspring)

            # Update Archive
            self.archive = self._update_archive(self.archive, offspring)
            self._assign_metrics(self.archive)

            self._notify(gen + 1, callbacks)

            if (gen + 1) % 10 == 0:
                logging.info(f"Gen {gen + 1}: Archive Size {len(self.archive)}")

        return self.archive

    def _update_archive(
        self, current: List[Solution], new_batch: List[Solution]
    ) -> List[Solution]:
        """核心逻辑：合并 -> 排序 -> 截断"""
        combined = current + new_batch

        # Sorting
        fronts = MOEAUtils.fast_non_dominated_sort(combined)

        new_archive = []
        for front in fronts:
            if len(new_archive) + len(front) <= self.archive_size:
                new_archive.extend(front)
            else:
                # Crowding
                MOEAUtils.calculate_crowding_distance(front)
                front.sort(key=lambda s: s.crowding_distance, reverse=True)
                needed = self.archive_size - len(new_archive)
                new_archive.extend(front[:needed])
                break
        return new_archive

    # --- Genetic Operators ---
    def _create_offspring(self, pool: List[Solution]) -> List[Solution]:
        children = []
        while len(children) < self.pop_size:
            p1 = self._select(pool)
            p2 = self._select(pool)
            if random.random() < self.cx_prob:
                c1, c2 = self._crossover(p1, p2)
                self.op_log["crossover"] += 1
            else:
                c1, c2 = p1.clone(), p2.clone()
            self._mutate(c1)
            self._mutate(c2)
            children.extend([c1, c2])
        return children[: self.pop_size]

    def _select(self, pop: List[Solution]) -> Solution:
        """锦标赛选择"""
        a, b = random.choice(pop), random.choice(pop)
        # Comparison
        if MOEAUtils.constrained_dominates(a, b):
            return a
        if MOEAUtils.constrained_dominates(b, a):
            return b
        return a if a.crowding_distance > b.crowding_distance else b

    def _crossover(self, p1: Solution, p2: Solution):
        c1, c2 = p1.clone(), p2.clone()
        for task in self.network.tasks:
            if random.random() < 0.5:
                tid = task.task_id
                c1.path_selections[tid], c2.path_selections[tid] = (
                    p2.path_selections[tid],
                    p1.path_selections[tid],
                )
        return c1, c2

    def _mutate(self, sol: Solution):
        """
        变异：确保选出的新路径与当前路径不同，避免无效变异。
        """
        for task in self.network.tasks:
            if random.random() < self.mut_prob:
                tid = task.task_id
                opts = self.candidate_paths_map.get(tid, [])
                
                if len(opts) > 1:
                    # 获取当前选择的路径
                    current_path = sol.path_selections.get(tid)
                    
                    # 过滤出除了当前路径以外的所有候选路径
                    # 确保变异一定会改变基因，引入多样性
                    candidates = [p for p in opts if p != current_path]
                    
                    if candidates:
                        sol.path_selections[tid] = random.choice(candidates)
                        self.op_log["mutation"] += 1

    # --- Helpers ---
    def _init_pop(self, size: int = None) -> List[Solution]:
        count = size if size else self.pop_size
        pop = []
        for _ in range(count):
            sol = Solution()
            for task in self.network.tasks:
                opts = self.candidate_paths_map.get(task.task_id, [])
                if opts:
                    sol.path_selections[task.task_id] = random.choice(opts)
            pop.append(sol)
        return pop

    def _evaluate(self, pop: List[Solution]):
        for s in pop:
            self.evaluator.evaluate(s)

    def _assign_metrics(self, pop: List[Solution]):
        fronts = MOEAUtils.fast_non_dominated_sort(pop)
        for i, front in enumerate(fronts):
            for s in front:
                s.rank = i
            MOEAUtils.calculate_crowding_distance(front)

    def _notify(self, gen, cbs):
        for cb in cbs:
            cb.on_generation_end(gen, self.archive)
