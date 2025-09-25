# --- coding: utf-8 ---
# --- test-dtlz1.py ---

from core.problem_dtlz1 import DTLZ1
from core.nsga import NSGA3
from core.plotter import ParetoPlotter

# --- 主程序入口 ---
if __name__ == '__main__':

    # --- 实验设置 ---

    # DTLZ1 问题参数
    NUM_OBJECTIVES = 3
    NUM_VARIABLES = NUM_OBJECTIVES + 4  # DTLZ1 中 k=5

    # 算法参数
    POPULATION_SIZE = 150 # 初始建议值，会被自动调整
    CROSSOVER_PROB = 0.8
    MUTATION_PROB = 0.1
    MAX_GENERATIONS = 300
    ETA_C = 5
    ETA_M = 5
    
    # --- 实验执行 ---

    # 1. 定义问题
    # 到时候把基准测试函数 DTLZ1 替换成自己的 HazmatProblem 类
    my_problem = DTLZ1(n_vars=NUM_VARIABLES, n_obj=NUM_OBJECTIVES)
    
    # 2. 实例化求解器
    optimizer = NSGA3(
        problem=my_problem,
        n_pop=POPULATION_SIZE,
        pc=CROSSOVER_PROB,
        pm=MUTATION_PROB,
        eta_c=ETA_C,
        eta_m=ETA_M
    )
    
    # 3. 运行算法
    final_solutions, final_objectives = optimizer.run(max_gen=MAX_GENERATIONS)
    
    # -- 结果可视化 ---

    # 4. 实例化绘图器并调用 plot 方法绘图
    plotter = ParetoPlotter(title="DTLZ1 Final Pareto Front", save_dir="results/dtlz1")
    plotter.plot(final_objectives, file_name=f"pareto_gen_{MAX_GENERATIONS}.png")

    print("NSGA-III 求解器已成功封装为对象！")