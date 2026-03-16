# --- coding: utf-8 ---
# --- benchmark_plotter.py ---

import os

import pandas as pd

from app.utils.plotter import BenchmarkPlotter

folder_name = "2benchmark_small"
file_name = "detailed_metrics_raw.csv"


def main():
    # 1. 设置路径 (需与 benchmark.py 的 save_dir 对应)
    csv_path = "results/" + folder_name + "/benchmark/" + file_name
    save_plots_dir = "results/" + folder_name + "/benchmark"
    os.makedirs(save_plots_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        return

    # 2. 加载数据
    df = pd.read_csv(csv_path)

    # 3. 转换格式为 plotter 所需的字典结构
    # stats_data = { "AlgoName": { "HV": [run1, run2...], ... } }
    stats_data = {}
    metrics = ["HV", "IGD", "SM", "CPU Time"]

    for algo in df["Algorithm"].unique():
        algo_df = df[df["Algorithm"] == algo]
        stats_data[algo] = {m: algo_df[m].tolist() for m in metrics}

    # 4. 执行绘图
    print("📊 正在进行benchmark数据绘图...")
    plotter = BenchmarkPlotter(save_plots_dir)

    # 分别调用三大核心图表
    plotter.plot_metrics_comparison(stats_data)  # 小提琴图
    plotter.plot_normalized_metrics_bar(
        stats_data, "normalized_bar_comparison"
    )  # 柱状图
    plotter.plot_metric_radar(stats_data, "algorithm_radar_comparison")  # 雷达图

    print(f"✨ 绘图完成！图片已保存至: {save_plots_dir}")


if __name__ == "__main__":
    main()
