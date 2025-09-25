# --- coding: utf-8 ---
# --- test-hub_and_spoke.py ---
from core.network_generator import HaSConfig, HaSNetworkGenerator

# --- 主程序入口 ---
if __name__ == "__main__":
    # 创建一个网络配置实例
    config = HaSConfig(num_nodes=15, num_hubs=4, num_emergency_nodes=2, num_tasks=5, road_connect_prob=0.6)

    # 创建一个网络生成器实例
    # 这里的 generator 可以被无缝替换为任何其他继承了 AbstractNetworkGenerator 的生成器
    generator = HaSNetworkGenerator(config=config)
    
    # 调用 generator 方法，生成一个 hub-and-spoke 网络
    my_network = generator.generate()

    # 使用网络
    my_network.summary()
    my_network.visualize()