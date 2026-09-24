"""RHMVSP 模型参数与绘图常量。"""

from __future__ import annotations
import json
from dataclasses import asdict, dataclass


EMPTY_REPOSITION_HUB_IDS = (1, 12, 38)
EMPTY_REPOSITION_HUB_FLEET_SIZE = 50
INSTANCE_DATA_VERSION = "rhmvsp-real-case-v1"


def default_empty_reposition_hub_fleet() -> dict[str, int]:
	return {
		str(hub_id): EMPTY_REPOSITION_HUB_FLEET_SIZE
		for hub_id in EMPTY_REPOSITION_HUB_IDS
	}


@dataclass
class RHMVSPConfig:
	"""可靠危险品多式联运车辆调度问题的配置。"""

	# ============================================================
	# 模型参数
	# ============================================================

	T_max: int = 168  # 真实案例规划期和所有 OD 的统一运输期限（小时）

	# 运载能力
	q_road: float = 1.0  # 单辆公路车辆固定装载一个 40 英尺集装箱，即 1 FEU
	q_rail: float = 40.0  # 每次铁路发车最多装载 40 个 40 英尺集装箱，即 40 FEU

	# 表列费率均为应用第三类危险品成本系数之前的原始费率
	hazmat_cost_coefficient: float = 1.5
	c_road_fixed: float = 675.0  # 公路运输固定费率（元/FEU）
	c_road: float = 36.0  # 公路运输里程费率（元/FEU·公里）
	c_empty_reposition_per_km: float = 18.0  # 公路车辆空驶调位费率：载运里程费率的一半（元/车·公里）
	# 载运量区间依次为 [0,10)、[10,20)、[20,40] FEU；末档包含物理容量上限。
	# 元组项为 (下界, 上界, 固定费率元/FEU, 变动费率元/(FEU·公里))。
	rail_tariff_tiers: tuple = (
		(0.0, 10.0, 680.0, 2.754),
		(10.0, 20.0, 714.0, 2.203),
		(20.0, 40.0, 748.0, 1.928),
	)

	c_transfer_road_rail: float = 500.0  # 每次公铁或铁公转运行为的固定费用（元/次）
	c_transfer_rail_rail: float = 200.0  # 每次铁铁转运行为的固定费用（元/次）
	c_unmet_penalty: float = 100000.0  # 假设的未满足需求单位惩罚费（元/FEU）

	alpha_R: float = 0.9  # 风险可靠性阈值
	alpha_T: float = 0.8  # OD 任务的默认时间可靠性阈值
	road_accident_belief_bounds: tuple = (0.020, 0.080)
	rail_accident_belief_bounds: tuple = (0.002, 0.015)
	transfer_accident_belief_rail_rail_bounds: tuple = (0.003, 0.012)
	transfer_accident_belief_intermodal_bounds: tuple = (0.008, 0.030)
	# 归一化的逐小时事故信度乘数（日均值为 1）：
	# 23:00–07:00 为 5/6，07:00–11:00 和 17:00–21:00 为 7/6，其余时段为 1。
	hourly_accident_belief_multipliers: tuple = (
		5.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0,
		5.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0,
		7.0 / 6.0, 7.0 / 6.0, 7.0 / 6.0, 7.0 / 6.0,
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		7.0 / 6.0, 7.0 / 6.0, 7.0 / 6.0, 7.0 / 6.0,
		1.0, 1.0,
		5.0 / 6.0,
	)

	# 公路驾驶员休息规则；铁路运输不受这两个参数约束
	driver_fatigue_limit_hours: float = 4.0  # 公路车辆最长连续驾驶时间（小时）
	driver_rest_hours: float = 20.0 / 60.0  # 每次强制休息的最短时间（小时，即 20 分钟）

	road_speed: float = 80.0  # 公路行驶速度（公里/小时）
	rail_speed: float = 50.0  # 普快班列行驶速度（公里/小时）
	rail_intermediate_stop_hours: float = 2.0  # 每条铁路弧计入的中间停站时间（小时），只生成时刻表
	# 铁路行程时间的 Zigzag 不确定比例：（下界、中值、上界）
	rail_fuzzy_time_ratio: tuple = (0.9, 1.0, 1.3)
	# 公路行程时间的 Zigzag 不确定比例：（下界、中值、上界）
	road_fuzzy_time_ratio: tuple = (0.9, 1.0, 1.5)
	transfer_time_intermodal_hours: float = 8.0
	transfer_time_rail_rail_hours: float = 4.0
	pricing_overlap_ratio: float = 0.75  # 双向定价中正向和反向标签搜索区间各占统一期限的比例

	# ============================================================
	# 时变风险与求解模式开关
	# ============================================================
	enable_time_varying_risk: bool = True  # 是否用交通和疲劳时变系数修正事故信度
	lp_only: bool = False  # 是否仅求根节点列生成的线性松弛解
	skip_risk_constraint: bool = False  # 是否在定价中跳过路径风险可靠性约束，仅用于基准实验
	solver_name: str = "gurobi"  # 优化求解器名称，可选 gurobi 或 pulp

	# ============================================================
	# 求解器参数
	# ============================================================
	time_limit: int = 3600  # 求解时间上限（秒）

	# ============================================================
	# 分支定价参数
	# ============================================================
	bap_max_iterations: int = 1000  # 根节点列生成允许执行的最大迭代次数
	bap_pricing_columns_limit: int = 999  # 每轮定价最多返回的负检验数列数
	initial_columns_per_od_service_day: int = 2  # 每个 OD 在每个服务日保留的初始候选列数
	bap_ng_route_size: int = 15  # 路径松弛中每个节点的邻域大小
	bap_stabilization: bool = True  # 是否启用对偶变量稳定化
	stabilization_mu: float = 0.5  # 当前对偶解在平滑更新中的权重
	max_labels_exact: int = 500  # 精确定价中每个节点保留的最大标签数
	max_labels_heuristic: int = 50  # 启发式定价中每个节点保留的最大标签数
	total_labels_limit: int = 500  # 单次定价在全部节点上保留的标签总数上限

	# ============================================================
	# macOS 下使用 PuLP 时可调用的 CBC 求解器路径；路径不存在时使用 PuLP 默认求解器
	cbc_path: str = "/opt/homebrew/Caskroom/miniforge/base/envs/rhmvsp/bin/cbc"


	# ============================================================
	# 数值计算常量
	# ============================================================
	tolerance: float = 1e-6  # 判断变量取值、检验数和整数性的数值容差
	big_m: float = 1e12  # 虚拟列惩罚和逻辑约束使用的充分大常数

	seed: int = 7  # 数据生成与抽样实验使用的随机种子
	data_version: str = INSTANCE_DATA_VERSION  # 实例生成规则版本
	gurobi_threads: int = 1  # 单线程确保等价最优解的选择可复现

	def transfer_unit_cost(self, previous_mode: int, next_mode: int) -> float:
		"""返回相邻运输弧之间每发生一次转运行为的固定费用（元/次）。"""
		if previous_mode == 2 and next_mode == 2:
			return self.c_transfer_rail_rail
		return self.c_transfer_road_rail

	def transfer_time_hours(self, previous_mode: int, next_mode: int) -> float:
		"""返回相邻运输弧之间的固定转运时间（小时）。"""
		if previous_mode == 2 and next_mode == 2:
			return self.transfer_time_rail_rail_hours
		return self.transfer_time_intermodal_hours

	def road_loaded_cost(self, distance: float, load_feu: float = 1.0) -> float:
		"""返回装载公路运输成本（元）。"""
		return self.hazmat_cost_coefficient * (self.c_road_fixed + self.c_road * distance) * load_feu

	def rail_unit_cost(self, distance: float, tier_index: int) -> float:
		"""返回给定铁路运价档次的单位装载成本（元/FEU）。"""
		_, _, fixed_rate, variable_rate = self.rail_tariff_tiers[tier_index]
		return self.hazmat_cost_coefficient * (fixed_rate + variable_rate * distance)

	def to_dict(self) -> dict:
		"""转为字典，并将主要模型参数改成便于识别的完整键名。"""
		data = asdict(self)

		rev_map = {
			"T_max": "planning_horizon_hours_T_max",
			"q_road": "vehicle_capacity_road_feu_q_road",
			"q_rail": "service_capacity_rail_feu_q_rail",
			"c_road_fixed": "fixed_transport_cost_road_per_feu",
			"c_road": "unit_transport_cost_road_c_road",
			"c_transfer_road_rail": "fixed_transfer_cost_road_rail_per_occurrence",
			"c_transfer_rail_rail": "fixed_transfer_cost_rail_rail_per_occurrence",
			"alpha_R": "risk_reliability_threshold_alpha_R",
			"alpha_T": "time_reliability_threshold_alpha_T",
		}
		output_data = {}
		for k, v in data.items():
			target_k = rev_map.get(k, k)
			output_data[target_k] = v
		return output_data

	def save_to_json(self, filepath: str):
		"""将模型配置和绘图配置保存到同一个 JSON 文件。"""
		
		# 收集绘图配置类中公开且不可调用的类属性
		visual_config = {}
		for k, v in PlotProperties.__dict__.items():
			if not k.startswith("_") and not callable(v) and not isinstance(v, classmethod):
				visual_config[k] = v
		
		out = {
			"model_config": asdict(self),
			"visual_config": visual_config
		}
		with open(filepath, "w") as f:
			json.dump(out, f, indent=4)

from typing import Dict, List


class ColorPalette:
	"""用于高分辨率学术图形的莫兰迪配色。"""

	# 中性色
	DEFAULT_BLACK = "#000000"
	DEFAULT_GRAY = "#999999"
	DEFAULT_WHITE = "#FFFFFF"

	# 五种主要莫兰迪色
	TERRACOTTA = "#C26B48"  # 陶土色，用于危险品优先路径或应急元素
	CELADON = "#94C1B7"  # 青瓷色，用于主要多式联运路径
	SAGE_GREEN = "#ABBCAA"  # 鼠尾草绿，用于普通辐射节点或第三条路径
	WARM_SAND = "#DDBF91"  # 暖沙色，用于多式联运转运节点
	PALE_LIME = "#DDE4B6"  # 淡青柠色，用于辅助路径或缓冲元素

	# 敏感性分析双 Y 轴组合图专用色（不属于 docs 中的 colour set）
	SENSITIVITY_RISK_LINE = TERRACOTTA
	SENSITIVITY_TIME_LINE = CELADON
	SENSITIVITY_COST_MARKER = "#7F8582"  # 成本棒棒糖端点：石墨灰
	SENSITIVITY_COST_STEM = "#C9CDCA"  # 成本棒棒糖竖线：浅灰
	SENSITIVITY_COST_AXIS = "#555555"  # 成本坐标轴与刻度：深灰
	SENSITIVITY_GRID = "#D9D9D9"  # 组合图水平网格：浅灰
	SENSITIVITY_COST = TERRACOTTA  # in-sample gap: cost family
	SENSITIVITY_RISK = CELADON  # in-sample gap: risk family
	SENSITIVITY_TIME = "#7F8582"  # in-sample gap: neutral time family

	# 已调度运输任务循环使用的颜色
	TASK_LOOP: List[str] = [
		TERRACOTTA,
		CELADON,
		SAGE_GREEN,
		WARM_SAND,
		PALE_LIME,
	]

	# 默认中性色列表
	DEFAULT_COLOR: List[Dict[str, str]] = [
		{"BLACK": DEFAULT_BLACK},
		{"GRAY": DEFAULT_GRAY},
		{"WHITE": DEFAULT_WHITE},
	]

	# 网络节点与运输基础设施的颜色
	NETWORK_TOPOLOGY: List[Dict[str, str]] = [
		{"HUB": WARM_SAND},
		{"NON_HUB": SAGE_GREEN},
		{"EMERGENCY": TERRACOTTA},
		{"ROAD": DEFAULT_GRAY},
		{"RAILWAY": DEFAULT_BLACK},
	]


def get_color_by_key(config_list: List[Dict[str, str]], key: str) -> str:
	"""按键名从颜色配置列表中取出颜色值。"""
	for item in config_list:
		if key in item:
			return item[key]
	raise KeyError(f"Color key '{key}' not found in configuration list.")


class PlotProperties:
	"""统一的绘图参数和样式常量。"""

	# 字体
	FONT_FAMILY = "Times New Roman"
	FONT_SIZE_TITLE = 16
	FONT_SIZE_AXIS_LABEL = 14
	FONT_SIZE_LEGEND = 11
	FONT_SIZE_NODE_ID = 14
	FONT_SIZE_VEH_LABEL = 14
	FONT_SIZE_BASE = 12
	FONT_SIZE_TICK = 11
	FONT_COLOR_NODE_ID = "#000000"
	FONT_WEIGHT_BOLD = "bold"

	# 画布
	FIG_SIZE = (12, 10)
	DPI = 600
	PAD_INCHES = 0.1

	# 标记、节点和图例
	NODE_SHAPE_HUB = "s"  # 转运节点使用方形
	NODE_SHAPE_SPOKE = "o"  # 普通节点使用圆形
	NODE_SIZE_HUB = 500
	NODE_SIZE_SPOKE = 400
	NODE_LINE_WIDTH = 1.5
	MARKER_EDGE_COLOR = "#333333"
	LEGEND_MARKER_SIZE_HUB = 13
	LEGEND_MARKER_SIZE_SPOKE = 11
	LEGEND_MARKER_EDGE_WIDTH_HUB = 1.5
	LEGEND_MARKER_EDGE_WIDTH_SPOKE = 1.2
	LEGEND_MARKER_FACE_WHITE = "#FFFFFF"

	# 网络边、路径和网格
	LINE_WIDTH_ROAD = 2.5
	LINE_WIDTH_RAIL = 3.5
	LINE_WIDTH_ROUTE = 3.5
	SPINE_LINE_WIDTH = 1.2
	GRID_LINE_STYLE = "--"
	GRID_ALPHA = 0.4
	SENSITIVITY_COST_STEM_ALPHA = 0.75
	SENSITIVITY_COST_STEM_WIDTH = 1.5
	SENSITIVITY_COST_MARKER_SIZE = 42
	SENSITIVITY_RISK_MARKER = "o"
	SENSITIVITY_TIME_MARKER = "s"
	SENSITIVITY_RISK_LINE_STYLE = "-"
	SENSITIVITY_TIME_LINE_STYLE = "--"
	SENSITIVITY_RELIABILITY_MARKER_SIZE = 5
	SENSITIVITY_RELIABILITY_LINE_WIDTH = 1.8
	SENSITIVITY_Y_TARGET_INTERVALS = 5
	SENSITIVITY_GRID_ALPHA = 0.55
	SENSITIVITY_GRID_LINE_WIDTH = 0.6
	TICK_LENGTH = 5
	TICK_WIDTH = 1.0

	ALPHA_ROAD = 0.5
	ALPHA_RAIL = 0.5
	ALPHA_ROUTE = 0.95

	STYLE_ROAD = "-"
	STYLE_RAIL = "--"
	DASH_PATTERN_RAIL = (0, (5, 2))

	ARROW_STYLE = "-|>"
	ARROW_SIZE = 20

	# 布局偏移和曲率
	CURVATURE_BASE_RAD = 0.10
	CURVATURE_STEP_RAD = 0.15

	# 图形标签
	LABEL_LEGEND_HUB = "Hub"
	LABEL_LEGEND_SPOKE = "Spoke"
	LABEL_LEGEND_ROAD = "Road"
	LABEL_LEGEND_RAIL = "Railway"
	LABEL_LEGEND_TASK_PREFIX = "Task "
	LABEL_VEH_PREFIX = "Veh "

	# 透明文本框样式
	TRANSPARENT_BBOX = dict(boxstyle="round,pad=0.1", fc="none", ec="none")
