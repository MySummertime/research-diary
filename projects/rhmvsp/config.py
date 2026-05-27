"""RHMVSP Configuration - All model parameters and constants."""

from __future__ import annotations
import json
from dataclasses import asdict, dataclass


@dataclass
class RHMVSPConfig:
	"""
	Configuration for the Reliable Hazardous Materials Multi-modal VSP.

	Based on the paper's parameter definitions (§3.1.1).
	"""

	# ============================================================
	# Network Parameters
	# ============================================================
	M = {1: "road", 2: "rail"}  # Transport modes

	# Time discretization
	T_max: int = 48  # Planning horizon (hours)

	# Vehicle parameters
	K_max: int | None = None  # Max O-D pairs per vehicle (reuse cycles). None = unlimited.
	q_road: float = 20.0  # Road vehicle capacity (tons)
	# 60 t/wagon: Chinese railway 整车 standard for liquid ammonia (dangerous goods class 6)
	q_rail: float = 600.0  # Rail wagon capacity (tons, 整车 标重)

	# ============================================================
	# Cost Parameters
	# ============================================================
	# Cost parameters calibrated from 12306 official pricing:
	# Route: Ürümqi South → Chengdu (~3,334 km), 整车 (full carload, 60 t)
	c_road: float = 0.5    # Road: ¥/ton·km (market reference, tanker truck)
	c_deploy_road: float = 0.0  # Road vehicle deployment fixed cost (自有车为0)
	c_deploy_rent: float = 1000.0 # Rented vehicle deployment fixed cost (外租车起步价)
	c_rent_per_hour: float = 30.0 # Rented vehicle hourly rate (外租车小时租金)
	c_empty_reposition_per_km: float = 4.0  # Empty truck repositioning (¥/km, diesel+driver+depreciation)
	# ============================================================
	# Railway Freight Two-Part Tariff Pricing (基于《铁路货物运价规则》)
	# Formula per ton = (发到基价 + 运行基价 × 计费里程) × (1 + 危险品加成率) + 建设基金 × 计费里程
	# ============================================================
	c_rail_base_fixed: float = 16.30  # 发到基价 (¥/ton)
	c_rail_base_var: float = 0.0855  # 运行基价 (¥/ton·km)
	c_rail_hazmat_markup: float = 0.50  # 危险品加成率 (50% markup for hazmat)
	c_rail_fund: float = 0.033  # 铁路建设基金 (¥/ton·km)
	
	# c_deploy_rail includes 毒品专用车 reservation premium
	c_deploy_rail: float = 700.0  # Rail wagon deployment fixed cost (¥/wagon)
	# c_transfer = ¥37.75/ton (发站装车 + 到站卸车, hazmat防爆要求)
	c_transfer: float = 37.75  # Unit transfer/handling cost (¥/ton)
	c_hub: float = 1000.0  # Transfer facility setup cost (¥)
	max_cranes: int = 2  # Max concurrent transfer cranes per hub
	max_storage_tons: float = 500.0  # Max hazmat storage capacity per hub (tons)
	c_unmet: float = 0.0  # Unmet demand penalty (set to 0: all demand must be satisfied)
	c_opp_base: float = 500.0 # Base opportunity cost for unfulfilled demand (¥/ton, reduced to decrease total penalty)
	c_holding: float = 10.0    # Inventory holding cost per ton for remaining hazmat stock at origin (¥/ton)
	crane_daily_lifts: int = 200  # Daily lifts per crane
	max_hubs: int | None = None  # Max active transfer hubs (None means unlimited)
	alpha_c: float = 0.85  # Confidence level for fuzzy cost equivalents

	# ============================================================
	# Reliability Parameters (§2.4, §3.1.1)	# ============================================================
	alpha_max: float = 0.05  # Arc accident belief threshold (base risk param)
	# beta_od derived: beta_od = 1 - alpha_max
	# beta_net derived: beta_net = min(beta_od)
	alpha_T: float = 0.8  # Time reliability threshold (alpha_T > 0.5)
	# gamma_sys derived: gamma_sys = min(1 - alpha_max, alpha_T)

	# ============================================================
	# Risk Parameters (§2.2)	# ============================================================

	# ============================================================
	# Fuzzy Variable Parameters (§3.2)	# ============================================================
	road_speed: float = 60.0  # Road speed (km/h)
	rail_speed: float = 80.0  # Rail speed (km/h)
	# Trapezoidal fuzzy time ratios for rail: (a_ratio, b_ratio, c_ratio, d_ratio)
	# Rail time ~ Trap(base*a, base*b, base*c, base*d) with plateau [b,c]
	rail_fuzzy_time_ratio: tuple = (0.9, 1.0, 1.0, 1.3)
	# Bidirectional label correction overlap ratio (§4.5)
	pricing_overlap_ratio: float = 0.75  # each direction covers this fraction of time span

	empty_reposition_buffer_hours: float = 1.0  # Safety buffer (cleaning, inspection)

	# ============================================================
	# Demand and Cost Uncertainty Parameters (Wang 2025 Inspired)
	# ============================================================
	delta_D: float = 0.2  # Demand fluctuation factor
	delta_C: float = 0.3  # Penalty cost fluctuation factor
	delta_T: float = 0.8  # Time fluctuation factor (controls fuzzy spread of travel/transfer times)
	beta_D: float = 0.8   # Service level threshold for Demand Chance Constraint
	fuzzy_cost_spread: tuple = (0.9, 1.0, 1.2)  # Triangular fuzzy spread for unit transport costs

	# Solver settings
	solver_name: str = "gurobi"
	solver_backend: str = "gurobi"  # Default solver backend

	# ============================================================
	# Solver Parameters
	# ============================================================
	time_limit: int = 3600  # Solver time limit (seconds)

	# ============================================================
	# Branch-and-Price Parameters (§4.5)
	# ============================================================
	bap_max_iterations: int = 1000
	bap_pricing_columns_limit: int = 10  # Max columns returned per pricing OD
	bap_ng_route_size: int = 15  # ng-route neighborhood size
	bap_stabilization: bool = True  # Interior point stabilization
	stabilization_mu: float = 0.5  # Smoothing factor for duals
	max_labels_exact: int = 100  # Max labels to keep in exact pricing (reduced for speed)
	max_labels_heuristic: int = 20  # Max labels to keep in heuristic pricing

	# ============================================================
	# Soft Time Window Design
	# ============================================================
	# L_od = expected_arrival + L_od_buffer_hours.
	# A small buffer (e.g. 0.5 h) creates a meaningful but proportional
	# delay penalty that trades off against unmet/opportunity costs.
	L_od_buffer_hours: float = 1.5

	# Vehicle limit = num_vehicles × vehicle_limit_multiplier
	# Default overridden by instance-specific adaptive value in data_generator
	vehicle_limit_multiplier: float = 3.0

	# ============================================================
	# Paths & System
	# ============================================================
	# Default path for Conda-installed CBC on Mac
	cbc_path: str = "/opt/homebrew/Caskroom/miniforge/base/envs/rhmvsp/bin/cbc"

	# ============================================================
	# Numerical Constants
	# ============================================================
	tolerance: float = 1e-6
	big_m: float = 1e12

	# Random seed
	# ============================================================
	seed: int = 7

	# ============================================================
	# Numerical Integration Constants (Gauss-Legendre 5-point)
	# ============================================================
	gauss_legendre_nodes_5: tuple = (0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992)
	gauss_legendre_weights_5: tuple = (0.11846344, 0.23931434, 0.28444444, 0.23931434, 0.11846344)
	# ============================================================
	@property
	def beta_od(self) -> float:
		"""Path risk reliability threshold (derived from alpha_max)."""
		return 1.0 - self.alpha_max

	@property
	def beta_net(self) -> float:
		"""Network risk reliability threshold."""
		return self.beta_od

	@property
	def gamma_sys(self) -> float:
		"""Comprehensive reliability threshold."""
		return min(1.0 - self.alpha_max, self.alpha_T)

	@property
	def q_bar(self) -> float:
		"""Minimum vehicle capacity across all arc-mode combinations."""
		return min(self.q_road, self.q_rail)

	def time_coeff_upper(self, alpha_T: float | None = None):
		"""Coefficient for time window upper bound deterministic equivalent (Eq.12).

		For triangular fuzzy (a,b,c): (2*alpha-1)*c + 2*(1-alpha)*b
		For trapezoidal fuzzy (a,b,c,d): (2*alpha-1)*d + 2*(1-alpha)*c
		"""
		a = alpha_T if alpha_T is not None else self.alpha_T
		return {
			"tri_c": (2 * a - 1),
			"tri_b": 2 * (1 - a),
			"trap_d": (2 * a - 1),
			"trap_c": 2 * (1 - a),
		}

	def to_dict(self) -> dict:
		"""Convert config to dictionary including derived properties and full descriptive keys."""
		data = asdict(self)
		data["beta_od"] = self.beta_od
		data["beta_net"] = self.beta_net
		data["gamma_sys"] = self.gamma_sys
		data["q_bar"] = self.q_bar

		rev_map = {
			"T_max": "planning_horizon_hours_T_max",
			"dt_road": "time_granularity_road_hours_dt_road",
			"dt_rail": "time_granularity_rail_hours_dt_rail",
			"K_max": "max_reuse_cycles_per_vehicle_K_max",
			"q_road": "vehicle_capacity_road_tons_q_road",
			"q_rail": "wagon_capacity_rail_tons_q_rail",
			"c_road": "unit_transport_cost_road_c_road",
			"c_rail_base_fixed": "unit_transport_cost_rail_c_rail_base_fixed",
			"c_rail_base_var": "unit_transport_cost_rail_c_rail_base_var",
			"c_rail_hazmat_markup": "unit_transport_cost_rail_hazmat_markup",
			"c_rail_fund": "unit_transport_cost_rail_fund",
			"c_deploy_road": "fixed_dispatch_cost_road_d_road",
			"c_deploy_rail": "fixed_dispatch_cost_rail_d_rail",
			"c_transfer": "unit_transfer_cost_kappa_transfer",
			"c_hub": "fixed_transfer_facility_cost_H_transfer",
			"c_unmet": "unmet_demand_penalty_c_unmet",
			"alpha_max": "risk_threshold_alpha_max",
			"alpha_T": "time_reliability_threshold_alpha_T",
			"beta_D": "demand_chance_constraint_threshold_beta_D",
			"fuzzy_cost_spread": "fuzzy_multiplier_transport_cost_triangular",
		}
		output_data = {}
		for k, v in data.items():
			target_k = rev_map.get(k, k)
			output_data[target_k] = v
		return output_data

	def save_to_json(self, filepath: str):
		"""Save both model and visual configuration to a unified config.json file."""
		import json
		
		# Collect visual config from class attributes
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

# -*- coding: utf-8 -*-
"""
[Visual Style Configuration] Unified Visual Style Management Module for RHMVSP.
Centralizes color palettes, markers, linewidths, and fonts to ensure premium academic publication standards.
Based on the Twilight Gradation color system with high visibility.
"""

from typing import Dict, List


class ColorPalette:
	"""Refined Academic Morandi Earth & Sage Color Palette for High-Resolution Plotting."""

	# Base neutrals
	DEFAULT_BLACK = "#000000"
	DEFAULT_GRAY = "#999999"
	DEFAULT_WHITE = "#FFFFFF"

	# 5 Premium Morandi Colors requested by USER
	TERRACOTTA = "#C26B48"  # Warm Terracotta (Hazmat priority route / emergency)
	CELADON = "#94C1B7"  # Celadon Green (Primary multimodal green route)
	SAGE_GREEN = "#ABBCAA"  # Sage Green (Standard spoke nodes / third route)
	WARM_SAND = "#DDBF91"  # Warm Sand (Multimodal transfer hubs)
	PALE_LIME = "#DDE4B6"  # Pale Lime (Auxiliary routes / buffer)

	# Network tasks loop for active scheduled routes
	TASK_LOOP: List[str] = [
		TERRACOTTA,
		CELADON,
		SAGE_GREEN,
		WARM_SAND,
		PALE_LIME,
	]

	# Default neutrals list
	DEFAULT_COLOR: List[Dict[str, str]] = [
		{"BLACK": DEFAULT_BLACK},
		{"GRAY": DEFAULT_GRAY},
		{"WHITE": DEFAULT_WHITE},
	]

	# Network topology colors for background nodes and infrastructure
	NETWORK_TOPOLOGY: List[Dict[str, str]] = [
		{"HUB": WARM_SAND},
		{"NON_HUB": SAGE_GREEN},
		{"EMERGENCY": TERRACOTTA},
		{"ROAD": DEFAULT_GRAY},
		{"RAILWAY": DEFAULT_BLACK},
	]


def get_color_by_key(config_list: List[Dict[str, str]], key: str) -> str:
	"""Extract color string by key from a list of dict configurations."""
	for item in config_list:
		if key in item:
			return item[key]
	raise KeyError(f"Color key '{key}' not found in configuration list.")


class PlotProperties:
	"""Unified plotting parameters and styling constants."""

	# Fonts
	FONT_FAMILY = "Times New Roman"
	FONT_SIZE_TITLE = 16
	FONT_SIZE_AXIS_LABEL = 14
	FONT_SIZE_LEGEND = 11
	FONT_SIZE_NODE_ID = 14
	FONT_SIZE_NODE_NAME = 10
	FONT_SIZE_VEH_LABEL = 14
	FONT_SIZE_BASE = 12
	FONT_SIZE_TICK = 11
	FONT_COLOR_NODE_ID = "#000000"
	FONT_WEIGHT_BOLD = "bold"
	FONT_WEIGHT_NORMAL = "normal"

	# Figure
	FIG_SIZE = (12, 10)
	DPI = 600
	PAD_INCHES = 0.1

	# Markers & Nodes & Legend
	NODE_SHAPE = "o"
	NODE_SIZE_HUB = 900
	NODE_SIZE_SPOKE = 800
	NODE_LINE_WIDTH = 1.5
	MARKER_EDGE_COLOR = "#333333"
	LEGEND_MARKER_SIZE_HUB = 13
	LEGEND_MARKER_SIZE_SPOKE = 11
	LEGEND_MARKER_EDGE_WIDTH_HUB = 1.5
	LEGEND_MARKER_EDGE_WIDTH_SPOKE = 1.2
	LEGEND_MARKER_FACE_WHITE = "#FFFFFF"

	# Edges & Routes & Grids
	LINE_WIDTH_ROAD = 2.5
	LINE_WIDTH_RAIL = 3.5
	LINE_WIDTH_ROUTE = 3.5
	SPINE_LINE_WIDTH = 1.2
	GRID_LINE_STYLE = "--"
	GRID_ALPHA = 0.4
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

	# Layout offsets & Curvatures
	CURVATURE_OFFSET_RAD = 0.20
	NODE_LABEL_OFFSET = 3.2
	CURVATURE_BASE_RAD = 0.10
	CURVATURE_STEP_RAD = 0.15

	# Labels & Descriptions
	LABEL_X_AXIS = "Physical Coordinate X (km)"
	LABEL_Y_AXIS = "Physical Coordinate Y (km)"
	LABEL_LEGEND_HUB = "Hub"
	LABEL_LEGEND_SPOKE = "Spoke"
	LABEL_LEGEND_ROAD = "Road"
	LABEL_LEGEND_RAIL = "Railway"
	LABEL_LEGEND_TASK_PREFIX = "Task "
	LABEL_VEH_PREFIX = "Veh "

	# Transparent text box styles
	TRANSPARENT_BBOX = dict(boxstyle="round,pad=0.1", fc="none", ec="none")


