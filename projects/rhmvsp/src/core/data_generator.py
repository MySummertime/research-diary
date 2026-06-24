"""Instance Data Generator (§3.1, §4.4.2)."""


import math
import numpy as np
import networkx as nx

from config import RHMVSPConfig
from .uncertainty import UncertaintyEngine


def float_to_day_time(time_hours: float) -> str:
	"""Convert continuous float hours into Day 1/2 HH:MM real time format."""
	day = int(time_hours // 24) + 1
	rem_hours = time_hours % 24
	hours = int(rem_hours)
	minutes = int(round((rem_hours - hours) * 60))
	if minutes == 60:
		hours += 1
		minutes = 0
	if hours == 24:
		day += 1
		hours = 0
	return f"Day {day} {hours:02d}:{minutes:02d}"


class InstanceGenerator:
	"""
	Generate synthetic test instances for RHMVSP.

	Creates road-rail multimodal networks with:
	- Hub nodes (rail stations) forming an inner core.
	- Spoke nodes forming an outer ring.
	- Unrestricted road-rail-road hops via distance-based arc generation.
	"""

	def __init__(self, config: RHMVSPConfig, seed: int = 42):
		self.config = config
		self.rng = np.random.RandomState(seed)
		self.engine = UncertaintyEngine(config)

	def generate(self, size: str = "small") -> dict:
		"""Generate an instance of given size."""
		sizes = {
			"small":  {"nodes": 25,  "hubs": 5,  "od_pairs": 5,  "vehicles": 30, "T_max_init": 24},
			"medium": {"nodes": 60,  "hubs": 10, "od_pairs": 15, "vehicles": 90, "T_max_init": 36},
			"large":  {"nodes": 120, "hubs": 20, "od_pairs": 30, "vehicles": 150, "T_max_init": 48},
		}
		params = sizes[size]
		self.config.T_max = max(self.config.T_max, params["T_max_init"])
		return self._generate_instance(
			name=f"rhmvsp_{size}",
			num_nodes=params["nodes"],
			num_hubs=params["hubs"],
			num_od_pairs=params["od_pairs"],
			num_vehicles=params["vehicles"],
		)

	def _generate_instance(
		self, name: str, num_nodes: int, num_hubs: int = None, num_od_pairs: int = 5, num_vehicles: int = 50
	) -> dict:
		"""Generate a rigorously connected Hub-and-Spoke multimodal instance."""
		rng = self.rng

		# Dynamic timetable: scale services with network size
		num_services = max(8, num_nodes // 3)
		interval = 48.0 / num_services
		dynamic_schedule_hours = [round(interval / 2.0 + i * interval, 1) for i in range(num_services)]

		# 1. Partition Nodes into Hubs (Inner) and Spokes (Outer)
		if num_hubs is None:
			num_hubs = max(3, int(np.floor(0.3 * num_nodes)))
		hub_ids = list(range(1, num_hubs + 1))
		spoke_ids = list(range(num_hubs + 1, num_nodes + 1))

		# 2. Constructive Spatial Generation
		coords = {}
		coords[hub_ids[0]] = (0.0, 0.0)
		for h in hub_ids[1:]:
			for _ in range(100):
				theta = rng.uniform(0, 2 * np.pi)
				r = rng.uniform(450, 550)
				x, y = r * np.cos(theta), r * np.sin(theta)
				too_close = False
				for eh in coords:
					if np.hypot(x - coords[eh][0], y - coords[eh][1]) < 100:
						too_close = True
						break
				if not too_close:
					coords[h] = (x, y)
					break
			else:
				coords[h] = (x, y)

		for s in spoke_ids:
			ref_hub = rng.choice(hub_ids[1:]) if len(hub_ids) > 1 else hub_ids[0]
			ref_x, ref_y = coords[ref_hub]
			ref_angle = np.arctan2(ref_y, ref_x)
			for _ in range(100):
				theta = ref_angle + rng.uniform(-np.pi/3, np.pi/3)
				r = rng.uniform(150, 350)
				x = ref_x + r * np.cos(theta)
				y = ref_y + r * np.sin(theta)
				too_close = False
				for en, (ex, ey) in coords.items():
					if np.hypot(x - ex, y - ey) < 60:
						too_close = True
						break
				if not too_close:
					coords[s] = (x, y)
					break
			else:
				coords[s] = (x, y)

		# Build graphs
		G_rail = nx.Graph()
		G_road = nx.Graph()
		for i in range(len(hub_ids)):
			for j in range(i + 1, len(hub_ids)):
				u, v = hub_ids[i], hub_ids[j]
				d = np.hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1])
				if 400 <= d <= 1200:
					G_rail.add_edge(u, v, weight=d)
		all_nodes = hub_ids + spoke_ids
		for i in range(len(all_nodes)):
			for j in range(i + 1, len(all_nodes)):
				u, v = all_nodes[i], all_nodes[j]
				d = np.hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1])
				if 100 <= d <= 400:
					G_road.add_edge(u, v, weight=d)

		# 3. Assemble Network Attributes
		nodes = [{"id": i, "x": round(coords[i][0], 2), "y": round(coords[i][1], 2)} for i in all_nodes]
		arcs = []
		railway_timetable = []
		arc_id = 1
		train_id_counter = 101

		for u, v in G_rail.edges():
			dist = G_rail[u][v]['weight']
			base_time = dist / self.config.rail_speed
			unit_cost_rail = (self.config.c_rail_base_fixed + self.config.c_rail_base_var * dist) * (1 + self.config.c_rail_hazmat_markup) + self.config.c_rail_fund * dist
			alpha_rail_val = float(round(rng.uniform(0.0005, 0.003), 5))
			rail_ratios = getattr(self.config, 'rail_fuzzy_time_ratio', (0.9, 1.0, 1.0, 1.3))
			for src, dst in [(u, v), (v, u)]:
				arcs.append({
					"id": arc_id, "i": src, "j": dst, "mode": 2, "distance": round(float(dist), 2),
					"time_a": round(float(base_time * rail_ratios[0]), 2),
					"time_b": round(float(base_time * rail_ratios[1]), 2),
					"time_c": round(float(base_time * rail_ratios[2]), 2),
					"time_d": round(float(base_time * rail_ratios[3]), 2),
					"alpha": alpha_rail_val,
					"e_window": 0.0, "l_window": float(self.config.T_max),
					"e_window_fmt": float_to_day_time(0.0), "l_window_fmt": float_to_day_time(self.config.T_max),
					"capacity": float(self.config.q_rail),
					"unit_cost": round(float(unit_cost_rail), 2),
					"deploy_cost": float(self.config.c_deploy_rail), "min_units": 1,
				})
				arc_id += 1
			for dep_h in dynamic_schedule_hours:
				arr_h = round(dep_h + base_time, 2)
				railway_timetable.append({
					"train_id": f"X{train_id_counter}", "origin_hub": u, "destination_hub": v, "departure_time": dep_h,
					"departure_time_fmt": float_to_day_time(dep_h), "arrival_time": arr_h, "arrival_time_fmt": float_to_day_time(arr_h),
					"capacity": float(self.config.q_rail),
				})
				train_id_counter += 1
				railway_timetable.append({
					"train_id": f"X{train_id_counter}", "origin_hub": v, "destination_hub": u, "departure_time": dep_h,
					"departure_time_fmt": float_to_day_time(dep_h), "arrival_time": arr_h, "arrival_time_fmt": float_to_day_time(arr_h),
					"capacity": float(self.config.q_rail),
				})
				train_id_counter += 1

		for u, v in G_road.edges():
			dist = G_road[u][v]['weight']
			base_time = dist / self.config.road_speed
			alpha_road_val = float(round(rng.uniform(0.005, 0.03), 4))
			road_ratios = getattr(self.config, 'road_fuzzy_time_ratio', (0.8, 1.0, 1.0, 1.5))
			for src, dst in [(u, v), (v, u)]:
				arcs.append({
					"id": arc_id, "i": src, "j": dst, "mode": 1, "distance": round(float(dist), 2),
					"time_a": round(float(base_time * road_ratios[0]), 2),
					"time_b": round(float(base_time * road_ratios[1]), 2),
					"time_c": round(float(base_time * road_ratios[2]), 2),
					"time_d": round(float(base_time * road_ratios[3]), 2),
					"alpha": alpha_road_val,
					"e_window": 0.0, "l_window": float(self.config.T_max),
					"e_window_fmt": float_to_day_time(0.0), "l_window_fmt": float_to_day_time(self.config.T_max),
					"capacity": float(self.config.q_road),
					"unit_cost": round(float(self.config.c_road * dist), 2),
					"deploy_cost": float(self.config.c_deploy_road), "min_units": 1,
				})
				arc_id += 1

		# 4. Transfer facilities at Hub nodes
		transfer_nodes = []
		for hub in hub_ids:
			gamma_val = float(round(rng.uniform(0.001, 0.01), 5))
			transfer_nodes.append({
				"node_id": hub, "transfer_time_a": 0.5, "transfer_time_b": 1.0,
				"transfer_time_c": 2.0, "transfer_time_d": 4.0,
				"gamma": gamma_val,
				"unit_transfer_cost": float(self.config.c_transfer),
				"facility_cost": float(self.config.c_hub),
			})

		# 5. O-D Pairs: far apart, unique O and D nodes, different hubs
		G_road_dir = nx.DiGraph()
		for a in arcs:
			if a["mode"] == 1:
				G_road_dir.add_edge(a["i"], a["j"], weight=a["time_b"])

		od_pairs = []
		used_nodes = set()  # All O and D nodes must be globally unique

		def _angle_from_center(node_id):
			x, y = coords[node_id]
			return math.atan2(y, x)

		def _euclidean_dist(n1, n2):
			return np.hypot(coords[n1][0] - coords[n2][0], coords[n1][1] - coords[n2][1])

		MIN_OD_ANGLE = math.pi / 4.0  # 45 degrees minimum separation
		MIN_OD_EUCLIDEAN = 400.0

		for od_idx in range(num_od_pairs):
			found = False
			o, d = None, None
			t_road_1, t_road_2, t_rail = 0.0, 0.0, 0.0
			candidates = list(spoke_ids)
			rng.shuffle(candidates)
			for o_cand in candidates:
				if o_cand in used_nodes:
					continue
				for d_cand in candidates:
					if d_cand == o_cand or d_cand in used_nodes:
						continue
					# Directional constraint: O must be west of D (x_origin < x_destination)
					if coords[o_cand][0] >= coords[d_cand][0]:
						continue
					# Also require vertical separation to ensure multi-directional diversity
					y_diff = abs(coords[o_cand][1] - coords[d_cand][1])
					x_diff = abs(coords[d_cand][0] - coords[o_cand][0])
					if _euclidean_dist(o_cand, d_cand) < MIN_OD_EUCLIDEAN:
						continue
					try:
						o_dists = nx.single_source_dijkstra_path_length(G_road_dir, o_cand, weight='weight')
						o_hub_dists = {h: o_dists[h] for h in hub_ids if h in o_dists}
						if not o_hub_dists:
							continue
						h_o = min(o_hub_dists, key=o_hub_dists.get)
						_t_road_1 = o_hub_dists[h_o]
						d_hub_dists = {}
						for h in hub_ids:
							if h not in G_road_dir or d_cand not in G_road_dir:
								continue
							try:
								d_hub_dists[h] = nx.shortest_path_length(G_road_dir, h, d_cand, weight='weight')
							except nx.NetworkXNoPath:
								pass
						if not d_hub_dists:
							continue
						h_d = min(d_hub_dists, key=d_hub_dists.get)
						_t_road_2 = d_hub_dists[h_d]
						if h_o == h_d:
							if len(d_hub_dists) > 1:
								h_d = [h for h in d_hub_dists if h != h_o][0]
								_t_road_2 = d_hub_dists[h_d]
							else:
								continue
						if h_o not in G_rail or h_d not in G_rail:
							continue
						_t_rail = nx.shortest_path_length(G_rail, h_o, h_d, weight='weight') / self.config.rail_speed
						o, d = o_cand, d_cand
						t_road_1, t_road_2, t_rail = _t_road_1, _t_road_2, _t_rail
						used_nodes.add(o)
						used_nodes.add(d)
						found = True
						break
					except nx.NetworkXNoPath:
						continue
				if found:
					break

			if not found:
				remaining = [s for s in spoke_ids if s not in used_nodes]
				o = remaining[0] if remaining else rng.choice(spoke_ids)
				while True:
					d = int(rng.choice(spoke_ids))
					if d != o and d not in used_nodes:
						break
				o_dists = nx.single_source_dijkstra_path_length(G_road_dir, o, weight='weight')
				o_hub_dists = {h: o_dists[h] for h in hub_ids if h in o_dists}
				h_o = min(o_hub_dists, key=o_hub_dists.get)
				t_road_1 = o_hub_dists[h_o]
				d_hub_dists = {}
				for h in hub_ids:
					if h not in G_road_dir or d not in G_road_dir:
						continue
					try:
						d_hub_dists[h] = nx.shortest_path_length(G_road_dir, h, d, weight='weight')
					except nx.NetworkXNoPath:
						pass
				h_d = min(d_hub_dists, key=d_hub_dists.get)
				t_road_2 = d_hub_dists[h_d]
				if h_o == h_d and len(d_hub_dists) > 1:
					h_d = [h for h in d_hub_dists if h != h_o][0]
					t_road_2 = d_hub_dists[h_d]
				t_rail = nx.shortest_path_length(G_rail, h_o, h_d, weight='weight') / self.config.rail_speed
				used_nodes.add(o)
				used_nodes.add(d)

			multiplier = int(rng.uniform(1, 10))
			demand = float(multiplier * self.config.q_road)
			time_window_start = float(round(rng.uniform(0, 24), 1))
			transfer_time = 1.0
			wait_time_est = interval
			t_expected = time_window_start + t_road_1 + transfer_time + wait_time_est + t_rail + transfer_time + t_road_2
			time_window_end = float(round(t_expected + self.config.prep_time_hours + self.config.dispatch_buffer_hours + self.config.time_window_end_buffer_hours, 1))
			self.config.T_max = max(self.config.T_max, time_window_end + 12.0)

			od_pairs.append({
				"id": od_idx + 1,
				"origin": o, "destination": d, "demand": round(demand, 1),
				"time_window_start": round(time_window_start, 1), "time_window_end": round(time_window_end, 1),
				"time_window_start_fmt": float_to_day_time(time_window_start), "time_window_end_fmt": float_to_day_time(time_window_end),
				"alpha_T_od": float(self.config.alpha_T),
			})

		return {
			"name": name,
			"railway_timetable": railway_timetable,
			"network": {
				"nodes": nodes, "arcs": arcs, "transfer_nodes": transfer_nodes,
				"num_nodes": num_nodes, "num_arcs": len(arcs),
			},
			"od_pairs": od_pairs,
			"num_vehicles": num_vehicles,
			"config": {
				"T_max": self.config.T_max,
				"alpha_max": self.config.alpha_max,
				"alpha_T": self.config.alpha_T,
				"timetable_services_per_direction": num_services,
			},
		}