"""Instance Data Generator (§3.1, §4.4.2)."""


import numpy as np

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
	- Hub nodes (rail stations) connected by rail arcs
	- Road arcs connecting nearby hubs and providing last-mile access
	- Transfer nodes at rail stations
	- O-D pairs with hazmat demand
	"""

	def __init__(self, config: RHMVSPConfig, seed: int = 42):
		self.config = config
		self.rng = np.random.RandomState(seed)
		self.engine = UncertaintyEngine(config)

	def generate(self, size: str = "small") -> dict:
		"""Generate an instance of given size."""
		sizes = {
			"small": {"nodes": 20, "od_pairs": 5, "vehicles": 15},
			"medium": {"nodes": 50, "od_pairs": 15, "vehicles": 40},
			"large": {"nodes": 100, "od_pairs": 30, "vehicles": 80},
		}
		params = sizes[size]
		return self._generate_instance(
			name=f"rhmvsp_{size}",
			num_nodes=params["nodes"],
			num_od_pairs=params["od_pairs"],
			num_vehicles=params["vehicles"],
		)

	def _generate_instance(
		self, name: str, num_nodes: int, num_od_pairs: int, num_vehicles: int
	) -> dict:
		"""Generate a rigorous Hub-and-Spoke multimodal instance."""
		rng = self.rng

		# Dynamic timetable: scale services with network size, minimum every 6h
		# small(20)=8 services, medium(50)=12, large(100)=16
		num_services = max(8, num_nodes // 3)
		interval = 48.0 / num_services
		dynamic_schedule_hours = [round(interval / 2.0 + i * interval, 1) for i in range(num_services)]

		# 1. Partition Nodes into Central Hubs (Rail Stations) and Peripheral Spokes
		num_hubs = max(3, int(np.floor(0.4 * num_nodes)))

		hub_ids = list(range(1, num_hubs + 1))
		spoke_ids = list(range(num_hubs + 1, num_nodes + 1))

		# Split spokes into Origins (West/North) and Destinations (East/South)
		mid_spoke = len(spoke_ids) // 2
		origin_spokes = spoke_ids[:mid_spoke]
		dest_spokes = spoke_ids[mid_spoke:]
		if not origin_spokes or not dest_spokes:
			origin_spokes = spoke_ids
			dest_spokes = spoke_ids

		# Split hubs into West Hubs (for origin transfer) and East Hubs (for destination transfer)
		mid_hub = max(1, len(hub_ids) // 2)
		west_hubs = hub_ids[:mid_hub]
		east_hubs = hub_ids[mid_hub:]
		if not west_hubs or not east_hubs:
			west_hubs = hub_ids
			east_hubs = hub_ids

		# 2. Assign precise physical coordinates
		coords = {}
		
		# Scale coordinate span based on network size to simulate larger geographical area.
		# Larger networks occupy proportionally larger coordinate spans to preserve spatial realism.
		scale = float(num_nodes) / 20.0
		base_width = 100 * scale
		base_height = 100 * scale

		# West Hubs: X in [0.3*W, 0.48*W]
		for h in west_hubs:
			x = rng.uniform(0.3 * base_width, 0.48 * base_width)
			y = rng.uniform(0.35 * base_height, 0.65 * base_height)
			coords[h] = (round(x, 2), round(y, 2))

		# East Hubs: X in [0.52*W, 0.70*W]
		for h in east_hubs:
			x = rng.uniform(0.52 * base_width, 0.70 * base_width)
			y = rng.uniform(0.35 * base_height, 0.65 * base_height)
			coords[h] = (round(x, 2), round(y, 2))

		# Origin Spokes on West/North perimeter: X in [0.05*W, 0.25*W]
		for s in origin_spokes:
			x = rng.uniform(0.05 * base_width, 0.25 * base_width)
			y = rng.uniform(0.10 * base_height, 0.90 * base_height)
			coords[s] = (round(x, 2), round(y, 2))

		# Destination Spokes on East/South perimeter: X in [0.75*W, 0.95*W]
		for s in dest_spokes:
			x = rng.uniform(0.75 * base_width, 0.95 * base_width)
			y = rng.uniform(0.10 * base_height, 0.90 * base_height)
			coords[s] = (round(x, 2), round(y, 2))

		nodes = [{"id": i, "x": coords[i][0], "y": coords[i][1]} for i in list(hub_ids + spoke_ids)]

		arcs = []
		railway_timetable = []
		arc_id = 1

		# 3. Create Internal Railway Trunk Network (Mode 2) between Hubs
		# Exactly one physical rail arc per connected hub pair, timetable is decoupled
		train_id_counter = 101
		for i in range(len(hub_ids)):
			for j in range(i + 1, len(hub_ids)):
				u, v = hub_ids[i], hub_ids[j]
				dist = rng.uniform(400, 1500)
				base_time = dist / self.config.rail_speed
				
				# 铁路两部制运价公式：(发到基价 + 运行基价 × 里程) × (1 + 危险品加成) + 建设基金 × 里程
				unit_cost_rail = (self.config.c_rail_base_fixed + self.config.c_rail_base_var * dist) * (1 + self.config.c_rail_hazmat_markup) + self.config.c_rail_fund * dist

				# Unique risk parameters for the undirected rail edge (shared by both directions for realism)
				alpha_rail_val = float(round(rng.uniform(0.0005, 0.003), 5))
				c_conseq_rail_val = float(round(rng.uniform(30.0, 100.0), 2))
				pop_dens_rail_val = float(round(rng.uniform(0.5, 3.0), 2))

				# Single physical rail arc u -> v (trapezoidal fuzzy time, #9)
				rail_ratios = getattr(self.config, 'rail_fuzzy_time_ratio', (0.9, 1.0, 1.0, 1.3))
				rail_arc_u_v = {
					"id": arc_id, "i": u, "j": v, "mode": 2, "distance": round(float(dist), 2),
					"time_a": round(float(base_time * rail_ratios[0]), 2),
					"time_b": round(float(base_time * rail_ratios[1]), 2),
					"time_c": round(float(base_time * rail_ratios[2]), 2),
					"time_d": round(float(base_time * rail_ratios[3]), 2),
					"alpha": alpha_rail_val,
					"C_consequence": c_conseq_rail_val,
					"e_window": 0.0, "l_window": float(self.config.T_max),
					"capacity": float(self.config.q_rail),
					"unit_cost": round(float(unit_cost_rail), 2),
					"deploy_cost": float(self.config.c_deploy_rail), "min_units": 1,
					"population_density": pop_dens_rail_val,
				}
				arc_id += 1
				arcs.append(rail_arc_u_v)

				# Single physical rail arc v -> u (trapezoidal fuzzy time, #9)
				rail_arc_v_u = {
					"id": arc_id, "i": v, "j": u, "mode": 2, "distance": round(float(dist), 2),
					"time_a": round(float(base_time * rail_ratios[0]), 2),
					"time_b": round(float(base_time * rail_ratios[1]), 2),
					"time_c": round(float(base_time * rail_ratios[2]), 2),
					"time_d": round(float(base_time * rail_ratios[3]), 2),
					"alpha": alpha_rail_val,
					"C_consequence": c_conseq_rail_val,
					"e_window": 0.0, "l_window": float(self.config.T_max),
					"capacity": float(self.config.q_rail),
					"unit_cost": round(float(unit_cost_rail), 2),
					"deploy_cost": float(self.config.c_deploy_rail), "min_units": 1,
					"population_density": pop_dens_rail_val,
				}
				arc_id += 1
				arcs.append(rail_arc_v_u)

				# Generate Timetabled Train Services for u -> v and v -> u
				for dep_h in dynamic_schedule_hours:
					arr_h = round(dep_h + base_time, 2)
					railway_timetable.append({
						"train_id": f"X{train_id_counter}",
						"origin_hub": u, "destination_hub": v, "departure_time": dep_h,
						"departure_time_fmt": float_to_day_time(dep_h),
						"arrival_time": arr_h, "arrival_time_fmt": float_to_day_time(arr_h),
						"capacity": float(self.config.q_rail),
					})
					train_id_counter += 1
					
					arr_h_rev = round(dep_h + base_time, 2)
					railway_timetable.append({
						"train_id": f"X{train_id_counter}",
						"origin_hub": v, "destination_hub": u, "departure_time": dep_h,
						"departure_time_fmt": float_to_day_time(dep_h),
						"arrival_time": arr_h_rev, "arrival_time_fmt": float_to_day_time(arr_h_rev),
						"capacity": float(self.config.q_rail),
					})
					train_id_counter += 1




		# 4. Create Feeder/Distribution Road Network (Mode 1) from Spokes to Hubs
		for s in spoke_ids:
			# Strict topological island isolation: West spokes ONLY connect to West hubs; East spokes ONLY connect to East hubs
			target_hubs = west_hubs if s in origin_spokes else east_hubs
			hub_distances = [(h, np.linalg.norm(np.array(coords[s]) - np.array(coords[h]))) for h in target_hubs]
			hub_distances.sort(key=lambda x: x[1])
			closest_hubs = [h for h, _ in hub_distances[:min(2, len(target_hubs))]]

			for h in closest_hubs:
				dist = rng.uniform(50, 300)
				base_time = dist / self.config.road_speed

				# Unique risk parameters for the undirected road edge
				alpha_road_val = float(round(rng.uniform(0.005, 0.03), 4))
				c_conseq_road_val = float(round(rng.uniform(50.0, 200.0), 2))
				pop_dens_road_val = float(round(rng.uniform(1.0, 5.0), 2))

				feeder_arc_s_h = {
					"id": arc_id, "i": s, "j": h, "mode": 1, "distance": round(float(dist), 2),
					"time_a": round(float(base_time * 0.8), 2),
					"time_b": round(float(base_time), 2),
					"time_c": round(float(base_time * 1.5), 2),
					"alpha": alpha_road_val,
					"C_consequence": c_conseq_road_val,
					"e_window": 0.0, "l_window": float(self.config.T_max),
					"capacity": float(self.config.q_road),
					"unit_cost": round(float(self.config.c_road * dist), 2),
					"deploy_cost": float(self.config.c_deploy_road), "min_units": 1,
					"population_density": pop_dens_road_val,
				}
				arc_id += 1
				feeder_arc_h_s = {
					"id": arc_id, "i": h, "j": s, "mode": 1, "distance": round(float(dist), 2),
					"time_a": round(float(base_time * 0.8), 2),
					"time_b": round(float(base_time), 2),
					"time_c": round(float(base_time * 1.5), 2),
					"alpha": alpha_road_val,
					"C_consequence": c_conseq_road_val,
					"e_window": 0.0, "l_window": float(self.config.T_max),
					"capacity": float(self.config.q_road),
					"unit_cost": round(float(self.config.c_road * dist), 2),
					"deploy_cost": float(self.config.c_deploy_road), "min_units": 1,
					"population_density": pop_dens_road_val,
				}
				arc_id += 1
				arcs.extend([feeder_arc_s_h, feeder_arc_h_s])

		# 4.5 Create Peripheral Ring Road Network (Mode 1) strictly within each island (West only, East only)
		existing_spoke_edges = set()
		for spoke_group in [origin_spokes, dest_spokes]:
			for s1 in spoke_group:
				spoke_dists = [(s2, np.linalg.norm(np.array(coords[s1]) - np.array(coords[s2]))) for s2 in spoke_group if s2 != s1]
				spoke_dists.sort(key=lambda x: x[1])
				closest_spokes = [s2 for s2, _ in spoke_dists[:min(2, len(spoke_group) - 1)]]

				for s2 in closest_spokes:
					edge_key = frozenset({s1, s2})
					if edge_key not in existing_spoke_edges:
						existing_spoke_edges.add(edge_key)
						dist = rng.uniform(50, 300)
						base_time = dist / self.config.road_speed

						# Unique risk parameters for the undirected peripheral edge
						alpha_periph_val = float(round(rng.uniform(0.005, 0.03), 4))
						c_conseq_periph_val = float(round(rng.uniform(50.0, 200.0), 2))
						pop_dens_periph_val = float(round(rng.uniform(1.0, 5.0), 2))

						spoke_arc_1 = {
							"id": arc_id, "i": s1, "j": s2, "mode": 1, "distance": round(float(dist), 2),
							"time_a": round(float(base_time * 0.8), 2), "time_b": round(float(base_time), 2),
							"time_c": round(float(base_time * 1.5), 2), "alpha": alpha_periph_val,
							"C_consequence": c_conseq_periph_val, "e_window": 0.0,
							"l_window": float(self.config.T_max), "capacity": float(self.config.q_road),
							"unit_cost": round(float(self.config.c_road * dist), 2),
							"deploy_cost": float(self.config.c_deploy_road), "min_units": 1,
							"population_density": pop_dens_periph_val,
						}
						arc_id += 1
						spoke_arc_2 = {
							"id": arc_id, "i": s2, "j": s1, "mode": 1, "distance": round(float(dist), 2),
							"time_a": round(float(base_time * 0.8), 2), "time_b": round(float(base_time), 2),
							"time_c": round(float(base_time * 1.5), 2), "alpha": alpha_periph_val,
							"C_consequence": c_conseq_periph_val, "e_window": 0.0,
							"l_window": float(self.config.T_max), "capacity": float(self.config.q_road),
							"unit_cost": round(float(self.config.c_road * dist), 2),
							"deploy_cost": float(self.config.c_deploy_road), "min_units": 1,
							"population_density": pop_dens_periph_val,
						}
						arc_id += 1
						arcs.extend([spoke_arc_1, spoke_arc_2])

		# 4.6 Create Inter-Hub Highway Network (Mode 1) for fully connected road repositioning
		# This ensures empty trucks can reposition globally, but containers are still bound by ESPPRC stage transitions
		existing_hub_edges = set()
		for h1 in hub_ids:
			for h2 in hub_ids:
				if h1 == h2:
					continue
				edge_key = tuple(sorted([h1, h2]))
				if edge_key not in existing_hub_edges:
					existing_hub_edges.add(edge_key)
					dist = rng.uniform(50, 300)
					base_time = dist / self.config.road_speed
					
					alpha_hwy_val = float(round(rng.uniform(0.005, 0.03), 4))
					c_conseq_hwy_val = float(round(rng.uniform(50.0, 200.0), 2))
					pop_dens_hwy_val = float(round(rng.uniform(1.0, 5.0), 2))

					hwy_arc_1 = {
						"id": arc_id, "i": h1, "j": h2, "mode": 1, "distance": round(float(dist), 2),
						"time_a": round(float(base_time * 0.8), 2), "time_b": round(float(base_time), 2),
						"time_c": round(float(base_time * 1.5), 2), "alpha": alpha_hwy_val,
						"C_consequence": c_conseq_hwy_val, "e_window": 0.0,
						"l_window": float(self.config.T_max), "capacity": float(self.config.q_road),
						"unit_cost": round(float(self.config.c_road * dist), 2),
						"deploy_cost": float(self.config.c_deploy_road), "min_units": 1,
						"population_density": pop_dens_hwy_val,
					}
					arc_id += 1
					hwy_arc_2 = {
						"id": arc_id, "i": h2, "j": h1, "mode": 1, "distance": round(float(dist), 2),
						"time_a": round(float(base_time * 0.8), 2), "time_b": round(float(base_time), 2),
						"time_c": round(float(base_time * 1.5), 2), "alpha": alpha_hwy_val,
						"C_consequence": c_conseq_hwy_val, "e_window": 0.0,
						"l_window": float(self.config.T_max), "capacity": float(self.config.q_road),
						"unit_cost": round(float(self.config.c_road * dist), 2),
						"deploy_cost": float(self.config.c_deploy_road), "min_units": 1,
						"population_density": pop_dens_hwy_val,
					}
					arc_id += 1
					arcs.extend([hwy_arc_1, hwy_arc_2])

		# 5. Transfer facilities strictly at Hub nodes
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

		# 6. O-D Pairs: Strictly cross-network from Origin Spokes to Destination Spokes
		od_pairs = []
		existing_od_keys = set()
		for od_idx in range(num_od_pairs):
			while True:
				o = int(rng.choice(origin_spokes))
				d = int(rng.choice(dest_spokes))
				if o == d:
					d_cands = [s for s in dest_spokes if s != o]
					d = int(rng.choice(d_cands)) if d_cands else int(rng.choice(hub_ids))
				if (o, d) not in existing_od_keys:
					existing_od_keys.add((o, d))
					break

			demand = float(rng.uniform(20, 200))
			
			# Dynamically calculate physically feasible L_od using topology, timetable and travel speeds.
			# This makes feasible deliveries narrowly bound by actual network performance.
			# 1. Road leg: origin -> closest west hub
			o_hubs = [(h, np.linalg.norm(np.array(coords[o]) - np.array(coords[h]))) for h in west_hubs]
			o_hubs.sort(key=lambda x: x[1])
			best_o_hub = o_hubs[0][0]
			dist_1 = rng.uniform(50, 300)
			t_road_1 = dist_1 / self.config.road_speed
			
			# 2. Road leg: closest east hub -> destination
			d_hubs = [(h, np.linalg.norm(np.array(coords[d]) - np.array(coords[h]))) for h in east_hubs]
			d_hubs.sort(key=lambda x: x[1])
			best_d_hub = d_hubs[0][0]
			dist_2 = rng.uniform(50, 300)
			t_road_2 = dist_2 / self.config.road_speed
			
			# 3. Rail leg
			dist_rail = rng.uniform(400, 1500)
			t_rail_b = dist_rail / self.config.rail_speed
			t_rail_c = t_rail_b * 1.3
			
			# 4. Anchor earliest departure to the timetable to keep time windows tight and realistic
			# We select a random train from the schedule to anchor this OD pair's demand,
			# ensuring that demand is distributed uniformly across the 48-hour planning horizon
			# rather than clustered at the very first train.
			target_train_h = float(rng.choice(dynamic_schedule_hours))
			origin_ready_center = target_train_h - t_road_1 - 1.0
			E_od = float(round(max(0.0, rng.uniform(origin_ready_center - 2.0, origin_ready_center + 2.0)), 1))
			
			t_arrive_hub = E_od + t_road_1 + 1.0  # 1 hour transfer
			valid_deps = [h for h in dynamic_schedule_hours if h >= t_arrive_hub]
			t_dep = valid_deps[0] if valid_deps else t_arrive_hub + 12.0
			
			t_expected_arrival = t_dep + t_rail_b + 1.0 + t_road_2
			t_worst_arrival = t_dep + t_rail_c + 2.0 + t_road_2 * 1.5
			L_od = float(round(t_expected_arrival, 1))
			
			# Bound L_od up to T_max if needed, but ensure it's feasible
			self.config.T_max = max(self.config.T_max, L_od + 72.0)

			od_pairs.append({
				"id": od_idx + 1,
				"origin": o, "destination": d, "demand": round(demand, 1),
				"E_od": round(E_od, 1), "L_od": round(L_od, 1),
				"alpha_T_od": float(self.config.alpha_T),
			})

		# Ensure all arc time windows are consistent with the final planning horizon
		for arc in arcs:
			arc["l_window"] = float(self.config.T_max)
			arc["e_window_fmt"] = float_to_day_time(arc["e_window"])
			arc["l_window_fmt"] = float_to_day_time(arc["l_window"])

		for od in od_pairs:
			od["E_od_fmt"] = float_to_day_time(od["E_od"])
			od["L_od_fmt"] = float_to_day_time(od["L_od"])

		# Adaptive vehicle limit multiplier based on instance size
		# Small: 1.5x (tight fleet), Medium: 2.0x, Large: 2.5x
		if num_nodes <= 25:
			veh_limit_mult = 1.5
		elif num_nodes <= 60:
			veh_limit_mult = 2.0
		else:
			veh_limit_mult = 2.5

		instance = {
			"name": name,
			"railway_timetable": railway_timetable,
			"network": {
				"nodes": nodes, "arcs": arcs, "transfer_nodes": transfer_nodes,
				"num_nodes": num_nodes, "num_arcs": len(arcs),
			},
			"od_pairs": od_pairs,
			"num_vehicles": num_vehicles,
			"config": {
				"T_max": self.config.T_max, "K_max": self.config.K_max,
				"alpha_max": self.config.alpha_max, "alpha_T": self.config.alpha_T,
				"gamma_sys": self.config.gamma_sys,
				"vehicle_limit_multiplier": veh_limit_mult,
				"timetable_services_per_direction": num_services,
			},
		}

		return instance
