"""Real Geographic Instance Data Generator (Xinjiang - Shanghai Corridor)."""

import math
import numpy as np

from config import RHMVSPConfig

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

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

class RealInstanceGenerator:
    """
    Generate synthetic test instances for RHMVSP based on real Xinjiang-Shanghai geographical data.
    """

    def __init__(self, config: RHMVSPConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)

    def generate(self) -> dict:
        """Generate the real geographic instance."""
        
        # 1. Define nodes and coordinates
        #    18 Hubs + 10 Spokes
        #    Three corridors: North (陇海→京沪), Central (京广→沪宁), South (兰渝→成渝→渝贵→渝昆→京九→广深)
        node_defs = {
            # ── 18 Hubs ──
            1:  {"code": "URC",  "name": "Urumqi",    "type": "Hub", "lon": 87.62,  "lat": 43.83, "region": "North"},
            2:  {"code": "LHW",  "name": "Lanzhou",    "type": "Hub", "lon": 103.83, "lat": 36.06, "region": "North"},
            46: {"code": "XNN",  "name": "Xining",     "type": "Hub", "lon": 101.77, "lat": 36.62, "region": "North"},
            21: {"code": "SIA",  "name": "Xi'an",       "type": "Hub", "lon": 108.94, "lat": 34.26, "region": "North"},
            3:  {"code": "CGO",  "name": "Zhengzhou",  "type": "Hub", "lon": 113.66, "lat": 34.75, "region": "North"},
            22: {"code": "WUH",  "name": "Wuhan",       "type": "Hub", "lon": 114.30, "lat": 30.59, "region": "Central"},
            4:  {"code": "HFE",  "name": "Hefei",      "type": "Hub", "lon": 117.23, "lat": 31.82, "region": "Central"},
            5:  {"code": "SHA",  "name": "Shanghai",    "type": "Hub", "lon": 121.47, "lat": 31.23, "region": "Central"},
            12: {"code": "NKG",  "name": "Nanjing",    "type": "Hub", "lon": 118.80, "lat": 32.06, "region": "Central"},
            14: {"code": "HGH",  "name": "Hangzhou",   "type": "Hub", "lon": 120.12, "lat": 30.22, "region": "South"},
            29: {"code": "CTU",  "name": "Chengdu",     "type": "Hub", "lon": 104.07, "lat": 30.57, "region": "South"},
            30: {"code": "CKG",  "name": "Chongqing",   "type": "Hub", "lon": 106.55, "lat": 29.56, "region": "South"},
            31: {"code": "NCH",  "name": "Nanchang",    "type": "Hub", "lon": 115.86, "lat": 28.68, "region": "South"},
            33: {"code": "KWE",  "name": "Guiyang",    "type": "Hub", "lon": 106.63, "lat": 26.65, "region": "South"},
            38: {"code": "GZU",  "name": "Guangzhou",   "type": "Hub", "lon": 113.26, "lat": 23.13, "region": "South"},
            39: {"code": "NNG",  "name": "Nanning",     "type": "Hub", "lon": 108.37, "lat": 22.82, "region": "South"},
            44: {"code": "CSX",  "name": "Changsha",   "type": "Hub", "lon": 112.94, "lat": 28.23, "region": "South"},
            45: {"code": "FOC",  "name": "Fuzhou",     "type": "Hub", "lon": 119.30, "lat": 26.08, "region": "South"},

            # ── 10 Spokes ──
            # 5 Xinjiang origins (petrochemical production)
            6:  {"code": "KHG",  "name": "Kashgar",    "type": "Spoke", "lon": 75.99,  "lat": 39.47, "region": "North"},   # Xinjiang
            7:  {"code": "HMI",  "name": "Hami",       "type": "Spoke", "lon": 93.52,  "lat": 42.82, "region": "North"},   # Xinjiang
            8:  {"code": "KRL",  "name": "Korla",      "type": "Spoke", "lon": 86.17,  "lat": 41.73, "region": "North"},   # Xinjiang
            9:  {"code": "AKU",  "name": "Aksu",       "type": "Spoke", "lon": 80.27,  "lat": 41.17, "region": "North"},   # Xinjiang
            10: {"code": "HTN",  "name": "Hotan",      "type": "Spoke", "lon": 79.92,  "lat": 37.11, "region": "North"},   # Xinjiang
            # 3 Yangtze River Delta destinations
            13: {"code": "SZV",  "name": "Suzhou",     "type": "Spoke", "lon": 120.59, "lat": 31.30, "region": "YZR"},     # Jiangsu — chemical fibre parks
            41: {"code": "NTG",  "name": "Nantong",    "type": "Spoke", "lon": 120.86, "lat": 32.01, "region": "YZR"},     # Jiangsu — new chemical zone
            47: {"code": "WUX",  "name": "Wuxi",       "type": "Spoke", "lon": 120.30, "lat": 31.57, "region": "YZR"},     # Jiangsu — petrochemical base
            # 2 Pearl River Delta destinations
            37: {"code": "SZX",  "name": "Shenzhen",   "type": "Spoke", "lon": 114.07, "lat": 22.55, "region": "PRD"},     # Guangdong — port hazmat
            42: {"code": "HUZ",  "name": "Huizhou",    "type": "Spoke", "lon": 114.42, "lat": 23.11, "region": "PRD"},     # Guangdong — chemical zone

        }

        hub_ids = [k for k, v in node_defs.items() if v["type"] == "Hub"]
        spoke_ids = [k for k, v in node_defs.items() if v["type"] == "Spoke"]

        nodes = [{"id": k, "x": round(v["lon"], 2), "y": round(v["lat"], 2), "name": v["name"]}
                 for k, v in node_defs.items()]

        # ── Rail edges: hub–hub, based on real Chinese railway network ──
        rail_edges = [
            # 兰新线 + 兰青线: Urumqi → Lanzhou → Xining
            (1, 2),        # Urumqi → Lanzhou     [兰新线]
            (2, 46),       # Lanzhou → Xining     [兰青线]
            # 北 corridor (陇海线 + 西成高铁)
            (2, 21),       # Lanzhou → Xi'an      [陇海线]
            (21, 3),       # Xi'an → Zhengzhou    [陇海线]
            (21, 29),      # Xi'an → Chengdu      [西成高铁]
            # 京广线: Zhengzhou → Wuhan → Changsha → Guangzhou
            (3, 22),       # Zhengzhou → Wuhan    [京广线]
            (22, 44),      # Wuhan → Changsha     [京广线]
            (44, 38),      # Changsha → Guangzhou [京广线]
            # 郑渝高铁: Zhengzhou → Chongqing (direct)
            (3, 30),       # Zhengzhou → Chongqing [郑渝高铁 2022]
            # 郑州→东部长三角: Zhengzhou → Hefei / Nanjing / Shanghai (经徐州枢纽)
            (3, 4),        # Zhengzhou → Hefei     [京广→合蚌高铁]
            (3, 12),       # Zhengzhou → Nanjing   [陇海→京沪线 经徐州]
            (3, 5),        # Zhengzhou → Shanghai  [陇海→京沪线 经徐州]
            # 合武线/合宁线/沪宁线: Wuhan → Hefei → Nanjing → Shanghai
            (22, 4),       # Wuhan → Hefei        [合武线]
            (4, 12),       # Hefei → Nanjing      [合宁线]
            (12, 5),       # Nanjing → Shanghai   [沪宁线]
            # 宁杭高铁: Nanjing → Hangzhou (direct)
            (12, 14),      # Nanjing → Hangzhou   [宁杭高铁 2013]
            # 南 corridor (兰渝→成渝→渝贵→黔桂→南广)
            (2, 29),       # Lanzhou → Chengdu    [兰渝线]
            (29, 30),      # Chengdu → Chongqing  [成渝线]
            (30, 22),      # Chongqing → Wuhan    [渝利铁路/郑渝通道]
            (30, 44),      # Chongqing → Changsha [渝怀铁路/渝湘通道]
            (30, 33),      # Chongqing → Guiyang  [渝贵线]
            (33, 39),      # Guiyang → Nanning    [黔桂线]
            (39, 38),      # Nanning → Guangzhou  [南广高铁]
            # 成贵高铁: Chengdu → Guiyang (direct)
            (29, 33),      # Chengdu → Guiyang    [成贵高铁 2019]
            # 贵广高铁: Guiyang → Guangzhou (direct)
            (33, 38),      # Guiyang → Guangzhou  [贵广高铁 2014]
            # 沪昆线: Shanghai ↔ Hangzhou ↔ Nanchang ↔ Changsha
            (14, 5),       # Hangzhou → Shanghai  [沪昆线]
            (31, 14),      # Nanchang → Hangzhou  [沪昆线]
            (31, 44),      # Nanchang → Changsha  [沪昆线]
            # 杭深线/沿海通道: Hangzhou → Fuzhou → Guangzhou
            (14, 45),      # Hangzhou → Fuzhou    [杭深线]
            (45, 38),      # Fuzhou → Guangzhou   [杭深线]
            # 京九线/广深线
            (38, 31),      # Guangzhou → Nanchang [京九线/广深线]
            # 京九线/广深线: Nanchang → Shenzhen
            (31, 37),      # Nanchang → Shenzhen  [京九线/广深线]
            # 武九线: Wuhan → Nanchang (via Jiujiang)
            (22, 31),      # Wuhan → Nanchang    [武九线]
        ]

        # ── Road edges: spoke→nearest hub(s) + spoke↔spoke within group ──
        road_edges = []

        # Spoke-to-hub: each spoke connects to its 1–2 nearest hubs by road
        # Each spoke MUST connect to its provincial-capital Hub
        spoke_nearest_hubs = {
            # Xinjiang origins → Urumqi (sole rail entry)
            6:  [1],       # Kashgar → Urumqi
            7:  [1],       # Hami → Urumqi
            8:  [1],       # Korla → Urumqi
            9:  [1],       # Aksu → Urumqi
            10: [1],       # Hotan → Urumqi
            # Yangtze River Delta destinations
            13: [5],       # Suzhou → Shanghai
            41: [5],       # Nantong → Shanghai
            47: [5, 12],   # Wuxi → Shanghai, Nanjing
            # Pearl River Delta destinations
            37: [38],      # Shenzhen → Guangzhou
            42: [38],      # Huizhou → Guangzhou
        }
        for spoke, hubs in spoke_nearest_hubs.items():
            for h in hubs:
                road_edges.append((spoke, h))

        # Spoke↔spoke road mesh within each corridor group
        spoke_groups = {
            "North": [6, 7, 8, 9, 10],    # Xinjiang origins
            "YZR":   [13, 41, 47],         # Yangtze River Delta destinations
            "PRD":   [37, 42],             # Pearl River Delta destinations
        }
        for group_spokes in spoke_groups.values():
            for i in range(len(group_spokes)):
                for j in range(i + 1, len(group_spokes)):
                    road_edges.append((group_spokes[i], group_spokes[j]))
        
        arcs = []
        arc_id = 1
        
        # Construct Rail Arcs
        for u, v in rail_edges:
            dist = haversine_distance(node_defs[u]["lon"], node_defs[u]["lat"], node_defs[v]["lon"], node_defs[v]["lat"])
            base_time = dist / self.config.rail_speed
            
            # Rail specific costs
            discount = 0.85 if dist > 2000 else 1.0 # discount for long-haul
            unit_cost_rail = ((self.config.c_rail_base_fixed + self.config.c_rail_base_var * dist) * (1 + self.config.c_rail_hazmat_markup) + self.config.c_rail_fund * dist) * discount
            
            alpha_rail_val = float(round(self.rng.uniform(0.0005, 0.003), 5))
            rail_ratios = getattr(self.config, 'rail_fuzzy_time_ratio', (0.9, 1.0, 1.0, 1.3))
            
            for src, dst in [(u, v), (v, u)]:
                arcs.append({
                    "id": arc_id, "i": src, "j": dst, "mode": 2, "distance": round(float(dist), 2),
                    "time_a": round(float(base_time * rail_ratios[0]), 2),
                    "time_b": round(float(base_time * rail_ratios[1]), 2),
                    "time_c": round(float(base_time * rail_ratios[2]), 2),
                    "time_d": round(float(base_time * rail_ratios[3]), 2),
                    "alpha": alpha_rail_val,
                    "e_window": 0.0, "l_window": 999.0, # Will update max T_max later
                    "e_window_fmt": float_to_day_time(0.0), "l_window_fmt": float_to_day_time(999.0),
                    "capacity": float(self.config.q_rail),
                    "unit_cost": round(float(unit_cost_rail), 2),
                    "deploy_cost": float(self.config.c_deploy_rail), "min_units": 1,
                })
                arc_id += 1

        # Construct Road Arcs
        for u, v in road_edges:
            dist = max(1.0, haversine_distance(node_defs[u]["lon"], node_defs[u]["lat"], node_defs[v]["lon"], node_defs[v]["lat"]))
            base_time = dist / self.config.road_speed
            alpha_road_val = float(round(self.rng.uniform(0.005, 0.03), 4))
            road_ratios = getattr(self.config, 'road_fuzzy_time_ratio', (0.8, 1.0, 1.0, 1.5))
            
            for src, dst in [(u, v), (v, u)]:
                arcs.append({
                    "id": arc_id, "i": src, "j": dst, "mode": 1, "distance": round(float(dist), 2),
                    "time_a": round(float(base_time * road_ratios[0]), 2),
                    "time_b": round(float(base_time * road_ratios[1]), 2),
                    "time_c": round(float(base_time * road_ratios[2]), 2),
                    "time_d": round(float(base_time * road_ratios[3]), 2),
                    "alpha": alpha_road_val,
                    "e_window": 0.0, "l_window": 999.0,
                    "e_window_fmt": float_to_day_time(0.0), "l_window_fmt": float_to_day_time(999.0),
                    "capacity": float(self.config.q_road),
                    "unit_cost": round(float(self.config.c_road * dist), 2),
                    "deploy_cost": float(self.config.c_deploy_road), "min_units": 1,
                })
                arc_id += 1

        # 3. Transfer facilities at Hub nodes
        transfer_nodes = []
        for hub in hub_ids:
            gamma_val = float(round(self.rng.uniform(0.001, 0.01), 5))
            transfer_nodes.append({
                "node_id": hub, "transfer_time_a": 4.0, "transfer_time_b": 8.0,
                "transfer_time_c": 12.0, "transfer_time_d": 16.0,
                "gamma": gamma_val,
                "unit_transfer_cost": float(self.config.c_transfer),
                "facility_cost": float(self.config.c_hub),
            })

        # 4. Generate Specific OD Pairs (West to East)
        # All OD pairs route through the hub chain: origin→(road)→Lanzhou→(rail)→...→Shanghai→(road)→destination
        # origin_hub=2 (Lanzhou) is the network entry hub
        # destination_hub depends on which hub the destination spoke connects to
        tasks = [
            {"o": 6,  "d": 13, "dh": 5,  "multiplier": 150.0 / self.config.q_rail},  # OD 1: Kashgar → Suzhou    [→Shanghai hub 5]  (北 corridor)
            {"o": 10, "d": 41, "dh": 5,  "multiplier": 240.0 / self.config.q_rail},  # OD 2: Hotan → Nantong     [→Shanghai hub 5]  (北 corridor)
            {"o": 9,  "d": 47, "dh": 5,  "multiplier": 360.0 / self.config.q_rail},  # OD 3: Aksu → Wuxi        [→Shanghai hub 5]  (北 corridor)
            {"o": 8,  "d": 37, "dh": 38, "multiplier": 320.0 / self.config.q_rail},  # OD 4: Korla → Shenzhen   [→Guangzhou hub 38] (南 corridor)
            {"o": 7,  "d": 42, "dh": 38, "multiplier": 270.0 / self.config.q_rail},  # OD 5: Hami → Huizhou     [→Guangzhou hub 38] (南 corridor)
        ]

        od_pairs = []
        max_required_time = 0.0

        # Update T_max globally for this instance to 28 days (672 hours)
        self.config.T_max = 672.0

        for idx, task in enumerate(tasks):
            o, d, dh = task["o"], task["d"], task["dh"]
            demand = float(round(task["multiplier"] * self.config.q_rail, 0))

            # --- Compute total travel time (road1 + rail + road2) for this OD ---
            t_road_1 = haversine_distance(node_defs[o]["lon"], node_defs[o]["lat"],
                                          node_defs[2]["lon"], node_defs[2]["lat"]) / self.config.road_speed
            t_rail = 0.0
            rail_chains = {
                5:  [2, 21, 3, 22, 4, 5],       # Lanzhou → Xi'an → … → Shanghai (北 corridor)
                38: [2, 29, 30, 33, 38],          # Lanzhou → Chengdu → … → Guiyang → Guangzhou (南 corridor)
            }
            chain = rail_chains.get(dh, [2, 29, 30, 39, 38, 31, 14, 5])
            for k in range(len(chain) - 1):
                t_rail += haversine_distance(node_defs[chain[k]]["lon"], node_defs[chain[k]]["lat"],
                                             node_defs[chain[k+1]]["lon"], node_defs[chain[k+1]]["lat"]) / self.config.rail_speed
            t_road_2 = haversine_distance(node_defs[dh]["lon"], node_defs[dh]["lat"],
                                          node_defs[d]["lon"], node_defs[d]["lat"]) / self.config.road_speed
            # Physical travel time (road + rail only)
            t_travel_raw = t_road_1 + t_rail + t_road_2
            # Estimated overhead: transfers + rail waiting.
            # The reliability evaluator now includes rail waiting in z_d.
            # 60h gives enough slack for all ODs to be feasible while keeping
            # R_time < 1.0 for most ODs.
            t_overhead = 60.0
            t_travel = t_travel_raw + t_overhead

            # --- Per-OD delivery deadline ---
            # Deadline = start + t_travel, rounded to 12h grid for nicer windows.
            # The fuzzy upper bound z_d (now including rail waiting) ≈ 1.3×raw_time,
            # which should exceed the deadline, giving R_time < 1.0.
            raw_start = t_road_1 + 10.0 + idx * 6.0
            time_window_start = float(np.ceil(raw_start / 12.0) * 12)
            time_window_end = float(np.ceil((time_window_start + t_travel) / 12.0) * 12)
            tw_width = time_window_end - time_window_start

            max_required_time = max(max_required_time, time_window_end)

            od_pairs.append({
                "id": idx + 1,
                "origin": o, "destination": d, "demand": demand,
                "time_window_start": round(time_window_start, 1),
                "time_window_end": round(time_window_end, 1),
                "time_window_start_fmt": float_to_day_time(time_window_start),
                "time_window_end_fmt": float_to_day_time(time_window_end),
                "alpha_T_od": float(self.config.alpha_T),
                "origin_hub": 2,     # Lanzhou — network entry hub
                "destination_hub": dh,
                "dist_road_1": round(haversine_distance(node_defs[o]["lon"], node_defs[o]["lat"],
                                                       node_defs[2]["lon"], node_defs[2]["lat"]), 2),
                "dist_rail": round(sum(
                    haversine_distance(node_defs[chain[k]]["lon"], node_defs[chain[k]]["lat"],
                                       node_defs[chain[k+1]]["lon"], node_defs[chain[k+1]]["lat"])
                    for k in range(len(chain) - 1)
                ), 2),
                "dist_road_2": round(haversine_distance(node_defs[dh]["lon"], node_defs[dh]["lat"],
                                                       node_defs[d]["lon"], node_defs[d]["lat"]), 2),
            })
            
        for arc in arcs:
            arc["l_window"] = float(self.config.T_max)
            arc["l_window_fmt"] = float_to_day_time(float(self.config.T_max))

        # 5. Railway Timetable
        # Complete graph: generate 2 trains per day on EACH rail edge
        railway_timetable = []
        train_id_counter = 1
        
        # For 28 days
        for day in range(1, 29):
            for u, v in rail_edges:
                dist = haversine_distance(node_defs[u]["lon"], node_defs[u]["lat"], node_defs[v]["lon"], node_defs[v]["lat"])
                trip_time = dist / self.config.rail_speed
                
                # Network Propagation Delay (Cascading)
                if u < v: # Eastbound
                    if u == 1:
                        offset = 0.0
                    else:
                        offset = haversine_distance(node_defs[1]["lon"], node_defs[1]["lat"], node_defs[u]["lon"], node_defs[u]["lat"]) / self.config.rail_speed + 4.0
                else: # Westbound
                    if u == 5:
                        offset = 0.0
                    else:
                        offset = haversine_distance(node_defs[5]["lon"], node_defs[5]["lat"], node_defs[u]["lon"], node_defs[u]["lat"]) / self.config.rail_speed + 4.0
                        
                # 2 trains per day
                for train_num in range(2):
                    base_hour = 6 + train_num * 12
                    
                    # S2-1: Compute separate offsets for forward (u→v) and reverse (v→u) directions.
                    # Forward offset based on u's position, reverse offset based on v's position.
                    offset_fwd = haversine_distance(node_defs[1]["lon"], node_defs[1]["lat"], node_defs[u]["lon"], node_defs[u]["lat"]) / self.config.rail_speed + 4.0 if u != 1 else 0.0
                    offset_rev = haversine_distance(node_defs[1]["lon"], node_defs[1]["lat"], node_defs[v]["lon"], node_defs[v]["lat"]) / self.config.rail_speed + 4.0 if v != 1 else 0.0
                    
                    # Forward: u → v
                    dep_h_fwd = (day - 1) * 24 + base_hour + offset_fwd + self.rng.uniform(-0.5, 0.5)
                    arr_h_fwd = dep_h_fwd + trip_time
                    railway_timetable.append({
                        "train_id": train_id_counter, "origin_hub": u, "destination_hub": v,
                        "departure_time": round(dep_h_fwd, 2), "arrival_time": round(arr_h_fwd, 2),
                        "dep_fmt": float_to_day_time(dep_h_fwd), "arr_fmt": float_to_day_time(arr_h_fwd)
                    })
                    train_id_counter += 1
                    
                    # Reverse: v → u
                    dep_h_rev = (day - 1) * 24 + base_hour + offset_rev + self.rng.uniform(-0.5, 0.5)
                    arr_h_rev = dep_h_rev + trip_time
                    railway_timetable.append({
                        "train_id": train_id_counter, "origin_hub": v, "destination_hub": u,
                        "departure_time": round(dep_h_rev, 2), "arrival_time": round(arr_h_rev, 2),
                        "dep_fmt": float_to_day_time(dep_h_rev), "arr_fmt": float_to_day_time(arr_h_rev)
                    })
                    train_id_counter += 1

        instance_data = {
            "name": "rhmvsp_real_case",
            "network": {
                "num_nodes": len(nodes),
                "num_arcs": len(arcs),
                "nodes": nodes,
                "arcs": arcs,
                "transfer_nodes": transfer_nodes,
            },
            "od_pairs": od_pairs,
            "railway_timetable": railway_timetable,
            "hub_fleet_size": {"1": 40, "5": 10, "38": 24},  # per-hub fleet (disabled for now)
            "num_vehicles_west": 46,
            "num_vehicles_east": 46,
            "config": {
                "T_max": self.config.T_max
            }
        }
        return instance_data
