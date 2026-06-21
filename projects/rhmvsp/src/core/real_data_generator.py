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
        #    15 Hubs + 12 Spokes
        #    Three corridors: North (陇海→京沪), Central (京广→沪宁), South (兰渝→成渝→渝贵→渝昆→京九→广深)
        node_defs = {
            # ── 15 Hubs ──
            1:  {"code": "URC",  "name": "Urumqi",    "type": "Hub", "lon": 87.62,  "lat": 43.83, "region": "North"},
            2:  {"code": "LHW",  "name": "Lanzhou",    "type": "Hub", "lon": 103.83, "lat": 36.06, "region": "North"},
            21: {"code": "SIA",  "name": "Xi'an",       "type": "Hub", "lon": 108.94, "lat": 34.26, "region": "North"},
            3:  {"code": "CGO",  "name": "Zhengzhou",  "type": "Hub", "lon": 113.66, "lat": 34.75, "region": "North"},
            11: {"code": "XUZ",  "name": "Xuzhou",     "type": "Hub", "lon": 117.28, "lat": 34.20, "region": "North"},
            28: {"code": "LYG",  "name": "Lianyungang", "type": "Hub", "lon": 119.17, "lat": 34.60, "region": "North"},
            40: {"code": "JNA",  "name": "Jinan",       "type": "Hub", "lon": 117.00, "lat": 36.67, "region": "North"},
            22: {"code": "WUH",  "name": "Wuhan",       "type": "Hub", "lon": 114.30, "lat": 30.59, "region": "Central"},
            4:  {"code": "HFE",  "name": "Hefei",      "type": "Hub", "lon": 117.23, "lat": 31.82, "region": "Central"},
            5:  {"code": "SHA",  "name": "Shanghai",    "type": "Hub", "lon": 121.47, "lat": 31.23, "region": "Central"},
            29: {"code": "CTU",  "name": "Chengdu",     "type": "Hub", "lon": 104.07, "lat": 30.57, "region": "South"},
            30: {"code": "CKG",  "name": "Chongqing",   "type": "Hub", "lon": 106.55, "lat": 29.56, "region": "South"},
            31: {"code": "NCH",  "name": "Nanchang",    "type": "Hub", "lon": 115.86, "lat": 28.68, "region": "South"},
            38: {"code": "GZU",  "name": "Guangzhou",   "type": "Hub", "lon": 113.26, "lat": 23.13, "region": "South"},
            39: {"code": "NNG",  "name": "Nanning",     "type": "Hub", "lon": 108.37, "lat": 22.82, "region": "South"},

            # ── North Spokes (Xinjiang + Longhai corridor) ──
            6:  {"code": "KHG",  "name": "Kashgar",    "type": "Spoke", "lon": 75.99,  "lat": 39.47, "region": "North"},
            7:  {"code": "HMI",  "name": "Hami",       "type": "Spoke", "lon": 93.52,  "lat": 42.82, "region": "North"},
            8:  {"code": "KRL",  "name": "Korla",      "type": "Spoke", "lon": 86.17,  "lat": 41.73, "region": "North"},
            9:  {"code": "AKU",  "name": "Aksu",       "type": "Spoke", "lon": 80.27,  "lat": 41.17, "region": "North"},
            10: {"code": "HTN",  "name": "Hotan",      "type": "Spoke", "lon": 79.92,  "lat": 37.11, "region": "North"},
            24: {"code": "BJI",  "name": "Baoji",      "type": "Spoke", "lon": 107.24, "lat": 34.36, "region": "North"},
            34: {"code": "RZH",  "name": "Rizhao",     "type": "Spoke", "lon": 119.46, "lat": 35.38, "region": "North"},

            # ── Central Spokes (Jingguang corridor + Yangtze Delta) ──
            12: {"code": "NKG",  "name": "Nanjing",    "type": "Spoke", "lon": 118.80, "lat": 32.06, "region": "Central"},
            13: {"code": "SZV",  "name": "Suzhou",     "type": "Spoke", "lon": 120.59, "lat": 31.30, "region": "Central"},

            # ── South Spokes ──
            14: {"code": "HGH",  "name": "Hangzhou",   "type": "Spoke", "lon": 120.12, "lat": 30.22, "region": "South"},
            32: {"code": "MYG",  "name": "Mianyang",   "type": "Spoke", "lon": 104.68, "lat": 31.47, "region": "South"},
            33: {"code": "KWE",  "name": "Guiyang",    "type": "Spoke", "lon": 106.63, "lat": 26.65, "region": "South"},
            34: {"code": "RZH",  "name": "Rizhao",     "type": "Spoke", "lon": 119.46, "lat": 35.38, "region": "North"},
            35: {"code": "LUZ",  "name": "Luzhou",     "type": "Spoke", "lon": 105.44, "lat": 28.88, "region": "South"},
            36: {"code": "QZH",  "name": "Qinzhou",    "type": "Spoke", "lon": 108.65, "lat": 21.98, "region": "South"},
            37: {"code": "SZX",  "name": "Shenzhen",   "type": "Spoke", "lon": 114.07, "lat": 22.55, "region": "South"},
        }

        hub_ids = [k for k, v in node_defs.items() if v["type"] == "Hub"]
        spoke_ids = [k for k, v in node_defs.items() if v["type"] == "Spoke"]

        nodes = [{"id": k, "x": round(v["lon"], 2), "y": round(v["lat"], 2), "name": v["name"]}
                 for k, v in node_defs.items()]

        # ── Rail edges: hub–hub only, three-corridor hub chains ──
        rail_edges = [
            # Shared entry: Urumqi → Lanzhou
            (1, 2),
            # North corridor (陇海线 → 京沪线 → 胶新线)
            (2, 21),    # Lanzhou → Xi'an
            (21, 3),    # Xi'an → Zhengzhou
            (3, 11),    # Zhengzhou → Xuzhou
            (11, 40),   # Xuzhou → Jinan     [京沪线]
            (40, 28),   # Jinan → Lianyungang [胶新线]
            # Central corridor (京广线 → 区域线 → 沪宁线)
            (3, 22),    # Zhengzhou → Wuhan
            (22, 4),    # Wuhan → Hefei
            (4, 5),     # Hefei → Shanghai
            # South corridor (兰渝→成渝→渝贵→京九→广深/广昆)
            (2, 29),    # Lanzhou → Chengdu
            (29, 30),   # Chengdu → Chongqing
            (30, 39),   # Chongqing → Nanning   [渝贵线/湘桂线]
            (39, 38),   # Nanning → Guangzhou    [南广高铁]
            (38, 31),   # Guangzhou → Nanchang   [京广线]
            (31, 14),   # Nanchang → Hangzhou    [沪昆线]
            (14, 5),    # Hangzhou → Shanghai    [沪昆线]
            (29, 35),   # Chengdu → Luzhou       [隆黄铁路]
            (31, 37),   # Nanchang → Shenzhen    [京九线/广深线]
            (31, 36),   # Nanchang → Qinzhou     [南防线]
        ]

        # ── Road edges: spoke→nearest hub(s) + spoke↔spoke within group ──
        road_edges = []

        # Spoke-to-hub: each spoke connects to its 1–2 nearest hubs by road
        spoke_nearest_hubs = {
            6:  [2],       # Kashgar → Lanzhou
            7:  [2],       # Hami → Lanzhou
            8:  [1, 2],    # Korla → Urumqi, Lanzhou
            9:  [2],       # Aksu → Lanzhou
            10: [2],       # Hotan → Lanzhou
            24: [21],      # Baoji → Xi'an
            34: [40],      # Rizhao → Jinan
            12: [4, 5],    # Nanjing → Hefei, Shanghai
            13: [5],       # Suzhou → Shanghai
            14: [5, 31],   # Hangzhou → Shanghai, Nanchang
            32: [29],      # Mianyang → Chengdu
            33: [30, 31],  # Guiyang → Chongqing, Nanchang
            35: [29],      # Luzhou → Chengdu
            36: [39],      # Qinzhou → Nanning
            37: [38],      # Shenzhen → Guangzhou
        }
        for spoke, hubs in spoke_nearest_hubs.items():
            for h in hubs:
                road_edges.append((spoke, h))

        # Spoke↔spoke road mesh within each corridor group
        spoke_groups = {
            "North":   [6, 7, 8, 9, 10, 24, 34],
            "Central":  [12, 13],
            "South":    [14, 32, 33, 35, 36, 37],
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
                "node_id": hub, "transfer_time_a": 0.5, "transfer_time_b": 1.0,
                "transfer_time_c": 2.0, "transfer_time_d": 4.0,
                "gamma": gamma_val,
                "unit_transfer_cost": float(self.config.c_transfer),
                "facility_cost": float(self.config.c_hub),
            })

        # 4. Generate Specific OD Pairs (West to East)
        # All OD pairs route through the hub chain: origin→(road)→Lanzhou→(rail)→...→Shanghai→(road)→destination
        # origin_hub=2 (Lanzhou) is the network entry hub; destination_hub=5 (Shanghai) is the exit hub
        tasks = [
            {"o": 6, "d": 13, "multiplier": 150.0 / self.config.q_rail},  # OD 1: Kashgar → Suzhou     [Central corridor]
            {"o": 10, "d": 35, "multiplier": 240.0 / self.config.q_rail}, # OD 2: Hotan → Luzhou       [South corridor]
            {"o": 9, "d": 38, "multiplier": 360.0 / self.config.q_rail}, # OD 3: Aksu → Guangzhou     [South corridor]
            {"o": 8, "d": 39, "multiplier": 510.0 / self.config.q_rail}, # OD 4: Korla → Nanning      [South corridor]
            {"o": 7, "d": 40, "multiplier": 300.0 / self.config.q_rail}, # OD 5: Hami → Jinan         [North corridor]
        ]

        od_pairs = []
        max_required_time = 0.0

        # Update T_max globally for this instance to 28 days (672 hours)
        self.config.T_max = 672.0

        for idx, task in enumerate(tasks):
            o, d = task["o"], task["d"]
            demand = float(round(task["multiplier"] * self.config.q_rail, 0))

            time_window_start = float(round(10.0 + idx * 20.0 + self.rng.uniform(-5, 5), 1))

            # Geographical distance-based deadline via the hub chain
            # Route: Origin(o) →(road)→ Lanzhou(2) →(rail chain)→ Shanghai(5) →(road)→ Destination(d)
            # Use the longest corridor (South: Lanzhou→Chengdu→Chongqing→Wuhan→Nanchang→Hangzhou→Shanghai) as upper bound
            t_road_1 = haversine_distance(node_defs[o]["lon"], node_defs[o]["lat"],
                                          node_defs[2]["lon"], node_defs[2]["lat"]) / self.config.road_speed
            t_rail = 0.0
            chain = [2, 29, 30, 22, 31, 14, 5]  # South corridor hub chain
            for k in range(len(chain) - 1):
                t_rail += haversine_distance(node_defs[chain[k]]["lon"], node_defs[chain[k]]["lat"],
                                             node_defs[chain[k+1]]["lon"], node_defs[chain[k+1]]["lat"]) / self.config.rail_speed
            t_road_2 = haversine_distance(node_defs[5]["lon"], node_defs[5]["lat"],
                                          node_defs[d]["lon"], node_defs[d]["lat"]) / self.config.road_speed

            # 20 hours buffer for waiting and transfers
            t_expected = time_window_start + t_road_1 + t_rail + t_road_2 + 20.0
            # 12 hour additional buffer so time_reliability lands in [0.3, 0.8]
            time_window_end = float(round(t_expected + 12.0, 1))

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
                "destination_hub": 5, # Shanghai — network exit hub
                "dist_road_1": round(haversine_distance(node_defs[o]["lon"], node_defs[o]["lat"],
                                                       node_defs[2]["lon"], node_defs[2]["lat"]), 2),
                "dist_rail": round(sum(
                    haversine_distance(node_defs[chain[k]]["lon"], node_defs[chain[k]]["lat"],
                                       node_defs[chain[k+1]]["lon"], node_defs[chain[k+1]]["lat"])
                    for k in range(len(chain) - 1)
                ), 2),
                "dist_road_2": round(haversine_distance(node_defs[5]["lon"], node_defs[5]["lat"],
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
                    
                    # Staggered departure with +/- 0.5h uniform jitter
                    dep_h = (day - 1) * 24 + base_hour + offset + self.rng.uniform(-0.5, 0.5)
                    arr_h = dep_h + trip_time
                    
                    # Both directions
                    railway_timetable.append({
                        "train_id": train_id_counter, "origin_hub": u, "destination_hub": v,
                        "departure_time": round(dep_h, 2), "arrival_time": round(arr_h, 2),
                        "dep_fmt": float_to_day_time(dep_h), "arr_fmt": float_to_day_time(arr_h)
                    })
                    train_id_counter += 1
                    
                    railway_timetable.append({
                        "train_id": train_id_counter, "origin_hub": v, "destination_hub": u,
                        "departure_time": round(dep_h, 2), "arrival_time": round(arr_h, 2),
                        "dep_fmt": float_to_day_time(dep_h), "arr_fmt": float_to_day_time(arr_h)
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
            "num_vehicles_west": 70,
            "num_vehicles_east": 70,
            "config": {
                "T_max": self.config.T_max
            }
        }
        return instance_data
