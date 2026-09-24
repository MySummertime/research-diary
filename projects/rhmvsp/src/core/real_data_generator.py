"""Real Geographic Instance Data Generator (Xinjiang - Shanghai Corridor)."""

import math

from config import RHMVSPConfig, default_empty_reposition_hub_fleet
from .synthetic_beliefs import arc_belief, draw_belief

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

    def generate(self) -> dict:
        """Generate the real geographic instance."""
        
        # 1. Define nodes and coordinates
        #    18 Hubs + 10 Spokes
        #    Three corridors: North (陇海→京沪), Central (京广→沪宁), South (兰渝→成渝→渝贵→渝昆→京九→广深)
        node_defs = {
            # ── 18 Hubs ──
            1: {
                "code": "URC",
                "name": "Urumqi",
                "type": "Hub",
                "lon": 87.62,
                "lat": 43.83,
            },
            2: {
                "code": "LHW",
                "name": "Lanzhou",
                "type": "Hub",
                "lon": 103.83,
                "lat": 36.06,
            },
            46: {
                "code": "XNN",
                "name": "Xining",
                "type": "Hub",
                "lon": 101.77,
                "lat": 36.62,
            },
            21: {
                "code": "SIA",
                "name": "Xi'an",
                "type": "Hub",
                "lon": 108.94,
                "lat": 34.26,
            },
            3: {
                "code": "CGO",
                "name": "Zhengzhou",
                "type": "Hub",
                "lon": 113.66,
                "lat": 34.75,
            },
            22: {
                "code": "WUH",
                "name": "Wuhan",
                "type": "Hub",
                "lon": 114.30,
                "lat": 30.59,
            },
            4: {
                "code": "HFE",
                "name": "Hefei",
                "type": "Hub",
                "lon": 117.23,
                "lat": 31.82,
            },
            5: {
                "code": "SHA",
                "name": "Shanghai",
                "type": "Hub",
                "lon": 121.47,
                "lat": 31.23,
            },
            12: {
                "code": "NKG",
                "name": "Nanjing",
                "type": "Hub",
                "lon": 118.80,
                "lat": 32.06,
            },
            14: {
                "code": "HGH",
                "name": "Hangzhou",
                "type": "Hub",
                "lon": 120.12,
                "lat": 30.22,
            },
            29: {
                "code": "CTU",
                "name": "Chengdu",
                "type": "Hub",
                "lon": 104.07,
                "lat": 30.57,
            },
            30: {
                "code": "CKG",
                "name": "Chongqing",
                "type": "Hub",
                "lon": 106.55,
                "lat": 29.56,
            },
            31: {
                "code": "NCH",
                "name": "Nanchang",
                "type": "Hub",
                "lon": 115.86,
                "lat": 28.68,
            },
            33: {
                "code": "KWE",
                "name": "Guiyang",
                "type": "Hub",
                "lon": 106.63,
                "lat": 26.65,
            },
            38: {
                "code": "GZU",
                "name": "Guangzhou",
                "type": "Hub",
                "lon": 113.26,
                "lat": 23.13,
            },
            39: {
                "code": "NNG",
                "name": "Nanning",
                "type": "Hub",
                "lon": 108.37,
                "lat": 22.82,
            },
            44: {
                "code": "CSX",
                "name": "Changsha",
                "type": "Hub",
                "lon": 112.94,
                "lat": 28.23,
            },
            45: {
                "code": "FOC",
                "name": "Fuzhou",
                "type": "Hub",
                "lon": 119.30,
                "lat": 26.08,
            },

            # ── 10 Spokes ──
            6: {
                "code": "KHG",
                "name": "Kashgar",
                "type": "Spoke",
                "lon": 75.99,
                "lat": 39.47,
            },
            7: {
                "code": "HMI",
                "name": "Hami",
                "type": "Spoke",
                "lon": 93.52,
                "lat": 42.82,
            },
            8: {
                "code": "KRL",
                "name": "Korla",
                "type": "Spoke",
                "lon": 86.17,
                "lat": 41.73,
            },
            9: {
                "code": "AKU",
                "name": "Aksu",
                "type": "Spoke",
                "lon": 80.27,
                "lat": 41.17,
            },
            10: {
                "code": "HTN",
                "name": "Hotan",
                "type": "Spoke",
                "lon": 79.92,
                "lat": 37.11,
            },
            13: {
                "code": "SZV",
                "name": "Suzhou",
                "type": "Spoke",
                "lon": 120.59,
                "lat": 31.30,
            },
            41: {
                "code": "NTG",
                "name": "Nantong",
                "type": "Spoke",
                "lon": 120.86,
                "lat": 32.01,
            },
            47: {
                "code": "WUX",
                "name": "Wuxi",
                "type": "Spoke",
                "lon": 120.30,
                "lat": 31.57,
            },
            37: {
                "code": "SZX",
                "name": "Shenzhen",
                "type": "Spoke",
                "lon": 114.07,
                "lat": 22.55,
            },
            42: {
                "code": "HUZ",
                "name": "Huizhou",
                "type": "Spoke",
                "lon": 114.42,
                "lat": 23.11,
            },
        }

        hub_ids = [k for k, v in node_defs.items() if v["type"] == "Hub"]
        reposition_hub_by_spoke = {
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1,
            13: 12, 41: 12, 47: 12,
            37: 38, 42: 38,
        }

        nodes = [
            {
                "id": k,
                "x": v["lon"],
                "y": v["lat"],
                "name": v["name"],
                "reposition_hub_id": reposition_hub_by_spoke.get(k, k if k in (1, 12, 38) else None),
            }
            for k, v in node_defs.items()
        ]

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

        # ── Road edges: each spoke connects to its provincial dispatch capital ──
        road_edges = []

        # Spoke-to-hub: each spoke connects to its 1–2 nearest hubs by road
        # Each spoke MUST connect to its provincial-capital Hub
        spoke_dispatch_hubs = {
            # Xinjiang origins → Urumqi (sole rail entry)
            6:  [1],       # Kashgar → Urumqi
            7:  [1],       # Hami → Urumqi
            8:  [1],       # Korla → Urumqi
            9:  [1],       # Aksu → Urumqi
            10: [1],       # Hotan → Urumqi
            # Yangtze River Delta destinations
            13: [12],      # Suzhou → Nanjing
            41: [12],      # Nantong → Nanjing
            47: [12],      # Wuxi → Nanjing
            # Pearl River Delta destinations
            37: [38],      # Shenzhen → Guangzhou
            42: [38],      # Huizhou → Guangzhou
        }
        for spoke, hubs in spoke_dispatch_hubs.items():
            for h in hubs:
                road_edges.append((spoke, h))

        
        arcs = []
        arc_id = 1
        
        # Construct Rail Arcs
        for u, v in rail_edges:
            dist = haversine_distance(node_defs[u]["lon"], node_defs[u]["lat"], node_defs[v]["lon"], node_defs[v]["lat"])
            base_time = dist / self.config.rail_speed
            
            unit_cost_rail = 0.0
            
            rail_ratios = getattr(self.config, 'rail_fuzzy_time_ratio', (0.9, 1.0, 1.3))
            
            for src, dst in [(u, v), (v, u)]:
                arcs.append({
                    "id": arc_id, "i": src, "j": dst, "mode": 2, "distance": round(float(dist), 2),
                    "time_a": round(float(base_time * rail_ratios[0]), 2),
                    "time_b": round(float(base_time * rail_ratios[1]), 2),
                    "time_c": round(float(base_time * rail_ratios[2]), 2),
                    "alpha": arc_belief(self.config.seed, arc_id, src, dst, 2,
                        self.config.road_accident_belief_bounds, self.config.rail_accident_belief_bounds),
                    "unit_cost": round(float(unit_cost_rail), 2),
                })
                arc_id += 1

        # Construct Road Arcs
        for u, v in road_edges:
            dist = max(1.0, haversine_distance(node_defs[u]["lon"], node_defs[u]["lat"], node_defs[v]["lon"], node_defs[v]["lat"]))
            base_time = dist / self.config.road_speed
            road_ratios = getattr(self.config, 'road_fuzzy_time_ratio', (0.9, 1.0, 1.5))
            
            for src, dst in [(u, v), (v, u)]:
                arcs.append({
                    "id": arc_id, "i": src, "j": dst, "mode": 1, "distance": round(float(dist), 2),
                    "time_a": round(float(base_time * road_ratios[0]), 2),
                    "time_b": round(float(base_time * road_ratios[1]), 2),
                    "time_c": round(float(base_time * road_ratios[2]), 2),
                    "alpha": arc_belief(self.config.seed, arc_id, src, dst, 1,
                        self.config.road_accident_belief_bounds, self.config.rail_accident_belief_bounds),
                    "unit_cost": round(float(self.config.road_loaded_cost(dist)), 2),
                })
                arc_id += 1

        # 3. Transfer facilities at Hub nodes
        transfer_nodes = []
        for hub in hub_ids:
            transfer_nodes.append({
                "node_id": hub,
                "gamma_rr": draw_belief(self.config.seed, f"transfer:{hub}:rr", self.config.transfer_accident_belief_rail_rail_bounds),
                "gamma_ir": draw_belief(self.config.seed, f"transfer:{hub}:ir", self.config.transfer_accident_belief_intermodal_bounds),
                "unit_transfer_cost": float(self.config.c_transfer_road_rail),
            })

        # 4. Generate Specific OD Pairs (West to East)
        # All OD pairs route through the hub chain: origin→(road)→Urumqi→(rail)→destination hub→(road)→destination
        # origin_hub=1 (Urumqi) is the network entry hub
        # destination_hub depends on which hub the destination spoke connects to
        tasks = [
            {"o": 6,  "d": 13, "dh": 12,  "demand": 20.0},
            {"o": 10, "d": 41, "dh": 12,  "demand": 10.0},
            {"o": 9,  "d": 47, "dh": 12,  "demand": 12.0},
            {"o": 8,  "d": 37, "dh": 38, "demand": 18.0},
            {"o": 7,  "d": 42, "dh": 38, "demand": 14.0},
        ]

        od_pairs = []
        max_required_time = 0.0

        for idx, task in enumerate(tasks):
            o, d, dh = task["o"], task["d"], task["dh"]
            demand = task["demand"]

            # --- Compute total travel time (road1 + rail + road2) for this OD ---
            t_road_1 = haversine_distance(node_defs[o]["lon"], node_defs[o]["lat"],
                                          node_defs[1]["lon"], node_defs[1]["lat"]) / self.config.road_speed
            t_rail = 0.0
            rail_chains = {
                12: [1, 2, 21, 3, 12],
                38: [1, 2, 29, 30, 33, 38],
            }
            chain = rail_chains.get(dh, [2, 29, 30, 39, 38, 31, 14, 5])
            for k in range(len(chain) - 1):
                t_rail += haversine_distance(node_defs[chain[k]]["lon"], node_defs[chain[k]]["lat"],
                                             node_defs[chain[k+1]]["lon"], node_defs[chain[k+1]]["lat"]) / self.config.rail_speed
            t_road_2 = haversine_distance(node_defs[dh]["lon"], node_defs[dh]["lat"],
                                          node_defs[d]["lon"], node_defs[d]["lat"]) / self.config.road_speed
            # Physical travel time (road + rail only)
            t_travel_raw = t_road_1 + t_rail + t_road_2
            # 真实案例统一允许从规划期起点发运，并在配置的规划期内完成。
            time_window_start = 0.0
            time_window_end = float(self.config.T_max)

            od_pairs.append({
                "id": idx + 1,
                "origin": o, "destination": d, "demand": demand,
                "time_window_start": round(time_window_start, 1),
                "time_window_end": round(time_window_end, 1),
                "time_window_start_fmt": float_to_day_time(time_window_start),
                "time_window_end_fmt": float_to_day_time(time_window_end),
                "alpha_T_od": float(self.config.alpha_T),
                "origin_hub": 1,
                "destination_hub": dh,
                "dist_road_1": round(haversine_distance(node_defs[o]["lon"], node_defs[o]["lat"],
                                                       node_defs[1]["lon"], node_defs[1]["lat"]), 2),
                "dist_rail": round(sum(
                    haversine_distance(node_defs[chain[k]]["lon"], node_defs[chain[k]]["lat"],
                                       node_defs[chain[k+1]]["lon"], node_defs[chain[k+1]]["lat"])
                    for k in range(len(chain) - 1)
                ), 2),
                "dist_road_2": round(haversine_distance(node_defs[dh]["lon"], node_defs[dh]["lat"],
                                                       node_defs[d]["lon"], node_defs[d]["lat"]), 2),
            })
            
        # 5. Railway Timetable
        # 每条铁路弧的两个方向每天均在 10:00 和 18:00 各发一班。
        railway_timetable = []
        train_id_counter = 1
        
        planning_days = int(math.ceil(float(self.config.T_max) / 24.0))
        for day in range(1, planning_days + 1):
            for u, v in rail_edges:
                dist = haversine_distance(node_defs[u]["lon"], node_defs[u]["lat"], node_defs[v]["lon"], node_defs[v]["lat"])
                service_time = dist / self.config.rail_speed + self.config.rail_intermediate_stop_hours
                for departure_hour in (10.0, 18.0):
                    departure_time = (day - 1) * 24 + departure_hour
                    arrival_time = departure_time + service_time
                    for origin_hub, destination_hub in ((u, v), (v, u)):
                        railway_timetable.append({
                            "train_id": train_id_counter,
                            "origin_hub": origin_hub,
                            "destination_hub": destination_hub,
                            "departure_time": round(departure_time, 2),
                            "arrival_time": round(arrival_time, 2),
                            "dep_fmt": float_to_day_time(departure_time),
                            "arr_fmt": float_to_day_time(arrival_time),
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
            "hub_fleet_size": default_empty_reposition_hub_fleet(),
            "config": {
                "T_max": self.config.T_max,
                "seed": self.config.seed,
                "alpha_R": self.config.alpha_R,
                "alpha_T": self.config.alpha_T,
                "data_version": self.config.data_version,
            }
        }
        return instance_data
