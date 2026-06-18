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
        node_defs = {
            # 5 Hubs
            1: {"code": "URC", "name": "Urumqi", "type": "Hub", "lon": 87.62, "lat": 43.83, "region": "West"},
            2: {"code": "LHW", "name": "Lanzhou", "type": "Hub", "lon": 103.83, "lat": 36.06, "region": "West"},
            3: {"code": "CGO", "name": "Zhengzhou", "type": "Hub", "lon": 113.66, "lat": 34.75, "region": "Mid"},
            4: {"code": "HFE", "name": "Hefei", "type": "Hub", "lon": 117.23, "lat": 31.82, "region": "East"},
            5: {"code": "SHA", "name": "Shanghai", "type": "Hub", "lon": 121.47, "lat": 31.23, "region": "East"},
            
            # 5 Specific West Spokes (Xinjiang)
            6: {"code": "KHG", "name": "Kashgar", "type": "Spoke", "lon": 75.99, "lat": 39.47, "region": "West"},
            7: {"code": "HMI", "name": "Hami", "type": "Spoke", "lon": 93.52, "lat": 42.82, "region": "West"},
            8: {"code": "KRL", "name": "Korla", "type": "Spoke", "lon": 86.17, "lat": 41.73, "region": "West"},
            9: {"code": "AKU", "name": "Aksu", "type": "Spoke", "lon": 80.27, "lat": 41.17, "region": "West"},
            10: {"code": "HTN", "name": "Hotan", "type": "Spoke", "lon": 79.92, "lat": 37.11, "region": "West"},
            
            # 5 Specific East Spokes
            11: {"code": "XUZ", "name": "Xuzhou", "type": "Spoke", "lon": 117.28, "lat": 34.20, "region": "East"},
            12: {"code": "NKG", "name": "Nanjing", "type": "Spoke", "lon": 118.80, "lat": 32.06, "region": "East"},
            13: {"code": "SZV", "name": "Suzhou", "type": "Spoke", "lon": 120.59, "lat": 31.30, "region": "East"},
            14: {"code": "HGH", "name": "Hangzhou", "type": "Spoke", "lon": 120.12, "lat": 30.22, "region": "East"},
            15: {"code": "FOC", "name": "Fuzhou", "type": "Spoke", "lon": 119.30, "lat": 26.07, "region": "East"},
        }
        
        # Add 2 West dummy spokes and 3 East dummy spokes
        for i in range(16, 18):
            node_defs[i] = {
                "code": f"WD_{i}", "name": f"West_Dummy_{i}", "type": "Spoke",
                "lon": node_defs[1]["lon"] + self.rng.uniform(-3, 3),
                "lat": node_defs[1]["lat"] + self.rng.uniform(-3, 3),
                "region": "West"
            }
        for i in range(18, 21):
            node_defs[i] = {
                "code": f"ED_{i}", "name": f"East_Dummy_{i}", "type": "Spoke",
                "lon": node_defs[5]["lon"] + self.rng.uniform(-3, 3),
                "lat": node_defs[5]["lat"] + self.rng.uniform(-3, 3),
                "region": "East"
            }
        hub_ids = [k for k, v in node_defs.items() if v["type"] == "Hub"]
        spoke_ids = [k for k, v in node_defs.items() if v["type"] == "Spoke"]
        
        nodes = [{"id": k, "x": round(v["lon"], 2), "y": round(v["lat"], 2), "name": v["name"]} for k, v in node_defs.items()]
        
        # Regional Hub logic
        rail_edges = []
        west_hubs = [k for k, v in node_defs.items() if v["region"] == "West" and v["type"] == "Hub"]
        mid_hubs = [k for k, v in node_defs.items() if v["region"] == "Mid" and v["type"] == "Hub"]
        east_hubs = [k for k, v in node_defs.items() if v["region"] == "East" and v["type"] == "Hub"]

        # 1. West hubs fully connected
        for i in range(len(west_hubs)):
            for j in range(i+1, len(west_hubs)):
                rail_edges.append((west_hubs[i], west_hubs[j]))

        # 2. Mid hubs fully connected
        for i in range(len(mid_hubs)):
            for j in range(i+1, len(mid_hubs)):
                rail_edges.append((mid_hubs[i], mid_hubs[j]))

        # 3. East hubs fully connected
        for i in range(len(east_hubs)):
            for j in range(i+1, len(east_hubs)):
                rail_edges.append((east_hubs[i], east_hubs[j]))

        # 4. West to Mid connected
        for w in west_hubs:
            for m in mid_hubs:
                rail_edges.append((w, m))

        # 5. Mid to East connected
        for m in mid_hubs:
            for e in east_hubs:
                rail_edges.append((m, e))
        
        road_edges = []
        
        # Region separation for Road
        west_spokes = [k for k, v in node_defs.items() if v["region"] == "West" and v["type"] == "Spoke"] # 20 spokes
        east_spokes = [k for k, v in node_defs.items() if v["region"] == "East" and v["type"] == "Spoke"] # 20 spokes
        west_hubs = [k for k, v in node_defs.items() if v["region"] == "West" and v["type"] == "Hub"] # 1, 2
        east_hubs = [k for k, v in node_defs.items() if v["region"] == "East" and v["type"] == "Hub"] # 9, 10
        
        # Connect West Spokes to West Hubs
        for s in west_spokes:
            for h in west_hubs:
                road_edges.append((s, h))
        
        # Connect East Spokes to East Hubs
        for s in east_spokes:
            for h in east_hubs:
                road_edges.append((s, h))
                
        # Inter-spoke road connections (West to West, East to East) to ensure robust routing and repositioning
        west_spoke_edges = []
        for i in range(len(west_spokes)):
            for j in range(i+1, len(west_spokes)):
                west_spoke_edges.append((west_spokes[i], west_spokes[j]))
        self.rng.shuffle(west_spoke_edges)
        road_edges.extend(west_spoke_edges) # Dense mesh in West
        
        east_spoke_edges = []
        for i in range(len(east_spokes)):
            for j in range(i+1, len(east_spokes)):
                east_spoke_edges.append((east_spokes[i], east_spokes[j]))
        self.rng.shuffle(east_spoke_edges)
        road_edges.extend(east_spoke_edges) # Dense mesh in East
        
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
        # Keep exactly the same task logic, updated to match new IDs
        # Previous origins: 6: Kashgar, 10: Hotan, 9: Aksu, 8: Korla, 7: Hami.
        # Previous destinations: 15: Fuzhou, 14: Hangzhou, 13: Suzhou, 12: Nanjing, 11: Xuzhou.
        # New origins: 11: Kashgar, 15: Hotan, 14: Aksu, 13: Korla, 12: Hami
        # New destinations: 20: Fuzhou, 19: Hangzhou, 18: Suzhou, 17: Nanjing, 16: Xuzhou
        
        tasks = [
            {"o": 6, "d": 15, "multiplier": 150.0 / self.config.q_rail},  # OD 1
            {"o": 10, "d": 14, "multiplier": 240.0 / self.config.q_rail}, # OD 2
            {"o": 9, "d": 13, "multiplier": 360.0 / self.config.q_rail}, # OD 3
            {"o": 8, "d": 12, "multiplier": 510.0 / self.config.q_rail}, # OD 4
            {"o": 7, "d": 11, "multiplier": 690.0 / self.config.q_rail}, # OD 5
        ]
        
        od_pairs = []
        max_required_time = 0.0
        
        # Update T_max globally for this instance to 28 days (672 hours)
        self.config.T_max = 672.0
        
        for idx, task in enumerate(tasks):
            o, d = task["o"], task["d"]
            demand = float(round(task["multiplier"] * self.config.q_rail, 0))
            
            est_dist = haversine_distance(node_defs[o]["lon"], node_defs[o]["lat"], node_defs[d]["lon"], node_defs[d]["lat"])
            est_trip_time = est_dist / 60.0 
            
            time_window_start = float(round(10.0 + idx * 20.0 + self.rng.uniform(-5, 5), 1)) # Staggered starts
            
            # Geographical distance-based deadline
            # Route: Origin(o) -> Hub(1) -> Hub(5) -> Destination(d)
            t_road_1 = haversine_distance(node_defs[o]["lon"], node_defs[o]["lat"], node_defs[1]["lon"], node_defs[1]["lat"]) / self.config.road_speed
            t_rail = haversine_distance(node_defs[1]["lon"], node_defs[1]["lat"], node_defs[5]["lon"], node_defs[5]["lat"]) / self.config.rail_speed
            t_road_2 = haversine_distance(node_defs[5]["lon"], node_defs[5]["lat"], node_defs[d]["lon"], node_defs[d]["lat"]) / self.config.road_speed
            
            # We add 20 hours buffer for waiting and transfers
            t_expected = time_window_start + t_road_1 + t_rail + t_road_2 + 20.0
            
            # To ensure time_reliability is not exactly 1.0 but around 0.3-0.8, we don't give a massive buffer
            # We just add a 12 hour buffer to the expected time.
            time_window_end = float(round(t_expected + 12.0, 1))
            
            max_required_time = max(max_required_time, time_window_end)
            
            od_pairs.append({
                "id": idx + 1,
                "origin": o, "destination": d, "demand": demand,
                "time_window_start": round(time_window_start, 1), "time_window_end": round(time_window_end, 1),
                "time_window_start_fmt": float_to_day_time(time_window_start), "time_window_end_fmt": float_to_day_time(time_window_end),
                "alpha_T_od": float(self.config.alpha_T),
                "origin_hub": 1,
                "destination_hub": 5,
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
