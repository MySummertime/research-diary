"""Shared utilities for sensitivity experiments."""
from __future__ import annotations
import json
import math
from collections import deque
from typing import Any

from src.core.uncertainty import UncertaintyEngine as UE2, TrapezoidalFuzzy as TrapFz


def _haversine_distance(lon1, lat1, lon2, lat2):
    """Great-circle distance in km."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c


def update_od_deadlines(instance: dict, config: Any, alpha_T_dependent: bool = False) -> None:
    """Recompute L_od for each OD pair based on current alpha_T and delta_T.

    Uses physical travel time + 60h overhead (same formula as real_data_generator.py).
    Mutates instance["od_pairs"] in place (updates alpha_T_od and time_window_end).
    """
    _arcs = instance["network"]["arcs"]
    _hub_ids = {tn["node_id"] for tn in instance["network"].get("transfer_nodes", [])}
    _rail_graph: dict[int, list] = {}
    for a in _arcs:
        if a["mode"] == 2:
            _rail_graph.setdefault(a["i"], []).append(a)
    _node_coords = {n["id"]: (n["x"], n["y"]) for n in instance["network"]["nodes"]}

    def _bfs_rail(start: int, end: int):
        visited = {start: []}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for arc in _rail_graph.get(node, []):
                if arc["j"] not in visited:
                    visited[arc["j"]] = visited[node] + [arc]
                    if arc["j"] == end:
                        return visited[arc["j"]]
                    queue.append(arc["j"])
        return None

    # Rail chain lookup (same as real_data_generator.py)
    rail_chains = {
        5:  [2, 21, 3, 22, 4, 5],       # Lanzhou → ... → Shanghai
        38: [2, 29, 30, 33, 38],          # Lanzhou → ... → Guangzhou
    }

    for od in instance.get("od_pairs", []):
        od["alpha_T_od"] = config.alpha_T
        od_o, od_d = od["origin"], od["destination"]
        roads_out = [a for a in _arcs if a["i"] == od_o and a["mode"] == 1 and a["j"] in _hub_ids]
        roads_in  = [a for a in _arcs if a["j"] == od_d and a["mode"] == 1 and a["i"] in _hub_ids]

        if roads_out and roads_in:
            # Use BFS to find rail path from entry hub to destination hub
            co = min(roads_out, key=lambda a: a["distance"])
            ci = min(roads_in,  key=lambda a: a["distance"])
            rail_path = _bfs_rail(co["j"], ci["i"])

            if rail_path:
                # --- Time window overhead ---
                if alpha_T_dependent:
                    # α_T-dependent: tighter α_T → tighter time window → binding time reliability.
                    _overhead_min = 40.0
                    _overhead_max = 60.0
                    _ratio = (1.0 - config.alpha_T) ** 0.5
                    _overhead = _overhead_min + _ratio * (_overhead_max - _overhead_min)
                else:
                    # Fixed overhead for experiments that don't sweep α_T
                    _overhead = 60.0

                # Physical travel time: road1 + rail + road2
                t_road_1 = co["distance"] / config.road_speed
                t_rail = sum(arc["distance"] / config.rail_speed for arc in rail_path)
                t_road_2 = ci["distance"] / config.road_speed
                t_travel_raw = t_road_1 + t_rail + t_road_2
                t_travel = t_travel_raw + _overhead

                start_time = od.get("time_window_start", 0.0)
                od["time_window_end"] = float(math.ceil((start_time + t_travel) / 12.0) * 12)
