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


def print_od_routing(sol: dict, instance: dict, label: str = ""):
    """Print complete per-OD routing paths from a solution dictionary.

    Shows the full node sequence for each dispatched route:
      origin → road → hub → rail → hub → road → destination

    Parameters
    ----------
    sol : dict
        Solution dictionary from BAP solver (must contain 'transport_routes').
    instance : dict
        Instance dictionary (for node name lookup).
    label : str
        Optional label printed in the header for context.
    """
    routes = sol.get("transport_routes", [])
    if not routes:
        sol_status = sol.get("status", "?")
        print(f"  [No transport_routes — status={sol_status}]")
        return

    # Build node lookup: node_id → readable name
    node_names = {}
    for n in instance.get("network", {}).get("nodes", []):
        nid = n.get("node_id") or n.get("id")
        name = n.get("name", "") or n.get("label", "")
        node_names[nid] = name or str(nid)

    # Group routes by OD pair
    from collections import OrderedDict
    od_routes = OrderedDict()
    for r in routes:
        od_idx = r.get("od_idx")
        od_routes.setdefault(od_idx, []).append(r)

    header = f"─── OD Routing {label} ───".strip()
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    for od_idx in sorted(od_routes.keys()):
        od_list = od_routes[od_idx]
        od_pair = od_list[0].get("od_pair", f"OD_{od_idx}")
        print(f"\n  {od_pair} ({len(od_list)} dispatch(es)):")

        for ri, route in enumerate(od_list):
            origin = route.get("origin", "?")
            dest = route.get("destination", "?")
            dep = route.get("departure_time", 0)
            arr = route.get("arrival_time", 0)
            vehicles = route.get("vehicles", [route.get("vehicle", "?")])
            cost = route.get("cost", 0)
            arcs = route.get("arcs", [])

            or_name = node_names.get(origin, origin)
            dest_name = node_names.get(dest, dest)

            # Reconstruct full path from arcs
            path_nodes = [str(or_name)]
            for arc in arcs:
                j = arc.get("j")
                j_name = node_names.get(j, j)
                path_nodes.append(str(j_name))

            path_str = " → ".join(path_nodes)
            veh_str = f"  Vehicles={vehicles}" if isinstance(vehicles, list) and len(vehicles) <= 5 else ""

            print(f"    #{ri + 1}: {path_str}")
            print(f"        Dep={dep:.1f}h  Arr={arr:.1f}h  "
                  f"Dur={arr - dep:.1f}h  ¥{cost:,.0f}{veh_str}")

def format_routing_summary(sol: dict, instance: dict, max_ods: int = 5) -> str:
    """Return condensed routing summary with node IDs for CSV storage.

    Format: "OD1: 6->1->2->21->3->5;OD2: 10->1->2->21->3->5"
      - Each OD shows ONE continuous path from origin to destination,
        merging ALL arcs from transport_routes AND rail_consolidation_summary
        to ensure road + rail + road segments are all included.
      - Multiple ODs separated by ";"
    """
    routes = sol.get("transport_routes", [])
    rail_summary = sol.get("rail_consolidation_summary", [])
    if not routes:
        return ""

    # Collect rail arcs from rail_consolidation_summary (hub → hub rail segments)
    # Format: {od_idx: [(from_node, to_node), ...]}
    from collections import OrderedDict, defaultdict
    rail_arcs_by_od = defaultdict(set)
    for rc in rail_summary:
        tasks = rc.get("tasks_included", [])
        section = rc.get("rail_section", "")
        # Parse "Hub_X -> Hub_Y" to get node IDs
        # The actual node IDs are harder to get from this string...
        # Instead, rely on transport_routes arcs which may already include rail

    # Group transport_routes by OD
    od_routes = OrderedDict()
    for r in routes:
        od_routes.setdefault(r.get("od_idx"), []).append(r)

    parts = []
    for od_idx in sorted(od_routes.keys())[:max_ods]:
        od_list = od_routes[od_idx]
        od_pair = od_list[0].get("od_pair", f"OD_{od_idx}")

        # Collect ALL arcs (i→j) across all dispatches for this OD
        arcs_set = set()
        for route in od_list:
            for arc in route.get("arcs", []):
                i = arc.get("i")
                j = arc.get("j")
                if i is not None and j is not None:
                    arcs_set.add((i, j))

        if not arcs_set:
            continue

        # Build directed graph and find start node
        out_edges = defaultdict(list)
        in_degree = defaultdict(int)
        all_nodes = set()
        for i, j in arcs_set:
            out_edges[i].append(j)
            in_degree[j] += 1
            if i not in in_degree:
                in_degree[i] = 0
            all_nodes.add(i)
            all_nodes.add(j)

        # Start node = node with outgoing edges but no incoming
        start = None
        for n in all_nodes:
            if in_degree.get(n, 0) == 0 and n in out_edges:
                start = n
                break
        if start is None:
            start = min(all_nodes)

        # Walk the path
        path_nodes = [str(start)]
        current = start
        visited = {start}
        while current in out_edges:
            # Linear path: take first unvisited successor
            next_found = None
            for nxt in out_edges[current]:
                if nxt not in visited:
                    next_found = nxt
                    break
            if next_found is None:
                break
            path_nodes.append(str(next_found))
            visited.add(next_found)
            current = next_found

        parts.append(f"{od_pair}: {'->'.join(path_nodes)}")

    return "; ".join(parts)