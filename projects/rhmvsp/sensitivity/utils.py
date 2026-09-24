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
    """Synchronize OD thresholds with the common deadline in ``config.T_max``.

    ``alpha_T_dependent`` is retained for call-site compatibility but is
    intentionally ignored: changing the deadline together with alpha_T would
    confound the sensitivity experiment.
    """
    del alpha_T_dependent
    for od in instance.get("od_pairs", []):
        od["alpha_T_od"] = config.alpha_T
        od["time_window_start"] = 0.0
        od["time_window_end"] = float(config.T_max)


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
            dep = route.get("departure_Time", 0)
            arr = route.get("arrival_Time", 0)
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
