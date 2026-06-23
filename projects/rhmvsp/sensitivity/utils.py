"""Shared utilities for sensitivity experiments."""
from __future__ import annotations
import json
from collections import deque
from typing import Any

from src.core.uncertainty import UncertaintyEngine as UE2, TrapezoidalFuzzy as TrapFz


def update_od_deadlines(instance: dict, config: Any) -> None:
    """Recompute L_od for each OD pair based on current alpha_T and delta_T.

    Uses BFS to find multi-hop rail paths and compute trapezoidal time.
    Mutates instance["od_pairs"] in place (updates alpha_T_od and time_window_end).
    """
    _temp_engine = UE2(config)
    _arcs = instance["network"]["arcs"]
    _hub_ids = {tn["node_id"] for tn in instance["network"].get("transfer_nodes", [])}
    _rail_graph: dict[int, list] = {}
    for a in _arcs:
        if a["mode"] == 2:
            _rail_graph.setdefault(a["i"], []).append(a)

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

    for od in instance.get("od_pairs", []):
        od["alpha_T_od"] = config.alpha_T
        od_o, od_d = od["origin"], od["destination"]
        roads_out = [a for a in _arcs if a["i"] == od_o and a["mode"] == 1 and a["j"] in _hub_ids]
        roads_in  = [a for a in _arcs if a["j"] == od_d and a["mode"] == 1 and a["i"] in _hub_ids]
        if roads_out and roads_in:
            co = min(roads_out, key=lambda a: a["distance"])
            ci = min(roads_in,  key=lambda a: a["distance"])
            rail_path = _bfs_rail(co["j"], ci["i"])
            if rail_path:
                r1 = _temp_engine.make_transport_time(co["distance"], 1)
                tf = _temp_engine.make_transfer_time()
                trap_a = r1.a + tf.a
                trap_b = r1.b + tf.b
                trap_c = r1.c + tf.c
                trap_d = r1.d + tf.d
                for ra_arc in rail_path:
                    ra = _temp_engine.make_transport_time(ra_arc["distance"], 2)
                    trap_a += ra.a; trap_b += ra.b; trap_c += ra.c; trap_d += ra.d
                tf2 = _temp_engine.make_transfer_time()
                trap_a += tf2.a; trap_b += tf2.b; trap_c += tf2.c; trap_d += tf2.d
                r2 = _temp_engine.make_transport_time(ci["distance"], 1)
                trap_a += r2.a; trap_b += r2.b; trap_c += r2.c; trap_d += r2.d
                trap = TrapFz(trap_a, trap_b, trap_c, trap_d)
                det_L = trap.deterministic_equivalent_upper(config.alpha_T)
                od["time_window_end"] = float(round(od.get("time_window_start", 0.0) + det_L * 1.10, 1))
