"""
Reconstructed network_builder.py for RHMVSP project.

Based on:
- S3 docs: network_requirements_full.md, network_data_structure.md
- Network generation report: generate_all_networks_report.md
- Full project recovery notes: FULLPROJECT_RECOVERY_GUIDE.md
- Original source analysis: network_builder_original.py

Key design notes:
    - Roads connect O/D to transfer nodes (hubs)
    - Rail connects transfer nodes (hubs) to transfer nodes (hubs)
    - Arcs use (i, j, mode) as key: mode 1=road, mode 2=rail
    - Hub-to-hub direct rail is not allowed (must be hub->hub)
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from .model_types import Arc, Hub, Network, Node

logger = logging.getLogger(__name__)

# Constants for network generation
ROAD_SPEED_AVG = 60.0   # km/h average road speed
RAIL_SPEED_AVG = 80.0   # km/h average rail speed
EARTH_RADIUS = 6371.0   # km


def _haversine_km(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Calculate haversine distance in km between two points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS * math.asin(math.sqrt(a))


class NetworkBuilder:
    """
    Build a road-rail intermodal network from node/arc CSV files.

    The builder reads pre-generated CSV data, constructs domain objects
    (Node, Arc, Hub), and provides helper queries for solver use.
    """

    def __init__(self, data_dir: str | Path, rng: random.Random | None = None):
        self.data_dir = Path(data_dir)
        self.rng = rng or random.Random()

        self.nodes: dict[int, Node] = {}
        self.arcs: dict[tuple[int, int, int], Arc] = {}
        self.hubs: dict[int, Hub] = {}

        self._load_nodes()
        self._load_arcs()
        self._classify_hubs()

    # ------------------------------------------------------------------ loading

    def _load_nodes(self) -> None:
        """Load nodes from CSV and create Node objects."""
        nodes_file = self.data_dir / "nodes.csv"
        if not nodes_file.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_file}")

        with open(nodes_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node = Node(
                    id=int(row["node_id"]),
                    name=row.get("name", f"node_{row['node_id']}"),
                    lon=float(row["longitude"]),
                    lat=float(row["latitude"]),
                    population=int(float(row.get("population", 0))),
                    node_type=row.get("node_type", "general"),
                )
                self.nodes[node.id] = node

        logger.info(f"Loaded {len(self.nodes)} nodes")

    def _load_arcs(self) -> None:
        """Load arcs from CSV and create Arc objects."""
        arcs_file = self.data_dir / "arcs.csv"
        if not arcs_file.exists():
            raise FileNotFoundError(f"Arcs file not found: {arcs_file}")

        with open(arcs_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                i_id = int(row["i"])
                j_id = int(row["j"])
                mode = int(row["mode"])

                distance = float(row.get("distance", 0.0))
                if distance <= 0 and i_id in self.nodes and j_id in self.nodes:
                    ni, nj = self.nodes[i_id], self.nodes[j_id]
                    distance = _haversine_km(ni.lon, ni.lat, nj.lon, nj.lat)

                speed = ROAD_SPEED_AVG if mode == 1 else RAIL_SPEED_AVG
                base_time = distance / speed if speed > 0 else 0.0

                arc = Arc(
                    i=i_id,
                    j=j_id,
                    mode=mode,
                    distance=distance,
                    base_time=base_time,
                    risk_weight=float(row.get("risk_weight", 1.0)),
                )
                self.arcs[(i_id, j_id, mode)] = arc

        logger.info(f"Loaded {len(self.arcs)} arcs")

    def _classify_hubs(self) -> None:
        """Identify transfer hubs from nodes with rail connections."""
        rail_nodes = set()
        for (i, j, mode) in self.arcs:
            if mode == 2:
                rail_nodes.add(i)
                rail_nodes.add(j)

        for nid in rail_nodes:
            if nid in self.nodes:
                self.hubs[nid] = Hub(
                    node_id=nid,
                    node=self.nodes[nid],
                    capacity=float("inf"),
                )

        logger.info(f"Identified {len(self.hubs)} transfer hubs")

    # ----------------------------------------------------------------- queries

    def get_arcs_from(self, node_id: int, mode: int | None = None) -> list[Arc]:
        """Get all arcs originating from a node, optionally filtered by mode."""
        return [
            arc for (i, j, m), arc in self.arcs.items()
            if i == node_id and (mode is None or m == mode)
        ]

    def get_arcs_to(self, node_id: int, mode: int | None = None) -> list[Arc]:
        """Get all arcs terminating at a node, optionally filtered by mode."""
        return [
            arc for (i, j, m), arc in self.arcs.items()
            if j == node_id and (mode is None or m == mode)
        ]

    def get_road_arcs(self) -> list[Arc]:
        """Get all road arcs (mode=1)."""
        return [arc for arc in self.arcs.values() if arc.mode == 1]

    def get_rail_arcs(self) -> list[Arc]:
        """Get all rail arcs (mode=2)."""
        return [arc for arc in self.arcs.values() if arc.mode == 2]

    def get_od_pairs(self) -> list[dict]:
        """Load OD pairs from CSV if available."""
        od_file = self.data_dir / "od_pairs.csv"
        if not od_file.exists():
            return []

        od_pairs = []
        with open(od_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                od_pairs.append({
                    "origin": int(row["origin"]),
                    "destination": int(row["destination"]),
                    "demand": float(row.get("demand", 0.0)),
                })
        return od_pairs

    def get_timetable(self) -> list[dict]:
        """Load railway timetable from CSV if available."""
        tt_file = self.data_dir / "railway_timetable.csv"
        if not tt_file.exists():
            return []

        timetable = []
        with open(tt_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timetable.append(dict(row))
        return timetable

    # -------------------------------------------------------- export / summary

    def summary(self) -> dict:
        """Return a summary dict of the network."""
        return {
            "num_nodes": len(self.nodes),
            "num_arcs": len(self.arcs),
            "num_hubs": len(self.hubs),
            "road_arcs": len(self.get_road_arcs()),
            "rail_arcs": len(self.get_rail_arcs()),
            "total_road_distance_km": sum(
                a.distance for a in self.get_road_arcs()
            ),
            "total_rail_distance_km": sum(
                a.distance for a in self.get_rail_arcs()
            ),
        }

    def to_dict(self) -> dict:
        """Export network as a plain dict (JSON-serializable)."""
        return {
            "nodes": [
                {
                    "node_id": n.id,
                    "name": n.name,
                    "longitude": n.lon,
                    "latitude": n.lat,
                    "population": n.population,
                    "node_type": n.node_type,
                }
                for n in self.nodes.values()
            ],
            "arcs": [
                {
                    "i": a.i,
                    "j": a.j,
                    "mode": a.mode,
                    "distance": a.distance,
                    "base_time": a.base_time,
                    "risk_weight": a.risk_weight,
                }
                for a in self.arcs.values()
            ],
            "hubs": [
                {
                    "node_id": h.node_id,
                    "capacity": h.capacity,
                }
                for h in self.hubs.values()
            ],
        }
