"""Multimodal Network Graph Model (§2.1, §3.1)."""

from __future__ import annotations
from dataclasses import dataclass

import networkx as nx


@dataclass
class Arc:
    """Represents an arc (i,j) in the multimodal network."""
    arc_id: int                     # Arc ID (1-based business ID)
    i: int                          # Start node
    j: int                          # End node
    mode: int                       # 1=road, 2=rail
    distance: float                 # Distance (km)
    # Fuzzy transport time parameters: trapezoidal (a, b, c, d)
    # Road: c=d (triangular equivalent); Rail: b<=c plateau
    time_a: float = 0.0
    time_b: float = 0.0
    time_c: float = 0.0
    time_d: float = 0.0
    # Risk parameters
    alpha: float = 0.01             # Accident belief (time-independent default)
    C_consequence: float = 100.0    # Accident consequence
    # Operational time window (mainly for rail)
    e_window: float = 0.0           # Earliest operational time
    l_window: float = 48.0          # Latest operational time
    # Capacity
    capacity: float = 100.0         # Per-vehicle capacity (tons)
    # Cost
    unit_cost: float = 0.5          # Transport cost per ton
    deploy_cost: float = 100.0      # Vehicle deployment cost
    min_units: int = 1              # Minimum transport units
    population_density: float = 1.0  # Population density multiplier for time-varying C (S2-2)

    @property
    def mode_name(self) -> str:
        return "road" if self.mode == 1 else "rail"

    @property
    def is_road(self) -> bool:
        return self.mode == 1

    @property
    def is_rail(self) -> bool:
        return self.mode == 2


@dataclass
class TransferNode:
    """A node where road-rail transfer is possible."""
    node_id: int
    # Fuzzy transfer time: trapezoidal (a, b, c, d)
    transfer_time_a: float = 0.5
    transfer_time_b: float = 1.0
    transfer_time_c: float = 2.0
    transfer_time_d: float = 4.0
    # Transfer risk
    gamma: float = 0.005            # Transfer accident belief
    # Cost
    unit_transfer_cost: float = 50.0
    facility_cost: float = 5000.0   # Fixed facility setup cost
    max_cranes: int = 2             # Max concurrent cranes per hub
    max_storage_tons: float = 500.0 # Max hazmat storage capacity per hub (tons)


@dataclass
class ODPair:
    """Origin-Destination pair with demand."""
    origin: int
    destination: int
    demand: float                   # Q_od (tons)
    E_od: float = 0.0              # Delivery time window lower bound
    L_od: float = 48.0             # Delivery time window upper bound
    alpha_T_od: float = 0.8        # Time reliability threshold


class MultimodalNetwork:
    """
    Road-rail multimodal transport network G = (N, A).

    Based on §2.1: Nodes = transport hubs, Arcs = transport segments.
    Each arc (i,j) has available modes M_ij ⊆ {1,2}.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.road_graph = nx.DiGraph()  # Subgraph strictly for road distance/routing
        self.arcs: dict[tuple[int, int, int], Arc] = {}  # (i,j,arc_id) -> Arc
        self.transfer_nodes: dict[int, TransferNode] = {}
        self.od_pairs: list[ODPair] = []
        self.ng_neighbors: dict[int, set[int]] = {}      # Node -> ng-route memory neighborhood
        self.timetable: dict[tuple[int, int], list[dict]] = {}  # (u, v) -> sorted list of train runs
        self._arcs_out: dict[int, list[Arc]] = {}
        self._arcs_in: dict[int, list[Arc]] = {}

    @classmethod
    def from_instance(cls, instance: dict) -> MultimodalNetwork:
        """Create network from a JSON instance dictionary."""
        net = cls()

        # Add nodes
        for node in instance["network"]["nodes"]:
            net.graph.add_node(node["id"], **node)
            net.road_graph.add_node(node["id"], **node)

        # Add arcs
        for arc_data in instance["network"]["arcs"]:
            arc = Arc(
                arc_id=arc_data.get("id", 0),
                i=arc_data["i"], j=arc_data["j"], mode=arc_data["mode"],
                distance=arc_data["distance"],
                time_a=arc_data.get("time_a", 0),
                time_b=arc_data.get("time_b", 0),
                time_c=arc_data.get("time_c", 0),
                time_d=arc_data.get("time_d", arc_data.get("time_c", 0)),  # #9: trapezoidal, default=c
                alpha=arc_data.get("alpha", 0.01),
                C_consequence=arc_data.get("C_consequence", 100.0),
                e_window=arc_data.get("e_window", 0.0),
                l_window=arc_data.get("l_window", 48.0),
                capacity=arc_data.get("capacity", 100.0),
                unit_cost=arc_data.get("unit_cost", 0.5),
                deploy_cost=arc_data.get("deploy_cost", 100.0),
                min_units=arc_data.get("min_units", 1),
                population_density=arc_data.get("population_density", 1.0),
            )
            net.add_arc(arc)

        # Populate timetable
        for t in instance.get("railway_timetable", []):
            pair = (t["origin_hub"], t["destination_hub"])
            if pair not in net.timetable:
                net.timetable[pair] = []
            net.timetable[pair].append(t)

        for pair in net.timetable:
            net.timetable[pair].sort(key=lambda x: x["departure_time"])

        # Transfer nodes
        for tn_data in instance["network"].get("transfer_nodes", []):
            tn = TransferNode(
                node_id=tn_data["node_id"],
                transfer_time_a=tn_data.get("transfer_time_a", 0.5),
                transfer_time_b=tn_data.get("transfer_time_b", 1.0),
                transfer_time_c=tn_data.get("transfer_time_c", 2.0),
                transfer_time_d=tn_data.get("transfer_time_d", 4.0),
                gamma=tn_data.get("gamma", 0.005),
                unit_transfer_cost=tn_data.get("unit_transfer_cost", 50.0),
                facility_cost=tn_data.get("facility_cost", 5000.0),
                max_cranes=tn_data.get("max_cranes", 2),
                max_storage_tons=tn_data.get("max_storage_tons", 500.0),
            )
            net.transfer_nodes[tn.node_id] = tn

        # O-D pairs
        for od_data in instance["od_pairs"]:
            od = ODPair(
                origin=od_data["origin"], destination=od_data["destination"],
                demand=od_data["demand"],
                E_od=od_data.get("E_od", 0.0),
                L_od=od_data.get("L_od", 48.0),
                alpha_T_od=od_data.get("alpha_T_od", 0.8),
            )
            net.od_pairs.append(od)

        # Build ng-route neighborhood
        ng_size = instance.get("config", {}).get("bap_ng_route_size", 15)
        net._build_ng_neighbors(ng_size)

        return net

    def add_arc(self, arc: Arc):
        """Add an arc to the network."""
        self.arcs[(arc.i, arc.j, arc.arc_id)] = arc
        self.graph.add_edge(arc.i, arc.j, mode=arc.mode, arc_id=arc.arc_id)
        if arc.is_road:
            self.road_graph.add_edge(arc.i, arc.j, distance=arc.distance)
        if arc.i not in self._arcs_out:
            self._arcs_out[arc.i] = []
        self._arcs_out[arc.i].append(arc)
        if arc.j not in self._arcs_in:
            self._arcs_in[arc.j] = []
        self._arcs_in[arc.j].append(arc)

    def get_road_arcs(self) -> list[Arc]:
        """Get all road arcs."""
        return [a for a in self.arcs.values() if a.is_road]

    def get_rail_arcs(self) -> list[Arc]:
        """Get all rail arcs."""
        return [a for a in self.arcs.values() if a.is_rail]

    def get_transfer_nodes(self) -> list[TransferNode]:
        """Get all transfer nodes."""
        return list(self.transfer_nodes.values())

    def get_arcs_out(self, node: int, mode: int | None = None) -> list[Arc]:
        """Get arcs leaving a node, optionally filtered by mode."""
        arcs = self._arcs_out.get(node, [])
        if mode is None:
            return arcs
        return [a for a in arcs if a.mode == mode]

    def get_arcs_in(self, node: int, mode: int | None = None) -> list[Arc]:
        """Get arcs entering a node, optionally filtered by mode."""
        arcs = self._arcs_in.get(node, [])
        if mode is None:
            return arcs
        return [a for a in arcs if a.mode == mode]

    def get_modes_at_arc(self, i: int, j: int) -> list[int]:
        """Get available transport modes on arc (i,j)."""
        return list(set([arc.mode for arc in self.arcs.values() if arc.i == i and arc.j == j]))

    def is_transfer_node(self, node: int) -> bool:
        """Check if a node is a road-rail transfer node."""
        return node in self.transfer_nodes

    def get_all_nodes(self) -> list[int]:
        """Get all node IDs."""
        return list(self.graph.nodes())

    def get_all_arc_keys(self) -> list[tuple[int, int, int]]:
        """Get all (i, j, arc_id) keys."""
        return list(self.arcs.keys())

    def neighbors_out(self, node: int) -> list[int]:
        """Get successor nodes."""
        return list(self.graph.successors(node))

    def neighbors_in(self, node: int) -> list[int]:
        """Get predecessor nodes."""
        return list(self.graph.predecessors(node))

    def get_node_coords(self, node_id: int) -> tuple[float, float]:
        """Get (x,y) coordinates of a node."""
        node_data = self.graph.nodes[node_id]
        return (node_data.get("x", 0.0), node_data.get("y", 0.0))

    def get_road_distance(self, node_u: int, node_v: int) -> float:
        """Calculate geographical Euclidean distance (km) between node_u and node_v."""
        if node_u == node_v:
            return 0.0
        import numpy as np
        coords_u = np.array(self.get_node_coords(node_u))
        coords_v = np.array(self.get_node_coords(node_v))
        return float(round(np.linalg.norm(coords_u - coords_v), 2))

    def _build_ng_neighbors(self, k: int):
        """Build ng-route memory neighborhood for each node."""
        all_nodes = self.get_all_nodes()
        for u in all_nodes:
            dists = []
            for v in all_nodes:
                if u == v:
                    dists.append((v, 0.0))
                else:
                    dists.append((v, self.get_road_distance(u, v)))
            dists.sort(key=lambda item: item[1])
            self.ng_neighbors[u] = set([n for n, d in dists[:min(k, len(dists))]])

    def get_exact_road_distance(self, node_u: int, node_v: int) -> float:
        """Get exact road distance using Dijkstra on road subgraph."""
        if node_u == node_v:
            return 0.0
        try:
            return float(round(nx.shortest_path_length(self.road_graph, node_u, node_v, weight="distance"), 2))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Strict topological constraint: if no path exists, return an infinite penalty distance
            # This enforces strictly '公-铁-公' operations instead of magically allowing trucks to bypass the graph.
            return 1e6

    def to_dict(self) -> dict:
        """Serialize network to dictionary."""
        return {
            "nodes": [
                {
                    "id": n,
                    "x": self.graph.nodes[n].get("x", 0.0),
                    "y": self.graph.nodes[n].get("y", 0.0)
                }
                for n in self.graph.nodes()
            ],
            "arcs": [
                {
                    "id": a.arc_id, "i": a.i, "j": a.j, "mode": a.mode,
                    "distance": a.distance,
                    "time_a": a.time_a, "time_b": a.time_b, "time_c": a.time_c, "time_d": a.time_d,
                    "alpha": a.alpha, "C_consequence": a.C_consequence,
                    "e_window": a.e_window, "l_window": a.l_window,
                    "capacity": a.capacity, "unit_cost": a.unit_cost,
                    "deploy_cost": a.deploy_cost, "min_units": a.min_units,
                    "population_density": a.population_density,
                }
                for a in self.arcs.values()
            ],
            "transfer_nodes": [
                {
                    "node_id": tn.node_id,
                    "transfer_time_a": tn.transfer_time_a,
                    "transfer_time_b": tn.transfer_time_b,
                    "transfer_time_c": tn.transfer_time_c,
                    "transfer_time_d": tn.transfer_time_d,
                    "gamma": tn.gamma,
                    "unit_transfer_cost": tn.unit_transfer_cost,
                    "facility_cost": tn.facility_cost,
                    "max_cranes": tn.max_cranes,
                    "max_storage_tons": tn.max_storage_tons,
                }
                for tn in self.transfer_nodes.values()
            ],
            "num_nodes": self.graph.number_of_nodes(),
            "num_arcs": len(self.arcs),
        }
