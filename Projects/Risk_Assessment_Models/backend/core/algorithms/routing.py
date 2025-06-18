# Projects/Risk_Assessment_Models/backend/core/algorithms/routing.py

import heapq
from typing import List, Dict, Tuple, Optional, Callable
from ..graph import Edge, Graph


class RoutingAlgorithms:
    """
    A container for graph routing algorithms.
    """

    @staticmethod
    def _reconstruct_path(
        parent: Dict[int, Optional[int]], src: int, dest: int
    ) -> Optional[List[int]]:
        """
        Redonstruct the path depending on dict 'parent'
        """
        path: List = []
        curr: Optional[int] = dest
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr)
        return path[::-1] if path and path[-1] == src else None

    @staticmethod
    def dijkstra(
        graph: Graph, src: int, dest: int, cost_func: Callable[[int, int, Edge], float]
    ) -> Tuple[Optional[float], Optional[List[int]]]:
        """
            Calculates the optimal path using Dijkstra's algorithm with a generic cost function.

            This implementation is flexible, allowing any custom logic to be used for
            calculating the weight of an edge during traversal.

            Args:
        graph: An instance of the Graph class.
                src: The ID of the starting node.
                dest: The ID of the destination node.
                cost_func: A callable function that takes (source_id, target_id, edge_object)
                           and returns a float representing the cost of that edge.

            Returns:
                A tuple containing:
                - The total cost of the shortest path (float), or None if no path exists.
                - A list of node IDs for the path, or None if no path exists.
        """
        # --- Initialization ---
        distances: Dict[int, float] = {node_id: float('inf') for node_id in graph.nodes}
        if src not in distances:
            return None, None
        distances[src] = 0

        previous_nodes: Dict[int, Optional[int]] = {node_id: None for node_id in graph.nodes}
        priority_queue: List[Tuple[float, int]] = [(0, src)]

        # Build adjacency list for efficient neighbor lookup
        adj: Dict[int, List[Edge]] = {node_id: [] for node_id in graph.nodes}
        for edge in graph.edges:
            if edge.source in adj:
                adj[edge.source].append(edge)

        # --- Main Loop ---
        while priority_queue:
            current_distance, current_node_id = heapq.heappop(priority_queue)

            if current_distance > distances[current_node_id]:
                continue

            if current_node_id == dest:
                break  # Destination reached

            for edge in adj.get(current_node_id, []):
                # The key change is here: Use the provided cost_func to get the edge weight
                cost = cost_func(edge.source, edge.target, edge)

                distance_through_current = current_distance + cost

                if distance_through_current < distances[edge.target]:
                    distances[edge.target] = distance_through_current
                    previous_nodes[edge.target] = current_node_id
                    heapq.heappush(priority_queue, (distance_through_current, edge.target))

        # --- Path Reconstruction ---
        if distances.get(dest) is None or distances[dest] == float('inf'):
            return None, None  # No path found

        path = RoutingAlgorithms._reconstruct_path(previous_nodes, src, dest)

        if path and path[0] == src:
            return distances[dest], path
        else:
            return None, None  # Path does not connect back to source
