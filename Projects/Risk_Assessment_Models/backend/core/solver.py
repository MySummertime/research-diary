# Projects/Risk_Assessment_Models/backend/core/solver.py

from typing import List, Dict, Any
from .graph import Node, Edge, Graph
from .algorithms.sorting import SortingAlgorithms
from .algorithms.routing import RoutingAlgorithms


def solve_var(graph: Graph, src: int, dest: int, alpha: float) -> Dict[str, Any]:
    """
    Main solver function for the VaR Model.

    This function iterates through possible consequence thresholds (β) to find the
    smallest threshold for which a path exists with a cumulative risk (f) below
    a calculated tolerance (α_bar).

    Args:
        graph: The Graph object containing nodes and edges.
        src: The source node ID.
        dest: The destination node ID.
        alpha: The desired confidence level (e.g., 0.95).

    Returns:
        A dictionary containing the full results if a path is found,
        or a failure status if no such path exists.
    """

    # Step 0: Initialization
    # Collect all unique consequence values from edges
    all_cons: set = set(edge.cons for edge in graph.edges)
    # all_cons.update(node.cons for node in graph.nodes.values())

    # Create a sorted list of consequence thresholds to test, starting with 0.0
    sorted_consequences: list = SortingAlgorithms.merge_sort(list(all_cons))

    # The list of beta values (cut-off risks) to iterate through
    test_betas: list = [0.0] + sorted_consequences

    # Step 1-3: Iteratively solve subproblems for each beta_alpha
    for beta_alpha in test_betas:

        def cost_func(u: int, v: int, edge: Edge) -> float:
            """
            Calculates the risk cost for traversing an edge.
            The cost is the sum of modified probabilities of incident (on the edge and at the
            target node) whose consequence exceeds the current beta_alpha threshold.
            """
            target_node: Node | None = graph.nodes.get(edge.target)

            # Modify probability based on the consequence value
            modified_edge_prob: float = edge.prob if edge.cons > beta_alpha else 0.0
            modified_node_prob: float = 0.0
            if target_node:
                modified_node_prob = (
                    target_node.prob if target_node and target_node.cons > beta_alpha else 0.0
                )

            return modified_edge_prob + modified_node_prob

        # Call the generic Dijkstra's algorithm to get the value of f and the optimal path
        f_value, path = RoutingAlgorithms.dijkstra(graph, src, dest, cost_func)

        def cal_alpha_bar(graph: Graph, alpha: float, path: List[int]) -> float:
            """
            Calculates the alpha_bar value for a specific path using the graph's
            built-in adjacency list for efficient edge lookup.
            """
            # # Use the graph's efficient adjacency list property
            # adj = graph.adjacency_list
            # path_prob_sum = 0.0

            # for i in range(len(path) - 1):
            #     source_node_id = path[i]
            #     target_node_id = path[i+1]

            #     # Find the specific edge in the list of outgoing edges
            #     found_edge = None
            #     for edge in adj.get(source_node_id, []):
            #         if edge.target == target_node_id:
            #             found_edge = edge
            #             break

            #     if found_edge is None:
            #         return 0

            #     path_prob_sum += found_edge.prob

            # prob_no_accident_on_path = 1.0 - path_prob_sum
            # alpha_bar = max(0, alpha - prob_no_accident_on_path)
            alpha_bar = 1 - alpha
            return alpha_bar

        # Check for optimality
        if f_value is not None and path:
            # Calculate alpha_bar - the threshod of the cumulative probabilities
            alpha_bar: float = cal_alpha_bar(graph, alpha, path)
            if f_value <= alpha_bar:
                return {
                    'status': 'success',
                    'alpha': alpha,
                    'optimal_var': beta_alpha,
                    'f_value': f_value,
                    'alpha_bar': alpha_bar,
                    'path': path,
                    'algorithm_used': 'Dijkstra (VaR Model)',
                }

    # If the loop finishes without find a solution
    return {
        'status': 'failure',
        'message': 'No path found satisfying the given confidence level alpha.',
    }
