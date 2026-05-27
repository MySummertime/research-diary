

from __future__ import annotations
import numpy as np


class BranchingStrategy:
    """
    Implements Ryan-Foster (node-pair) and Arc-based branching strategies.
    """

    @staticmethod
    def get_ryan_foster_candidate(columns: list, total_demand: int) -> tuple[int, int] | None:
        """
        Select a node pair (i,j) for Ryan-Foster branching.
        Chooses the pair with fractional co-occurrence closest to 0.5.
        """
        pair_usage = {}

        for col in columns:
            if not hasattr(col, 'lp_value') or col.lp_value < 1e-6:
                continue

            # Get all nodes in the path
            nodes = set()
            for arc in col.arcs:
                nodes.add(arc.i)
                nodes.add(arc.j)

            sorted_nodes = sorted(list(nodes))
            for idx_a in range(len(sorted_nodes)):
                for idx_b in range(idx_a + 1, len(sorted_nodes)):
                    pair = (sorted_nodes[idx_a], sorted_nodes[idx_b])
                    pair_usage[pair] = pair_usage.get(pair, 0.0) + col.lp_value

        # Find pair closest to 0.5 (or any fractional value)
        best_pair = None
        best_score = 1.0  # Max distance from 0.5 is 0.5, so 1.0 is safe

        for pair, usage in pair_usage.items():
            fractional_part = usage - np.floor(usage)
            if 1e-6 < fractional_part < 1 - 1e-6:
                score = abs(fractional_part - 0.5)
                if score < best_score:
                    best_score = score
                    best_pair = pair

        return best_pair

    @staticmethod
    def get_arc_candidate(columns: list) -> tuple[int, int, int] | None:
        """
        Select an arc (i,j,m) for branching.
        Chooses the arc with fractional usage closest to 0.5.
        """
        arc_usage = {}
        for col in columns:
            if not hasattr(col, 'lp_value') or col.lp_value < 1e-6:
                continue

            for arc in col.arcs:
                key = (arc.i, arc.j, arc.arc_id)
                arc_usage[key] = arc_usage.get(key, 0.0) + col.lp_value

        best_arc = None
        best_score = 1.0

        for key, usage in arc_usage.items():
            fractional_part = usage - np.floor(usage)
            if 1e-6 < fractional_part < 1 - 1e-6:
                score = abs(fractional_part - 0.5)
                if score < best_score:
                    best_score = score
                    best_arc = key

        return best_arc
