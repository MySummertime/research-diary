from __future__ import annotations
import numpy as np


class BranchingStrategy:
    """
    Implements OD-specific Arc branching strategy.

    Ryan-Foster (node-pair) branching has been removed because it is mathematically
    invalid for this problem: paths can visit the same physical node multiple times
    or partially, making the binary node-visit decision meaningless.

    Arc branching is now OD-specific: each branch constraint is scoped to a single
    OD index, so branching on arc (u,v,m) for OD k only forbids/enforces that arc
    in columns belonging to OD k, leaving all other ODs unaffected.
    """

    @staticmethod
    def get_arc_candidate_for_od(columns: list, target_od: int) -> tuple[int, int, int] | None:
        """
        Select an arc (i,j,arc_id) for branching, scoped to a specific OD index.

        Chooses the arc with fractional usage closest to 0.5 among columns
        belonging to target_od only.
        """
        arc_usage = {}
        for col in columns:
            if not hasattr(col, 'lp_value') or col.lp_value < 1e-6:
                continue
            if col.od_idx != target_od:
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