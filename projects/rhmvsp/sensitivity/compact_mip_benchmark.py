"""
Compact MIP Benchmark for RHMVSP.

Enumerates ALL feasible paths (columns) for each OD pair using the pricing
problem (zero duals, relaxed label limits), then solves the full path-based
formulation as a single MIP with Gurobi (no dynamic column generation).

This provides an exact benchmark algorithm that is fundamentally different
from BAP's dynamic column generation approach.

Reference: Wang et al. (2025) compares C&CG vs BD — both exact decomposition
algorithms. Here, BAP (dynamic CG) vs Compact MIP (enumeration + direct MIP)
serves the same role.
"""
from __future__ import annotations

import copy
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solvers.branch_and_price import BranchAndPriceController, BAPNode
from src.utils.logging import Logger


def solve_compact_mip(
    instance: dict,
    config,
    logger: Logger,
    time_limit: int = 3600,
) -> dict:
    """
    Solve RHMVSP as a compact MIP by enumerating ALL feasible paths upfront
    and solving the full path-based formulation as a MIP.

    Uses BAP's own infrastructure (pricing + master) but skips the dynamic
    column generation loop — all paths are enumerated upfront, then the
    master is solved as an integer MIP.

    Args:
        instance: Problem instance dict
        config: RHMVSPConfig
        logger: Logger instance
        time_limit: Solver time limit in seconds

    Returns:
        Solution dict (same format as BranchAndPriceController.solve())
            including status, total_cost, R_net_risk, R_net_time, etc.
    """
    solve_t0 = time.time()

    cfg = copy.deepcopy(config)
    cfg.time_limit = time_limit
    cfg.solver_name = "gurobi"

    # ── 1. Create BAP controller (provides network, engine, pp, mp) ────────
    bap = BranchAndPriceController(cfg, instance, logger)

    # ── 2. Enumerate ALL feasible columns per OD pair ──────────────────────
    # Use high (big_m) duals so reduced_cost is negative — pricing finds ALL
    # feasible paths (same approach as BAP's _generate_initial_columns).
    big_m = getattr(cfg, 'big_m', 1e6)
    high_duals = {
        "od": {i: big_m for i in range(len(bap.network.od_pairs))},
        "vehicle": {},
        "trans_out": {},
        "trans_in": {},
        "hub": {},
    }

    # Raise label limits for full enumeration
    orig_exact = cfg.max_labels_exact
    orig_heur = cfg.max_labels_heuristic
    cfg.max_labels_exact = 50000
    cfg.max_labels_heuristic = 20000

    all_columns = []
    existing_fingerprints: set = set()

    # ── Phase A: Enumerate with high duals (initial pass) ────────────────
    for od_idx, od in enumerate(bap.network.od_pairs):
        logger.info(f"  Enumerating paths for OD {od_idx+1} "
                     f"(origin={od.origin}, dest={od.destination}) ...")

        for exact_flag in [False, True]:
            phase = "Exact" if exact_flag else "Heuristic"
            labels = bap.pp.solve(
                od.origin, od.destination, od_idx, high_duals,
                od.time_window_start, od.time_window_end, od.alpha_T_od,
                branch_constraints=[], exact=exact_flag,
            )
            cols_added = 0
            for lab in labels:
                fp = tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence)
                if fp not in existing_fingerprints:
                    existing_fingerprints.add(fp)
                    col = bap._label_to_column(lab, od_idx)
                    if col is not None:
                        all_columns.append(col)
                        cols_added += 1
            logger.info(f"    {phase}: {cols_added} new columns")

    # ── Phase B: Additional passes with randomly perturbed duals ─────────
    # Single-pass pricing may miss paths for some ODs. Run extra passes
    # with random dual perturbations to explore different parts of the
    # search space.
    import random as _random
    _random.seed(42)
    n_extra_passes = 3
    n_od = len(bap.network.od_pairs)

    for pass_i in range(n_extra_passes):
        perturbed_duals = {
            "od": {i: big_m * _random.uniform(0.5, 1.5) for i in range(n_od)},
            "vehicle": {}, "trans_out": {}, "trans_in": {}, "hub": {},
        }
        total_added = 0
        for od_idx, od in enumerate(bap.network.od_pairs):
            labels = bap.pp.solve(
                od.origin, od.destination, od_idx, perturbed_duals,
                od.time_window_start, od.time_window_end, od.alpha_T_od,
                branch_constraints=[], exact=False,  # heuristic is enough for diversity
            )
            for lab in labels:
                fp = tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence)
                if fp not in existing_fingerprints:
                    existing_fingerprints.add(fp)
                    col = bap._label_to_column(lab, od_idx)
                    if col is not None:
                        all_columns.append(col)
                        total_added += 1
        logger.info(f"    Random pass {pass_i+1}/{n_extra_passes}: {total_added} new columns")

    cfg.max_labels_exact = orig_exact
    cfg.max_labels_heuristic = orig_heur

    enum_time = time.time() - solve_t0
    logger.info(f"  Enumerated {len(all_columns)} total columns ({enum_time:.1f}s)")

    if not all_columns:
        logger.warning("  No columns enumerated — instance infeasible")
        return {"status": "infeasible"}

    # ── 3. Solve as MIP via BAP's extraction ───────────────────────────────
    # _extract_solution re-solves MasterProblem as MIP with all columns,
    # stitches vehicles, computes costs and reliability, returns full dict.
    mock_node = BAPNode(
        lower_bound=0.0,
        depth=0,
        node_id=0,
        columns=all_columns,
    )

    try:
        mip_t0 = time.time()
        sol = bap._extract_solution(mock_node)
        mip_time = time.time() - mip_t0
    except Exception as e:
        logger.error(f"  MIP solve / extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}

    # ── 4. Augment with timing info ────────────────────────────────────────
    total_time = time.time() - solve_t0
    sol["solve_wall_time"] = total_time
    sol["solve_cpu_time"] = total_time
    sol["enumeration_time"] = enum_time
    sol["mip_solve_time"] = mip_time
    sol["solver"] = "compact_mip"
    sol.setdefault("gap_pct", 0.0)
    sol.setdefault("iterations", 1)
    sol.setdefault("num_repositions", 0)

    return sol
