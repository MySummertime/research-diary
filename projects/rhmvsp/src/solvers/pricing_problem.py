from __future__ import annotations
from dataclasses import dataclass, field
import heapq
from itertools import count

from config import RHMVSPConfig
from ..core.network import Arc, MultimodalNetwork
from ..core.uncertainty import TrapezoidalFuzzy, TriangularFuzzy, UncertaintyEngine


@dataclass
class Label:
    """Label for ESPPRC with resource consumption, reduced cost tracking, and ng-route memory."""
    cost: float = 0.0
    reduced_cost: float = 0.0
    time_upper: float = 0.0
    time_lower: float = 0.0
    risk_R: float = 1.0
    visited: frozenset = field(default_factory=frozenset)
    ng_memory: frozenset = field(default_factory=frozenset)
    node: int = -1
    path: tuple = ()  # sequence of (node, mode)
    arc_sequence: tuple = ()
    prev_mode: int = 0
    feasible: bool = True
    rail_started: bool = False  # True once a rail arc has been traversed
    rail_ended: bool = False    # True once a road arc is traversed AFTER rail_started

    def __lt__(self, other):
        return self.reduced_cost < other.reduced_cost


class PricingProblem:
    """
    ESPPRC solver with exact Bidirectional Label-Correcting and ng-route relaxation.

    S3-1 fix: _join_labels loop bug corrected — all arc iterations are now inside
              the for-arc loop body so every midpoint arc is properly evaluated.
    S2-6 fix: Arc reliability cache keyed by (arc_id, time_bin) to preserve
              time-dependent risk computation; same arc at different departure
              times now gets independently computed reliability values.
    S3-2 fix: Bidirectional half-horizon cutoff uses resource-balanced midpoint
              instead of fixed 4-hour buffer.
    S3-3 fix: Dominance rule includes ng_memory subset check so that labels with
              incompatible ng sets cannot dominate each other.
    """

    def __init__(self, config: RHMVSPConfig, network: MultimodalNetwork, engine: UncertaintyEngine, logger):
        self.config = config
        self.network = network
        self.engine = engine
        self.logger = logger
        # S2-6: time-keyed arc reliability cache: (arc_id, hour_bin) -> R
        self._arc_R_cache: dict[tuple, float] = {}
        # S2-6: transfer node time cache: (node_id, hour_bin) -> transfer_time
        self._transfer_time_cache: dict[tuple, float] = {}
        # Timers for exact ESPPRC execution profiling
        self.total_espprc_cpu_time = 0.0
        self.total_espprc_wall_time = 0.0

    def _get_arc_R(self, arc: Arc, start_time: float, alpha_T: float) -> float:
        """
        S2-6: Fast continuous arc reliability computation with high-precision cache.
        Uses 0.1 hour (6 minutes) granularity to balance exact time-varying risk integration 
        with B&P algorithm performance.
        """
        # Round start_time to 1 decimal place (0.1 hour granularity)
        time_bin = round(start_time, 1)
        cache_key = (arc.arc_id, time_bin)
        
        if cache_key not in self._arc_R_cache:
            tri = self.engine.make_transport_time(arc.distance, arc.mode)
            self._arc_R_cache[cache_key] = self.engine.arc_reliability(
                arc.alpha, start_time=time_bin,
                fuzzy_duration=(tri.a, tri.b, tri.c, tri.d),
                time_dependent=True,
                C_consequence=arc.C_consequence,
            )
        return self._arc_R_cache[cache_key]

    def _get_transfer_time(self, tn, alpha_T: float) -> float:
        """
        Transfer time is fixed to 1.0 hour for unloading from road to rail.
        """
        return 1.0

    def solve(self, origin: int, destination: int, od_idx: int, duals: dict,
              od_E: float, od_L: float, alpha_T: float, branch_constraints: list = None,
              exact: bool = True) -> list[Label]:
        """
        Solve bidirectional ESPPRC, optionally heuristically (exact=False).
        """
        self.origin = origin
        self.destination = destination
        self.od_idx = od_idx
        self.od_E = od_E
        self.od_L = od_L
        self.branch_constraints = branch_constraints if branch_constraints is not None else []

        import time
        start_wall = time.time()
        start_cpu = time.process_time()

        # 1. Forward Labeling
        forward_labels = self._desrochers_label_correcting(origin, destination, duals, od_E, od_L, alpha_T, forward=True, exact=exact)
        num_f = sum(len(v) for v in forward_labels.values())

        # 2. Backward Labeling
        backward_labels = self._desrochers_label_correcting(destination, origin, duals, od_E, od_L, alpha_T, forward=False, exact=exact)
        num_b = sum(len(v) for v in backward_labels.values())

        # 3. Join Step (S3-1 fix applied inside _join_labels)
        all_columns = self._join_labels(forward_labels, backward_labels, od_L, alpha_T, duals)
        if od_idx == 3:
            self.logger.warning(f"DEBUG OD 4: F labels: {sum(len(v) for v in forward_labels.values())}, B labels: {sum(len(v) for v in backward_labels.values())}, Joined: {len(all_columns)}")
        all_columns.sort(key=lambda x: x.reduced_cost)
        if len(all_columns) > 10:
            all_columns = all_columns[:self.config.bap_pricing_columns_limit]

        self.logger.debug(f"  Pricing OD {od_idx} (exact={exact}): F={num_f}, B={num_b}, Joined={len(all_columns)}")

        self.total_espprc_wall_time += (time.time() - start_wall)
        self.total_espprc_cpu_time += (time.process_time() - start_cpu)

        return all_columns

    def _desrochers_label_correcting(self, start_node: int, end_node: int, duals: dict,
                         od_E: float, od_L: float, alpha_T: float, forward: bool,
                         exact: bool = True) -> dict[int, list[Label]]:
        """
        Desrochers Label-correcting algorithm (Dynamic Programming) with ng-route, S3-2 resource-balanced midpoint,
        and heapq priority queue.
        """
        labels_at: dict[int, list[Label]] = {n: [] for n in self.network.get_all_nodes()}

        max_L = max(self.config.T_max, od_L)
        init_time = od_E if forward else max_L
        init_ng = frozenset({start_node})

        q_veh = self.config.q_road
        dual_od = duals.get("od", {}).get(self.od_idx, 0.0)
        dual_v = duals.get("vehicle", {}).get(0, 0.0)
        dual_veh = duals.get("vehicle", [])
        dual_trans_out = duals.get("trans_out", {}).get(self.od_idx, 0.0)
        dual_trans_in = duals.get("trans_in", {}).get(self.od_idx, 0.0)

        # Initialize reduced cost with base OD and fleet duals
        # #13: Dual signs verified against PuLP/Gurobi convention:
        # capacity_link (GE): dual_od <= 0 → -dual_od*q_veh >= 0 (adds cost)
        # vehicle_balance (LE): dual_v >= 0 → -dual_v <= 0 (reduces cost)
        # reposition_flow (LE): dual_trans >= 0 → +dual_trans >= 0 (adds cost)
        init_rc = - (dual_od * q_veh) - dual_v + dual_trans_out + dual_trans_in

        init = Label(
            cost=0.0, reduced_cost=init_rc, time_upper=init_time, time_lower=init_time,
            risk_R=1.0, visited=frozenset({start_node}), ng_memory=init_ng,
            node=start_node, path=((start_node, 0),), feasible=True, rail_started=False, rail_ended=False
        )
        labels_at[start_node].append(init)

        counter = count()
        queue = []
        heapq.heappush(queue, (init.reduced_cost, next(counter), init))

        # S3-2: Resource-balanced midpoint — use 40% of time horizon for each
        # direction with 20% overlap buffer, rather than a fixed 4h constant.
        span = max_L - od_E
        half_ratio = getattr(self.config, 'pricing_overlap_ratio', 0.75)  # from config
        max_time_f = od_E + half_ratio * span
        min_time_b = max(od_E, max_L - half_ratio * span)

        while queue:
            _, _, curr = heapq.heappop(queue)

            # If the label has been dominated and removed, skip
            if curr not in labels_at[curr.node]:
                continue

            adj_arcs = self.network.get_arcs_out(curr.node) if forward else self.network.get_arcs_in(curr.node)

            for arc in adj_arcs:
                next_node = arc.j if forward else arc.i
                
                # Prevent containers from using the inter-hub highway network (Mode 1 between two hubs)
                # This ensures the new fully-connected road topology is strictly used for empty truck repositioning,
                # avoiding a massive combinatorial explosion in ESPPRC.
                if arc.mode == 1 and self.network.is_transfer_node(arc.i) and self.network.is_transfer_node(arc.j):
                    continue
                
                # Elementarity check using ng-route memory
                if next_node in curr.ng_memory and next_node != end_node:
                    continue

                new_labels = self._extend(curr, arc, duals, alpha_T, forward)

                for new_label in new_labels:
                    if new_label and new_label.feasible:
                        if forward and new_label.time_upper > max_time_f and next_node != end_node:
                            continue
                        if not forward and new_label.time_upper < min_time_b and next_node != end_node:
                            continue

                        if self._is_dominated(new_label, labels_at[new_label.node]):
                            continue

                        # S3-3: Remove only those labels actually dominated by new_label
                        labels_at[new_label.node] = [
                            l for l in labels_at[new_label.node]
                            if not self._dominates(new_label, l)
                        ]
                        labels_at[new_label.node].append(new_label)

                        # Heuristic pruning: keep top labels by reduced cost
                        max_keep = self.config.max_labels_heuristic if not exact else self.config.max_labels_exact
                        if len(labels_at[new_label.node]) > max_keep:
                            labels_at[new_label.node].sort(key=lambda x: x.reduced_cost)
                            labels_at[new_label.node] = labels_at[new_label.node][:max_keep]

                        heapq.heappush(queue, (new_label.reduced_cost, next(counter), new_label))

        return labels_at

    def _extend(self, parent: Label, arc: Arc, duals: dict, alpha_T: float, forward: bool) -> list[Label]:
        """Extend label along arc, generating multiple labels for all valid rail timetable branches."""
        tri = self.engine.make_transport_time(arc.distance, arc.mode)
        dt = getattr(tri, 'c', tri.b)  # Expected worst-case peak time

        q_veh = self.config.q_road
        q_rail = self.config.q_rail

        transfer_cost = 0.0
        transfer_time = 0.0
        node_R = 1.0
        dual_hub = 0.0

        # Safe transfer hub extraction
        if parent.prev_mode != 0 and parent.prev_mode != arc.mode:
            for c_type, c_data in self.branch_constraints:
                if c_type == "hub_close" and parent.node == c_data:
                    return []
            tn = self.network.transfer_nodes.get(parent.node)
            if tn is None:
                return []
            transfer_time = self._get_transfer_time(tn, alpha_T)
            transfer_cost = tn.unit_transfer_cost * q_veh
            node_R = 1.0 - tn.gamma
            dual_hub = duals.get("hub", {}).get(parent.node, 0.0)

        if arc.mode == 2:
            c_rail_base_fixed = self.config.c_rail_base_fixed
            c_rail_base_var = self.config.c_rail_base_var
            c_rail_hazmat_markup = self.config.c_rail_hazmat_markup
            c_rail_fund = self.config.c_rail_fund
            dist = arc.distance
            actual_unit_cost = round(
                (c_rail_base_fixed + c_rail_base_var * dist) * (1 + c_rail_hazmat_markup)
                + c_rail_fund * dist, 2
            )
        else:
            actual_unit_cost = arc.unit_cost

        ratio = (q_veh / q_rail) if arc.mode == 2 else 1.0
        arc_deploy_cost = arc.deploy_cost * arc.min_units * ratio
        base_transport_cost_run = actual_unit_cost * q_veh

        spread = self.config.fuzzy_cost_spread
        alpha_c = self.config.alpha_c
        # Expected fuzzy cost multiplier: E^M[c~] = (a + 2b + c) / 4 for triangular
        expected_multiplier = (spread[0] + 2.0 * spread[1] + spread[2]) / 4.0
        transport_cost_run_expected = base_transport_cost_run * expected_multiplier
        # Upper bound deterministic equivalent coefficient
        if alpha_c > 0.5:
            upper_coeff = (2.0 * alpha_c - 1.0) * spread[2] + 2.0 * (1.0 - alpha_c) * spread[1]
        else:
            upper_coeff = spread[1]

        transport_cost_run_upper = base_transport_cost_run * upper_coeff
        # #7: Use expected cost (matches master problem's col.total_cost)
        actual_cost_run = arc_deploy_cost + transport_cost_run_expected + transfer_cost

        next_node = arc.j if forward else arc.i

        if hasattr(self.network, 'ng_neighbors'):
            N_j = self.network.ng_neighbors.get(next_node, set([next_node]))
            new_ng = frozenset((parent.ng_memory | {next_node}).intersection(N_j))
        else:
            new_ng = parent.ng_memory | {next_node}

        new_rail_started = parent.rail_started
        new_rail_ended = parent.rail_ended
        if forward:
            if arc.mode == 2:  # rail
                if parent.rail_ended:
                    return []  # cannot use rail after rail segment ended
                new_rail_started = True
                # rail_ended stays False (still in rail segment)
            elif arc.mode == 1:  # road
                if parent.rail_started and not parent.rail_ended:
                    new_rail_ended = True  # first road arc after rail → rail segment ended
                # else: flags unchanged (road before rail, or road after rail already ended)
        else:
            # Backward extension: symmetric to forward, but reversed.
            # We track rail_started and rail_ended to prevent road-rail-road-rail
            # sequences from filling the priority queue and bypassing heuristic pruning.
            if arc.mode == 2:  # rail (backward)
                if parent.rail_ended:
                    return []
                new_rail_started = True
                new_rail_ended = False
            elif arc.mode == 1:  # road (backward)
                if parent.rail_started and not parent.rail_ended:
                    new_rail_ended = True  # A road after rail (in reverse) ends the rail segment
                else:
                    new_rail_ended = parent.rail_ended
            else:
                new_rail_ended = parent.rail_ended

        new_labels = []

        if forward:
            t_arr = parent.time_upper + transfer_time
            if arc.mode == 2:
                runs = getattr(self.network, 'timetable', {}).get((arc.i, arc.j), [])
                valid_runs = sorted(
                    [r for r in runs if r["departure_time"] >= t_arr and r["departure_time"] <= t_arr + 48.0],
                    key=lambda x: x["departure_time"]
                )[:2]
                if not valid_runs:
                    return []
                for r in valid_runs:
                    t_start = r["departure_time"]
                    new_time = r["arrival_time"]
                    # S2-6: use time-keyed cache
                    arc_R = self._get_arc_R(arc, t_start, alpha_T)
                    
                    actual_cost_total = actual_cost_run
                    
                    new_cost_base = parent.cost + actual_cost_total
                    new_rc_base = parent.reduced_cost + actual_cost_total - dual_hub
                    
                    new_risk = min(parent.risk_R, arc_R, node_R)
                    if new_risk >= self.config.beta_od:
                        new_labels.append(Label(
                            cost=new_cost_base, reduced_cost=new_rc_base,
                            time_upper=new_time, time_lower=new_time, risk_R=new_risk,
                            visited=parent.visited | {next_node}, ng_memory=new_ng, node=next_node,
                            path=parent.path + ((next_node, arc.mode),),
                            arc_sequence=parent.arc_sequence + (arc,), prev_mode=arc.mode, feasible=True, rail_started=new_rail_started, rail_ended=new_rail_ended
                        ))
            else:
                t_start = t_arr
                new_time = t_start + dt
                arc_R = self._get_arc_R(arc, t_start, alpha_T)
                
                actual_cost_total = actual_cost_run
                new_cost_base = parent.cost + actual_cost_total
                new_rc_base = parent.reduced_cost + actual_cost_total - dual_hub
                
                new_risk = min(parent.risk_R, arc_R, node_R)
                if new_risk >= self.config.beta_od:
                    new_labels.append(Label(
                        cost=new_cost_base, reduced_cost=new_rc_base,
                        time_upper=new_time, time_lower=new_time, risk_R=new_risk,
                        visited=parent.visited | {next_node}, ng_memory=new_ng, node=next_node,
                        path=parent.path + ((next_node, arc.mode),),
                        arc_sequence=parent.arc_sequence + (arc,), prev_mode=arc.mode, feasible=True, rail_started=new_rail_started, rail_ended=new_rail_ended
                    ))
        else:
            t_dep_latest = parent.time_upper - transfer_time
            if arc.mode == 2:
                runs = getattr(self.network, 'timetable', {}).get((arc.i, arc.j), [])
                valid_runs = sorted(
                    [r for r in runs if r["arrival_time"] <= t_dep_latest and r["arrival_time"] >= t_dep_latest - 48.0],
                    key=lambda x: x["arrival_time"],
                    reverse=True
                )[:2]
                if not valid_runs:
                    return []
                for r in valid_runs:
                    new_time = r["departure_time"]
                    arc_R = self._get_arc_R(arc, new_time, alpha_T)
                    
                    t_end = r["arrival_time"]
                    actual_cost_total = actual_cost_run
                    new_cost_base = parent.cost + actual_cost_total
                    new_rc_base = parent.reduced_cost + actual_cost_total - dual_hub
                    new_risk = min(parent.risk_R, arc_R, node_R)
                    if new_risk >= self.config.beta_od and new_time >= self.od_E:
                        new_labels.append(Label(
                            cost=new_cost_base, reduced_cost=new_rc_base,
                            time_upper=new_time, time_lower=new_time, risk_R=new_risk,
                            visited=parent.visited | {next_node}, ng_memory=new_ng, node=next_node,
                            path=parent.path + ((next_node, arc.mode),),
                            arc_sequence=(arc,) + parent.arc_sequence, prev_mode=arc.mode, feasible=True, rail_started=new_rail_started, rail_ended=new_rail_ended
                        ))
            else:
                new_time = t_dep_latest - dt
                arc_R = self._get_arc_R(arc, new_time, alpha_T)
                
                t_end = t_dep_latest
                actual_cost_total = actual_cost_run
                new_cost_base = parent.cost + actual_cost_total
                new_rc_base = parent.reduced_cost + actual_cost_total - dual_hub
                new_risk = min(parent.risk_R, arc_R, node_R)
                if new_risk >= self.config.beta_od and new_time >= self.od_E:
                    new_labels.append(Label(
                        cost=new_cost_base, reduced_cost=new_rc_base,
                        time_upper=new_time, time_lower=new_time, risk_R=new_risk,
                        visited=parent.visited | {next_node}, ng_memory=new_ng, node=next_node,
                        path=parent.path + ((next_node, arc.mode),),
                        arc_sequence=(arc,) + parent.arc_sequence, prev_mode=arc.mode, feasible=True, rail_started=new_rail_started, rail_ended=new_rail_ended
                    ))

        return new_labels

    def _join_labels(self, forward: dict[int, list[Label]], backward: dict[int, list[Label]],
                    od_L: float, alpha_T: float, duals: dict) -> list[Label]:
        """
        Fast midpoint join of forward and backward labels across network connecting arcs.

        S3-1 fix: The inner f_lab x b_lab double-loop is now correctly placed
        INSIDE the for-arc loop body, so every midpoint arc is evaluated.
        Previously this loop was outside the for-arc loop, using only the last arc.
        """
        complete_paths = []
        dual_od = duals.get("od", {}).get(self.od_idx, 0.0)
        dual_veh = duals.get("vehicle", [])
        dual_trans_out = duals.get("trans_out", {}).get(self.od_idx, 0.0)
        dual_trans_in = duals.get("trans_in", {}).get(self.od_idx, 0.0)

        q_veh = self.config.q_road
        q_rail = self.config.q_rail

        for j in self.network.get_all_nodes():
            if backward[j]:
                backward[j].sort(key=lambda x: x.reduced_cost)

        max_join_f = self.config.bap_pricing_columns_limit * 5  # limit forward labels
        max_join_b = self.config.bap_pricing_columns_limit * 5  # limit backward labels

        for i in self.network.get_all_nodes():
            if not forward[i]:
                continue
            forward[i].sort(key=lambda x: x.reduced_cost)

            for arc in self.network.get_arcs_out(i):
                # S3-1 FIX: everything below is now INSIDE this for-arc loop
                j = arc.j
                if not backward[j]:
                    continue

                tri = self.engine.make_transport_time(arc.distance, arc.mode)
                dt_arc = tri.deterministic_equivalent_upper(alpha_T)

                # Adjusted duals_base usage after removing budget_dual
                duals_base = (dual_od * q_veh) - dual_trans_out - dual_trans_in

                for f_lab in forward[i][:max_join_f]:
                    for b_lab in backward[j][:max_join_b]:
                        # Pruning: check reduced cost bound
                        if f_lab.reduced_cost + b_lab.reduced_cost + duals_base >= -1e-4:
                            break  # Since backward[j] is sorted by reduced_cost, all subsequent will also fail
                        # S3-3: Elementarity check — no shared non-endpoint visited nodes
                        if not f_lab.visited.isdisjoint(b_lab.visited):
                            continue

                        transfer_time = 0.0
                        transfer_cost = 0.0
                        node_R = 1.0
                        if f_lab.prev_mode != 0 and f_lab.prev_mode != arc.mode:
                            tn = self.network.transfer_nodes.get(i)
                            if tn is None:
                                continue
                            transfer_time = self._get_transfer_time(tn, alpha_T)
                            transfer_cost = tn.unit_transfer_cost * q_veh
                            node_R = 1.0 - tn.gamma

                        transfer_time_j = 0.0
                        transfer_cost_j = 0.0
                        node_R_j = 1.0
                        if b_lab.prev_mode != 0 and b_lab.prev_mode != arc.mode:
                            tn_j = self.network.transfer_nodes.get(j)
                            if tn_j is None:
                                continue
                            transfer_time_j = self._get_transfer_time(tn_j, alpha_T)
                            transfer_cost_j = tn_j.unit_transfer_cost * q_veh
                            node_R_j = 1.0 - tn_j.gamma

                        t_arrive_i = f_lab.time_upper + transfer_time
                        if arc.mode == 2:
                            runs = getattr(self.network, 'timetable', {}).get((arc.i, arc.j), [])
                            valid_runs = [r for r in runs if r["departure_time"] >= t_arrive_i]
                            if not valid_runs:
                                continue
                            r = valid_runs[0]
                            t_start_arc = r["departure_time"]
                            t_end_arc = r["arrival_time"] + transfer_time_j
                        else:
                            t_start_arc = t_arrive_i
                            t_end_arc = t_start_arc + dt_arc + transfer_time_j

                        if t_end_arc > b_lab.time_upper:
                            continue

                        # S2-6: time-keyed reliability
                        arc_R = self._get_arc_R(arc, t_start_arc, alpha_T)
                        path_risk = min(f_lab.risk_R, b_lab.risk_R, arc_R, node_R, node_R_j)
                        if path_risk < self.config.beta_od:
                            continue

                        full_visited = f_lab.visited | b_lab.visited
                        full_arcs = f_lab.arc_sequence + (arc,) + b_lab.arc_sequence

                        # Validate road-rail-road sequence:
                        # Must start and end with road, have rail, and only one contiguous rail segment
                        # NOTE: b_lab.arc_sequence stores arcs in forward direction
                        # (current-node→destination), so b_lab.arc_sequence[-1] is
                        # the arc closest to destination (should be road, mode=1).
                        if not f_lab.arc_sequence or not b_lab.arc_sequence:
                            continue
                        all_modes = [a.mode for a in f_lab.arc_sequence] + [arc.mode] + [a.mode for a in b_lab.arc_sequence]
                        if f_lab.arc_sequence[0].mode != 1 or b_lab.arc_sequence[-1].mode != 1 or 2 not in all_modes:
                            continue
                        # Check rail arcs are contiguous (no gap = no rail→road→rail)
                        rail_pos = [k for k, m in enumerate(all_modes) if m == 2]
                        has_gap = any(rail_pos[k + 1] - rail_pos[k] > 1 for k in range(len(rail_pos) - 1))
                        if has_gap:
                            continue

                        trans_hubs = set()
                        for k in range(len(full_arcs) - 1):
                            a1, a2 = full_arcs[k], full_arcs[k + 1]
                            if a1.mode != a2.mode:
                                tn_node = self.network.transfer_nodes.get(a1.j)
                                if tn_node:
                                    trans_hubs.add(a1.j)

                        if self.branch_constraints:
                            valid_bc = True
                            for bc_type, bc_data in self.branch_constraints:
                                if bc_type in ["rf_same", "rf_diff"]:
                                    i_bc, j_bc = bc_data
                                    has_i = i_bc in full_visited
                                    has_j = j_bc in full_visited
                                    if bc_type == "rf_same" and has_i != has_j:
                                        valid_bc = False
                                        break
                                    elif bc_type == "rf_diff" and (has_i and has_j):
                                        valid_bc = False
                                        break
                                elif bc_type == "hub_close":
                                    hub_id = bc_data
                                    if hub_id in trans_hubs:
                                        valid_bc = False
                                        break
                                elif bc_type == "hub_open":
                                    pass
                            if not valid_bc:
                                continue

                        # S3-1 FIX: Compute exact midpoint arc cost and combine with labels
                        if arc.mode == 2:
                            c_rfx = self.config.c_rail_base_fixed
                            c_rv = self.config.c_rail_base_var
                            c_rh = self.config.c_rail_hazmat_markup
                            c_rf = self.config.c_rail_fund
                            a_unit = round((c_rfx + c_rv * arc.distance) * (1 + c_rh) + c_rf * arc.distance, 2)
                        else:
                            a_unit = arc.unit_cost

                        r_a = (q_veh / q_rail) if arc.mode == 2 else 1.0
                        arc_deploy_cost = arc.deploy_cost * arc.min_units * r_a
                        transport_cost_run = a_unit * q_veh
                        # #7: Use expected cost (matches pricing _extend and master problem)
                        spread_j = self.config.fuzzy_cost_spread
                        expected_mult_j = (spread_j[0] + 2.0 * spread_j[1] + spread_j[2]) / 4.0
                        actual_cost_arc = arc_deploy_cost + transport_cost_run * expected_mult_j + transfer_cost + transfer_cost_j

                        # Delay penalty is assessed on completed routes only.
                        dual_hub_i = duals.get("hub", {}).get(i, 0.0) if transfer_cost > 0 else 0.0
                        dual_hub_j = duals.get("hub", {}).get(j, 0.0) if transfer_cost_j > 0 else 0.0
                        midpoint_rc = actual_cost_arc - dual_hub_i - dual_hub_j

                        max_L = max(self.config.T_max, od_L)
                        total_time = f_lab.time_upper + (max_L - b_lab.time_upper)
                        total_cost = f_lab.cost + b_lab.cost + actual_cost_arc
                        reduced_cost = f_lab.reduced_cost + b_lab.reduced_cost + duals_base + midpoint_rc

                        if reduced_cost < -1e-4:
                            complete_paths.append(Label(
                                cost=total_cost, reduced_cost=reduced_cost,
                                time_upper=total_time, risk_R=path_risk,
                                visited=full_visited, arc_sequence=full_arcs, feasible=True, rail_started=True, rail_ended=True
                            ))

        complete_paths.sort(key=lambda x: (x.reduced_cost, x.cost))
        return complete_paths

    def _dominates(self, l1: Label, l2: Label) -> bool:
        """
        S3-3: ng-route dominance rule.

        l1 dominates l2 if and only if:
        (a) l1 has no worse reduced_cost, time_upper, and risk_R, AND
        (b) l1.ng_memory is a subset of l2.ng_memory
            (l1 has fewer forced non-revisit obligations, so it can be extended
             to at least all paths that l2 can reach without cycling).
        """
        # Branch constraint checks
        if self.branch_constraints:
            for bc_type, bc_data in self.branch_constraints:
                if bc_type in ["rf_same", "rf_diff"]:
                    i_bc, j_bc = bc_data
                    has_i1 = i_bc in l1.visited
                    has_j1 = j_bc in l1.visited
                    has_i2 = i_bc in l2.visited
                    has_j2 = j_bc in l2.visited
                    if (has_i1 != has_i2) or (has_j1 != has_j2):
                        return False

        # #15: visited subset check for ng-route correctness
        # l1 can only dominate l2 if l1 has visited a subset of l2's non-ng nodes
        # (otherwise l1 might revisit nodes that l2 has already used)
        non_ng1 = l1.visited - l1.ng_memory
        non_ng2 = l2.visited - l2.ng_memory
        if not non_ng1.issubset(non_ng2):
            return False

        return (
            l1.reduced_cost <= l2.reduced_cost
            and l1.time_upper <= l2.time_upper
            and l1.risk_R >= l2.risk_R
            and l1.ng_memory <= l2.ng_memory  # S3-3: ng_memory subset check
            and l1.prev_mode == l2.prev_mode
            and l1.rail_started == l2.rail_started
            and l1.rail_ended == l2.rail_ended
        )

    def _is_dominated(self, label: Label, existing: list[Label]) -> bool:
        for l in existing:
            if self._dominates(l, label):
                return True
        return False
