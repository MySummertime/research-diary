from __future__ import annotations
from dataclasses import dataclass, field
import heapq
from itertools import count

from config import RHMVSPConfig
from ..core.network import Arc, MultimodalNetwork
from ..core.uncertainty import ZigzagUncertainVariable, UncertaintyEngine


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
    rail_services: tuple = ()  # 按行驶顺序保存所选铁路班次
    prev_mode: int = 0
    continuous_drive: float = 0.0
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
        # Timers and resource-limit diagnostics for pricing calls
        self.total_espprc_cpu_time = 0.0
        self.total_espprc_wall_time = 0.0
        self.last_solve_resource_limited = False
        self.resource_limit_hits = 0

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
                fuzzy_duration=(tri.a, tri.b, tri.c),
                time_dependent=True,
            )
        return self._arc_R_cache[cache_key]

    @staticmethod
    def _combine_risk_reliabilities(
        partial_reliabilities: tuple[float, ...],
        component_reliabilities: tuple[float, ...],
    ) -> float:
        """Combine series-system component reliabilities by the minimum rule."""
        values = partial_reliabilities + component_reliabilities
        return min(values) if values else 1.0

    def _get_transfer_time(self, previous_mode: int, next_mode: int) -> float:
        """Return the fixed transfer duration for the modal transition."""
        return self.config.transfer_time_hours(previous_mode, next_mode)

    def solve(self, origin: int, destination: int, od_idx: int, duals: dict,
              od_E: float, od_L: float, alpha_T: float, branch_constraints: list = None,
              exact: bool = True) -> list[Label]:
        """
        Solve bidirectional ESPPRC, optionally heuristically (exact=False).
        """
        self.last_solve_resource_limited = False
        self._exact_pricing = exact
        self.origin = origin
        self.destination = destination
        self.od_idx = od_idx
        self.od_E = od_E
        self.od_L = od_L
        self.branch_constraints = branch_constraints if branch_constraints is not None else []

        import time
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        # 1. Forward Labeling
        forward_labels = self._desrochers_label_correcting(origin, destination, duals, od_E, od_L, alpha_T, forward=True, exact=exact)
        num_f = sum(len(v) for v in forward_labels.values())

        # 2. Backward Labeling
        backward_labels = self._desrochers_label_correcting(destination, origin, duals, od_E, od_L, alpha_T, forward=False, exact=exact)
        num_b = sum(len(v) for v in backward_labels.values())

        # 3. Join Step (S3-1 fix applied inside _join_labels)
        all_columns = self._join_labels(forward_labels, backward_labels, od_L, alpha_T, duals)
        all_columns.sort(key=lambda x: x.reduced_cost)
        if len(all_columns) > self.config.bap_pricing_columns_limit:
            all_columns = all_columns[:self.config.bap_pricing_columns_limit]

        self.logger.debug(f"  Pricing OD {od_idx} (exact={exact}): F={num_f}, B={num_b}, Joined={len(all_columns)}")

        self.total_espprc_wall_time += (time.perf_counter() - start_wall)
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

        max_L = od_L
        init_time = od_E if forward else max_L
        init_ng = frozenset({start_node})

        q_veh = self.config.q_road
        dual_od = duals.get("od", {}).get(self.od_idx, 0.0)
        dual_v = duals.get("vehicle", {}).get(0, 0.0)
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

        # S3-2: Resource-balanced midpoint — tighter overlap for better pruning
        span = max_L - od_E
        half_ratio = getattr(self.config, 'pricing_overlap_ratio', 0.75)  # from config
        max_time_f = od_E + half_ratio * span
        min_time_b = max(od_E, max_L - half_ratio * span)

        # Global label cap to prevent join explosion
        total_labels_limit = self.config.total_labels_limit

        while queue:
            _, _, curr = heapq.heappop(queue)

            # If the label has been dominated and removed, skip
            if curr not in labels_at[curr.node]:
                continue

            adj_arcs = self.network.get_arcs_out(curr.node) if forward else self.network.get_arcs_in(curr.node)

            for arc in adj_arcs:
                next_node = arc.j if forward else arc.i
                
                
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
                            if exact:
                                self.last_solve_resource_limited = True
                                self.resource_limit_hits += 1
                            labels_at[new_label.node].sort(key=lambda x: x.reduced_cost)
                            labels_at[new_label.node] = labels_at[new_label.node][:max_keep]

                        # Global label cap: stop expanding if total labels exceed limit
                        total_labels = sum(len(v) for v in labels_at.values())
                        if total_labels > total_labels_limit:
                            if exact:
                                self.last_solve_resource_limited = True
                                self.resource_limit_hits += 1
                            break
                        heapq.heappush(queue, (new_label.reduced_cost, next(counter), new_label))

        return labels_at

    def _extend(self, parent: Label, arc: Arc, duals: dict, alpha_T: float, forward: bool) -> list[Label]:
        """Extend label along arc, generating multiple labels for all valid rail timetable branches."""
        tri = self.engine.make_transport_time(arc.distance, arc.mode)
        # All feasibility, timetable, and risk clocks use the same
        # alpha_T-quantile schedule.
        dt = tri.deterministic_equivalent_upper(alpha_T)
        if arc.mode == 1:
            prior_drive = parent.continuous_drive if parent.prev_mode == 1 else 0.0
            rest_time, new_continuous_drive = self.engine.road_drive_transition(prior_drive, dt)
        else:
            rest_time, new_continuous_drive = 0.0, 0.0

        q_veh = self.config.q_road
        q_rail = self.config.q_rail

        transfer_cost = 0.0
        transfer_time = 0.0
        node_R = 1.0

        # Safe transfer hub extraction
        if parent.prev_mode != 0 and not (parent.prev_mode == 1 and arc.mode == 1):
            tn = self.network.transfer_nodes.get(parent.node)
            if tn is None:
                return []
            transfer_time = self._get_transfer_time(parent.prev_mode, arc.mode)
            transfer_cost = self.config.transfer_unit_cost(parent.prev_mode, arc.mode)
            transfer_belief = tn.gamma_rr if parent.prev_mode == arc.mode == 2 else tn.gamma_ir
            node_R = 1.0 - transfer_belief

        actual_unit_cost = arc.unit_cost

        transport_cost_run = actual_unit_cost * q_veh
        actual_cost_run = transport_cost_run + transfer_cost

        # S4-1: 事故信度仅用于可靠性约束，不计入成本
        risk_penalty_ext = 0.0

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
                    [r for r in runs if r["departure_time"] >= t_arr],
                    key=lambda x: x["departure_time"]
                )
                if not self._exact_pricing:
                    valid_runs = valid_runs[:2]
                if not valid_runs:
                    return []
                for r in valid_runs:
                    t_start = r["departure_time"]
                    new_time = r["arrival_time"]
                    # S2-6: use time-keyed cache
                    arc_R = self._get_arc_R(arc, t_start, alpha_T)
                    
                    rail_dual = duals.get("rail", {}).get(r.get("train_id"), 0.0)
                    actual_cost_total = actual_cost_run + risk_penalty_ext
                    
                    new_cost_base = parent.cost + actual_cost_total
                    new_rc_base = parent.reduced_cost + actual_cost_total - rail_dual * q_veh
                    
                    new_risk = self._combine_risk_reliabilities((parent.risk_R,), (arc_R, node_R))
                    if self.config.skip_risk_constraint or new_risk >= self.config.alpha_R:
                        new_labels.append(Label(
                            cost=new_cost_base, reduced_cost=new_rc_base,
                            time_upper=new_time, time_lower=new_time, risk_R=new_risk,
                            visited=parent.visited | {next_node}, ng_memory=new_ng, node=next_node,
                            path=parent.path + ((next_node, arc.mode),),
                            arc_sequence=parent.arc_sequence + (arc,), rail_services=parent.rail_services + (r,),
                            prev_mode=arc.mode, continuous_drive=0.0, feasible=True, rail_started=new_rail_started, rail_ended=new_rail_ended
                        ))
            else:
                t_start = t_arr
                new_time = t_start + dt + rest_time
                arc_R = self._get_arc_R(arc, t_start, alpha_T)
                
                actual_cost_total = actual_cost_run + risk_penalty_ext
                new_cost_base = parent.cost + actual_cost_total
                new_rc_base = parent.reduced_cost + actual_cost_total
                
                new_risk = self._combine_risk_reliabilities((parent.risk_R,), (arc_R, node_R))
                if self.config.skip_risk_constraint or new_risk >= self.config.alpha_R:
                    new_labels.append(Label(
                        cost=new_cost_base, reduced_cost=new_rc_base,
                        time_upper=new_time, time_lower=new_time, risk_R=new_risk,
                        visited=parent.visited | {next_node}, ng_memory=new_ng, node=next_node,
                        path=parent.path + ((next_node, arc.mode),),
                        arc_sequence=parent.arc_sequence + (arc,), rail_services=parent.rail_services,
                        prev_mode=arc.mode, continuous_drive=new_continuous_drive, feasible=True, rail_started=new_rail_started, rail_ended=new_rail_ended
                    ))
        else:
            t_dep_latest = parent.time_upper - transfer_time
            if arc.mode == 2:
                runs = getattr(self.network, 'timetable', {}).get((arc.i, arc.j), [])
                valid_runs = sorted(
                    [r for r in runs if r["arrival_time"] <= t_dep_latest],
                    key=lambda x: x["arrival_time"],
                    reverse=True
                )
                if not self._exact_pricing:
                    valid_runs = valid_runs[:2]
                if not valid_runs:
                    return []
                for r in valid_runs:
                    new_time = r["departure_time"]
                    arc_R = self._get_arc_R(arc, new_time, alpha_T)
                    
                    t_end = r["arrival_time"]
                    rail_dual = duals.get("rail", {}).get(r.get("train_id"), 0.0)
                    actual_cost_total = actual_cost_run + risk_penalty_ext
                    new_cost_base = parent.cost + actual_cost_total
                    new_rc_base = parent.reduced_cost + actual_cost_total - rail_dual * q_veh
                    new_risk = self._combine_risk_reliabilities((parent.risk_R,), (arc_R, node_R))
                    if (self.config.skip_risk_constraint or new_risk >= self.config.alpha_R) and new_time >= self.od_E:
                        new_labels.append(Label(
                            cost=new_cost_base, reduced_cost=new_rc_base,
                            time_upper=new_time, time_lower=new_time, risk_R=new_risk,
                            visited=parent.visited | {next_node}, ng_memory=new_ng, node=next_node,
                            path=parent.path + ((next_node, arc.mode),),
                            arc_sequence=(arc,) + parent.arc_sequence, rail_services=(r,) + parent.rail_services,
                            prev_mode=arc.mode, continuous_drive=0.0, feasible=True, rail_started=new_rail_started, rail_ended=new_rail_ended
                        ))
            else:
                new_time = t_dep_latest - dt - rest_time
                arc_R = self._get_arc_R(arc, new_time, alpha_T)
                
                t_end = t_dep_latest
                actual_cost_total = actual_cost_run + risk_penalty_ext
                new_cost_base = parent.cost + actual_cost_total
                new_rc_base = parent.reduced_cost + actual_cost_total
                new_risk = self._combine_risk_reliabilities((parent.risk_R,), (arc_R, node_R))
                if (self.config.skip_risk_constraint or new_risk >= self.config.alpha_R) and new_time >= self.od_E:
                    new_labels.append(Label(
                        cost=new_cost_base, reduced_cost=new_rc_base,
                        time_upper=new_time, time_lower=new_time, risk_R=new_risk,
                        visited=parent.visited | {next_node}, ng_memory=new_ng, node=next_node,
                        path=parent.path + ((next_node, arc.mode),),
                        arc_sequence=(arc,) + parent.arc_sequence, rail_services=parent.rail_services,
                        prev_mode=arc.mode, continuous_drive=new_continuous_drive, feasible=True, rail_started=new_rail_started, rail_ended=new_rail_ended
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
                        if f_lab.prev_mode != 0 and not (f_lab.prev_mode == 1 and arc.mode == 1):
                            tn = self.network.transfer_nodes.get(i)
                            if tn is None:
                                continue
                            transfer_time = self._get_transfer_time(f_lab.prev_mode, arc.mode)
                            transfer_cost = self.config.transfer_unit_cost(f_lab.prev_mode, arc.mode)
                            transfer_belief = tn.gamma_rr if f_lab.prev_mode == arc.mode == 2 else tn.gamma_ir
                            node_R = 1.0 - transfer_belief

                        transfer_time_j = 0.0
                        transfer_cost_j = 0.0
                        node_R_j = 1.0
                        if b_lab.prev_mode != 0 and not (b_lab.prev_mode == 1 and arc.mode == 1):
                            tn_j = self.network.transfer_nodes.get(j)
                            if tn_j is None:
                                continue
                            transfer_time_j = self._get_transfer_time(arc.mode, b_lab.prev_mode)
                            transfer_cost_j = self.config.transfer_unit_cost(arc.mode, b_lab.prev_mode)
                            transfer_belief_j = tn_j.gamma_rr if arc.mode == b_lab.prev_mode == 2 else tn_j.gamma_ir
                            node_R_j = 1.0 - transfer_belief_j

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

                        midpoint_services = (r,) if arc.mode == 2 else ()

                        if t_end_arc > b_lab.time_upper:
                            continue

                        # S2-6: time-keyed reliability
                        arc_R = self._get_arc_R(arc, t_start_arc, alpha_T)
                        path_risk = self._combine_risk_reliabilities(
                            (f_lab.risk_R, b_lab.risk_R),
                            (arc_R, node_R, node_R_j),
                        )
                        if not self.config.skip_risk_constraint and path_risk < self.config.alpha_R:
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
                            if not (a1.mode == 1 and a2.mode == 1):
                                tn_node = self.network.transfer_nodes.get(a1.j)
                                if tn_node:
                                    trans_hubs.add(a1.j)

                        if self.branch_constraints:
                            valid_bc = True
                            for bc_type, bc_data in self.branch_constraints:
                                if bc_type in ("arc_forbid", "arc_enforce"):
                                    od_bc, u_bc, v_bc, m_bc = bc_data
                                    if od_bc == self.od_idx:
                                        has_arc = any(a.i == u_bc and a.j == v_bc and a.arc_id == m_bc for a in full_arcs)
                                        if bc_type == "arc_forbid" and has_arc:
                                            valid_bc = False
                                            break
                                        elif bc_type == "arc_enforce" and not has_arc:
                                            valid_bc = False
                                            break
                            if not valid_bc:
                                continue

                        max_L = od_L

                        # S3-1 FIX: Compute exact midpoint arc cost and combine with labels
                        a_unit = arc.unit_cost
                        transport_cost_run = a_unit * q_veh
                        actual_cost_arc = transport_cost_run + transfer_cost + transfer_cost_j

                        # 事故信度仅用于可靠性约束，不计入成本
                        rail_dual = 0.0
                        if arc.mode == 2:
                            rail_dual = duals.get("rail", {}).get(r.get("train_id"), 0.0)
                        midpoint_rc = actual_cost_arc - rail_dual * q_veh

                        if arc.mode == 1:
                            prior_drive = f_lab.continuous_drive if f_lab.prev_mode == 1 else 0.0
                            midpoint_rest, _ = self.engine.road_drive_transition(prior_drive, dt_arc)
                        else:
                            midpoint_rest = 0.0
                        total_time = f_lab.time_upper + dt_arc + midpoint_rest + (max_L - b_lab.time_upper)
                        total_cost = f_lab.cost + b_lab.cost + actual_cost_arc
                        full_services = f_lab.rail_services + midpoint_services + b_lab.rail_services
                        reduced_cost = (
                            f_lab.reduced_cost + b_lab.reduced_cost + duals_base
                            + midpoint_rc
                        )

                        if reduced_cost < -1e-4:
                            complete_paths.append(Label(
                                cost=total_cost, reduced_cost=reduced_cost,
                                time_upper=total_time, risk_R=path_risk,
                                visited=full_visited, arc_sequence=full_arcs,
                                rail_services=full_services,
                                feasible=True, rail_started=True, rail_ended=True
                            ))

        complete_paths.sort(key=lambda x: (x.reduced_cost, x.cost))
        return complete_paths

    def _dominates(self, l1: Label, l2: Label) -> bool:
        """
        S3-3: ng-route dominance rule.

        l1 dominates l2 if and only if:
        (a) l1 has no worse reduced_cost, time_upper, and remaining risk reliability, AND
        (b) l1.ng_memory is a subset of l2.ng_memory
            (l1 has fewer forced non-revisit obligations, so it can be extended
             to at least all paths that l2 can reach without cycling).
        """
        # Branch constraint checks (OD-specific arc branching only)
        # No rf_same/rf_diff constraints — removed with Ryan-Foster branching.

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
            and l1.continuous_drive <= l2.continuous_drive
            and l1.rail_started == l2.rail_started
            and l1.rail_ended == l2.rail_ended
        )

    def _is_dominated(self, label: Label, existing: list[Label]) -> bool:
        for l in existing:
            if self._dominates(l, label):
                return True
        return False
