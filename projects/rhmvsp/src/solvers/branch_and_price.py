from __future__ import annotations
import heapq
import time
import os
import numpy as np
import networkx as nx
import pulp
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None
    print("Gurobipy not found. Falling back to Pulp solvers.")

from ..utils.timing import Timer
from dataclasses import dataclass, field

from ..core.network import Arc, MultimodalNetwork
from ..core.reliability import ReliabilityAnalyzer, PathReliabilityResult
from ..core.uncertainty import TrapezoidalFuzzy, ZigzagUncertainVariable, UncertaintyEngine
from .branching import BranchingStrategy
from .master_problem import MasterProblem
from .pricing_problem import PricingProblem


@dataclass
class Column:
    """A column in the master problem representing a feasible vehicle route with exact timing."""

    col_id: int
    od_idx: int
    vehicle_idx: int
    arcs: list[Arc]
    arc_modes: list[int]
    transfer_nodes: list[int]
    departure_time: float
    total_cost: float
    transport_cost: float = 0.0
    transfer_cost: float = 0.0
    arrival_time: float = 0.0
    time_R: float = 1.0
    bottleneck_capacity: float = 20.0
    lp_value: float = 0.0
    transport_cost_upper_bound: float = 0.0
    risk_R: float = 1.0
    origin_hub: int = -1       # hub ID for west-side origin hub (fleet base)
    destination_hub: int = -1  # hub ID used for east-side delivery
    rail_services: list[dict] = field(default_factory=list)  # 按行驶顺序使用的铁路班次


@dataclass(order=True)
class BAPNode:
    """Node in the Branch-and-Price tree."""

    lower_bound: float
    depth: int
    node_id: int
    parent_id: int = -1
    branch_constraints: list[tuple] = field(default_factory=list, compare=False)
    columns: list = field(default_factory=list, compare=False)

    def __post_init__(self):
        self.priority = (self.lower_bound, -self.depth, self.node_id)


class BranchAndPriceController:
    """
    Main controller for the exact Branch-and-Price algorithm.
    Implements Best-First Search tree exploration with exact single-stage recourse.
    """

    def __init__(self, config, instance, logger):
        self.config = config
        self.instance = instance
        self.logger = logger
        self.network = MultimodalNetwork.from_instance(instance)
        self.network._build_ng_neighbors(self.config.bap_ng_route_size)
        # Ensure per-OD time reliability uses active config.alpha_T so
        # sensitivity runs that modify config.alpha_T take effect.
        for od in self.network.od_pairs:
            od.alpha_T_od = self.config.alpha_T
        self.engine = UncertaintyEngine(config)
        self.reliability = ReliabilityAnalyzer(config, self.engine)

        # Vehicle limit is dynamically provided by the instance configuration
        self.mp = MasterProblem(config, self.network, logger, instance=instance)
        self.pp = PricingProblem(config, self.network, self.engine, logger)

        self.best_ub = float("inf")
        self.best_solution = None
        self.open_nodes = []
        self.node_counter = 0
        self.col_gen_iterations = 0
        self.cg_history: list[tuple[int, float, float]] = []  # [(iter, lp_bound, best_ub)]

        # Observational benchmark metrics; these do not alter solver decisions.
        self.processed_node_count = 0
        self.root_lp_bound = None
        self.root_pricing_termination = None
        self.root_bound_available = False
        self.last_cg_termination = None

    def solve(self) -> dict:
        """Main solve loop using Best-First Search."""
        import time
        self.solve_start_time = time.time()
        self.solve_start_wall = time.perf_counter()
        self.solve_start_cpu = time.process_time()

        root_cols = self._generate_initial_columns()
        self.cg_history = []
        root = BAPNode(lower_bound=-self.config.big_m, depth=0, node_id=0, columns=root_cols)
        heapq.heappush(self.open_nodes, root)
        self.node_counter += 1

        while self.open_nodes:
            if time.time() - self.solve_start_time > self.config.time_limit:
                self.logger.warning(f"Time limit of {self.config.time_limit}s reached. Terminating branch-and-bound early.")
                break

            curr_node = heapq.heappop(self.open_nodes)

            if curr_node.lower_bound >= self.best_ub:
                continue

            self.logger.info(
                f"Solving Node {curr_node.node_id} (LB={curr_node.lower_bound:.2f}, depth={curr_node.depth})"
            )

            self.processed_node_count += 1
            lp_val, is_optimal = self._column_generation(curr_node)
            if curr_node.node_id == 0 and lp_val is not None:
                self.root_lp_bound = lp_val
                self.root_pricing_termination = self.last_cg_termination
                self.root_bound_available = is_optimal

            if lp_val is None:
                continue

            curr_node.lower_bound = lp_val

            if lp_val >= self.best_ub:
                continue

            # LP-only mode: use fractional lambda values from CG (no MIP re-solve)
            if self.config.lp_only:
                self.logger.info("  LP-only mode: extracting fractional LP solution.")
                sol = self._extract_lp_solution(curr_node, lp_val)
                sol["lp_bound"] = lp_val
                sol["gap_pct"] = 0.0
                return self._attach_performance_metrics(sol)

            # In root node or when columns are sufficient, extract exact physical integer MIP solution immediately
            sol = self._extract_solution(curr_node)
            if sol and sol.get("total_cost", float("inf")) < self.best_ub:
                self.best_ub = sol["total_cost"]
                self.best_solution = sol
                self.logger.info(
                    f"  Integer solution found at Node {curr_node.node_id} "
                    f"(Total Cost (c_total)=¥{self.best_ub:,.2f}, UB updated)."
                )
                
                self.logger.info("  Continuing search to prove optimality (or until time limit).")

            if self._is_integer(curr_node):
                continue

            self._branch(curr_node)

        if self.best_solution:
            return self._attach_performance_metrics(self.best_solution)
        return self._attach_performance_metrics({"status": "infeasible"})

    def _attach_performance_metrics(self, solution: dict) -> dict:
        """Attach reproducible timing and root-node benchmark metadata."""
        total_wall = time.perf_counter() - self.solve_start_wall
        total_cpu = time.process_time() - self.solve_start_cpu
        master_wall = self.mp.total_master_wall_time
        pricing_wall = self.pp.total_espprc_wall_time
        other_wall = max(0.0, total_wall - master_wall - pricing_wall)
        root_gap_vs_incumbent = None
        incumbent = solution.get("total_cost")
        if (
            self.root_bound_available
            and incumbent is not None
            and abs(float(incumbent)) > self.config.tolerance
        ):
            root_gap_vs_incumbent = (
                100.0 * (float(incumbent) - float(self.root_lp_bound))
                / abs(float(incumbent))
            )
        solution.update({
            "solve_wall_time": total_wall,
            "solve_cpu_time": total_cpu,
            "master_wall_time": master_wall,
            "master_cpu_time": self.mp.total_master_cpu_time,
            "master_lp_wall_time": self.mp.total_master_lp_wall_time,
            "master_mip_wall_time": self.mp.total_master_mip_wall_time,
            "pricing_wall_time": pricing_wall,
            "pricing_cpu_time": self.pp.total_espprc_cpu_time,
            "other_wall_time": other_wall,
            "processed_node_count": self.processed_node_count,
            "root_lp_bound_bp": self.root_lp_bound,
            "root_pricing_termination": self.root_pricing_termination,
            "root_bound_available": self.root_bound_available,
            "pricing_resource_limit_hits": self.pp.resource_limit_hits,
            "root_gap_vs_bp_incumbent_pct": root_gap_vs_incumbent,
            # gapBP requires a common cross-method Z_best and is aggregated later.
            "gap_bp_pct": None,
        })
        return solution

    def _column_generation(self, node: BAPNode) -> tuple[float | None, bool]:
        """Run exact column generation loop at a node."""
        iteration = 0
        max_iters = self.config.bap_max_iterations
        lp_val = None
        pricing_resource_limited = False
        self.last_cg_termination = "unknown"

        existing_fingerprints = set()
        for col in node.columns:
            fp = (tuple((a.i, a.j, a.arc_id) for a in col.arcs),
                  tuple(run["train_id"] for run in col.rail_services), round(col.arrival_time, 1))
            existing_fingerprints.add(fp)

        previous_duals = None
        alpha_stab = self.config.stabilization_mu

        import time
        while iteration < max_iters:
            if time.time() - getattr(self, 'solve_start_time', time.time()) > self.config.time_limit:
                self.logger.warning(f"Time limit of {self.config.time_limit}s reached during Column Generation.")
                self.last_cg_termination = "time_limit"
                break

            iteration += 1
            self.col_gen_iterations += 1
            self.logger.info(
                f"  CG iteration {iteration}: solving RMP with {len(node.columns)} columns"
            )
            lp_val, raw_duals = self.mp.solve(node.columns, node.branch_constraints)
            if lp_val is None:
                self.last_cg_termination = "rmp_failure"
                return None, False

            self.cg_history.append((self.col_gen_iterations, lp_val, self.best_ub))

            # Dual Stabilization (Moving Average)
            if self.config.bap_stabilization and previous_duals is not None:
                duals = {}
                for k in raw_duals:
                    if isinstance(raw_duals[k], dict):
                        duals[k] = {}
                        for sub_k in raw_duals[k]:
                            old_val = previous_duals.get(k, {}).get(sub_k, raw_duals[k][sub_k])
                            duals[k][sub_k] = alpha_stab * old_val + (1 - alpha_stab) * raw_duals[k][sub_k]
                    else:
                        old_val = previous_duals.get(k, raw_duals[k])
                        duals[k] = alpha_stab * old_val + (1 - alpha_stab) * raw_duals[k]
            else:
                duals = raw_duals
            
            # Boost duals for ODs that still have no columns, so pricing can find paths for them
            big_m = getattr(self.config, 'big_m', 1e6)
            od_duals = duals.get("od", {})
            for i in range(len(self.network.od_pairs)):
                has_col = any(col.od_idx == i for col in node.columns)
                if not has_col:
                    od_duals[i] = max(od_duals.get(i, 0.0), big_m * 0.1)
            
            previous_duals = duals

            new_cols_found = False
            # 1. Heuristic Pricing (exact=False) first to find columns quickly
            for i, od in enumerate(self.network.od_pairs):
                self.logger.info(f"  CG iteration {iteration}: heuristic pricing OD_{i + 1}")
                earliest_start = self._od_collection_ready_time(od)
                labels = self.pp.solve(
                    od.origin, od.destination, i, duals, earliest_start, od.time_window_end, od.alpha_T_od,
                    node.branch_constraints, exact=False
                )

                new_node_cols = []
                for lab in labels:
                    fp = (tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence),
                          tuple(run["train_id"] for run in lab.rail_services), round(lab.time_upper, 1))
                    if fp not in existing_fingerprints:
                        col = self._label_to_column(lab, i)
                        if col is not None:
                            new_node_cols.append(col)
                        existing_fingerprints.add(fp)

                if new_node_cols:
                    node.columns.extend(new_node_cols)
                    new_cols_found = True

            # 2. Always run Exact Pricing (exact=True) to prove optimality
            for i, od in enumerate(self.network.od_pairs):
                self.logger.info(f"  CG iteration {iteration}: exact pricing OD_{i + 1}")
                earliest_start = self._od_collection_ready_time(od)
                labels = self.pp.solve(
                    od.origin, od.destination, i, duals, earliest_start, od.time_window_end, od.alpha_T_od,
                    node.branch_constraints, exact=True
                )

                if self.pp.last_solve_resource_limited:
                    pricing_resource_limited = True

                new_node_cols = []
                for lab in labels:
                    fp = (tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence),
                          tuple(run["train_id"] for run in lab.rail_services), round(lab.time_upper, 1))
                    if fp not in existing_fingerprints:
                        col = self._label_to_column(lab, i)
                        if col is not None:
                            new_node_cols.append(col)
                        existing_fingerprints.add(fp)

                if new_node_cols:
                    node.columns.extend(new_node_cols)
                    new_cols_found = True

            if not new_cols_found:
                self.last_cg_termination = (
                    "pricing_resource_limit" if pricing_resource_limited else "no_new_column"
                )
                break

            self.logger.debug(
                f"  Iteration {iteration}: LP={lp_val:.2f}, PoolSize={len(node.columns)}"
            )

        if iteration >= max_iters and self.last_cg_termination == "unknown":
            self.last_cg_termination = "iteration_limit"
            self.logger.warning(f"  Column generation reached max iterations ({max_iters})")

        return lp_val, self.last_cg_termination == "no_new_column"

    def _is_integer(self, node: BAPNode) -> bool:
        """Check integrality of the column-dispatch variables."""
        tol = self.config.tolerance
        for col in node.columns:
            if tol < abs(col.lp_value - round(col.lp_value)) < 1.0 - tol:
                return False
        return True

    def _apply_branch_constraints(self, columns, constraints):
        """Filter columns based on OD-specific arc branching constraints."""
        if not constraints:
            return columns

        filtered = []
        for col in columns:
            valid = True
            for c_type, c_data in constraints:
                if c_type == "arc_forbid":
                    od_idx, u, v, m = c_data
                    if col.od_idx == od_idx and self._contains_arc(col, u, v, m):
                        valid = False
                        break
                elif c_type == "arc_enforce":
                    od_idx, u, v, m = c_data
                    if col.od_idx == od_idx and not self._contains_arc(col, u, v, m):
                        valid = False
                        break
            if valid:
                filtered.append(col)
        return filtered

    def _label_to_column(self, label, od_idx) -> Column | None:
        """Convert a PricingProblem Label to a MasterProblem Column with exact Wait-Shift timing."""
        q_veh = self.config.q_road
        q_rail = self.config.q_rail

        transport_cost = 0.0
        transport_cost_upper_bound = 0.0
        for a in label.arc_sequence:
            cost_a = a.unit_cost * q_veh
            transport_cost += cost_a
            transport_cost_upper_bound += cost_a

        transfer_nodes_list = []
        transfer_cost = 0.0
        for k in range(len(label.arc_sequence) - 1):
            a1, a2 = label.arc_sequence[k], label.arc_sequence[k + 1]
            if not (a1.mode == 1 and a2.mode == 1):
                tn_id = a1.j
                transfer_nodes_list.append(tn_id)
                tn = self.network.transfer_nodes.get(tn_id)
                if tn:
                    transfer_cost += self.config.transfer_unit_cost(a1.mode, a2.mode)
        
        actual_cost_without_risk = transport_cost + transfer_cost
        # Wait-Shift Strategy: Recalculate deterministic upper time to capture early waiting time at rail hubs
        od = self.network.od_pairs[od_idx]
        alpha_T = self.config.alpha_T
        first_rail_idx = next(
            (i for i, arc in enumerate(label.arc_sequence) if arc.mode == 2),
            len(label.arc_sequence),
        )
        west_arcs = label.arc_sequence[:first_rail_idx]
        west_duration = 0.0
        west_continuous_drive = 0.0
        for arc in west_arcs:
            road_time = self.engine.make_transport_time(
                arc.distance, arc.mode
            ).deterministic_equivalent_upper(alpha_T)
            rest_time, west_continuous_drive = self.engine.road_drive_transition(
                west_continuous_drive, road_time
            )
            west_duration += road_time + rest_time
        west_transfer_duration = 0.0
        if west_arcs:
            transfer_node = self.network.transfer_nodes.get(west_arcs[-1].j)
            if transfer_node:
                west_transfer_duration = self.config.transfer_time_hours(1, 2)
        first_rail_departure = (
            label.rail_services[0]["departure_time"]
            if label.rail_services else od.time_window_start + west_duration
        )
        # Demand may wait at its origin.  Start each collection batch as late
        # as possible while still catching the selected first rail service.
        actual_departure_time = max(
            od.time_window_start,
            first_rail_departure - west_transfer_duration - west_duration,
        )
        t_arr = actual_departure_time
        total_wait_time = 0.0
        prev_mode = 0
        continuous_drive = 0.0
        rail_service_idx = 0

        # S2-4: Variables for rigorous M{Z~ <= time_window_end} measure tracking
        z_a_dur = 0.0
        z_b_dur = 0.0
        z_c_dur = 0.0
        z_d_dur = 0.0

        for i, arc in enumerate(label.arc_sequence):
            if i > 0:
                if prev_mode != 0 and not (prev_mode == 1 and arc.mode == 1):
                    tn = self.network.transfer_nodes.get(arc.i)
                    if tn:
                        transfer_time = self.config.transfer_time_hours(prev_mode, arc.mode)
                        t_arr += transfer_time
                        z_a_dur += transfer_time
                        z_b_dur += transfer_time
                        z_c_dur += transfer_time
                        z_d_dur += transfer_time
            if arc.mode == 2:
                continuous_drive = 0.0
                # A route-service combination is one immutable plan.  Reuse
                # the service selected by pricing instead of silently changing
                # it while materializing the column.
                if rail_service_idx >= len(label.rail_services):
                    return None
                selected_run = label.rail_services[rail_service_idx]
                rail_service_idx += 1
                if (
                    selected_run.get("origin_hub") != arc.i
                    or selected_run.get("destination_hub") != arc.j
                    or selected_run["departure_time"] < t_arr
                ):
                    return None
                t_start = selected_run["departure_time"]
                wait = max(0.0, t_start - t_arr)
                total_wait_time += wait
                t_end = selected_run["arrival_time"]
                t_arr = t_end
                # S2-1: Add wait time to fuzzy accumulators (deterministic component).
                # Without this, the fuzzy Z~ underestimates total time, making time_R
                # systematically too optimistic.
                z_a_dur += arc.time_a + wait
                z_b_dur += arc.time_b + wait
                z_c_dur += arc.time_b + wait
                z_d_dur += arc.time_c + wait
            else:
                tri = self.engine.make_transport_time(arc.distance, arc.mode)
                arc_de = tri.deterministic_equivalent_upper(alpha_T)
                if prev_mode != 1:
                    continuous_drive = 0.0
                rest_time, continuous_drive = self.engine.road_drive_transition(
                    continuous_drive, arc_de
                )
                t_start = t_arr
                t_end = t_start + arc_de + rest_time
                t_arr = t_end
                
                z_a_dur += tri.a + rest_time
                z_b_dur += tri.b + rest_time
                z_c_dur += tri.b + rest_time
                z_d_dur += tri.c + rest_time

            prev_mode = arc.mode

        # The deadline for this column. Always use the raw instance time_window_end.
        l_effective = od.time_window_end

        # ---------------------------------------------------------------
        # COLUMN COST = transport + transfer (no delay penalty)
        # ---------------------------------------------------------------
        # NOTE: z_*_dur includes arc travel times, transfer times, AND rail timetable
        # wait times. Using od.time_window_start as the base so z_*_dur measures
        # total elapsed time from the earliest possible start.
        trap_Z = TrapezoidalFuzzy(
            a=actual_departure_time + z_a_dur,
            b=actual_departure_time + z_b_dur,
            c=actual_departure_time + z_c_dur,
            d=actual_departure_time + z_d_dur,
        )
        actual_cost = actual_cost_without_risk
        
        # S2-4: Compute continuous time_R = M{Z~ <= time_window_end} in [0,1].
        time_R = trap_Z.uncertain_measure_leq(l_effective)
        if t_arr > min(l_effective, self.config.T_max) or time_R < alpha_T:
            return None

        transfer_objects = [
            self.network.transfer_nodes[node_id]
            for node_id in transfer_nodes_list
            if node_id in self.network.transfer_nodes
        ]
        exact_risk = self.reliability.analyze_path(
            list(label.arc_sequence),
            [arc.mode for arc in label.arc_sequence],
            transfer_objects,
            od,
            actual_departure_time,
            t_arr,
            time_R,
            list(label.rail_services),
        ).plan_risk_reliability
        if not self.config.skip_risk_constraint and exact_risk < self.config.alpha_R:
            return None

        bottleneck_capacity = q_veh

        col = Column(
            col_id=self.mp.next_col_id,
            od_idx=od_idx,
            vehicle_idx=0,
            arcs=list(label.arc_sequence),
            arc_modes=[a.mode for a in label.arc_sequence],
            transfer_nodes=transfer_nodes_list,
            departure_time=actual_departure_time,
            arrival_time=t_arr,
            total_cost=actual_cost,
            transport_cost=transport_cost,
            transfer_cost=transfer_cost,
            risk_R=exact_risk,
            time_R=time_R,
            bottleneck_capacity=bottleneck_capacity,
            transport_cost_upper_bound=transport_cost_upper_bound,
            destination_hub=self.network.od_pairs[od_idx].destination_hub,
            origin_hub=self.network.od_pairs[od_idx].origin_hub,
            rail_services=[dict(run) for run in label.rail_services],
        )
        self.mp.next_col_id += 1
        return col

    def _od_collection_ready_time(self, od) -> float:
        """Earliest loaded departure; vehicles are pre-positioned before Day 1."""
        return od.time_window_start

    def _generate_initial_columns(self):
        """Generate rigorous initial columns by solving the Pricing Problem with high (big_m) duals."""
        cols = []
        existing_fingerprints = set()
        dummy_duals = {
            "od": {i: getattr(self.config, 'big_m', 1e6) for i in range(len(self.network.od_pairs))},
            "vehicle": {},
            "trans_out": {},
            "trans_in": {},
        }
        for i, od in enumerate(self.network.od_pairs):
            earliest_start = self._od_collection_ready_time(od)
            # Seed every service day in the configured horizon. A single solve
            # from time zero would let the heuristic's earliest-run shortlist
            # hide later-day dispatches, even though each dispatch has its own
            # 40-FEU capacity in the master problem.
            release_times = []
            release = float(earliest_start)
            while release < min(float(od.time_window_end), float(self.config.T_max)):
                release_times.append(release)
                release += 24.0

            for release_time in release_times:
                labels = self.pp.solve(
                    od.origin, od.destination, i, dummy_duals, release_time,
                    od.time_window_end, od.alpha_T_od,
                    branch_constraints=[], exact=False
                )
                labels.sort(key=lambda lab: (
                    lab.rail_services[0]["departure_time"] if lab.rail_services else float("inf"),
                    lab.rail_services[-1]["arrival_time"] if lab.rail_services else float("inf"),
                    lab.reduced_cost,
                ))
                retained_for_release = 0
                for lab in labels:
                    fp = (tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence),
                          tuple(run["train_id"] for run in lab.rail_services), round(lab.time_upper, 1))
                    if fp not in existing_fingerprints:
                        col = self._label_to_column(lab, i)
                        if col is not None:
                            existing_fingerprints.add(fp)
                            cols.append(col)
                            retained_for_release += 1
                            if retained_for_release >= self.config.initial_columns_per_od_service_day:
                                break
        self.logger.info(f"  Generated {len(cols)} initial columns")
        return cols

    def _branch(self, node: BAPNode):
        """Branch on an OD-specific arc used by fractional columns.

        Find the OD with the most fractional columns, then branch on an OD-specific
                 arc (i,j,mode) scoped to that OD only. This avoids the mathematical
                 invalidity of Ryan-Foster node-pair branching (which incorrectly assumes
                 that visiting a physical node is a binary decision, when paths can visit
                 a node multiple times or partially).
        """
        tol = self.config.tolerance

        # Find the OD index with the most fractional column values
        od_fractional_counts = {}
        for col in node.columns:
            if tol < abs(col.lp_value - round(col.lp_value)) < 1.0 - tol:
                od_fractional_counts[col.od_idx] = od_fractional_counts.get(col.od_idx, 0) + 1

        # Pick the OD with the most fractional columns
        target_od = max(od_fractional_counts, key=od_fractional_counts.get) if od_fractional_counts else None

        if target_od is not None:
            # Find the best arc within this OD's columns for branching
            candidate = BranchingStrategy.get_arc_candidate_for_od(node.columns, target_od, node.branch_constraints)
            if candidate:
                u, v, m = candidate
                self.logger.info(f"  OD-specific Arc Branching: OD_{target_od} Arc ({u}, {v}, mode={m})")
                
                child1 = BAPNode(
                    lower_bound=node.lower_bound, depth=node.depth + 1, node_id=self.node_counter, parent_id=node.node_id,
                    branch_constraints=node.branch_constraints + [("arc_forbid", (target_od, u, v, m))],
                    columns=[c for c in node.columns if not (c.od_idx == target_od and self._contains_arc(c, u, v, m))],
                )
                self.node_counter += 1
                heapq.heappush(self.open_nodes, child1)
                
                child2 = BAPNode(
                    lower_bound=node.lower_bound, depth=node.depth + 1, node_id=self.node_counter, parent_id=node.node_id,
                    branch_constraints=node.branch_constraints + [("arc_enforce", (target_od, u, v, m))],
                    columns=[c for c in node.columns if not (c.od_idx == target_od) or self._contains_arc(c, u, v, m)],
                )
                self.node_counter += 1
                heapq.heappush(self.open_nodes, child2)

    def _contains_arc(self, col, u, v, arc_id) -> bool:
        for arc in col.arcs:
            if arc.i == u and arc.j == v and arc.arc_id == arc_id:
                return True
        return False

    # Note: _check_rf (Ryan-Foster node-pair branching) has been removed.
    # It is mathematically invalid for this problem because paths can visit
    # the same physical node multiple times or partially, making the binary
    # node-visit decision meaningless. Replaced by OD-specific arc branching.

    def _stitch_sub_fleet(self, sub_tasks: list, fleet_name: str, start_v_id: int, max_vehicles: int = None) -> tuple[int, list[dict], float, list[dict]]:
        """求解一个省级调度中心所辖公路车队的任务衔接问题。"""
        if not sub_tasks:
            return 0, [], 0.0, []

        sub_tasks.sort(key=lambda t: t['departure_time'])

        if self.config.solver_name == "gurobi" and gp:
            try:
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
                stitch_prob = gp.Model(f"Stitching_{fleet_name}", env=env)
                stitch_prob.setParam('OutputFlag', 0)  # Suppress Gurobi output
                stitch_prob.setParam("Seed", self.config.seed)
                stitch_prob.setParam("Threads", self.config.gurobi_threads)
            except gp.GurobiError as e:
                self.logger.warning(f"Gurobi license error: {e}. Falling back to Pulp.")
                stitch_prob = pulp.LpProblem(f"Stitching_{fleet_name}", pulp.LpMinimize)
                use_gurobi = False
            else:
                use_gurobi = True
        else:
            stitch_prob = pulp.LpProblem(f"Stitching_{fleet_name}", pulp.LpMinimize)
            use_gurobi = False

        x_vars = {}
        start_vars = {}
        valid_edges = []
        initial_moves = {}
        q_veh = self.config.q_road

        for u_idx, u in enumerate(sub_tasks):
            for v_idx, v in enumerate(sub_tasks):
                if u_idx != v_idx:
                    if u.get("province_hub_id") != v.get("province_hub_id"):
                        continue
                    if u['destination'] == v['origin']:
                        req_gap = 0.0
                        dist = 0.0
                    else:
                        dist = self.network.get_exact_road_distance(u['destination'], v['origin'])
                        # Use nominal travel time (not fuzzy upper bound) for stitching.
                        # Fuzzy uncertainty is already handled by path reliability constraints.
                        drive_time = dist / self.config.road_speed
                        rest_time = self.engine.road_rest_time(drive_time)
                        req_gap = drive_time + rest_time

                    if u['arrival_time'] + req_gap <= v['departure_time']:
                        empty_drive_fee = self.config.c_empty_reposition_per_km * dist
                        valid_edges.append((u_idx, v_idx, empty_drive_fee, dist))

        # Initial vehicles are assumed to be pre-positioned at their first OD
        # tasks before Day 1 00:00, so no initial travel test or cost applies.
        for task_idx in range(len(sub_tasks)):
            initial_moves[task_idx] = {"feasible": True}

        if use_gurobi:
            for edge in valid_edges:
                u, v, empty_fee, dist = edge
                x_vars[(u, v)] = stitch_prob.addVar(vtype=GRB.BINARY, name=f"x_{fleet_name}_{u}_{v}")
            for task_idx in range(len(sub_tasks)):
                start_vars[task_idx] = stitch_prob.addVar(
                    vtype=GRB.BINARY,
                    ub=1.0 if initial_moves[task_idx]["feasible"] else 0.0,
                    name=f"start_{fleet_name}_{task_idx}",
                )
            stitch_prob.update()
            
            obj_expr = 0
            for edge in valid_edges:
                u, v, empty_fee, dist = edge
                obj_expr += empty_fee * x_vars[(u, v)]
            stitch_prob.setObjective(obj_expr, GRB.MINIMIZE)

            for u_idx in range(len(sub_tasks)):
                if not initial_moves[u_idx]["feasible"]:
                    stitch_prob.addConstr(
                        start_vars[u_idx] == 0,
                        name=f"initial_feasible_{fleet_name}_{u_idx}",
                    )
                out_edges = [edge for edge in valid_edges if edge[0] == u_idx]
                if out_edges:
                    stitch_prob.addConstr(
                        gp.quicksum(x_vars[(edge[0], edge[1])] for edge in out_edges) <= 1,
                        name=f"out_{fleet_name}_{u_idx}"
                    )
                in_edges = [edge for edge in valid_edges if edge[1] == u_idx]
                stitch_prob.addConstr(
                    start_vars[u_idx]
                    + gp.quicksum(x_vars[(edge[0], edge[1])] for edge in in_edges)
                    == 1,
                    name=f"balance_{fleet_name}_{u_idx}",
                )

            if max_vehicles is not None:
                stitch_prob.addConstr(
                    gp.quicksum(start_vars.values()) <= max_vehicles,
                    name=f"fleet_limit_{fleet_name}",
                )
            
            stitch_prob.optimize()
            
            # Check optimization status
            if stitch_prob.status == GRB.OPTIMAL or stitch_prob.status == GRB.SUBOPTIMAL:
                solution_found = True
            elif stitch_prob.status == GRB.INFEASIBLE:
                raise RuntimeError(
                    f"[{fleet_name}] 车辆拼接不可行；主问题不得返回违反车队规模的方案"
                )
            else:
                solution_found = False
                self.logger.warning(f"Gurobi did not find an optimal solution for stitching problem. Status: {stitch_prob.status}")

        else: # Use Pulp
            for edge in valid_edges:
                u, v, empty_fee, dist = edge
                x_vars[(u, v)] = pulp.LpVariable(f"x_{fleet_name}_{u}_{v}", 0, 1, pulp.LpBinary)
            for task_idx in range(len(sub_tasks)):
                start_vars[task_idx] = pulp.LpVariable(
                    f"start_{fleet_name}_{task_idx}",
                    0,
                    1 if initial_moves[task_idx]["feasible"] else 0,
                    pulp.LpBinary,
                )

            obj_expr = []
            for edge in valid_edges:
                u, v, empty_fee, dist = edge
                obj_expr.append(empty_fee * x_vars[(u, v)])
            stitch_prob += pulp.lpSum(obj_expr)

            for u_idx in range(len(sub_tasks)):
                if not initial_moves[u_idx]["feasible"]:
                    stitch_prob += (
                        start_vars[u_idx] == 0,
                        f"initial_feasible_{fleet_name}_{u_idx}",
                    )
                out_edges = [edge for edge in valid_edges if edge[0] == u_idx]
                if out_edges:
                    stitch_prob += pulp.lpSum(x_vars[(edge[0], edge[1])] for edge in out_edges) <= 1, f"out_{fleet_name}_{u_idx}"
                in_edges = [edge for edge in valid_edges if edge[1] == u_idx]
                stitch_prob += (
                    start_vars[u_idx]
                    + pulp.lpSum(x_vars[(edge[0], edge[1])] for edge in in_edges)
                    == 1,
                    f"balance_{fleet_name}_{u_idx}",
                )

            if max_vehicles is not None:
                stitch_prob += (
                    pulp.lpSum(start_vars.values()) <= max_vehicles,
                    f"fleet_limit_{fleet_name}",
                )

            # Existing solver selection logic for Pulp
            if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
                solver = pulp.HiGHS_CMD(msg=0)
            elif hasattr(self.config, 'cbc_path') and os.path.exists(self.config.cbc_path):
                solver = pulp.COIN_CMD(path=self.config.cbc_path, msg=0)
            else:
                solver = pulp.PULP_CBC_CMD(msg=0)
            
            stitch_prob.solve(solver)
            
            if stitch_prob.status == pulp.LpStatusOptimal:
                solution_found = True
            elif stitch_prob.status == pulp.LpStatusInfeasible:
                raise RuntimeError(
                    f"[{fleet_name}] 车辆拼接不可行；主问题不得返回违反车队规模的方案"
                )
            else:
                solution_found = False
                self.logger.warning(f"Pulp did not find an optimal solution for stitching problem. Status: {stitch_prob.status}")

        next_task = {}
        edge_fees = {}
        if solution_found:
            for edge in valid_edges:
                u, v, empty_fee, dist = edge
                if ((use_gurobi and x_vars[(u, v)].X > 0.5) or
                    (not use_gurobi and x_vars[(u, v)].varValue is not None and x_vars[(u, v)].varValue > 0.5)):
                    next_task[u] = v
                    edge_fees[(u, v)] = empty_fee

        in_degree = {u: 0 for u in range(len(sub_tasks))}
        for u, v in next_task.items():
            in_degree[v] += 1

        v_counter = start_v_id
        num_vehicles_used = 0
        scheduled = []
        total_empty_fee = 0.0
        reposition_events = []  # list of dicts for logging

        for start_task in range(len(sub_tasks)):
            if in_degree[start_task] == 0:
                num_vehicles_used += 1
                curr = start_task
                prev_task = None
                while True:
                    t = sub_tasks[curr]
                    t['vehicle'] = v_counter
                    scheduled.append(t)
                    if curr in next_task:
                        nxt = next_task[curr]
                        total_empty_fee += edge_fees[(curr, nxt)]
                        # Log reposition event
                        dist = 0.0
                        for edge in valid_edges:
                            if edge[0] == curr and edge[1] == nxt:
                                # valid_edges entries are (u_idx, v_idx,
                                # empty_fee, distance).  The distance is the
                                # fourth element, not a fifth one.
                                dist = edge[3]
                                break
                        reposition_events.append({
                            "event_type": "inter_task_repositioning",
                            "vehicle": v_counter,
                            "from_od": t.get('od_pair', f'OD_{t.get("od_idx",-1)+1}'),
                            "to_od": sub_tasks[nxt].get('od_pair', f'OD_{sub_tasks[nxt].get("od_idx",-1)+1}'),
                            "from_node": t['destination'],
                            "to_node": sub_tasks[nxt]['origin'],
                            "from_arrival": t['arrival_time'],
                            "to_departure": sub_tasks[nxt]['departure_time'],
                            "distance_km": round(dist, 2),
                            "empty_fee": round(edge_fees[(curr, nxt)], 2),
                        })
                        prev_task = curr
                        curr = nxt
                    else:
                        break
                v_counter += 1

        return num_vehicles_used, scheduled, total_empty_fee, reposition_events

    def _split_route_segments(self, col: Column) -> tuple[list, list, list]:
        """
        S1-1: Dynamically identify road-rail-road segments from a column's arc sequence.

        The problem mandates public-transport mode sequence: road -> rail -> road.
        Multiple consecutive road arcs before/after the rail segment are allowed.
        Returns (west_road_arcs, rail_arcs, east_road_arcs).

        Raises ValueError if the column does not follow the road-rail-road sequence.
        """
        arcs = col.arcs
        modes = col.arc_modes

        if not arcs:
            return [], [], []

        # Find the index where mode first changes from road to rail
        rail_start = None
        for k, m in enumerate(modes):
            if m == 2:
                rail_start = k
                break

        # Find the index where mode last changes from rail back to road
        rail_end = None
        for k in range(len(modes) - 1, -1, -1):
            if modes[k] == 2:
                rail_end = k
                break

        if rail_start is None or rail_end is None:
            # Pure-road route: no rail segment (should not occur in valid problem)
            return arcs, [], []

        west_road_arcs = arcs[:rail_start]
        rail_arcs = arcs[rail_start:rail_end + 1]
        east_road_arcs = arcs[rail_end + 1:]

        return west_road_arcs, rail_arcs, east_road_arcs

    def _extract_lp_solution(self, node: BAPNode, lp_val: float) -> dict:
        """Extract a fractional LP solution using column lambda values already set by CG.

        Unlike _extract_solution which re-solves as MIP, this method uses the
        fractional col.lp_value values directly to compute weighted sub-costs
        and reliability — the correct approach for LP relaxation benchmarking.
        """
        q_veh = self.config.q_road
        tol = self.config.tolerance

        # --- Weighted sub-costs from fractional lambdas ---
        total_transport = 0.0
        total_transfer = 0.0
        total_repo = 0.0
        total_penalty = 0.0
        used_hubs = set()

        # Track per-OD fractional deliveries for penalty calculation
        od_delivered_frac = {}
        od_columns_frac = {}

        for col in node.columns:
            val = col.lp_value
            if val <= tol:
                continue

            total_transport += col.transport_cost * val
            used_hubs.update(col.transfer_nodes)

            # Per-OD fractional delivery
            od_idx = col.od_idx
            od_delivered_frac[od_idx] = od_delivered_frac.get(od_idx, 0.0) + q_veh * val
            od_columns_frac.setdefault(od_idx, []).append((col, val))

        total_transfer = sum(
            col.transfer_cost * col.lp_value
            for col in node.columns
            if col.lp_value > tol
        )
        rail_tariff = getattr(self.mp, "rail_tariff_solution", {})
        total_transport += sum(
            rail_tariff.get("unit_costs", {}).get(key, 0.0) * load
            for key, load in rail_tariff.get("loads", {}).items()
        )

        # Unmet penalty from fractional shortfall
        for i, od in enumerate(self.network.od_pairs):
            delivered = min(od_delivered_frac.get(i, 0.0), od.demand)
            shortfall = max(0.0, od.demand - delivered)
            total_penalty += self.config.c_unmet_penalty * shortfall

        total_cost = total_transport + total_transfer + total_penalty

        # --- Reliability: evaluate the dominant column per OD ---
        od_reliability_details = {}
        od_batches = {}
        active_cols = []

        for col in node.columns:
            val = col.lp_value
            if val <= tol:
                continue
            active_cols.append((col, val))

        # Evaluate every positive column; risk and time are aggregated
        # independently over the active plans of each OD.
        for col, _ in active_cols:
            od_idx = col.od_idx
            od_req = self.network.od_pairs[od_idx]
            transfer_objs = [self.network.transfer_nodes[tid]
                             for tid in col.transfer_nodes
                             if tid in self.network.transfer_nodes]

            result = self.reliability.analyze_path(
                col.arcs, col.arc_modes, transfer_objs, od_req, col.departure_time,
                col.arrival_time, col.time_R, col.rail_services,
            )
            od_batches.setdefault(od_idx, []).append(result)

        # Aggregate R_net
        network_risk_reliability = 1.0
        network_time_reliability = 1.0
        network_risk = 0.0
        network_time = 0.0

        if od_batches:
            for od_idx, results in od_batches.items():
                od_risk_reliability = min(r.plan_risk_reliability for r in results)
                od_time_reliability = min(r.plan_time_reliability for r in results)
                od_risk = max(r.plan_risk for r in results)
                od_time = max(r.plan_arrival_time for r in results)
                od_reliability_details[od_idx] = {
                    "od_risk_reliability": od_risk_reliability,
                    "od_time_reliability": od_time_reliability,
                    "od_joint_reliability": min(od_risk_reliability, od_time_reliability),
                    "binding": "risk" if od_risk_reliability <= od_time_reliability else "time",
                    "od_risk": od_risk,
                    "od_time": od_time,
                }
            network_risk_reliability = min(d["od_risk_reliability"] for d in od_reliability_details.values())
            network_time_reliability = min(d["od_time_reliability"] for d in od_reliability_details.values())
            network_risk = 1.0 - network_risk_reliability
            network_time = max(d["od_time"] for d in od_reliability_details.values())

        return {
            "status": "optimal",
            "total_cost": total_cost,
            "transport_cost": total_transport,
            "transfer_cost": total_transfer,
            "reposition_cost": total_repo,
            "unmet_penalty": total_penalty,
            "unmet_penalty_per_feu": self.config.c_unmet_penalty,
            "network_risk_reliability": network_risk_reliability,
            "network_time_reliability": network_time_reliability,
            "network_risk": network_risk,
            "network_time": network_time,
            "od_reliability_details": od_reliability_details,
            "iterations": getattr(self, 'col_gen_iterations', 1),
            "total_columns": len(node.columns),
            "num_vehicles_used": len(active_cols),
            "total_dispatches": len(active_cols),
            "active_hubs": sorted(list(used_hubs)),
        }

    def _extract_solution(self, node: BAPNode) -> dict:
        """
        Extract solution, stitch empty vehicle deadheads, and consolidate rail wagon shipments.

        S1-1: Uses _split_route_segments to dynamically identify west-road, rail, east-road
              segments rather than hardcoding col.arcs[0/1/2].
        S2-5b: network_risk_reliability and network_time_reliability computed by calling self.reliability to evaluate the solution.
        """
        # --- Re-solve master as MIP to get integer lambdas AND repositioning (pi) values ---
        # The CG LP relaxation gives fractional lambdas; MIP gives proper integer dispatches.
        # S1-1: Pass branch_constraints from current node so MIP respects branching decisions.
        mip_val, _ = self.mp.solve(node.columns, branch_constraints=node.branch_constraints, is_integer=True)

        collection_tasks = []
        distribution_tasks = []
        rail_tasks = []
        used_hubs = set()

        alpha_T = self.config.alpha_T
        dispatch_counter = 0
        q_veh = self.config.q_road

        for col in node.columns:
            tol = getattr(self.config, 'tolerance', 1e-6)
            dispatches = int(round(col.lp_value)) if col.lp_value > tol else 0
            
            if dispatches > 0:
                od = self.network.od_pairs[col.od_idx]
                used_hubs.update(col.transfer_nodes)

                # S1-1: Dynamically split route into road-rail-road segments
                west_road_arcs, rail_arcs_seg, east_road_arcs = self._split_route_segments(col)

                if not west_road_arcs or not rail_arcs_seg or not east_road_arcs:
                    # Skip columns that do not follow the required road-rail-road pattern
                    continue

                # Compute west road segment timing
                t_west_dep = col.departure_time
                t_west_arr = t_west_dep
                west_continuous_drive = 0.0
                for wa in west_road_arcs:
                    tri_w = self.engine.make_transport_time(wa.distance, wa.mode)
                    road_time = tri_w.deterministic_equivalent_upper(alpha_T)
                    rest_time, west_continuous_drive = self.engine.road_drive_transition(
                        west_continuous_drive, road_time
                    )
                    t_west_arr += road_time + rest_time

                # Transfer time at west hub (last west road arc destination = first rail arc origin)
                west_hub_node = west_road_arcs[-1].j
                tn_west = self.network.transfer_nodes.get(west_hub_node)
                transfer_time_west = 0.0
                if tn_west:
                    transfer_time_west = self.config.transfer_time_hours(1, 2)

                # Compute east road segment timing (backward from arrival)
                t_east_arr = col.arrival_time
                t_east_dep = t_east_arr
                east_continuous_drive = 0.0
                for ea in reversed(east_road_arcs):
                    tri_e = self.engine.make_transport_time(ea.distance, ea.mode)
                    road_time = tri_e.deterministic_equivalent_upper(alpha_T)
                    rest_time, east_continuous_drive = self.engine.road_drive_transition(
                        east_continuous_drive, road_time
                    )
                    t_east_dep -= road_time + rest_time

                # Transfer time at east hub (first east road arc origin = last rail arc destination)
                east_hub_node = east_road_arcs[0].i
                tn_east = self.network.transfer_nodes.get(east_hub_node)
                transfer_time_east = 0.0
                if tn_east:
                    transfer_time_east = self.config.transfer_time_hours(2, 1)

                # Rail segment timing
                rail_dep = t_west_arr + transfer_time_west
                rail_arr = t_east_dep - transfer_time_east

                def create_task():
                    return {
                        "dispatch_id": 0,
                        "col_id": col.col_id,
                        "od_idx": col.od_idx,
                        "od_pair": f"OD_{col.od_idx + 1}",
                        "origin": None,
                        "destination": None,
                        "departure_time": 0.0,
                        "arrival_time": 0.0,
                        "availability_time": od.time_window_start,
                        "arcs": [],
                        "cost": col.total_cost,
                        "transport_cost": col.transport_cost,
                        "transfer_cost": col.transfer_cost,
                        "bottleneck_capacity": col.bottleneck_capacity,
                    }

                for _ in range(dispatches):
                    w = create_task()
                    w["origin"] = west_road_arcs[0].i
                    w["destination"] = west_road_arcs[-1].j
                    w["departure_time"] = t_west_dep
                    w["arrival_time"] = t_west_arr
                    w["arcs"] = [{"i": a.i, "j": a.j, "mode": 1, "alpha": a.alpha} for a in west_road_arcs]
                    w["dispatch_id"] = dispatch_counter
                    w["origin_hub"] = col.origin_hub
                    w["province_hub_id"] = col.origin_hub
                    w["task_type"] = "collection"
                    collection_tasks.append(w)
                    
                    e = create_task()
                    e["origin"] = east_road_arcs[0].i
                    e["destination"] = east_road_arcs[-1].j
                    e["departure_time"] = t_east_dep
                    e["arrival_time"] = t_east_arr
                    e["arcs"] = [{"i": a.i, "j": a.j, "mode": 1, "alpha": a.alpha} for a in east_road_arcs]
                    e["dispatch_id"] = dispatch_counter
                    e["destination_hub"] = col.destination_hub
                    e["province_hub_id"] = col.destination_hub
                    e["task_type"] = "distribution"
                    distribution_tasks.append(e)

                    r = create_task()
                    r["vehicle"] = 2000 + dispatch_counter
                    r["origin"] = rail_arcs_seg[0].i
                    r["destination"] = rail_arcs_seg[-1].j
                    r["departure_time"] = rail_dep
                    r["arrival_time"] = rail_arr
                    r["arcs"] = [{"i": a.i, "j": a.j, "mode": 2, "alpha": a.alpha} for a in rail_arcs_seg]
                    r["dispatch_id"] = dispatch_counter
                    r["rail_services"] = [dict(run) for run in col.rail_services]
                    rail_tasks.append(r)
                    dispatch_counter += 1

        # S1-2: Per-hub vehicle stitching.  Collection and distribution tasks
        # share one provincial fleet and may connect when their times permit.
        hub_fleet = self.instance.get("hub_fleet_size", {})
        road_tasks_by_hub = {}
        for task in collection_tasks + distribution_tasks:
            hub_id = task["province_hub_id"]
            road_tasks_by_hub.setdefault(hub_id, []).append(task)

        num_vehicles_used = 0
        scheduled_road = []
        reposition_events = []
        repo_empty_fee_total = 0.0
        v_counter = 1

        for hub_id, hub_tasks in road_tasks_by_hub.items():
            fleet_limit = hub_fleet.get(str(hub_id), None)
            n_veh, sched, ef, ev = self._stitch_sub_fleet(
                hub_tasks, f"Province_Hub{hub_id}", start_v_id=v_counter,
                max_vehicles=fleet_limit
            )
            num_vehicles_used += n_veh
            scheduled_road.extend(sched)
            reposition_events.extend(ev)
            repo_empty_fee_total += ef
            v_counter += n_veh

        # Combine schedules
        schedule = scheduled_road + rail_tasks

        # Calculate exact consolidation costs using dynamic segment identification (S1-1)
        rail_consolidated_runs = {}
        for t in rail_tasks:
            col = next(c for c in node.columns if c.col_id == t['col_id'])
            west_arcs_c, rail_arcs_c, east_arcs_c = self._split_route_segments(col)

            # Group rail tasks by rail segment origin-destination and departure for consolidation
            if rail_arcs_c:
                rail_key = (rail_arcs_c[0].i, rail_arcs_c[-1].j, round(t['departure_time'], 1))
                if rail_key not in rail_consolidated_runs:
                    rail_consolidated_runs[rail_key] = {"tasks": [], "rail_arcs": rail_arcs_c}
                rail_consolidated_runs[rail_key]["tasks"].append(t)

        consolidated_summary = []
        for key, data in rail_consolidated_runs.items():
            u, v, dep = key
            task_list = data["tasks"]
            r_arcs = data["rail_arcs"]
            tasks_per_wagon = self.config.q_rail / self.config.q_road
            wagons = int(np.ceil(len(task_list) / tasks_per_wagon))
            load_factor = (len(task_list) / (wagons * tasks_per_wagon)) * 100.0
            consolidated_summary.append({
                "rail_section": f"Hub_{u} -> Hub_{v}",
                "departure_time": dep,
                "truck_dispatches": len(task_list),
                "rail_wagons": wagons,
                "load_factor_percent": round(load_factor, 1),
                "tasks_included": [tk['od_pair'] for tk in task_list]
            })

        
        rail_tariff = getattr(self.mp, "rail_tariff_solution", {})
        rail_transport_total = sum(
            rail_tariff.get("unit_costs", {}).get(key, 0.0) * load
            for key, load in rail_tariff.get("loads", {}).items()
        )
        road_transport_total = sum(t["transport_cost"] for t in rail_tasks)
        total_transport = road_transport_total + rail_transport_total
        total_transfer = sum(
            col.transfer_cost * (col.lp_value or 0.0)
            for col in node.columns
        )

        s2_allocation_details = []
        actual_total_penalty_cost = 0.0

        for i, od in enumerate(self.network.od_pairs):
            od_tasks = [t for t in rail_tasks if t['od_idx'] == i]
            total_capacity_available = sum(t['bottleneck_capacity'] for t in od_tasks)

            # Demand satisfaction: delivered cannot exceed actual demand
            delivered = min(total_capacity_available, od.demand)
            shortfall = max(0.0, od.demand - delivered)

            # Unmet penalty penalty (deterministic shortfall + reposition delay)
            penalty_cost_i = self.config.c_unmet_penalty * shortfall
            actual_total_penalty_cost += penalty_cost_i

            required_dispatches = int(np.ceil(od.demand / q_veh)) if q_veh > 0 else 0
            actual_dispatches = len(od_tasks)

            s2_allocation_details.append({
                "od_idx": i,
                "od_pair": f"OD_{i+1}",
                "demand": od.demand,
                "delivered": round(delivered, 2),
                "shortfall": round(shortfall, 2),
                "required_dispatches": required_dispatches,
                "scheduled_dispatches": actual_dispatches,
                "penalty_cost": round(penalty_cost_i, 2),
            })

        # Aggregate identical vehicle paths into transport_routes
        aggregated_routes = {}
        for t in schedule:
            key = (t['od_pair'], tuple((a['i'], a['j'], a['mode']) for a in t['arcs']), round(t['departure_time'], 2), round(t['arrival_time'], 2))
            if key not in aggregated_routes:
                aggregated_routes[key] = dict(t)
                aggregated_routes[key]['num_vehicles'] = 1
                if 'vehicle' in aggregated_routes[key]:
                    aggregated_routes[key]['vehicles'] = [t['vehicle']]
                    del aggregated_routes[key]['vehicle']
            else:
                aggregated_routes[key]['num_vehicles'] += 1
                if 'vehicle' in t and 'vehicles' in aggregated_routes[key]:
                    aggregated_routes[key]['vehicles'].append(t['vehicle'])

        transport_routes = list(aggregated_routes.values())

        # Log reposition events summary
        if reposition_events:
            by_route = {}
            for e in reposition_events:
                key = (e['from_od'], e['to_od'], e['from_node'], e['to_node'])
                by_route.setdefault(key, []).append(e)
            self.logger.info(
                f"  Reposition events: {len(reposition_events)} total | "
                f"empty_fee=¥{repo_empty_fee_total:.2f}"
            )
            for key, events in by_route.items():
                avg_fee = sum(e['empty_fee'] for e in events) / len(events)
                self.logger.info(
                    f"    {key[0]} → {key[1]} (node {key[2]}→{key[3]}): "
                    f"{len(events)} times | avg empty=¥{avg_fee:.2f}/ea"
                )

        penalty_cost = actual_total_penalty_cost

        batch_train_services = []
        batch_number_by_od = {}
        for task in sorted(rail_tasks, key=lambda item: (item["od_idx"], item["dispatch_id"])):
            od_idx = task["od_idx"]
            batch_number_by_od[od_idx] = batch_number_by_od.get(od_idx, 0) + 1
            batch_id = batch_number_by_od[od_idx]
            for service_order, run in enumerate(task.get("rail_services", []), 1):
                batch_train_services.append({
                    "od_id": od_idx + 1,
                    "batch_id": batch_id,
                    "service_order": service_order,
                    "train_id": run["train_id"],
                    "origin": run["origin_hub"],
                    "destination": run["destination_hub"],
                    # Some real-data timetable records only contain numeric
                    # times.  Keep report generation robust for both formats.
                    "departure_time_fmt": run.get(
                        "departure_time_fmt", str(run.get("departure_time", "N/A"))
                    ),
                    "arrival_time_fmt": run.get(
                        "arrival_time_fmt", str(run.get("arrival_time", "N/A"))
                    ),
                })

        total_cost = (
            total_transport + total_transfer
            + repo_empty_fee_total + actual_total_penalty_cost
        )

        # S2-5b: Compute true network reliability network_risk_reliability and network_time_reliability using ReliabilityAnalyzer.
        # Build vehicle_assignments dict expected by analyze_solution.
        try:
            R_net_result = self._compute_solution_metrics(node)
        except Exception as e:
            self.logger.warning(f"R_net computation failed: {e}")
            R_net_result = None

        # Unpack detailed reliability results if available
        od_reliability_details = {}
        if isinstance(R_net_result, dict):
            od_reliability_details = R_net_result.get('od_results', {})
            network_risk_reliability = R_net_result.get('network_risk_reliability', -1.0)
            network_time_reliability = R_net_result.get('network_time_reliability', -1.0)
            network_risk = R_net_result.get('network_risk', 0.0)
            network_time = R_net_result.get('network_time', 0.0)
        else:
            network_risk_reliability = None
            network_time_reliability = None
            network_risk = 0.0
            network_time = 0.0

        # od_time_reliability = min(R_T across all dispatched routes) — guaranteed >= alpha_T by pricing.
        # od_risk_reliability is the minimum reliability across dispatched routes and is at least alpha_R.
        # delivery_ratio is stored for display but does NOT scale od_time_reliability or od_risk_reliability.
        if isinstance(R_net_result, dict) and 'od_results' in R_net_result:
            for i, od in enumerate(self.network.od_pairs):
                if i in od_reliability_details:
                    od_delivered = s2_allocation_details[i]['delivered'] if i < len(s2_allocation_details) else od.demand
                    delivery_ratio = od_delivered / od.demand if od.demand > 0 else 1.0
                    od_reliability_details[i]['delivery_ratio'] = delivery_ratio
                    od_reliability_details[i]['od_joint_reliability'] = min(
                        od_reliability_details[i].get('od_time_reliability', 1.0),
                        od_reliability_details[i].get('od_risk_reliability', 1.0)
                    )
                    od_reliability_details[i]['binding'] = 'risk' if od_reliability_details[i].get('od_risk_reliability', 1.0) <= od_reliability_details[i].get('od_time_reliability', 1.0) else 'time'
            od_times = [od_reliability_details[i]['od_time_reliability'] for i in od_reliability_details]
            od_risks = [od_reliability_details[i].get('od_risk_reliability', 1.0) for i in od_reliability_details]
            network_time_reliability = min(od_times) if od_times else 1.0
            network_risk_reliability = min(od_risks) if od_risks else 1.0

        _lb = getattr(node, 'lower_bound', 0.0)
        _ub = getattr(self, 'best_ub', total_cost)
        gap_pct = max(0.0, (_ub - _lb) / _ub * 100.0) if _ub > 0 else 0.0

        return {
            "status": "optimal",
            "total_cost": total_cost,
            "transport_cost": total_transport,
            "road_transport_cost": road_transport_total,
            "rail_transport_cost": rail_transport_total,
            "transfer_cost": total_transfer,
            "reposition_cost": repo_empty_fee_total,
            "reposition_events": reposition_events,
            "num_repositions": len(reposition_events),
            "unmet_penalty": actual_total_penalty_cost,
            "unmet_penalty_per_feu": self.config.c_unmet_penalty,
            "penalty_cost": penalty_cost,
            "active_hubs": sorted(list(used_hubs)),
            "network_risk_reliability": network_risk_reliability,
            "network_time_reliability": network_time_reliability,
            "network_risk": network_risk,
            "network_time": network_time,
            "od_reliability_details": od_reliability_details,
            "max_vehicles": sum(int(v) for v in self.instance.get("hub_fleet_size", {}).values()),
            "num_vehicles_used": num_vehicles_used,
            "total_dispatches": len(scheduled_road),
            "transport_routes": transport_routes,
            "s2_allocation_details": s2_allocation_details,
            "rail_consolidation_summary": consolidated_summary,
            "batch_train_services": batch_train_services,
            "iterations": getattr(self, 'col_gen_iterations', 1),
            "cg_history": list(self.cg_history),
            "total_columns": len(node.columns),
            "gap_pct": round(gap_pct, 4),
        }

    def _compute_solution_metrics(self, node: BAPNode) -> dict:
        """
        S2-5b: Compute true network reliability R_net by evaluating each active column.

        Uses ReliabilityAnalyzer.analyze_path for each dispatched column to get
        per-path risk and time reliability, then aggregates via min over OD pairs.
        """
        # Compute per-path and per-OD reliability details and aggregate to R_net
        od_batches: dict[int, list[PathReliabilityResult]] = {}

        for col in node.columns:
            if not (hasattr(col, "lp_value") and col.lp_value > self.config.tolerance):
                continue

            od_idx = col.od_idx
            od_req = self.network.od_pairs[od_idx]

            # Gather transfer node objects for this column
            transfer_objs = []
            for tn_id in col.transfer_nodes:
                tn = self.network.transfer_nodes.get(tn_id)
                if tn:
                    transfer_objs.append(tn)

            # Use col.departure_time as the actual departure time for R_T computation.
            # od.time_window_start would underestimate the fuzzy arrival, inflating R_T.
            result = self.reliability.analyze_path(
                col.arcs, col.arc_modes, transfer_objs, od_req, col.departure_time,
                col.arrival_time, col.time_R, col.rail_services,
            )

            od_batches.setdefault(od_idx, []).append(result)

        if not od_batches:
            return 1.0

        od_results = {}
        od_reliabilities = []
        od_risks = []
        od_times = []
        for od_idx, batches in od_batches.items():
            plan_risk_reliabilities = [b.plan_risk_reliability for b in batches]
            plan_time_reliabilities = [b.plan_time_reliability for b in batches]
            plan_risks = [b.plan_risk for b in batches]
            plan_times = [b.plan_time for b in batches]
            
            od_risk_reliability = min(plan_risk_reliabilities) if plan_risk_reliabilities else 1.0
            od_time_reliability = min(plan_time_reliabilities) if plan_time_reliabilities else 1.0
            # Determine which component binds (risk or time) for the worst OD
            binding = "risk" if od_risk_reliability <= od_time_reliability else "time"

            od_results[od_idx] = {
                "od_joint_reliability": min(od_risk_reliability, od_time_reliability),
                "od_risk_reliability": od_risk_reliability,
                "od_time_reliability": od_time_reliability,
                "num_batches": len(batches),
                "plan_risk_reliabilities": plan_risk_reliabilities,
                "plan_time_reliabilities": plan_time_reliabilities,
                "plan_risks": plan_risks,
                "plan_times": plan_times,
                "binding": binding,
                "od_risk": max(plan_risks) if plan_risks else 0.0,
                "od_time": max(plan_times) if plan_times else 0.0,
            }
            od_risks.append(od_risk_reliability)
            od_times.append(od_time_reliability)

        network_risk_reliability = min(od_risks) if od_risks else 1.0
        network_time_reliability = min(od_times) if od_times else 1.0
        
        network_risk = 1.0 - network_risk_reliability
        network_time = max((od_r["od_time"] for od_r in od_results.values()), default=0.0)

        return {
            "network_risk_reliability": network_risk_reliability,
            "network_time_reliability": network_time_reliability,
            "network_risk": network_risk,
            "network_time": network_time,
            "od_results": od_results
        }
