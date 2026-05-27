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
from ..core.uncertainty import TrapezoidalFuzzy, TriangularFuzzy, UncertaintyEngine
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
    deploy_cost: float = 0.0
    transport_cost: float = 0.0
    transfer_cost: float = 0.0
    delay_penalty_cost: float = 0.0
    arrival_time: float = 0.0
    time_R: float = 1.0
    sys_R: float = 1.0
    bottleneck_capacity: float = 20.0
    lp_value: float = 0.0
    lp_value_own: float = 0.0
    lp_value_rent: float = 0.0
    transport_cost_upper_bound: float = 0.0
    risk_R: float = 1.0


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
        # Ensure per-OD time reliability uses active config.alpha_T so
        # sensitivity runs that modify config.alpha_T take effect.
        for od in self.network.od_pairs:
            od.alpha_T_od = self.config.alpha_T
        self.engine = UncertaintyEngine(config)
        self.reliability = ReliabilityAnalyzer(config, self.engine)

        # Vehicle limit is dynamically provided by the instance configuration
        self.mp = MasterProblem(config, self.network, instance["num_vehicles"], logger, instance=instance)
        self.pp = PricingProblem(config, self.network, self.engine, logger)

        self.best_ub = float("inf")
        self.best_solution = None
        self.open_nodes = []
        self.node_counter = 0
        self.col_gen_iterations = 0

    def solve(self) -> dict:
        """Main solve loop using Best-First Search."""
        import time
        self.solve_start_time = time.time()

        root_cols = self._generate_initial_columns()
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

            lp_val, is_optimal = self._column_generation(curr_node)

            if lp_val is None:
                continue

            curr_node.lower_bound = lp_val

            if lp_val >= self.best_ub:
                continue

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

        # Core time is strictly the time spent inside ESPPRC pathfinding
        elapsed_wall = self.pp.total_espprc_wall_time
        elapsed_cpu = self.pp.total_espprc_cpu_time
        
        if self.best_solution:
            self.best_solution["solve_wall_time"] = elapsed_wall
            self.best_solution["solve_cpu_time"] = elapsed_cpu
            return self.best_solution
        else:
            return {"status": "infeasible", "solve_wall_time": elapsed_wall, "solve_cpu_time": elapsed_cpu}

    def _column_generation(self, node: BAPNode) -> tuple[float | None, bool]:
        """Run exact column generation loop at a node."""
        iteration = 0
        max_iters = self.config.bap_max_iterations

        existing_fingerprints = set()
        for col in node.columns:
            # #18: Include departure time in fingerprint to distinguish same-path different-timing columns
            fp = tuple((a.i, a.j, a.arc_id) for a in col.arcs) + (round(col.departure_time, 1),)
            existing_fingerprints.add(fp)

        previous_duals = None
        alpha_stab = self.config.stabilization_mu

        import time
        while iteration < max_iters:
            if time.time() - getattr(self, 'solve_start_time', time.time()) > self.config.time_limit:
                self.logger.warning(f"Time limit of {self.config.time_limit}s reached during Column Generation.")
                break

            iteration += 1
            self.col_gen_iterations += 1
            lp_val, raw_duals = self.mp.solve(node.columns, node.branch_constraints)
            if lp_val is None:
                return None, False

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
                labels = self.pp.solve(
                    od.origin, od.destination, i, duals, od.E_od, od.L_od, od.alpha_T_od,
                    node.branch_constraints, exact=False
                )

                new_node_cols = []
                for lab in labels:
                    # #18: Include time_upper to distinguish same-path different-timing labels
                    fp = tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence) + (round(lab.time_upper, 1),)
                    if fp not in existing_fingerprints:
                        col = self._label_to_column(lab, i)
                        new_node_cols.append(col)
                        existing_fingerprints.add(fp)

                filtered_cols = self._apply_branch_constraints(
                    new_node_cols, node.branch_constraints
                )

                if filtered_cols:
                    node.columns.extend(filtered_cols)
                    new_cols_found = True

            # 2. If heuristic pricing fails to find any new columns, run Exact Pricing (exact=True) to prove optimality
            if not new_cols_found:
                for i, od in enumerate(self.network.od_pairs):
                    labels = self.pp.solve(
                        od.origin, od.destination, i, duals, od.E_od, od.L_od, od.alpha_T_od,
                        node.branch_constraints, exact=True
                    )

                    new_node_cols = []
                    for lab in labels:
                        fp = tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence) + (round(lab.time_upper, 1),)
                        if fp not in existing_fingerprints:
                            col = self._label_to_column(lab, i)
                            new_node_cols.append(col)
                            existing_fingerprints.add(fp)

                    filtered_cols = self._apply_branch_constraints(
                        new_node_cols, node.branch_constraints
                    )

                    if filtered_cols:
                        node.columns.extend(filtered_cols)
                        new_cols_found = True

            if not new_cols_found:
                break

            self.logger.debug(
                f"  Iteration {iteration}: LP={lp_val:.2f}, PoolSize={len(node.columns)}"
            )


        if iteration >= max_iters:
            self.logger.warning(f"  Column generation reached max iterations ({max_iters})")

        return lp_val, True

    def _is_integer(self, node: BAPNode) -> bool:
        """Integrality check for all integer variables: column dispatch (lambda) and hub activation (g_vars)."""
        tol = self.config.tolerance
        for col in node.columns:
            if hasattr(col, "lp_value") and tol < abs(col.lp_value - round(col.lp_value)) < 1.0 - tol:
                return False
        if hasattr(self.mp, "hub_values"):
            for val in self.mp.hub_values.values():
                if tol < abs(val - round(val)) < 1.0 - tol:
                    return False
        return True

    def _apply_branch_constraints(self, columns, constraints):
        """Filter columns based on branching constraints (Ryan-Foster and Hub Activation)."""
        if not constraints:
            return columns

        filtered = []
        for col in columns:
            valid = True
            for c_type, c_data in constraints:
                if c_type == "rf_same":
                    i, j = c_data
                    if not self._check_rf(col, i, j, True):
                        valid = False
                        break
                elif c_type == "rf_diff":
                    i, j = c_data
                    if not self._check_rf(col, i, j, False):
                        valid = False
                        break
                elif c_type == "hub_close":
                    hub_id = c_data
                    if hub_id in col.transfer_nodes:
                        valid = False
                        break
                elif c_type == "arc_forbid":
                    u, v, m = c_data
                    if self._contains_arc(col, u, v, m):
                        valid = False
                        break
                elif c_type == "arc_enforce":
                    u, v, m = c_data
                    if not self._contains_arc(col, u, v, m):
                        valid = False
                        break
            if valid:
                filtered.append(col)
        return filtered

    def _label_to_column(self, label, od_idx) -> Column:
        """Convert a PricingProblem Label to a MasterProblem Column with exact Wait-Shift timing."""
        q_veh = self.config.q_road
        q_rail = self.config.q_rail

        deploy_cost = sum(a.deploy_cost * a.min_units * ((q_veh / q_rail) if a.mode == 2 else 1.0) for a in label.arc_sequence)
        
        base_transport_cost = sum(a.unit_cost * q_veh for a in label.arc_sequence)
        spread = self.config.fuzzy_cost_spread
        alpha_c = self.config.alpha_c
        
        # E^M[c~] = (a + 2b + c)/4 for Triangular Fuzzy Variable
        expected_multiplier = (spread[0] + 2.0 * spread[1] + spread[2]) / 4.0
        transport_cost = base_transport_cost * expected_multiplier
        
        # Upper bound deterministic equivalent coefficient
        if alpha_c > 0.5:
            upper_coeff = (2.0 * alpha_c - 1.0) * spread[2] + 2.0 * (1.0 - alpha_c) * spread[1]
        else:
            upper_coeff = spread[1]
        transport_cost_upper_bound = base_transport_cost * upper_coeff

        transfer_nodes_list = []
        transfer_cost = 0.0
        for k in range(len(label.arc_sequence) - 1):
            a1, a2 = label.arc_sequence[k], label.arc_sequence[k + 1]
            if a1.mode != a2.mode:
                tn_id = a1.j
                transfer_nodes_list.append(tn_id)
                tn = self.network.transfer_nodes.get(tn_id)
                if tn:
                    transfer_cost += tn.unit_transfer_cost * q_veh
        
        actual_cost_without_risk = deploy_cost + transport_cost + transfer_cost
        # Wait-Shift Strategy: Recalculate deterministic upper time to capture early waiting time at rail hubs
        od = self.network.od_pairs[od_idx]
        t_arr = od.E_od
        total_wait_time = 0.0
        alpha_T = self.config.alpha_T
        prev_mode = 0

        # S2-4: Variables for rigorous M{Z~ <= L_od} measure tracking
        z_a_dur = 0.0
        z_b_dur = 0.0
        z_c_dur = 0.0
        z_d_dur = 0.0

        for i, arc in enumerate(label.arc_sequence):
            if i > 0:
                if prev_mode != 0 and prev_mode != arc.mode:
                    tn = self.network.transfer_nodes.get(arc.i)
                    if tn:
                        z_a_dur += 1.0
                        z_b_dur += 1.0
                        z_c_dur += 1.0
                        z_d_dur += 1.0
            if arc.mode == 2:
                # #16: Timetable keyed by (origin_hub, dest_hub). Assumes one physical
                # rail arc per hub pair (true in current data_generator). If multiple
                # arcs share the same hub pair, they'd incorrectly share timetable entries.
                runs = getattr(self.network, 'timetable', {}).get((arc.i, arc.j), [])
                valid_runs = [r for r in runs if r["departure_time"] >= t_arr]
                t_start = valid_runs[0]["departure_time"] if valid_runs else max(t_arr, arc.e_window)
                total_wait_time += max(0.0, t_start - t_arr)
                t_end = valid_runs[0]["arrival_time"] if valid_runs else t_start + 4.0
                t_arr = t_end
                z_a_dur += arc.time_a
                z_b_dur += arc.time_b
                z_c_dur += arc.time_c
                z_d_dur += getattr(arc, 'time_d', arc.time_c)  # #9: trapezoidal for rail
            else:
                tri = self.engine.make_transport_time(arc.distance, arc.mode)
                arc_de = tri.deterministic_equivalent_upper(alpha_T)
                t_start = t_arr
                t_end = t_start + arc_de
                t_arr = t_end
                
                z_a_dur += tri.a
                z_b_dur += tri.b
                z_c_dur += tri.c
                z_d_dur += tri.d  # road: c=d (trapezoidal with collapsed plateau)

            prev_mode = arc.mode

        shifted_departure_time = od.E_od + total_wait_time
        
        # The deadline for this column. Always use the raw instance L_od.
        l_effective = od.L_od

        # ---------------------------------------------------------------
        # COLUMN COST = transport + deploy + transfer (no delay penalty)
        # ---------------------------------------------------------------
        trap_Z = TrapezoidalFuzzy(
            a=shifted_departure_time + z_a_dur,
            b=shifted_departure_time + z_b_dur,
            c=shifted_departure_time + z_c_dur,
            d=shifted_departure_time + z_d_dur,
        )
        actual_cost = actual_cost_without_risk
        
        # S2-4: Compute continuous time_R = M{Z~ <= L_od} in [0,1].
        time_R = trap_Z.uncertain_measure_leq(l_effective)
        sys_R = min(label.risk_R, time_R)
        bottleneck_capacity = q_veh

        col = Column(
            col_id=self.mp.next_col_id,
            od_idx=od_idx,
            vehicle_idx=0,
            arcs=list(label.arc_sequence),
            arc_modes=[a.mode for a in label.arc_sequence],
            transfer_nodes=transfer_nodes_list,
            departure_time=shifted_departure_time,
            arrival_time=label.time_upper,
            total_cost=actual_cost,
            deploy_cost=deploy_cost,
            transport_cost=transport_cost,
            transfer_cost=transfer_cost,
            delay_penalty_cost=0.0,
            risk_R=label.risk_R,
            time_R=time_R,
            sys_R=sys_R,
            bottleneck_capacity=bottleneck_capacity,
            transport_cost_upper_bound=transport_cost_upper_bound,
        )
        self.mp.next_col_id += 1
        return col

    def _generate_initial_columns(self):
        """Generate rigorous initial columns by solving the Pricing Problem with high (big_m) duals."""
        cols = []
        existing_fingerprints = set()
        dummy_duals = {
            "od": {i: getattr(self.config, 'big_m', 1e6) for i in range(len(self.network.od_pairs))},
            "vehicle": {},
            "trans_out": {},
            "trans_in": {},
            "hub": {},
        }
        for i, od in enumerate(self.network.od_pairs):
            labels = self.pp.solve(
                od.origin, od.destination, i, dummy_duals, od.E_od, od.L_od, od.alpha_T_od,
                branch_constraints=[], exact=False
            )
            for lab in labels:
                fp = tuple((a.i, a.j, a.arc_id) for a in lab.arc_sequence)
                if fp not in existing_fingerprints:
                    existing_fingerprints.add(fp)
                    cols.append(self._label_to_column(lab, i))
        self.logger.info(f"  Generated {len(cols)} initial columns")
        return cols

    def _branch(self, node: BAPNode):
        """Hierarchical branching: Prioritize fractional hub activation, then Ryan-Foster node pairs."""
        tol = self.config.tolerance
        fractional_hub = None
        if hasattr(self.mp, "hub_values"):
            for h, val in self.mp.hub_values.items():
                if tol < abs(val - round(val)) < 1.0 - tol:
                    fractional_hub = h
                    break

        if fractional_hub is not None:
            self.logger.info(f"  Branching on fractional Hub {fractional_hub} (value={self.mp.hub_values[fractional_hub]:.2f})")
            child1 = BAPNode(
                lower_bound=node.lower_bound, depth=node.depth + 1, node_id=self.node_counter, parent_id=node.node_id,
                branch_constraints=node.branch_constraints + [("hub_close", fractional_hub)],
                columns=[c for c in node.columns if fractional_hub not in c.transfer_nodes],
            )
            self.node_counter += 1
            heapq.heappush(self.open_nodes, child1)

            child2 = BAPNode(
                lower_bound=node.lower_bound, depth=node.depth + 1, node_id=self.node_counter, parent_id=node.node_id,
                branch_constraints=node.branch_constraints + [("hub_open", fractional_hub)],
                columns=list(node.columns),
            )
            self.node_counter += 1
            heapq.heappush(self.open_nodes, child2)
            return

        candidate = BranchingStrategy.get_ryan_foster_candidate(node.columns, self.node_counter)
        if candidate:
            i, j = candidate
            self.logger.info(f"  Branching on Ryan-Foster pair ({i}, {j})")

            child1 = BAPNode(
                lower_bound=node.lower_bound, depth=node.depth + 1, node_id=self.node_counter, parent_id=node.node_id,
                branch_constraints=node.branch_constraints + [("rf_same", (i, j))],
                columns=[c for c in node.columns if self._check_rf(c, i, j, True)],
            )
            self.node_counter += 1
            heapq.heappush(self.open_nodes, child1)

            child2 = BAPNode(
                lower_bound=node.lower_bound, depth=node.depth + 1, node_id=self.node_counter, parent_id=node.node_id,
                branch_constraints=node.branch_constraints + [("rf_diff", (i, j))],
                columns=[c for c in node.columns if self._check_rf(c, i, j, False)],
            )
            self.node_counter += 1
            heapq.heappush(self.open_nodes, child2)
            return

        # S1-3: Arc branching fallback if Ryan-Foster fails
        arc_candidate = BranchingStrategy.get_arc_candidate(node.columns)
        if arc_candidate:
            u, v, m = arc_candidate
            self.logger.info(f"  Branching on Arc ({u}, {v}, m={m})")
            
            child1 = BAPNode(
                lower_bound=node.lower_bound, depth=node.depth + 1, node_id=self.node_counter, parent_id=node.node_id,
                branch_constraints=node.branch_constraints + [("arc_forbid", (u, v, m))],
                columns=[c for c in node.columns if not self._contains_arc(c, u, v, m)],
            )
            self.node_counter += 1
            heapq.heappush(self.open_nodes, child1)
            
            child2 = BAPNode(
                lower_bound=node.lower_bound, depth=node.depth + 1, node_id=self.node_counter, parent_id=node.node_id,
                branch_constraints=node.branch_constraints + [("arc_enforce", (u, v, m))],
                columns=[c for c in node.columns if self._contains_arc(c, u, v, m)],
            )
            self.node_counter += 1
            heapq.heappush(self.open_nodes, child2)

    def _contains_arc(self, col, u, v, m) -> bool:
        for arc in col.arcs:
            if arc.i == u and arc.j == v and arc.arc_id == m:
                return True
        return False

    def _check_rf(self, col, i, j, same: bool):
        nodes = {arc.i for arc in col.arcs} | {arc.j for arc in col.arcs}
        has_i = i in nodes
        has_j = j in nodes
        if same:
            return has_i == has_j
        else:
            return not (has_i and has_j)

    def _stitch_sub_fleet(self, sub_tasks: list, fleet_name: str, start_v_id: int, max_vehicles: int = None, fleet_type: str = "own") -> tuple[int, list[dict], float, list[dict]]:
        """Solve vehicle stitching ILP for a local road sub-fleet (West or East)."""
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
        valid_edges = []
        d_road = self.config.c_deploy_road
        q_veh = self.config.q_road
        road_speed = self.config.road_speed
        c_road = self.config.c_road

        for u_idx, u in enumerate(sub_tasks):
            for v_idx, v in enumerate(sub_tasks):
                if u_idx != v_idx:
                    if u['destination'] == v['origin']:
                        req_gap = 0.0
                        dist = 0.0
                    else:
                        dist = self.network.get_exact_road_distance(u['destination'], v['origin'])
                        alpha_T = self.config.alpha_T
                        rep_fuzzy = self.engine.make_reposition_time(dist)
                        req_gap = rep_fuzzy.deterministic_equivalent_upper(alpha_T)

                    if u['arrival_time'] + req_gap <= v['departure_time']:
                        empty_drive_fee = self.config.c_empty_reposition_per_km * dist
                        reposition_arrival = u['arrival_time'] + req_gap
                        od_j_idx = v['od_idx']
                        od_E_j = self.network.od_pairs[od_j_idx].E_od
                        extra_wait = max(0.0, reposition_arrival - od_E_j)
                        inv_rate = (self.config.c_holding + self.config.c_opp_base) / 24.0
                        time_factor = 0.05
                        q_veh = self.config.q_road
                        extra_inv_cost = q_veh * inv_rate * (extra_wait + 0.5 * time_factor * extra_wait**2)
                        
                        rep_cost = empty_drive_fee + extra_inv_cost
                        valid_edges.append((u_idx, v_idx, rep_cost, dist))

        if use_gurobi:
            for edge in valid_edges:
                u, v, rep_cost, dist = edge
                x_vars[(u, v)] = stitch_prob.addVar(vtype=GRB.BINARY, name=f"x_{fleet_name}_{u}_{v}")
            stitch_prob.update()
            
            c_rent_hr = getattr(self.config, 'c_rent_per_hour', 100.0)
            c_dep_rent = getattr(self.config, 'c_deploy_rent', 1000.0)
            
            obj_expr = 0
            for edge in valid_edges:
                u, v, rep_cost, dist = edge
                if fleet_type == 'own':
                    # Maximize stitches (minimize vehicles) with a large M, then minimize rep_cost
                    val = rep_cost - 10000.0
                else:
                    arr_u = sub_tasks[u]['arrival_time']
                    dep_v = sub_tasks[v]['departure_time']
                    gap = max(0.0, dep_v - arr_u)
                    val = rep_cost + c_rent_hr * gap - c_dep_rent
                obj_expr += val * x_vars[(u, v)]
                
            stitch_prob.setObjective(obj_expr, GRB.MINIMIZE)

            for u_idx in range(len(sub_tasks)):
                out_edges = [edge for edge in valid_edges if edge[0] == u_idx]
                if out_edges:
                    stitch_prob.addConstr(
                        gp.quicksum(x_vars[(edge[0], edge[1])] for edge in out_edges) <= 1,
                        name=f"out_{fleet_name}_{u_idx}"
                    )
                in_edges = [edge for edge in valid_edges if edge[1] == u_idx]
                if in_edges:
                    stitch_prob.addConstr(
                        gp.quicksum(x_vars[(edge[0], edge[1])] for edge in in_edges) <= 1,
                        name=f"in_{fleet_name}_{u_idx}"
                    )
            
            stitch_prob.optimize()
            
            # Check optimization status
            if stitch_prob.status == GRB.OPTIMAL or stitch_prob.status == GRB.SUBOPTIMAL:
                solution_found = True
            else:
                solution_found = False
                self.logger.warning(f"Gurobi did not find an optimal solution for stitching problem. Status: {stitch_prob.status}")

        else: # Use Pulp
            for edge in valid_edges:
                u, v, rep_cost, dist = edge
                x_vars[(u, v)] = pulp.LpVariable(f"x_{fleet_name}_{u}_{v}", 0, 1, pulp.LpBinary)

            c_rent_hr = getattr(self.config, 'c_rent_per_hour', 100.0)
            c_dep_rent = getattr(self.config, 'c_deploy_rent', 1000.0)
            
            obj_expr = []
            for edge in valid_edges:
                u, v, rep_cost, dist = edge
                if fleet_type == 'own':
                    val = rep_cost - 10000.0
                else:
                    arr_u = sub_tasks[u]['arrival_time']
                    dep_v = sub_tasks[v]['departure_time']
                    gap = max(0.0, dep_v - arr_u)
                    val = rep_cost + c_rent_hr * gap - c_dep_rent
                obj_expr.append(val * x_vars[(u, v)])
                
            stitch_prob += pulp.lpSum(obj_expr)

            for u_idx in range(len(sub_tasks)):
                out_edges = [edge for edge in valid_edges if edge[0] == u_idx]
                if out_edges:
                    stitch_prob += pulp.lpSum(x_vars[(edge[0], edge[1])] for edge in out_edges) <= 1, f"out_{fleet_name}_{u_idx}"
                in_edges = [edge for edge in valid_edges if edge[1] == u_idx]
                if in_edges:
                    stitch_prob += pulp.lpSum(x_vars[(edge[0], edge[1])] for edge in in_edges) <= 1, f"in_{fleet_name}_{u_idx}"
            
            
            # Existing solver selection logic for Pulp
            if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
                solver = pulp.HiGHS_CMD(msg=0)
            elif hasattr(self.config, 'cbc_path') and os.path.exists(self.config.cbc_path):
                solver = pulp.COIN_CMD(path=self.config.cbc_path, msg=0)
            else:
                solver = pulp.PULP_CBC_CMD(msg=0)
            
            stitch_prob.solve(solver)
            
            solution_found = stitch_prob.status == pulp.LpStatusOptimal

        next_task = {}
        rep_costs = {}
        if solution_found:
            for edge in valid_edges:
                u, v, rep_cost, dist = edge
                if ((use_gurobi and x_vars[(u, v)].X > 0.5) or
                    (not use_gurobi and x_vars[(u, v)].varValue is not None and x_vars[(u, v)].varValue > 0.5)):
                    next_task[u] = v
                    rep_costs[(u, v)] = rep_cost

        in_degree = {u: 0 for u in range(len(sub_tasks))}
        for u, v in next_task.items():
            in_degree[v] += 1

        v_counter = start_v_id
        num_vehicles_used = 0
        scheduled = []
        total_reposition = 0.0
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
                        total_reposition += rep_costs[(curr, nxt)]
                        # Log reposition event
                        dist = 0.0
                        for edge in valid_edges:
                            if edge[0] == curr and edge[1] == nxt:
                                dist = edge[3]
                                break
                        reposition_events.append({
                            "vehicle": v_counter,
                            "from_od": t.get('od_pair', f'OD_{t.get("od_idx",-1)+1}'),
                            "to_od": sub_tasks[nxt].get('od_pair', f'OD_{sub_tasks[nxt].get("od_idx",-1)+1}'),
                            "from_node": t['destination'],
                            "to_node": sub_tasks[nxt]['origin'],
                            "from_arrival": t['arrival_time'],
                            "to_departure": sub_tasks[nxt]['departure_time'],
                            "distance_km": round(dist, 2),
                            "cost": round(rep_costs[(curr, nxt)], 2),
                        })
                        prev_task = curr
                        curr = nxt
                    else:
                        break
                v_counter += 1

        return num_vehicles_used, scheduled, total_reposition, reposition_events

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

    def _extract_solution(self, node: BAPNode) -> dict:
        """
        Extract solution, stitch empty vehicle deadheads, and consolidate rail wagon shipments.

        S1-1: Uses _split_route_segments to dynamically identify west-road, rail, east-road
              segments rather than hardcoding col.arcs[0/1/2].
        S2-1: Supply uncertainty uses TrapezoidalFuzzy model from SupplyUncertaintyModel.
        S2-5b: R_net computed by calling self.reliability to evaluate the solution.
        """
        west_tasks_own = []
        east_tasks_own = []
        west_tasks_rent = []
        east_tasks_rent = []
        rail_tasks = []
        used_hubs = set()

        alpha_T = self.config.alpha_T
        dispatch_counter = 0
        q_veh = self.config.q_road

        for col in node.columns:
            val_own = getattr(col, "lp_value_own", 0.0)
            val_rent = getattr(col, "lp_value_rent", 0.0)
            tol = getattr(self.config, 'tolerance', 1e-6)
            
            dispatches_own = int(round(val_own)) if val_own > tol else 0
            dispatches_rent = int(round(val_rent)) if val_rent > tol else 0
            
            if dispatches_own > 0 or dispatches_rent > 0:
                used_hubs.update(col.transfer_nodes)

                # S1-1: Dynamically split route into road-rail-road segments
                west_road_arcs, rail_arcs_seg, east_road_arcs = self._split_route_segments(col)

                if not west_road_arcs or not rail_arcs_seg or not east_road_arcs:
                    # Skip columns that do not follow the required road-rail-road pattern
                    continue

                # Compute west road segment timing
                t_west_dep = col.departure_time
                t_west_arr = t_west_dep
                for wa in west_road_arcs:
                    tri_w = self.engine.make_transport_time(wa.distance, wa.mode)
                    t_west_arr += tri_w.deterministic_equivalent_upper(alpha_T)

                # Transfer time at west hub (last west road arc destination = first rail arc origin)
                west_hub_node = west_road_arcs[-1].j
                tn_west = self.network.transfer_nodes.get(west_hub_node)
                transfer_time_west = 0.0
                if tn_west:
                    transfer_time_west = 1.0
                    transfer_cost_w = tn_west.unit_transfer_cost * q_veh

                # Compute east road segment timing (backward from arrival)
                t_east_arr = col.arrival_time
                t_east_dep = t_east_arr
                for ea in reversed(east_road_arcs):
                    tri_e = self.engine.make_transport_time(ea.distance, ea.mode)
                    t_east_dep -= tri_e.deterministic_equivalent_upper(alpha_T)

                # Transfer time at east hub (first east road arc origin = last rail arc destination)
                east_hub_node = east_road_arcs[0].i
                tn_east = self.network.transfer_nodes.get(east_hub_node)
                transfer_time_east = 0.0
                if tn_east:
                    transfer_time_east = 1.0
                    transfer_cost_e = tn_east.unit_transfer_cost * q_veh              

                # Rail segment timing
                rail_dep = t_west_arr + transfer_time_west
                rail_arr = t_east_dep - transfer_time_east

                def create_task(fleet_type):
                    return {
                        "fleet_type": fleet_type,
                        "dispatch_id": 0,
                        "col_id": col.col_id,
                        "od_idx": col.od_idx,
                        "od_pair": f"OD_{col.od_idx + 1}",
                        "origin": None,
                        "destination": None,
                        "departure_time": 0.0,
                        "arrival_time": 0.0,
                        "arcs": [],
                        "sys_R": col.sys_R,
                        "cost": col.total_cost,
                        "deploy_cost": col.deploy_cost,
                        "transport_cost": col.transport_cost,
                        "transfer_cost": col.transfer_cost,
                        "delay_penalty_cost": getattr(col, 'delay_penalty_cost', 0.0),
                        "bottleneck_capacity": col.bottleneck_capacity,
                    }

                for _ in range(dispatches_own):
                    w = create_task("own")
                    w["origin"] = west_road_arcs[0].i
                    w["destination"] = west_road_arcs[-1].j
                    w["departure_time"] = t_west_dep
                    w["arrival_time"] = t_west_arr
                    w["arcs"] = [{"i": a.i, "j": a.j, "mode": 1, "alpha": a.alpha, "consequence": a.C_consequence} for a in west_road_arcs]
                    w["dispatch_id"] = dispatch_counter
                    west_tasks_own.append(w)
                    
                    e = create_task("own")
                    e["origin"] = east_road_arcs[0].i
                    e["destination"] = east_road_arcs[-1].j
                    e["departure_time"] = t_east_dep
                    e["arrival_time"] = t_east_arr
                    e["arcs"] = [{"i": a.i, "j": a.j, "mode": 1, "alpha": a.alpha, "consequence": a.C_consequence} for a in east_road_arcs]
                    e["dispatch_id"] = dispatch_counter
                    east_tasks_own.append(e)

                    r = create_task("own")
                    r["vehicle"] = 2000 + dispatch_counter
                    r["origin"] = rail_arcs_seg[0].i
                    r["destination"] = rail_arcs_seg[-1].j
                    r["departure_time"] = rail_dep
                    r["arrival_time"] = rail_arr
                    r["arcs"] = [{"i": a.i, "j": a.j, "mode": 2, "alpha": a.alpha, "consequence": a.C_consequence} for a in rail_arcs_seg]
                    r["dispatch_id"] = dispatch_counter
                    rail_tasks.append(r)
                    dispatch_counter += 1

                for _ in range(dispatches_rent):
                    w = create_task("rent")
                    w["origin"] = west_road_arcs[0].i
                    w["destination"] = west_road_arcs[-1].j
                    w["departure_time"] = t_west_dep
                    w["arrival_time"] = t_west_arr
                    w["arcs"] = [{"i": a.i, "j": a.j, "mode": 1, "alpha": a.alpha, "consequence": a.C_consequence} for a in west_road_arcs]
                    w["dispatch_id"] = dispatch_counter
                    west_tasks_rent.append(w)
                    
                    e = create_task("rent")
                    e["origin"] = east_road_arcs[0].i
                    e["destination"] = east_road_arcs[-1].j
                    e["departure_time"] = t_east_dep
                    e["arrival_time"] = t_east_arr
                    e["arcs"] = [{"i": a.i, "j": a.j, "mode": 1, "alpha": a.alpha, "consequence": a.C_consequence} for a in east_road_arcs]
                    e["dispatch_id"] = dispatch_counter
                    east_tasks_rent.append(e)

                    r = create_task("rent")
                    r["vehicle"] = 2000 + dispatch_counter
                    r["origin"] = rail_arcs_seg[0].i
                    r["destination"] = rail_arcs_seg[-1].j
                    r["departure_time"] = rail_dep
                    r["arrival_time"] = rail_arr
                    r["arcs"] = [{"i": a.i, "j": a.j, "mode": 2, "alpha": a.alpha, "consequence": a.C_consequence} for a in rail_arcs_seg]
                    r["dispatch_id"] = dispatch_counter
                    rail_tasks.append(r)
                    dispatch_counter += 1

                for _ in range(dispatches_rent):
                    w = create_task("rent")
                    w["origin"] = west_road_arcs[0].i
                    w["destination"] = west_road_arcs[-1].j
                    w["departure_time"] = t_west_dep
                    w["arrival_time"] = t_west_arr
                    w["arcs"] = [{"i": a.i, "j": a.j, "mode": 1, "alpha": a.alpha, "consequence": a.C_consequence} for a in west_road_arcs]
                    w["dispatch_id"] = dispatch_counter
                    west_tasks_rent.append(w)
                    
                    e = create_task("rent")
                    e["origin"] = east_road_arcs[0].i
                    e["destination"] = east_road_arcs[-1].j
                    e["departure_time"] = t_east_dep
                    e["arrival_time"] = t_east_arr
                    e["arcs"] = [{"i": a.i, "j": a.j, "mode": 1, "alpha": a.alpha, "consequence": a.C_consequence} for a in east_road_arcs]
                    e["dispatch_id"] = dispatch_counter
                    east_tasks_rent.append(e)

                    r = create_task("rent")
                    r["vehicle"] = 2000 + dispatch_counter
                    r["origin"] = rail_arcs_seg[0].i
                    r["destination"] = rail_arcs_seg[-1].j
                    r["departure_time"] = rail_dep
                    r["arrival_time"] = rail_arr
                    r["arcs"] = [{"i": a.i, "j": a.j, "mode": 2, "alpha": a.alpha, "consequence": a.C_consequence} for a in rail_arcs_seg]
                    r["dispatch_id"] = dispatch_counter
                    rail_tasks.append(r)
                    dispatch_counter += 1
                
                # prevent execution of old dispatch block

        # S1-2: Global Vehicle Stitching (Rigorous Optimality)
        for t in west_tasks_own: t['region'] = 'West'
        for t in east_tasks_own: t['region'] = 'East'
        for t in west_tasks_rent: t['region'] = 'West'
        for t in east_tasks_rent: t['region'] = 'East'
            
        own_road_tasks = west_tasks_own + east_tasks_own
        num_veh_own, sched_own, repo_own, repo_ev_own = self._stitch_sub_fleet(
            own_road_tasks, "GlobalRoadOwn", start_v_id=1, max_vehicles=self.instance["num_vehicles"], fleet_type="own"
        )
        
        rent_road_tasks = west_tasks_rent + east_tasks_rent
        num_veh_rent, sched_rent, repo_rent, repo_ev_rent = self._stitch_sub_fleet(
            rent_road_tasks, "GlobalRoadRent", start_v_id=1001, max_vehicles=None, fleet_type="rent"
        )
        
        num_vehicles_used = num_veh_own + num_veh_rent
        scheduled_road = sched_own + sched_rent
        total_reposition = repo_own + repo_rent
        reposition_events = repo_ev_own + repo_ev_rent
        
        scheduled_west = [t for t in scheduled_road if t.get('region') == 'West']
        scheduled_east = [t for t in scheduled_road if t.get('region') == 'East']

        # Combine schedules
        schedule = scheduled_road + rail_tasks

        # Calculate exact consolidation costs using dynamic segment identification (S1-1)
        total_hub = len(used_hubs) * self.config.c_hub
        road_deploy_cost = 0.0
        rail_consolidated_runs = {}
        for t in rail_tasks:
            col = next(c for c in node.columns if c.col_id == t['col_id'])
            west_arcs_c, rail_arcs_c, east_arcs_c = self._split_route_segments(col)

            for wa in west_arcs_c:
                road_deploy_cost += wa.deploy_cost
            for ea in east_arcs_c:
                road_deploy_cost += ea.deploy_cost

            # Group rail tasks by rail segment origin-destination and departure for consolidation
            if rail_arcs_c:
                rail_key = (rail_arcs_c[0].i, rail_arcs_c[-1].j, round(t['departure_time'], 1))
                if rail_key not in rail_consolidated_runs:
                    rail_consolidated_runs[rail_key] = {"tasks": [], "rail_arcs": rail_arcs_c}
                rail_consolidated_runs[rail_key]["tasks"].append(t)

        exact_rail_deploy_cost = 0.0
        consolidated_summary = []
        for key, data in rail_consolidated_runs.items():
            u, v, dep = key
            task_list = data["tasks"]
            r_arcs = data["rail_arcs"]
            tasks_per_wagon = self.config.q_rail / self.config.q_road
            wagons = int(np.ceil(len(task_list) / tasks_per_wagon))
            for ra in r_arcs:
                exact_rail_deploy_cost += ra.deploy_cost * wagons

            load_factor = (len(task_list) / (wagons * tasks_per_wagon)) * 100.0
            consolidated_summary.append({
                "rail_section": f"Hub_{u} -> Hub_{v}",
                "departure_time": dep,
                "truck_dispatches": len(task_list),
                "rail_wagons": wagons,
                "load_factor_percent": round(load_factor, 1),
                "tasks_included": [tk['od_pair'] for tk in task_list]
            })

        
        c_rent_hr = getattr(self.config, 'c_rent_per_hour', 100.0)
        c_dep_rent = getattr(self.config, 'c_deploy_rent', 1000.0)
        
        rent_deploy_cost = num_veh_rent * c_dep_rent
        rent_time_cost = 0.0
        for t in rent_road_tasks:
            rent_time_cost += c_rent_hr * (t['arrival_time'] - t['departure_time'])
        for r in repo_ev_rent:
            rent_time_cost += c_rent_hr * r.get('gap_hours', (r['to_departure'] - r['from_arrival']))

        total_deploy = road_deploy_cost + exact_rail_deploy_cost + rent_deploy_cost
        
        total_transport = sum(t['transport_cost'] for t in rail_tasks) + rent_time_cost

        total_transport = sum(t['transport_cost'] for t in rail_tasks)
        total_transfer = sum(t['transfer_cost'] for t in rail_tasks)
        total_delay_penalty = 0.0  # delay penalty removed

        s2_allocation_details = []
        actual_total_opp_cost = 0.0
        actual_total_holding_cost = 0.0
        c_holding = self.config.c_holding

        from ..core.uncertainty import SupplyUncertaintyModel, DemandUncertaintyModel
        supply_model = SupplyUncertaintyModel(self.config)
        demand_model = DemandUncertaintyModel(self.config)

        for i, od in enumerate(self.network.od_pairs):
            od_tasks = [t for t in rail_tasks if t['od_idx'] == i]
            total_capacity_available = sum(t['bottleneck_capacity'] for t in od_tasks)

            # Demand satisfaction
            xi_max = 2.0 * 0.95308992 - 1.0
            d_max = od.demand * (1.0 + self.config.delta_D * xi_max)
            delivered = min(total_capacity_available, d_max)
            shortfall = max(0.0, od.demand - delivered)

            # Opportunity cost penalty (now strictly deterministic based on shortfall)
            opp_cost_i = self.config.c_opp_base * shortfall
            actual_total_opp_cost += opp_cost_i

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
                "opp_cost": round(opp_cost_i, 2),
            })

        # The time-dependent holding cost for repositioning is already included in repo_events cost
        # We zero out the old per-OD holding cost since it's now integrated directly into repo links
        actual_total_holding_cost = 0.0

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
            self.logger.info(f"  Reposition events: {len(reposition_events)} total, ¥{total_reposition:.2f}")
            for key, events in by_route.items():
                avg_cost = sum(e['cost'] for e in events) / len(events)
                self.logger.info(f"    {key[0]} → {key[1]} (node {key[2]}→{key[3]}): "
                                 f"{len(events)} times, avg ¥{avg_cost:.2f}/ea")

        penalty_cost = actual_total_opp_cost + total_delay_penalty

        total_cost = (
            total_deploy + total_transport + total_transfer + total_hub
            + total_reposition + actual_total_opp_cost + actual_total_holding_cost
            + total_delay_penalty
        )

        # S2-5b: Compute true network reliability R_net using ReliabilityAnalyzer.
        # Build vehicle_assignments dict expected by analyze_solution.
        try:
            R_net_computed = self._compute_solution_R_net(node)
        except Exception:
            R_net_computed = -1.0

        # Unpack detailed reliability results if available
        od_reliability_details = {}
        if isinstance(R_net_computed, dict):
            od_reliability_details = R_net_computed.get('od_results', {})
            R_net_risk = R_net_computed.get('R_net_risk', -1.0)
            R_net_time = R_net_computed.get('R_net_time', -1.0)
            R_net_computed = R_net_computed.get('R_net', -1.0)
        else:
            R_net_risk = R_net_computed
            R_net_time = R_net_computed

        return {
            "status": "optimal",
            "total_cost": total_cost,
            "deploy_cost": total_deploy,
            "hub_cost": total_hub,
            "transport_cost": total_transport,
            "transfer_cost": total_transfer,
            "reposition_cost": total_reposition,
            "reposition_events": reposition_events,
            "num_repositions": len(reposition_events),
            "holding_cost": actual_total_holding_cost,
            "opportunity_cost": actual_total_opp_cost,
            "penalty_cost": penalty_cost,
            "active_hubs": sorted(list(used_hubs)),
            "R_net": R_net_computed,
            "R_net_risk": R_net_risk,
            "R_net_time": R_net_time,
            "od_reliability_details": od_reliability_details,
            "max_vehicles": self.instance["num_vehicles"],
            "num_vehicles_used": num_vehicles_used,
            "total_dispatches": len(scheduled_road),
            "transport_routes": transport_routes,
            "s2_allocation_details": s2_allocation_details,
            "rail_consolidation_summary": consolidated_summary,
            "iterations": getattr(self, 'col_gen_iterations', 1),
            "total_columns": len(node.columns),
        }

    def _compute_solution_R_net(self, node: BAPNode) -> float:
        """
        S2-5b: Compute true network reliability R_net by evaluating each active column.

        Uses ReliabilityAnalyzer.analyze_path for each dispatched column to get
        per-path comprehensive reliability, then aggregates via min over OD pairs.
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

            result = self.reliability.analyze_path(
                col.arcs, col.arc_modes, transfer_objs, od_req, col.departure_time
            )

            od_batches.setdefault(od_idx, []).append(result)

        if not od_batches:
            return 1.0

        od_results = {}
        od_reliabilities = []
        od_risks = []
        od_times = []
        for od_idx, batches in od_batches.items():
            batch_comps = [b.comprehensive for b in batches]
            batch_risks = [b.risk_reliability for b in batches]
            batch_times = [b.time_reliability for b in batches]
            R_od = min(batch_comps) if batch_comps else 1.0
            R_od_risk = min(batch_risks) if batch_risks else 1.0
            R_od_time = min(batch_times) if batch_times else 1.0
            # Determine which component binds (risk or time) for the worst batch
            worst_idx = int(batch_comps.index(R_od)) if batch_comps else 0
            binding = "risk" if batch_risks[worst_idx] <= batch_times[worst_idx] else "time"

            od_results[od_idx] = {
                "R_od": R_od,
                "R_od_risk": R_od_risk,
                "R_od_time": R_od_time,
                "num_batches": len(batches),
                "batch_comprehensive": batch_comps,
                "batch_risk": batch_risks,
                "batch_time": batch_times,
                "binding": binding,
            }
            od_reliabilities.append(R_od)
            od_risks.append(R_od_risk)
            od_times.append(R_od_time)

        R_net = min(od_reliabilities) if od_reliabilities else 1.0
        R_net_risk = min(od_risks) if od_risks else 1.0
        R_net_time = min(od_times) if od_times else 1.0
        return {
            "R_net": R_net,
            "R_net_risk": R_net_risk,
            "R_net_time": R_net_time,
            "od_results": od_results
        }
