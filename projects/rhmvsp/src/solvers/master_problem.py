from __future__ import annotations
import os
import numpy as np
import pulp
import networkx as nx
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None
    print("Gurobipy not found. Falling back to Pulp solvers.")

from config import RHMVSPConfig
from ..core.network import ODPair
from ..core.uncertainty import TrapezoidalFuzzy, UncertaintyEngine


class MasterProblem:
    """
    Restricted Master Problem (RMP) for RHMVSP.

    Objective:
      min  sum_col cost * lambda_col
         + sum_od [p_pen + c_unmet_penalty] * shortfall_i
         + empty-vehicle repositioning costs
    """

    def __init__(self, config: RHMVSPConfig, network, logger, instance: dict = None):
        self.config = config
        self.network = network
        self.od_pairs = network.od_pairs
        self.hub_fleet = (instance or {}).get("hub_fleet_size", {})
        self.logger = logger
        self.instance = instance or {}
        self.engine = UncertaintyEngine(config)

        # Stabilization state
        self.mu = getattr(config, 'stabilization_mu', 0.5)
        self.prev_duals = None
        self.use_stabilization = getattr(config, 'bap_stabilization', True)
        self.next_col_id = 0
        self.total_master_wall_time = 0.0
        self.total_master_cpu_time = 0.0
        self.total_master_lp_wall_time = 0.0
        self.total_master_mip_wall_time = 0.0

    def _compute_unmet_penaltys(self):
        """
        Compute opportunity costs per OD pair.
        """
        c_unmet_penalty = {}
        for i, od in enumerate(self.od_pairs):
            # 未满足需求惩罚按 FEU 计，不再按距离调整。
            # under Wang 2025, without additional O-D distance adjustment.
            c_unmet_penalty[i] = self.config.c_unmet_penalty

        return c_unmet_penalty

    def solve(self, columns: list, branch_constraints: list = None, is_integer: bool = False) -> tuple[float | None, dict]:
        """
        Solve exact RMP LP relaxation / MIP.

        Objective:
          min  sum_col cost * lambda_col
             + sum_od [p_pen + c_unmet_penalty] * shortfall_i
             + empty-vehicle repositioning costs
        """
        columns = sorted(columns, key=lambda col: col.col_id)
        branch_constraints = sorted(branch_constraints) if branch_constraints else []

        # 1. Always instantiate all transfer hubs in the Master Problem to support branch constraints safely

        # 2. Compute opportunity costs
        c_unmet_penalty = self._compute_unmet_penaltys()

        q_veh = self.config.q_road
        def column_train_ids(col):
            return {
                int(run["train_id"])
                for run in col.rail_services
                if "train_id" in run
            }

        used_train_ids = set().union(
            *(column_train_ids(col) for col in columns)
        ) if columns else set()
        # The RMP only needs timetable resources referenced by its current
        # columns. Newly priced columns add their dispatches on the next rebuild.
        rail_runs = {
            int(run["train_id"]): run
            for run in self.instance.get("railway_timetable", [])
            if "train_id" in run and int(run["train_id"]) in used_train_ids
        }
        rail_distances = {
            (int(arc["i"]), int(arc["j"])): float(arc["distance"])
            for arc in self.instance.get("network", {}).get("arcs", [])
            if int(arc["mode"]) == 2
        }
        rail_tiers = tuple(self.config.rail_tariff_tiers)
        rail_unit_cost = {
            (train_id, p): self.config.rail_unit_cost(
                rail_distances[(int(run["origin_hub"]), int(run["destination_hub"]))], p
            )
            for train_id, run in rail_runs.items()
            for p in range(len(rail_tiers))
        }

        def column_variable_cost(col):
            # Rail transport cost is load-tier dependent and is modeled separately.
            return col.total_cost

        # 3. Create LP Problem
        if self.config.solver_name == "gurobi" and gp:
            try:
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
                prob = gp.Model("RMP", env=env)
                prob.setParam('OutputFlag', 0)  # Suppress Gurobi output
                prob.setParam("Seed", self.config.seed)
                prob.setParam("Threads", self.config.gurobi_threads)
            except gp.GurobiError as e:
                self.logger.warning(f"Gurobi license error: {e}. Falling back to Pulp.")
                prob = pulp.LpProblem("RMP", pulp.LpMinimize)
                use_gurobi = False
            else:
                use_gurobi = True
        else:
            prob = pulp.LpProblem("RMP", pulp.LpMinimize)
            use_gurobi = False

        # Column-level empty repositioning variables.  Feasibility uses the
        # exact road-task times stored by each generated column so that the
        # master fleet constraints match final vehicle stitching.
        valid_reposition_transitions = {}

        def road_task_times(col):
            first_rail = next((i for i, arc in enumerate(col.arcs) if arc.mode == 2), len(col.arcs))
            last_rail = max((i for i, arc in enumerate(col.arcs) if arc.mode == 2), default=-1)
            west = col.arcs[:first_rail]
            east = col.arcs[last_rail + 1:]
            if not west or not east:
                return None
            def road_block_duration(block):
                duration = 0.0
                continuous_drive = 0.0
                for road_arc in block:
                    drive_time = self.engine.make_transport_time(
                        road_arc.distance, 1
                    ).deterministic_equivalent_upper(self.config.alpha_T)
                    rest_time, continuous_drive = self.engine.road_drive_transition(
                        continuous_drive, drive_time
                    )
                    duration += drive_time + rest_time
                return duration

            west_duration = road_block_duration(west)
            east_duration = road_block_duration(east)
            return {
                "west_start": col.departure_time,
                "west_end": col.departure_time + west_duration,
                "west_origin": west[0].i,
                "west_destination": west[-1].j,
                "east_start": col.arrival_time - east_duration,
                "east_end": col.arrival_time,
                "east_origin": east[0].i,
                "east_destination": east[-1].j,
            }

        column_times = {col.col_id: road_task_times(col) for col in columns}
        column_by_id = {col.col_id: col for col in columns}
        road_task_nodes = {}
        for col in columns:
            times = column_times[col.col_id]
            if times is None:
                continue
            road_task_nodes[(col.col_id, "collection")] = {
                "province_hub_id": col.origin_hub,
                "start": times["west_start"],
                "end": times["west_end"],
                "origin": times["west_origin"],
                "destination": times["west_destination"],
            }
            road_task_nodes[(col.col_id, "distribution")] = {
                "province_hub_id": col.destination_hub,
                "start": times["east_start"],
                "end": times["east_end"],
                "origin": times["east_origin"],
                "destination": times["east_destination"],
            }

        for task_i, data_i in road_task_nodes.items():
            for task_j, data_j in road_task_nodes.items():
                if task_i == task_j:
                    continue
                # Empty repositioning is allowed only inside one provincial fleet area.
                if data_i["province_hub_id"] != data_j["province_hub_id"]:
                    continue
                distance = self.network.get_exact_road_distance(
                    data_i["destination"], data_j["origin"]
                )
                drive_time = distance / self.config.road_speed
                required_gap = drive_time + self.engine.road_rest_time(drive_time)
                if data_i["end"] + required_gap <= data_j["start"]:
                    empty_fee = self.config.c_empty_reposition_per_km * distance
                    valid_reposition_transitions[(task_i, task_j)] = (distance, empty_fee)

        self.logger.debug(
            f"  Provincial road-task nodes={len(road_task_nodes)}, "
            f"reposition transitions={len(valid_reposition_transitions)}; "
            "all initial vehicles are pre-positioned before Day 1 00:00"
        )

        # Decision Variables
        if use_gurobi:
            lambdas = {
                col.col_id: prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS, name=f"lambda_{col.col_id}")
                for col in columns
            }
            dummy_vars = {
                i: prob.addVar(
                    lb=0,
                    ub=0 if is_integer else GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"dummy_{i}",
                )
                for i in range(len(self.od_pairs))
            }
            rail_tier_load = {
                (train_id, p): prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS,
                    name=f"rail_tier_load_{train_id}_{p}")
                for train_id in rail_runs for p in range(len(rail_tiers))
            }
            rail_tier_active = {
                (train_id, p): prob.addVar(lb=0, ub=1,
                    vtype=GRB.BINARY if is_integer else GRB.CONTINUOUS,
                    name=f"rail_tier_active_{train_id}_{p}")
                for train_id in rail_runs for p in range(len(rail_tiers))
            }
        else:
            lambdas = {
                col.col_id: pulp.LpVariable(f"lambda_{col.col_id}", 0, None, pulp.LpContinuous)
                for col in columns
            }
            dummy_vars = {
                i: pulp.LpVariable(
                    f"dummy_{i}", 0, 0 if is_integer else None, pulp.LpContinuous
                )
                for i in range(len(self.od_pairs))
            }
            rail_tier_load = {
                (train_id, p): pulp.LpVariable(
                    f"rail_tier_load_{train_id}_{p}", 0, None,
                    pulp.LpInteger if is_integer else pulp.LpContinuous)
                for train_id in rail_runs for p in range(len(rail_tiers))
            }
            rail_tier_active = {
                (train_id, p): pulp.LpVariable(
                    f"rail_tier_active_{train_id}_{p}", 0, 1,
                    pulp.LpBinary if is_integer else pulp.LpContinuous)
                for train_id in rail_runs for p in range(len(rail_tiers))
            }

        # Decision Variables (repositioning, delivery, shortfall)
        if use_gurobi:
            pi_reposition = {
                pair: prob.addVar(
                    lb=0,
                    vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS,
                    name=f"pi_{pair[0][0]}_{pair[0][1]}_{pair[1][0]}_{pair[1][1]}",
                )
                for pair in valid_reposition_transitions
            }
            initial_vehicle = {
                task: prob.addVar(
                    lb=0,
                    ub=GRB.INFINITY,
                    vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS,
                    name=f"initial_vehicle_{task[0]}_{task[1]}",
                )
                for task in road_task_nodes
            }
            q_delivered = {
                i: prob.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q_del_{i}")
                for i, od in enumerate(self.od_pairs)
            }
            shortage_vars = {}
            for i in range(len(self.od_pairs)):
                shortage_vars[i] = prob.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"shortage_{i}")
            prob.update()
        else:
            pi_category = pulp.LpInteger if is_integer else pulp.LpContinuous
            pi_reposition = {
                pair: pulp.LpVariable(
                    f"pi_{pair[0][0]}_{pair[0][1]}_{pair[1][0]}_{pair[1][1]}",
                    0, None, pi_category,
                )
                for pair in valid_reposition_transitions
            }
            initial_vehicle = {
                task: pulp.LpVariable(
                    f"initial_vehicle_{task[0]}_{task[1]}",
                    0,
                    None,
                    pi_category,
                )
                for task in road_task_nodes
            }
            q_delivered = {
                i: pulp.LpVariable(f"q_del_{i}", 0, None, pulp.LpContinuous)
                for i, od in enumerate(self.od_pairs)
            }
            shortage_vars = {}
            for i in range(len(self.od_pairs)):
                shortage_vars[i] = pulp.LpVariable(f"shortage_{i}", 0, None, pulp.LpContinuous)

        # 4. Objective function

        # Removed _get_q_c_vals since supply uncertainty is disabled
        if use_gurobi:
            transport_cost_term = gp.quicksum(
                column_variable_cost(col) * lambdas[col.col_id] for col in columns
            )
            rail_transport_cost_term = gp.quicksum(
                rail_unit_cost[key] * rail_tier_load[key]
                for key in rail_tier_load
            )
            repo_cost_term = gp.quicksum(
                valid_reposition_transitions[pair][1] * pi_reposition[pair]
                for pair in valid_reposition_transitions
            )
            demand_penalty_cost_term = gp.quicksum(
                c_unmet_penalty[i] * shortage_vars[i] for i, od in enumerate(self.od_pairs)
            )
            
            dummy_cost_term = gp.quicksum(self.config.big_m * dummy_vars[i] for i in range(len(self.od_pairs)))
            
            prob.setObjective(
                transport_cost_term + rail_transport_cost_term + repo_cost_term
                + demand_penalty_cost_term + dummy_cost_term,
                GRB.MINIMIZE
            )
        else:
            transport_cost_term = pulp.lpSum(
                column_variable_cost(col) * lambdas[col.col_id] for col in columns
            )
            rail_transport_cost_term = pulp.lpSum(
                rail_unit_cost[key] * rail_tier_load[key]
                for key in rail_tier_load
            )
            repo_cost_term = pulp.lpSum(
                valid_reposition_transitions[pair][1] * pi_reposition[pair]
                for pair in valid_reposition_transitions
            )
            demand_penalty_cost_term = pulp.lpSum(
                c_unmet_penalty[i] * shortage_vars[i] for i, od in enumerate(self.od_pairs)
            )
            
            dummy_cost_term = pulp.lpSum(self.config.big_m * dummy_vars[i] for i in range(len(self.od_pairs)))
            
            prob += (
                transport_cost_term + rail_transport_cost_term + repo_cost_term
                + demand_penalty_cost_term + dummy_cost_term
            )

        # 5. Constraints

        # Exact demand partition: every FEU is either delivered or unmet.
        for i, od in enumerate(self.od_pairs):
            if use_gurobi:
                prob.addConstr(
                    q_delivered[i] + shortage_vars[i] == od.demand,
                    name=f"demand_balance_{i}",
                )
            else:
                prob += (
                    q_delivered[i] + shortage_vars[i] == od.demand,
                    f"demand_balance_{i}",
                )

        # A loaded road dispatch always carries exactly one 40-foot container (1 FEU).
        capacity_constraints = {}
        for i, od in enumerate(self.od_pairs):
            cols_for_od = [col for col in columns if col.od_idx == i]
            if use_gurobi:
                prob.addConstr(
                    gp.quicksum(lambdas[col.col_id] * q_veh for col in cols_for_od) + dummy_vars[i] == q_delivered[i],
                    name=f"capacity_link_{i}"
                )
            else:
                c_cap = (
                    pulp.lpSum(lambdas[col.col_id] * q_veh for col in cols_for_od)
                    + dummy_vars[i]
                    - q_delivered[i]
                )
                capacity_constraints[i] = pulp.LpConstraint(c_cap, pulp.LpConstraintEQ, name=f"capacity_link_{i}", rhs=0)
                prob += capacity_constraints[i]

        # Each timetable dispatch selects at most one load-dependent FEU tariff tier.
        rail_capacity_constraints = {}
        for train_id in rail_runs:
            using_columns = [
                col for col in columns if train_id in column_train_ids(col)
            ]
            if use_gurobi:
                constr = prob.addConstr(
                    gp.quicksum(q_veh * lambdas[col.col_id] for col in using_columns)
                    == gp.quicksum(rail_tier_load[train_id, p] for p in range(len(rail_tiers))),
                    name=f"rail_capacity_{train_id}",
                )
                rail_capacity_constraints[train_id] = constr
                prob.addConstr(
                    gp.quicksum(rail_tier_active[train_id, p] for p in range(len(rail_tiers))) <= 1,
                    name=f"rail_tier_select_{train_id}",
                )
                for p, (lower, upper, _, _) in enumerate(rail_tiers):
                    tier_upper = upper if p == len(rail_tiers) - 1 else upper - self.config.tolerance
                    prob.addConstr(
                        rail_tier_load[train_id, p] >= lower * rail_tier_active[train_id, p],
                        name=f"rail_tier_lower_{train_id}_{p}",
                    )
                    prob.addConstr(
                        rail_tier_load[train_id, p] <= tier_upper * rail_tier_active[train_id, p],
                        name=f"rail_tier_upper_{train_id}_{p}",
                    )
            else:
                constr = pulp.LpConstraint(
                    pulp.lpSum(q_veh * lambdas[col.col_id] for col in using_columns)
                    - pulp.lpSum(rail_tier_load[train_id, p] for p in range(len(rail_tiers))),
                    pulp.LpConstraintEQ,
                    name=f"rail_capacity_{train_id}",
                    rhs=0,
                )
                prob += constr
                rail_capacity_constraints[train_id] = constr
                prob += (
                    pulp.lpSum(rail_tier_active[train_id, p] for p in range(len(rail_tiers))) <= 1,
                    f"rail_tier_select_{train_id}",
                )
                for p, (lower, upper, _, _) in enumerate(rail_tiers):
                    tier_upper = upper if p == len(rail_tiers) - 1 else upper - self.config.tolerance
                    prob += (
                        rail_tier_load[train_id, p] >= lower * rail_tier_active[train_id, p],
                        f"rail_tier_lower_{train_id}_{p}",
                    )
                    prob += (
                        rail_tier_load[train_id, p] <= tier_upper * rail_tier_active[train_id, p],
                        f"rail_tier_upper_{train_id}_{p}",
                    )

        # Demand satisfaction at origin/destination (S2-8)


        # Per-hub fleet hard constraints
        if self.hub_fleet:
            for hub_id_str, fleet_size in self.hub_fleet.items():
                hub_id = int(hub_id_str)
                hub_tasks = [
                    task for task, data in road_task_nodes.items()
                    if data["province_hub_id"] == hub_id
                ]
                if hub_tasks:
                    initial_count = (
                        gp.quicksum(initial_vehicle[task] for task in hub_tasks)
                        if use_gurobi else
                        pulp.lpSum(initial_vehicle[task] for task in hub_tasks)
                    )
                    if use_gurobi:
                        prob.addConstr(
                            initial_count <= fleet_size,
                            name=f"hub_fleet_{hub_id}",
                        )
                    else:
                        prob += (
                            initial_count <= fleet_size,
                            f"hub_fleet_{hub_id}",
                        )

        

        # Flow conservation for repositioning vehicles
        for task in road_task_nodes:
            incoming = [
                pi_reposition[pair] for pair in valid_reposition_transitions
                if pair[1] == task
            ]
            outgoing = [
                pi_reposition[pair] for pair in valid_reposition_transitions
                if pair[0] == task
            ]
            task_flow = lambdas[task[0]]
            if use_gurobi:
                prob.addConstr(
                    initial_vehicle[task] + gp.quicksum(incoming) == task_flow,
                    name=f"repo_balance_{task[0]}_{task[1]}",
                )
                prob.addConstr(
                    gp.quicksum(incoming) <= task_flow,
                    name=f"repo_in_{task[0]}_{task[1]}",
                )
                prob.addConstr(
                    gp.quicksum(outgoing) <= task_flow,
                    name=f"repo_out_{task[0]}_{task[1]}",
                )
            else:
                prob += (
                    initial_vehicle[task] + pulp.lpSum(incoming) == task_flow,
                    f"repo_balance_{task[0]}_{task[1]}",
                )
                prob += (
                    pulp.lpSum(incoming) <= task_flow,
                    f"repo_in_{task[0]}_{task[1]}",
                )
                prob += (
                    pulp.lpSum(outgoing) <= task_flow,
                    f"repo_out_{task[0]}_{task[1]}",
                )

        # Branching constraints
        for c_type, c_data in branch_constraints:
            if c_type == "rf_same":
                i, j = c_data
                if use_gurobi:
                    prob.addConstr(
                        gp.quicksum(lambdas[col.col_id] for col in columns if not self._check_rf(col, i, j, True)) == 0,
                        name=f"rf_same_{i}_{j}"
                    )
                else:
                    prob += (
                        pulp.lpSum(
                            lambdas[col.col_id] for col in columns if not self._check_rf(col, i, j, True)
                        )
                        == 0,
                        f"rf_same_{i}_{j}",
                    )
            elif c_type == "rf_diff":
                i, j = c_data
                if use_gurobi:
                    prob.addConstr(
                        gp.quicksum(lambdas[col.col_id] for col in columns if not self._check_rf(col, i, j, False)) == 0,
                        name=f"rf_diff_{i}_{j}"
                    )
                else:
                    prob += (
                        pulp.lpSum(
                            lambdas[col.col_id] for col in columns if not self._check_rf(col, i, j, False)
                        )
                        == 0,
                        f"rf_diff_{i}_{j}",
                    )

        # S2-8: Optional stabilization: add trust region around previous duals
        if self.use_stabilization and self.prev_duals and use_gurobi:
            # Gurobi-specific stabilization implementation
            # ... (implementation omitted for brevity)
            pass # Placeholder for Gurobi stabilization logic if needed

        # 6. Solve LP
        import time
        optimize_wall_start = time.perf_counter()
        optimize_cpu_start = time.process_time()
        if use_gurobi:
            self.logger.info(
                f"  RMP optimize: columns={len(columns)}, active rail dispatches={len(rail_runs)}"
            )
            prob.optimize()
            optimize_wall = time.perf_counter() - optimize_wall_start
            optimize_cpu = time.process_time() - optimize_cpu_start
            self.total_master_wall_time += optimize_wall
            self.total_master_cpu_time += optimize_cpu
            if is_integer:
                self.total_master_mip_wall_time += optimize_wall
            else:
                self.total_master_lp_wall_time += optimize_wall
            if prob.status == GRB.INFEASIBLE:
                prob.computeIIS()
                prob.write("rmp_infeasible.ilp")
                self.logger.warning("RMP is INFEASIBLE. Computed IIS and saved to rmp_infeasible.ilp")
                return None, {}
            elif prob.status == GRB.OPTIMAL or prob.status == GRB.SUBOPTIMAL:
                lp_val = prob.ObjVal
                # Skip dual extraction for MIP
                if is_integer:
                    duals = {}
                else:
                    duals = self._extract_gurobi_duals(
                        prob, capacity_constraints, rail_capacity_constraints,
                    )
            else:
                self.logger.warning(f"Gurobi did not find an optimal solution. Status: {prob.status}")
                return None, {}
        else:
            if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
                solver = pulp.HiGHS_CMD(msg=0)
            elif hasattr(self.config, 'cbc_path') and os.path.exists(self.config.cbc_path):
                solver = pulp.COIN_CMD(path=self.config.cbc_path, msg=0)
            else:
                solver = pulp.PULP_CBC_CMD(msg=0)

            prob.solve(solver)
            optimize_wall = time.perf_counter() - optimize_wall_start
            optimize_cpu = time.process_time() - optimize_cpu_start
            self.total_master_wall_time += optimize_wall
            self.total_master_cpu_time += optimize_cpu
            if is_integer:
                self.total_master_mip_wall_time += optimize_wall
            else:
                self.total_master_lp_wall_time += optimize_wall

            if prob.status == pulp.LpStatusOptimal:
                lp_val = pulp.value(prob.objective)
                # Skip dual extraction for MIP
                if is_integer:
                    duals = {}
                else:
                    duals = self._extract_pulp_duals(
                        prob, capacity_constraints, rail_capacity_constraints,
                    )
            else:
                self.logger.warning(f"Pulp did not find an optimal solution. Status: {prob.status}")
                return None, {}


        # 7. Update column LP values
        for col in columns:
            col.lp_value = lambdas[col.col_id].X if use_gurobi else lambdas[col.col_id].varValue

        self.reposition_solution = {
            "initial": {
                task: (var.X if use_gurobi else var.varValue or 0.0)
                for task, var in initial_vehicle.items()
            },
            "transitions": {
                pair: (var.X if use_gurobi else var.varValue or 0.0)
                for pair, var in pi_reposition.items()
            },
            "task_data": road_task_nodes,
        }
        self.rail_tariff_solution = {
            "loads": {
                key: (var.X if use_gurobi else var.varValue or 0.0)
                for key, var in rail_tier_load.items()
            },
            "unit_costs": rail_unit_cost,
        }
        # S2-8: Update prev_duals for stabilization in next iteration
        self.prev_duals = duals

        return lp_val, duals

    def _extract_gurobi_duals(self, prob, capacity_constraints, rail_capacity_constraints):
        duals = {
            "od": {},
            "vehicle": {},
            "trans_out": {},
            "trans_in": {},
            "rail": {},
        }
        for i in range(len(self.od_pairs)):
            duals["od"][i] = prob.getConstrByName(f"capacity_link_{i}").Pi
            # Reposition-flow constraints are column/task-specific and have no
            # mathematically valid one-to-one OD dual for the pricing problem.
            duals["trans_out"][i] = 0.0
            duals["trans_in"][i] = 0.0
        
        duals["vehicle"][0] = 0.0
        for train_id, constr in rail_capacity_constraints.items():
            duals["rail"][train_id] = constr.Pi
        return duals

    def _extract_pulp_duals(self, prob, capacity_constraints, rail_capacity_constraints):
        duals = {
            "od": {},
            "vehicle": {},
            "trans_out": {},
            "trans_in": {},
            "rail": {},
        }
        for i in range(len(self.od_pairs)):
            cap_pi = capacity_constraints[i].pi or 0.0
            duals["od"][i] = cap_pi
            duals["trans_out"][i] = 0.0
            duals["trans_in"][i] = 0.0
            
        duals["vehicle"][0] = 0.0
        for train_id, constr in rail_capacity_constraints.items():
            duals["rail"][train_id] = constr.pi or 0.0

        return duals

    def _check_rf(self, col, i, j, same: bool):
        nodes = {arc.i for arc in col.arcs} | {arc.j for arc in col.arcs}
        has_i = i in nodes
        has_j = j in nodes
        if same:
            return has_i == has_j
        else:
            return not (has_i and has_j)
