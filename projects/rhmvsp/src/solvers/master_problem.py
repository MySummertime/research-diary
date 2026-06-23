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
         + sum_hub c_hub * g_hub
         + sum_od [p_pen + c_unmet_penalty] * shortfall_i
         + repositioning costs (vehicle idle + empty drive + opportunity)
    """

    def __init__(self, config: RHMVSPConfig, network, logger, instance: dict = None):
        self.config = config
        self.network = network
        self.od_pairs = network.od_pairs
        self.num_vehicles_west = instance.get("num_vehicles_west", 2500)
        self.num_vehicles_east = instance.get("num_vehicles_east", 2500)
        self.hub_fleet = instance.get("hub_fleet_size", {})  # per-hub fleet: {hub_id: fleet_size}
        self.logger = logger
        self.instance = instance or {}
        self.engine = UncertaintyEngine(config)

        # Stabilization state
        self.mu = getattr(config, 'stabilization_mu', 0.5)
        self.prev_duals = None
        self.use_stabilization = getattr(config, 'bap_stabilization', True)
        self.next_col_id = 0

    def _compute_unmet_penaltys(self):
        """
        Compute opportunity costs per OD pair.
        """
        c_unmet_penalty = {}
        for i, od in enumerate(self.od_pairs):
            # Unmet penalty per ton is defined strictly as a fixed unit cost
            # under Wang 2025, without additional O-D distance adjustment.
            c_unmet_penalty[i] = self.config.c_unmet_penalty

        return c_unmet_penalty

    def solve(self, columns: list, branch_constraints: list = None, is_integer: bool = False) -> tuple[float | None, dict]:
        """
        Solve exact RMP LP relaxation / MIP.

        Objective:
          min  sum_col cost * lambda_col
             + sum_hub c_hub * g_hub
             + sum_od [p_pen + c_unmet_penalty] * shortfall_i
             + repositioning costs (vehicle idle + empty drive + opportunity)
        """
        if not columns:
            return None, {}
        branch_constraints = branch_constraints if branch_constraints is not None else []

        # 1. Always instantiate all transfer hubs in the Master Problem to support branch constraints safely
        transfer_node_ids = sorted(list(self.network.transfer_nodes.keys()))

        # 2. Compute opportunity costs
        c_unmet_penalty = self._compute_unmet_penaltys()

        q_veh = self.config.q_road
        c_idle = self.config.c_vehicle_idle
        d_road = self.config.c_deploy_road

        # 3. Create LP Problem
        if self.config.solver_name == "gurobi" and gp:
            try:
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
                prob = gp.Model("RMP", env=env)
                prob.setParam('OutputFlag', 0)  # Suppress Gurobi output
            except gp.GurobiError as e:
                self.logger.warning(f"Gurobi license error: {e}. Falling back to Pulp.")
                prob = pulp.LpProblem("RMP", pulp.LpMinimize)
                use_gurobi = False
            else:
                use_gurobi = True
        else:
            prob = pulp.LpProblem("RMP", pulp.LpMinimize)
            use_gurobi = False

        # Decision Variables
        if use_gurobi:
            lambdas_own = {
                col.col_id: prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS, name=f"lambda_own_{col.col_id}")
                for col in columns
            }
            lambdas_rent = {
                col.col_id: prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS, name=f"lambda_rent_{col.col_id}")
                for col in columns
            }
            g_vars = {
                i: prob.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"g_{i}")
                for i in transfer_node_ids
            }
            dummy_vars = {
                i: prob.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"dummy_{i}")
                for i in range(len(self.od_pairs))
            }
        else:
            lambdas_own = {
                col.col_id: pulp.LpVariable(f"lambda_own_{col.col_id}", 0, None, pulp.LpContinuous)
                for col in columns
            }
            lambdas_rent = {
                col.col_id: pulp.LpVariable(f"lambda_rent_{col.col_id}", 0, None, pulp.LpContinuous)
                for col in columns
            }
            g_vars = {
                i: pulp.LpVariable(f"g_{i}", 0, 1, pulp.LpContinuous)
                for i in transfer_node_ids
            }
            dummy_vars = {
                i: pulp.LpVariable(f"dummy_{i}", 0, None, pulp.LpContinuous)
                for i in range(len(self.od_pairs))
            }

        # OD-level Empty Repositioning Variables pi_{i,j}
        # Decision: reuse a vehicle from OD_i for OD_j vs. deploying a new one.
        # Full comparison (user's insight):
        #   reposition total = empty_drive_fee + extra_inventory_cost_during_wait
        #   new_veh total    = deploy_fee (available immediately at time_window_start_j)
        #   → only reposition if total_repo - d_road < 0
        valid_transitions_west = {}
        valid_transitions_east = {}
        for i, od_i in enumerate(self.od_pairs):
            for j, od_j in enumerate(self.od_pairs):
                # West: Reposition from OD_i West Hub to OD_j Origin
                dist_w = self.network.get_exact_road_distance(od_i.origin, od_j.origin)
                prep_h = getattr(self.config, 'prep_time_hours', 2.0)
                dispatch_buf = getattr(self.config, 'dispatch_buffer_hours', 1.0)
                prep_repos = getattr(self.config, 'c_prep_reposition', 200.0)
                req_gap_w = max(0.1, dist_w / self.config.road_speed) + prep_h + dispatch_buf
                arr_w_i = od_i.time_window_start + 10.0
                if arr_w_i + req_gap_w <= od_j.time_window_end:
                    empty_fee = self.config.c_empty_reposition_per_km * dist_w
                    repo_arr = arr_w_i + req_gap_w
                    extra_wait = max(0.0, repo_arr - od_j.time_window_start)
                    inv_rate = self.config.c_vehicle_idle / 24.0
                    time_factor = 0.05
                    extra_inv_cost = q_veh * inv_rate * (extra_wait + 0.5 * time_factor * extra_wait**2)
                    valid_transitions_west[(i, j)] = (dist_w, prep_repos + empty_fee + extra_inv_cost)

                # East: Reposition from OD_i Destination to OD_j East Hub
                dist_e = self.network.get_exact_road_distance(od_i.destination, od_j.destination)
                req_gap_e = max(0.1, dist_e / self.config.road_speed) + prep_h + dispatch_buf
                arr_e_i = od_i.time_window_end - getattr(self.config, 'time_window_end_buffer_hours', 1.5)
                if arr_e_i + req_gap_e <= od_j.time_window_end:
                    empty_fee = self.config.c_empty_reposition_per_km * dist_e
                    repo_arr = arr_e_i + req_gap_e
                    extra_wait = max(0.0, repo_arr - od_j.time_window_start)
                    extra_inv_cost = q_veh * inv_rate * (extra_wait + 0.5 * time_factor * extra_wait**2)
                    valid_transitions_east[(i, j)] = (dist_e, prep_repos + empty_fee + extra_inv_cost)

        # Decision Variables
        if use_gurobi:
            lambdas_own = {
                col.col_id: prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS, name=f"lambda_own_{col.col_id}")
                for col in columns
            }
            lambdas_rent = {
                col.col_id: prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS, name=f"lambda_rent_{col.col_id}")
                for col in columns
            }
            g_vars = {
                i: prob.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"g_{i}")
                for i in transfer_node_ids
            }
            dummy_vars = {
                i: prob.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"dummy_{i}")
                for i in range(len(self.od_pairs))
            }
        else:
            lambdas_own = {
                col.col_id: pulp.LpVariable(f"lambda_own_{col.col_id}", 0, None, pulp.LpContinuous)
                for col in columns
            }
            lambdas_rent = {
                col.col_id: pulp.LpVariable(f"lambda_rent_{col.col_id}", 0, None, pulp.LpContinuous)
                for col in columns
            }
            g_vars = {
                i: pulp.LpVariable(f"g_{i}", 0, 1, pulp.LpContinuous)
                for i in transfer_node_ids
            }
            dummy_vars = {
                i: pulp.LpVariable(f"dummy_{i}", 0, None, pulp.LpContinuous)
                for i in range(len(self.od_pairs))
            }

        # Decision Variables (repositioning, delivery, shortfall)
        if use_gurobi:
            # OD-level Empty Repositioning Variables pi_{i,j}
            pi_west_own = { pair: prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS, name=f"pi_west_own_{pair[0]}_{pair[1]}") for pair in valid_transitions_west }
            pi_east_own = { pair: prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS, name=f"pi_east_own_{pair[0]}_{pair[1]}") for pair in valid_transitions_east }
            pi_west_rent = { pair: prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS, name=f"pi_west_rent_{pair[0]}_{pair[1]}") for pair in valid_transitions_west }
            pi_east_rent = { pair: prob.addVar(lb=0, vtype=GRB.INTEGER if is_integer else GRB.CONTINUOUS, name=f"pi_east_rent_{pair[0]}_{pair[1]}") for pair in valid_transitions_east }
            q_delivered = {
                i: prob.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q_del_{i}")
                for i, od in enumerate(self.od_pairs)
            }
            shortage_vars = {}
            for i in range(len(self.od_pairs)):
                shortage_vars[i] = prob.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"shortage_{i}")
            prob.update()
        else:
            pi_west_own = { pair: pulp.LpVariable(f"pi_west_own_{pair[0]}_{pair[1]}", 0, None, pulp.LpContinuous) for pair in valid_transitions_west }
            pi_east_own = { pair: pulp.LpVariable(f"pi_east_own_{pair[0]}_{pair[1]}", 0, None, pulp.LpContinuous) for pair in valid_transitions_east }
            pi_west_rent = { pair: pulp.LpVariable(f"pi_west_rent_{pair[0]}_{pair[1]}", 0, None, pulp.LpContinuous) for pair in valid_transitions_west }
            pi_east_rent = { pair: pulp.LpVariable(f"pi_east_rent_{pair[0]}_{pair[1]}", 0, None, pulp.LpContinuous) for pair in valid_transitions_east }
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
            
            c_rent_hr = getattr(self.config, 'c_rent_per_hour', 100.0)
            c_dep_rent = getattr(self.config, 'c_deploy_rent', 1000.0)
            
            # Own costs
            transport_own = gp.quicksum(col.total_cost * lambdas_own[col.col_id] for col in columns)
            repo_own = gp.quicksum(valid_transitions_west[pair][1] * pi_west_own[pair] for pair in valid_transitions_west) + gp.quicksum(valid_transitions_east[pair][1] * pi_east_own[pair] for pair in valid_transitions_east)
            
            # Rent costs
            transport_rent = gp.quicksum(
                (col.total_cost + c_dep_rent + c_rent_hr * (col.arrival_time - col.departure_time)) * lambdas_rent[col.col_id]
                for col in columns
            )
            # Gap duration = od_j.time_window_end - buffer - arrival_time_i
            def get_gap(p):
                i, j = p
                od_i = self.od_pairs[i]
                od_j = self.od_pairs[j]
                buf = getattr(self.config, 'time_window_end_buffer_hours', 1.5)
                arr_i = od_i.time_window_end - buf
                dep_j = od_j.time_window_end - buf
                return max(0.0, dep_j - arr_i)
                
            repo_rent = gp.quicksum((valid_transitions_west[pair][1] + c_rent_hr * get_gap(pair) - c_dep_rent) * pi_west_rent[pair] for pair in valid_transitions_west) + gp.quicksum((valid_transitions_east[pair][1] + c_rent_hr * get_gap(pair) - c_dep_rent) * pi_east_rent[pair] for pair in valid_transitions_east)
            
            transport_cost_term = transport_own + transport_rent
            repo_cost_term = repo_own + repo_rent
            hub_cost_term = gp.quicksum(self.config.c_hub * g_vars[i] for i in transfer_node_ids)
            
            demand_penalty_cost_term = gp.quicksum(
                c_unmet_penalty[i] * shortage_vars[i] for i, od in enumerate(self.od_pairs)
            )
            
            dummy_cost_term = gp.quicksum(self.config.big_m * dummy_vars[i] for i in range(len(self.od_pairs)))
            
            prob.setObjective(
                transport_cost_term + repo_cost_term + hub_cost_term
                + demand_penalty_cost_term + dummy_cost_term,
                GRB.MINIMIZE
            )
        else:
            
            c_rent_hr = getattr(self.config, 'c_rent_per_hour', 100.0)
            c_dep_rent = getattr(self.config, 'c_deploy_rent', 1000.0)
            
            transport_own = pulp.lpSum(col.total_cost * lambdas_own[col.col_id] for col in columns)
            repo_own = pulp.lpSum(valid_transitions_west[pair][1] * pi_west_own[pair] for pair in valid_transitions_west) + pulp.lpSum(valid_transitions_east[pair][1] * pi_east_own[pair] for pair in valid_transitions_east)
            
            transport_rent = pulp.lpSum(
                (col.total_cost + c_dep_rent + c_rent_hr * (col.arrival_time - col.departure_time)) * lambdas_rent[col.col_id]
                for col in columns
            )
            def get_gap(p):
                i, j = p
                od_i = self.od_pairs[i]
                od_j = self.od_pairs[j]
                buf = getattr(self.config, 'time_window_end_buffer_hours', 1.5)
                arr_i = od_i.time_window_end - buf
                dep_j = od_j.time_window_end - buf
                return max(0.0, dep_j - arr_i)
                
            repo_rent = pulp.lpSum((valid_transitions_west[pair][1] + c_rent_hr * get_gap(pair) - c_dep_rent) * pi_west_rent[pair] for pair in valid_transitions_west) + pulp.lpSum((valid_transitions_east[pair][1] + c_rent_hr * get_gap(pair) - c_dep_rent) * pi_east_rent[pair] for pair in valid_transitions_east)
            
            transport_cost_term = transport_own + transport_rent
            repo_cost_term = repo_own + repo_rent
            hub_cost_term = pulp.lpSum(self.config.c_hub * g_vars[i] for i in transfer_node_ids)
            
            demand_penalty_cost_term = pulp.lpSum(
                c_unmet_penalty[i] * shortage_vars[i] for i, od in enumerate(self.od_pairs)
            )
            
            dummy_cost_term = pulp.lpSum(self.config.big_m * dummy_vars[i] for i in range(len(self.od_pairs)))
            
            prob += (
                transport_cost_term + repo_cost_term + hub_cost_term
                + demand_penalty_cost_term + dummy_cost_term
            )

        # 5. Constraints

        # Demand shortfall definition
        for i, od in enumerate(self.od_pairs):
            if use_gurobi:
                prob.addConstr(shortage_vars[i] >= od.demand - q_delivered[i], name=f"short_def1_{i}")
            else:
                prob += shortage_vars[i] >= od.demand - q_delivered[i], f"short_def1_{i}"

        # S2-8: Capacity link — delivered quantity <= physical capacity from dispatched columns + dummy variables
        capacity_constraints = {}
        for i, od in enumerate(self.od_pairs):
            cols_for_od = [col for col in columns if col.od_idx == i]
            if use_gurobi:
                prob.addConstr(
                    gp.quicksum((lambdas_own[col.col_id] + lambdas_rent[col.col_id]) * q_veh for col in cols_for_od) + dummy_vars[i] >= q_delivered[i],
                    name=f"capacity_link_{i}"
                )
            else:
                c_cap = (
                    pulp.lpSum((lambdas_own[col.col_id] + lambdas_rent[col.col_id]) * q_veh for col in cols_for_od)
                    + dummy_vars[i]
                    - q_delivered[i]
                )
                capacity_constraints[i] = pulp.LpConstraint(c_cap, pulp.LpConstraintGE, name=f"capacity_link_{i}", rhs=0)
                prob += capacity_constraints[i]

        # Demand satisfaction at origin/destination (S2-8)


        # Vehicle balance: per-hub fleet model (each hub has its own fleet of vehicles)
        # When hub_fleet is set, use per-hub constraints; otherwise fall back to global fleet.
        # Per-hub constraint disabled for now — column generation needs more iterations
        # to produce enough repositioning-capable columns before the constraint can bind.
        if False and self.hub_fleet:
            for hub_id_str, fleet_size in self.hub_fleet.items():
                hub_id = int(hub_id_str)
                cols_for_hub = [col for col in columns if col.destination_hub == hub_id]
                if cols_for_hub:
                    # Count ALL repositionings where source uses hub h (including intra-hub)
                    # Each repositioning reuses one vehicle, reducing effective fleet consumption
                    repo_reuse = [pi_east_own[p] for p in valid_transitions_east 
                                  if self.od_pairs[p[0]].destination_hub == hub_id]
                    repo_reuse_rent = [pi_east_rent[p] for p in valid_transitions_east 
                                      if self.od_pairs[p[0]].destination_hub == hub_id]
                    if use_gurobi:
                        prob.addConstr(
                            gp.quicksum(lambdas_own[col.col_id] for col in cols_for_hub) 
                            - gp.quicksum(repo_reuse) - gp.quicksum(repo_reuse_rent) <= fleet_size,
                            name=f"fleet_hub_{hub_id}"
                        )
                    else:
                        prob += (
                            pulp.lpSum(lambdas_own[col.col_id] for col in cols_for_hub) 
                            - pulp.lpSum(repo_reuse) - pulp.lpSum(repo_reuse_rent) <= fleet_size,
                            f"fleet_hub_{hub_id}"
                        )
        else:
            # Fallback: global fleet balance (original model)
            veh_limit_w = self.num_vehicles_west
            veh_limit_e = self.num_vehicles_east
            if use_gurobi:
                prob.addConstr(gp.quicksum(lambdas_own.values()) - gp.quicksum(pi_west_own.values()) <= veh_limit_w, name="vehicle_balance_west")
                prob.addConstr(gp.quicksum(lambdas_own.values()) - gp.quicksum(pi_east_own.values()) <= veh_limit_e, name="vehicle_balance_east")
                prob.addConstr(gp.quicksum(lambdas_rent.values()) - gp.quicksum(pi_west_rent.values()) >= 0, name="vehicle_balance_rent_w")
                prob.addConstr(gp.quicksum(lambdas_rent.values()) - gp.quicksum(pi_east_rent.values()) >= 0, name="vehicle_balance_rent_e")
            else:
                prob += (pulp.lpSum(lambdas_own.values()) - pulp.lpSum(pi_west_own.values()) <= veh_limit_w, "vehicle_balance_west")
                prob += (pulp.lpSum(lambdas_own.values()) - pulp.lpSum(pi_east_own.values()) <= veh_limit_e, "vehicle_balance_east")
                prob += (pulp.lpSum(lambdas_rent.values()) - pulp.lpSum(pi_west_rent.values()) >= 0, "vehicle_balance_rent_w")
                prob += (pulp.lpSum(lambdas_rent.values()) - pulp.lpSum(pi_east_rent.values()) >= 0, "vehicle_balance_rent_e")

        

        # Flow conservation for repositioning vehicles
        for k in range(len(self.od_pairs)):
            in_w_own = [pi_west_own[p] for p in valid_transitions_west if p[1] == k]
            out_w_own = [pi_west_own[p] for p in valid_transitions_west if p[0] == k]
            in_e_own = [pi_east_own[p] for p in valid_transitions_east if p[1] == k]
            out_e_own = [pi_east_own[p] for p in valid_transitions_east if p[0] == k]
            cols_own = [lambdas_own[c.col_id] for c in columns if c.od_idx == k]
            
            if use_gurobi:
                prob.addConstr(gp.quicksum(in_w_own) <= gp.quicksum(cols_own), name=f"repo_w_in_own_{k}")
                prob.addConstr(gp.quicksum(out_w_own) <= gp.quicksum(cols_own), name=f"repo_w_out_own_{k}")
                prob.addConstr(gp.quicksum(in_e_own) <= gp.quicksum(cols_own), name=f"repo_e_in_own_{k}")
                prob.addConstr(gp.quicksum(out_e_own) <= gp.quicksum(cols_own), name=f"repo_e_out_own_{k}")
            else:
                prob += (pulp.lpSum(in_w_own) <= pulp.lpSum(cols_own), f"repo_w_in_own_{k}")
                prob += (pulp.lpSum(out_w_own) <= pulp.lpSum(cols_own), f"repo_w_out_own_{k}")
                prob += (pulp.lpSum(in_e_own) <= pulp.lpSum(cols_own), f"repo_e_in_own_{k}")
                prob += (pulp.lpSum(out_e_own) <= pulp.lpSum(cols_own), f"repo_e_out_own_{k}")

        # Hub activation constraints
        for t_node in transfer_node_ids:
            cols_via_hub = [
                lambdas_own[col.col_id] + lambdas_rent[col.col_id] for col in columns if t_node in col.transfer_nodes
            ]
            tn = self.network.transfer_nodes[t_node]
            if use_gurobi:
                prob.addConstr(
                    gp.quicksum(cols_via_hub) <= self.config.big_m * g_vars[t_node],
                    name=f"hub_activate_{t_node}"
                )
            else:
                prob += (
                    pulp.lpSum(cols_via_hub) <= self.config.big_m * g_vars[t_node],
                    f"hub_activate_{t_node}",
                )
                
            # Hub physical processing capacity constraint
            max_lifts = tn.max_cranes * self.config.crane_daily_lifts
            if use_gurobi:
                prob.addConstr(
                    gp.quicksum(cols_via_hub) <= max_lifts * g_vars[t_node],
                    name=f"hub_crane_capacity_{t_node}"
                )
            else:
                prob += (
                    pulp.lpSum(cols_via_hub) <= max_lifts * g_vars[t_node],
                    f"hub_crane_capacity_{t_node}",
                )

        # Max network active hubs constraint
        if use_gurobi:
            prob.addConstr(
                gp.quicksum(g_vars.values()) <= (self.config.max_hubs if self.config.max_hubs is not None else len(transfer_node_ids)),
                name="max_hubs_limit"
            )
        else:
            prob += (
                pulp.lpSum(g_vars.values()) <= (self.config.max_hubs if self.config.max_hubs is not None else len(transfer_node_ids)),
                "max_hubs_limit",
            )

        # Branching constraints (S2-8: Ryan-Foster and Hub Activation)
        for c_type, c_data in branch_constraints:
            if c_type == "rf_same":
                i, j = c_data
                if use_gurobi:
                    prob.addConstr(
                        gp.quicksum(lambdas_own[col.col_id] + lambdas_rent[col.col_id] for col in columns if not self._check_rf(col, i, j, True)) == 0,
                        name=f"rf_same_{i}_{j}"
                    )
                else:
                    prob += (
                        pulp.lpSum(
                            lambdas_own[col.col_id] + lambdas_rent[col.col_id] for col in columns if not self._check_rf(col, i, j, True)
                        )
                        == 0,
                        f"rf_same_{i}_{j}",
                    )
            elif c_type == "rf_diff":
                i, j = c_data
                if use_gurobi:
                    prob.addConstr(
                        gp.quicksum(lambdas_own[col.col_id] + lambdas_rent[col.col_id] for col in columns if not self._check_rf(col, i, j, False)) == 0,
                        name=f"rf_diff_{i}_{j}"
                    )
                else:
                    prob += (
                        pulp.lpSum(
                            lambdas_own[col.col_id] + lambdas_rent[col.col_id] for col in columns if not self._check_rf(col, i, j, False)
                        )
                        == 0,
                        f"rf_diff_{i}_{j}",
                    )
            elif c_type == "hub_close":
                hub_id = c_data
                if use_gurobi:
                    prob.addConstr(g_vars[hub_id] == 0, name=f"hub_close_{hub_id}")
                else:
                    prob += (g_vars[hub_id] == 0, f"hub_close_{hub_id}")
            elif c_type == "hub_open":
                hub_id = c_data
                if use_gurobi:
                    prob.addConstr(g_vars[hub_id] == 1, name=f"hub_open_{hub_id}")
                else:
                    prob += (g_vars[hub_id] == 1, f"hub_open_{hub_id}")

        # S2-8: Optional stabilization: add trust region around previous duals
        if self.use_stabilization and self.prev_duals and use_gurobi:
            # Gurobi-specific stabilization implementation
            # ... (implementation omitted for brevity)
            pass # Placeholder for Gurobi stabilization logic if needed

        # 6. Solve LP
        if use_gurobi:
            prob.optimize()
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
                    duals = self._extract_gurobi_duals(prob, capacity_constraints, transfer_node_ids)
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

            if prob.status == pulp.LpStatusOptimal:
                lp_val = pulp.value(prob.objective)
                # Skip dual extraction for MIP
                if is_integer:
                    duals = {}
                else:
                    duals = self._extract_pulp_duals(prob, capacity_constraints, transfer_node_ids)
            else:
                self.logger.warning(f"Pulp did not find an optimal solution. Status: {prob.status}")
                return None, {}


        # 7. Update column LP values and hub activation values
        for col in columns:
            val_own = lambdas_own[col.col_id].X if use_gurobi else lambdas_own[col.col_id].varValue
            val_rent = lambdas_rent[col.col_id].X if use_gurobi else lambdas_rent[col.col_id].varValue
            col.lp_value_own = val_own
            col.lp_value_rent = val_rent
            col.lp_value = val_own + val_rent

        self.hub_values = {i: g_vars[i].X if use_gurobi else g_vars[i].varValue for i in transfer_node_ids}

        # S2-8: Update prev_duals for stabilization in next iteration
        self.prev_duals = duals

        return lp_val, duals

    def _extract_gurobi_duals(self, prob, capacity_constraints, transfer_node_ids):
        duals = {
            "od": {},
            "vehicle": {},
            "trans_out": {},
            "trans_in": {},
            "hub": {},
        }
        for i in range(len(self.od_pairs)):
            duals["od"][i] = prob.getConstrByName(f"capacity_link_{i}").Pi
            demand_pi = prob.getConstrByName(f"capacity_link_{i}").Pi
            repo_pi_in = (prob.getConstrByName(f"repo_w_in_own_{i}").Pi + prob.getConstrByName(f"repo_e_in_own_{i}").Pi)
            repo_pi_out = (prob.getConstrByName(f"repo_w_out_own_{i}").Pi + prob.getConstrByName(f"repo_e_out_own_{i}").Pi)
            duals["trans_out"][i] = repo_pi_out
            duals["trans_in"][i] = repo_pi_in
        
        duals["vehicle"][0] = (prob.getConstrByName("vehicle_balance_west").Pi + prob.getConstrByName("vehicle_balance_east").Pi)

        for t_node in transfer_node_ids:
            act_pi = prob.getConstrByName(f"hub_activate_{t_node}").Pi
            cap_pi = prob.getConstrByName(f"hub_crane_capacity_{t_node}").Pi
            duals["hub"][t_node] = act_pi + cap_pi
            
        return duals

    def _extract_pulp_duals(self, prob, capacity_constraints, transfer_node_ids):
        duals = {
            "od": {},
            "vehicle": {},
            "trans_out": {},
            "trans_in": {},
            "hub": {},
        }
        for i in range(len(self.od_pairs)):
            cap_pi = capacity_constraints[i].pi or 0.0
            duals["od"][i] = cap_pi
            demand_pi = prob.constraints[f"capacity_link_{i}"].pi or 0.0
            repo_pi_in = ((prob.constraints[f"repo_w_in_own_{i}"].pi or 0.0) + (prob.constraints[f"repo_e_in_own_{i}"].pi or 0.0))
            repo_pi_out = ((prob.constraints[f"repo_w_out_own_{i}"].pi or 0.0) + (prob.constraints[f"repo_e_out_own_{i}"].pi or 0.0))
            duals["trans_out"][i] = repo_pi_out
            duals["trans_in"][i] = repo_pi_in
            
        duals["vehicle"][0] = ((prob.constraints["vehicle_balance_west"].pi or 0.0) + (prob.constraints["vehicle_balance_east"].pi or 0.0))
        
        for t_node in transfer_node_ids:
            act_pi = prob.constraints[f"hub_activate_{t_node}"].pi or 0.0
            cap_pi = prob.constraints[f"hub_crane_capacity_{t_node}"].pi or 0.0
            duals["hub"][t_node] = act_pi + cap_pi
            
        return duals

    def _check_rf(self, col, i, j, same: bool):
        nodes = {arc.i for arc in col.arcs} | {arc.j for arc in col.arcs}
        has_i = i in nodes
        has_j = j in nodes
        if same:
            return has_i == has_j
        else:
            return not (has_i and has_j)
