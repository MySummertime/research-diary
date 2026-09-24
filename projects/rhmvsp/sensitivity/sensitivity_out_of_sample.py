"""Fixed-plan out-of-sample evaluation of MR designs and one MW baseline."""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from sensitivity.parameter_grids import ALPHA_R_OOS_LEVELS, ALPHA_T_OOS_LEVELS
from sensitivity.utils import format_routing_summary, update_od_deadlines
from src.core.scenario_sampler import sample_scenario
from src.core.uncertainty import TrapezoidalFuzzy, UncertaintyEngine
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.utils.numeric_format import format_alpha_R, format_alpha_T, format_csv_value

N_SCENARIOS = 200
SCENARIO_SEED = RHMVSPConfig.seed
INSTANCE_PATH = "data_real_case/instance.json"
PARAM_GRID = [(a_t, a_r) for a_t in ALPHA_T_OOS_LEVELS for a_r in ALPHA_R_OOS_LEVELS]


def _solve_nominal(config, instance, model):
    """Optimize an ex-ante plan once, before any scenario is observed."""
    solver = BAPSolver(config, json.loads(json.dumps(instance)), Logger(level="INFO", log_file=None))
    solution = solver.solve()
    if solution.get("status") in ("infeasible", "error", None):
        raise RuntimeError(f"Nominal {model} solve failed: {solution.get('status')}")
    solution["model"] = model
    solution["routes"] = format_routing_summary(solution, instance)
    return solution


def _arc_index(instance):
    return {(int(a["i"]), int(a["j"]), int(a["mode"])): a for a in instance["network"]["arcs"]}


def _evaluate_route(route, scenario, config):
    """Evaluate one frozen route using scenario-realized time and risk."""
    arcs_by_key = _arc_index(scenario)
    transfers = {int(x["node_id"]): x for x in scenario["network"].get("transfer_nodes", [])}
    engine = UncertaintyEngine(config)
    departure = float(route.get("departure_time", 0.0))
    clock = departure
    supports, reliabilities = [], []
    previous_mode, continuous_drive = None, 0.0

    for ref in route.get("arcs", []):
        key = (int(ref["i"]), int(ref["j"]), int(ref["mode"]))
        arc = arcs_by_key.get(key)
        if arc is None:
            return {"valid": False}
        mode = key[2]
        if previous_mode is not None and not (previous_mode == mode == 1):
            transfer_time = config.transfer_time_hours(previous_mode, mode)
            supports.append((transfer_time,) * 4)
            node = transfers.get(key[0])
            if node:
                belief = "gamma_rr" if previous_mode == mode == 2 else "gamma_ir"
                reliabilities.append(1.0 - float(node.get(belief, 0.0)))
            clock += transfer_time
            continuous_drive = 0.0

        a = float(arc.get("time_a", 0.0))
        b = float(arc.get("time_b", a))
        c = float(arc.get("time_c", b))
        d = float(arc.get("time_d", c))
        support = (a, b, c, d)
        supports.append(support)
        duration = sum(support) / 4.0
        reliabilities.append(engine.arc_reliability(
            float(arc.get("alpha", 0.0)), clock,
            time_dependent=True, fuzzy_duration=support,
        ))
        clock += duration
        if mode == 1:
            rest, continuous_drive = engine.road_drive_transition(continuous_drive, duration)
            supports.append((rest,) * 4)
            clock += rest
        else:
            continuous_drive = 0.0
        previous_mode = mode

    if not supports:
        return {"valid": False}
    totals = [departure + sum(values) for values in zip(*supports)]
    deadline = min(float(scenario["od_pairs"][int(route["od_idx"])]["time_window_end"]), float(config.T_max))
    time_reliability = TrapezoidalFuzzy(*totals).uncertain_measure_leq(deadline)
    risk_reliability = min(reliabilities) if reliabilities else 1.0
    return {"valid": True, "risk_reliability": risk_reliability,
            "time_reliability": time_reliability, "risk": 1.0 - risk_reliability,
            "time": totals[-1]}


def _evaluate_fixed_plan(solution, scenario, config, alpha_t, alpha_r):
    """Evaluate frozen route/capacity decisions without re-optimization."""
    routes = solution.get("transport_routes", [])
    route_results = [_evaluate_route(route, scenario, config) for route in routes]
    valid = [result for result in route_results if result.get("valid")]
    capacity = {int(x["od_idx"]): float(x.get("delivered", 0.0))
                for x in solution.get("s2_allocation_details", [])}
    total_demand = total_delivered = unmet = 0.0
    for od_idx, od in enumerate(scenario["od_pairs"]):
        demand = float(od["demand"])
        delivered = min(demand, capacity.get(od_idx, 0.0))
        total_demand += demand
        total_delivered += delivered
        unmet += demand - delivered
    fixed_cost = sum(float(solution.get(k, 0.0) or 0.0)
                     for k in ("transport_cost", "transfer_cost", "reposition_cost"))
    has_valid_transport = bool(routes) and len(valid) == len(routes)
    risk_rel = min((x["risk_reliability"] for x in valid), default=None)
    time_rel = min((x["time_reliability"] for x in valid), default=None)
    service_failure = unmet > config.tolerance or len(valid) != len(routes)
    return {
        "cost": fixed_cost + config.c_unmet_penalty * unmet,
        "transport_cost": float(solution.get("transport_cost", 0.0) or 0.0),
        "transfer_cost": float(solution.get("transfer_cost", 0.0) or 0.0),
        "reposition_cost": float(solution.get("reposition_cost", 0.0) or 0.0),
        "unmet_penalty": config.c_unmet_penalty * unmet,
        "network_risk_reliability": risk_rel,
        "network_time_reliability": time_rel,
        # Risk, time, and their reliabilities are undefined when the nominal
        # plan dispatches no valid transport route.  Do not encode that state
        # as the fictitious extremes risk=1 or time=0.
        "network_risk": max((x["risk"] for x in valid), default=None),
        "network_time": max((x["time"] for x in valid), default=None),
        "unmet_demand": unmet,
        "delivery_ratio": total_delivered / total_demand if total_demand else 1.0,
        "service_failure": int(service_failure),
        "risk_violation": (int(risk_rel + config.tolerance < alpha_r)
                           if has_valid_transport else None),
        "time_violation": (int(time_rel + config.tolerance < alpha_t)
                           if has_valid_transport else None),
        "status": ("no_feasible_transport_plan" if not has_valid_transport
                   else "service_failure" if service_failure else "evaluated"),
    }


RESULT_FIELDS = ["cost", "transport_cost", "transfer_cost", "reposition_cost",
                 "unmet_penalty", "network_risk_reliability", "network_time_reliability",
                 "network_risk", "network_time", "unmet_demand", "delivery_ratio",
                 "service_failure", "risk_violation", "time_violation"]


def _write_row(writer, label, alpha_t, alpha_r, scenario_id, model, result):
    writer.writerow([label,
                     format_csv_value("alpha_T", alpha_t) if alpha_t is not None else "",
                     format_csv_value("alpha_R", alpha_r) if alpha_r is not None else "",
                     scenario_id, model,
                     *[format_csv_value(field, result[field]) for field in RESULT_FIELDS],
                     result["status"]])


def run_out_of_sample():
    with open(INSTANCE_PATH, encoding="utf-8") as handle:
        base_instance = json.load(handle)
    exp_dir = os.path.join("results", f"out_of_sample_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(exp_dir, exist_ok=True)

    plans = []
    for alpha_t, alpha_r in PARAM_GRID:
        config = RHMVSPConfig()
        config.alpha_T, config.alpha_R = alpha_t, alpha_r
        config.time_limit, config.solver_name = 120, "gurobi"
        nominal = json.loads(json.dumps(base_instance))
        update_od_deadlines(nominal, config)
        label = f"({format_alpha_T(alpha_t)}, {format_alpha_R(alpha_r)})"
        plans.append((label, alpha_t, alpha_r, config, _solve_nominal(config, nominal, "MR")))

    mw_config = RHMVSPConfig()
    mw_config.alpha_T = mw_config.alpha_R = 0.0
    mw_config.skip_risk_constraint = True
    mw_config.time_limit, mw_config.solver_name = 120, "gurobi"
    nominal_mw = json.loads(json.dumps(base_instance))
    update_od_deadlines(nominal_mw, mw_config)
    mw_solution = _solve_nominal(mw_config, nominal_mw, "MW")

    with open(os.path.join(exp_dir, "nominal_plans.json"), "w", encoding="utf-8") as handle:
        json.dump({"MR": {x[0]: x[4] for x in plans}, "MW": mw_solution}, handle, indent=2)
    with open(os.path.join(exp_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump({"design": "fixed-plan common-scenario out-of-sample evaluation",
                   "n_scenarios": N_SCENARIOS, "scenario_seed": SCENARIO_SEED,
                   "param_grid": PARAM_GRID, "mw_is_common_baseline": True,
                   "instance": INSTANCE_PATH}, handle, indent=2)

    scenarios = []
    rng = np.random.default_rng(SCENARIO_SEED)
    for _ in range(N_SCENARIOS):
        scenarios.append(sample_scenario(base_instance, rng))
    csv_path = os.path.join(exp_dir, "experiment.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_label", "alpha_T", "alpha_R", "scenario_id", "model",
                         "cost", "transport_cost", "transfer_cost", "reposition_cost",
                         "unmet_penalty", "network_risk_reliability",
                         "network_time_reliability", "Network_Risk", "Network_Time",
                         "unmet_demand", "delivery_ratio", "service_failure",
                         "risk_violation", "time_violation", "status"])
        for scenario_id, scenario in enumerate(scenarios):
            mw_result = _evaluate_fixed_plan(mw_solution, scenario, mw_config, 0.0, 0.0)
            _write_row(writer, "MW baseline", None, None, scenario_id, "MW", mw_result)
            for label, alpha_t, alpha_r, config, solution in plans:
                result = _evaluate_fixed_plan(solution, scenario, config, alpha_t, alpha_r)
                _write_row(writer, label, alpha_t, alpha_r, scenario_id, "MR", result)
    print(f"Fixed-plan out-of-sample results saved to {csv_path}")
    return exp_dir


if __name__ == "__main__":
    run_out_of_sample()
