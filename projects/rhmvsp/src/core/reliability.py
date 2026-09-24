"""
Reliability Analysis Module (§2).

Implements the four-level risk, time, risk-reliability, and time-reliability
framework: operation, plan, OD, and network.

S2-3 fix: Time reliability R_T is now a continuous uncertain measure value
          M{Z~ <= time_window_end} in [alpha_T, 1], not a binary {0,1} indicator.
S2-5 fix: analyze_solution() correctly computes R_net and is called by the
          Branch-and-Price controller to populate solution output.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .network import Arc, ODPair, TransferNode
from .uncertainty import TrapezoidalFuzzy, ZigzagUncertainVariable, UncertaintyEngine


@dataclass
class PathReliabilityResult:
    """Four-metric analysis result for one dispatched transport plan."""

    # Operation level: transport arcs and transfer operations.
    operation_risks: dict[str, float] = field(default_factory=dict)
    operation_times: dict[str, float] = field(default_factory=dict)
    operation_risk_reliabilities: dict[str, float] = field(default_factory=dict)

    # Plan level.
    plan_risk: float = 0.0
    plan_time: float = 0.0
    plan_risk_reliability: float = 1.0
    plan_time_reliability: float = 1.0
    plan_arrival_time: float = 0.0

    # Feasibility flags
    arc_risk_feasible: bool = True
    time_window_feasible: bool = True  # Arrival within [time_window_start, time_window_end]

    # Arrival time (deterministic equivalent)
    arrival_time_upper: float = 0.0  # Upper bound arrival time
    arrival_time_lower: float = 0.0  # Lower bound arrival time

class ReliabilityAnalyzer:
    """
    Computes reliability measures at all levels (§2.2-2.5).

    All four metrics use the hierarchy operation -> plan -> OD -> network.
    Risk and time are reported with B and T semantics; R-prefixed quantities
    are reserved for reliability measures.
    """

    def __init__(self, config, engine: UncertaintyEngine):
        self.config = config
        self.engine = engine
        # Cache for dynamically computed transport times per arc
        # key: (arc_id, config_signature) -> TrapezoidalFuzzy
        self._arc_time_cache: dict[tuple, 'TrapezoidalFuzzy'] = {}

    def check_arc_risk(self, arc: Arc, start_time: float = 0.0, end_time: float = 0.0,
                       time_dependent: bool = True,
                       fuzzy_duration: tuple[float, float, float] | None = None) -> tuple[bool, float]:
        """
        Evaluate one arc's risk reliability.

        Returns: (feasible, reliability)
        Each component is checked against the common series-system threshold.
        """
        R = self.engine.arc_reliability(
            arc.alpha, start_time, end_time, time_dependent,
            fuzzy_duration=fuzzy_duration,
        )
        return self.config.skip_risk_constraint or R >= self.config.alpha_R, R

    def check_node_risk(
        self, node: TransferNode, previous_mode: int, next_mode: int
    ) -> tuple[bool, float]:
        """
        Check transfer node risk (Eq.2).

        Returns: (feasible, reliability)
        R_i = 1 - gamma_i
        """
        belief = node.gamma_rr if previous_mode == next_mode == 2 else node.gamma_ir
        R = 1.0 - belief
        return self.config.skip_risk_constraint or R >= self.config.alpha_R, R

    def compute_path_risk(
        self, arcs: list[Arc], transfer_nodes: list[TransferNode],
        arrival_times: list[float] = None, departure_time: float = 0.0
    ) -> tuple[bool, float]:
        """
        Compute plan risk reliability from the minimum component reliability.
        """
        arr = arrival_times if arrival_times else [departure_time] * len(arcs)
        arc_R = [
            self.engine.arc_reliability(
                a.alpha,
                arr[k - 1] if k > 0 else departure_time,
                arr[k],
                time_dependent=True,
            )
            for k, a in enumerate(arcs)
        ]
        transfer_modes = [
            (a1.mode, a2.mode)
            for a1, a2 in zip(arcs, arcs[1:])
            if not (a1.mode == 1 and a2.mode == 1)
        ]
        node_R = [
            self.check_node_risk(tn, previous_mode, next_mode)[1]
            for tn, (previous_mode, next_mode) in zip(transfer_nodes, transfer_modes)
        ]

        R_P = self.engine.path_reliability(arc_R, node_R)

        # Check path risk-reliability threshold: R(P) >= alpha_R
        feasible = R_P >= self.config.alpha_R
        return feasible, R_P

    def compute_time_reliability_od(
        self,
        departure_time: float,
        arcs: list[Arc],
        transfer_nodes: list[TransferNode],
        od: ODPair,
        arc_modes: list[int],
    ) -> tuple[bool, float, float, float, list[float]]:
        """
        Compute O-D level time reliability (§2.3.2, Eq.10). S2-3 fix.

        Uses engine.make_transport_time() for dynamic fuzzy time parameters
        so that changes to the configured transport-time ratios take effect.

        Returns the plan feasibility, plan time reliability, time bounds,
        and traversal arrival bounds. Time reliability is defined only for
        completion of the complete plan by the common planning horizon.
        """
        alpha_T = od.alpha_T_od

        # Initialize accumulators for fuzzy time parameters
        z_a = 0.0
        z_b = 0.0
        z_c = 0.0
        z_d = 0.0
        arrival_upper = 0.0
        arrival_lower = 0.0
        arrival_times_upper: list[float] = []
        prev_mode: int | None = None
        continuous_drive = 0.0
        transfer_idx = 0
        for k, (arc, mode) in enumerate(zip(arcs, arc_modes)):
            # Transfer handling between modes
            if prev_mode is not None and not (prev_mode == 1 and mode == 1):
                if transfer_idx < len(transfer_nodes):
                    tn = transfer_nodes[transfer_idx]
                    transfer_time = self.config.transfer_time_hours(prev_mode, mode)
                    z_a += transfer_time
                    z_b += transfer_time
                    z_c += transfer_time
                    z_d += transfer_time
                    arrival_upper += transfer_time
                    arrival_lower += transfer_time
                    transfer_idx += 1
            # Transport time: use engine for dynamic fuzzy parameters (#9)
            # This ensures configured transport-time ratios are reflected in
            # the evaluation, not just in pricing.
            trap_time = self.engine.make_transport_time(arc.distance, mode)
            if mode == 1:
                if prev_mode != 1:
                    continuous_drive = 0.0
                deterministic_drive = trap_time.deterministic_equivalent_upper(alpha_T)
                rest_time, continuous_drive = self.engine.road_drive_transition(
                    continuous_drive, deterministic_drive
                )
            else:
                rest_time = 0.0
                continuous_drive = 0.0
            arrival_upper += trap_time.c
            arrival_lower += trap_time.a
            z_a += trap_time.a
            z_b += trap_time.b
            z_c += trap_time.b  # Triangular mode contributes to both plateau endpoints.
            z_d += trap_time.c
            arrival_upper += rest_time
            arrival_lower += rest_time
            z_a += rest_time
            z_b += rest_time
            z_c += rest_time
            z_d += rest_time
            arrival_times_upper.append(arrival_upper)
            prev_mode = mode

        # Feasibility: deterministic equivalent arrival <= time_window_end
        if alpha_T <= 0.5:
            # Left branch: x = z_a + 2*alpha*(z_b - z_a)
            det_equiv_coeff = z_a + 2.0 * alpha_T * (z_b - z_a)
        else:
            # Right branch (alpha_T > 0.5): det_equiv = (2*alpha_T-1)*z_d + 2*(1-alpha_T)*z_c
            det_equiv_coeff = (2.0 * alpha_T - 1.0) * z_d + 2.0 * (1.0 - alpha_T) * z_c
        det_upper_arrival = departure_time + det_equiv_coeff
        deadline = min(float(od.time_window_end), float(self.config.T_max))
        time_feasible = det_upper_arrival <= deadline

        # The fuzzy trapezoid Z~ must include the departure time offset so that
        # uncertain_measure_leq(time_window_end) correctly evaluates M{departure + Z_arcs <= time_window_end}.
        trap_Z = TrapezoidalFuzzy(
            a=departure_time + z_a,
            b=departure_time + z_b,
            c=departure_time + z_c,
            d=departure_time + z_d,
        )
        R_T = trap_Z.uncertain_measure_leq(deadline)

        return time_feasible, R_T, arrival_upper, arrival_lower, arrival_times_upper

    def analyze_path(
        self,
        arcs: list[Arc],
        arc_modes: list[int],
        transfer_nodes: list[TransferNode],
        od: ODPair,
        departure_time: float,
        arrival_time: float | None = None,
        plan_time_reliability: float | None = None,
        rail_services: list[dict] | tuple[dict, ...] | None = None,
    ) -> PathReliabilityResult:
        """
        Full reliability analysis for a single path (§2.2-2.5).

        Returns comprehensive PathReliabilityResult with continuous R_T (S2-3).
        """
        result = PathReliabilityResult()

        # 4. Time reliability & arrival sequence (Eq.10-15) with continuous R_T
        time_ok, R_T, t_upper, t_lower, arr_times = self.compute_time_reliability_od(
            departure_time, arcs, transfer_nodes, od, arc_modes
        )
        result.plan_time_reliability = R_T if plan_time_reliability is None else plan_time_reliability
        result.plan_arrival_time = departure_time + t_upper if arrival_time is None else arrival_time
        result.arrival_time_upper = t_upper
        result.arrival_time_lower = t_lower
        result.time_window_feasible = time_ok

        # 1. Arc-level risk reliability (Eq.1, Eq.9) with time-dependent risk (S2-2)
        risk_clock = departure_time
        previous_mode = 0
        continuous_drive = 0.0
        rail_service_idx = 0
        for k, (arc, mode) in enumerate(zip(arcs, arc_modes)):
            if k > 0 and not (previous_mode == 1 and mode == 1):
                matching_transfer = next(
                    (item for item in transfer_nodes if item.node_id == arc.i),
                    None,
                )
                if matching_transfer is not None:
                    risk_clock += self.config.transfer_time_hours(previous_mode, mode)

            arc_rest_time = 0.0
            if mode == 2 and rail_services and rail_service_idx < len(rail_services):
                continuous_drive = 0.0
                service = rail_services[rail_service_idx]
                rail_service_idx += 1
                start_t = float(service["departure_time"])
                risk_clock = float(service["arrival_time"])
            else:
                start_t = risk_clock
                duration = self.engine.make_transport_time(arc.distance, mode)
                duration_upper = duration.deterministic_equivalent_upper(od.alpha_T_od)
                risk_clock += duration_upper
                if mode == 1:
                    if previous_mode != 1:
                        continuous_drive = 0.0
                    arc_rest_time, continuous_drive = self.engine.road_drive_transition(
                        continuous_drive, duration_upper
                    )
                    risk_clock += arc_rest_time
                else:
                    continuous_drive = 0.0
            # Use engine-computed fuzzy duration for consistency with time reliability
            trap_dur = self.engine.make_transport_time(arc.distance, mode)
            f_dur = (trap_dur.a, trap_dur.b, trap_dur.c)
            feasible, R = self.check_arc_risk(arc, start_t, time_dependent=True, fuzzy_duration=f_dur)
            operation_key = f"transport:{k}:{arc.i}->{arc.j}:{mode}"
            result.operation_risk_reliabilities[operation_key] = R
            result.operation_risks[operation_key] = 1.0 - R
            result.operation_times[operation_key] = trap_dur.c + arc_rest_time
            previous_mode = mode

        # 2. Node-level risk reliability (Eq.2)
        transfer_modes = [
            (a1.mode, a2.mode)
            for a1, a2 in zip(arcs, arcs[1:])
            if not (a1.mode == 1 and a2.mode == 1)
        ]
        for transfer_idx, (tn, (previous_mode, next_mode)) in enumerate(zip(transfer_nodes, transfer_modes)):
            feasible, R = self.check_node_risk(tn, previous_mode, next_mode)
            operation_key = f"transfer:{transfer_idx}:node:{tn.node_id}:{previous_mode}->{next_mode}"
            result.operation_risk_reliabilities[operation_key] = R
            result.operation_risks[operation_key] = 1.0 - R
            result.operation_times[operation_key] = self.config.transfer_time_hours(previous_mode, next_mode)

        # Report complete scheduled elapsed time, including timetable waits.
        result.plan_time = result.plan_arrival_time - departure_time

        # 3. Path-level risk reliability (Eq.4)
        result.plan_risk_reliability = self.engine.path_reliability(
            list(result.operation_risk_reliabilities.values()),
            [],
        )
        result.plan_risk = 1.0 - result.plan_risk_reliability
        result.arc_risk_feasible = (
            self.config.skip_risk_constraint
            or result.plan_risk_reliability >= self.config.alpha_R
        )

        return result

    def verify_vehicle_reuse_schedule(self, vehicle_schedule: list[dict], network) -> dict:
        """
        Verify empty vehicle repositioning time window constraints across serial tasks (§3.3.2 iv, Eq.15).
        """
        v_groups = {}
        for task in vehicle_schedule:
            v_id = task.get("vehicle")
            if v_id not in v_groups:
                v_groups[v_id] = []
            v_groups[v_id].append(task)

        feasible = True
        violations = []

        for v_id, tasks in v_groups.items():
            if len(tasks) <= 1:
                continue
            tasks.sort(key=lambda t: t.get("departure_time", 0.0))
            for i in range(len(tasks) - 1):
                t1 = tasks[i]
                t2 = tasks[i + 1]
                arr1 = t1.get("arrival_time", 0.0)
                dep2 = t2.get("departure_time", 0.0)
                dest1 = t1.get("destination")
                orig2 = t2.get("origin")

                if dest1 == orig2 and dest1 is not None:
                    req_gap = 0.0
                else:
                    dist = network.get_exact_road_distance(dest1, orig2) if dest1 is not None and orig2 is not None else 50.0
                    base_time = dist / self.config.road_speed
                    rep_trap = TrapezoidalFuzzy(base_time * 0.8, base_time, base_time * 1.2, base_time * 1.5)
                    req_gap = rep_trap.deterministic_equivalent_upper(self.config.alpha_T)

                if arr1 + req_gap > dep2:
                    feasible = False
                    violations.append({
                        "vehicle": v_id,
                        "task1_od": t1.get("od_pair"),
                        "task1_arr": arr1,
                        "task2_od": t2.get("od_pair"),
                        "task2_dep": dep2,
                        "req_reposition": req_gap,
                        "actual_gap": dep2 - arr1
                    })

        return {"feasible": feasible, "violations": violations}

    def analyze_solution(
        self,
        vehicle_assignments: dict,
        network,
    ) -> dict:
        """
        Analyze reliability of a complete solution. S2-5 fix.

        Returns dict with per-OD and network-level risk and time reliability
        metrics computed independently from path analysis.
        """
        od_results = {}
        od_risks = []
        od_times = []

        for od_idx, od in enumerate(network.od_pairs):
            if od_idx not in vehicle_assignments:
                continue

            assignments = vehicle_assignments[od_idx]
            plan_risk_reliabilities = []
            plan_time_reliabilities = []
            plan_risks = []
            plan_times = []
            plan_arrivals = []

            for assignment in assignments:
                arcs = assignment.get("arcs", [])
                modes = assignment.get("modes", [])
                dep_time = assignment.get("departure_time", 0)
                transfers = assignment.get("transfer_nodes", [])

                result = self.analyze_path(
                    arcs, modes, transfers, od, dep_time,
                    rail_services=assignment.get("rail_services"),
                )
                plan_risk_reliabilities.append(result.plan_risk_reliability)
                plan_time_reliabilities.append(result.plan_time_reliability)
                plan_risks.append(result.plan_risk)
                plan_times.append(result.plan_time)
                plan_arrivals.append(result.plan_arrival_time)

            # O-D risk = min over batches (serial, Eq.5a)
            od_risk_reliability = min(plan_risk_reliabilities) if plan_risk_reliabilities else 1.0
            # O-D time = min over batches (serial, Eq.5a)
            od_time_reliability = min(plan_time_reliabilities) if plan_time_reliabilities else 1.0

            od_results[od_idx] = {
                "od_risk_reliability": od_risk_reliability,
                "od_time_reliability": od_time_reliability,
                "od_risk": max(plan_risks) if plan_risks else 0.0,
                "od_time": max(plan_arrivals) if plan_arrivals else 0.0,
                "num_plans": len(plan_risk_reliabilities),
                "plan_risks": plan_risks,
                "plan_times": plan_times,
                "plan_risk_reliabilities": plan_risk_reliabilities,
                "plan_time_reliabilities": plan_time_reliabilities,
            }
            od_risks.append(od_risk_reliability)
            od_times.append(od_time_reliability)

        network_risk_reliability = min(od_risks) if od_risks else 1.0
        network_time_reliability = min(od_times) if od_times else 1.0

        network_risk = 1.0 - network_risk_reliability
        network_time = max((od_r["od_time"] for od_r in od_results.values()), default=0.0)

        return {
            "od_results": od_results,
            "network_risk_reliability": network_risk_reliability,
            "network_time_reliability": network_time_reliability,
            "network_risk": network_risk,
            "network_time": network_time,
        }
