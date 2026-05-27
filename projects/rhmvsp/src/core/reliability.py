"""
Reliability Analysis Module (§2).

Implements the three-level risk reliability, three-level time reliability,
and comprehensive reliability framework.

S2-3 fix: Time reliability R_T is now a continuous uncertain measure value
          M{Z~ <= L_od} in [alpha_T, 1], not a binary {0,1} indicator.
S2-5 fix: analyze_solution() correctly computes R_net and is called by the
          Branch-and-Price controller to populate solution output.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .network import Arc, ODPair, TransferNode
from .uncertainty import TrapezoidalFuzzy, TriangularFuzzy, UncertaintyEngine


@dataclass
class PathReliabilityResult:
    """Reliability analysis result for a single path."""

    # Component-level
    arc_reliabilities: dict[tuple[int, int], float] = field(default_factory=dict)
    node_reliabilities: dict[int, float] = field(default_factory=dict)

    # Path-level
    risk_reliability: float = 1.0  # R(P) = min arc/node R (Eq.4)
    time_reliability: float = 1.0  # R_T^od(v) as continuous M{Z~<=L} (Eq.10, S2-3)
    comprehensive: float = 1.0  # R_sys = min{R, R_T} (Eq.5c)

    # Feasibility flags
    arc_risk_feasible: bool = True  # All arcs satisfy alpha_max
    time_window_feasible: bool = True  # Arrival within [E_od, L_od]
    operational_feasible: bool = True  # Rail arcs within operational windows
    transfer_feasible: bool = True  # Transfer timing feasible

    # Arrival time (deterministic equivalent)
    arrival_time_upper: float = 0.0  # Upper bound arrival time
    arrival_time_lower: float = 0.0  # Lower bound arrival time


class ReliabilityAnalyzer:
    """
    Computes reliability measures at all levels (§2.2-2.5).

    Hierarchy:
    - Risk: arc -> path -> O-D -> network
    - Time: arc -> O-D -> network  (continuous uncertain measure, S2-3)
    - Comprehensive: min(risk, time) at path level
    """

    def __init__(self, config, engine: UncertaintyEngine):
        self.config = config
        self.engine = engine

    def check_arc_risk(self, arc: Arc, start_time: float = 0.0, end_time: float = 0.0,
                       time_dependent: bool = True,
                       fuzzy_duration: tuple[float, float, float] | None = None) -> tuple[bool, float]:
        """
        Check arc risk reliability constraint (Eq.9).

        Returns: (feasible, reliability)
        Constraint: alpha_ij^m(t) <= alpha_max
        """
        R = self.engine.arc_reliability(
            arc.alpha, start_time, end_time, time_dependent,
            fuzzy_duration=fuzzy_duration,
            C_consequence=arc.C_consequence,
        )
        feasible = (1.0 - R) <= self.config.alpha_max
        return feasible, R

    def check_node_risk(self, node: TransferNode) -> tuple[bool, float]:
        """
        Check transfer node risk (Eq.2).

        Returns: (feasible, reliability)
        R_i = 1 - gamma_i
        """
        R = 1.0 - node.gamma
        feasible = node.gamma <= self.config.alpha_max
        return feasible, R

    def compute_path_risk(
        self, arcs: list[Arc], transfer_nodes: list[TransferNode],
        arrival_times: list[float] = None, departure_time: float = 0.0
    ) -> tuple[bool, float]:
        """
        Compute path-level risk reliability (Eq.4).

        R(P) = min{min_l R_ij^m, min_l R_i^{m->m'}}
        """
        arr = arrival_times if arrival_times else [departure_time] * len(arcs)
        arc_R = [
            self.engine.arc_reliability(
                a.alpha,
                arr[k - 1] if k > 0 else departure_time,
                arr[k],
                time_dependent=True,
                C_consequence=a.C_consequence,
            )
            for k, a in enumerate(arcs)
        ]
        node_R = [1.0 - tn.gamma for tn in transfer_nodes]

        all_R = arc_R + node_R
        R_P = min(all_R) if all_R else 1.0

        # Check path constraint (Eq.9a, 9b): R(P) >= beta_od
        feasible = self.config.beta_od <= R_P
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

        Returns: (feasible, R_T, arrival_upper, arrival_lower, arrival_times_upper)
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
        transfer_idx = 0

        for k, (arc, mode) in enumerate(zip(arcs, arc_modes)):
            # Transfer handling between modes
            if prev_mode is not None and prev_mode != mode:
                if transfer_idx < len(transfer_nodes):
                    tn = transfer_nodes[transfer_idx]
                    z_a += tn.transfer_time_a
                    z_b += tn.transfer_time_b
                    z_c += tn.transfer_time_c
                    z_d += tn.transfer_time_d
                    transfer_idx += 1
            # Rail waiting window
            if mode == 2:
                arrival_upper = max(arrival_upper, arc.e_window)
                arrival_lower = max(arrival_lower, arc.e_window)
            # Transport time accumulation (#9: trapezoidal for rail)
            arrival_upper += getattr(arc, 'time_d', arc.time_c)
            arrival_lower += arc.time_a
            z_a += arc.time_a
            z_b += arc.time_b
            z_c += arc.time_c
            z_d += getattr(arc, 'time_d', arc.time_c)
            arrival_times_upper.append(arrival_upper)
            prev_mode = mode
        time_feasible = (departure_time + z_c) <= od.L_od
        # The fuzzy trapezoid Z~ must include the departure time offset so that
        # uncertain_measure_leq(L_od) correctly evaluates M{departure + Z_arcs <= L_od}.
        trap_Z = TrapezoidalFuzzy(
            a=departure_time + z_a,
            b=departure_time + z_b,
            c=departure_time + z_c,
            d=departure_time + z_d,
        )
        R_T = trap_Z.uncertain_measure_leq(od.L_od)

        return time_feasible, R_T, arrival_upper, arrival_lower, arrival_times_upper

    def compute_operational_window_feasibility(
        self,
        arcs: list[Arc],
        arc_modes: list[int],
        arrival_times_upper: list[float],
        alpha_T: float,
    ) -> tuple[bool, list[bool]]:
        """
        Check rail arc operational time window constraints (Eq.14a, 14b).
        """
        feasible = True
        flags = []
        for arc, mode, t_arr in zip(arcs, arc_modes, arrival_times_upper):
            if mode == 2:  # Rail
                ok = (t_arr >= arc.e_window) and (t_arr <= arc.l_window)
                if not ok:
                    feasible = False
                flags.append(ok)
            else:
                flags.append(True)  # Road always OK
        return feasible, flags

    def compute_transfer_feasibility(
        self,
        arcs: list[Arc],
        arc_modes: list[int],
        transfer_nodes: list[TransferNode],
        arrival_times: list[float],
        alpha_T: float,
    ) -> tuple[bool, list[bool]]:
        """
        Check road-rail transfer timing constraints (Eq.15a, 15b).
        """
        feasible = True
        flags = []
        prev_mode = None
        transfer_idx = 0

        for k, (arc, mode) in enumerate(zip(arcs, arc_modes)):
            if prev_mode is not None and prev_mode != mode:
                if transfer_idx < len(transfer_nodes):
                    tn = transfer_nodes[transfer_idx]
                    tri = TriangularFuzzy(
                        a=tn.transfer_time_a,
                        b=tn.transfer_time_b,
                        c=tn.transfer_time_c
                    )
                    t_arr = arrival_times[k - 1] if k > 0 else 0

                    if prev_mode == 1 and mode == 2:
                        earliest = t_arr + tri.deterministic_equivalent_lower(alpha_T)
                        latest = t_arr + tri.deterministic_equivalent_upper(alpha_T)
                        if k < len(arcs):
                            rail_arc = arcs[k]
                            ok = earliest >= rail_arc.e_window and latest <= rail_arc.l_window
                        else:
                            ok = True
                    else:
                        ok = True

                    if not ok:
                        feasible = False
                    flags.append(ok)
                    transfer_idx += 1
            prev_mode = mode

        return feasible, flags

    def analyze_path(
        self,
        arcs: list[Arc],
        arc_modes: list[int],
        transfer_nodes: list[TransferNode],
        od: ODPair,
        departure_time: float,
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
        result.time_reliability = R_T
        result.arrival_time_upper = t_upper
        result.arrival_time_lower = t_lower
        result.time_window_feasible = time_ok

        # 1. Arc-level risk reliability (Eq.1, Eq.9) with time-dependent risk (S2-2)
        for k, arc in enumerate(arcs):
            start_t = arr_times[k - 1] if k > 0 else departure_time
            f_dur = (arc.time_a, arc.time_b, arc.time_c, getattr(arc, 'time_d', arc.time_c))
            feasible, R = self.check_arc_risk(arc, start_t, time_dependent=True, fuzzy_duration=f_dur)
            result.arc_reliabilities[(arc.i, arc.j)] = R
            if not feasible:
                result.arc_risk_feasible = False

        # 2. Node-level risk reliability (Eq.2)
        for tn in transfer_nodes:
            feasible, R = self.check_node_risk(tn)
            result.node_reliabilities[tn.node_id] = R
            if not feasible:
                result.arc_risk_feasible = False

        # 3. Path-level risk reliability (Eq.4)
        result.risk_reliability = self.engine.path_reliability(
            list(result.arc_reliabilities.values()),
            list(result.node_reliabilities.values()),
        )

        # Operational window & Transfer feasibility
        op_ok, op_flags = self.compute_operational_window_feasibility(arcs, arc_modes, arr_times, od.alpha_T_od)
        tr_ok, tr_flags = self.compute_transfer_feasibility(arcs, arc_modes, transfer_nodes, arr_times, od.alpha_T_od)
        result.operational_feasible = op_ok
        result.transfer_feasible = tr_ok
        if not (op_ok and tr_ok):
            result.time_window_feasible = False

        # 5. Comprehensive reliability (Eq.5c)
        result.comprehensive = self.engine.comprehensive_reliability(
            result.risk_reliability, result.time_reliability
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

        Returns dict with per-OD and network-level reliability metrics,
        including the true continuous R_net computed from path analysis.
        """
        od_results = {}
        od_reliabilities = []

        for od_idx, od in enumerate(network.od_pairs):
            if od_idx not in vehicle_assignments:
                continue

            assignments = vehicle_assignments[od_idx]
            batch_reliabilities = []

            for assignment in assignments:
                arcs = assignment.get("arcs", [])
                modes = assignment.get("modes", [])
                dep_time = assignment.get("departure_time", 0)
                transfers = assignment.get("transfer_nodes", [])

                result = self.analyze_path(arcs, modes, transfers, od, dep_time)
                batch_reliabilities.append(result.comprehensive)

            # O-D reliability = min over batches (Eq.5a)
            R_od = self.engine.od_reliability(batch_reliabilities)
            od_results[od_idx] = {
                "R_od": R_od,
                "num_batches": len(batch_reliabilities),
                "batch_reliabilities": batch_reliabilities,
            }
            od_reliabilities.append(R_od)

        # Network reliability (Eq.5b): min over all O-D pairs
        R_net = self.engine.network_reliability(od_reliabilities)

        return {
            "od_results": od_results,
            "R_net": R_net,
            "all_od_reliable": all(r >= self.config.beta_net for r in od_reliabilities),
        }
