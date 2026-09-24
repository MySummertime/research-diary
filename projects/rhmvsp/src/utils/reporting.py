from src.utils.numeric_format import (
    RELIABILITY_DIGITS,
    RISK_DIGITS,
    TIME_RELIABILITY_DIGITS,
    format_number,
)


def float_to_day_time(time_hours: float) -> str:
    """Convert continuous float hours (0~48) into Day 1/2 24-hour real time format (Day X HH:MM)."""
    day = int(time_hours // 24) + 1
    rem_hours = time_hours % 24
    hours = int(rem_hours)
    minutes = int(round((rem_hours - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    if hours == 24:
        day += 1
        hours = 0
    return f"Day {day} {hours:02d}:{minutes:02d}"


def format_solution_report(solution: dict, instance: dict, total_wall: float, total_cpu: float) -> str:
    """Format a human-readable solution report with benchmark timing fields."""
    def format_metric(key: str, spec: str, default: str = "N/A") -> str:
        value = solution.get(key)
        return default if value is None else format(value, spec)

    def append_performance_metrics(lines: list[str]) -> None:
        lines.append(f"Algorithm wall time: {format_metric('solve_wall_time', '.6f')}s")
        lines.append(f"Master time:         {format_metric('master_wall_time', '.6f')}s")
        lines.append(f"Pricing time:        {format_metric('pricing_wall_time', '.6f')}s")
        lines.append(f"Other time:          {format_metric('other_wall_time', '.6f')}s")
        lines.append(f"Processed nodes:     {solution.get('processed_node_count', 'N/A')}")
        lines.append(f"Root LP value:       {format_metric('root_lp_bound_bp', '.6f')}")
        lines.append(f"Root pricing status: {solution.get('root_pricing_termination', 'N/A')}")
        lines.append(f"Root bound usable:   {solution.get('root_bound_available', False)}")

    lines = []
    lines.append("=" * 60)
    lines.append("RHMVSP Solution Report")
    lines.append("=" * 60)

    if solution.get("status") == "infeasible":
        lines.append("Status: INFEASIBLE")
        lines.append(f"Total CPU time: {total_cpu:.2f}s (Wall: {total_wall:.2f}s)")
        append_performance_metrics(lines)
        return "\n".join(lines)

    lines.append(f"Status: {solution.get('status', 'unknown').upper()}")
    lines.append(f"Total CPU time: {total_cpu:.2f}s (Wall: {total_wall:.2f}s)")
    append_performance_metrics(lines)
    lines.append(f"B&P iterations: {solution.get('iterations', 'N/A')}")
    lines.append(f"Total columns generated: {solution.get('total_columns', 'N/A')}")
    lines.append("")

    lines.append("--- Cost Breakdown ---")
    lines.append(f"  Total cost:          ¥{solution.get('total_cost', 0):>12,.2f}")
    lines.append(f"  Transport cost:      ¥{solution.get('transport_cost', 0):>12,.2f}")
    lines.append(f"  Transfer cost:       ¥{solution.get('transfer_cost', 0):>12,.2f}")
    lines.append(f"  Reposition cost:     ¥{solution.get('reposition_cost', 0):>12,.2f}")
    lines.append(f"  Unmet penalty:       ¥{solution.get('unmet_penalty', 0):>12,.2f}")
    if "unmet_penalty_per_feu" in solution:
        lines.append(f"  Penalty rate:        ¥{solution['unmet_penalty_per_feu']:>12,.2f}/FEU")
    lines.append("")

    lines.append("--- Resource Budget & Infrastructure ---")

    active_hubs = solution.get('active_hubs', [])
    lines.append(f"  Active transfer hubs: {len(active_hubs)} {active_hubs if active_hubs else ''}")
    lines.append("")

    lines.append("--- Reliability & Fleet ---")
    if 'network_risk_reliability' in solution:
        lines.append(
            "  Network risk reliability: "
            f"{format_number(solution['network_risk_reliability'], RELIABILITY_DIGITS)}"
        )
    if 'network_time_reliability' in solution:
        lines.append(
            "  Network time reliability: "
            f"{format_number(solution['network_time_reliability'], TIME_RELIABILITY_DIGITS)}"
        )
    if 'network_risk' in solution:
        lines.append(
            f"  Network risk: {format_number(solution['network_risk'], RISK_DIGITS)}"
        )
    if 'network_time' in solution:
        lines.append(f"  Network time: {format_metric('network_time', '.2f', '0.00')} h")
    lines.append(f"  Max available vehicles: {solution.get('max_vehicles', 0)}")
    lines.append(f"  Actual used vehicles: {solution.get('num_vehicles_used', 0)}")
    lines.append(f"  Dispatched road trips: {solution.get('total_dispatches', 0)}")
    lines.append("")

    # Per-OD reliability details if available
    od_rel = solution.get('od_reliability_details', {})
    if od_rel:
        lines.append("--- O-D Reliability Details ---")
        for od_idx in sorted(od_rel.keys(), key=int):
            od_r = od_rel[od_idx]
            od_name = f"OD_{int(od_idx)+1}"
            r_str = (
                "Joint reliability="
                f"{format_number(od_r.get('od_joint_reliability', 1.0), TIME_RELIABILITY_DIGITS)}"
            )
            if 'od_risk_reliability' in od_r and 'od_time_reliability' in od_r:
                r_str += (
                    " (Risk reliability: "
                    f"{format_number(od_r['od_risk_reliability'], RELIABILITY_DIGITS)}, "
                    f"Time reliability: {format_number(od_r['od_time_reliability'], TIME_RELIABILITY_DIGITS)})"
                )
                if 'od_risk' in od_r and 'od_time' in od_r:
                    r_str += (
                        f" [Risk: {format_number(od_r['od_risk'], RISK_DIGITS)}, "
                        f"Time: {od_r['od_time']:.2f} h]"
                    )
            lines.append(
                f"  {od_name:<8s} | {r_str} | Batches={od_r.get('num_batches', 0)} | Binding={od_r.get('binding','N/A')} | Deliver={od_r.get('delivery_ratio', 1.0):.0%}"
            )
        lines.append("")

    details = solution.get("s2_allocation_details", [])
    if details:
        lines.append("--- Dispatch Details ---")
        for d in details:
            lines.append(
                f"  {d['od_pair']:<8s} | Demand: {d['demand']:>6.1f} FEU | Delivered: {d.get('delivered', 0):>6.1f} FEU | Shortfall: {d.get('shortfall', 0):>5.1f} FEU"
            )
        lines.append("")

    # Reposition events (optimal solution)
    repo_events = solution.get("reposition_events", [])
    if repo_events:
        lines.append("--- Reposition Details ---")
        lines.append(f"  Total repositions: {len(repo_events)} | Cost: ¥{solution.get('reposition_cost', 0):,.2f}")
        lines.append(f"  {'Veh':>4s} | {'From':>10s} | {'To':>10s} | {'Dep-Node':>8s} | {'Arr-Node':>8s} | {'Time':>16s} | {'Dist':>8s} | {'Cost':>10s}")
        for r in repo_events:
            r_dep = float_to_day_time(r['from_arrival'])
            r_arr = float_to_day_time(r['to_departure'])
            gap = max(0.0, r['to_departure'] - r['from_arrival'])
            lines.append(
                f"  {r['vehicle']:>4d} | {r['from_od']:>10s} | {r['to_od']:>10s} | {str(r['from_node']):>8s} | {str(r['to_node']):>8s} | {r_dep} to {r_arr} ({gap:.1f}h) | {r.get('distance_km', 0):>6.1f}km | ¥{r.get('empty_fee', 0):>9.1f}"
            )
        lines.append("")

    schedule = solution.get("transport_routes", [])
    if schedule:
        lines.append("--- Active Transport Routes ---")
        total_vehicles = 0
        for t in schedule:
            dep = float_to_day_time(t['departure_time'])
            arr = float_to_day_time(t['arrival_time'])
            n_veh = t.get('num_vehicles', 1)
            total_vehicles += n_veh
            # Build full node path from arcs: [(i,j,mode), ...]
            arcs = t.get('arcs', [])
            if arcs:
                node_path = [str(arcs[0]['i'])]
                for a in arcs:
                    node_path.append(str(a['j']))
                path_str = "->".join(node_path)
            else:
                path_str = f"{t.get('origin', '')} -> {t.get('destination', '')}"
            lines.append(
                f"  {t.get('od_pair', '')} | {path_str} | {dep} to {arr} | {n_veh} vehicle(s)"
            )
        # Use num_vehicles_used (unique vehicles after stitching) instead of summing route segments
        unique_vehicles = solution.get('num_vehicles_used', total_vehicles)
        lines.append(f"  {'':>4s}   Unique vehicles used: {unique_vehicles} (route segments: {total_vehicles})")
        lines.append("")

    rail_summary = solution.get("rail_consolidation_summary", [])
    if rail_summary:
        lines.append("")
        lines.append("--- Rail Wagon Consolidation Summary ---")
        for rs in rail_summary:
            dep_fmt = float_to_day_time(rs['departure_time'])
            lines.append(
                f"  {rs['rail_section']:<22s} | Dep: {dep_fmt:<14s} | "
                f"Trucks: {rs['truck_dispatches']:2d} -> Wagons: {rs['rail_wagons']:2d} "
                f"(Load: {rs['load_factor_percent']:>5.1f}%) | Included: {', '.join(rs['tasks_included'])}"
            )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
