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
    """Format a human-readable solution report recording exact CPU and wall-clock times."""
    lines = []
    lines.append("=" * 60)
    lines.append("RHMVSP Solution Report")
    lines.append("=" * 60)

    core_wall = solution.get("solve_wall_time", total_wall)
    core_cpu = solution.get("solve_cpu_time", total_cpu)

    if solution.get("status") == "infeasible":
        lines.append("Status: INFEASIBLE")
        lines.append(f"Total CPU time: {total_cpu:.2f}s (Wall: {total_wall:.2f}s)")
        lines.append(f"Core  CPU time: {core_cpu:.2f}s (Wall: {core_wall:.2f}s)")
        return "\n".join(lines)

    lines.append(f"Status: {solution.get('status', 'unknown').upper()}")
    lines.append(f"Total CPU time: {total_cpu:.2f}s (Wall: {total_wall:.2f}s)")
    lines.append(f"Core  CPU time: {core_cpu:.2f}s (Wall: {core_wall:.2f}s)")
    lines.append(f"B&P iterations: {solution.get('iterations', 'N/A')}")
    lines.append(f"Total columns generated: {solution.get('total_columns', 'N/A')}")
    lines.append("")

    lines.append("--- Cost Breakdown ---")
    lines.append(f"  Total cost:          ¥{solution.get('total_cost', 0):>12,.2f}")
    lines.append(f"  Transport cost:      ¥{solution.get('transport_cost', 0):>12,.2f}")
    lines.append(f"  Transfer cost:       ¥{solution.get('transfer_cost', 0):>12,.2f}")
    reposition_total = (solution.get('reposition_cost', 0) + solution.get('prep_cost', 0)
                        + solution.get('idle_cost', 0) + solution.get('repo_extra_penalty', 0))
    lines.append(f"  Reposition cost:     ¥{reposition_total:>12,.2f}")
    lines.append(f"    └─ empty drive:    ¥{solution.get('reposition_cost', 0):>12,.2f}")
    lines.append(f"    └─ preparation:    ¥{solution.get('total_prep_cost', 0):>12,.2f}")
    lines.append(f"    └─ vehicle idle:   ¥{solution.get('idle_cost', 0):>12,.2f}")
    lines.append(f"    └─ repo delay:     ¥{solution.get('repo_extra_penalty', 0):>12,.2f}")
    shortfall = solution.get('unmet_penalty', 0) - solution.get('repo_extra_penalty', 0)
    lines.append(f"  Unmet penalty:       ¥{shortfall:>12,.2f}")
    lines.append("")

    lines.append("--- Resource Budget & Infrastructure ---")

    active_hubs = solution.get('active_hubs', [])
    lines.append(f"  Active transfer hubs: {len(active_hubs)} {active_hubs if active_hubs else ''}")
    lines.append("")

    lines.append("--- Reliability & Fleet ---")
    if 'R_net_risk' in solution:
        lines.append(f"  Network R_risk:      {solution.get('R_net_risk', 0):.6f}")
    if 'R_net_time' in solution:
        lines.append(f"  Network R_time:      {solution.get('R_net_time', 0):.6f}")
    if 'raw_net_risk' in solution:
        lines.append(f"  Network Raw Risk:    {solution.get('raw_net_risk', 0):.2f}")
    if 'raw_net_time' in solution:
        lines.append(f"  Network Raw Time:    {solution.get('raw_net_time', 0):.2f}h")
    lines.append(f"  Max available vehicles: {(solution.get('max_vehicles_west', 0) + solution.get('max_vehicles_east', 0))}")
    lines.append(f"  Actual used vehicles: {solution.get('num_vehicles_used', 0)}")
    lines.append(f"  Dispatched road trips: {solution.get('total_dispatches', 0)}")
    lines.append("")

    # Per-OD reliability details if available
    od_rel = solution.get('od_reliability_details', {})
    if od_rel:
        lines.append("--- O-D Reliability Details ---")
        for od_idx in sorted(od_rel.keys()):
            od_r = od_rel[od_idx]
            od_name = f"OD_{od_idx+1}"
            r_str = f"R_od={od_r.get('R_od', 1.0):.4f}"
            if 'R_od_risk' in od_r and 'R_od_time' in od_r:
                r_str += f" (Risk: {od_r['R_od_risk']:.4f}, Time: {od_r['R_od_time']:.4f})"
                if 'raw_risk' in od_r and 'raw_time' in od_r:
                    r_str += f" [Raw Risk: {od_r['raw_risk']:.2f}, Raw Time: {od_r['raw_time']:.2f}h]"
            lines.append(
                f"  {od_name:<8s} | {r_str} | Batches={od_r.get('num_batches', 0)} | Binding={od_r.get('binding','N/A')}"
            )
        lines.append("")

    details = solution.get("s2_allocation_details", [])
    if details:
        lines.append("--- Dispatch Details ---")
        for d in details:
            lines.append(
                f"  {d['od_pair']:<8s} | Demand: {d['demand']:>6.1f}t | Delivered: {d.get('delivered', 0):>6.1f}t | Shortfall: {d.get('shortfall', 0):>5.1f}t"
            )
        lines.append("")

    # Reposition events (optimal solution)
    repo_events = solution.get("reposition_events", [])
    if repo_events:
        lines.append("--- Reposition Details ---")
        lines.append(f"  Total repositions: {len(repo_events)} | Empty: ¥{solution.get('reposition_cost', 0):,.2f} | Idle: ¥{solution.get('repo_pure_idle', 0):,.2f} | Opp: ¥{solution.get('repo_extra_penalty', 0):,.2f}")
        lines.append(f"  {'Veh':>4s} | {'From':>10s} | {'To':>10s} | {'Dep-Node':>8s} | {'Arr-Node':>8s} | {'Time':>16s} | {'Dist':>8s} | {'Empty':>10s} | {'Idle':>10s} | {'Opp':>10s}")
        lines.append(f"  {'-'*4:>4s} | {'-'*10:>10s} | {'-'*10:>10s} | {'-'*8:>8s} | {'-'*8:>8s} | {'-'*16:>16s} | {'-'*8:>8s} | {'-'*10:>10s} | {'-'*10:>10s} | {'-'*10:>10s}")
        for r in repo_events:
            r_dep = float_to_day_time(r['from_arrival'])
            r_arr = float_to_day_time(r['to_departure'])
            gap = max(0.0, r['to_departure'] - r['from_arrival'])
            lines.append(
                f"  {r['vehicle']:>4d} | {r['from_od']:>10s} | {r['to_od']:>10s} | {str(r['from_node']):>8s} | {str(r['to_node']):>8s} | {r_dep} to {r_arr} ({gap:.1f}h) | {r.get('distance_km', 0):>6.1f}km | ¥{r.get('empty_fee', 0):>9.1f} | ¥{r.get('pure_idle', 0):>9.1f} | ¥{r.get('extra_penalty', 0):>9.1f}"
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
            lines.append(
                f"  {t.get('od_pair', '')} | {t.get('origin', '')} -> {t.get('destination', '')} | {dep} to {arr} | {n_veh} vehicle(s)"
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