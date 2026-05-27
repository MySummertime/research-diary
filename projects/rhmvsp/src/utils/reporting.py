
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
    lines.append(f"  Vehicle deploy cost: ¥{solution.get('deploy_cost', 0):>12,.2f}")
    lines.append(f"  Hub setup cost:      ¥{solution.get('hub_cost', 0):>12,.2f}")
    lines.append(f"  Transport cost:      ¥{solution.get('transport_cost', 0):>12,.2f}")
    lines.append(f"  Transfer cost:       ¥{solution.get('transfer_cost', 0):>12,.2f}")
    lines.append(f"  Reposition cost:     ¥{solution.get('reposition_cost', 0):>12,.2f}")
    lines.append(f"  Inventory holding:   ¥{solution.get('holding_cost', 0):>12,.2f}")
    lines.append(f"  Opportunity cost:    ¥{solution.get('opportunity_cost', 0):>12,.2f}")

    lines.append(f"  Total penalty sum:   ¥{solution.get('penalty_cost', 0):>12,.2f}")
    lines.append("")

    lines.append("--- Resource Budget & Infrastructure ---")

    active_hubs = solution.get('active_hubs', [])
    lines.append(f"  Active transfer hubs: {len(active_hubs)} {active_hubs if active_hubs else ''}")
    lines.append("")

    lines.append("--- Reliability & Fleet ---")
    lines.append(f"  Network R_sys:       {solution.get('R_net', 0):.6f}")
    if 'R_net_risk' in solution:
        lines.append(f"  Network R_risk:      {solution.get('R_net_risk', 0):.6f}")
    if 'R_net_time' in solution:
        lines.append(f"  Network R_time:      {solution.get('R_net_time', 0):.6f}")
    lines.append(f"  Max available vehicles: {solution.get('max_vehicles', 0)}")
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
            lines.append(
                f"  {od_name:<8s} | {r_str} | Batches={od_r.get('num_batches', 0)} | Binding={od_r.get('binding','N/A')}"
            )
        lines.append("")

    details = solution.get("s2_allocation_details", [])
    if details:
        lines.append("--- Supply Inventory & Dispatch Details ---")
        for d in details:
            lines.append(
                f"  {d['od_pair']:<8s} | Demand: {d['demand']:>6.1f}t | Delivered: {d.get('delivered', 0):>6.1f}t"
            )
            lines.append(
                f"           | Shortfall: {d.get('shortfall', 0):>5.1f}t | Opp Cost: ¥{d.get('opp_cost', 0):.1f}"
            )
        lines.append("")

    schedule = solution.get("vehicle_schedule", [])
    repo_events = solution.get("reposition_events", [])
    
    if schedule:
        lines.append("--- Detailed Vehicle Itineraries ---")
        
        veh_tasks = {}
        for t in schedule:
            v = t['vehicle']
            if v not in veh_tasks:
                veh_tasks[v] = []
            veh_tasks[v].append(t)
            
        veh_repos = {}
        for r in repo_events:
            v = r['vehicle']
            if v not in veh_repos:
                veh_repos[v] = []
            veh_repos[v].append(r)
            
        for v in sorted(veh_tasks.keys()):
            tasks = sorted(veh_tasks[v], key=lambda x: x['departure_time'])
            repos = sorted(veh_repos.get(v, []), key=lambda x: x.get('from_arrival', 0))
            
            fleet_str = "Own Fleet" if tasks and tasks[0].get('fleet_type') == 'own' else "Rented Fleet" if tasks and tasks[0].get('fleet_type') == 'rent' else "Rail Wagon"
            lines.append(f"  [Vehicle {v:4d}] ({fleet_str})")
            
            # Print tasks and interleaved repos
            for i, task in enumerate(tasks):
                dep = float_to_day_time(task['departure_time'])
                arr = float_to_day_time(task['arrival_time'])
                lines.append(f"    - Task: {task.get('od_pair', '')} | {task.get('origin', '')} -> {task.get('destination', '')} | {dep} to {arr}")
                
                # Check if there is a repo starting after this task
                for r in repos:
                    if abs(r['from_arrival'] - task['arrival_time']) < 1e-4:
                        r_dep = float_to_day_time(r['from_arrival'])
                        r_arr = float_to_day_time(r['to_departure'])
                        gap = max(0.0, r['to_departure'] - r['from_arrival'])
                        lines.append(f"      * Reposition: {r['from_node']} -> {r['to_node']} | {r_dep} to {r_arr} ({gap:.1f}h) | Cost: ¥{r['cost']:.1f}")
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
