import sys, os, json, time
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RHMVSPConfig
from src.solvers.branch_and_price import BranchAndPriceController as BAPSolver
from src.utils.logging import Logger
from src.utils.reporting import format_solution_report
from sensitivity.plot_helpers import set_style, COLOURS

COLOR_TIME = COLOURS["C1"]
COLOR_ITER = COLOURS["C2"]

def _update_od_deadlines(instance, config):
    from src.core.uncertainty import UncertaintyEngine as UE2, TrapezoidalFuzzy as TrapFz
    _temp_engine = UE2(config)
    _arcs = instance["network"]["arcs"]
    _hub_ids = {tn["node_id"] for tn in instance["network"].get("transfer_nodes", [])}
    for od in instance.get("od_pairs", []):
        od["alpha_T_od"] = config.alpha_T
        od_o, od_d = od["origin"], od["destination"]
        roads_out = [a for a in _arcs if a["i"] == od_o and a["mode"] == 1 and a["j"] in _hub_ids]
        roads_in  = [a for a in _arcs if a["j"] == od_d and a["mode"] == 1 and a["i"] in _hub_ids]
        rails     = [a for a in _arcs if a["mode"] == 2]
        if roads_out and roads_in and rails:
            co = min(roads_out, key=lambda a: a["distance"])
            ci = min(roads_in,  key=lambda a: a["distance"])
            rm = [a for a in rails if a["i"] == co["j"] and a["j"] == ci["i"]]
            if rm:
                r1 = _temp_engine.make_transport_time(co["distance"], 1)
                tf = _temp_engine.make_transfer_time()
                ra = _temp_engine.make_transport_time(rm[0]["distance"], 2)
                r2 = _temp_engine.make_transport_time(ci["distance"], 1)
                trap = TrapFz(r1.a+tf.a+ra.a+r2.a, r1.b+tf.b+ra.b+r2.b,
                              r1.c+tf.c+ra.c+r2.c, r1.d+tf.d+ra.d+r2.d)
                det_L = trap.deterministic_equivalent_upper(config.alpha_T)
                od["L_od"] = float(round(od.get("E_od", 0.0) + det_L * 1.10, 1))

def _plot_algorithm_performance(x_labels, time_vals, iter_vals, save_path):
    set_style()
    fig, ax1 = plt.subplots(figsize=(7, 5))

    x_idx = list(range(len(x_labels)))

    ax1.set_xlabel("Instance Scale", fontsize=14)
    ax1.set_ylabel('Run time (s)', fontsize=14, color=COLOR_TIME)
    # Use dual line chart
    line1, = ax1.plot(x_idx, time_vals, color=COLOR_TIME, marker='o', markersize=6,
                       linewidth=1.8, label='Run time', zorder=5)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Iteration', fontsize=14, color=COLOR_ITER)
    line2, = ax2.plot(x_idx, iter_vals, color=COLOR_ITER, marker='s', markersize=6,
                       linewidth=1.8, label='Iteration', zorder=4)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=12)

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')

    ax1.tick_params(axis='x', direction='in', labelsize=11)
    ax1.tick_params(axis='y', direction='in', labelsize=11, colors=COLOR_TIME)
    ax2.tick_params(axis='y', direction='in', labelsize=11, colors=COLOR_ITER)
    ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path + ".png", dpi=600, bbox_inches='tight')
    fig.savefig(save_path + ".tiff", dpi=600, format='tiff', bbox_inches='tight')
    plt.close(fig)

def run_algorithm_performance():
    scales = ['small'] # , 'medium', 'large']
    instances_per_scale = [1, 2]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"perf_scale_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    from src.utils.logging import Tee
    tee = Tee(os.path.join(exp_dir, "experiment.log"))
    sys.stdout = tee

    csv_path = os.path.join(exp_dir, "algorithm_performance.csv")
    cost_cols = ["deploy_cost","hub_cost","transport_cost","transfer_cost",
                 "reposition_cost","prep_cost","idle_cost","unmet_penalty"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Scale", "Instance", "Status", "Time", "Iter", "Cost"]
                    + cost_cols
                    + ["R_risk", "R_time", "Raw_Risk", "Raw_Time"])
        
    results = []
        
    for scale in scales:
        for inst_idx in instances_per_scale:
            instance_path = f"data_performance/{scale}/instance_{inst_idx}.json"
            if not os.path.exists(instance_path):
                print(f"Skipping {instance_path} (File not found. Please generate this data.)")
                continue
                
            with open(instance_path) as f:
                instance = json.load(f)
                
            config = RHMVSPConfig()
            config.solver_name = "gurobi"
            config.solver_backend = "gurobi"
            
            _update_od_deadlines(instance, config)
            logger = Logger(level="ERROR", log_file=None)
            
            t0 = time.time()
            sol = None
            try:
                solver = BAPSolver(config, instance, logger)
                sol = solver.solve()
            except Exception as e:
                print(f"Scale={scale}, Inst={inst_idx} | ERROR: {e}")
            elapsed_bap = time.time() - t0

            bap_ok = sol and sol.get("status") != "infeasible"

            row = [scale, str(inst_idx)]
            if bap_ok:
                # 记录完整 solution json (仿照 main_experiment)
                sol_path = f"data_performance/{scale}/solution_{inst_idx}.json"
                with open(sol_path, "w") as f:
                    json.dump(sol, f, indent=2, default=str)
                print(f"Saved solution instance to {sol_path}")

                # 打印完整格式化报告 (和 main_experiment 相同)
                report = format_solution_report(sol, instance, total_wall=elapsed_bap, total_cpu=0.0)
                print("\n" + "="*60)
                print(f"SOLUTION REPORT for {scale} Instance {inst_idx}")
                print("="*60)
                print(report)
                print("="*60 + "\n")

                cost_bap = sol["total_cost"]
                r_risk_bap = sol.get("R_net_risk", 0)
                r_time_bap = sol.get("R_net_time", 0)
                raw_risk = sol.get("raw_net_risk", 0.0)
                raw_time = sol.get("raw_net_time", 0.0)
                iter_bap = sol.get("iterations", 0)
                row += ["optimal", f"{elapsed_bap:.2f}", str(iter_bap), f"{cost_bap:.0f}"]
                for col in cost_cols:
                    row.append(f"{sol.get(col, 0):.0f}")
                row += [f"{r_risk_bap:.4f}", f"{r_time_bap:.4f}", f"{raw_risk:.2f}", f"{raw_time:.2f}"]
                
                print(f"Scale={scale}, Inst={inst_idx} | Time: {elapsed_bap:.2f}s, Iter: {iter_bap}, Cost: ¥{cost_bap:,.0f}")
                print(f"  Reliability: R_risk={r_risk_bap:.4f}, R_time={r_time_bap:.4f}")
                print(f"  Sub-Costs: Deploy=¥{sol.get('deploy_cost',0):,.0f}, Hub=¥{sol.get('hub_cost',0):,.0f}, Transport=¥{sol.get('transport_cost',0):,.0f}, Transfer=¥{sol.get('transfer_cost',0):,.0f}, Repo=¥{sol.get('reposition_cost',0):,.0f}, Prep=¥{sol.get('prep_cost',0):,.0f}, Idle=¥{sol.get('idle_cost',0):,.0f}, Opp=¥{sol.get('unmet_penalty',0):,.0f}")
                print(f"  Raw Metrics: RawRisk={raw_risk:.2f}, RawTime={raw_time:.2f}h\n")
                
                results.append({
                    "label": f"{scale}-{inst_idx}",
                    "time": elapsed_bap,
                    "iter": iter_bap
                })
            else:
                row += ["infeasible", "", "", ""]
                row += [""] * len(cost_cols)
                row += ["", "", "", ""]
                print(f"Scale={scale}, Inst={inst_idx} | INFEASIBLE")

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    if results:
        x_labels = [r["label"] for r in results]
        time_vals = [r["time"] for r in results]
        iter_vals = [r["iter"] for r in results]
        _plot_algorithm_performance(
            x_labels=x_labels, 
            time_vals=time_vals, 
            iter_vals=iter_vals, 
            save_path=os.path.join(exp_dir, "perf_scale")
        )
        print("Performance plot generated successfully!")

    tee.close()

if __name__ == "__main__":
    run_algorithm_performance()
