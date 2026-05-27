"""
K_max Sensitivity Analysis for RHMVSP.

Compares empty-vehicle repositioning limits:
    K_max ∈ [2, 4, 6, 8, 10, ∞]

K_max = maximum O-D pairs a single vehicle can serve (reuse cycles).
K_max = ∞ (None) means unlimited repositioning.

Usage:
    python src/scripts/run_sensitivity_Kmax.py [--size small|medium|large]
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from projects.rhmvsp.config import RHMVSPConfig
from projects.rhmvsp.src.solvers.branch_and_price import BranchAndPriceController
from projects.rhmvsp.src.utils.logging import Logger


def run_kmax_sensitivity(size: str = "small", time_limit: int = 300):
    """Run K_max sensitivity for all K ∈ [2,4,6,8,10,∞] on a given instance size."""
    
    base_config = RHMVSPConfig()
    instance_path = f"data/{size}/instance.json"
    
    if not os.path.exists(instance_path):
        print(f"Instance not found: {instance_path}")
        print("Run 'python main.py --mode generate --size {size}' first.")
        return
    
    with open(instance_path, "r", encoding="utf-8") as f:
        base_instance = json.load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results/sensitivity_Kmax_{timestamp}_{size}"
    os.makedirs(out_dir, exist_ok=True)
    
    logger = Logger(level="INFO", log_file=os.path.join(out_dir, "sensitivity.log"))
    logger.info(f"{'='*60}")
    logger.info(f"K_max Sensitivity Analysis — Instance: {size}")
    logger.info(f"{'='*60}")
    
    # K_max values to test: 2, 4, 6, 8, 10, None (∞)
    kmax_values = [2, 4, 6, 8, 10, None]
    
    results = []
    
    for K in kmax_values:
        label = f"K_max={K}" if K is not None else "K_max=∞"
        logger.info(f"\n{'─'*50}")
        logger.info(f"Running: {label}")
        logger.info(f"{'─'*50}")
        
        # Build config with this K_max
        cfg_dict = {
            k: getattr(base_config, k)
            for k in RHMVSPConfig.__dataclass_fields__.keys()
        }
        cfg_dict["K_max"] = K
        cfg_dict["time_limit"] = time_limit
        config = RHMVSPConfig(**cfg_dict)
        
        inst = deepcopy(base_instance)
        
        solver = BranchAndPriceController(config, inst, logger)
        sol = solver.solve()
        
        metrics = {
            "K_max": K,
            "label": label,
            "status": sol.get("status", "unknown"),
            "total_cost": sol.get("total_cost", 0.0),
            "deploy_cost": sol.get("deploy_cost", 0.0),
            "hub_cost": sol.get("hub_cost", 0.0),
            "transport_cost": sol.get("transport_cost", 0.0),
            "transfer_cost": sol.get("transfer_cost", 0.0),
            "delay_penalty_cost": sol.get("delay_penalty_cost", 0.0),
            "reposition_cost": sol.get("reposition_cost", 0.0),
            "holding_cost": sol.get("holding_cost", 0.0),
            "opportunity_cost": sol.get("opportunity_cost", 0.0),
            "vehicles_used": sol.get("num_vehicles_used", 0),
            "R_sys": sol.get("R_net", -1.0),
            "R_sys_risk": sol.get("R_net_risk", -1.0),
            "R_sys_time": sol.get("R_net_time", -1.0),
        }
        
        logger.info(f"  Total Cost: ¥{metrics['total_cost']:,.2f}")
        logger.info(f"  Reposition Cost: ¥{metrics['reposition_cost']:,.2f}")
        logger.info(f"  Vehicles Used: {metrics['vehicles_used']}")
        logger.info(f"  R_time: {metrics['R_sys_time']:.4f}")
        
        results.append(metrics)
    
    # Save results
    out_path = os.path.join(out_dir, "sensitivity_Kmax_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary table
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"K_max Sensitivity Summary — {size.upper()} Instance")
    lines.append(f"{'='*80}")
    header = f"{'K_max':<10} {'Total Cost':<14} {'Repos Cost':<12} {'Delay Cost':<12} {'Veh Used':<10} {'R_time':<8}"
    sep = f"{'─'*10} {'─'*14} {'─'*12} {'─'*12} {'─'*10} {'─'*8}"
    lines.append(header)
    lines.append(sep)
    for r in results:
        label = r["label"]
        lines.append(
            f"{label:<10} ¥{r['total_cost']:<10,.2f}  ¥{r['reposition_cost']:<8,.2f}  "
            f"¥{r['delay_penalty_cost']:<8,.2f}  {r['vehicles_used']:<8}  {r['R_sys_time']:<.4f}"
        )
    summary = "\n".join(lines)
    logger.info(summary)
    
    # Save summary as CSV
    csv_path = os.path.join(out_dir, "sensitivity_Kmax_table.csv")
    with open(csv_path, "w") as f:
        f.write("K_max,total_cost,reposition_cost,delay_penalty_cost,holding_cost,"
                "opportunity_cost,vehicles_used,R_sys,R_sys_risk,R_sys_time\n")
        for r in results:
            K = r["K_max"] if r["K_max"] is not None else -1
            f.write(f"{K},{r['total_cost']:.2f},{r['reposition_cost']:.2f},"
                    f"{r['delay_penalty_cost']:.2f},{r['holding_cost']:.2f},"
                    f"{r['opportunity_cost']:.2f},{r['vehicles_used']},"
                    f"{r['R_sys']:.6f},{r['R_sys_risk']:.6f},{r['R_sys_time']:.6f}\n")
    
    logger.info(f"\nResults saved to: {out_dir}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K_max Sensitivity Analysis")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--time-limit", type=int, default=300, help="Solver time limit per run (s)")
    args = parser.parse_args()
    run_kmax_sensitivity(size=args.size, time_limit=args.time_limit)
