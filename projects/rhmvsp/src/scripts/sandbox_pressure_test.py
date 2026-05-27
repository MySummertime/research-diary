import os
import json
import numpy as np
import logging
import copy
import sys

# Add project root to sys.path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from projects.rhmvsp.src.core.data_generator import InstanceGenerator
from projects.rhmvsp.config import RHMVSPConfig
from projects.rhmvsp.src.solvers.branch_and_price import BranchAndPriceController

def setup_logger():
    logging.basicConfig(level=logging.WARNING)
    return logging.getLogger("SandboxPressure")

def run_delta_t_pressure_test(base_inst, logger):
    print("\n" + "="*50)
    print("MODULE A: DELTA_T EXTREME PRESSURE TEST")
    print("="*50)
    
    # Sweep values
    delta_T_vals = [0.05, 0.20, 0.40, 0.60]
    buffers = [4.0, 6.0, 8.0, 10.0, 12.0]
    
    for buf in buffers:
        print(f"\n--- Testing L_od Buffer: {buf} hours ---")
        results = []
        for dt in delta_T_vals:
            cfg = RHMVSPConfig()
            cfg.delta_T = dt
            cfg.alpha_T = 0.95  # Extremely strict time reliability
            cfg.bap_early_break = True  # Fast mode
            cfg.bap_max_iterations = 15 # Prevent hanging
            
            inst = copy.deepcopy(base_inst)
            
            # 1. Extreme Time Window Tightening
            for od in inst["od_pairs"]:
                # Use expected travel time or just E_od + buffer
                tight_L = od["E_od"] + buf
                od["L_od"] = tight_L
                
            solver = BranchAndPriceController(cfg, inst, logger)
            sol = solver.solve()
            
            cost = sol.get("total_cost", 0.0)
            unmet = sol.get("unmet_penalty_cost", 0.0)
            r_comp = sol.get("R_net", 1.0)
            r_risk = sol.get("R_net_risk", 1.0)
            r_time = sol.get("R_net_time", 1.0)
            print(f"delta_T: {dt:.2f} | Total Cost: {cost:10.2f} | Unmet: {unmet:10.2f} | R_comp: {r_comp:.4f} | R_risk: {r_risk:.4f} | R_time: {r_time:.4f}")
            results.append(cost)
            
        variance = np.var(results)
        print(f"--> Variance for Buffer {buf}: {variance:.2f}")
        if variance > 0:
            print(f"SUCCESS: Buffer {buf} triggers delta_T sensitivity!")

def run_delta_s_pressure_test(base_inst, logger):
    print("\n" + "="*50)
    print("MODULE B: DELTA_S EXTREME PRESSURE TEST")
    print("="*50)
    
    delta_S_vals = [0.05, 0.20, 0.40, 0.60]
    results = []
    
    for ds in delta_S_vals:
        cfg = RHMVSPConfig()
        cfg.delta_S = ds
        cfg.bap_early_break = True
        
        inst = copy.deepcopy(base_inst)
        
        # 1. Extreme Supply Tightening
        # Calculate total demand per origin
        demand_per_origin = {}
        for od in inst["od_pairs"]:
            orig = od["origin"]
            demand_per_origin[orig] = demand_per_origin.get(orig, 0) + od["demand"]
            
        for orig, total_demand in demand_per_origin.items():
            # Force origin supply to be exactly 101% of demand
            # So ANY fluctuation (even 5%) will cause severe shortfalls
            tight_supply = total_demand * 1.01
            # nodes is a list of dicts. Find the node with id == orig
            for n in inst["network"]["nodes"]:
                if n["id"] == orig:
                    n["supply_capacity"] = tight_supply
                    break
            
        solver = BranchAndPriceController(cfg, inst, logger)
        sol = solver.solve()
        
        cost = sol.get("total_cost", 0.0)
        unmet = sol.get("unmet_penalty_cost", 0.0)
        r_comp = sol.get("R_net", 1.0)
        r_risk = sol.get("R_net_risk", 1.0)
        r_time = sol.get("R_net_time", 1.0)
        print(f"delta_S: {ds:.2f} | Total Cost: {cost:10.2f} | Unmet: {unmet:10.2f} | R_comp: {r_comp:.4f} | R_risk: {r_risk:.4f} | R_time: {r_time:.4f}")
        results.append(cost)

    variance = np.var(results)
    print(f"--> Variance of Total Cost over delta_S: {variance:.2f}")
    if variance > 0:
        print("SUCCESS: We found a configuration where delta_S is highly sensitive!")
    else:
        print("FAILED: delta_S remains insensitive. We may need to tighten further.")

def main():
    logger = setup_logger()
    base_config = RHMVSPConfig()
    
    # Generate a fresh small instance
    print("Generating base small instance...")
    gen = InstanceGenerator(base_config, seed=42)
    inst = gen.generate("small")
    
    run_delta_t_pressure_test(inst, logger)
    # run_delta_s_pressure_test(inst, logger)  # Already verified working

if __name__ == "__main__":
    main()
