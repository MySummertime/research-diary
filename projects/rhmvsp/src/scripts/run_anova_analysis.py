import os
import sys
import json
import time
import logging
import itertools
import pandas as pd
from datetime import datetime
from copy import deepcopy
import concurrent.futures

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Ensure src is in Python path
sys.path.insert(0, "/Users/frances/Codes/rhmvsp")

from projects.rhmvsp.config import RHMVSPConfig
from projects.rhmvsp.src.solvers.branch_and_price import BranchAndPriceController

# Suppress debug logs from solver
logging.basicConfig(level=logging.WARNING)

def solve_instance(params_dict, base_instance_path):
    """Solve a single instance with given parameters."""
    with open(base_instance_path, "r") as f:
        inst = json.load(f)

    # Update config
    valid_keys = set(RHMVSPConfig.__dataclass_fields__.keys())
    cfg_dict = deepcopy(inst.get("config", {}))
    
    # Apply param updates
    for k, v in params_dict.items():
        if k in valid_keys:
            cfg_dict[k] = v

    # Filter cfg_dict to only valid keys before instantiating
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}

    # Re-initialize config
    config = RHMVSPConfig(**filtered_cfg)
    inst["config"] = {k: getattr(config, k) for k in valid_keys if isinstance(getattr(config, k), (int, float, str, tuple, list))}
    
    # We use a dummy logger to suppress output during parallel execution
    logger = logging.getLogger(f"solver_{id(params_dict)}")
    logger.setLevel(logging.CRITICAL)

    solver = BranchAndPriceController(config, inst, logger)
    try:
        sol = solver.solve()
        total_cost = sol.get("total_cost", float('nan'))
    except Exception as e:
        total_cost = float('nan')

    res = params_dict.copy()
    res["total_cost"] = total_cost
    return res

def run_anova_analysis():
    print("=" * 60)
    print("Starting Full Factorial ANOVA Sensitivity Analysis")
    print("=" * 60)
    
    base_instance_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "small", "instance.json"))
    
    # 6 Levels per parameter
    param_levels = {
        "alpha_max": [0.01, 0.03, 0.05, 0.07, 0.09, 0.10],
        "delta_D": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "c_unmet": [100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0],
        "c_hub": [1000.0, 5000.0, 10000.0, 20000.0, 35000.0, 50000.0]
    }

    # Generate full factorial combinations (6^4 = 1296)
    keys = list(param_levels.keys())
    combinations = list(itertools.product(*(param_levels[k] for k in keys)))
    tasks = [dict(zip(keys, combo)) for combo in combinations]
    
    print(f"[INFO] Generated {len(tasks)} parallel instances to solve.")
    
    start_time = time.time()
    results = []
    
    # Process Pool
    max_workers = min(10, os.cpu_count() or 4)
    print(f"[INFO] Running with {max_workers} parallel workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(solve_instance, task, base_instance_path): task for task in tasks}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                print(f"[ERROR] Exception occurred: {exc}")
            
            if i % 50 == 0 or i == len(tasks):
                print(f"[PROGRESS] Completed {i}/{len(tasks)} runs...")

    elapsed = time.time() - start_time
    print(f"[INFO] Parallel execution finished in {elapsed:.2f} seconds.")
    
    # Data processing
    df = pd.DataFrame(results)
    df.dropna(inplace=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", f"anova_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "raw_data.csv")
    df.to_csv(csv_path, index=False)
    
    # Statistical Modeling: MLR & ANOVA
    print("\n" + "=" * 60)
    print("Fitting Multiple Linear Regression & Running F-Test (ANOVA)")
    print("=" * 60)
    
    # Formula includes all main effects and two-way interactions
    formula = "total_cost ~ (alpha_max + delta_D + c_unmet + c_hub)**2"
    
    model = smf.ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Save ANOVA table
    anova_csv = os.path.join(out_dir, "anova_table.csv")
    anova_table.to_csv(anova_csv)
    
    print("\n--- ANOVA RESULTS ---")
    print(anova_table.to_markdown())
    print("\n[INFO] Summary saved to:", anova_csv)

if __name__ == "__main__":
    run_anova_analysis()
