import os
import sys
import json
import random
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from projects.rhmvsp.config import RHMVSPConfig, PlotProperties, ColorPalette
from projects.rhmvsp.src.solvers.branch_and_price import BranchAndPriceController
from projects.rhmvsp.src.utils.logging import Logger

class StatisticalAnalyzer:
    def __init__(self, num_samples: int, instance_path: str, output_dir: str):
        self.num_samples = num_samples
        self.instance_path = instance_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = Logger(level="INFO", log_file=os.path.join(self.output_dir, "statistical_tests.log"))
        
        with open(instance_path, "r", encoding="utf-8") as f:
            self.base_instance = json.load(f)
            
        self.base_config = RHMVSPConfig()
        
        # Ranges for LHS/Random sampling
        self.param_ranges = {
            "alpha_max": (0.01, 0.10),
            "alpha_T": (0.6, 0.95),
            "delta_D": (0.0, 1.0),
            "c_unmet": (100.0, 10000.0),
            "c_hub": (1000.0, 50000.0)
        }
        
    def generate_samples(self):
        """Generate random samples within the specified ranges."""
        self.logger.info(f"Generating {self.num_samples} random parameter samples...")
        samples = []
        for _ in range(self.num_samples):
            sample = {}
            for param, (low, high) in self.param_ranges.items():
                sample[param] = random.uniform(low, high)
            samples.append(sample)
        return samples
        
    def run_model(self, param_dict, sample_idx):
        """Runs the model for a single parameter configuration."""
        self.logger.info("-" * 40)
        self.logger.info(f"Running Sample {sample_idx + 1}/{self.num_samples}")
        
        valid_keys = set(RHMVSPConfig.__dataclass_fields__.keys())
        cfg_dict = {k: getattr(self.base_config, k) for k in valid_keys}
        for k, v in param_dict.items():
            if k in cfg_dict:
                cfg_dict[k] = v
                
        config = RHMVSPConfig(**cfg_dict)
        
        inst = deepcopy(self.base_instance)
        inst["config"] = {k: getattr(config, k) for k in valid_keys if isinstance(getattr(config, k), (int, float, str, tuple, list))}

            
        solver = BranchAndPriceController(config, inst, self.logger)
        try:
            sol = solver.solve()
            
            # Extract time: earliest departure to latest arrival
            total_time = 0.0
            if sol.get("status") == "optimal" and "scheduled_tasks" in sol:
                tasks = sol["scheduled_tasks"]
                if tasks:
                    earliest_dep = min(t.get("departure_time", float('inf')) for t in tasks)
                    latest_arr = max(t.get("arrival_time", 0.0) for t in tasks)
                    if earliest_dep < float('inf'):
                        total_time = latest_arr - earliest_dep
            
            # Fallback if scheduled_tasks is not structured as expected
            if total_time == 0.0 and "transport_cost" in sol:
                # If we cannot compute time directly, use transport_cost proxy or just 0
                pass
                
            metrics = {
                "total_cost": sol.get("total_cost", np.nan),
                "total_time": total_time,
                "R_sys": sol.get("R_net", sol.get("R_sys", np.nan)) # Fallbacks for reliability
            }
        except Exception as e:
            self.logger.error(f"Error solving sample {sample_idx}: {e}")
            metrics = {"total_cost": np.nan, "total_time": np.nan, "R_sys": np.nan}
            
        return metrics

    def run_all(self):
        samples = self.generate_samples()
        results = []
        
        for idx, sample in enumerate(samples):
            metrics = self.run_model(sample, idx)
            combined = {**sample, **metrics}
            results.append(combined)
            
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, "raw_samples_results.csv"), index=False)
        
        # Filter out failed runs
        df = df.dropna()
        if len(df) == 0:
            self.logger.error("All solver runs failed. Cannot perform statistical tests.")
            return
            
        self.perform_statistical_tests(df)
        
    def perform_statistical_tests(self, df):
        self.logger.info("=" * 50)
        self.logger.info("Performing Statistical Analysis")
        self.logger.info("=" * 50)
        
        params = list(self.param_ranges.keys())
        outputs = ["total_cost", "total_time", "R_sys"]
        
        # Check if output has variation
        valid_outputs = []
        for out in outputs:
            if df[out].std() > 1e-6:
                valid_outputs.append(out)
            else:
                self.logger.warning(f"Output '{out}' has zero variance across samples. It will be excluded from regression.")
        
        # 1. Correlation Analysis
        corr_pearson = df[params + valid_outputs].corr(method='pearson')
        corr_spearman = df[params + valid_outputs].corr(method='spearman')
        
        # Save Correlation Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_pearson.loc[params, valid_outputs], annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("Pearson Correlation: Parameters vs Outputs")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"), dpi=PlotProperties.DPI)
        plt.close()
        
        # 2. Regression (F-test and T-test)
        X = df[params]
        X = sm.add_constant(X)
        
        stats_results = []
        param_pvalues = {p: [] for p in params}
        
        for out in valid_outputs:
            y = df[out]
            model = sm.OLS(y, X).fit()
            
            f_pval = model.f_pvalue
            rsquared = model.rsquared
            
            stats_results.append({
                "Output": out,
                "Model_R2": rsquared,
                "Model_F_pvalue": f_pval,
                "Overall_Significant": "Yes" if f_pval < 0.05 else "No"
            })
            
            self.logger.info(f"\n--- Regression for {out} ---")
            self.logger.info(f"F-test p-value: {f_pval:.4f} (R^2: {rsquared:.4f})")
            
            for p in params:
                p_val = model.pvalues[p]
                param_pvalues[p].append(p_val)
                significance = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
                self.logger.info(f"  {p}: t-test p-value = {p_val:.4f} {significance}")

        stats_df = pd.DataFrame(stats_results)
        stats_df.to_csv(os.path.join(self.output_dir, "model_ftest_summary.csv"), index=False)
        
        # 3. Identify Useless Parameters
        # A parameter is useless if its t-test p-value is > 0.05 across ALL valid models.
        param_summary = []
        for p in params:
            p_vals = param_pvalues[p]
            min_pval = min(p_vals) if p_vals else 1.0
            is_useless = min_pval > 0.05
            
            row = {"Parameter": p, "Min_PValue": min_pval, "Is_Useless": "YES" if is_useless else "NO"}
            for i, out in enumerate(valid_outputs):
                row[f"PValue_{out}"] = p_vals[i]
            param_summary.append(row)
            
        summary_df = pd.DataFrame(param_summary)
        summary_df.to_csv(os.path.join(self.output_dir, "parameter_ttest_summary.csv"), index=False)
        
        self.logger.info("\n" + "=" * 50)
        self.logger.info("FINAL CONCLUSION: USELESS PARAMETERS")
        self.logger.info("=" * 50)
        useless_params = [p["Parameter"] for p in param_summary if p["Is_Useless"] == "YES"]
        if useless_params:
            self.logger.info(f"The following parameters DO NOT significantly affect any output (p > 0.05):")
            for up in useless_params:
                self.logger.info(f"  - {up}")
        else:
            self.logger.info("All parameters significantly affect at least one output. No useless parameters found.")
            
        self.logger.info(f"\nAll results and plots have been saved to {self.output_dir}")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"statistical_tests_{timestamp}")
    
    analyzer = StatisticalAnalyzer(
        num_samples=30,  # User requested 30
        instance_path=os.path.join(project_root, "data", "small", "instance.json"),
        output_dir=output_dir
    )
    analyzer.run_all()

if __name__ == "__main__":
    main()
