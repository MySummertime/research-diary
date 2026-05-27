import json
import os
import sys
from copy import deepcopy
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import make_interp_spline
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from projects.rhmvsp.config import RHMVSPConfig, PlotProperties, ColorPalette
from projects.rhmvsp.src.solvers.branch_and_price import BranchAndPriceController
from projects.rhmvsp.src.utils.logging import Logger
from projects.rhmvsp.src.scripts.config_sensitivity import SensitivityConfig


class SensitivityAnalyzer:
    """
    Modular Sensitivity Analyzer for the Reliable Hazardous Materials Multi-Modal
    Vehicle Scheduling Problem (RHMVSP). Executes systematic parameter variations
    and extracts structured operational metrics.
    """
    def __init__(self, config_path: str, instance_path: str, output_path: str, logger: Logger):
        self.config_path = config_path
        self.instance_path = instance_path
        self.output_path = output_path
        self.logger = logger
        self.results_dir = os.path.dirname(self.output_path)
        os.makedirs(self.results_dir, exist_ok=True)

        self.base_config = RHMVSPConfig()
        self.sens_config = SensitivityConfig()
        with open(instance_path, "r", encoding="utf-8") as f:
            self.base_instance = json.load(f)

    def _run_variation(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a single solver instance with the specified parameter variations."""
        param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items())
        self.logger.info("-" * 50)
        self.logger.info(f"Running Sensitivity: {param_str}")
        self.logger.info("-" * 50)

        valid_keys = set(RHMVSPConfig.__dataclass_fields__.keys())
        cfg_dict = {k: getattr(self.base_config, k) for k in valid_keys}
        for k, v in param_dict.items():
            if k in cfg_dict:
                cfg_dict[k] = v

        # Set a hard time limit for sensitivity analysis to prevent combinatorial explosion delays
        cfg_dict["time_limit"] = self.sens_config.time_limit
        config = RHMVSPConfig(**cfg_dict)

        inst = deepcopy(self.base_instance)
        inst["config"] = {k: getattr(config, k) for k in valid_keys if isinstance(getattr(config, k), (int, float, str, tuple, list))}
        if "budget" in param_dict:
            inst["budget"] = param_dict["budget"]

        # Sandbox-discovered boundaries for proper sensitivity demonstration
        if "delta_T" in param_dict:
            for od in inst.get("od_pairs", []):
                od["L_od"] = od["E_od"] + self.sens_config.time_window_buffer_hours
                
        if "delta_S" in param_dict:
            demand_per_origin = {}
            for od in inst.get("od_pairs", []):
                orig = od["origin"]
                demand_per_origin[orig] = demand_per_origin.get(orig, 0) + od["demand"]
            for orig, total_demand in demand_per_origin.items():
                tight_supply = total_demand * self.sens_config.tight_supply_multiplier
                for n in inst["network"]["nodes"]:
                    if n["id"] == orig:
                        n["supply_capacity"] = tight_supply
                        break

        solver = BranchAndPriceController(config, inst, self.logger)
        sol = solver.solve()

        metrics = {
            "status": sol.get("status", "unknown"),
            "total_cost": sol.get("total_cost", 0.0),
            "deploy_cost": sol.get("deploy_cost", 0.0),
            "hub_cost": sol.get("hub_cost", 0.0),
            "transport_cost": sol.get("transport_cost", 0.0),
            "c_unmet": sol.get("c_unmet", 0.0),
            "holding_cost": sol.get("holding_cost", 0.0),
            "reposition_cost": sol.get("reposition_cost", 0.0),
            "opportunity_cost": sol.get("opportunity_cost", 0.0),
            "active_hubs": len(sol.get("active_hubs", [])),
            "vehicles_used": sol.get("num_vehicles_used", 0),
            "R_sys": sol.get("R_net", getattr(config, 'alpha_max', 0.95)),
            "R_sys_risk": sol.get("R_net_risk", getattr(config, 'alpha_max', 0.95)),
            "R_sys_time": sol.get("R_net_time", getattr(config, 'alpha_max', 0.95)),
            "iterations": sol.get("iterations", 0),
            "columns": sol.get("total_columns", 0)
        }

        self.logger.info(f"Result [{param_str}]: Total Cost (c_total) = ¥{metrics['total_cost']:,.2f}, Shortage Penalty (\\hat{{c}}_{{unmet}}) = ¥{metrics['c_unmet']:,.2f}, Hubs = {metrics['active_hubs']}")
        return metrics

    def analyze_1d(self, param_name: str, test_values: List[Any]) -> Dict[str, Any]:
        """Evaluates model sensitivity across a single parameter."""
        self.logger.info(f"--- Starting 1D Sensitivity Analysis: {param_name} ({test_values}) ---")
        results = {}
        for val in test_values:
            results[str(val)] = self._run_variation({param_name: val})
        return results

    def analyze_2d(self, p1_name: str, p1_vals: List[Any], p2_name: str, p2_vals: List[Any]) -> Dict[str, Any]:
        """Evaluates model sensitivity across a 2D grid of parameters."""
        self.logger.info(f"--- Starting 2D Sensitivity Analysis: {p1_name} x {p2_name} ---")
        results = {}
        for v1 in p1_vals:
            for v2 in p2_vals:
                key = f"{v1}_{v2}"
                results[key] = self._run_variation({p1_name: v1, p2_name: v2})
        return {"p1": p1_name, "p2": p2_name, "p1_vals": p1_vals, "p2_vals": p2_vals, "data": results}

    def run_all(self, param_name: str, test_values: List[float], param2_name: str = None, test2_values: List[float] = None) -> Dict[str, Any]:
        """Executes sensitivity analyses for the specific parameter and persists results."""
        self.logger.info("=" * 60)
        self.logger.info(f"Beginning Comprehensive RHMVSP Sensitivity Analysis")
        self.logger.info("=" * 60)

        full_results = {
            "1D": {},
            "2D": {}
        }
        
        if param2_name is None:
            full_results["1D"][param_name] = self.analyze_1d(param_name, test_values)
        else:
            key = f"{param_name}_x_{param2_name}"
            full_results["2D"][key] = self.analyze_2d(param_name, test_values, param2_name, test2_values)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

        self.logger.info("=" * 60)
        self.logger.info(f"Sensitivity analysis successfully completed. Saved to {self.output_path}")
        
        self.generate_plots(full_results)
        
        return full_results

    def generate_plots(self, full_results: Dict[str, Any]):
        """Generates advanced plots and extra tables based on the full sensitivity results."""
        self.logger.info("Generating advanced plots and tables...")
        sns.set_theme(style="whitegrid", font=PlotProperties.FONT_FAMILY)
        
        # 1D Plots and Tables
        for param, res in full_results["1D"].items():
            df = pd.DataFrame.from_dict(res, orient="index")
            
            # Avoid ValueError if the param name is exactly the same as an output metric (e.g. c_unmet)
            output_metric_renamed = False
            if param in df.columns:
                df.rename(columns={param: f"{param}_metric_out"}, inplace=True)
                output_metric_renamed = True
                
            df.index.name = param
            df.reset_index(inplace=True)
            df[param] = df[param].astype(float)
            
            # --- Save Standard Raw Table ---
            csv_path = os.path.join(self.results_dir, f"table_1d_{param}.csv")
            df.to_csv(csv_path, index=False)
            
            # --- Generate Extra Tables ---
            # 1. Cost Percentage Table
            df_pct = df[[param]].copy()
            cost_cols = ["deploy_cost", "hub_cost", "transport_cost", "c_unmet", "holding_cost", "opportunity_cost"]
            for col in cost_cols:
                actual_col = f"{col}_metric_out" if (col == param and output_metric_renamed) else col
                df_pct[f"{col}_pct"] = (df[actual_col] / df["total_cost"] * 100).round(2)
            df_pct.to_csv(os.path.join(self.results_dir, f"table_1d_{param}_cost_pct.csv"), index=False)
            
            # 2. Operational Metrics Table
            df_ops = df[[param, "active_hubs", "vehicles_used", "R_sys", "R_sys_risk", "R_sys_time"]].copy()
            df_ops.to_csv(os.path.join(self.results_dir, f"table_1d_{param}_operations.csv"), index=False)
            
            # --- Advanced Plotting ---
            self._plot_custom_dual_y(param, df)

        # 2D Plots (3D Surface Plot)
        for key, res in full_results["2D"].items():
            self._plot_3d_surface(res["p1"], res["p2"], res)
        
        self.logger.info("All advanced plots and extra tables generated successfully.")

    def _plot_3d_surface(self, p1_name: str, p2_name: str, res: Dict[str, Any]):
        """Generates a true 3D surface plot from a 2D parameter sweep meshgrid."""
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        param_labels = {
            "alpha_max": r"Risk Tolerance ($\alpha_{max}$)",
            "delta_D": r"Demand Uncertainty ($\delta_D$)",
            "delta_S": r"Supply Uncertainty ($\delta_S$)",
            "delta_C": r"Cost Uncertainty ($\delta_C$)",
            "delta_T": r"Time Uncertainty ($\delta_T$)",
            "alpha_T": r"Time Reliability ($\alpha_T$)",
            "c_unmet": r"Shortage Penalty Cost ($\hat{c}_{unmet}$)"
        }
        
        p1_vals = res["p1_vals"]
        p2_vals = res["p2_vals"]
        data = res["data"]
        
        X, Y = np.meshgrid(p1_vals, p2_vals)
        Z_cost = np.zeros_like(X, dtype=float)
        
        for i, v2 in enumerate(p2_vals):
            for j, v1 in enumerate(p1_vals):
                key = f"{v1}_{v2}"
                Z_cost[i, j] = data[key]["total_cost"]
                
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a custom colormap from our Academic Morandi color set
        # Transition: Steel Blue (Low) -> Celadon -> Sage Green -> Terracotta (High)
        morandi_colors = ["#4682B4", ColorPalette.CELADON, ColorPalette.SAGE_GREEN, ColorPalette.TERRACOTTA]
        custom_cmap = LinearSegmentedColormap.from_list("academic_morandi", morandi_colors)
        
        # Draw the true surface plot (Wang 2025 flat-faceted style)
        surf = ax.plot_surface(X, Y, Z_cost, cmap=custom_cmap, edgecolor='k', 
                               linewidth=0.5, alpha=0.9, antialiased=False, shade=True)
        
        ax.set_xlabel(param_labels.get(p1_name, p1_name), fontsize=14, labelpad=15)
        ax.set_ylabel(param_labels.get(p2_name, p2_name), fontsize=14, labelpad=15)
        ax.set_zlabel(r"Total Cost ($C_{total}$)", fontsize=14, labelpad=15)
        
        ax.view_init(elev=35, azim=225)
        ax.set_title(f"3D Objective Surface: {p1_name} vs {p2_name}", fontsize=16, pad=20)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        
        fig.tight_layout()
        out_name = f"plot_3d_surface_{p1_name}_{p2_name}"
        plt.savefig(os.path.join(self.results_dir, f"{out_name}.png"), dpi=PlotProperties.DPI, transparent=True)
        plt.savefig(os.path.join(self.results_dir, f"{out_name}.tiff"), dpi=PlotProperties.DPI, transparent=True)
        plt.close()

    def _plot_custom_dual_y(self, param: str, df: pd.DataFrame):
        """Generates a pure 2-Y axis line chart for parameter vs Time & Risk Reliability."""
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        param_labels = {
            "alpha_max": r"Risk Tolerance ($\alpha_{max}$)",
            "delta_D": r"Demand Uncertainty ($\delta_D$)",
            "delta_S": r"Supply Uncertainty ($\delta_S$)",
            "delta_C": r"Cost Uncertainty ($\delta_C$)",
            "delta_T": r"Time Uncertainty ($\delta_T$)",
            "alpha_T": r"Time Reliability ($\alpha_T$)",
            "c_unmet": r"Shortage Penalty Cost ($\hat{c}_{unmet}$)"
        }
        x_label = param_labels.get(param, f"Parameter (${param}$)")
        
        x = df[param].values.astype(float)
        rel_risk = df["R_sys_risk"].values.astype(float)
        rel_time = df["R_sys_time"].values.astype(float)
        
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        rel_risk = rel_risk[sort_idx]
        rel_time = rel_time[sort_idx]
        
        fig, ax1 = plt.subplots(figsize=(9, 6))
        
        # Left Y-Axis: Risk Reliability
        color_rel_risk = ColorPalette.TERRACOTTA  # Warm Terracotta
        line1 = ax1.plot(x, rel_risk, '-o', color=color_rel_risk, linewidth=2.5, markersize=8, markeredgecolor='white', label=r"$\mathit{R_{net}^{risk}}$")
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel(r"Risk Reliability ($R_{net}^{risk}$)", color=color_rel_risk, fontsize=14)
        ax1.tick_params(axis='y', labelcolor=color_rel_risk, direction='in')
        ax1.tick_params(axis='x', direction='in')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{val:g}" for val in x])
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Right Y-Axis: Time Reliability
        ax2 = ax1.twinx()
        color_rel_time = ColorPalette.CELADON  # Celadon Green
        line2 = ax2.plot(x, rel_time, '-^', color=color_rel_time, linewidth=2.5, markersize=8, markeredgecolor='white', label=r"$\mathit{R_{net}^{time}}$")
        ax2.set_ylabel(r"Time Reliability ($R_{net}^{time}$)", color=color_rel_time, fontsize=14)
        ax2.tick_params(axis='y', labelcolor=color_rel_time, direction='in')
        
        # 强制视觉分离两个 Y 轴的曲线 (Prevent overlap)
        risk_min, risk_max = min(rel_risk), max(rel_risk)
        time_min, time_max = min(rel_time), max(rel_time)
        
        risk_range = max(1e-4, risk_max - risk_min)
        time_range = max(1e-4, time_max - time_min)
        
        # 风险可靠性 (Left Axis) 压在图表下半区：上半部分留白 150%
        ax1.set_ylim(risk_min - risk_range * 0.2, risk_max + risk_range * 1.5)
        
        # 时间可靠性 (Right Axis) 托在图表上半区：下半部分留白 150%
        ax2.set_ylim(time_min - time_range * 1.5, time_max + time_range * 0.2)

        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        combined_lines = lines + lines2
        combined_labels = [r"Risk Reliability ($R_{net}^{risk}$)", r"Time Reliability ($R_{net}^{time}$)"]
        ax1.legend(combined_lines, combined_labels, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=12)
        
        fig.tight_layout()
        
        out_name = f"plot_dual_{param}"
        plt.savefig(os.path.join(self.results_dir, f"{out_name}.png"), dpi=PlotProperties.DPI, transparent=True)
        plt.savefig(os.path.join(self.results_dir, f"{out_name}.tiff"), dpi=PlotProperties.DPI, transparent=True)
        plt.close()

def main():
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(description="Run RHMVSP Sensitivity Analysis")
    parser.add_argument("--param", type=str, required=True, help="Parameter to test (e.g. delta_D)")
    parser.add_argument("--values", type=str, required=True, help="Comma-separated values to test (e.g. 0.0,0.1,0.5)")
    parser.add_argument("--param2", type=str, required=False, help="Second parameter for 2D surface plot (e.g. delta_T)")
    parser.add_argument("--values2", type=str, required=False, help="Comma-separated values for second parameter")
    args = parser.parse_args()
    
    test_values = [float(v.strip()) for v in args.values.split(",")]
    
    test2_values = None
    if args.param2 and args.values2:
        test2_values = [float(v.strip()) for v in args.values2.split(",")]
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = args.param if not args.param2 else f"{args.param}_x_{args.param2}"
    results_dir = os.path.join(project_root, "results", f"sensitivity_{timestamp}_{folder_name}")
    os.makedirs(results_dir, exist_ok=True)
    
    logger = Logger(level="INFO", log_file=os.path.join(results_dir, "sensitivity.log"))
    analyzer = SensitivityAnalyzer(
        config_path=os.path.join(project_root, "config.json"),
        instance_path=os.path.join(project_root, "data", "sandbox", "instance.json"),
        output_path=os.path.join(results_dir, "sensitivity_results.json"),
        logger=logger
    )
    
    analyzer.run_all(args.param, test_values, args.param2, test2_values)

if __name__ == "__main__":
    main()
