"""Run multiple 1D sensitivity sweeps programmatically.

This script reuses the SensitivityAnalyzer to run several 1D sweeps
for parameters: delta_D, delta_S, delta_C, delta_T, alpha_T, c_unmet.
Results are saved in `results/` with timestamped subfolders.
"""
from __future__ import annotations
import os
import datetime
import subprocess


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(project_root, "results", f"multi_sensitivity_{timestamp}")
    os.makedirs(base_results_dir, exist_ok=True)

    # Use simple prints instead of importing project logger to avoid import path issues
    def logger_info(msg: str):
        print(msg)

    # Define parameter grids
    param_grid = {
        "delta_D": [0.0, 0.1, 0.2, 0.3, 0.5],
        "delta_S": [0.0, 0.1, 0.2, 0.3, 0.5],
        "delta_C": [0.0, 0.1, 0.2, 0.3, 0.5],
        "delta_T": [0.1, 0.2, 0.4, 0.6, 0.8],
        "alpha_T": [0.1, 0.3, 0.5, 0.7, 0.9],
        "c_unmet": [100.0, 500.0, 1000.0, 2000.0, 5000.0],
    }

    for param, vals in param_grid.items():
        out_dir = os.path.join(base_results_dir, f"sensitivity_{param}")
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            "conda", "run", "-n", "rhmvsp", "python",
            os.path.join(project_root, "src", "scripts", "run_sensitivity.py"),
            "--param", param,
            "--values", ",".join(str(v) for v in vals),
        ]
        logger_info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    logger_info(f"Multi-parameter sweeps complete. Results base: {base_results_dir}")


if __name__ == "__main__":
    main()
