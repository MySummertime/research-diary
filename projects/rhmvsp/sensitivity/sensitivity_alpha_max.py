"""Sensitivity: alpha_max (risk reliability threshold) vs R_risk, R_time"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensitivity.runner import run_sensitivity

# Reliability thresholds (paper convention: higher = stricter)
# Internally converted to unreliability via beta_od = 1 - alpha_max
alpha_max_values = [0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985]
run_sensitivity(
    param_name="alpha_max",
    param_values=alpha_max_values,
    param_attr_path="alpha_max",
    instance_path="data_real_case/instance.json",
    label="alpha_max",
    time_limit=120,
    invert_param=True,  # reliability threshold → unreliability for solver
)