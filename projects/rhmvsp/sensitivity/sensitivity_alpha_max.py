"""Sensitivity: alpha_max vs R_risk, R_time"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensitivity.runner import run_sensitivity

alpha_max_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.10]
run_sensitivity(
    param_name="alpha_max",
    param_values=alpha_max_values,
    param_attr_path="alpha_max",
    instance_path="data/small/instance.json",
    label="alpha_max",
    #time_limit=120,
)