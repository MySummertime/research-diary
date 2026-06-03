"""Sensitivity: alpha_T vs R_risk, R_time"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensitivity.runner import run_sensitivity

alpha_T_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
run_sensitivity(
    param_name="alpha_T",
    param_values=alpha_T_values,
    param_attr_path="alpha_T",
    instance_path="data/small/instance.json",
    label="alpha_T",
    #time_limit=120,
)