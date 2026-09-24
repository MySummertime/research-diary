"""Shared reliability-threshold grids calibrated to the model inputs."""

# Main-experiment threshold and the common sensitivity grid.
BASELINE_ALPHA_R = 0.90
AUDITED_RISK_RELIABILITY = 0.90

# The risk grid resolves the expected plan-reliability range instead of using
# degenerate endpoints or an uninformative 0.1 step.
ALPHA_R_SCAN = [0.890, 0.895, 0.900, 0.905, 0.910, 0.920, 0.930, 0.935]
ALPHA_T_SCAN = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Three progressive MR designs selected from the coupled experiment.  Keep
# these as explicit pairs: they are not intended to form a Cartesian grid.
VALUE_OF_RELIABILITY_PARAM_GRID = [
    (0.75, 0.890),
    (0.75, 0.910),
    (0.75, 0.930),
    (0.95, 0.890),
    (0.95, 0.910),
    (0.95, 0.930),
]

# Out-of-sample settings are independent from the value-of-reliability study.
ALPHA_R_OOS_LEVELS = [0.890, 0.910, 0.930]
ALPHA_T_OOS_LEVELS = [0.75, 0.95]

# Algorithm-performance settings are named independently so changing one
# experiment cannot silently change another experiment's design.
ALPHA_R_ALGORITHM_LEVELS = [0.890, 0.910, 0.930]
ALPHA_T_ALGORITHM_LEVELS = [0.75, 0.95]
