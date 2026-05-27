from dataclasses import dataclass

@dataclass
class SensitivityConfig:
    """
    Configuration parameters specifically for the Sensitivity Analysis scripts.
    Isolated from the main business logic RHMVSPConfig.
    """
    time_limit: int = 30  # Hard time limit (seconds) for sensitivity analysis runs
    time_window_buffer_hours: float = 10.0  # Buffer added to E_od to construct L_od in sensitivity tests
    tight_supply_multiplier: float = 1.01  # Demand multiplier to set tight supply constraints
