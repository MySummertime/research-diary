"""Consistent presentation precision for reports and experiment tables.

Solution JSON files retain full-precision numeric values.  These helpers are
for human-readable logs and CSV exports only.
"""
from decimal import Decimal

RISK_DIGITS = 4
RELIABILITY_DIGITS = 4
TIME_RELIABILITY_DIGITS = 4
ALPHA_R_DIGITS = 3
ALPHA_T_DIGITS = 2
TIME_DIGITS = 2
COST_DIGITS = 2
PERCENT_DIGITS = 4


def format_number(value, digits: int, default: str = "") -> str:
    """Return a fixed-decimal string while preserving missing values."""
    if value is None or value == "":
        return default
    return f"{float(value):.{digits}f}"


def parameter_digits(values, max_digits: int = 12) -> int:
    """Infer the displayed precision from the supplied parameter values.

    For a grid, this preserves its finest explicitly supplied resolution.  It
    deliberately uses ``str(value)`` rather than binary-float arithmetic.
    """
    digits = 0
    for value in values:
        exponent = Decimal(str(value)).as_tuple().exponent
        digits = max(digits, max(0, -exponent))
    return min(digits, max_digits)


def format_parameter(value, values=None) -> str:
    """Format a parameter using its own or its experiment grid's precision."""
    context = list(values) if values is not None else [value]
    return format_number(value, parameter_digits(context))


def format_alpha_R(value) -> str:
    """Format the risk-reliability threshold like risk reliability."""
    return format_number(value, ALPHA_R_DIGITS)


def format_alpha_T(value) -> str:
    """Format the time-reliability threshold like time reliability."""
    return format_number(value, ALPHA_T_DIGITS)


def format_csv_value(field: str, value) -> str:
    """Format a CSV value according to the metric represented by *field*."""
    if value is None or value == "":
        return ""
    name = field.lower()
    if name in {"alpha_r", "risk_reliability_threshold_alpha_r"}:
        return format_alpha_R(value)
    if name in {"alpha_t", "time_reliability_threshold_alpha_t"}:
        return format_alpha_T(value)
    if "gap" in name or "percent" in name or "pct" in name:
        return format_number(value, PERCENT_DIGITS)
    if "time" in name and "reliability" in name:
        return format_number(value, TIME_RELIABILITY_DIGITS)
    if "reliability" in name:
        return format_number(value, RELIABILITY_DIGITS)
    if name == "risk" or name.endswith("_risk") or "networkrisk" in name:
        return format_number(value, RISK_DIGITS)
    if "cost" in name or "penalty" in name:
        return format_number(value, COST_DIGITS)
    if name == "time" or name.endswith("_time") or "networktime" in name:
        return format_number(value, TIME_DIGITS)
    if "elapsed" in name:
        return format_number(value, TIME_DIGITS)
    return str(value)


def format_csv_row(row: dict) -> dict:
    """Return a copy of a named CSV row with consistent display precision."""
    return {field: format_csv_value(field, value) for field, value in row.items()}
