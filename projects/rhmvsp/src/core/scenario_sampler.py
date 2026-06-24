"""
Monte Carlo Scenario Sampler for Out-of-Sample Testing.

Randomly perturbs demand, arc travel times, and risk parameters in a
deep-copied instance, creating diverse "out-of-sample" realizations
for robustness evaluation.

Perturbation sources:
  1. OD-pair demand: ±demand_pct (uniform, round to q_road)
  2. Arc travel times: ±time_pct (uniform, preserving trapezoidal shape)
  3. Arc risk alpha: ±risk_pct (uniform, on arcs with alpha > 0)
"""
from __future__ import annotations
import json
import numpy as np


def sample_scenario(
    base_instance: dict,
    rng: np.random.Generator | np.random.RandomState,
    demand_pct: float = 0.20,
    time_pct: float = 0.15,
    risk_pct: float = 0.30,
) -> dict:
    """
    Return a NEW deep-copied instance with perturbed demand, arc times,
    and risk parameters.

    Parameters
    ----------
    base_instance : dict   — Original instance JSON dict (not mutated).
    rng           : Generator — Numpy RNG (supports both legacy RandomState
                                and new Generator).
    demand_pct    : float  — Max fractional demand change (±20 % default).
    time_pct      : float  — Max fractional arc-time change (±15 % default).
    risk_pct      : float  — Max fractional arc risk-alpha change (±30 % default).

    Returns
    -------
    dict — New instance with perturbed demand, arc times, and risk.
    """
    inst = json.loads(json.dumps(base_instance))          # deep copy

    _perturb_demand(inst.get("od_pairs", []), rng, demand_pct)
    _perturb_arc_times(inst.get("network", {}).get("arcs", []), rng, time_pct)
    _perturb_risk(inst.get("network", {}).get("arcs", []), rng, risk_pct)

    return inst


# ── internal helpers ──────────────────────────────────────────────────────────

Q_ROAD = 30.0  # Road vehicle capacity (tons) — from config.py


def _perturb_demand(
    od_pairs: list[dict],
    rng: np.random.Generator | np.random.RandomState,
    pct: float,
) -> None:
    """
    Perturb each OD-pair demand by Uniform(1-pct, 1+pct) and round
    upward to the nearest q_road so that every realization remains
    feasible for integer vehicle allocations.
    """
    for od in od_pairs:
        noise = rng.uniform(1.0 - pct, 1.0 + pct)
        new_demand = od["demand"] * noise
        # Round up to nearest q_road to keep feasibility (≤ capacity)
        new_demand = max(Q_ROAD, np.ceil(new_demand / Q_ROAD) * Q_ROAD)
        od["demand"] = float(new_demand)


def _perturb_arc_times(
    arcs: list[dict],
    rng: np.random.Generator | np.random.RandomState,
    pct: float,
) -> None:
    """
    Scale each arc's trapezoidal time support (a, b, c, d) by
    Uniform(1-pct, 1+pct).  Every arc gets its own independent draw.
    This widens or narrows the fuzzy support while preserving the
    trapezoidal shape (a ≤ b ≤ c ≤ d).
    """
    for arc in arcs:
        noise = rng.uniform(1.0 - pct, 1.0 + pct)
        for key in ("time_a", "time_b", "time_c", "time_d"):
            arc[key] = round(arc[key] * noise, 6)
        # Guard: ensure a ≤ b ≤ c ≤ d
        times = [arc["time_a"], arc["time_b"], arc["time_c"], arc["time_d"]]
        times_sorted = sorted(times)
        arc["time_a"], arc["time_b"], arc["time_c"], arc["time_d"] = [
            round(t, 6) for t in times_sorted
        ]


def _perturb_risk(
    arcs: list[dict],
    rng: np.random.Generator | np.random.RandomState,
    pct: float,
) -> None:
    """
    Perturb each arc's accident belief alpha by Uniform(1-pct, 1+pct).
    Only arcs with alpha > 0 are perturbed.  The result is clamped to
    (0, 0.95] so that R = 1-alpha stays positive.
    """
    for arc in arcs:
        if arc.get("alpha", 0) <= 0:
            continue
        noise = rng.uniform(1.0 - pct, 1.0 + pct)
        new_alpha = arc["alpha"] * noise
        arc["alpha"] = round(max(1e-6, min(0.95, new_alpha)), 5)
