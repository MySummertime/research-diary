"""Reproducible synthetic accident-belief generation for experimental instances."""

from __future__ import annotations

import hashlib


def _unit_draw(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def draw_belief(seed: int, key: str, bounds: tuple[float, float]) -> float:
    """Return a deterministic pseudorandom belief in the closed interval."""
    lower, upper = map(float, bounds)
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(f"Invalid accident-belief bounds: {bounds!r}")
    return lower + (upper - lower) * _unit_draw(seed, key)


def arc_belief(seed: int, arc_id: int, i: int, j: int, mode: int,
               road_bounds: tuple[float, float],
               rail_bounds: tuple[float, float]) -> float:
    bounds = road_bounds if int(mode) == 1 else rail_bounds
    return draw_belief(seed, f"arc:{arc_id}:{i}:{j}:{mode}", bounds)
