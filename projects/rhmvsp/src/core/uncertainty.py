from __future__ import annotations
import math
"""
Uncertainty Theory Engine (§2, §3.2).

Transport: Zigzag uncertain variables (Liu, Uncertainty Theory, 6th ed.,
Definition 3.8, Eq. 3.50). Transfer: trapezoidal uncertain-set time measures
(Chapter 9). These objects are not identified with probability or fuzzy
credibility merely because some numerical formulas coincide.

Time-varying risk (§2.2): the accident belief alpha_ij^m(t) scales with the
traffic and fatigue multiplier phi(t). Risk is reported as accident belief.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


def _validate_alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("Confidence must be finite and in [0, 1]")


def _validate_support(values) -> None:
    if not all(math.isfinite(x) for x in values) or any(x > y for x, y in zip(values, values[1:])):
        raise ValueError(f"Support must be finite and ordered: {values}")


@dataclass
class ZigzagUncertainVariable:
    """Z(a,b,c): lower bound, median, upper bound (book Eq. 3.50).

    Strict a<b<c is the book's regular case. Equal endpoints are supported
    as limiting distributions, including the deterministic constant a=b=c.
    """
    a: float
    b: float
    c: float

    def __post_init__(self):
        _validate_support((self.a, self.b, self.c))

    def membership(self, x: float) -> float:
        """Legacy triangular shape helper, NOT the uncertainty distribution."""
        if x < self.a or x > self.c:
            return 0.0
        if x == self.b:
            return 1.0
        if x < self.b:
            return (x-self.a)/(self.b-self.a)
        return (self.c-x)/(self.c-self.b)

    def uncertain_measure_leq(self, x: float) -> float:
        """Uncertainty distribution, book Eq. 3.50; constants use a step."""
        if x < self.a:
            return 0.0
        if x >= self.c:
            return 1.0
        if x < self.b:
            return (x-self.a)/(2.0*(self.b-self.a))
        return 1.0-(self.c-x)/(2.0*(self.c-self.b))

    def uncertain_measure_geq(self, x: float) -> float:
        """Use reflection so inclusive inequalities remain correct at atoms."""
        return ZigzagUncertainVariable(-self.c, -self.b, -self.a).uncertain_measure_leq(-x)

    def deterministic_equivalent_upper(self, alpha: float) -> float:
        """Finite upper-quantile duration for a reliability level in [0, 1]."""
        _validate_alpha(alpha)
        if alpha == 0:
            # A vacuous time-reliability constraint still needs a finite
            # duration for path scheduling.  The lower support is the least
            # restrictive physically meaningful value; -inf would propagate
            # through label clocks and produce NaN departure times.
            return self.a
        if alpha <= 0.5:
            return self.a+2.0*alpha*(self.b-self.a)
        return (2.0*alpha-1.0)*self.c+2.0*(1.0-alpha)*self.b

    def deterministic_equivalent_lower(self, alpha: float) -> float:
        """Largest E with M{T>=E}>=alpha, including constants."""
        return -ZigzagUncertainVariable(-self.c, -self.b, -self.a).deterministic_equivalent_upper(alpha)

    def mean(self) -> float:
        """Book Eq. 3.192: E[T]=(a+2b+c)/4."""
        return (self.a+2*self.b+self.c)/4.0


# Backward-compatible import for callers from earlier versions.
TriangularFuzzy = ZigzagUncertainVariable


@dataclass
class TrapezoidalFuzzy:
    """
    Trapezoidal fuzzy variable tau~ = (a, b, c, d) where a <= b <= c <= d.

    Membership function:
        mu(x) = (x-a)/(b-a)  if a <= x <= b
        mu(x) = 1            if b <= x <= c
        mu(x) = (d-x)/(d-c)  if c <= x <= d
        mu(x) = 0            otherwise

    Used for: transfer time (§3.2, Eq.8) and aggregated path duration.
    """
    a: float  # Minimum
    b: float  # Lower mode start
    c: float  # Upper mode end
    d: float  # Maximum

    def __post_init__(self):
        _validate_support((self.a, self.b, self.c, self.d))

    def membership(self, x: float) -> float:
        """Trapezoidal shape; book Eq. 9.59, extended to collapsed supports."""
        if x < self.a or x > self.d:
            return 0.0
        if self.b <= x <= self.c:
            return 1.0
        if x < self.b:
            return (x-self.a)/(self.b-self.a)
        return (self.d-x)/(self.d-self.c)

    def uncertain_measure_leq(self, x: float) -> float:
        """Imaginary-inclusion time measure (book Eq. 9.191).

        Nondegenerate supports have continuous tails and a 0.5 plateau.
        Collapsed endpoints use inclusive limiting values; a=b=c=d is constant.
        """
        if x < self.a:
            return 0.0
        if x >= self.d:
            return 1.0
        if x < self.b:
            return (x-self.a)/(2.0*(self.b-self.a))
        if x <= self.c:
            return 0.5
        return 1.0-(self.d-x)/(2.0*(self.d-self.c))

    def uncertain_measure_geq(self, x: float) -> float:
        """Reflect the support; do not subtract an inclusive CDF at atoms."""
        return TrapezoidalFuzzy(-self.d, -self.c, -self.b, -self.a).uncertain_measure_leq(-x)

    def deterministic_equivalent_upper(self, alpha: float) -> float:
        """Finite upper-quantile duration; at 0.5 use the upper core c."""
        _validate_alpha(alpha)
        if alpha == 0:
            # See ZigzagUncertainVariable: scheduling must remain finite even
            # when the time-reliability constraint is disabled.
            return self.a
        if alpha < 0.5:
            return self.a+2.0*alpha*(self.b-self.a)
        if alpha == 0.5:
            return self.c
        return (2.0*alpha-1.0)*self.d+2.0*(1.0-alpha)*self.c

    def deterministic_equivalent_lower(self, alpha: float) -> float:
        """Largest feasible E; at alpha=0.5 the plateau ends at c, not b."""
        return -TrapezoidalFuzzy(-self.d, -self.c, -self.b, -self.a).deterministic_equivalent_upper(alpha)

    def mean(self) -> float:
        """
        Expected value under uncertainty theory (Liu 2007).
        For trapezoidal (a,b,c,d): E = (a + b + c + d) / 4
        """
        return (self.a + self.b + self.c + self.d) / 4.0

    def __repr__(self):
        return f"TrapFuzzy({self.a:.2f}, {self.b:.2f}, {self.c:.2f}, {self.d:.2f})"


@dataclass
class BoolUncertainVariable:
    """
    Boolean uncertain variable xi (§2.2, §3.2).

    M{xi = 1} = alpha (accident belief)
    M{xi = 0} = 1 - alpha

    Used for: arc/node accident occurrence.
    """
    alpha: float  # Accident belief

    def reliability(self) -> float:
        """R = M{xi = 0} = 1 - alpha (Eq.1)."""
        return 1.0 - self.alpha

# ============================================================
# Time-varying accident-belief multipliers
# ============================================================

def _traffic_multiplier(hour_of_day: float, profile: tuple[float, ...]) -> float:
    """
    Time-varying accident-belief multiplier phi(t) (S2-2).

    The day is divided into 24 one-hour intervals. Each interval has one
    assumed constant value; no interpolation is used between adjacent hours.
    """
    h = hour_of_day % 24.0
    if len(profile) != 24 or any(value <= 0.0 for value in profile):
        raise ValueError("Hourly accident-belief profile requires 24 positive multipliers")
    return float(profile[int(h)])


class UncertaintyEngine:
    """
    Central engine for all uncertainty computations.

    Creates fuzzy variables from network parameters and computes
    deterministic equivalents for reliability constraints.
    """

    def __init__(self, config):
        self.config = config

    def make_transport_time(self, distance: float, mode: int) -> ZigzagUncertainVariable:
        """Create Z(base*a, base*b, base*c), base=distance/speed."""
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Distance must be finite and nonnegative")
        if mode == 1:
            speed = self.config.road_speed
            ratios = self.config.road_fuzzy_time_ratio
        elif mode == 2:
            speed = self.config.rail_speed
            ratios = self.config.rail_fuzzy_time_ratio
        else:
            raise ValueError(f"Unknown transport mode: {mode}")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Transport speed must be finite and positive")
        if len(ratios) != 3:
            raise ValueError("Transport ratios require three values (lower, median, upper)")
        _validate_support(ratios)
        if ratios[0] < 0:
            raise ValueError("Transport time ratios must be nonnegative")
        base = distance/speed
        return ZigzagUncertainVariable(*(base*r for r in ratios))

    def road_rest_time(self, drive_time: float) -> float:
        """Return rests for a road block that starts with zero driving state."""
        return self.road_drive_transition(0.0, drive_time)[0]

    def road_drive_transition(
        self, continuous_drive: float, drive_time: float
    ) -> tuple[float, float]:
        """Advance cumulative road-driving state and insert mandatory rests.

        Returns ``(rest_time, residual_continuous_drive)``.  A rest is inserted
        immediately when accumulated driving reaches the limit.  Non-road
        operations reset the state in the calling scheduling logic.
        """
        limit = self.config.driver_fatigue_limit_hours
        total_drive = max(0.0, continuous_drive) + max(0.0, drive_time)
        rest_count = int((total_drive + self.config.tolerance) // limit)
        residual_drive = total_drive - rest_count * limit
        return rest_count * self.config.driver_rest_hours, residual_drive

    def _integrate_risk_multiplier(self, t1: float, t2: float) -> float:
        """
        Calculate the average accident-belief multiplier over [t1, t2].

        The multiplier is constant in each of the 24 one-hour intervals, so
        the exact integral is the sum of interval value times overlap length.
        """
        if t1 >= t2:
            return _traffic_multiplier(t1, self.config.hourly_accident_belief_multipliers)

        total_integral = 0.0
        current_t = t1
        while current_t < t2:
            day = int(current_t // 24)
            h = current_t - day * 24.0
            next_h = min(24.0, math.floor(h) + 1.0)
            seg_end = min(t2, day * 24.0 + next_h)
            step = seg_end - current_t
            total_integral += _traffic_multiplier(
                current_t, self.config.hourly_accident_belief_multipliers
            ) * step
            current_t = seg_end

        return total_integral / (t2 - t1)

    def arc_reliability(self, alpha: float, start_time: float = 0.0, end_time: float = 0.0,
                        time_dependent: bool = False, arrival_time: Optional[float] = None,
                        fuzzy_duration: Optional[Tuple[float, ...]] = None) -> float:
        """
        Arc reliability R_ij^m = 1 - alpha_ij^m(t) (Eq.1).

        Time-dependent mode:
        - the baseline belief is adjusted by the bounded odds mapping
        - phi_bar is the exact piecewise integral of phi(t) over the expected
          uncertain traversal duration specified by fuzzy_duration.
        """
        if arrival_time is not None and start_time == 0.0:
            start_time = arrival_time
            end_time = arrival_time + 1.0

        # Config switch: when enable_time_varying_risk=False, treat as constant alpha
        if not time_dependent or not getattr(self.config, 'enable_time_varying_risk', True):
            return 1.0 - alpha

        if fuzzy_duration is not None:
            if len(fuzzy_duration) == 4:
                a, b, c, d = fuzzy_duration
                expected_duration = (a + b + c + d) / 4.0
            else:
                a, b, c = fuzzy_duration
                expected_duration = (a + 2.0 * b + c) / 4.0
            avg_phi = self._integrate_risk_multiplier(
                start_time, start_time + expected_duration
            )
            effective_alpha = self.time_adjusted_belief(alpha, avg_phi)
        else:
            t_end = end_time if end_time > start_time else start_time + 1.0
            avg_phi = self._integrate_risk_multiplier(start_time, t_end)
            effective_alpha = self.time_adjusted_belief(alpha, avg_phi)

        return 1.0 - effective_alpha

    def path_reliability(self, arc_reliabilities: list, node_reliabilities: list) -> float:
        """
        Plan risk reliability for a series uncertain system.
        """
        all_r = arc_reliabilities + node_reliabilities
        return min(all_r) if all_r else 1.0

    @staticmethod
    def time_adjusted_belief(baseline: float, multiplier: float) -> float:
        """Apply the bounded odds multiplier to a baseline accident belief."""
        baseline = min(1.0, max(0.0, baseline))
        multiplier = max(0.0, multiplier)
        denominator = 1.0 - baseline + multiplier * baseline
        return 0.0 if denominator == 0.0 else multiplier * baseline / denominator

    def od_reliability(self, batch_reliabilities: list) -> float:
        """
        O-D pair reliability R_od = min over batches (Eq.5a).

        Serial system across batches: any batch failure = O-D failure.
        """
        return min(batch_reliabilities) if batch_reliabilities else 1.0

    def network_reliability(self, od_reliabilities: list) -> float:
        """
        Network reliability R_net = min over O-D pairs (Eq.5b).
        """
        return min(od_reliabilities) if od_reliabilities else 1.0

    def comprehensive_reliability(self, risk_R: float, time_R: float) -> float:
        """
        Comprehensive reliability R_sys = min{R_risk, R_T} (Eq.5c).
        """
        return min(risk_R, time_R)

    def time_upper_constraint_value(self, fuzzy_vars: list, is_trapezoidal: list,
                                     alpha_T: float) -> float:
        """
        Compute the deterministic equivalent LHS coefficient sum for
        M{Z <= L} >= alpha_T (Eq.12).

        Z = s + sum(t~_ij * x_ij) + sum(tau~_i * y_i)
        Deterministic equivalent LHS:
            s + sum[(2a-1)*c + 2(1-a)*b] * x + sum[(2a-1)*d + 2(1-a)*c] * y

        Returns the per-unit coefficient for each fuzzy variable.
        """
        coeffs = []
        for fv, is_trap in zip(fuzzy_vars, is_trapezoidal):
            coeffs.append(fv.deterministic_equivalent_upper(alpha_T))
        return coeffs

    def time_lower_constraint_value(self, fuzzy_vars: list, is_trapezoidal: list,
                                     alpha_T: float) -> float:
        """
        Compute the deterministic equivalent RHS for M{Z >= E} >= alpha_T (Eq.15).
        """
        coeffs = []
        for fv, is_trap in zip(fuzzy_vars, is_trapezoidal):
            coeffs.append(fv.deterministic_equivalent_lower(alpha_T))
        return coeffs
