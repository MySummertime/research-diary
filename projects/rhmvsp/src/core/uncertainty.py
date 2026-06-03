from __future__ import annotations
"""
Uncertainty Theory Engine (§2, §3.2).

Implements fuzzy variables (triangular/trapezoidal) and uncertainty measures M{...}.
Based on Liu (2007/2015) Uncertainty Theory.

When alpha_T > 0.5, uncertain measure M{...} is mathematically equivalent to
credibility measure Cr{...} (Liu 2007, §1.12).

Time-varying risk (§2.2): Both accident probability alpha_ij^m(t) and consequence
C_ij^m(t) are time-dependent. alpha scales with a traffic/fatigue multiplier
phi(t) and C scales with a population exposure multiplier psi(t) reflecting
diurnal activity patterns (Fabiano et al. 2002, Abkowitz & Cheng 1988).
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TriangularFuzzy:
    """
    Triangular fuzzy variable t~ = (a, b, c) where a <= b <= c.

    Membership function:
        mu(x) = (x-a)/(b-a)  if a <= x <= b
        mu(x) = (c-x)/(c-b)  if b <= x <= c
        mu(x) = 0            otherwise

    Used for: transport time on arcs (§3.2, Eq.7).
    """
    a: float  # Minimum
    b: float  # Most likely
    c: float  # Maximum

    def __post_init__(self):
        assert self.a <= self.b <= self.c, f"Invalid triangular fuzzy: ({self.a}, {self.b}, {self.c})"

    def membership(self, x: float) -> float:
        """Membership function mu(x)."""
        if x <= self.a or x >= self.c:
            return 0.0
        elif x <= self.b:
            return (x - self.a) / (self.b - self.a) if self.b > self.a else 1.0
        else:
            return (self.c - x) / (self.c - self.b) if self.c > self.b else 1.0

    def uncertain_measure_leq(self, x: float) -> float:
        """
        M{t~ <= x} - Uncertain measure that the fuzzy variable is <= x.
        For alpha > 0.5 (right tail), deterministic equivalent for M{t~ <= L} >= alpha:
            L >= (2*alpha-1)*c + 2*(1-alpha)*b
        """
        if x <= self.a:
            return 0.0
        elif x >= self.c:
            return 1.0
        elif x <= self.b:
            mu = (x - self.a) / (self.b - self.a) if self.b > self.a else 1.0
            return 0.5 * mu
        else:  # b < x < c
            mu = (self.c - x) / (self.c - self.b) if self.c > self.b else 1.0
            return 1.0 - 0.5 * mu

    def uncertain_measure_geq(self, x: float) -> float:
        """M{t~ >= x} = 1 - M{t~ <= x} (by self-duality of uncertain measure)."""
        return 1.0 - self.uncertain_measure_leq(x)

    def deterministic_equivalent_upper(self, alpha: float) -> float:
        """
        Deterministic equivalent for M{t~ <= L} >= alpha (Eq.12 / A.1).

        When alpha > 0.5, equivalent to: L >= (2*alpha-1)*c + 2*(1-alpha)*b
        Returns the minimum L satisfying the constraint.
        """
        # Support full alpha range (0,1]. For alpha < 0.5 the required L
        # lies on the left slope [a, b] of the triangular membership where
        # M{t~ <= x} = 0.5 * (x - a) / (b - a).
        if alpha < 0.5:
            # Solve 0.5 * (x - a) / (b - a) = alpha => x = a + 2*alpha*(b-a)
            return self.a + 2.0 * alpha * (self.b - self.a)
        # alpha >= 0.5: use right-tail deterministic equivalent
        return (2.0 * alpha - 1.0) * self.c + 2.0 * (1.0 - alpha) * self.b

    def deterministic_equivalent_lower(self, alpha: float) -> float:
        """
        Deterministic equivalent for M{t~ >= E} >= alpha (Eq.15 / A.2).

        When alpha > 0.5, equivalent to: E <= (2*alpha-1)*a + 2*(1-alpha)*b
        Returns the maximum E satisfying the constraint.
        """
        # Use symmetry: M{t~ >= E} >= alpha  <=>  M{t~ <= E} <= 1 - alpha
        # So deterministic-equivalent lower bound equals the deterministic
        # equivalent upper bound for (1 - alpha).
        return self.deterministic_equivalent_upper(1.0 - alpha)

    def mean(self) -> float:
        """
        Expected value under uncertainty theory (Liu 2007, Theorem 1.23).
        E[xi] = integral_0^inf M{xi>=r}dr - integral_{-inf}^0 M{xi<=r}dr
        For triangular (a,b,c): E = (a + 2b + c) / 4
        """
        return (self.a + 2 * self.b + self.c) / 4.0

    def __repr__(self):
        return f"TriFuzzy({self.a:.2f}, {self.b:.2f}, {self.c:.2f})"


@dataclass
class TrapezoidalFuzzy:
    """
    Trapezoidal fuzzy variable tau~ = (a, b, c, d) where a <= b <= c <= d.

    Membership function:
        mu(x) = (x-a)/(b-a)  if a <= x <= b
        mu(x) = 1            if b <= x <= c
        mu(x) = (d-x)/(d-c)  if c <= x <= d
        mu(x) = 0            otherwise

    Used for: transport time (§3.2, Eq.7) and transfer time (§3.2, Eq.8).
    """
    a: float  # Minimum
    b: float  # Lower mode start
    c: float  # Upper mode end
    d: float  # Maximum

    def __post_init__(self):
        assert self.a <= self.b <= self.c <= self.d, \
            f"Invalid trapezoidal fuzzy: ({self.a}, {self.b}, {self.c}, {self.d})"

    def membership(self, x: float) -> float:
        """Membership function mu(x)."""
        if x <= self.a or x >= self.d:
            return 0.0
        elif x <= self.b:
            return (x - self.a) / (self.b - self.a) if self.b > self.a else 1.0
        elif x <= self.c:
            return 1.0
        else:
            return (self.d - x) / (self.d - self.c) if self.d > self.c else 1.0

    def uncertain_measure_leq(self, x: float) -> float:
        """
        M{tau~ <= x} for trapezoidal fuzzy variable.

        Credibility: Cr{tau~ <= x} =
            0                           if x <= a
            (x-a) / (2*(b-a))           if a < x <= b
            0.5                         if b < x <= c  (plateau: mu=1 everywhere)
            1 - (d-x) / (2*(d-c))       if c < x <= d
            1                           if x >= d
        """
        if x <= self.a:
            return 0.0
        elif x >= self.d:
            return 1.0
        elif x <= self.b:
            return (x - self.a) / (2 * (self.b - self.a)) if self.b > self.a else 0.5
        elif x <= self.c:
            return 0.5
        else:
            return 1.0 - (self.d - x) / (2 * (self.d - self.c)) if self.d > self.c else 0.5

    def uncertain_measure_geq(self, x: float) -> float:
        """M{tau~ >= x} = 1 - M{tau~ <= x}."""
        return 1.0 - self.uncertain_measure_leq(x)

    def deterministic_equivalent_upper(self, alpha: float) -> float:
        """
        Deterministic equivalent for M{tau~ <= L} >= alpha.

        For trapezoidal, when alpha > 0.5 (right tail):
            L >= (2*alpha-1)*d + 2*(1-alpha)*c
        """
        # For alpha < 0.5, the required L lies on the rising slope [a,b]
        # where M{tau~ <= x} = (x - a) / (2*(b-a)). Solve for x:
        #   (x - a) / (2*(b-a)) = alpha  => x = a + 2*alpha*(b-a)
        # For alpha >= 0.5, use right-tail formula to be consistent with
        # compute_time_reliability_od which always uses the right-branch.
        if alpha < 0.5:
            return self.a + 2.0 * alpha * (self.b - self.a)
        # alpha >= 0.5: right-tail formula
        return (2.0 * alpha - 1.0) * self.d + 2.0 * (1.0 - alpha) * self.c

    def deterministic_equivalent_lower(self, alpha: float) -> float:
        """
        Deterministic equivalent for M{tau~ >= E} >= alpha.

        For trapezoidal, when alpha > 0.5 (left tail):
            E <= (2*alpha-1)*a + 2*(1-alpha)*b
        """
        return self.deterministic_equivalent_upper(1.0 - alpha)

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

    def expected_consequence(self, C: float) -> float:
        """E^M[r~] = alpha * C (Eq.3)."""
        return self.alpha * C


# ============================================================
# Time-varying Risk Multipliers (§2.2, S2-2 fix)
# ============================================================

def _traffic_multiplier(hour_of_day: float, risk_scale: float = 1.0) -> float:
    """
    Time-varying accident probability multiplier phi(t) (S2-2).

    risk_scale in [0,1]: 0 → constant 1.0, 1 → full time-varying formula.
    phi_eff(t) = 1 + risk_scale * (phi(t) - 1).

    Models elevated hazmat accident risk during:
    - Night/early morning (22:00-06:00): multiplier=1.5 (fatigue, low visibility)
    - Morning/evening rush (07:00-09:00, 17:00-19:00): multiplier=1.3 (congestion)
    - Daytime off-peak (09:00-17:00, 19:00-22:00): multiplier=1.0 (normal)

    Reference: Abkowitz & Cheng (1988), Fabiano et al. (2002).
    """
    h = hour_of_day % 24.0
    if h < 6.0:
        raw = 1.5
    elif h < 7.0:
        raw = 1.5 - 0.2 * (h - 6.0)
    elif h < 9.0:
        raw = 1.3
    elif h < 17.0:
        raw = 1.0
    elif h < 19.0:
        raw = 1.3
    elif h < 22.0:
        raw = 1.0
    else:
        raw = 1.0 + 0.5 * ((h - 22.0) / 2.0)
    return 1.0 + risk_scale * (raw - 1.0)


def _population_exposure_multiplier(hour_of_day: float, risk_scale: float = 1.0) -> float:
    """
    Time-varying consequence multiplier psi(t) (S2-2).

    risk_scale in [0,1]: 0 → constant 1.0, 1 → full time-varying formula.
    psi_eff(t) = 1 + risk_scale * (psi(t) - 1).

    Hazmat accident consequence C(t) scales with population exposure:
    - Daytime (08:00-20:00): peak activity, psi=1.0 (reference)
    - Evening transition (20:00-22:00): psi=0.8
    - Late night (22:00-06:00): low exposure, psi=0.5 (fewer people outdoors)
    - Early morning (06:00-08:00): psi=0.7

    Reference: Erkut & Verter (1998), Batta & Chiu (1988).
    """
    h = hour_of_day % 24.0
    if h < 6.0:
        raw = 0.5
    elif h < 8.0:
        raw = 0.5 + 0.25 * ((h - 6.0) / 2.0)
    elif h < 20.0:
        raw = 1.0
    elif h < 22.0:
        raw = 1.0 - 0.25 * ((h - 20.0) / 2.0)
    else:
        raw = 0.75 - 0.25 * ((h - 22.0) / 2.0)
    return 1.0 + risk_scale * (raw - 1.0)


def time_varying_alpha(alpha_base: float, start_time: float, end_time: float,
                       risk_scale: float = 1.0) -> float:
    """
    Compute time-averaged accident probability over transit window [start_time, end_time].

    alpha_ij^m(t) = alpha_base * phi(t)

    Uses exact piecewise integration of phi(t) over the transit window.
    Returns effective alpha for the transit.
    """
    if start_time >= end_time:
        return min(0.95, alpha_base * _traffic_multiplier(start_time, risk_scale))

    total_integral = 0.0
    current_t = start_time
    while current_t < end_time:
        day = int(current_t // 24)
        h = current_t - day * 24.0
        # Find end of current piecewise segment
        breakpoints = [6.0, 7.0, 9.0, 17.0, 19.0, 22.0, 24.0]
        next_h = next((bp for bp in breakpoints if bp > h), 24.0)
        seg_end = min(end_time, day * 24.0 + next_h)
        step = seg_end - current_t
        # Trapezoidal integration within segment
        phi_start = _traffic_multiplier(current_t, risk_scale)
        phi_end = _traffic_multiplier(seg_end - 1e-9, risk_scale)
        total_integral += (phi_start + phi_end) / 2.0 * step
        current_t = seg_end

    avg_multiplier = total_integral / (end_time - start_time)
    return min(0.95, alpha_base * avg_multiplier)


def time_varying_consequence(C_base: float, hour_of_day: float,
                             risk_scale: float = 1.0) -> float:
    """
    Compute time-dependent consequence C(t) = C_base * psi(t).

    Consequence reflects population exposure at time of accident.
    """
    return C_base * _population_exposure_multiplier(hour_of_day, risk_scale)



class UncertaintyEngine:
    """
    Central engine for all uncertainty computations.

    Creates fuzzy variables from network parameters and computes
    deterministic equivalents for reliability constraints.
    """

    def __init__(self, config):
        self.config = config
        self.risk_scale = getattr(config, 'risk_scale', 1.0)

    def make_transport_time(self, distance: float, mode: int) -> TrapezoidalFuzzy:
        """Create transport time fuzzy variable for an arc (§3.2, Eq.7).
        Returns TrapezoidalFuzzy for all modes (road: c=d equivalent to triangular).
        """
        dt = getattr(self.config, 'delta_T', 0.2)
        scale = dt / 0.2 if dt > 0 else 0.001

        if mode == 1:  # Road: triangular equivalent (c=d)
            base = distance / self.config.road_speed
            a_ratio = max(0.1, 1.0 - 0.2 * scale)
            c_ratio = 1.0 + 0.5 * scale
            return TrapezoidalFuzzy(
                a=base * a_ratio,
                b=base,
                c=base,  # plateau collapses to point (triangular)
                d=base * c_ratio,
            )
        else:  # Rail: trapezoidal with plateau (§3.2, #9)
            base = distance / self.config.rail_speed
            ratios = getattr(self.config, 'rail_fuzzy_time_ratio', (0.9, 1.0, 1.0, 1.3))
            return TrapezoidalFuzzy(
                a=base * ratios[0],
                b=base * ratios[1],
                c=base * ratios[2],
                d=base * ratios[3],
            )

    def make_transfer_time(self) -> TrapezoidalFuzzy:
        """Create transfer time fuzzy variable (§3.2, Eq.8)."""
        dt = getattr(self.config, 'delta_T', 0.2)
        scale = dt / 0.2 if dt > 0 else 0.001
        base = 1.0  # Base transfer time is ~1 hour
        return TrapezoidalFuzzy(
            a=base * max(0.1, 1.0 - 0.5 * scale), 
            b=base, 
            c=base * (1.0 + 1.0 * scale), 
            d=base * (1.0 + 3.0 * scale)
        )

    def _integrate_risk_multiplier(self, t1: float, t2: float) -> float:
        """
        Calculate time-averaged accident probability multiplier phi(t) over [t1, t2].
        Uses exact piecewise integration of the traffic-dependent multiplier.
        """
        if t1 >= t2:
            return _traffic_multiplier(t1, self.risk_scale)

        total_integral = 0.0
        current_t = t1
        while current_t < t2:
            day = int(current_t // 24)
            h = current_t - day * 24.0
            breakpoints = [6.0, 7.0, 9.0, 17.0, 19.0, 22.0, 24.0]
            next_h = next((bp for bp in breakpoints if bp > h), 24.0)
            seg_end = min(t2, day * 24.0 + next_h)
            step = seg_end - current_t
            phi_s = _traffic_multiplier(current_t, self.risk_scale)
            phi_e = _traffic_multiplier(seg_end - 1e-9, self.risk_scale)
            total_integral += (phi_s + phi_e) / 2.0 * step
            current_t = seg_end

        return total_integral / (t2 - t1)

    def _integrate_risk_consequence_product(self, t1: float, t2: float) -> float:
        """
        Calculate time-averaged product of phi(t) and psi(t) over [t1, t2].
        Uses exact piecewise integration (Simpson's 1/3 rule) of the multipliers.
        """
        if t1 >= t2:
            return _traffic_multiplier(t1, self.risk_scale) * _population_exposure_multiplier(t1, self.risk_scale)

        total_integral = 0.0
        current_t = t1
        while current_t < t2:
            day = int(current_t // 24)
            h = current_t - day * 24.0
            breakpoints = [6.0, 7.0, 8.0, 9.0, 17.0, 19.0, 20.0, 22.0, 24.0]
            next_h = next((bp for bp in breakpoints if bp > h), 24.0)
            seg_end = min(t2, day * 24.0 + next_h)
            step = seg_end - current_t
            
            mid_t_seg = (current_t + seg_end) / 2.0
            v_start = _traffic_multiplier(current_t, self.risk_scale) * _population_exposure_multiplier(current_t, self.risk_scale)
            v_mid = _traffic_multiplier(mid_t_seg, self.risk_scale) * _population_exposure_multiplier(mid_t_seg, self.risk_scale)
            v_end = _traffic_multiplier(seg_end - 1e-9, self.risk_scale) * _population_exposure_multiplier(seg_end - 1e-9, self.risk_scale)
            
            total_integral += (step / 6.0) * (v_start + 4.0 * v_mid + v_end)
            current_t = seg_end

        return total_integral / (t2 - t1)

    def arc_reliability(self, alpha: float, start_time: float = 0.0, end_time: float = 0.0,
                        time_dependent: bool = False, arrival_time: Optional[float] = None,
                        fuzzy_duration: Optional[Tuple[float, ...]] = None,
                        C_consequence: float = 100.0) -> float:
        """
        Arc reliability R_ij^m = 1 - alpha_ij^m(t) (Eq.1).

        Time-dependent mode (S2-2):
        - alpha_ij^m(t) = alpha_base * phi(t)  [probability multiplier]
        - C_ij^m(t)     = C_base * psi(t)      [consequence multiplier]

        Risk measure = alpha(t) * C(t) / C_ref, normalized so that
        alpha_ij^m used for reliability = effective probability only.

        If fuzzy_duration=(a,b,c) is provided, integrates over triangular fuzzy
        arrival times using 10-point quadrature.
        """
        if arrival_time is not None and start_time == 0.0:
            start_time = arrival_time
            end_time = arrival_time + 1.0

        # Config switch: when enable_time_varying_risk=False, treat as constant alpha
        if not time_dependent or not getattr(self.config, 'enable_time_varying_risk', True):
            return 1.0 - alpha

        if fuzzy_duration is not None:
            # #9: Support both triangular (a,b,c) and trapezoidal (a,b,c,d) fuzzy durations
            if len(fuzzy_duration) == 4:
                a, b, c, d = fuzzy_duration
            else:
                a, b, c = fuzzy_duration
                d = c  # triangular: d=c
            n_steps = 10
            total_alpha = 0.0
            for k in range(1, n_steps + 1):
                r = (k - 0.5) / n_steps
                # Alpha-cut of trapezoidal fuzzy duration
                if r <= 0.5:
                    t_r = a + 2.0 * r * (b - a)
                elif r <= 0.5 + 1e-9:
                    t_r = b  # plateau start
                else:
                    # Map r from (0.5, 1] to plateau + right slope
                    r_adj = (r - 0.5) * 2.0  # [0, 1]
                    if r_adj <= 0.5:
                        t_r = b + 2.0 * r_adj * (c - b)
                    else:
                        t_r = c + 2.0 * (r_adj - 0.5) * (d - c)
                avg_phi = self._integrate_risk_multiplier(start_time, start_time + t_r)
                effective_alpha_k = alpha * avg_phi
                total_alpha += effective_alpha_k
            effective_alpha = min(0.95, total_alpha / n_steps)
        else:
            t_end = end_time if end_time > start_time else start_time + 1.0
            avg_phi = self._integrate_risk_multiplier(start_time, t_end)
            # Effective alpha accounts for probability time-variation only
            effective_alpha = min(0.95, alpha * avg_phi)

        return 1.0 - effective_alpha

    def path_reliability(self, arc_reliabilities: list, node_reliabilities: list) -> float:
        """
        Path reliability R(P) = min of all component reliabilities (Eq.4).

        Serial system: min over all arcs and transfer nodes on the path.
        """
        all_r = arc_reliabilities + node_reliabilities
        return min(all_r) if all_r else 1.0

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




