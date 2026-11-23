"""
Trust network — V1.1 §4.

Design rules enforced here:
  1. Raw-in / trust-gates-out (§4.1): estimates are computed ONLY from
     RawReading objects for neighbors. Passing a CorrectedValue raises
     TypeError, so the invariant can't silently regress.
  2. This module NEVER imports the env's parameters. It holds its own
     `AssumedParams` (what the trust network *believes* the physics is),
     separate from the env's ground truth. Use
     AssumedParams.from_env(p, calibration_error={...}) to build one with
     a deliberate mismatch.

Variables: T, H have two evidence sources (cross-zone + actuator history).
M (soil moisture) has actuator history only (no coupling, §2.4).
Light is not in the env yet, so it is not here yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

ZONE_IDS = ["zone_1", "zone_2", "zone_3"]
NEIGHBORS = {
    "zone_1": ["zone_2", "zone_3"],
    "zone_2": ["zone_1", "zone_3"],
    "zone_3": ["zone_1", "zone_2"],
}
VARS = ("T", "H", "M")
CZ_VARS = ("T", "H")  # variables with a cross-zone evidence source


# ---------------------------------------------------------------------------
# Type-level enforcement of raw-in / trust-gates-out
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawReading:
    value: float


@dataclass(frozen=True)
class CorrectedValue:
    value: float


def _require_raw(items):
    for it in items:
        if not isinstance(it, RawReading):
            raise TypeError(
                f"cross-zone estimate requires RawReading inputs, got {type(it).__name__}. "
                f"Raw-in/trust-gates-out invariant (V1.1 §4.1)."
            )


# ---------------------------------------------------------------------------
# Assumed physics (the trust network's OWN belief, not the env's truth)
# ---------------------------------------------------------------------------

_PARAM_NAMES = [
    "Q_heat", "C_thermal", "h_loss", "h_vent", "k_T", "T_ambient",
    "W_mist", "C_humidity", "g_loss", "g_vent", "k_H", "H_ambient",
    "alpha", "beta", "dt",
]


@dataclass
class AssumedParams:
    Q_heat: float
    C_thermal: float
    h_loss: float
    h_vent: float
    k_T: float
    T_ambient: float
    W_mist: float
    C_humidity: float
    g_loss: float
    g_vent: float
    k_H: float
    H_ambient: float
    alpha: float
    beta: float
    dt: float

    @classmethod
    def from_env(cls, env_params, calibration_error: dict | None = None):
        """
        Copy values from the env's params (duck-typed, no import of env),
        then scale the named ones by (1 + error). Example:
            AssumedParams.from_env(p, {"k_T": 0.5})   # k_T assumed 50% too high
        """
        vals = {n: float(getattr(env_params, n)) for n in _PARAM_NAMES}
        for name, err in (calibration_error or {}).items():
            if name not in vals:
                raise KeyError(name)
            vals[name] *= (1.0 + err)
        return cls(**vals)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrustConfig:
    alpha_trust: float = 0.1      # EMA constant — needs a sweep (V1.1 §4.5)
    trust_floor: float = 0.02
    init_trust: float = 1.0
    tail_ref: float = 0.05        # see HealthyResiduals.match_score
    eps: float = 1e-3
    horizon_s: float = 60.0       # horizon for coupling-signal confidence
    noise_floor: dict = field(default_factory=lambda: {"T": 0.1, "H": 0.5, "M": 0.5})
    conf_act: dict = field(default_factory=lambda: {"T": 0.5, "H": 0.5, "M": 0.5})  # placeholder


# ---------------------------------------------------------------------------
# Estimators (one-step physics predictors)
# ---------------------------------------------------------------------------

def _self_rate(var, x, u, p: AssumedParams):
    """d(var)/dt from this zone's own dynamics — actuators + self loss, NO neighbors."""
    if var == "T":
        heat = p.Q_heat if u["heat"] else 0.0
        loss = p.h_loss * (x - p.T_ambient)
        vent = p.h_vent * (x - p.T_ambient) if u["fan"] else 0.0
        return (heat - loss - vent) / p.C_thermal
    if var == "H":
        mist = p.W_mist if u["mist"] else 0.0
        loss = p.g_loss * (x - p.H_ambient)
        vent = p.g_vent * (x - p.H_ambient) if u["fan"] else 0.0
        return (mist - loss - vent) / p.C_humidity
    if var == "M":
        return p.alpha * (1.0 if u["water"] else 0.0) - p.beta * x
    raise KeyError(var)


def _clip(var, x):
    return float(np.clip(x, 0.0, 100.0)) if var in ("H", "M") else float(x)


def estimate_act(var, prev_est, u, p: AssumedParams) -> float:
    """Actuator-history estimate: previous estimate + modeled own-dynamics step."""
    return _clip(var, prev_est + p.dt * _self_rate(var, prev_est, u, p))


def estimate_cz(var, prev_est, u, neighbor_prev_raw, p: AssumedParams) -> float:
    """
    Cross-zone estimate: actuator-history step PLUS coupling to neighbors,
    computed from neighbors' RAW readings only (V1.1 §4.1).
    """
    _require_raw(neighbor_prev_raw)
    k, C = (p.k_T, p.C_thermal) if var == "T" else (p.k_H, p.C_humidity)
    coupling = -sum(k * (prev_est - n.value) for n in neighbor_prev_raw) / C
    return _clip(var, prev_est + p.dt * (_self_rate(var, prev_est, u, p) + coupling))


def confidence_cz(var, raw_i, raw_j, trust_j, p: AssumedParams, cfg: TrustConfig) -> float:
    """
    SNR-style confidence that neighbor j carries information about zone i.

    Unit-consistent version of V1.1 §4.2: signal = the shift (in the
    variable's own units) that coupling would produce over `horizon_s`,
    compared against the sensor noise floor. Weak coupling or a small
    inter-zone gap -> low confidence. Scaled by neighbor trust AFTER
    (trust gates influence, doesn't compute the estimate).
    """
    k, C = (p.k_T, p.C_thermal) if var == "T" else (p.k_H, p.C_humidity)
    signal = (k / C) * abs(raw_i - raw_j) * cfg.horizon_s
    return (signal / (signal + cfg.noise_floor[var])) * trust_j


def fuse(est_cz, conf_cz, est_act, conf_act, eps):
    """Inverse-variance-style fusion (V1.1 §4.4): precision = max(conf, eps)."""
    w_cz, w_act = max(conf_cz, eps), max(conf_act, eps)
    return (est_cz * w_cz + est_act * w_act) / (w_cz + w_act), w_cz / (w_cz + w_act)


# ---------------------------------------------------------------------------
# Healthy residual distribution (empirical, not Gaussian)
# ---------------------------------------------------------------------------

class HealthyResiduals:
    """Per-variable empirical distribution of residuals from fault-free runs."""

    def __init__(self, samples: dict | None = None):
        self._sorted = {}
        for var, arr in (samples or {}).items():
            self._sorted[var] = np.sort(np.asarray(arr, dtype=float))

    def survival(self, var, x) -> float:
        """Fraction of healthy residuals >= x."""
        arr = self._sorted[var]
        return float((len(arr) - np.searchsorted(arr, x, side="left")) / len(arr))

    def match_score(self, var, x, tail_ref: float) -> float:
        """
        1.0 while the residual sits inside the healthy bulk; falls to 0 as it
        moves into/past the healthy tail.

        NOTE (spec patch): V1.1 §4.5 writes MatchScore = ECDF(residual). Taken
        literally that is HIGH for large residuals, and 1 - ECDF is uniform on
        [0,1] for healthy data, so a perfectly healthy sensor would settle at
        trust ~0.5. Dividing the survival value by `tail_ref` (default 5%) and
        capping at 1 keeps healthy sensors near 1.0.
        """
        return min(1.0, self.survival(var, x) / tail_ref)


# ---------------------------------------------------------------------------
# Trust network
# ---------------------------------------------------------------------------

@dataclass
class TrustRecord:
    raw: float
    est_cz: float | None
    est_act: float
    conf_cz: float | None
    conf_act: float
    cz_weight: float | None
    combined: float
    residual: float
    match: float
    trust: float
    belief: float   # trust*raw + (1-trust)*combined: what the agent/explanations use
    raw_weight: float  # = trust; estimate weight is 1 - raw_weight


class TrustNetwork:
    def __init__(self, assumed: AssumedParams, healthy: HealthyResiduals | None = None,
                 cfg: TrustConfig | None = None):
        self.p = assumed
        self.healthy = healthy
        self.cfg = cfg or TrustConfig()

    def reset(self, raw: dict):
        """raw[zone][var] -> RawReading. Cold start: anchor estimates on first raw reading."""
        self.prev_raw = raw
        self.est = {z: {v: raw[z][v].value for v in VARS} for z in ZONE_IDS}
        self.trust = {z: {v: self.cfg.init_trust for v in VARS} for z in ZONE_IDS}

    def step(self, raw: dict, actions: dict) -> dict:
        """
        raw: readings at time t. actions: the actions applied during the step
        that PRODUCED these readings (i.e. what you passed to env.step).
        Neighbor trust used for confidence is last step's (no same-step loop).
        """
        cfg, p = self.cfg, self.p
        u = {z: dict(heat=bool(a[0]), fan=bool(a[1]), mist=bool(a[2]), water=bool(a[3]))
             for z, a in actions.items()}
        records, new_est, new_trust = {}, {z: {} for z in ZONE_IDS}, {z: {} for z in ZONE_IDS}

        for z in ZONE_IDS:
            records[z] = {}
            for v in VARS:
                prev = self.est[z][v]
                act = estimate_act(v, prev, u[z], p)
                if v in CZ_VARS:
                    nb_prev = [self.prev_raw[n][v] for n in NEIGHBORS[z]]
                    cz = estimate_cz(v, prev, u[z], nb_prev, p)
                    conf_cz = float(np.mean([
                        confidence_cz(v, raw[z][v].value, raw[n][v].value,
                                      self.trust[n][v], p, cfg) for n in NEIGHBORS[z]]))
                    combined, w = fuse(cz, conf_cz, act, cfg.conf_act[v], cfg.eps)
                else:
                    cz = conf_cz = w = None
                    combined = act

                residual = abs(raw[z][v].value - combined)
                match = (1.0 if self.healthy is None
                         else self.healthy.match_score(v, residual, cfg.tail_ref))
                tr = (1 - cfg.alpha_trust) * self.trust[z][v] + cfg.alpha_trust * match
                tr = float(np.clip(tr, cfg.trust_floor, 1.0))

                # belief = trust-gated blend of raw and combined. It is for the
                # observation/explanations ONLY; estimator anchors stay on `combined`,
                # so a faulty raw can't leak back into the estimates.
                belief = tr * raw[z][v].value + (1.0 - tr) * combined
                new_est[z][v], new_trust[z][v] = combined, tr
                records[z][v] = TrustRecord(raw[z][v].value, cz, act, conf_cz,
                                            cfg.conf_act[v], w, combined, residual,
                                            match, tr, belief, tr)

        self.est, self.trust, self.prev_raw = new_est, new_trust, raw
        return records