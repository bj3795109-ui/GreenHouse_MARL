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
import math
import numpy as np

ZONE_IDS = ["zone_1", "zone_2", "zone_3"]
NEIGHBORS = {
    "zone_1": ["zone_2", "zone_3"],
    "zone_2": ["zone_1", "zone_3"],
    "zone_3": ["zone_1", "zone_2"],
}
VARS = ("T", "H", "M")
CZ_VARS = ("T", "H")  # variables with a cross-zone evidence source


def _is_finite_number(val) -> bool:
    if isinstance(val, bool) or val is None:
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _is_valid_raw_reading(item) -> bool:
    return isinstance(item, RawReading) and _is_finite_number(item.value)


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
        if not _is_finite_number(it.value):
            raise ValueError(
                f"cross-zone estimate requires finite RawReading values, got {it.value!r}."
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
    if not _is_finite_number(prev_est):
        raise ValueError(f"estimate_act requires finite prev_est, got {prev_est!r}")
    return _clip(var, prev_est + p.dt * _self_rate(var, prev_est, u, p))


def estimate_cz(var, prev_est, u, neighbor_prev_raw, p: AssumedParams) -> float:
    """
    Cross-zone estimate: actuator-history step PLUS coupling to neighbors,
    computed from neighbors' RAW readings only (V1.1 §4.1).
    """
    _require_raw(neighbor_prev_raw)
    if not _is_finite_number(prev_est):
        raise ValueError(f"estimate_cz requires finite prev_est, got {prev_est!r}")
    k, C = (p.k_T, p.C_thermal) if var == "T" else (p.k_H, p.C_humidity)
    coupling = -sum(k * (prev_est - float(n.value)) for n in neighbor_prev_raw) / C
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
    if not (_is_finite_number(raw_i) and _is_finite_number(raw_j) and _is_finite_number(trust_j)):
        raise ValueError(
            f"confidence_cz requires finite inputs, got raw_i={raw_i!r}, raw_j={raw_j!r}, trust_j={trust_j!r}"
        )
    k, C = (p.k_T, p.C_thermal) if var == "T" else (p.k_H, p.C_humidity)
    signal = (k / C) * abs(float(raw_i) - float(raw_j)) * cfg.horizon_s
    return (signal / (signal + cfg.noise_floor[var])) * float(trust_j)


def fuse(est_cz, conf_cz, est_act, conf_act, eps):
    """Inverse-variance-style fusion (V1.1 §4.4): precision = max(conf, eps)."""
    if not (_is_finite_number(est_cz) and _is_finite_number(conf_cz)
            and _is_finite_number(est_act) and _is_finite_number(conf_act)
            and _is_finite_number(eps)):
        raise ValueError(
            f"fuse requires finite inputs, got est_cz={est_cz!r}, conf_cz={conf_cz!r}, "
            f"est_act={est_act!r}, conf_act={conf_act!r}, eps={eps!r}"
        )
    w_cz, w_act = max(float(conf_cz), float(eps)), max(float(conf_act), float(eps))
    return (float(est_cz) * w_cz + float(est_act) * w_act) / (w_cz + w_act), w_cz / (w_cz + w_act)


# ---------------------------------------------------------------------------
# Healthy residual distribution (empirical, not Gaussian)
# ---------------------------------------------------------------------------

class HealthyResiduals:
    """Per-variable empirical distribution of residuals from fault-free runs."""

    def __init__(self, samples: dict | None = None):
        self._sorted = {}
        for var, arr in (samples or {}).items():
            vec = np.asarray(arr, dtype=float)
            if not np.all(np.isfinite(vec)):
                raise ValueError(f"HealthyResiduals samples for '{var}' contain non-finite values")
            self._sorted[var] = np.sort(vec)

    def survival(self, var, x) -> float:
        """Fraction of healthy residuals >= x."""
        if not _is_finite_number(x):
            return 0.0
        arr = self._sorted.get(var)
        if arr is None or len(arr) == 0:
            return 0.0
        return float((len(arr) - np.searchsorted(arr, float(x), side="left")) / len(arr))

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
        if not _is_finite_number(tail_ref) or float(tail_ref) <= 0.0:
            raise ValueError(f"tail_ref must be > 0 and finite, got {tail_ref!r}")
        if not _is_finite_number(x):
            return 0.0
        return min(1.0, self.survival(var, x) / float(tail_ref))


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

    def _extract_raw_item(self, raw: dict, z: str, v: str) -> tuple[bool, float]:
        item = (raw.get(z) or {}).get(v) if isinstance(raw, dict) else None
        if item is not None and not isinstance(item, RawReading):
            raise TypeError(
                f"TrustNetwork requires RawReading inputs, got {type(item).__name__} for {z}.{v}. "
                f"Raw-in/trust-gates-out invariant (V1.1 §4.1)."
            )
        if _is_valid_raw_reading(item):
            return True, float(item.value)
        if isinstance(item, RawReading) and not isinstance(item.value, bool):
            try:
                return False, float(item.value)
            except (TypeError, ValueError):
                pass
        return False, float("nan")

    def reset(self, raw: dict):
        """raw[zone][var] -> RawReading. Cold start: anchor estimates on first valid raw reading."""
        self.prev_raw = {z: {} for z in ZONE_IDS}
        self._raw_valid = {z: {} for z in ZONE_IDS}
        self.est = {z: {} for z in ZONE_IDS}
        self._est_valid = {z: {} for z in ZONE_IDS}
        self.trust = {z: {} for z in ZONE_IDS}

        for z in ZONE_IDS:
            for v in VARS:
                ok, val = self._extract_raw_item(raw, z, v)
                if ok:
                    self.prev_raw[z][v] = RawReading(val)
                    self._raw_valid[z][v] = True
                    self.est[z][v] = val
                    self._est_valid[z][v] = True
                    self.trust[z][v] = self.cfg.init_trust
                else:
                    self.prev_raw[z][v] = RawReading(float("nan"))
                    self._raw_valid[z][v] = False
                    self.est[z][v] = float("nan")
                    self._est_valid[z][v] = False
                    self.trust[z][v] = self.cfg.trust_floor

    def step(self, raw: dict, actions: dict) -> dict:
        """
        raw: readings at time t. actions: the actions applied during the step
        that PRODUCED these readings (i.e. what you passed to env.step).
        Neighbor trust used for confidence is last step's (no same-step loop).

        Invalid raw readings (NaN/Inf/missing) are rejected before fusion,
        residual matching, belief blending, and neighbor cross-zone coupling so
        one zone's faulty sensor cannot contaminate healthy neighbors.
        """
        cfg, p = self.cfg, self.p
        u = {
            z: dict(heat=bool(a[0]), fan=bool(a[1]), mist=bool(a[2]), water=bool(a[3]))
            for z, a in actions.items()
        }

        parsed_raw = {z: {v: self._extract_raw_item(raw, z, v) for v in VARS} for z in ZONE_IDS}

        records = {}
        new_est = {z: {} for z in ZONE_IDS}
        new_est_valid = {z: {} for z in ZONE_IDS}
        new_trust = {z: {} for z in ZONE_IDS}
        new_prev_raw = {z: {} for z in ZONE_IDS}
        new_raw_valid = {z: {} for z in ZONE_IDS}

        for z in ZONE_IDS:
            records[z] = {}
            for v in VARS:
                raw_ok, raw_val = parsed_raw[z][v]
                est_ok = self._est_valid[z][v]

                if est_ok:
                    prev = self.est[z][v]
                    act = estimate_act(v, prev, u[z], p)

                    if v in CZ_VARS:
                        nb_prev = [
                            self.prev_raw[n][v]
                            for n in NEIGHBORS[z]
                            if self._raw_valid[n][v]
                        ]
                        cz = estimate_cz(v, prev, u[z], nb_prev, p) if nb_prev else None

                        if cz is not None and raw_ok:
                            valid_confs = [
                                confidence_cz(
                                    v,
                                    raw_val,
                                    parsed_raw[n][v][1],
                                    self.trust[n][v],
                                    p,
                                    cfg,
                                )
                                for n in NEIGHBORS[z]
                                if parsed_raw[n][v][0]
                            ]
                            conf_cz = float(np.mean(valid_confs)) if valid_confs else 0.0
                            combined, w = fuse(cz, conf_cz, act, cfg.conf_act[v], cfg.eps)
                        elif cz is not None:
                            conf_cz = 0.0
                            combined, w = fuse(cz, 0.0, act, cfg.conf_act[v], cfg.eps)
                        else:
                            conf_cz = None
                            w = None
                            combined = act
                    else:
                        cz = conf_cz = w = None
                        combined = act

                    if raw_ok:
                        residual = abs(raw_val - combined)
                        match = (
                            1.0
                            if self.healthy is None
                            else self.healthy.match_score(v, residual, cfg.tail_ref)
                        )
                        tr = (1.0 - cfg.alpha_trust) * self.trust[z][v] + cfg.alpha_trust * match
                        tr = float(np.clip(tr, cfg.trust_floor, 1.0))
                        raw_weight = tr
                        # belief = trust-gated blend of raw and combined; estimator anchors stay on `combined`
                        belief = tr * raw_val + (1.0 - tr) * combined
                    else:
                        residual = float("nan")
                        match = 0.0
                        tr = float(
                            np.clip(
                                (1.0 - cfg.alpha_trust) * self.trust[z][v],
                                cfg.trust_floor,
                                1.0,
                            )
                        )
                        raw_weight = 0.0
                        belief = combined

                    new_est[z][v] = combined
                    new_est_valid[z][v] = True
                    new_trust[z][v] = tr

                elif raw_ok:
                    # Cold-anchor onto first valid reading after an invalid initialization
                    act = raw_val
                    cz = conf_cz = w = None
                    combined = raw_val
                    residual = 0.0
                    match = 1.0
                    tr = float(
                        np.clip(
                            (1.0 - cfg.alpha_trust) * self.trust[z][v] + cfg.alpha_trust * match,
                            cfg.trust_floor,
                            1.0,
                        )
                    )
                    raw_weight = tr
                    belief = raw_val
                    new_est[z][v] = combined
                    new_est_valid[z][v] = True
                    new_trust[z][v] = tr

                else:
                    # Neither valid prior anchor nor valid reading: do not fabricate plausible values
                    act = float("nan")
                    cz = conf_cz = w = None
                    combined = float("nan")
                    residual = float("nan")
                    match = 0.0
                    tr = cfg.trust_floor
                    raw_weight = 0.0
                    belief = float("nan")
                    new_est[z][v] = float("nan")
                    new_est_valid[z][v] = False
                    new_trust[z][v] = tr

                new_prev_raw[z][v] = RawReading(raw_val) if raw_ok else RawReading(float("nan"))
                new_raw_valid[z][v] = raw_ok

                records[z][v] = TrustRecord(
                    raw_val,
                    cz,
                    act,
                    conf_cz,
                    cfg.conf_act[v],
                    w,
                    combined,
                    residual,
                    match,
                    tr,
                    belief,
                    raw_weight,
                )

        self.est = new_est
        self._est_valid = new_est_valid
        self.trust = new_trust
        self.prev_raw = new_prev_raw
        self._raw_valid = new_raw_valid
        return records