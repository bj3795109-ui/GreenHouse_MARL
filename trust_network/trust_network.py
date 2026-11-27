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

import math
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
    """One sensor sample. value is None (or non-finite) for a dropout."""
    value: float | None


def _missing(r: "RawReading") -> bool:
    return r.value is None or not math.isfinite(r.value)


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
    raw: float | None
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
    def __init__(
        self,
        assumed: AssumedParams,
        healthy: HealthyResiduals | None = None,
        cfg: TrustConfig | None = None,
    ):
        self.p = assumed
        self.healthy = healthy
        self.cfg = cfg or TrustConfig()

    def _extract_raw_item(
        self, raw: dict, z: str, v: str
    ) -> tuple[str, float | None]:

        zone_data = raw.get(z) if isinstance(raw, dict) else None
        item = zone_data.get(v) if isinstance(zone_data, dict) else None

        if item is None:
            return "missing", None

        if not isinstance(item, RawReading):
            raise TypeError(
                f"TrustNetwork requires RawReading inputs, got "
                f"{type(item).__name__} for {z}.{v}. "
                "Raw-in/trust-gates-out invariant (V1.1 §4.1)."
            )

        if item.value is None:
            return "missing", None

        if isinstance(item.value, bool):
            return "invalid", float("nan")

        try:
            value = float(item.value)
        except (TypeError, ValueError):
            return "invalid", float("nan")

        if not math.isfinite(value):
            return "invalid", value

        return "valid", value

    def reset(self, raw: dict):
        """
        Initialize the network.

        Missing initial readings are rejected.
        Non-finite readings are allowed but remain unanchored: such a sensor enters
        probation (trust starts at the floor and must climb past the SensorInterface's
        trust threshold before it is TRUSTED). A biased FIRST valid reading cannot be
        detected without redundancy, so fault scenarios must not start at t=0
        (see fault_injection.scenario.validate_scenario).
        """

        self.prev_raw = {z: {} for z in ZONE_IDS}
        self._raw_valid = {z: {} for z in ZONE_IDS}
        self.est = {z: {} for z in ZONE_IDS}
        self._est_valid = {z: {} for z in ZONE_IDS}
        self.trust = {z: {} for z in ZONE_IDS}

        for z in ZONE_IDS:
            for v in VARS:

                status, val = self._extract_raw_item(raw, z, v)

                if status == "missing":
                    raise ValueError(
                        f"Missing initial sensor reading for {z}.{v}"
                    )

                if status == "valid":
                    self.prev_raw[z][v] = RawReading(val)
                    self._raw_valid[z][v] = True

                    self.est[z][v] = val
                    self._est_valid[z][v] = True

                    self.trust[z][v] = self.cfg.init_trust

                else:
                    self.prev_raw[z][v] = RawReading(val)
                    self._raw_valid[z][v] = False

                    self.est[z][v] = float("nan")
                    self._est_valid[z][v] = False

                    self.trust[z][v] = self.cfg.trust_floor

    def step(self, raw: dict, actions: dict) -> dict:
        """
        Process one timestep.

        Valid readings:
          - Participate in trust evaluation and belief blending.
          - Update the trust EMA.

        Invalid readings (NaN/Inf):
          - Are excluded from fusion and cross-zone coupling.
          - Decrease trust.
          - Preserve their original value in TrustRecord.raw.

        Missing readings (dropout):
          - Are excluded from fusion and cross-zone coupling.
          - Do not change trust.
          - Are represented by None in TrustRecord.raw.

        Estimates continue propagating from previous valid estimates.
        """

        cfg, p = self.cfg, self.p

        u = {
            z: {
                "heat": bool(a[0]),
                "fan": bool(a[1]),
                "mist": bool(a[2]),
                "water": bool(a[3]),
            }
            for z, a in actions.items()
        }

        parsed_raw = {
            z: {
                v: self._extract_raw_item(raw, z, v)
                for v in VARS
            }
            for z in ZONE_IDS
        }

        records = {}

        new_est = {z: {} for z in ZONE_IDS}
        new_est_valid = {z: {} for z in ZONE_IDS}

        new_trust = {z: {} for z in ZONE_IDS}

        new_prev_raw = {z: {} for z in ZONE_IDS}
        new_raw_valid = {z: {} for z in ZONE_IDS}

        for z in ZONE_IDS:
            records[z] = {}

            for v in VARS:

                raw_status, raw_val = parsed_raw[z][v]

                raw_ok = raw_status == "valid"
                raw_missing = raw_status == "missing"
                raw_invalid = raw_status == "invalid"

                est_ok = self._est_valid[z][v]

                # ======================================================
                # Existing valid estimate
                # ======================================================

                if est_ok:

                    prev = self.est[z][v]

                    act = estimate_act(v, prev, u[z], p)

                    # --------------------------------------------------
                    # Cross-zone estimation
                    # --------------------------------------------------

                    if v in CZ_VARS:

                        nb_prev = [
                            self.prev_raw[n][v]
                            for n in NEIGHBORS[z]
                            if (
                                self._raw_valid[n][v]
                                and parsed_raw[n][v][0] == "valid"
                            )
                        ]

                        cz = (
                            estimate_cz(
                                v, prev, u[z], nb_prev, p
                            )
                            if nb_prev
                            else None
                        )

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
                                if parsed_raw[n][v][0] == "valid"
                                and self._raw_valid[n][v]
                            ]

                            conf_cz = (
                                float(np.mean(valid_confs))
                                if valid_confs
                                else 0.0
                            )

                            combined, w = fuse(
                                cz,
                                conf_cz,
                                act,
                                cfg.conf_act[v],
                                cfg.eps,
                            )

                        elif cz is not None:

                            conf_cz = 0.0

                            combined, w = fuse(
                                cz,
                                0.0,
                                act,
                                cfg.conf_act[v],
                                cfg.eps,
                            )

                        else:

                            conf_cz = None
                            w = None
                            combined = act

                    else:

                        cz = None
                        conf_cz = None
                        w = None

                        combined = act

                    # --------------------------------------------------
                    # Valid current reading
                    # --------------------------------------------------

                    if raw_ok:

                        residual = abs(raw_val - combined)

                        match = (
                            1.0
                            if self.healthy is None
                            else self.healthy.match_score(
                                v,
                                residual,
                                cfg.tail_ref,
                            )
                        )

                        tr = (
                            (1.0 - cfg.alpha_trust)
                            * self.trust[z][v]
                            + cfg.alpha_trust * match
                        )

                        tr = float(
                            np.clip(
                                tr,
                                cfg.trust_floor,
                                1.0,
                            )
                        )

                        raw_weight = tr

                        belief = (
                            tr * raw_val
                            + (1.0 - tr) * combined
                        )

                    # --------------------------------------------------
                    # Invalid or missing current reading
                    # --------------------------------------------------

                    else:

                        residual = float("nan")
                        match = 0.0

                        if raw_invalid:

                            # Invalid sensor evidence decreases trust.
                            tr = (
                                (1.0 - cfg.alpha_trust)
                                * self.trust[z][v]
                                + cfg.alpha_trust * match
                            )

                            tr = float(
                                np.clip(
                                    tr,
                                    cfg.trust_floor,
                                    1.0,
                                )
                            )

                        else:

                            # Missing data is not evidence of failure.
                            tr = self.trust[z][v]

                        raw_weight = 0.0
                        belief = combined

                    new_est[z][v] = combined
                    new_est_valid[z][v] = True
                    new_trust[z][v] = tr

                # ======================================================
                # Cold recovery: no previous valid estimate
                # ======================================================

                elif raw_ok:

                    act = raw_val

                    cz = None
                    conf_cz = None
                    w = None

                    combined = raw_val
                    residual = 0.0
                    match = 1.0

                    tr = (
                        (1.0 - cfg.alpha_trust)
                        * self.trust[z][v]
                        + cfg.alpha_trust * match
                    )

                    tr = float(
                        np.clip(
                            tr,
                            cfg.trust_floor,
                            1.0,
                        )
                    )

                    raw_weight = tr
                    belief = raw_val

                    new_est[z][v] = combined
                    new_est_valid[z][v] = True
                    new_trust[z][v] = tr

                # ======================================================
                # Still no valid estimate or current reading
                # ======================================================

                else:

                    act = float("nan")

                    cz = None
                    conf_cz = None
                    w = None

                    combined = float("nan")
                    residual = float("nan")
                    match = 0.0

                    tr = self.cfg.trust_floor

                    raw_weight = 0.0
                    belief = float("nan")

                    new_est[z][v] = float("nan")
                    new_est_valid[z][v] = False
                    new_trust[z][v] = tr

                # ======================================================
                # Save current raw reading
                # ======================================================

                if raw_missing:

                    new_prev_raw[z][v] = RawReading(None)

                else:

                    new_prev_raw[z][v] = RawReading(raw_val)

                new_raw_valid[z][v] = raw_ok

                # ======================================================
                # Build trust record
                # ======================================================

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

        # ==============================================================
        # Commit state after processing all zones
        # ==============================================================

        self.est = new_est
        self._est_valid = new_est_valid

        self.trust = new_trust

        self.prev_raw = new_prev_raw
        self._raw_valid = new_raw_valid

        return records
