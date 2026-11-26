"""
Sensor interface adapter decoupling PID/Supervisor from TrustNetwork internals.

Enforces finite-value checks so NaN/Inf raw sensor readings cannot poison
`belief = trust * raw + (1 - trust) * combined`, checks optional measurement
staleness/missing keys safely without KeyError, tracks open-loop estimation
duration (especially for uncoupled soil moisture M), and manages transient
holds before escalating to UNRELIABLE_FAILSAFE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from control.types import (
    MeasurementQuality,
    SensorMeasurement,
    VARS,
    ZONE_IDS,
)


@dataclass
class SensorInterfaceConfig:
    """
    Configurable measurement acceptance thresholds.

    NOTE: `theta_trust_good` and `theta_est_min` are empirical gating thresholds,
    NOT calibrated probabilities. Tune via fault-injection sweeps.
    """

    theta_trust_good: float = 0.60
    theta_est_min: float = 0.15
    t_hold_max: int = 5
    t_est_max: dict[str, int] = field(
        default_factory=lambda: {"T": 120, "H": 120, "M": 15}
    )
    max_staleness_s: Optional[float] = None
    max_identical_timestamps: Optional[int] = None


def _is_finite(val: Any) -> bool:
    if val is None:
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


class SensorInterface:
    """Adapts TrustRecords or raw observations into validated SensorMeasurements."""

    def __init__(self, cfg: Optional[SensorInterfaceConfig] = None):
        self.cfg = cfg or SensorInterfaceConfig()
        self.reset()

    def reset(self) -> None:
        self._open_loop_steps: dict[tuple[str, str], int] = {
            (z, v): 0 for z in ZONE_IDS for v in VARS
        }
        self._hold_steps: dict[tuple[str, str], int] = {
            (z, v): 0 for z in ZONE_IDS for v in VARS
        }
        self._last_timestamps: dict[tuple[str, str], Optional[float]] = {
            (z, v): None for z in ZONE_IDS for v in VARS
        }
        self._same_ts_count: dict[tuple[str, str], int] = {
            (z, v): 0 for z in ZONE_IDS for v in VARS
        }

    def _check_stale(
        self,
        key: tuple[str, str],
        timestamp: Optional[float],
        current_time: Optional[float],
    ) -> bool:
        if timestamp is None or not _is_finite(timestamp):
            return False
        ts = float(timestamp)
        if (
            self.cfg.max_staleness_s is not None
            and current_time is not None
            and _is_finite(current_time)
            and (float(current_time) - ts) > self.cfg.max_staleness_s
        ):
            return True

        prev_ts = self._last_timestamps[key]
        if prev_ts is not None and ts == prev_ts:
            self._same_ts_count[key] += 1
        else:
            self._last_timestamps[key] = ts
            self._same_ts_count[key] = 1

        if (
            self.cfg.max_identical_timestamps is not None
            and self._same_ts_count[key] > self.cfg.max_identical_timestamps
        ):
            return True
        return False

    def _classify_and_select(
        self,
        zone_id: str,
        var: str,
        raw_val: Optional[float],
        combined_val: Optional[float],
        belief_val: Optional[float],
        trust: float,
        confidence: float,
        timestamp: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> SensorMeasurement:
        key = (zone_id, var)
        stale = self._check_stale(key, timestamp, current_time)
        valid_raw = _is_finite(raw_val) and not stale
        valid_est = _is_finite(combined_val)
        safe_trust = float(trust) if _is_finite(trust) else 0.0
        safe_conf = float(confidence) if _is_finite(confidence) else 0.0

        raw_f = float(raw_val) if _is_finite(raw_val) else None
        est_f = float(combined_val) if valid_est else None
        belief_f = float(belief_val) if (valid_raw and _is_finite(belief_val)) else None
        ts_f = float(timestamp) if _is_finite(timestamp) else None

        # Tier 1: Trusted, fresh, finite raw reading -> use belief (or raw)
        if valid_raw and safe_trust >= self.cfg.theta_trust_good:
            self._open_loop_steps[key] = 0
            self._hold_steps[key] = 0
            chosen = belief_f if belief_f is not None else raw_f
            return SensorMeasurement(
                var=var,
                value=chosen,
                raw=raw_f,
                combined=est_f,
                belief=belief_f,
                trust=safe_trust,
                confidence=safe_conf,
                quality=MeasurementQuality.TRUSTED,
                valid_raw=True,
                valid_est=valid_est,
                open_loop_steps=0,
                hold_steps=0,
                timestamp=ts_f,
                stale=False,
            )

        # Tier 2: Raw missing/NaN/stale or low trust -> use physics estimate `combined`
        # Never use `belief` when raw is invalid because trust * NaN == NaN.
        max_open_loop = self.cfg.t_est_max.get(var, 60)
        if (
            valid_est
            and safe_conf >= self.cfg.theta_est_min
            and self._open_loop_steps[key] < max_open_loop
        ):
            self._open_loop_steps[key] += 1
            self._hold_steps[key] = 0
            return SensorMeasurement(
                var=var,
                value=est_f,
                raw=raw_f,
                combined=est_f,
                belief=belief_f,
                trust=safe_trust,
                confidence=safe_conf,
                quality=MeasurementQuality.ESTIMATE_ONLY,
                valid_raw=valid_raw,
                valid_est=True,
                open_loop_steps=self._open_loop_steps[key],
                hold_steps=0,
                timestamp=ts_f,
                stale=stale,
            )

        # Tier 3 / 4: Both unavailable or estimator unreliable/expired -> hold then failsafe
        self._hold_steps[key] += 1
        hold_count = self._hold_steps[key]
        if hold_count <= self.cfg.t_hold_max:
            quality = MeasurementQuality.HOLD_TRANSIENT
        else:
            quality = MeasurementQuality.UNRELIABLE_FAILSAFE

        return SensorMeasurement(
            var=var,
            value=None,
            raw=raw_f,
            combined=est_f,
            belief=belief_f,
            trust=safe_trust,
            confidence=safe_conf,
            quality=quality,
            valid_raw=valid_raw,
            valid_est=valid_est,
            open_loop_steps=self._open_loop_steps[key],
            hold_steps=hold_count,
            timestamp=ts_f,
            stale=stale,
        )

    def from_trust_record(
        self,
        zone_id: str,
        var: str,
        record: Any,
        current_time: Optional[float] = None,
    ) -> SensorMeasurement:
        """
        Convert a duck-typed TrustRecord (or None if missing) into a validated SensorMeasurement.
        """
        if record is None:
            return self._classify_and_select(
                zone_id=zone_id,
                var=var,
                raw_val=None,
                combined_val=None,
                belief_val=None,
                trust=0.0,
                confidence=0.0,
                timestamp=None,
                current_time=current_time,
            )

        conf_act = getattr(record, "conf_act", 0.0)
        conf_cz = getattr(record, "conf_cz", None)
        if conf_cz is not None and _is_finite(conf_cz):
            confidence = max(float(conf_cz), float(conf_act) if _is_finite(conf_act) else 0.0)
        else:
            confidence = float(conf_act) if _is_finite(conf_act) else 0.0

        return self._classify_and_select(
            zone_id=zone_id,
            var=var,
            raw_val=getattr(record, "raw", None),
            combined_val=getattr(record, "combined", None),
            belief_val=getattr(record, "belief", None),
            trust=getattr(record, "trust", 0.0),
            confidence=confidence,
            timestamp=getattr(record, "timestamp", None),
            current_time=current_time,
        )

    def from_trust_records(
        self,
        records: Optional[Mapping[str, Mapping[str, Any]]],
        current_time: Optional[float] = None,
    ) -> dict[str, dict[str, SensorMeasurement]]:
        """
        Convert nested records[zone_id][var] -> SensorMeasurement dict.
        Always populates every (zone_id, var) pair even when zones or vars are missing.
        """
        rec_map = records or {}
        out: dict[str, dict[str, SensorMeasurement]] = {}
        for z in ZONE_IDS:
            var_map = rec_map.get(z) or {}
            out[z] = {
                v: self.from_trust_record(z, v, var_map.get(v), current_time=current_time)
                for v in VARS
            }
        return out

    def from_raw_dict(
        self,
        raw_values: Optional[Mapping[str, Mapping[str, Any]]],
        trust: float = 1.0,
        confidence: float = 1.0,
        timestamps: Optional[Mapping[str, Mapping[str, float]]] = None,
        current_time: Optional[float] = None,
    ) -> dict[str, dict[str, SensorMeasurement]]:
        """Standalone adapter when running PID directly on raw readings without TrustNetwork."""
        raw_map = raw_values or {}
        ts_map = timestamps or {}
        out: dict[str, dict[str, SensorMeasurement]] = {}
        for z in ZONE_IDS:
            var_map = raw_map.get(z) or {}
            z_ts = ts_map.get(z) or {}
            out[z] = {}
            for v in VARS:
                val = var_map.get(v)
                out[z][v] = self._classify_and_select(
                    zone_id=z,
                    var=v,
                    raw_val=val,
                    combined_val=None,
                    belief_val=val,
                    trust=trust,
                    confidence=confidence,
                    timestamp=z_ts.get(v),
                    current_time=current_time,
                )
        return out

    def from_env_observations(
        self,
        obs: Optional[Mapping[str, Any]],
        trust: float = 1.0,
        confidence: float = 1.0,
        current_time: Optional[float] = None,
    ) -> dict[str, dict[str, SensorMeasurement]]:
        """
        Convert GreenhouseEnv observation dict {zone_id: np.ndarray([T, H, M, ...])}
        into per-zone SensorMeasurements for all ZONE_IDS.
        """
        obs_map = obs or {}
        raw_dict: dict[str, dict[str, Any]] = {}
        for z in ZONE_IDS:
            vec = obs_map.get(z)
            if vec is None or len(vec) < 3:
                raw_dict[z] = {"T": None, "H": None, "M": None}
            else:
                raw_dict[z] = {"T": vec[0], "H": vec[1], "M": vec[2]}
        return self.from_raw_dict(raw_dict, trust=trust, confidence=confidence, current_time=current_time)
