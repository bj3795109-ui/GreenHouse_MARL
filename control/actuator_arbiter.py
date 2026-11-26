"""
Ambient-aware shared fan arbitration and intra-zone actuator mutual exclusion.

Resolves conflicts between Temperature and Humidity control loops over the
single binary fan actuator (`u_fan`), taking into account that ventilation
equilibrates toward (T_ambient, H_ambient) rather than unconditionally cooling
or dehumidifying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from control.actuator_mapper import (
    DwellTracker,
    MappedRequests,
    _validate_binary_bit,
    _validate_finite_float,
    validate_actuator_config,
)
from control.types import (
    ActuatorConfig,
    AmbientState,
    ArbitrationBlockFlags,
    CropBounds,
    SensorMeasurement,
)


@dataclass(frozen=True)
class ArbitrationResult:
    """Final safe binary action for one zone plus feedback flags for PID anti-windup."""

    action: tuple[int, int, int, int]  # (heat, fan, mist, water)
    block_flags: ArbitrationBlockFlags
    reason: str


def _validate_optional_finite(val: Any, name: str) -> Optional[float]:
    if val is None:
        return None
    return _validate_finite_float(val, name)


def _validate_bounds_finite(bounds: CropBounds, name: str) -> None:
    if not isinstance(bounds, CropBounds):
        raise TypeError(f"Expected CropBounds for {name}, got {type(bounds).__name__}")
    min_a = _validate_finite_float(bounds.min_acceptable, f"{name}.min_acceptable")
    opt_min = _validate_finite_float(bounds.optimal_min, f"{name}.optimal_min")
    opt_max = _validate_finite_float(bounds.optimal_max, f"{name}.optimal_max")
    max_a = _validate_finite_float(bounds.max_acceptable, f"{name}.max_acceptable")
    if opt_min > opt_max or min_a > max_a:
        raise ValueError(f"Invalid bounds ordering in {name}")


def _validate_mapped_requests(req: MappedRequests) -> None:
    if not isinstance(req, MappedRequests):
        raise TypeError(f"Expected MappedRequests, got {type(req).__name__}")
    _validate_binary_bit(req.heat_cmd, "req.heat_cmd")
    _validate_binary_bit(req.mist_cmd, "req.mist_cmd")
    _validate_binary_bit(req.water_cmd, "req.water_cmd")
    for fname in ("cool_demand", "heat_assist_demand", "dehum_demand", "hum_assist_demand"):
        d = _validate_finite_float(getattr(req, fname), f"req.{fname}")
        if not (0.0 <= d <= 1.0):
            raise ValueError(f"req.{fname} must be within [0.0, 1.0], got {d}")
    if not isinstance(req.water_soak_blocked, bool):
        raise ValueError(f"req.water_soak_blocked must be bool, got {type(req.water_soak_blocked).__name__}")


def _normalized_excursion(val: Optional[float], bounds: CropBounds) -> float:
    """Normalized distance outside optimal band relative to acceptable margin."""
    if val is None:
        return 0.0
    if val > bounds.optimal_max:
        span = max(1e-3, bounds.max_acceptable - bounds.optimal_max)
        return (val - bounds.optimal_max) / span
    if val < bounds.optimal_min:
        span = max(1e-3, bounds.optimal_min - bounds.min_acceptable)
        return (bounds.optimal_min - val) / span
    return 0.0


def _outside_acceptable(val: Optional[float], bounds: CropBounds) -> bool:
    if val is None:
        return False
    return val > bounds.max_acceptable or val < bounds.min_acceptable


class ActuatorArbiter:
    """Deterministic shared-fan arbiter and mutual-exclusion enforcer."""

    def __init__(self, cfg: ActuatorConfig | None = None):
        self.cfg = cfg or ActuatorConfig()
        validate_actuator_config(self.cfg)
        self.fan_dwell = DwellTracker(self.cfg.min_on_steps, self.cfg.min_off_steps)
        self._last_fan_decision: int = 0

    def reset(self) -> None:
        self.fan_dwell.reset()
        self._last_fan_decision = 0

    def sync_fan(self, actual_fan: int) -> None:
        fan_bit = _validate_binary_bit(actual_fan, "actual_fan")
        self.fan_dwell.sync_external(fan_bit)
        self._last_fan_decision = fan_bit

    def _evaluate_fan_utility(
        self,
        demand_pos: float,
        demand_neg: float,
        current_val: Optional[float],
        ambient_val: Optional[float],
        margin: float,
        active_actuator_on: bool,
    ) -> tuple[int, float, bool]:
        """
        Return (utility_sign, strength, direction_blocked_by_ambient).

        utility_sign:
          +1 -> running fan helps this loop
           0 -> loop is neutral / inside deadband
          -1 -> running fan actively harms this loop (fights heater/mister or pushes away from target)
        """
        if current_val is None or ambient_val is None:
            # Ambient unknown: cannot safely assume fan cools/dries; avoid counterproductive actuation
            if demand_neg >= self.cfg.fan_on_thresh:
                return (0, 0.0, True)
            return (0, 0.0, False)

        diff = current_val - ambient_val  # positive => fan lowers variable toward ambient

        if demand_neg >= self.cfg.fan_off_thresh:
            # Loop wants to DECREASE variable (cooling or dehumidification)
            if diff > margin:
                return (+1, demand_neg, False)
            if diff < -margin:
                # Fan would move variable FURTHER UP (away from target)
                return (-1, demand_neg, True)
            return (0, 0.0, True)

        if demand_pos >= self.cfg.fan_off_thresh:
            # Loop wants to INCREASE variable (heating or misting)
            if diff < -margin and not active_actuator_on:
                # Ambient is warmer/more humid than zone -> free ambient assistance
                return (+1, 0.5 * demand_pos, False)
            if diff > margin:
                # Ambient is colder/drier -> running fan vents away heat/mist
                return (-1, demand_pos, False)
            return (0, 0.0, False)

        return (0, 0.0, False)

    def arbitrate(
        self,
        req: MappedRequests,
        meas_T: SensorMeasurement,
        meas_H: SensorMeasurement,
        bounds_T: CropBounds,
        bounds_H: CropBounds,
        ambient: AmbientState,
    ) -> ArbitrationResult:
        _validate_mapped_requests(req)
        _validate_bounds_finite(bounds_T, "bounds_T")
        _validate_bounds_finite(bounds_H, "bounds_H")
        if not isinstance(ambient, AmbientState):
            raise TypeError(f"Expected AmbientState, got {type(ambient).__name__}")
        amb_T = _validate_optional_finite(ambient.T_ambient, "ambient.T_ambient")
        amb_H = _validate_optional_finite(ambient.H_ambient, "ambient.H_ambient")
        T_val = _validate_optional_finite(meas_T.value, "meas_T.value")
        H_val = _validate_optional_finite(meas_H.value, "meas_H.value")

        c = self.cfg

        u_T_util, u_T_str, T_amb_blocked = self._evaluate_fan_utility(
            req.heat_assist_demand,
            req.cool_demand,
            T_val,
            amb_T,
            c.ambient_margin_T,
            active_actuator_on=(req.heat_cmd == 1),
        )
        u_H_util, u_H_str, H_amb_blocked = self._evaluate_fan_utility(
            req.hum_assist_demand,
            req.dehum_demand,
            H_val,
            amb_H,
            c.ambient_margin_H,
            active_actuator_on=(req.mist_cmd == 1),
        )

        # Hysteresis check on whether any beneficial demand crosses on/off threshold
        thresh = c.fan_off_thresh if self._last_fan_decision == 1 else c.fan_on_thresh
        T_wants = u_T_util == +1 and u_T_str >= thresh
        H_wants = u_H_util == +1 and u_H_str >= thresh

        desired_fan = 0
        reason = "fan_idle"

        if not T_wants and not H_wants:
            desired_fan = 0
            reason = "no_beneficial_vent_demand"
        elif (u_T_util >= 0 and u_H_util >= 0) and (T_wants or H_wants):
            desired_fan = 1
            reason = "unopposed_beneficial_vent"
        else:
            # Conflict: one loop benefits (+1) while the other is harmed (-1)
            T_crit = _outside_acceptable(T_val, bounds_T)
            H_crit = _outside_acceptable(H_val, bounds_H)
            T_sev = _normalized_excursion(T_val, bounds_T)
            H_sev = _normalized_excursion(H_val, bounds_H)

            if T_wants and u_H_util == -1:
                if T_crit and not H_crit:
                    desired_fan = 1
                    reason = "conflict_T_critical_priority"
                elif H_crit and not T_crit:
                    desired_fan = 0
                    reason = "conflict_H_critical_blocks_fan"
                elif T_val is not None and T_val > bounds_T.max_acceptable:
                    desired_fan = 1
                    reason = "conflict_overtemp_priority"
                else:
                    # Cooling via fan can be compensated on H by mist; compare severity + hysteresis
                    margin = -c.arbiter_hysteresis if self._last_fan_decision == 1 else c.arbiter_hysteresis
                    if (T_sev + 0.15) > (H_sev + margin):
                        desired_fan = 1
                        reason = "conflict_T_severity_wins"
                    else:
                        desired_fan = 0
                        reason = "conflict_H_protection_blocks_fan"

            elif H_wants and u_T_util == -1:
                if H_crit and not T_crit:
                    desired_fan = 1
                    reason = "conflict_H_critical_priority"
                elif T_crit and not H_crit:
                    desired_fan = 0
                    reason = "conflict_T_critical_blocks_fan"
                else:
                    margin = -c.arbiter_hysteresis if self._last_fan_decision == 1 else c.arbiter_hysteresis
                    if H_sev > (T_sev + margin):
                        desired_fan = 1
                        reason = "conflict_H_severity_wins"
                    else:
                        desired_fan = 0
                        reason = "conflict_T_protection_blocks_fan"

        fan_cmd = self.fan_dwell.apply(desired_fan)
        self._last_fan_decision = fan_cmd

        # Intra-zone mutual exclusion: never fight active cooling/dehumidifying fan
        heat_cmd = req.heat_cmd
        mist_cmd = req.mist_cmd
        water_cmd = req.water_cmd

        fan_is_cooling = (
            fan_cmd == 1
            and T_val is not None
            and ambient.T_ambient is not None
            and T_val > ambient.T_ambient + c.ambient_margin_T
        )
        fan_is_drying = (
            fan_cmd == 1
            and H_val is not None
            and ambient.H_ambient is not None
            and H_val > ambient.H_ambient + c.ambient_margin_H
        )

        block_T_pos = False
        block_H_pos = False

        if fan_is_cooling and heat_cmd == 1:
            if not (c.allow_cold_dehumidification and _outside_acceptable(H_val, bounds_H)):
                heat_cmd = 0
                block_T_pos = True
                reason = f"{reason}|heat_blocked_by_cooling_fan"

        if fan_is_drying and mist_cmd == 1:
            mist_cmd = 0
            block_H_pos = True
            reason = f"{reason}|mist_blocked_by_drying_fan"

        # Inform PID anti-windup if requested cooling/dehumidification was blocked
        block_T_neg = T_amb_blocked or (req.cool_demand >= c.fan_on_thresh and fan_cmd == 0)
        block_H_neg = H_amb_blocked or (req.dehum_demand >= c.fan_on_thresh and fan_cmd == 0)

        return ArbitrationResult(
            action=(heat_cmd, fan_cmd, mist_cmd, water_cmd),
            block_flags=ArbitrationBlockFlags(
                block_T_pos=block_T_pos,
                block_T_neg=block_T_neg,
                block_H_pos=block_H_pos,
                block_H_neg=block_H_neg,
                block_M_pos=req.water_soak_blocked,
            ),
            reason=reason,
        )
