"""
Continuous PID output to binary actuator request mapping with software PWM,
minimum on/off dwell timers, and irrigation pulse/soak lockout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from control.types import ACTUATOR_NAMES, ActuatorConfig


def _validate_finite_float(val: Any, name: str) -> float:
    if isinstance(val, (bool, str, bytes)) or val is None:
        raise ValueError(f"{name} must be a finite numeric value, got {val!r}")
    try:
        fval = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {val!r}") from exc
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return fval


def _validate_binary_bit(val: Any, name: str) -> int:
    if isinstance(val, (str, bytes)) or val is None:
        raise ValueError(f"{name} must be binary 0 or 1, got {val!r}")
    if isinstance(val, bool):
        return 1 if val else 0
    try:
        fval = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be binary 0 or 1, got {val!r}") from exc
    if not math.isfinite(fval) or fval not in (0.0, 1.0):
        raise ValueError(f"{name} must be binary 0 or 1, got {val!r}")
    return int(fval)


def validate_actuator_name(name: str) -> str:
    if not isinstance(name, str) or name not in ACTUATOR_NAMES:
        raise ValueError(f"Unknown actuator name {name!r}; expected one of {ACTUATOR_NAMES}")
    return name


def validate_binary_action(action: Any) -> tuple[int, int, int, int]:
    if isinstance(action, (str, bytes, dict)) or action is None:
        raise ValueError(f"Action must be a 4-element binary sequence, got {type(action).__name__}")
    try:
        seq = list(action)
    except TypeError as exc:
        raise ValueError(f"Action must be iterable of length 4, got {type(action).__name__}") from exc
    if len(seq) != 4:
        raise ValueError(f"Action must have length 4, got {len(seq)}")
    return (
        _validate_binary_bit(seq[0], "action[0](heat)"),
        _validate_binary_bit(seq[1], "action[1](fan)"),
        _validate_binary_bit(seq[2], "action[2](mist)"),
        _validate_binary_bit(seq[3], "action[3](water)"),
    )


def validate_actuator_config(cfg: ActuatorConfig) -> None:
    if not isinstance(cfg, ActuatorConfig):
        raise TypeError(f"Expected ActuatorConfig, got {type(cfg).__name__}")
    if isinstance(cfg.pwm_period_heat, bool) or int(cfg.pwm_period_heat) < 1:
        raise ValueError(f"pwm_period_heat must be >= 1, got {cfg.pwm_period_heat}")
    if isinstance(cfg.pwm_period_mist, bool) or int(cfg.pwm_period_mist) < 1:
        raise ValueError(f"pwm_period_mist must be >= 1, got {cfg.pwm_period_mist}")
    if isinstance(cfg.min_on_steps, bool) or int(cfg.min_on_steps) < 1:
        raise ValueError(f"min_on_steps must be >= 1, got {cfg.min_on_steps}")
    if isinstance(cfg.min_off_steps, bool) or int(cfg.min_off_steps) < 1:
        raise ValueError(f"min_off_steps must be >= 1, got {cfg.min_off_steps}")
    if isinstance(cfg.water_max_pulse_steps, bool) or int(cfg.water_max_pulse_steps) < 0:
        raise ValueError(f"water_max_pulse_steps must be >= 0, got {cfg.water_max_pulse_steps}")
    if isinstance(cfg.water_soak_steps, bool) or int(cfg.water_soak_steps) < 0:
        raise ValueError(f"water_soak_steps must be >= 0, got {cfg.water_soak_steps}")
    cutoff = _validate_finite_float(cfg.min_duty_cutoff, "min_duty_cutoff")
    if not (0.0 <= cutoff <= 1.0):
        raise ValueError(f"min_duty_cutoff must be in [0, 1], got {cutoff}")
    fan_on = _validate_finite_float(cfg.fan_on_thresh, "fan_on_thresh")
    fan_off = _validate_finite_float(cfg.fan_off_thresh, "fan_off_thresh")
    if fan_off < 0.0 or fan_on < 0.0 or fan_off > fan_on:
        raise ValueError(f"Invalid fan thresholds: fan_off_thresh={fan_off}, fan_on_thresh={fan_on}")
    if _validate_finite_float(cfg.ambient_margin_T, "ambient_margin_T") < 0.0:
        raise ValueError("ambient_margin_T must be >= 0")
    if _validate_finite_float(cfg.ambient_margin_H, "ambient_margin_H") < 0.0:
        raise ValueError("ambient_margin_H must be >= 0")
    if _validate_finite_float(cfg.arbiter_hysteresis, "arbiter_hysteresis") < 0.0:
        raise ValueError("arbiter_hysteresis must be >= 0")


@dataclass(frozen=True)
class MappedRequests:
    """Intermediate requests before shared-fan arbitration."""

    heat_cmd: int
    mist_cmd: int
    water_cmd: int
    cool_demand: float
    heat_assist_demand: float
    dehum_demand: float
    hum_assist_demand: float
    water_soak_blocked: bool


class DwellTracker:
    """Enforces minimum ON and minimum OFF durations to prevent relay chatter."""

    def __init__(self, min_on_steps: int, min_off_steps: int):
        self.min_on_steps = max(1, int(min_on_steps))
        self.min_off_steps = max(1, int(min_off_steps))
        self.state: int = 0
        self.steps_in_state: int = self.min_off_steps

    def reset(self) -> None:
        self.state = 0
        self.steps_in_state = self.min_off_steps

    def sync_external(self, actual_state: int) -> None:
        """Keep dwell timer synchronized when MARL or SAFE_MODE drives the actuator."""
        s = 1 if actual_state else 0
        if s == self.state:
            self.steps_in_state += 1
        else:
            self.state = s
            self.steps_in_state = 1

    def apply(self, desired_state: int) -> int:
        desired = 1 if desired_state else 0
        if desired == self.state:
            self.steps_in_state += 1
            return self.state

        if self.state == 1 and self.steps_in_state < self.min_on_steps:
            self.steps_in_state += 1
            return 1
        if self.state == 0 and self.steps_in_state < self.min_off_steps:
            self.steps_in_state += 1
            return 0

        self.state = desired
        self.steps_in_state = 1
        return self.state

    def can_switch_to(self, desired_state: int) -> bool:
        desired = 1 if desired_state else 0
        if desired == self.state:
            return True
        if self.state == 1:
            return self.steps_in_state >= self.min_on_steps
        return self.steps_in_state >= self.min_off_steps


class ActuatorMapper:
    """
    Converts normalized continuous PID outputs (u_T, u_H, u_M) into:
    - Time-proportioned (PWM) + dwell-protected binary commands for heat & mist
    - Pulse-limited + soak-locked binary command for water
    - Continuous scored ventilation demands for ActuatorArbiter
    """

    def __init__(self, cfg: ActuatorConfig | None = None):
        self.cfg = cfg or ActuatorConfig()
        validate_actuator_config(self.cfg)
        self.dwell: dict[str, DwellTracker] = {
            name: DwellTracker(self.cfg.min_on_steps, self.cfg.min_off_steps)
            for name in ACTUATOR_NAMES
        }
        self._pwm_phase: dict[str, int] = {"heat": 0, "mist": 0}
        self._water_pulse_steps: int = 0
        self._water_soak_remaining: int = 0

    def reset(self) -> None:
        for d in self.dwell.values():
            d.reset()
        self._pwm_phase = {"heat": 0, "mist": 0}
        self._water_pulse_steps = 0
        self._water_soak_remaining = 0

    def sync_executed(self, action: tuple[int, int, int, int]) -> None:
        """Synchronize dwell timers and irrigation pulse/soak counters with executed action."""
        parsed = validate_binary_action(action)
        for name, val in zip(ACTUATOR_NAMES, parsed):
            self.dwell[name].sync_external(val)
        if parsed[3] == 1:
            self._water_soak_remaining = 0
            self._water_pulse_steps += 1
        elif self._water_pulse_steps > 0:
            self._water_pulse_steps = 0
            self._water_soak_remaining = max(0, self.cfg.water_soak_steps - 1)
        elif self._water_soak_remaining > 0:
            self._water_soak_remaining -= 1

    def _pwm_decision(self, name: str, demand: float, period: int) -> int:
        if demand < self.cfg.min_duty_cutoff:
            self._pwm_phase[name] = (self._pwm_phase[name] + 1) % max(1, period)
            return 0
        period = max(1, int(period))
        phase = self._pwm_phase[name] % period
        self._pwm_phase[name] = (phase + 1) % period
        on_steps = round(min(1.0, max(0.0, demand)) * period)
        return 1 if phase < on_steps else 0

    def map_outputs(self, u_T: float, u_H: float, u_M: float) -> MappedRequests:
        val_T = _validate_finite_float(u_T, "u_T")
        val_H = _validate_finite_float(u_H, "u_H")
        val_M = _validate_finite_float(u_M, "u_M")
        for name, val in (("u_T", val_T), ("u_H", val_H), ("u_M", val_M)):
            if not (-1.0 <= val <= 1.0):
                raise ValueError(f"{name} must be within [-1.0, 1.0], got {val}")

        c = self.cfg

        heat_demand = max(0.0, val_T)
        cool_demand = max(0.0, -val_T)
        mist_demand = max(0.0, val_H)
        dehum_demand = max(0.0, -val_H)
        water_demand = max(0.0, val_M)

        raw_heat = self._pwm_decision("heat", heat_demand, c.pwm_period_heat)
        raw_mist = self._pwm_decision("mist", mist_demand, c.pwm_period_mist)

        heat_cmd = self.dwell["heat"].apply(raw_heat)
        mist_cmd = self.dwell["mist"].apply(raw_mist)

        # Water: pulse-bounded + soak lockout to handle soil infiltration lag
        soak_blocked = False
        if self._water_soak_remaining > 0:
            raw_water = 0
            self._water_soak_remaining -= 1
            if water_demand >= c.min_duty_cutoff:
                soak_blocked = True
        elif water_demand >= c.min_duty_cutoff:
            if self._water_pulse_steps < c.water_max_pulse_steps:
                raw_water = 1
                self._water_pulse_steps += 1
            else:
                raw_water = 0
                self._water_pulse_steps = 0
                self._water_soak_remaining = max(0, c.water_soak_steps - 1)
                soak_blocked = True
        else:
            raw_water = 0
            self._water_pulse_steps = 0

        # Water bypasses min_on_steps if max_pulse is smaller than min_on_steps,
        # so respect explicit pulse bound cleanly while preventing sub-min_off bounce.
        if raw_water == 1 and self.dwell["water"].state == 0 and not self.dwell["water"].can_switch_to(1):
            water_cmd = 0
            soak_blocked = True
        else:
            self.dwell["water"].sync_external(raw_water)
            water_cmd = raw_water

        return MappedRequests(
            heat_cmd=heat_cmd,
            mist_cmd=mist_cmd,
            water_cmd=water_cmd,
            cool_demand=cool_demand,
            heat_assist_demand=heat_demand,
            dehum_demand=dehum_demand,
            hum_assist_demand=mist_demand,
            water_soak_blocked=soak_blocked,
        )
