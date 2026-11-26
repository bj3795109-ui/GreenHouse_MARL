"""
Per-zone MARL-PID Supervisor state machine.

Manages transitions among MARL_ACTIVE, PID_FALLBACK, RECOVERY_PENDING, and SAFE_MODE
independently for a single greenhouse zone, including strict action validation,
persistent boundary-divergence detection, recovery validation, and bumpless handover.

NOTE: Software SAFE_MODE and safety checks complement—but NEVER replace—the
hardware microcontroller (ESP32) safety interlocks defined in V1.1 §0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from control.types import (
    ACTUATOR_NAMES,
    AmbientState,
    ControlMode,
    CropBounds,
    MeasurementQuality,
    SensorMeasurement,
    VARS,
    ZoneStatus,
)
from control.zone_controller import ZoneController


@dataclass
class SupervisorConfig:
    """Configurable supervisor thresholds and safety limits."""

    N_confirm: int = 10
    W_persist: int = 15
    K_contra: int = 5
    theta_marl_min_trust: float = 0.35
    theta_recovery_trust: float = 0.60
    marl_requires_trusted_raw: bool = True
    handover_dwell_protect_steps: int = 2

    # Software defense-in-depth limits (mirrors V1.1 §0 hardware interlocks)
    T_max_hard: float = 45.0
    t_max_continuous_heat: int = 120
    t_max_continuous_water: int = 20


@dataclass
class ZoneMetrics:
    total_steps: int = 0
    marl_active_steps: int = 0
    pid_fallback_steps: int = 0
    recovery_pending_steps: int = 0
    safe_mode_steps: int = 0
    safe_mode_events: int = 0
    unsafe_actions_intercepted: int = 0
    marl_to_pid_switches: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0
    actuator_toggles: int = 0
    transition_log: list[tuple[int, str, str, str]] = field(default_factory=list)


def validate_marl_action(raw_action: Any) -> tuple[bool, Optional[tuple[int, int, int, int]], Optional[str]]:
    """
    Strictly validate a MARL action BEFORE any bool() coercion.

    Rejects None, wrong length, non-numeric types, NaN/Inf (preventing Python's
    `bool(np.nan) == True` hazard), and values outside {0, 1}.
    """
    if raw_action is None:
        return (False, None, "missing_action")

    if isinstance(raw_action, (str, bytes, dict)):
        return (False, None, f"invalid_type:{type(raw_action).__name__}")

    try:
        arr = np.asarray(raw_action)
    except Exception as exc:
        return (False, None, f"conversion_error:{exc}")

    if arr.shape != (4,):
        return (False, None, f"invalid_shape:{arr.shape}")

    parsed: list[int] = []
    for idx, val in enumerate(arr.tolist()):
        if isinstance(val, bool):
            parsed.append(1 if val else 0)
            continue
        if not isinstance(val, (int, float, np.integer, np.floating)):
            return (False, None, f"non_numeric_element_{idx}:{type(val).__name__}")
        fval = float(val)
        if not math.isfinite(fval):
            return (False, None, f"non_finite_element_{idx}:{fval}")
        if fval not in (0.0, 1.0):
            return (False, None, f"non_binary_element_{idx}:{fval}")
        parsed.append(int(fval))

    return (True, (parsed[0], parsed[1], parsed[2], parsed[3]), None)


class ZoneSupervisor:
    """Independent per-zone state machine governing MARL vs. PID vs. SAFE_MODE."""

    def __init__(
        self,
        zone_id: str,
        zone_controller: ZoneController,
        cfg: Optional[SupervisorConfig] = None,
        initial_mode: ControlMode = ControlMode.MARL_ACTIVE,
    ):
        self.zone_id = zone_id
        self.controller = zone_controller
        self.cfg = cfg or SupervisorConfig()
        self.initial_mode = initial_mode
        self.reset()

    def reset(self) -> None:
        self.mode: ControlMode = self.initial_mode
        self.controller.reset()
        self.metrics = ZoneMetrics()
        self._recovery_streak: int = 0
        self._persist_bad_steps: int = 0
        self._contra_steps: int = 0
        self._cont_heat_steps: int = 0
        self._cont_water_steps: int = 0
        self._handover_protect_remaining: int = 0
        self._prev_executed: Optional[tuple[int, int, int, int]] = None

    def _transition(self, new_mode: ControlMode, reason: str) -> tuple[Optional[ControlMode], Optional[str]]:
        if new_mode == self.mode:
            return (None, None)
        old = self.mode
        self.mode = new_mode
        self.metrics.transition_log.append((self.metrics.total_steps, old.value, new_mode.value, reason))
        if old == ControlMode.MARL_ACTIVE and new_mode == ControlMode.PID_FALLBACK:
            self.metrics.marl_to_pid_switches += 1
        if old == ControlMode.PID_FALLBACK and new_mode == ControlMode.RECOVERY_PENDING:
            self.metrics.recovery_attempts += 1
        if old == ControlMode.RECOVERY_PENDING and new_mode == ControlMode.MARL_ACTIVE:
            self.metrics.recovery_successes += 1
        if new_mode == ControlMode.SAFE_MODE:
            self.metrics.safe_mode_events += 1
        return (old, reason)

    def _sensors_unreliable(self, measurements: dict[str, SensorMeasurement]) -> bool:
        return any(
            measurements[v].quality == MeasurementQuality.UNRELIABLE_FAILSAFE for v in VARS
        )

    def _sensors_ready_for_marl(self, measurements: dict[str, SensorMeasurement]) -> bool:
        for v in VARS:
            m = measurements[v]
            if m.quality in (MeasurementQuality.HOLD_TRANSIENT, MeasurementQuality.UNRELIABLE_FAILSAFE):
                return False
            if self.cfg.marl_requires_trusted_raw and m.trust < self.cfg.theta_recovery_trust:
                return False
        return True

    def _check_safety_and_divergence(
        self,
        action: tuple[int, int, int, int],
        measurements: dict[str, SensorMeasurement],
        bounds: dict[str, CropBounds],
        ambient: AmbientState,
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluate whether a syntactically valid MARL action is safe and non-divergent.
        Returns (is_safe, failure_reason).
        """
        heat, fan, mist, water = action
        T_val = measurements["T"].value
        H_val = measurements["H"].value
        M_val = measurements["M"].value

        # 1. Hard over-temperature check
        if T_val is not None and T_val > self.cfg.T_max_hard and heat == 1:
            return (False, "unsafe_heat_above_T_max_hard")

        # 2. Continuous-on watchdog checks (projected next step)
        if heat == 1 and (self._cont_heat_steps + 1) > self.cfg.t_max_continuous_heat:
            return (False, "unsafe_continuous_heat_exceeded")
        if water == 1 and (self._cont_water_steps + 1) > self.cfg.t_max_continuous_water:
            return (False, "unsafe_continuous_water_exceeded")

        # 3. Contradictory actuator combinations (heat + cooling fan, mist + drying fan)
        fan_cools = (
            fan == 1
            and T_val is not None
            and ambient.T_ambient is not None
            and T_val > ambient.T_ambient + 0.5
        )
        fan_dries = (
            fan == 1
            and H_val is not None
            and ambient.H_ambient is not None
            and H_val > ambient.H_ambient + 1.5
        )
        if (heat == 1 and fan_cools) or (mist == 1 and fan_dries):
            self._contra_steps += 1
            if self._contra_steps >= self.cfg.K_contra:
                return (False, "persistent_contradictory_actuation")
        else:
            self._contra_steps = 0

        # 4. Monitored persistent boundary-divergence detection
        diverging = False
        if T_val is not None:
            if T_val > bounds["T"].max_acceptable and heat == 1:
                diverging = True
            elif T_val < bounds["T"].min_acceptable and heat == 0:
                diverging = True
        if H_val is not None:
            if H_val < bounds["H"].min_acceptable and mist == 0:
                diverging = True
            elif H_val > bounds["H"].max_acceptable and mist == 1:
                diverging = True
        if M_val is not None:
            if M_val < bounds["M"].min_acceptable and water == 0:
                diverging = True
            elif M_val > bounds["M"].max_acceptable and water == 1:
                diverging = True

        if diverging:
            self._persist_bad_steps += 1
            if self._persist_bad_steps >= self.cfg.W_persist:
                return (False, "persistent_boundary_divergence")
        else:
            self._persist_bad_steps = 0

        return (True, None)

    def _apply_handover_dwell_protection(
        self, candidate: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Prevent abrupt multi-actuator bounce during initial steps back in MARL_ACTIVE."""
        if self._handover_protect_remaining <= 0 or self._prev_executed is None:
            return candidate
        self._handover_protect_remaining -= 1
        out: list[int] = []
        for idx, name in enumerate(ACTUATOR_NAMES):
            desired = candidate[idx]
            dwell = (
                self.controller.arbiter.fan_dwell
                if name == "fan"
                else self.controller.mapper.dwell[name]
            )
            if dwell.can_switch_to(desired):
                out.append(desired)
            else:
                out.append(self._prev_executed[idx])
        return (out[0], out[1], out[2], out[3])

    def _enforce_software_interlocks(
        self,
        action: tuple[int, int, int, int],
        measurements: dict[str, SensorMeasurement],
    ) -> tuple[int, int, int, int]:
        """Last-line software safety clamp (never replaces ESP32 hardware interlocks)."""
        heat, fan, mist, water = action
        T_val = measurements["T"].value
        if T_val is not None and T_val > self.cfg.T_max_hard:
            heat = 0
        if heat == 1 and (self._cont_heat_steps + 1) > self.cfg.t_max_continuous_heat:
            heat = 0
        if water == 1 and (self._cont_water_steps + 1) > self.cfg.t_max_continuous_water:
            water = 0
        return (heat, fan, mist, water)

    def step(
        self,
        raw_marl_action: Any,
        measurements: dict[str, SensorMeasurement],
        bounds: dict[str, CropBounds],
        ambient: AmbientState,
        agent_failed: bool = False,
    ) -> ZoneStatus:
        self.metrics.total_steps += 1
        trans_from: Optional[ControlMode] = None
        trans_reason: Optional[str] = None

        # Validate MARL action first so RECOVERY_PENDING completion doesn't advance PID dwell timers needlessly
        if agent_failed:
            marl_valid, marl_parsed, marl_err = (False, None, "agent_crashed_or_timeout")
        else:
            marl_valid, marl_parsed, marl_err = validate_marl_action(raw_marl_action)

        if not marl_valid:
            self.metrics.unsafe_actions_intercepted += 1

        # Check sensor reliability for SAFE_MODE escalation
        sensors_failsafe = self._sensors_unreliable(measurements)
        marl_obs_degraded = (
            self.cfg.marl_requires_trusted_raw
            and any(
                measurements[v].trust < self.cfg.theta_marl_min_trust
                or measurements[v].quality != MeasurementQuality.TRUSTED
                for v in VARS
            )
        )

        pid_action = self._prev_executed or (0, 0, 0, 0)
        arb_reason = ""

        def _run_pid() -> tuple[tuple[int, int, int, int], str]:
            out = self.controller.step(measurements, bounds, ambient)
            return out.action, out.arbiter_reason

        if sensors_failsafe:
            trans_from, trans_reason = self._transition(
                ControlMode.SAFE_MODE, "measurements_unreliable_failsafe"
            )
            self._recovery_streak = 0
            executed = (0, 0, 0, 0)
            self.controller.track_shadow(measurements, bounds, executed, ambient)

        elif self.mode == ControlMode.SAFE_MODE:
            # Recover from SAFE_MODE to PID_FALLBACK once measurements return
            if not sensors_failsafe:
                trans_from, trans_reason = self._transition(
                    ControlMode.PID_FALLBACK, "measurements_restored_from_safe_mode"
                )
                pid_action, arb_reason = _run_pid()
                executed = pid_action
            else:
                executed = (0, 0, 0, 0)

        elif self.mode == ControlMode.MARL_ACTIVE:
            if not marl_valid:
                trans_from, trans_reason = self._transition(
                    ControlMode.PID_FALLBACK, f"marl_invalid:{marl_err}"
                )
            elif marl_obs_degraded:
                trans_from, trans_reason = self._transition(
                    ControlMode.PID_FALLBACK, "sensor_trust_below_marl_threshold"
                )
            else:
                assert marl_parsed is not None
                safe, safety_err = self._check_safety_and_divergence(
                    marl_parsed, measurements, bounds, ambient
                )
                if not safe:
                    self.metrics.unsafe_actions_intercepted += 1
                    trans_from, trans_reason = self._transition(
                        ControlMode.PID_FALLBACK, f"marl_unsafe:{safety_err}"
                    )

            if self.mode == ControlMode.PID_FALLBACK:
                self._recovery_streak = 0
                pid_action, arb_reason = _run_pid()
                executed = pid_action
            else:
                assert marl_parsed is not None
                executed = self._apply_handover_dwell_protection(marl_parsed)
                self.controller.track_shadow(measurements, bounds, executed, ambient)

        elif self.mode == ControlMode.PID_FALLBACK:
            if marl_valid and self._sensors_ready_for_marl(measurements):
                assert marl_parsed is not None
                safe, _ = self._check_safety_and_divergence(
                    marl_parsed, measurements, bounds, ambient
                )
                if safe:
                    self._recovery_streak = 1
                    trans_from, trans_reason = self._transition(
                        ControlMode.RECOVERY_PENDING, "marl_candidate_healthy"
                    )
                    if self._recovery_streak >= self.cfg.N_confirm:
                        trans_from, trans_reason = self._transition(
                            ControlMode.MARL_ACTIVE, "recovery_confirmed"
                        )
                        self._handover_protect_remaining = self.cfg.handover_dwell_protect_steps
                        executed = self._apply_handover_dwell_protection(marl_parsed)
                        self.controller.track_shadow(measurements, bounds, executed, ambient)
                    else:
                        pid_action, arb_reason = _run_pid()
                        executed = pid_action
                else:
                    pid_action, arb_reason = _run_pid()
                    executed = pid_action
            else:
                pid_action, arb_reason = _run_pid()
                executed = pid_action

        elif self.mode == ControlMode.RECOVERY_PENDING:
            if not marl_valid or not self._sensors_ready_for_marl(measurements):
                self._recovery_streak = 0
                trans_from, trans_reason = self._transition(
                    ControlMode.PID_FALLBACK, f"recovery_aborted:{marl_err or 'obs_unreliable'}"
                )
                pid_action, arb_reason = _run_pid()
                executed = pid_action
            else:
                assert marl_parsed is not None
                safe, safety_err = self._check_safety_and_divergence(
                    marl_parsed, measurements, bounds, ambient
                )
                if not safe:
                    self.metrics.unsafe_actions_intercepted += 1
                    self._recovery_streak = 0
                    trans_from, trans_reason = self._transition(
                        ControlMode.PID_FALLBACK, f"recovery_aborted:{safety_err}"
                    )
                    pid_action, arb_reason = _run_pid()
                    executed = pid_action
                else:
                    self._recovery_streak += 1
                    if self._recovery_streak >= self.cfg.N_confirm:
                        trans_from, trans_reason = self._transition(
                            ControlMode.MARL_ACTIVE, "recovery_confirmed"
                        )
                        self._handover_protect_remaining = self.cfg.handover_dwell_protect_steps
                        executed = self._apply_handover_dwell_protection(marl_parsed)
                        self.controller.track_shadow(measurements, bounds, executed, ambient)
                    else:
                        pid_action, arb_reason = _run_pid()
                        executed = pid_action

        # Software last-line safety clamp
        executed = self._enforce_software_interlocks(executed, measurements)

        # Update continuous actuator duration & toggle metrics
        self._cont_heat_steps = self._cont_heat_steps + 1 if executed[0] == 1 else 0
        self._cont_water_steps = self._cont_water_steps + 1 if executed[3] == 1 else 0
        if self._prev_executed is not None:
            self.metrics.actuator_toggles += sum(
                1 for a, b in zip(self._prev_executed, executed) if a != b
            )
        self._prev_executed = executed

        if self.mode == ControlMode.MARL_ACTIVE:
            self.metrics.marl_active_steps += 1
        elif self.mode == ControlMode.PID_FALLBACK:
            self.metrics.pid_fallback_steps += 1
        elif self.mode == ControlMode.RECOVERY_PENDING:
            self.metrics.recovery_pending_steps += 1
        elif self.mode == ControlMode.SAFE_MODE:
            self.metrics.safe_mode_steps += 1

        return ZoneStatus(
            zone_id=self.zone_id,
            mode=self.mode,
            executed_action=executed,
            pid_action=pid_action,
            marl_valid=marl_valid,
            marl_failure_reason=marl_err,
            recovery_streak=self._recovery_streak,
            transitioned_from=trans_from,
            transition_reason=trans_reason,
            arbiter_reason=arb_reason,
            measurement_qualities={v: measurements[v].quality for v in VARS},
        )
