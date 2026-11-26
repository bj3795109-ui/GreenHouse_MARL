"""
Per-zone MARL-PID Supervisor state machine.

Manages transitions among MARL_ACTIVE, PID_FALLBACK, RECOVERY_PENDING, and SAFE_MODE
independently for a single greenhouse zone, including strict action validation,
persistent boundary-divergence detection, recovery validation, and bumpless handover.

NOTE: Software SAFE_MODE and safety checks complement—but NEVER replace—the
hardware microcontroller (ESP32) safety interlocks defined in V1.1 §0.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

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

logger = logging.getLogger(__name__)

VALID_TRANSITIONS: dict[ControlMode, frozenset[ControlMode]] = {
    ControlMode.MARL_ACTIVE: frozenset({
        ControlMode.PID_FALLBACK,
        ControlMode.SAFE_MODE,
    }),
    ControlMode.PID_FALLBACK: frozenset({
        ControlMode.RECOVERY_PENDING,
        ControlMode.SAFE_MODE,
    }),
    ControlMode.RECOVERY_PENDING: frozenset({
        ControlMode.MARL_ACTIVE,
        ControlMode.PID_FALLBACK,
        ControlMode.SAFE_MODE,
    }),
    ControlMode.SAFE_MODE: frozenset({
        ControlMode.PID_FALLBACK,
    }),
}

_SAFE_ACTION: tuple[int, int, int, int] = (0, 0, 0, 0)
_PID_RECOVERABLE_ERRORS = (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError)


def _is_finite_num(val: Any) -> bool:
    if isinstance(val, (bool, str, bytes)) or val is None:
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


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
    safe_mode_retry_steps: int = 1

    # Software defense-in-depth limits (mirrors V1.1 §0 hardware interlocks)
    T_max_hard: float = 45.0
    t_max_continuous_heat: int = 120
    t_max_continuous_water: int = 20

    def __post_init__(self) -> None:
        if self.N_confirm < 1:
            raise ValueError(f"N_confirm must be >= 1, got {self.N_confirm}")
        if self.W_persist < 1:
            raise ValueError(f"W_persist must be >= 1, got {self.W_persist}")
        if self.K_contra < 1:
            raise ValueError(f"K_contra must be >= 1, got {self.K_contra}")
        if not (_is_finite_num(self.theta_marl_min_trust) and 0.0 <= float(self.theta_marl_min_trust) <= 1.0):
            raise ValueError(f"theta_marl_min_trust must be in [0, 1], got {self.theta_marl_min_trust}")
        if not (_is_finite_num(self.theta_recovery_trust) and 0.0 <= float(self.theta_recovery_trust) <= 1.0):
            raise ValueError(f"theta_recovery_trust must be in [0, 1], got {self.theta_recovery_trust}")
        if self.handover_dwell_protect_steps < 0:
            raise ValueError(f"handover_dwell_protect_steps must be >= 0, got {self.handover_dwell_protect_steps}")
        if self.safe_mode_retry_steps < 1:
            raise ValueError(f"safe_mode_retry_steps must be >= 1, got {self.safe_mode_retry_steps}")
        if not _is_finite_num(self.T_max_hard):
            raise ValueError(f"T_max_hard must be finite, got {self.T_max_hard}")
        if self.t_max_continuous_heat < 1:
            raise ValueError(f"t_max_continuous_heat must be >= 1, got {self.t_max_continuous_heat}")
        if self.t_max_continuous_water < 1:
            raise ValueError(f"t_max_continuous_water must be >= 1, got {self.t_max_continuous_water}")


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
    pid_failures: int = 0
    invalid_transitions_rejected: int = 0
    actuator_toggles: int = 0
    last_fault_reason: Optional[str] = None
    fault_history: list[tuple[int, str]] = field(default_factory=list)
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
        if not isinstance(zone_id, str) or not zone_id.strip():
            raise ValueError(f"zone_id must be a non-empty string, got {zone_id!r}")
        if not isinstance(initial_mode, ControlMode):
            raise ValueError(f"initial_mode must be a ControlMode, got {initial_mode!r}")
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
        self._pid_fault_active: bool = False
        self._safe_mode_cooldown: int = 0

    def _record_fault(self, reason: str) -> None:
        self.metrics.last_fault_reason = reason
        self.metrics.fault_history.append((self.metrics.total_steps, reason))

    def transition_to(self, new_mode: ControlMode, reason: str) -> tuple[Optional[ControlMode], Optional[str]]:
        """Validate and perform an explicit FSM transition using VALID_TRANSITIONS."""
        if not isinstance(new_mode, ControlMode):
            self.metrics.invalid_transitions_rejected += 1
            msg = f"invalid_transition_target:{new_mode!r}"
            self._record_fault(msg)
            logger.warning("Zone %s rejected non-ControlMode transition: %s", self.zone_id, msg)
            raise ValueError(msg)

        if new_mode == self.mode:
            return (None, None)

        allowed = VALID_TRANSITIONS.get(self.mode, frozenset())
        if new_mode not in allowed:
            self.metrics.invalid_transitions_rejected += 1
            msg = f"invalid_transition:{self.mode.value}->{new_mode.value}:{reason}"
            self._record_fault(msg)
            logger.warning("Zone %s rejected illegal FSM transition: %s", self.zone_id, msg)
            raise ValueError(msg)

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

        # Reset mode-scoped transient counters at transition boundaries.
        # Keep _persist_bad_steps / _contra_steps across MARL_ACTIVE -> PID_FALLBACK
        # so an agent that just tripped persistent divergence cannot re-enter recovery
        # while still proposing the same faulty action; clear them on MARL_ACTIVE / SAFE_MODE.
        if new_mode == ControlMode.PID_FALLBACK:
            self._recovery_streak = 0
            self._handover_protect_remaining = 0
        elif new_mode in (ControlMode.SAFE_MODE, ControlMode.MARL_ACTIVE):
            self._recovery_streak = 0
            self._persist_bad_steps = 0
            self._contra_steps = 0
            if new_mode == ControlMode.SAFE_MODE:
                self._handover_protect_remaining = 0

        return (old, reason)

    def _transition(self, new_mode: ControlMode, reason: str) -> tuple[Optional[ControlMode], Optional[str]]:
        return self.transition_to(new_mode, reason)

    def _validate_measurements_structure(
        self, measurements: Any
    ) -> tuple[bool, Optional[str]]:
        if not isinstance(measurements, Mapping):
            return (False, "malformed_measurements_container")
        for v in VARS:
            m = measurements.get(v)
            if not isinstance(m, SensorMeasurement):
                return (False, f"missing_or_invalid_measurement:{v}")
            if m.quality == MeasurementQuality.UNRELIABLE_FAILSAFE:
                return (False, f"unreliable_failsafe:{v}")
            if m.quality in (MeasurementQuality.TRUSTED, MeasurementQuality.ESTIMATE_ONLY):
                if not _is_finite_num(m.value):
                    return (False, f"non_finite_measurement:{v}")
            elif m.quality == MeasurementQuality.HOLD_TRANSIENT:
                if m.value is not None and not _is_finite_num(m.value):
                    return (False, f"non_finite_hold_measurement:{v}")
            if not _is_finite_num(m.trust) or not _is_finite_num(m.confidence):
                return (False, f"non_finite_trust_or_conf:{v}")
        return (True, None)

    def _validate_bounds_and_ambient(
        self, bounds: Any, ambient: Any
    ) -> tuple[bool, Optional[str]]:
        if not isinstance(bounds, Mapping):
            return (False, "malformed_bounds_container")
        for v in VARS:
            b = bounds.get(v)
            if not isinstance(b, CropBounds):
                return (False, f"missing_or_invalid_bounds:{v}")
            if not (
                _is_finite_num(b.min_acceptable)
                and _is_finite_num(b.optimal_min)
                and _is_finite_num(b.optimal_max)
                and _is_finite_num(b.max_acceptable)
            ):
                return (False, f"non_finite_bounds:{v}")
            if b.optimal_min > b.optimal_max or b.min_acceptable > b.max_acceptable:
                return (False, f"invalid_bounds_order:{v}")
        if not isinstance(ambient, AmbientState):
            return (False, "invalid_ambient_type")
        if ambient.T_ambient is not None and not _is_finite_num(ambient.T_ambient):
            return (False, "non_finite_ambient_T")
        if ambient.H_ambient is not None and not _is_finite_num(ambient.H_ambient):
            return (False, "non_finite_ambient_H")
        return (True, None)

    def _sensors_unreliable(self, measurements: Mapping[str, SensorMeasurement]) -> bool:
        ok, _ = self._validate_measurements_structure(measurements)
        return not ok

    def _sensors_ready_for_marl(self, measurements: Mapping[str, SensorMeasurement]) -> bool:
        ok, _ = self._validate_measurements_structure(measurements)
        if not ok:
            return False
        for v in VARS:
            m = measurements[v]
            if m.quality in (MeasurementQuality.HOLD_TRANSIENT, MeasurementQuality.UNRELIABLE_FAILSAFE):
                return False
            if self.cfg.marl_requires_trusted_raw and float(m.trust) < self.cfg.theta_recovery_trust:
                return False
        return True

    def _check_safety_and_divergence(
        self,
        action: tuple[int, int, int, int],
        measurements: Mapping[str, SensorMeasurement],
        bounds: Mapping[str, CropBounds],
        ambient: AmbientState,
        strict_recovery: bool = False,
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluate whether a syntactically valid MARL action is safe and non-divergent.
        When `strict_recovery=True` (used for recovery validation), any single-step
        contradictory or boundary-divergent action is rejected without waiting for
        W_persist / K_contra accumulation.
        Returns (is_safe, failure_reason).
        """
        heat, fan, mist, water = action
        T_val = measurements["T"].value
        H_val = measurements["H"].value
        M_val = measurements["M"].value

        for name, val in (("T", T_val), ("H", H_val), ("M", M_val)):
            if val is not None and not _is_finite_num(val):
                return (False, f"non_finite_measurement:{name}")

        if ambient.T_ambient is not None and not _is_finite_num(ambient.T_ambient):
            return (False, "non_finite_ambient_T")
        if ambient.H_ambient is not None and not _is_finite_num(ambient.H_ambient):
            return (False, "non_finite_ambient_H")

        # 1. Hard over-temperature check (conservative when T unknown during heat)
        if T_val is not None and float(T_val) > self.cfg.T_max_hard and heat == 1:
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
            and float(T_val) > float(ambient.T_ambient) + 0.5
        )
        fan_dries = (
            fan == 1
            and H_val is not None
            and ambient.H_ambient is not None
            and float(H_val) > float(ambient.H_ambient) + 1.5
        )
        contradictory = (heat == 1 and fan_cools) or (mist == 1 and fan_dries)
        if strict_recovery:
            if contradictory:
                return (False, "contradictory_actuation")
        else:
            if contradictory:
                self._contra_steps += 1
                if self._contra_steps >= self.cfg.K_contra:
                    return (False, "persistent_contradictory_actuation")
            else:
                self._contra_steps = 0

        # 4. Monitored persistent boundary-divergence detection
        diverging = False
        if T_val is not None:
            t_f = float(T_val)
            if t_f > bounds["T"].max_acceptable and heat == 1:
                diverging = True
            elif t_f < bounds["T"].min_acceptable and heat == 0:
                diverging = True
        if H_val is not None:
            h_f = float(H_val)
            if h_f < bounds["H"].min_acceptable and mist == 0:
                diverging = True
            elif h_f > bounds["H"].max_acceptable and mist == 1:
                diverging = True
        if M_val is not None:
            m_f = float(M_val)
            if m_f < bounds["M"].min_acceptable and water == 0:
                diverging = True
            elif m_f > bounds["M"].max_acceptable and water == 1:
                diverging = True

        if strict_recovery:
            if diverging:
                return (False, "boundary_divergence")
        else:
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
        action: Any,
        measurements: Any,
    ) -> tuple[int, int, int, int]:
        """Last-line NaN-safe software safety clamp (never replaces ESP32 hardware interlocks)."""
        ok, parsed, _ = validate_marl_action(action)
        if not ok or parsed is None:
            return _SAFE_ACTION

        heat, fan, mist, water = parsed
        if not isinstance(measurements, Mapping):
            return _SAFE_ACTION

        m_T = measurements.get("T")
        T_val = m_T.value if isinstance(m_T, SensorMeasurement) else None
        if T_val is not None and (not _is_finite_num(T_val) or float(T_val) > self.cfg.T_max_hard):
            heat = 0

        m_H = measurements.get("H")
        H_val = m_H.value if isinstance(m_H, SensorMeasurement) else None
        if H_val is not None and not _is_finite_num(H_val):
            mist = 0

        m_M = measurements.get("M")
        M_val = m_M.value if isinstance(m_M, SensorMeasurement) else None
        if M_val is not None and not _is_finite_num(M_val):
            water = 0

        if heat == 1 and (self._cont_heat_steps + 1) > self.cfg.t_max_continuous_heat:
            heat = 0
        if water == 1 and (self._cont_water_steps + 1) > self.cfg.t_max_continuous_water:
            water = 0
        return (heat, fan, mist, water)

    def _safe_track_shadow(
        self,
        measurements: Any,
        bounds: Any,
        executed: tuple[int, int, int, int],
        ambient: Any,
    ) -> None:
        try:
            self.controller.track_shadow(measurements, bounds, executed, ambient)
        except _PID_RECOVERABLE_ERRORS as exc:
            logger.debug("Zone %s track_shadow skipped on invalid state: %s", self.zone_id, exc)

    def step(
        self,
        raw_marl_action: Any,
        measurements: Mapping[str, SensorMeasurement],
        bounds: Mapping[str, CropBounds],
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

        # Count intercepted invalid MARL actions only when MARL is active, recovering,
        # or explicitly supplied a malformed non-None payload while not marked offline.
        if not marl_valid and (
            self.mode in (ControlMode.MARL_ACTIVE, ControlMode.RECOVERY_PENDING)
            or (not agent_failed and raw_marl_action is not None)
        ):
            self.metrics.unsafe_actions_intercepted += 1

        meas_ok, meas_err = self._validate_measurements_structure(measurements)
        bounds_ok, bounds_err = self._validate_bounds_and_ambient(bounds, ambient)
        inputs_failsafe = not meas_ok or not bounds_ok
        failsafe_reason = meas_err or bounds_err or "measurements_unreliable_failsafe"

        marl_obs_degraded = (
            meas_ok
            and self.cfg.marl_requires_trusted_raw
            and any(
                float(measurements[v].trust) < self.cfg.theta_marl_min_trust
                or measurements[v].quality != MeasurementQuality.TRUSTED
                for v in VARS
            )
        )

        pid_action = _SAFE_ACTION
        arb_reason = ""

        def _try_run_pid() -> tuple[bool, tuple[int, int, int, int], str, Optional[str]]:
            try:
                out = self.controller.step(measurements, bounds, ambient)
                ok_act, parsed_act, act_err = validate_marl_action(out.action)
                if not ok_act or parsed_act is None:
                    raise ValueError(f"PID produced invalid action {out.action!r}: {act_err}")
                if not (_is_finite_num(out.u_T) and _is_finite_num(out.u_H) and _is_finite_num(out.u_M)):
                    raise ValueError("PID produced non-finite continuous output")
                return (True, parsed_act, out.arbiter_reason, None)
            except _PID_RECOVERABLE_ERRORS as exc:
                err_msg = f"pid_failure:{type(exc).__name__}:{exc}"
                logger.exception("Zone %s controller failure; forcing SAFE_MODE: %s", self.zone_id, err_msg)
                return (False, _SAFE_ACTION, err_msg, err_msg)

        if inputs_failsafe:
            trans_from, trans_reason = self._transition(ControlMode.SAFE_MODE, failsafe_reason)
            if trans_reason is not None:
                self._record_fault(failsafe_reason)
            executed = _SAFE_ACTION
            if bounds_ok and isinstance(measurements, Mapping):
                self._safe_track_shadow(measurements, bounds, executed, ambient)

        elif self.mode == ControlMode.SAFE_MODE:
            if self._pid_fault_active:
                if self._safe_mode_cooldown > 1:
                    self._safe_mode_cooldown -= 1
                    executed = _SAFE_ACTION
                    arb_reason = "safe_mode_pid_cooldown"
                else:
                    pid_ok, probed_act, probed_reason, pid_err = _try_run_pid()
                    if pid_ok:
                        self._pid_fault_active = False
                        self._safe_mode_cooldown = 0
                        trans_from, trans_reason = self._transition(
                            ControlMode.PID_FALLBACK, "pid_and_measurements_restored_from_safe_mode"
                        )
                        pid_action = probed_act
                        arb_reason = probed_reason
                        executed = pid_action
                    else:
                        self.metrics.pid_failures += 1
                        self._safe_mode_cooldown = self.cfg.safe_mode_retry_steps
                        self._record_fault(pid_err or "pid_failure")
                        executed = _SAFE_ACTION
                        arb_reason = probed_reason
            else:
                pid_ok, probed_act, probed_reason, pid_err = _try_run_pid()
                if pid_ok:
                    trans_from, trans_reason = self._transition(
                        ControlMode.PID_FALLBACK, "measurements_restored_from_safe_mode"
                    )
                    pid_action = probed_act
                    arb_reason = probed_reason
                    executed = pid_action
                else:
                    self.metrics.pid_failures += 1
                    self._pid_fault_active = True
                    self._safe_mode_cooldown = self.cfg.safe_mode_retry_steps
                    self._record_fault(pid_err or "pid_failure")
                    executed = _SAFE_ACTION
                    arb_reason = probed_reason

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
                pid_ok, pid_act, pid_arb, pid_err = _try_run_pid()
                if pid_ok:
                    pid_action, arb_reason = pid_act, pid_arb
                    executed = pid_action
                else:
                    self.metrics.pid_failures += 1
                    self._pid_fault_active = True
                    self._safe_mode_cooldown = self.cfg.safe_mode_retry_steps
                    self._record_fault(pid_err or "pid_failure")
                    trans_from, trans_reason = self._transition(
                        ControlMode.SAFE_MODE, pid_err or "pid_failure"
                    )
                    executed = _SAFE_ACTION
                    arb_reason = pid_arb
            else:
                assert marl_parsed is not None
                executed = self._apply_handover_dwell_protection(marl_parsed)
                self._safe_track_shadow(measurements, bounds, executed, ambient)

        elif self.mode == ControlMode.PID_FALLBACK:
            entered_recovery = False
            if marl_valid and self._sensors_ready_for_marl(measurements):
                assert marl_parsed is not None
                safe, _ = self._check_safety_and_divergence(
                    marl_parsed, measurements, bounds, ambient
                )
                if safe:
                    entered_recovery = True
                    trans_from, trans_reason = self._transition(
                        ControlMode.RECOVERY_PENDING, "marl_candidate_healthy"
                    )
                    self._recovery_streak = 1
                    if self._recovery_streak >= self.cfg.N_confirm:
                        trans_from, trans_reason = self._transition(
                            ControlMode.MARL_ACTIVE, "recovery_confirmed"
                        )
                        self._handover_protect_remaining = self.cfg.handover_dwell_protect_steps
                        executed = self._apply_handover_dwell_protection(marl_parsed)
                        self._safe_track_shadow(measurements, bounds, executed, ambient)
                    else:
                        pid_ok, pid_act, pid_arb, pid_err = _try_run_pid()
                        if pid_ok:
                            pid_action, arb_reason = pid_act, pid_arb
                            executed = pid_action
                        else:
                            self.metrics.pid_failures += 1
                            self._pid_fault_active = True
                            self._safe_mode_cooldown = self.cfg.safe_mode_retry_steps
                            self._record_fault(pid_err or "pid_failure")
                            trans_from, trans_reason = self._transition(
                                ControlMode.SAFE_MODE, pid_err or "pid_failure"
                            )
                            executed = _SAFE_ACTION
                            arb_reason = pid_arb

            if not entered_recovery:
                pid_ok, pid_act, pid_arb, pid_err = _try_run_pid()
                if pid_ok:
                    pid_action, arb_reason = pid_act, pid_arb
                    executed = pid_action
                else:
                    self.metrics.pid_failures += 1
                    self._pid_fault_active = True
                    self._safe_mode_cooldown = self.cfg.safe_mode_retry_steps
                    self._record_fault(pid_err or "pid_failure")
                    trans_from, trans_reason = self._transition(
                        ControlMode.SAFE_MODE, pid_err or "pid_failure"
                    )
                    executed = _SAFE_ACTION
                    arb_reason = pid_arb

        elif self.mode == ControlMode.RECOVERY_PENDING:
            if not marl_valid or not self._sensors_ready_for_marl(measurements):
                trans_from, trans_reason = self._transition(
                    ControlMode.PID_FALLBACK, f"recovery_aborted:{marl_err or 'obs_unreliable'}"
                )
                pid_ok, pid_act, pid_arb, pid_err = _try_run_pid()
                if pid_ok:
                    pid_action, arb_reason = pid_act, pid_arb
                    executed = pid_action
                else:
                    self.metrics.pid_failures += 1
                    self._pid_fault_active = True
                    self._safe_mode_cooldown = self.cfg.safe_mode_retry_steps
                    self._record_fault(pid_err or "pid_failure")
                    trans_from, trans_reason = self._transition(
                        ControlMode.SAFE_MODE, pid_err or "pid_failure"
                    )
                    executed = _SAFE_ACTION
                    arb_reason = pid_arb
            else:
                assert marl_parsed is not None
                safe, safety_err = self._check_safety_and_divergence(
                    marl_parsed, measurements, bounds, ambient
                )
                if not safe:
                    self.metrics.unsafe_actions_intercepted += 1
                    trans_from, trans_reason = self._transition(
                        ControlMode.PID_FALLBACK, f"recovery_aborted:{safety_err}"
                    )
                    pid_ok, pid_act, pid_arb, pid_err = _try_run_pid()
                    if pid_ok:
                        pid_action, arb_reason = pid_act, pid_arb
                        executed = pid_action
                    else:
                        self.metrics.pid_failures += 1
                        self._pid_fault_active = True
                        self._safe_mode_cooldown = self.cfg.safe_mode_retry_steps
                        self._record_fault(pid_err or "pid_failure")
                        trans_from, trans_reason = self._transition(
                            ControlMode.SAFE_MODE, pid_err or "pid_failure"
                        )
                        executed = _SAFE_ACTION
                        arb_reason = pid_arb
                else:
                    self._recovery_streak += 1
                    if self._recovery_streak >= self.cfg.N_confirm:
                        trans_from, trans_reason = self._transition(
                            ControlMode.MARL_ACTIVE, "recovery_confirmed"
                        )
                        self._handover_protect_remaining = self.cfg.handover_dwell_protect_steps
                        executed = self._apply_handover_dwell_protection(marl_parsed)
                        self._safe_track_shadow(measurements, bounds, executed, ambient)
                    else:
                        pid_ok, pid_act, pid_arb, pid_err = _try_run_pid()
                        if pid_ok:
                            pid_action, arb_reason = pid_act, pid_arb
                            executed = pid_action
                        else:
                            self.metrics.pid_failures += 1
                            self._pid_fault_active = True
                            self._safe_mode_cooldown = self.cfg.safe_mode_retry_steps
                            self._record_fault(pid_err or "pid_failure")
                            trans_from, trans_reason = self._transition(
                                ControlMode.SAFE_MODE, pid_err or "pid_failure"
                            )
                            executed = _SAFE_ACTION
                            arb_reason = pid_arb

        # Software last-line safety clamp (never replaces hardware interlocks)
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

        qualities = (
            {
                v: (
                    measurements[v].quality
                    if isinstance(measurements.get(v), SensorMeasurement)
                    else MeasurementQuality.UNRELIABLE_FAILSAFE
                )
                for v in VARS
            }
            if isinstance(measurements, Mapping)
            else {v: MeasurementQuality.UNRELIABLE_FAILSAFE for v in VARS}
        )

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
            measurement_qualities=qualities,
        )
