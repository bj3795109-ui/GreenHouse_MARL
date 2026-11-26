"""
Pure discrete-time PID controller with deadband, derivative-on-measurement,
back-calculation + external-block anti-windup, and measurement-quality gating.

This module intentionally has ZERO imports from PettingZoo, MARL, or TrustNetwork.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from control.types import (
    CropBounds,
    MeasurementQuality,
    PIDGains,
    SensorMeasurement,
)


@dataclass
class PIDState:
    """Inspectable internal state of a single PID loop."""

    integral: float = 0.0
    derivative: float = 0.0
    prev_measurement: Optional[float] = None
    last_output: float = 0.0
    last_error: float = 0.0
    step_counter: int = 0
    elapsed_since_update: float = 0.0
    has_updated: bool = False
    active_bounds: Optional[CropBounds] = None
    enabled: bool = True


def _check_finite_number(val: object, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {type(val).__name__}")
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {val}")
    return fval


def validate_pid_gains(gains: PIDGains) -> None:
    """Validate finiteness and physical/timing bounds for PIDGains."""
    if not isinstance(gains, PIDGains):
        raise TypeError(f"Expected PIDGains, got {type(gains).__name__}")

    kp = _check_finite_number(gains.Kp, "Kp")
    ki = _check_finite_number(gains.Ki, "Ki")
    kd = _check_finite_number(gains.Kd, "Kd")
    tau_f = _check_finite_number(gains.tau_f, "tau_f")
    t_t = _check_finite_number(gains.T_t, "T_t")
    ts = _check_finite_number(gains.Ts, "Ts")
    dt = _check_finite_number(gains.dt, "dt")
    u_min = _check_finite_number(gains.u_min, "u_min")
    u_max = _check_finite_number(gains.u_max, "u_max")

    if kp < 0.0:
        raise ValueError(f"Kp must be >= 0, got {kp}")
    if ki < 0.0:
        raise ValueError(f"Ki must be >= 0, got {ki}")
    if kd < 0.0:
        raise ValueError(f"Kd must be >= 0, got {kd}")
    if dt <= 0.0:
        raise ValueError(f"Base timestep dt must be > 0, got {dt}")
    if ts < dt:
        raise ValueError(f"Sampling period Ts ({ts}) must be >= dt ({dt})")
    if u_min >= u_max:
        raise ValueError(f"u_min ({u_min}) must be < u_max ({u_max})")
    if gains.unidirectional_positive and u_max <= 0.0:
        raise ValueError(f"u_max ({u_max}) must be > 0 for unidirectional_positive loop")
    if tau_f < 0.0:
        raise ValueError(f"tau_f must be >= 0, got {tau_f}")
    if t_t <= 0.0:
        raise ValueError(f"T_t must be > 0, got {t_t}")


def _validate_bounds(bounds: CropBounds) -> None:
    if not isinstance(bounds, CropBounds):
        raise TypeError(f"Expected CropBounds, got {type(bounds).__name__}")
    min_a = _check_finite_number(bounds.min_acceptable, "bounds.min_acceptable")
    opt_min = _check_finite_number(bounds.optimal_min, "bounds.optimal_min")
    opt_max = _check_finite_number(bounds.optimal_max, "bounds.optimal_max")
    max_a = _check_finite_number(bounds.max_acceptable, "bounds.max_acceptable")
    if opt_min > opt_max:
        raise ValueError(
            f"CropBounds violates optimal_min <= optimal_max: ({opt_min}, {opt_max})"
        )
    if min_a > max_a:
        raise ValueError(
            f"CropBounds violates min_acceptable <= max_acceptable: ({min_a}, {max_a})"
        )


class PIDController:
    """
    Independent single-variable discrete PID controller.

    Features:
    - Range deadband error against [optimal_min, optimal_max].
    - Filtered derivative on measurement (no derivative kick on stage/target changes).
    - Output saturation [u_min, u_max] (clamped to >= 0 when unidirectional_positive)
      with back-calculation anti-windup (T_t) and integral unwinding above optimal_max.
    - External actuator-block clamping so blocked fan/irrigation never winds up I.
    - Multi-rate sampling gate (dt vs. Ts) and optional variable timestep `dt` in step().
    - Measurement quality handling (TRUSTED / ESTIMATE_ONLY / HOLD_TRANSIENT / UNRELIABLE_FAILSAFE).
    - Explicit enable/disable lifecycle and shadow-tracking mode for bumpless handover.
    """

    def __init__(self, var: str, gains: PIDGains, enabled: bool = True):
        validate_pid_gains(gains)
        self.var = var
        self.gains = gains
        self.state = PIDState(enabled=bool(enabled))

    @property
    def enabled(self) -> bool:
        return self.state.enabled

    def enable(self) -> None:
        """Enable controller execution cleanly without derivative kick."""
        if not self.state.enabled:
            self.state.enabled = True
            self.state.derivative = 0.0
            self.state.prev_measurement = None
            self.state.elapsed_since_update = 0.0
            self.state.has_updated = False

    def disable(self) -> None:
        """Disable controller and clear active output/derivative drive."""
        self.state.enabled = False
        self.state.derivative = 0.0
        self.state.prev_measurement = None
        self.state.last_error = 0.0
        self.state.last_output = 0.0
        self.state.elapsed_since_update = 0.0
        self.state.has_updated = False

    def reset(self) -> None:
        self.state = PIDState(enabled=self.state.enabled)

    def update_gains(self, gains: PIDGains) -> None:
        validate_pid_gains(gains)
        self.gains = gains

    def set_targets(self, bounds: CropBounds) -> None:
        """
        Update crop target range without modifying PID gains.
        Because derivative acts on measurement, changing targets introduces
        zero derivative kick. If the optimal band shifts past the old band in
        a direction opposite to stored integral, clear integral to avoid stale drive.
        """
        _validate_bounds(bounds)
        prev = self.state.active_bounds
        if prev is not None:
            if bounds.optimal_min > prev.optimal_max and self.state.integral < 0.0:
                self.state.integral = 0.0
            elif bounds.optimal_max < prev.optimal_min and self.state.integral > 0.0:
                self.state.integral = 0.0
        self.state.active_bounds = bounds

    def _sync_active_bounds(self, bounds: CropBounds) -> None:
        _validate_bounds(bounds)
        prev = self.state.active_bounds
        if prev is None or (
            prev.crop_id != bounds.crop_id
            or prev.stage != bounds.stage
            or prev.optimal_min != bounds.optimal_min
            or prev.optimal_max != bounds.optimal_max
            or prev.min_acceptable != bounds.min_acceptable
            or prev.max_acceptable != bounds.max_acceptable
        ):
            self.set_targets(bounds)

    def _effective_u_min(self) -> float:
        return max(0.0, self.gains.u_min) if self.gains.unidirectional_positive else self.gains.u_min

    def compute_deadband_error(self, measurement: float, bounds: CropBounds) -> float:
        """Compute signed deadband error outside [optimal_min, optimal_max]."""
        y = _check_finite_number(measurement, "measurement")
        _validate_bounds(bounds)
        if y < bounds.optimal_min:
            return bounds.optimal_min - y
        if y > bounds.optimal_max:
            if self.gains.unidirectional_positive:
                return 0.0
            return bounds.optimal_max - y
        return 0.0

    def _clear_active_drive(self) -> None:
        self.state.derivative = 0.0
        self.state.prev_measurement = None
        self.state.last_error = 0.0
        self.state.last_output = 0.0

    def step(
        self,
        measurement: SensorMeasurement,
        bounds: CropBounds,
        block_positive: bool = False,
        block_negative: bool = False,
        dt: Optional[float] = None,
    ) -> float:
        """
        Compute control output for one control tick.

        Parameters
        ----------
        measurement : SensorMeasurement
        bounds : CropBounds
        block_positive : bool
            True when external arbiter/safety blocks positive actuation (e.g., soak lockout).
        block_negative : bool
            True when external arbiter blocks negative actuation (e.g., counterproductive fan).
        dt : float | None
            Optional actual elapsed timestep in seconds (> 0, finite). Defaults to gains.dt.
        """
        self._sync_active_bounds(bounds)
        g = self.gains

        step_dt = g.dt if dt is None else _check_finite_number(dt, "dt")
        if step_dt <= 0.0:
            raise ValueError(f"Timestep dt must be > 0, got {step_dt}")

        quality = measurement.quality

        # Fail-safe: prolonged missing/unreliable sensing -> zero energy/water output
        if quality == MeasurementQuality.UNRELIABLE_FAILSAFE:
            self._clear_active_drive()
            self.state.derivative = 0.0
            self.state.prev_measurement = None
            self.state.last_error = 0.0
            self.state.last_output = 0.0
            # Stale effort must not survive an outage and drive actuators on recovery.
            self.state.integral = 0.0
            return 0.0

        # Validate measurement value strictly for non-failsafe inputs:
        # HOLD_TRANSIENT permits value=None (holds last safe output), but never NaN/Inf.
        # TRUSTED and ESTIMATE_ONLY require a finite numeric value.
        if measurement.quality == MeasurementQuality.HOLD_TRANSIENT:
            if measurement.value is not None:
                try:
                    _check_finite_number(measurement.value, "measurement.value")
                except ValueError:
                    self._clear_active_drive()
                    raise
            if not self.state.enabled:
                self._clear_active_drive()
                return 0.0
            self.state.derivative = 0.0
            self.state.prev_measurement = None
            return self.state.last_output

        if measurement.value is None:
            self._clear_active_drive()
            raise ValueError(f"Missing measurement value for {self.var} with quality {quality.value}")

        try:
            y = _check_finite_number(measurement.value, "measurement.value")
        except ValueError:
            self._clear_active_drive()
            raise

        if not self.state.enabled:
            self._clear_active_drive()
            return 0.0

        # Determine effective elapsed timestep and multi-rate sampling gate
        if g.Ts > g.dt:
            if not self.state.has_updated:
                dt_eff = max(g.Ts, step_dt) if dt is not None else g.Ts
            else:
                self.state.elapsed_since_update += step_dt
                if self.state.elapsed_since_update + 1e-9 < g.Ts:
                    return self.state.last_output
                dt_eff = self.state.elapsed_since_update
            self.state.elapsed_since_update = 0.0
            self.state.has_updated = True
        else:
            dt_eff = step_dt
            self.state.elapsed_since_update = 0.0
            self.state.has_updated = True

        u_low = self._effective_u_min()
        u_high = g.u_max

        # Unidirectional positive loop (e.g. soil moisture irrigation):
        # never output positive drive above optimal_max, and unwind any stored positive integral.
        if g.unidirectional_positive and y > bounds.optimal_max:
            excess_err = bounds.optimal_max - y  # < 0
            decayed = self.state.integral * math.exp(-dt_eff / g.T_t) + g.Ki * dt_eff * excess_err
            self.state.integral = max(u_low, min(u_high, decayed))
            self.state.derivative = 0.0
            self.state.prev_measurement = y
            self.state.last_error = 0.0
            self.state.last_output = max(0.0, u_low)
            self.state.step_counter += 1
            return self.state.last_output

        e = self.compute_deadband_error(y, bounds)
        self.state.last_error = e

        # Proportional
        P = g.Kp * e

        # Filtered Derivative-on-Measurement.
        # Inside the deadband the loop is deliberately not regulating, so D must not
        # request actuation there either (it still brakes while outside the band).
        if e == 0.0 or self.state.prev_measurement is None or g.Kd == 0.0:
            D = 0.0
        else:
            dy = y - self.state.prev_measurement
            denom = g.tau_f + dt_eff
            D = (g.tau_f / denom) * self.state.derivative - (g.Kd / denom) * dy

        # Provisional Integral.
        # Inside the deadband (e == 0) stored effort bleeds off with time constant T_t
        # instead of being held forever and driving actuators while the zone is in range.
        if e == 0.0:
            I_cand = self.state.integral * (1.0 - min(1.0, dt_eff / g.T_t))
        else:
            I_cand = self.state.integral + g.Ki * dt_eff * e

        # Unsaturated command & saturation
        v = P + I_cand + D
        u = max(u_low, min(u_high, v))

        # Back-calculation anti-windup
        I_new = I_cand + (dt_eff / g.T_t) * (u - v)

        # External actuator-block clamping (prevents windup during fan conflict / soak lockout)
        if block_positive and e > 0.0 and I_new > self.state.integral:
            I_new = self.state.integral
        if block_negative and e < 0.0 and I_new < self.state.integral:
            I_new = self.state.integral

        # Clamp integral within effective actuator bounds
        I_new = max(u_low, min(u_high, I_new))

        self.state.integral = I_new
        self.state.derivative = D
        self.state.prev_measurement = y
        self.state.last_output = u
        self.state.step_counter += 1
        return u

    def track_shadow(
        self,
        measurement: SensorMeasurement,
        bounds: CropBounds,
        effective_u: float,
    ) -> None:
        """
        Shadow-tracking update while MARL/SAFE_MODE is active, enabling bumpless handover.

        Aligns integral state when outside the optimal deadband so P + I matches the
        bounded effective command `effective_u`, avoids unintended integral accumulation
        inside the optimal deadband, and keeps prev_measurement warm so D starts at 0.
        """
        self._sync_active_bounds(bounds)
        eff_u = _check_finite_number(effective_u, "effective_u")

        g = self.gains
        self.state.derivative = 0.0
        self.state.elapsed_since_update = 0.0
        self.state.has_updated = False

        if (
            measurement.quality in (MeasurementQuality.TRUSTED, MeasurementQuality.ESTIMATE_ONLY)
            and measurement.value is not None
        ):
            y = _check_finite_number(measurement.value, "measurement.value")
            e = self.compute_deadband_error(y, bounds)
            P = g.Kp * e
            u_low = self._effective_u_min()
            target_u = max(u_low, min(g.u_max, eff_u))
            alpha_track = min(1.0, g.Ts / g.T_t)

            if e == 0.0:
                # Inside optimal deadband (or unidirectional excess): do not wind up integral
                desired_I = 0.0
                target_u = max(0.0, u_low) if g.unidirectional_positive else 0.0
            else:
                desired_I = max(u_low, min(g.u_max, target_u - P))

            self.state.integral = (1.0 - alpha_track) * self.state.integral + alpha_track * desired_I
            self.state.prev_measurement = y
            self.state.last_error = e
            self.state.last_output = target_u
        else:
            if measurement.quality == MeasurementQuality.HOLD_TRANSIENT and measurement.value is not None:
                _check_finite_number(measurement.value, "measurement.value")
            self.state.prev_measurement = None
