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


class PIDController:
    """
    Independent single-variable discrete PID controller.

    Features:
    - Range deadband error against [optimal_min, optimal_max].
    - Filtered derivative on measurement (no derivative kick on stage/target changes).
    - Output saturation [u_min, u_max] with back-calculation anti-windup (T_t).
    - External actuator-block clamping so blocked fan/irrigation never winds up I.
    - Multi-rate sampling gate (dt vs. Ts) so Ts > dt never over-integrates.
    - Measurement quality handling (TRUSTED / ESTIMATE_ONLY / HOLD_TRANSIENT / UNRELIABLE_FAILSAFE).
    - Shadow-tracking mode for bumpless MARL -> PID handover.
    """

    def __init__(self, var: str, gains: PIDGains):
        if gains.dt <= 0.0:
            raise ValueError(f"Base timestep dt must be > 0, got {gains.dt}")
        if gains.Ts < gains.dt:
            raise ValueError(f"Sampling period Ts ({gains.Ts}) must be >= dt ({gains.dt})")
        if gains.u_min >= gains.u_max:
            raise ValueError(f"u_min ({gains.u_min}) must be < u_max ({gains.u_max})")
        if gains.tau_f < 0.0:
            raise ValueError(f"tau_f must be >= 0, got {gains.tau_f}")
        if gains.T_t <= 0.0:
            raise ValueError(f"T_t must be > 0, got {gains.T_t}")

        self.var = var
        self.gains = gains
        self.state = PIDState()

    def reset(self) -> None:
        self.state = PIDState()

    def update_gains(self, gains: PIDGains) -> None:
        self.gains = gains

    def set_targets(self, bounds: CropBounds) -> None:
        """
        Update crop target range without modifying PID gains.
        Because derivative acts on measurement, changing targets introduces
        zero derivative kick. If the optimal band shifts past the old band in
        a direction opposite to stored integral, clear integral to avoid stale drive.
        """
        prev = self.state.active_bounds
        if prev is not None:
            if bounds.optimal_min > prev.optimal_max and self.state.integral < 0.0:
                self.state.integral = 0.0
            elif bounds.optimal_max < prev.optimal_min and self.state.integral > 0.0:
                self.state.integral = 0.0
        self.state.active_bounds = bounds

    def compute_deadband_error(self, measurement: float, bounds: CropBounds) -> float:
        """Compute signed deadband error outside [optimal_min, optimal_max]."""
        if measurement < bounds.optimal_min:
            return bounds.optimal_min - measurement
        if measurement > bounds.optimal_max:
            if self.gains.unidirectional_positive:
                return 0.0
            return bounds.optimal_max - measurement
        return 0.0

    def step(
        self,
        measurement: SensorMeasurement,
        bounds: CropBounds,
        block_positive: bool = False,
        block_negative: bool = False,
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
        """
        if self.state.active_bounds is None or (
            self.state.active_bounds.crop_id != bounds.crop_id
            or self.state.active_bounds.stage != bounds.stage
            or self.state.active_bounds.optimal_min != bounds.optimal_min
            or self.state.active_bounds.optimal_max != bounds.optimal_max
        ):
            self.set_targets(bounds)

        quality = measurement.quality

        # Fail-safe: prolonged missing/unreliable sensing -> zero energy/water output
        if quality == MeasurementQuality.UNRELIABLE_FAILSAFE:
            self.state.derivative = 0.0
            self.state.prev_measurement = None
            self.state.last_error = 0.0
            self.state.last_output = 0.0
            return 0.0

        # Transient hold: pause integration, zero D, hold last safe output
        if (
            quality == MeasurementQuality.HOLD_TRANSIENT
            or measurement.value is None
            or not math.isfinite(measurement.value)
        ):
            self.state.derivative = 0.0
            self.state.prev_measurement = None
            return self.state.last_output

        g = self.gains
        # Multi-rate sampling gate when Ts > dt: hold last output between sample ticks
        if self.state.has_updated:
            self.state.elapsed_since_update += g.dt
            if self.state.elapsed_since_update + 1e-9 < g.Ts:
                return self.state.last_output
        self.state.elapsed_since_update = 0.0
        self.state.has_updated = True

        y = float(measurement.value)
        e = self.compute_deadband_error(y, bounds)
        self.state.last_error = e

        # Proportional
        P = g.Kp * e

        # Filtered Derivative-on-Measurement
        if self.state.prev_measurement is None or g.Kd == 0.0:
            D = 0.0
        else:
            dy = y - self.state.prev_measurement
            denom = g.tau_f + g.Ts
            D = (g.tau_f / denom) * self.state.derivative - (g.Kd / denom) * dy

        # Provisional Integral
        I_cand = self.state.integral + g.Ki * g.Ts * e

        # Unsaturated command & saturation
        v = P + I_cand + D
        u = max(g.u_min, min(g.u_max, v))

        # Back-calculation anti-windup
        I_new = I_cand + (g.Ts / g.T_t) * (u - v)

        # External actuator-block clamping (prevents windup during fan conflict / soak lockout)
        if block_positive and e > 0.0 and I_new > self.state.integral:
            I_new = self.state.integral
        if block_negative and e < 0.0 and I_new < self.state.integral:
            I_new = self.state.integral

        # Clamp integral within actuator bounds
        I_new = max(g.u_min, min(g.u_max, I_new))

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
        Shadow-tracking update while MARL is active, enabling bumpless MARL -> PID handover.

        Aligns integral state so P + I matches the bounded effective command `effective_u`
        and keeps prev_measurement warm so D starts at 0 without a step spike.
        """
        if self.state.active_bounds is None or (
            self.state.active_bounds.optimal_min != bounds.optimal_min
            or self.state.active_bounds.optimal_max != bounds.optimal_max
        ):
            self.set_targets(bounds)

        g = self.gains
        self.state.derivative = 0.0

        if (
            measurement.quality in (MeasurementQuality.TRUSTED, MeasurementQuality.ESTIMATE_ONLY)
            and measurement.value is not None
            and math.isfinite(measurement.value)
        ):
            y = float(measurement.value)
            e = self.compute_deadband_error(y, bounds)
            P = g.Kp * e
            target_u = max(g.u_min, min(g.u_max, float(effective_u)))
            # Smoothly track integral toward target_u - P
            desired_I = max(g.u_min, min(g.u_max, target_u - P))
            alpha_track = min(1.0, g.Ts / g.T_t)
            self.state.integral = (1.0 - alpha_track) * self.state.integral + alpha_track * desired_I
            self.state.prev_measurement = y
            self.state.last_error = e
            self.state.last_output = target_u
        else:
            self.state.prev_measurement = None
