"""Unit tests for PIDController (deadband, anti-windup, D-on-measurement, quality gating)."""

import math
import pytest

from control.pid_controller import PIDController
from control.types import CropBounds, MeasurementQuality, PIDGains, SensorMeasurement


def _bounds(opt_min=20.0, opt_max=25.0):
    return CropBounds(
        crop_id="tomato",
        stage="vegetative",
        var="T",
        unit="degC",
        min_acceptable=15.0,
        optimal_min=opt_min,
        optimal_max=opt_max,
        max_acceptable=32.0,
    )


def _meas(val, quality=MeasurementQuality.TRUSTED, var="T"):
    return SensorMeasurement(
        var=var,
        value=val,
        raw=val,
        combined=val,
        belief=val,
        trust=1.0 if quality == MeasurementQuality.TRUSTED else 0.1,
        confidence=0.9,
        quality=quality,
        valid_raw=(val is not None and math.isfinite(val)),
        valid_est=(val is not None and math.isfinite(val)),
    )


def test_deadband_returns_zero_inside_optimal_range():
    pid = PIDController("T", PIDGains(Kp=0.5, Ki=0.01, Kd=1.0))
    u = pid.step(_meas(22.5), _bounds(20.0, 25.0))
    assert u == 0.0
    assert pid.state.integral == 0.0


def test_derivative_on_measurement_has_no_kick_on_target_change():
    pid = PIDController("T", PIDGains(Kp=0.1, Ki=0.0, Kd=5.0, tau_f=2.0, Ts=1.0))
    b1 = _bounds(20.0, 25.0)
    b2 = _bounds(28.0, 30.0)
    # Warm up at constant measurement 22.0
    pid.step(_meas(22.0), b1)
    pid.step(_meas(22.0), b1)
    # Target jumps upward while measurement remains 22.0
    u = pid.step(_meas(22.0), b2)
    assert pid.state.derivative == 0.0
    assert u == pytest.approx(0.1 * (28.0 - 22.0))


def test_saturation_and_back_calculation_anti_windup():
    pid = PIDController("T", PIDGains(Kp=1.0, Ki=0.5, Kd=0.0, T_t=2.0, Ts=1.0, u_min=-1.0, u_max=1.0))
    b = _bounds(22.0, 25.0)
    for _ in range(50):
        u = pid.step(_meas(10.0), b)
        assert -1.0 <= u <= 1.0
    # Integral must stay bounded rather than growing to +300
    assert pid.state.integral <= 1.0


def test_external_block_prevents_integral_windup():
    pid = PIDController("T", PIDGains(Kp=0.2, Ki=0.1, Kd=0.0, T_t=5.0, Ts=1.0))
    b = _bounds(20.0, 22.0)
    pid.step(_meas(26.0), b, block_negative=True)
    i_before = pid.state.integral
    for _ in range(20):
        pid.step(_meas(26.0), b, block_negative=True)
    assert pid.state.integral == pytest.approx(i_before)


def test_hold_transient_freezes_integral_and_holds_output():
    pid = PIDController("T", PIDGains(Kp=0.5, Ki=0.05, Kd=1.0))
    b = _bounds(22.0, 25.0)
    u1 = pid.step(_meas(20.0), b)
    i1 = pid.state.integral
    u_hold = pid.step(_meas(None, quality=MeasurementQuality.HOLD_TRANSIENT), b)
    assert u_hold == pytest.approx(u1)
    assert pid.state.integral == pytest.approx(i1)
    assert pid.state.derivative == 0.0


def test_unreliable_failsafe_outputs_zero():
    pid = PIDController("T", PIDGains(Kp=0.5, Ki=0.05, Kd=1.0))
    b = _bounds(22.0, 25.0)
    pid.step(_meas(18.0), b)
    u_safe = pid.step(_meas(None, quality=MeasurementQuality.UNRELIABLE_FAILSAFE), b)
    assert u_safe == 0.0


def test_unidirectional_moisture_ignores_excess_above_optimal_max():
    pid = PIDController(
        "M", PIDGains(Kp=0.2, Ki=0.05, Kd=0.0, u_min=0.0, u_max=1.0, unidirectional_positive=True)
    )
    b = _bounds(40.0, 55.0)
    u = pid.step(_meas(75.0, var="M"), b)
    assert u == 0.0
    assert pid.state.integral == 0.0


def test_shadow_tracking_prepares_bumpless_handover():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.02, Kd=2.0, T_t=2.0, Ts=1.0))
    b = _bounds(22.0, 25.0)
    for _ in range(10):
        pid.track_shadow(_meas(20.0), b, effective_u=0.8)
    u_takeover = pid.step(_meas(20.0), b)
    assert u_takeover == pytest.approx(0.8, abs=0.15)


def test_multirate_sampling_gates_integration_when_ts_exceeds_dt():
    pid = PIDController("T", PIDGains(Kp=0.0, Ki=0.1, Kd=0.0, Ts=3.0, dt=1.0))
    b = _bounds(22.0, 25.0)
    # Step 0 updates immediately (I = 0.1 * 3.0 * 2.0 = 0.6)
    u0 = pid.step(_meas(20.0), b)
    assert u0 == pytest.approx(0.6)
    # Steps 1 and 2 hold without over-integrating
    u1 = pid.step(_meas(20.0), b)
    u2 = pid.step(_meas(20.0), b)
    assert u1 == pytest.approx(0.6)
    assert u2 == pytest.approx(0.6)
    assert pid.state.step_counter == 1
    # Step 3 hits elapsed == Ts == 3.0 -> integrates again (capped at u_max=1.0)
    u3 = pid.step(_meas(20.0), b)
    assert pid.state.step_counter == 2
    assert u3 > u2
