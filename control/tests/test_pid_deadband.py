"""Deadband semantics: no stale effort, no D-driven actuation in band."""
from control.pid_controller import PIDController
from control.types import CropBounds, MeasurementQuality, PIDGains, SensorMeasurement

B = CropBounds("c", "s", "T", "C", 10.0, 18.0, 26.0, 34.0)
G = dict(Kp=0.4, Ki=0.002, Kd=2.0, tau_f=5.0, T_t=20.0)


def m(v, q=MeasurementQuality.TRUSTED):
    return SensorMeasurement("T", v, v, v, v, 1.0, 1.0, q, v is not None, v is not None)


def test_integral_bleeds_off_inside_deadband():
    pid = PIDController("T", PIDGains(**G))
    pid.state.integral = 0.6
    for _ in range(100):
        pid.step(m(22.0), B)
    assert abs(pid.state.integral) < 0.01


def test_no_actuation_inside_deadband_even_when_rising():
    pid = PIDController("T", PIDGains(**G))
    for i in range(10):
        assert pid.step(m(20.0 + 0.3 * i), B) == 0.0


def test_derivative_still_brakes_outside_band():
    with_d = PIDController("T", PIDGains(**G))
    no_d = PIDController("T", PIDGains(**{**G, "Kd": 0.0}))
    for y in (27.0, 28.0):
        ua, ub = with_d.step(m(y), B), no_d.step(m(y), B)
    assert ua < ub


def test_failsafe_clears_integral():
    pid = PIDController("T", PIDGains(**G))
    pid.state.integral = 0.6
    assert pid.step(m(None, MeasurementQuality.UNRELIABLE_FAILSAFE), B) == 0.0
    assert pid.state.integral == 0.0


def test_shadow_tracking_does_not_import_effort_inside_band():
    pid = PIDController("T", PIDGains(**G))
    for _ in range(60):
        pid.track_shadow(m(22.0), B, 1.0)   # MARL heating while zone is in range
    assert abs(pid.state.integral) < 1e-9
    assert abs(pid.step(m(22.0), B)) < 1e-9


def test_shadow_tracking_still_tracks_outside_band():
    pid = PIDController("T", PIDGains(**G))
    for _ in range(60):
        pid.track_shadow(m(17.0), B, 1.0)   # e=1, P=0.4 -> desired I = 0.6
    assert 0.5 < pid.state.integral <= 0.6


def test_update_gains_is_validated_like_the_constructor():
    import pytest
    pid = PIDController("T", PIDGains(**G))
    with pytest.raises(ValueError):
        pid.update_gains(PIDGains(Kp=1.0, Ki=0.0, Ts=0.5, dt=1.0))   # Ts < dt
    with pytest.raises(ValueError):
        pid.update_gains(PIDGains(Kp=1.0, Ki=0.0, T_t=0.0))
    assert pid.gains.Kp == G["Kp"]                                    # rejected update left state intact
    pid.update_gains(PIDGains(**{**G, "Kp": 0.5}))
    assert pid.gains.Kp == 0.5
