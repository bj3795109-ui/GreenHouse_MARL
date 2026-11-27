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


# ---------------------------------------------------------------------------
# Phase 2 hardened unit & regression tests (PID-01 .. PID-12 + lifecycle)
# ---------------------------------------------------------------------------

def test_pid01_temperature_above_target_requests_cooling():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.002, Kd=0.0))
    u = pid.step(_meas(28.0), _bounds(20.0, 25.0))
    assert u < 0.0


def test_pid02_temperature_below_target_requests_heating():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.002, Kd=0.0))
    u = pid.step(_meas(17.0), _bounds(20.0, 25.0))
    assert u > 0.0


def test_pid03_temperature_inside_target_returns_neutral():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.01, Kd=2.0))
    u = pid.step(_meas(22.0), _bounds(20.0, 25.0))
    assert u == pytest.approx(0.0)


def test_pid04_large_persistent_error_accumulates_integral():
    pid = PIDController("T", PIDGains(Kp=0.05, Ki=0.02, Kd=0.0, u_min=-1.0, u_max=1.0))
    b = _bounds(22.0, 25.0)
    u_prev = pid.step(_meas(19.0), b)  # e = +3.0 -> I = 0.06
    assert pid.state.integral == pytest.approx(0.06)
    for _ in range(4):
        u_now = pid.step(_meas(19.0), b)
        assert u_now > u_prev
        u_prev = u_now
    assert pid.state.integral == pytest.approx(0.30)


def test_pid05_and_pid06_anti_windup_at_max_and_min_saturation():
    # PID-05: positive saturation at u_max
    pid_max = PIDController("T", PIDGains(Kp=1.0, Ki=0.5, Kd=0.0, T_t=2.0, u_min=-1.0, u_max=1.0))
    b = _bounds(22.0, 25.0)
    for _ in range(30):
        assert pid_max.step(_meas(10.0), b) == pytest.approx(1.0)
    assert pid_max.state.integral <= 1.0
    # When error reverses slightly, output should recover promptly rather than remaining stuck
    u_rev = pid_max.step(_meas(27.0), b)  # e = -2.0 -> P = -2.0, I <= 1.0 -> u < 0
    assert u_rev < 0.0

    # PID-06: negative saturation at u_min
    pid_min = PIDController("T", PIDGains(Kp=1.0, Ki=0.5, Kd=0.0, T_t=2.0, u_min=-1.0, u_max=1.0))
    for _ in range(30):
        assert pid_min.step(_meas(38.0), b) == pytest.approx(-1.0)
    assert pid_min.state.integral >= -1.0
    u_rev_pos = pid_min.step(_meas(20.0), b)  # e = +2.0 -> P = +2.0, I >= -1.0 -> u > 0
    assert u_rev_pos > 0.0


def test_pid07_derivative_responds_to_rapid_measurement_change():
    pid = PIDController("T", PIDGains(Kp=0.0, Ki=0.0, Kd=4.0, tau_f=2.0, Ts=1.0, dt=1.0))
    b = _bounds(20.0, 25.0)
    pid.step(_meas(24.0), b)
    # Crossing above the band by +3.0 -> D = -(4.0 / (2.0 + 1.0)) * 3.0 = -4.0 -> clamped -1.0
    u_rise = pid.step(_meas(27.0), b)
    assert pid.state.derivative == pytest.approx(-4.0)
    assert u_rise == pytest.approx(-1.0)

    # A rapid drop that remains outside the band -> D moves positive to oppose rapid cooling
    u_drop = pid.step(_meas(25.5), b)
    expected_d = (2.0 / 3.0) * (-4.0) - (4.0 / 3.0) * (-1.5)
    assert pid.state.derivative == pytest.approx(expected_d)
    assert u_drop > u_rise


def test_pid08_variable_timestep_scales_integral_derivative_and_antiwindup():
    pid = PIDController("T", PIDGains(Kp=0.0, Ki=0.1, Kd=2.0, tau_f=2.0, T_t=10.0, Ts=1.0, dt=1.0))
    b = _bounds(22.0, 25.0)
    # First step with dt=2.5s: I = 0.1 * 2.5 * 2.0 = 0.50
    u1 = pid.step(_meas(20.0), b, dt=2.5)
    assert u1 == pytest.approx(0.50)
    assert pid.state.integral == pytest.approx(0.50)

    # Second step with dt=0.5s: dy = 1.0, denom = 2.0 + 0.5 = 2.5, D = -(2.0/2.5)*1.0 = -0.8
    # e = 1.0, I_cand = 0.50 + 0.1 * 0.5 * 1.0 = 0.55, u = 0.55 - 0.8 = -0.25
    u2 = pid.step(_meas(21.0), b, dt=0.5)
    assert pid.state.derivative == pytest.approx(-0.8)
    assert pid.state.integral == pytest.approx(0.55)
    assert u2 == pytest.approx(-0.25)


@pytest.mark.parametrize("bad_dt", [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True])
def test_invalid_timestep_rejected(bad_dt):
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.01))
    with pytest.raises(ValueError):
        pid.step(_meas(20.0), _bounds(22.0, 25.0), dt=bad_dt)


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("quality", [MeasurementQuality.TRUSTED, MeasurementQuality.ESTIMATE_ONLY, MeasurementQuality.HOLD_TRANSIENT])
def test_pid09_and_pid10_nan_and_inf_measurements_rejected_without_holding_stale_output(bad_val, quality):
    pid = PIDController("T", PIDGains(Kp=0.5, Ki=0.05))
    b = _bounds(22.0, 25.0)
    assert pid.step(_meas(18.0), b) > 0.0  # non-zero prior output
    with pytest.raises(ValueError):
        pid.step(_meas(bad_val, quality=quality), b)
    # Active drive must be cleared, not left holding previous high output
    assert pid.state.last_output == 0.0


def test_none_measurement_with_trusted_quality_rejected():
    pid = PIDController("T", PIDGains(Kp=0.5, Ki=0.05))
    b = _bounds(22.0, 25.0)
    pid.step(_meas(18.0), b)
    with pytest.raises(ValueError):
        pid.step(_meas(None, quality=MeasurementQuality.TRUSTED), b)
    assert pid.state.last_output == 0.0


def test_pid11_reset_clears_integral_derivative_and_timing_history():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.05, Kd=2.0, Ts=2.0, dt=1.0))
    b = _bounds(22.0, 25.0)
    pid.step(_meas(18.0), b)
    pid.step(_meas(19.0), b)
    assert pid.state.integral != 0.0
    assert pid.state.has_updated is True

    pid.reset()
    assert pid.state.integral == 0.0
    assert pid.state.derivative == 0.0
    assert pid.state.prev_measurement is None
    assert pid.state.last_output == 0.0
    assert pid.state.last_error == 0.0
    assert pid.state.step_counter == 0
    assert pid.state.elapsed_since_update == 0.0
    assert pid.state.has_updated is False
    assert pid.state.active_bounds is None


def test_pid12_setpoint_changes_no_spike_and_clears_opposing_integral():
    pid = PIDController("T", PIDGains(Kp=0.2, Ki=0.05, Kd=5.0, tau_f=2.0))
    b_low = _bounds(18.0, 20.0)
    b_high = _bounds(24.0, 26.0)
    pid.step(_meas(22.0), b_low)
    pid.step(_meas(22.0), b_low)
    assert pid.state.integral < 0.0  # accumulated negative cooling integral

    # Setpoint shifts upward past previous band: negative integral cleared, D stays 0
    u = pid.step(_meas(22.0), b_high)
    assert pid.state.derivative == 0.0
    assert pid.state.integral > 0.0
    assert u == pytest.approx(0.2 * 2.0 + 0.05 * 2.0)


def test_unidirectional_moisture_unwinds_nonzero_integral_above_optimal_max():
    """Regression: nonzero positive integral must not keep watering when M > optimal_max."""
    pid = PIDController(
        "M",
        PIDGains(Kp=0.10, Ki=0.004, Kd=0.0, T_t=10.0, u_min=0.0, u_max=1.0, unidirectional_positive=True),
    )
    b = CropBounds("tomato", "vegetative", "M", "relative_percent", 25.0, 40.0, 55.0, 70.0)

    # Dry soil builds positive integral
    for _ in range(25):
        u = pid.step(_meas(30.0, var="M"), b)
        assert u > 0.0
    i_accum = pid.state.integral
    assert i_accum > 0.25

    # Soil rises slightly above optimal_max (56.0 > 55.0): pump MUST stop immediately and integral unwinds
    u_over = pid.step(_meas(56.0, var="M"), b)
    assert u_over == 0.0
    assert pid.state.integral < i_accum

    for _ in range(40):
        u_over = pid.step(_meas(56.0, var="M"), b)
        assert u_over == 0.0
    assert pid.state.integral < 0.05


def test_unidirectional_positive_clamps_output_nonnegative_even_if_u_min_default():
    pid = PIDController("M", PIDGains(Kp=0.2, Ki=0.05, unidirectional_positive=True))
    b = _bounds(40.0, 55.0)
    assert pid.step(_meas(60.0, var="M"), b) == 0.0
    assert pid.state.integral >= 0.0


@pytest.mark.parametrize(
    "bad_gains",
    [
        PIDGains(Kp=float("nan"), Ki=0.01),
        PIDGains(Kp=0.4, Ki=float("inf")),
        PIDGains(Kp=0.4, Ki=0.01, Kd=-0.1),
        PIDGains(Kp=-0.4, Ki=0.01),
        PIDGains(Kp=0.4, Ki=0.01, dt=0.0),
        PIDGains(Kp=0.4, Ki=0.01, dt=float("nan")),
        PIDGains(Kp=0.4, Ki=0.01, Ts=0.5, dt=1.0),
        PIDGains(Kp=0.4, Ki=0.01, tau_f=-1.0),
        PIDGains(Kp=0.4, Ki=0.01, T_t=0.0),
        PIDGains(Kp=0.4, Ki=0.01, u_min=1.0, u_max=1.0),
        PIDGains(Kp=0.4, Ki=0.01, u_min=-1.0, u_max=0.0, unidirectional_positive=True),
    ],
)
def test_invalid_gains_rejected_on_init_and_update(bad_gains):
    with pytest.raises(ValueError):
        PIDController("T", bad_gains)

    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.01))
    with pytest.raises(ValueError):
        pid.update_gains(bad_gains)
    assert pid.gains.Kp == 0.4


def test_invalid_setpoint_bounds_rejected():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.01))
    bad_nan = CropBounds("tomato", "veg", "T", "degC", 15.0, float("nan"), 25.0, 32.0)
    bad_inf = CropBounds("tomato", "veg", "T", "degC", 15.0, 20.0, float("inf"), 32.0)
    bad_order = CropBounds("tomato", "veg", "T", "degC", 15.0, 26.0, 22.0, 32.0)
    for bad in (bad_nan, bad_inf, bad_order):
        with pytest.raises(ValueError):
            pid.step(_meas(22.0), bad)


def test_disabled_controller_outputs_zero_and_freezes_integral():
    pid = PIDController("T", PIDGains(Kp=0.5, Ki=0.05, Kd=1.0))
    b = _bounds(22.0, 25.0)
    assert pid.step(_meas(18.0), b) > 0.0
    i_before = pid.state.integral

    pid.disable()
    assert pid.enabled is False
    for _ in range(5):
        assert pid.step(_meas(10.0), b) == 0.0
        assert pid.state.integral == pytest.approx(i_before)
        assert pid.state.last_output == 0.0

    # Also remains zero on HOLD_TRANSIENT while disabled
    assert pid.step(_meas(None, quality=MeasurementQuality.HOLD_TRANSIENT), b) == 0.0

    pid.enable()
    assert pid.enabled is True
    assert pid.step(_meas(18.0), b) > 0.0


def test_shadow_tracking_inside_deadband_does_not_accumulate_integral():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.02, T_t=2.0, Ts=1.0))
    b = _bounds(22.0, 25.0)
    for _ in range(15):
        pid.track_shadow(_meas(23.5), b, effective_u=1.0)
    assert pid.state.integral == pytest.approx(0.0)
    assert pid.step(_meas(23.5), b) == pytest.approx(0.0)


def test_shadow_tracking_crop_stage_change_and_elapsed_reset():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.02, T_t=2.0, Ts=3.0, dt=1.0))
    b1 = CropBounds("tomato", "seedling", "T", "degC", 18.0, 22.0, 26.0, 30.0)
    b2 = CropBounds("lettuce", "mature", "T", "degC", 10.0, 15.0, 20.0, 24.0)

    # Simulate partial multi-rate elapsed state then shadow tracking across crop change
    pid.step(_meas(20.0), b1)
    assert pid.state.integral > 0.0
    pid.step(_meas(20.0), b1)  # elapsed_since_update = 1.0 < Ts=3.0
    assert pid.state.elapsed_since_update == pytest.approx(1.0)

    pid.track_shadow(_meas(20.5), b2, effective_u=-0.5)
    assert pid.state.active_bounds == b2
    assert pid.state.integral < 0.0  # positive integral cleared on downward band shift
    assert pid.state.elapsed_since_update == 0.0
    assert pid.state.has_updated is False

    # Immediate takeover computes PID right away instead of holding stale output
    u_takeover = pid.step(_meas(20.5), b2)
    assert pid.state.has_updated is True
    assert u_takeover < 0.0


def test_load_pid_tuning_rejects_invalid_gains():
    from control.crop_config_loader import CropConfigError, load_pid_tuning

    bad_cfg = {
        "default": {
            "T": {"Kp": float("nan"), "Ki": 0.002},
            "H": {"Kp": 0.12, "Ki": 0.005},
            "M": {"Kp": 0.10, "Ki": 0.004},
        }
    }
    with pytest.raises(CropConfigError):
        load_pid_tuning(bad_cfg)


def test_shadow_tracking_rejects_non_finite_inputs():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.02))
    b = _bounds(22.0, 25.0)
    with pytest.raises(ValueError):
        pid.track_shadow(_meas(20.0), b, effective_u=float("nan"))
    with pytest.raises(ValueError):
        pid.track_shadow(_meas(float("inf")), b, effective_u=0.5)


def test_backward_compatibility_existing_call_signatures():
    pid = PIDController("T", PIDGains(Kp=0.4, Ki=0.002, Kd=2.0))
    b = _bounds(22.0, 25.0)
    u1 = pid.step(_meas(20.0), b)
    u2 = pid.step(_meas(20.0), b, True, False)
    u3 = pid.step(_meas(20.0), b, block_positive=False, block_negative=True)
    pid.track_shadow(_meas(20.0), b, 0.5)
    assert all(math.isfinite(u) for u in (u1, u2, u3))
