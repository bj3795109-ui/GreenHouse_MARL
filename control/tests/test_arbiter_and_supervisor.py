"""Unit tests for ActuatorMapper, ActuatorArbiter, and ZoneSupervisor FSM."""

import math
import pytest

from control.actuator_arbiter import ActuatorArbiter
from control.actuator_mapper import ActuatorMapper
from control.supervisor import SupervisorConfig, ZoneSupervisor, validate_marl_action
from control.types import (
    ActuatorConfig,
    AmbientState,
    ControlMode,
    CropBounds,
    MeasurementQuality,
    PIDGains,
    SensorMeasurement,
)
from control.zone_controller import ZoneController


def _bounds(var, min_a, opt_min, opt_max, max_a):
    return CropBounds("tomato", "vegetative", var, "unit", min_a, opt_min, opt_max, max_a)


def _meas(var, val, quality=MeasurementQuality.TRUSTED, trust=1.0):
    return SensorMeasurement(
        var=var,
        value=val,
        raw=val,
        combined=val,
        belief=val,
        trust=trust,
        confidence=0.9,
        quality=quality,
        valid_raw=(val is not None and math.isfinite(val)),
        valid_est=(val is not None and math.isfinite(val)),
    )


def _make_zone_controller():
    gains = {
        "T": PIDGains(Kp=0.4, Ki=0.002, Kd=2.0),
        "H": PIDGains(Kp=0.12, Ki=0.005, Kd=0.0),
        "M": PIDGains(Kp=0.1, Ki=0.004, Kd=0.0, u_min=0.0, u_max=1.0, unidirectional_positive=True),
    }
    return ZoneController("zone_1", gains, ActuatorConfig(min_on_steps=2, min_off_steps=2))


def test_validate_marl_action_rejects_nan_before_bool_coercion():
    ok, parsed, reason = validate_marl_action([float("nan"), 0, 0, 0])
    assert not ok
    assert parsed is None
    assert "non_finite" in reason


def test_validate_marl_action_rejects_non_binary_and_wrong_shape():
    assert not validate_marl_action([1, 2, 0, 0])[0]
    assert not validate_marl_action([1, 0, 0])[0]
    assert not validate_marl_action(None)[0]
    ok, parsed, _ = validate_marl_action([1, 0, 1, 0])
    assert ok and parsed == (1, 0, 1, 0)


def test_arbiter_inhibits_fan_when_ambient_is_counterproductive():
    mapper = ActuatorMapper(ActuatorConfig(min_on_steps=1, min_off_steps=1))
    arbiter = ActuatorArbiter(ActuatorConfig(min_on_steps=1, min_off_steps=1))
    bT = _bounds("T", 16.0, 21.0, 25.0, 32.0)
    bH = _bounds("H", 45.0, 60.0, 75.0, 85.0)
    # Zone is hot (28°C) and humid (80%), BUT outside ambient is even hotter (34°C) and wetter (90%)!
    req = mapper.map_outputs(u_T=-0.9, u_H=-0.9, u_M=0.0)
    res = arbiter.arbitrate(
        req,
        _meas("T", 28.0),
        _meas("H", 80.0),
        bT,
        bH,
        AmbientState(T_ambient=34.0, H_ambient=90.0),
    )
    assert res.action[1] == 0  # fan inhibited
    assert res.block_flags.block_T_neg is True
    assert res.block_flags.block_H_neg is True


def test_arbiter_resolves_T_vs_H_fan_conflict_with_overtemp_priority():
    mapper = ActuatorMapper(ActuatorConfig(min_on_steps=1, min_off_steps=1))
    arbiter = ActuatorArbiter(ActuatorConfig(min_on_steps=1, min_off_steps=1))
    bT = _bounds("T", 16.0, 21.0, 25.0, 32.0)
    bH = _bounds("H", 45.0, 60.0, 75.0, 85.0)
    # Zone over-temperature (33.5 > 32 max_acceptable) AND dry (55 < 60), ambient cool & dry
    req = mapper.map_outputs(u_T=-1.0, u_H=0.5, u_M=0.0)
    res = arbiter.arbitrate(
        req,
        _meas("T", 33.5),
        _meas("H", 55.0),
        bT,
        bH,
        AmbientState(T_ambient=20.0, H_ambient=35.0),
    )
    assert res.action[1] == 1  # fan ON for critical cooling


def test_water_soak_lockout_caps_continuous_irrigation():
    mapper = ActuatorMapper(ActuatorConfig(water_max_pulse_steps=2, water_soak_steps=4, min_on_steps=1, min_off_steps=1))
    cmds = [mapper.map_outputs(0.0, 0.0, 1.0).water_cmd for _ in range(7)]
    assert cmds[:2] == [1, 1]
    assert cmds[2:6] == [0, 0, 0, 0]


def test_supervisor_requires_n_confirm_steps_for_recovery():
    zc = _make_zone_controller()
    sup = ZoneSupervisor("zone_1", zc, cfg=SupervisorConfig(N_confirm=4))
    bounds = {
        "T": _bounds("T", 16, 21, 26, 32),
        "H": _bounds("H", 45, 60, 75, 85),
        "M": _bounds("M", 25, 40, 55, 70),
    }
    meas = {"T": _meas("T", 22.0), "H": _meas("H", 65.0), "M": _meas("M", 45.0)}
    amb = AmbientState(24.0, 45.0)

    # Fault -> immediate PID_FALLBACK
    st = sup.step(None, meas, bounds, amb, agent_failed=True)
    assert st.mode == ControlMode.PID_FALLBACK

    # 1..3 valid steps -> RECOVERY_PENDING (NOT yet MARL_ACTIVE)
    for k in range(1, 4):
        st = sup.step([0, 0, 0, 0], meas, bounds, amb, agent_failed=False)
        assert st.mode == ControlMode.RECOVERY_PENDING
        assert st.recovery_streak == k

    # 4th consecutive valid step -> MARL_ACTIVE
    st = sup.step([0, 0, 0, 0], meas, bounds, amb, agent_failed=False)
    assert st.mode == ControlMode.MARL_ACTIVE


def test_supervisor_aborts_recovery_on_invalid_step_and_enters_safe_mode_on_lost_sensors():
    zc = _make_zone_controller()
    sup = ZoneSupervisor("zone_1", zc, cfg=SupervisorConfig(N_confirm=4))
    bounds = {
        "T": _bounds("T", 16, 21, 26, 32),
        "H": _bounds("H", 45, 60, 75, 85),
        "M": _bounds("M", 25, 40, 55, 70),
    }
    meas = {"T": _meas("T", 22.0), "H": _meas("H", 65.0), "M": _meas("M", 45.0)}
    amb = AmbientState(24.0, 45.0)

    sup.step(None, meas, bounds, amb, agent_failed=True)
    sup.step([0, 0, 0, 0], meas, bounds, amb)
    assert sup.mode == ControlMode.RECOVERY_PENDING

    # Interrupted recovery -> back to PID_FALLBACK
    sup.step([float("nan"), 0, 0, 0], meas, bounds, amb)
    assert sup.mode == ControlMode.PID_FALLBACK

    # Sensor loss -> SAFE_MODE forcing all actuators OFF
    lost_meas = {
        "T": _meas("T", None, quality=MeasurementQuality.UNRELIABLE_FAILSAFE, trust=0.0),
        "H": _meas("H", 65.0),
        "M": _meas("M", 45.0),
    }
    st = sup.step([1, 0, 1, 1], lost_meas, bounds, amb)
    assert st.mode == ControlMode.SAFE_MODE
    assert st.executed_action == (0, 0, 0, 0)
