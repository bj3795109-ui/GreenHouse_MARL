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


# ---------------------------------------------------------------------------
# Phase 5: FB-07, FB-09, FB-10, NaN-safe interlocks, metrics lifecycle
# ---------------------------------------------------------------------------

def _standard_bounds():
    return {
        "T": _bounds("T", 16, 21, 26, 32),
        "H": _bounds("H", 45, 60, 75, 85),
        "M": _bounds("M", 25, 40, 55, 70),
    }


def _nominal_meas(T=22.0, H=65.0, M=45.0):
    return {"T": _meas("T", T), "H": _meas("H", H), "M": _meas("M", M)}


def test_fb07_repeated_recovery_failures_keep_zone_in_pid():
    zc = _make_zone_controller()
    sup = ZoneSupervisor("zone_1", zc, cfg=SupervisorConfig(N_confirm=3))
    bounds = _standard_bounds()
    meas = _nominal_meas(T=18.0)
    amb = AmbientState(24.0, 45.0)

    sup.step(None, meas, bounds, amb, agent_failed=True)
    assert sup.mode == ControlMode.PID_FALLBACK

    # Repeatedly start recovery (2 valid steps < N_confirm=3) then fail on 3rd step
    for _ in range(4):
        assert sup.step([1, 0, 0, 0], meas, bounds, amb).mode == ControlMode.RECOVERY_PENDING
        assert sup.step([1, 0, 0, 0], meas, bounds, amb).mode == ControlMode.RECOVERY_PENDING
        st_fail = sup.step([float("nan"), 0, 0, 0], meas, bounds, amb)
        assert st_fail.mode == ControlMode.PID_FALLBACK
        assert st_fail.recovery_streak == 0

    assert sup.metrics.recovery_attempts == 4
    assert sup.metrics.recovery_successes == 0


def test_fb09_pid_controller_exception_enters_safe_mode_and_recovers_via_pid_fallback(monkeypatch):
    zc = _make_zone_controller()
    sup = ZoneSupervisor("zone_2", zc, cfg=SupervisorConfig(N_confirm=3, safe_mode_retry_steps=2))
    bounds = _standard_bounds()
    meas = _nominal_meas(T=18.0)
    amb = AmbientState(24.0, 45.0)

    # Trigger PID fallback first
    sup.step(None, meas, bounds, amb, agent_failed=True)
    assert sup.mode == ControlMode.PID_FALLBACK

    # Inject a runtime fault inside ZoneController.step()
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated PID failure")

    orig_step = zc.step
    monkeypatch.setattr(zc, "step", _boom)

    st_err = sup.step(None, meas, bounds, amb, agent_failed=True)
    assert st_err.mode == ControlMode.SAFE_MODE
    assert st_err.executed_action == (0, 0, 0, 0)
    assert st_err.transitioned_from == ControlMode.PID_FALLBACK
    assert sup.metrics.pid_failures == 1
    assert sup.metrics.safe_mode_events == 1
    assert "pid_failure" in (sup.metrics.last_fault_reason or "")

    # Next step during cooldown stays in SAFE_MODE without duplicating safe_mode_events
    st_cool = sup.step([0, 0, 0, 0], meas, bounds, amb)
    assert st_cool.mode == ControlMode.SAFE_MODE
    assert st_cool.executed_action == (0, 0, 0, 0)
    assert sup.metrics.safe_mode_events == 1

    # Restore ZoneController.step() -> allowed recovery path SAFE_MODE -> PID_FALLBACK -> RECOVERY_PENDING -> MARL_ACTIVE
    monkeypatch.setattr(zc, "step", orig_step)
    st_pid = sup.step([1, 0, 0, 0], meas, bounds, amb)
    assert st_pid.mode == ControlMode.PID_FALLBACK

    for _ in range(2):
        assert sup.step([1, 0, 0, 0], meas, bounds, amb).mode == ControlMode.RECOVERY_PENDING
    assert sup.step([1, 0, 0, 0], meas, bounds, amb).mode == ControlMode.MARL_ACTIVE


def test_fb10_valid_and_invalid_fsm_transitions():
    from control.supervisor import VALID_TRANSITIONS

    assert ControlMode.MARL_ACTIVE not in VALID_TRANSITIONS[ControlMode.SAFE_MODE]
    assert ControlMode.RECOVERY_PENDING not in VALID_TRANSITIONS[ControlMode.SAFE_MODE]
    assert ControlMode.MARL_ACTIVE not in VALID_TRANSITIONS[ControlMode.PID_FALLBACK]

    zc = _make_zone_controller()
    sup = ZoneSupervisor("zone_1", zc, initial_mode=ControlMode.SAFE_MODE)

    # Repeated same-state request is an idempotent no-op
    assert sup.transition_to(ControlMode.SAFE_MODE, "still_safe") == (None, None)
    assert sup.metrics.safe_mode_events == 0

    # Illegal bypass SAFE_MODE -> MARL_ACTIVE rejected
    with pytest.raises(ValueError, match="invalid_transition"):
        sup.transition_to(ControlMode.MARL_ACTIVE, "illegal_bypass")
    assert sup.mode == ControlMode.SAFE_MODE
    assert sup.metrics.invalid_transitions_rejected == 1

    # Illegal bypass SAFE_MODE -> RECOVERY_PENDING rejected
    with pytest.raises(ValueError, match="invalid_transition"):
        sup.transition_to(ControlMode.RECOVERY_PENDING, "illegal_bypass")
    assert sup.mode == ControlMode.SAFE_MODE
    assert sup.metrics.invalid_transitions_rejected == 2

    # Permitted recovery sequence: SAFE_MODE -> PID_FALLBACK -> RECOVERY_PENDING -> MARL_ACTIVE
    sup.transition_to(ControlMode.PID_FALLBACK, "sensors_ok")
    with pytest.raises(ValueError, match="invalid_transition"):
        sup.transition_to(ControlMode.MARL_ACTIVE, "skip_recovery_pending")
    sup.transition_to(ControlMode.RECOVERY_PENDING, "candidate_ok")
    sup.transition_to(ControlMode.MARL_ACTIVE, "confirmed")
    assert sup.mode == ControlMode.MARL_ACTIVE


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_inf_safety_interlocks_cannot_be_bypassed(bad_val):
    zc = _make_zone_controller()
    sup = ZoneSupervisor("zone_1", zc)
    bounds = _standard_bounds()
    amb = AmbientState(24.0, 45.0)

    # Even if a malformed measurement claims TRUSTED, NaN/Inf triggers SAFE_MODE with (0,0,0,0)
    bad_meas = {
        "T": _meas("T", bad_val, quality=MeasurementQuality.TRUSTED),
        "H": _meas("H", 65.0),
        "M": _meas("M", 45.0),
    }
    st = sup.step([1, 1, 1, 1], bad_meas, bounds, amb)
    assert st.mode == ControlMode.SAFE_MODE
    assert st.executed_action == (0, 0, 0, 0)

    # Direct interlock check must also clamp heat/mist/water on non-finite values
    clamped = sup._enforce_software_interlocks((1, 1, 1, 1), bad_meas)
    assert clamped[0] == 0
    # Non-finite action tuple rejected completely
    assert sup._enforce_software_interlocks([float("nan"), 1, 1, 1], _nominal_meas()) == (0, 0, 0, 0)

    # Non-finite ambient must also fail safely rather than silently bypassing fan checks
    sup2 = ZoneSupervisor("zone_1", _make_zone_controller())
    st_amb = sup2.step([1, 1, 0, 0], _nominal_meas(), bounds, AmbientState(bad_val, 45.0))
    assert st_amb.mode == ControlMode.SAFE_MODE
    assert st_amb.executed_action == (0, 0, 0, 0)


def test_metrics_and_counter_lifecycle_across_transitions_and_reset():
    zc = _make_zone_controller()
    sup = ZoneSupervisor("zone_1", zc, cfg=SupervisorConfig(N_confirm=3, W_persist=3, K_contra=2))
    bounds = _standard_bounds()
    amb = AmbientState(20.0, 40.0)
    meas = _nominal_meas(T=22.0)

    # 1. Offline agent in PID_FALLBACK must not inflate unsafe_actions_intercepted every tick
    sup.step(None, meas, bounds, amb, agent_failed=True)  # MARL_ACTIVE -> PID_FALLBACK (1 intercepted)
    for _ in range(5):
        sup.step(None, meas, bounds, amb, agent_failed=True)
    assert sup.metrics.unsafe_actions_intercepted == 1
    assert sup.metrics.marl_to_pid_switches == 1

    # 2. Repeated SAFE_MODE steps increment safe_mode_steps but NOT safe_mode_events
    lost = {"T": _meas("T", None, MeasurementQuality.UNRELIABLE_FAILSAFE, trust=0.0), "H": _meas("H", 65.0), "M": _meas("M", 45.0)}
    for _ in range(4):
        sup.step(None, lost, bounds, amb)
    assert sup.metrics.safe_mode_events == 1
    assert sup.metrics.safe_mode_steps == 4

    # 3. Reset clears all episode metrics and transient counters cleanly
    sup.reset()
    assert sup.mode == ControlMode.MARL_ACTIVE
    assert sup.metrics.total_steps == 0
    assert sup.metrics.safe_mode_events == 0
    assert sup.metrics.unsafe_actions_intercepted == 0
    assert sup.metrics.transition_log == []
    assert sup.metrics.fault_history == []


def test_fb09_pid_failure_from_marl_transition_sensor_restore_and_recovery_pending(monkeypatch):
    bounds = _standard_bounds()
    meas = _nominal_meas(T=18.0)
    amb = AmbientState(24.0, 45.0)

    # 1. PID fails immediately when MARL -> PID_FALLBACK triggers
    zc1 = _make_zone_controller()
    sup1 = ZoneSupervisor("zone_1", zc1)
    monkeypatch.setattr(zc1, "step", lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad pid")))
    st1 = sup1.step(None, meas, bounds, amb, agent_failed=True)
    assert st1.mode == ControlMode.SAFE_MODE
    assert st1.executed_action == (0, 0, 0, 0)
    assert sup1.metrics.pid_failures == 1

    # 2. PID fails when sensor-induced SAFE_MODE attempts restoration
    zc2 = _make_zone_controller()
    sup2 = ZoneSupervisor("zone_2", zc2)
    lost = {"T": _meas("T", None, MeasurementQuality.UNRELIABLE_FAILSAFE, trust=0.0), "H": _meas("H", 65.0), "M": _meas("M", 45.0)}
    assert sup2.step(None, lost, bounds, amb).mode == ControlMode.SAFE_MODE
    monkeypatch.setattr(zc2, "step", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("pid down")))
    st2 = sup2.step([1, 0, 0, 0], meas, bounds, amb)
    assert st2.mode == ControlMode.SAFE_MODE
    assert st2.executed_action == (0, 0, 0, 0)
    assert sup2.metrics.pid_failures == 1

    # 3. PID fails during RECOVERY_PENDING
    zc3 = _make_zone_controller()
    sup3 = ZoneSupervisor("zone_3", zc3, cfg=SupervisorConfig(N_confirm=4))
    sup3.step(None, meas, bounds, amb, agent_failed=True)
    assert sup3.step([1, 0, 0, 0], meas, bounds, amb).mode == ControlMode.RECOVERY_PENDING
    monkeypatch.setattr(zc3, "step", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("pid failed in recovery")))
    st3 = sup3.step([1, 0, 0, 0], meas, bounds, amb)
    assert st3.mode == ControlMode.SAFE_MODE
    assert st3.executed_action == (0, 0, 0, 0)


def test_software_interlocks_and_supervisor_config_validation():
    with pytest.raises(ValueError):
        SupervisorConfig(N_confirm=0)
    with pytest.raises(ValueError):
        SupervisorConfig(theta_marl_min_trust=1.5)
    with pytest.raises(ValueError):
        SupervisorConfig(T_max_hard=float("nan"))

    zc = _make_zone_controller()
    sup = ZoneSupervisor("zone_1", zc, cfg=SupervisorConfig(T_max_hard=40.0, t_max_continuous_heat=2, t_max_continuous_water=2))
    bounds = _standard_bounds()
    amb = AmbientState(24.0, 45.0)

    # Continuous heat/water software interlock clamp after 2 steps
    m_ok = _nominal_meas(T=22.0, H=65.0, M=45.0)
    assert sup.step([1, 0, 0, 1], m_ok, bounds, amb).executed_action == (1, 0, 0, 1)
    assert sup.step([1, 0, 0, 1], m_ok, bounds, amb).executed_action == (1, 0, 0, 1)
    # 3rd consecutive step exceeds t_max_continuous_heat/water -> clamped to 0 and triggers PID fallback
    st_clamp = sup.step([1, 0, 0, 1], m_ok, bounds, amb)
    assert st_clamp.mode == ControlMode.PID_FALLBACK
    assert st_clamp.executed_action[0] == 0

    # Hard over-temperature interlock clamps heat=0 even on direct call
    hot_meas = _nominal_meas(T=42.0)
    assert sup._enforce_software_interlocks((1, 1, 0, 0), hot_meas) == (0, 1, 0, 0)


def test_fallback_manager_isolates_zone_crop_target_failure_to_safe_mode(monkeypatch):
    from pathlib import Path
    from control import CropConfigLoader, FallbackManager, load_pid_tuning

    root = Path(__file__).resolve().parent.parent
    loader = CropConfigLoader.from_json_file(root / "configs" / "crops_dataset.json")
    loader.assign_zones_from_json_file(root / "configs" / "zone_assignments.json")
    fm = FallbackManager(loader, load_pid_tuning(root / "configs" / "pid_tuning.json"))

    orig_targets = loader.get_zone_targets

    def _fail_zone2(z):
        if z == "zone_2":
            raise KeyError("corrupted zone_2 crop profile")
        return orig_targets(z)

    monkeypatch.setattr(loader, "get_zone_targets", _fail_zone2)
    obs = {
        "zone_1": [22.0, 65.0, 45.0],
        "zone_2": [22.0, 65.0, 45.0],
        "zone_3": [22.0, 65.0, 45.0],
    }
    acts = {"zone_1": [0, 0, 0, 0], "zone_2": [1, 1, 1, 1], "zone_3": [0, 0, 0, 0]}
    res = fm.step(acts, obs, ambient=AmbientState(24.0, 45.0), inputs_are_trust_records=False)

    assert res.statuses["zone_2"].mode == ControlMode.SAFE_MODE
    assert tuple(res.actions["zone_2"].tolist()) == (0, 0, 0, 0)
    assert res.statuses["zone_1"].mode == ControlMode.MARL_ACTIVE
    assert res.statuses["zone_3"].mode == ControlMode.MARL_ACTIVE
    assert fm.get_metrics()["zone_2"]["pid_failures"] == 1
