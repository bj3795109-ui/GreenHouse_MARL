"""Phase 3 unit and regression tests for ZoneController, ActuatorMapper, and ActuatorArbiter.

Covers ZONE-01 through ZONE-06, ACT-01 through ACT-06, and input/handover hardening.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from control.actuator_arbiter import ActuatorArbiter
from control.actuator_mapper import ActuatorMapper, MappedRequests
from control.crop_config_loader import CropConfigLoader, load_pid_tuning
from control.types import (
    ActuatorConfig,
    AmbientState,
    CropBounds,
    MeasurementQuality,
    PIDGains,
    SensorMeasurement,
    VARS,
    ZONE_IDS,
)
from control.zone_controller import ZoneController

DATASET_PATH = Path(__file__).resolve().parent.parent / "configs" / "crops_dataset.json"
TUNING_PATH = Path(__file__).resolve().parent.parent / "configs" / "pid_tuning.json"


def _bounds(var: str, min_a: float, opt_min: float, opt_max: float, max_a: float, crop: str = "tomato") -> CropBounds:
    return CropBounds(crop, "vegetative", var, "unit", min_a, opt_min, opt_max, max_a)


def _default_bounds(crop: str = "tomato") -> dict[str, CropBounds]:
    return {
        "T": _bounds("T", 16.0, 21.0, 26.0, 32.0, crop=crop),
        "H": _bounds("H", 45.0, 60.0, 75.0, 85.0, crop=crop),
        "M": _bounds("M", 25.0, 40.0, 55.0, 70.0, crop=crop),
    }


def _meas(var: str, val: float | None, quality: MeasurementQuality = MeasurementQuality.TRUSTED) -> SensorMeasurement:
    finite = val is not None and math.isfinite(val)
    return SensorMeasurement(
        var=var,
        value=val,
        raw=val,
        combined=val,
        belief=val,
        trust=1.0 if quality == MeasurementQuality.TRUSTED else 0.0,
        confidence=0.9 if quality != MeasurementQuality.UNRELIABLE_FAILSAFE else 0.0,
        quality=quality,
        valid_raw=finite,
        valid_est=finite,
    )


def _make_meas_dict(T: float | None = 23.0, H: float | None = 65.0, M: float | None = 45.0) -> dict[str, SensorMeasurement]:
    return {"T": _meas("T", T), "H": _meas("H", H), "M": _meas("M", M)}


def _default_gains() -> dict[str, PIDGains]:
    return {
        "T": PIDGains(Kp=0.40, Ki=0.002, Kd=2.0, tau_f=5.0, T_t=20.0),
        "H": PIDGains(Kp=0.12, Ki=0.005, Kd=0.0, tau_f=2.0, T_t=10.0),
        "M": PIDGains(Kp=0.10, Ki=0.004, Kd=0.0, u_min=0.0, u_max=1.0, unidirectional_positive=True),
    }


# ---------------------------------------------------------------------------
# ZONE-01 .. ZONE-06: Three-zone independence
# ---------------------------------------------------------------------------

def test_zone01_all_zones_operate_normally_and_independently():
    controllers = {z: ZoneController(z, _default_gains()) for z in ZONE_IDS}
    bounds = _default_bounds()
    amb = AmbientState(24.0, 45.0)
    nominal = _make_meas_dict(23.0, 65.0, 45.0)

    outputs = {z: controllers[z].step(nominal, bounds, amb) for z in ZONE_IDS}
    for z in ZONE_IDS:
        assert outputs[z].action == (0, 0, 0, 0)
        assert outputs[z].u_T == 0.0
        assert outputs[z].u_H == 0.0
        assert outputs[z].u_M == 0.0


def test_zone02_zone1_temperature_change_only_affects_zone1():
    controllers = {z: ZoneController(z, _default_gains()) for z in ZONE_IDS}
    bounds = _default_bounds()
    amb = AmbientState(24.0, 45.0)

    out1 = controllers["zone_1"].step(_make_meas_dict(T=15.0), bounds, amb)
    out2 = controllers["zone_2"].step(_make_meas_dict(T=23.0), bounds, amb)
    out3 = controllers["zone_3"].step(_make_meas_dict(T=23.0), bounds, amb)

    assert out1.u_T > 0.5
    assert out1.action[0] == 1  # zone_1 heats
    assert out2.action == (0, 0, 0, 0)
    assert out3.action == (0, 0, 0, 0)


def test_zone03_zone2_soil_moisture_change_only_affects_zone2():
    cfg = ActuatorConfig(water_max_pulse_steps=2, water_soak_steps=4, min_on_steps=1, min_off_steps=1)
    controllers = {z: ZoneController(z, _default_gains(), cfg) for z in ZONE_IDS}
    bounds = _default_bounds()
    amb = AmbientState(24.0, 45.0)

    out1 = controllers["zone_1"].step(_make_meas_dict(M=45.0), bounds, amb)
    out2 = controllers["zone_2"].step(_make_meas_dict(M=20.0), bounds, amb)
    out3 = controllers["zone_3"].step(_make_meas_dict(M=45.0), bounds, amb)

    assert out2.u_M > 0.5
    assert out2.action[3] == 1
    assert out1.action[3] == 0
    assert out3.action[3] == 0
    assert controllers["zone_2"].mapper._water_pulse_steps == 1
    assert controllers["zone_1"].mapper._water_pulse_steps == 0
    assert controllers["zone_3"].mapper._water_pulse_steps == 0


def test_zone04_zone1_pid_integral_changes_leave_zone2_and_zone3_unchanged():
    controllers = {z: ZoneController(z, _default_gains()) for z in ZONE_IDS}
    bounds = _default_bounds()
    amb = AmbientState(24.0, 45.0)

    for _ in range(10):
        controllers["zone_1"].step(_make_meas_dict(T=19.5, H=55.0, M=35.0), bounds, amb)
        controllers["zone_2"].step(_make_meas_dict(T=23.0, H=65.0, M=45.0), bounds, amb)
        controllers["zone_3"].step(_make_meas_dict(T=23.0, H=65.0, M=45.0), bounds, amb)

    for v in VARS:
        assert controllers["zone_1"].pids[v].state.integral > 0.0
        assert controllers["zone_2"].pids[v].state.integral == 0.0
        assert controllers["zone_3"].pids[v].state.integral == 0.0


def test_zone05_zone3_sensor_failsafe_does_not_affect_zones_1_and_2():
    controllers = {z: ZoneController(z, _default_gains()) for z in ZONE_IDS}
    bounds = _default_bounds()
    amb = AmbientState(20.0, 40.0)

    failed_z3 = {
        "T": _meas("T", None, quality=MeasurementQuality.UNRELIABLE_FAILSAFE),
        "H": _meas("H", None, quality=MeasurementQuality.UNRELIABLE_FAILSAFE),
        "M": _meas("M", None, quality=MeasurementQuality.UNRELIABLE_FAILSAFE),
    }

    out1 = controllers["zone_1"].step(_make_meas_dict(T=17.0), bounds, amb)
    out2 = controllers["zone_2"].step(_make_meas_dict(M=25.0), bounds, amb)
    out3 = controllers["zone_3"].step(failed_z3, bounds, amb)

    assert out3.action == (0, 0, 0, 0)
    assert out1.action[0] == 1  # zone_1 keeps heating
    assert out2.action[3] == 1  # zone_2 keeps watering


def test_zone06_different_crop_configs_and_per_zone_gains():
    loader = CropConfigLoader.from_json_file(DATASET_PATH)
    loader.assign_zone("zone_1", "tomato", "vegetative")
    loader.assign_zone("zone_2", "lettuce", "mature")
    loader.assign_zone("zone_3", "cucumber", "flowering")

    custom_tuning = {
        "zone_1": {
            "T": {"Kp": 0.40, "Ki": 0.002, "Kd": 2.0},
            "H": {"Kp": 0.12, "Ki": 0.005},
            "M": {"Kp": 0.10, "Ki": 0.004, "u_min": 0.0, "u_max": 1.0, "unidirectional_positive": True},
        },
        "zone_2": {
            "T": {"Kp": 0.60, "Ki": 0.010, "Kd": 1.0},
            "H": {"Kp": 0.15, "Ki": 0.005},
            "M": {"Kp": 0.20, "Ki": 0.008, "u_min": 0.0, "u_max": 1.0, "unidirectional_positive": True},
        },
        "zone_3": {
            "T": {"Kp": 0.25, "Ki": 0.001, "Kd": 0.5},
            "H": {"Kp": 0.10, "Ki": 0.002},
            "M": {"Kp": 0.08, "Ki": 0.002, "u_min": 0.0, "u_max": 1.0, "unidirectional_positive": True},
        },
    }
    gains = load_pid_tuning(custom_tuning)
    controllers = {z: ZoneController(z, gains[z]) for z in ZONE_IDS}
    amb = AmbientState(15.0, 40.0)

    # Identical 22°C reading: tomato (21-26) is optimal, lettuce (15-20) cools, cucumber (23-28) heats
    meas = _make_meas_dict(T=22.0, H=65.0, M=45.0)
    out1 = controllers["zone_1"].step(meas, loader.get_zone_targets("zone_1"), amb)
    out2 = controllers["zone_2"].step(meas, loader.get_zone_targets("zone_2"), amb)
    out3 = controllers["zone_3"].step(meas, loader.get_zone_targets("zone_3"), amb)

    assert out1.u_T == 0.0
    assert out2.u_T < 0.0 and out2.action[1] == 1  # lettuce cools
    assert out3.u_T > 0.0 and out3.action[0] == 1  # cucumber heats
    assert controllers["zone_2"].pids["T"].gains.Kp == pytest.approx(0.60)
    assert controllers["zone_3"].pids["T"].gains.Kp == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# ACT-01 .. ACT-06 & Phase 3 validation / handover regressions
# ---------------------------------------------------------------------------

def test_act01_heater_and_fan_conflict_applies_mutual_exclusion():
    cfg = ActuatorConfig(min_on_steps=1, min_off_steps=1, pwm_period_heat=10)
    arbiter = ActuatorArbiter(cfg)
    bT = _bounds("T", 16.0, 21.0, 26.0, 32.0)
    bH = _bounds("H", 45.0, 60.0, 75.0, 85.0)

    # Both heat_cmd=1 and strong dehum_demand=0.9 (fan dries because H=82 > amb_H=40)
    # T=25 is cooler than 32 and inside acceptable, while H=82 is near max_acceptable; fan cools (25 > 18)
    req = MappedRequests(
        heat_cmd=1,
        mist_cmd=0,
        water_cmd=0,
        cool_demand=0.0,
        heat_assist_demand=0.2,
        dehum_demand=0.9,
        hum_assist_demand=0.0,
        water_soak_blocked=False,
    )
    res = arbiter.arbitrate(req, _meas("T", 25.0), _meas("H", 82.0), bT, bH, AmbientState(18.0, 40.0))
    assert res.action[1] == 1  # fan ON for dehumidification
    assert res.action[0] == 0  # heater suppressed (mutual exclusion)
    assert res.block_flags.block_T_pos is True


def test_act01b_mister_and_fan_conflict_applies_mutual_exclusion():
    cfg = ActuatorConfig(min_on_steps=1, min_off_steps=1)
    arbiter = ActuatorArbiter(cfg)
    bT = _bounds("T", 16.0, 21.0, 26.0, 32.0)
    bH = _bounds("H", 45.0, 60.0, 75.0, 85.0)

    # Over-temperature (33.5 > 32) drives cooling fan ON while mist_cmd=1; ambient is drier (35 < 55)
    req = MappedRequests(
        heat_cmd=0,
        mist_cmd=1,
        water_cmd=0,
        cool_demand=1.0,
        heat_assist_demand=0.0,
        dehum_demand=0.0,
        hum_assist_demand=0.5,
        water_soak_blocked=False,
    )
    res = arbiter.arbitrate(req, _meas("T", 33.5), _meas("H", 55.0), bT, bH, AmbientState(20.0, 35.0))
    assert res.action[1] == 1  # fan ON
    assert res.action[2] == 0  # mist suppressed
    assert res.block_flags.block_H_pos is True


def test_act02_pump_requested_repeatedly_enforces_pulse_and_soak_limits():
    cfg = ActuatorConfig(water_max_pulse_steps=2, water_soak_steps=5, min_on_steps=1, min_off_steps=1)
    mapper = ActuatorMapper(cfg)
    cmds = [mapper.map_outputs(0.0, 0.0, 1.0).water_cmd for _ in range(14)]
    assert cmds == [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_act03_fan_minimum_on_and_off_dwell_respected():
    cfg = ActuatorConfig(min_on_steps=3, min_off_steps=3, fan_on_thresh=0.2, fan_off_thresh=0.08)
    mapper = ActuatorMapper(cfg)
    arbiter = ActuatorArbiter(cfg)
    bT = _bounds("T", 16.0, 21.0, 25.0, 32.0)
    bH = _bounds("H", 45.0, 60.0, 75.0, 85.0)
    amb = AmbientState(18.0, 40.0)

    # Step 0: high cool demand -> fan turns ON
    r0 = arbiter.arbitrate(mapper.map_outputs(-0.9, 0.0, 0.0), _meas("T", 29.0), _meas("H", 65.0), bT, bH, amb)
    assert r0.action[1] == 1
    # Steps 1..2: demand drops to 0, fan MUST stay ON for min_on_steps=3
    for _ in range(2):
        r = arbiter.arbitrate(mapper.map_outputs(0.0, 0.0, 0.0), _meas("T", 23.0), _meas("H", 65.0), bT, bH, amb)
        assert r.action[1] == 1
    # Step 3: min_on_steps satisfied -> fan turns OFF
    r3 = arbiter.arbitrate(mapper.map_outputs(0.0, 0.0, 0.0), _meas("T", 23.0), _meas("H", 65.0), bT, bH, amb)
    assert r3.action[1] == 0
    # Steps 4..5: demand spikes back, fan MUST stay OFF for min_off_steps=3
    for _ in range(2):
        r = arbiter.arbitrate(mapper.map_outputs(-0.9, 0.0, 0.0), _meas("T", 29.0), _meas("H", 65.0), bT, bH, amb)
        assert r.action[1] == 0
    # Step 6: min_off_steps satisfied -> fan turns ON again
    r6 = arbiter.arbitrate(mapper.map_outputs(-0.9, 0.0, 0.0), _meas("T", 29.0), _meas("H", 65.0), bT, bH, amb)
    assert r6.action[1] == 1


@pytest.mark.parametrize("bad_u", [float("nan"), float("inf"), float("-inf"), 1.5, -1.5, True])
def test_act04_invalid_actuator_outputs_rejected_by_mapper(bad_u):
    mapper = ActuatorMapper()
    with pytest.raises(ValueError):
        mapper.map_outputs(bad_u, 0.0, 0.0)
    with pytest.raises(ValueError):
        mapper.map_outputs(0.0, bad_u, 0.0)
    with pytest.raises(ValueError):
        mapper.map_outputs(0.0, 0.0, bad_u)


def test_act04_malformed_requests_and_action_sync_rejected():
    mapper = ActuatorMapper()
    arbiter = ActuatorArbiter()
    bT = _bounds("T", 16.0, 21.0, 25.0, 32.0)
    bH = _bounds("H", 45.0, 60.0, 75.0, 85.0)
    amb = AmbientState(20.0, 40.0)

    with pytest.raises(ValueError):
        mapper.sync_executed((1, 0, 0))  # wrong length
    with pytest.raises(ValueError):
        mapper.sync_executed((2, 0, 0, 0))  # non-binary
    with pytest.raises(ValueError):
        arbiter.sync_fan(2)

    bad_req = MappedRequests(
        heat_cmd=2,
        mist_cmd=0,
        water_cmd=0,
        cool_demand=float("nan"),
        heat_assist_demand=0.0,
        dehum_demand=0.0,
        hum_assist_demand=0.0,
        water_soak_blocked=False,
    )
    with pytest.raises(ValueError):
        arbiter.arbitrate(bad_req, _meas("T", 23.0), _meas("H", 65.0), bT, bH, amb)


@pytest.mark.parametrize(
    "bad_amb",
    [
        AmbientState(T_ambient=float("nan"), H_ambient=45.0),
        AmbientState(T_ambient=24.0, H_ambient=float("inf")),
        AmbientState(T_ambient=float("-inf"), H_ambient=45.0),
    ],
)
def test_invalid_ambient_measurements_rejected(bad_amb):
    zc = ZoneController("zone_1", _default_gains())
    bounds = _default_bounds()
    meas = _make_meas_dict(23.0, 65.0, 45.0)
    with pytest.raises(ValueError):
        zc.step(meas, bounds, bad_amb)
    with pytest.raises(ValueError):
        zc.track_shadow(meas, bounds, (0, 0, 0, 0), bad_amb)


def test_invalid_measurement_values_rejected_by_arbiter_directly():
    arbiter = ActuatorArbiter()
    mapper = ActuatorMapper()
    bT = _bounds("T", 16.0, 21.0, 25.0, 32.0)
    bH = _bounds("H", 45.0, 60.0, 75.0, 85.0)
    req = mapper.map_outputs(-0.8, 0.0, 0.0)
    with pytest.raises(ValueError):
        arbiter.arbitrate(req, _meas("T", float("nan")), _meas("H", 65.0), bT, bH, AmbientState(20.0, 40.0))


def test_act05_and_irrigation_handover_preserves_pulse_budget_and_soak_lockout():
    """Regression: MARL watering must advance pulse counter and enforce soak lockout on PID takeover."""
    cfg = ActuatorConfig(water_max_pulse_steps=2, water_soak_steps=4, min_on_steps=1, min_off_steps=1)
    zc = ZoneController("zone_1", _default_gains(), cfg)
    bounds = _default_bounds()
    amb = AmbientState(24.0, 45.0)
    dry_meas = _make_meas_dict(T=23.0, H=65.0, M=20.0)

    # Case 1: MARL waters for 1 step, then hands over to PID -> PID gets only 1 remaining pulse step, then soak!
    zc.track_shadow(dry_meas, bounds, (0, 0, 0, 1), amb)
    out_p1 = zc.step(dry_meas, bounds, amb)
    out_p2 = zc.step(dry_meas, bounds, amb)
    assert out_p1.action[3] == 1
    assert out_p2.action[3] == 0  # soak lockout triggered after 2 total pulse steps

    # Case 2: Continuous MARL watering reaches pulse limit (2 steps), then immediate PID fallback -> soak immediately!
    zc2 = ZoneController("zone_2", _default_gains(), cfg)
    zc2.track_shadow(dry_meas, bounds, (0, 0, 0, 1), amb)
    zc2.track_shadow(dry_meas, bounds, (0, 0, 0, 1), amb)
    soak_cmds = [zc2.step(dry_meas, bounds, amb).action[3] for _ in range(4)]
    assert soak_cmds == [0, 0, 0, 0]
    # After 4 soak steps, PID may start a fresh pulse
    assert zc2.step(dry_meas, bounds, amb).action[3] == 1


def test_independent_irrigation_state_across_three_zones():
    cfg = ActuatorConfig(water_max_pulse_steps=2, water_soak_steps=4, min_on_steps=1, min_off_steps=1)
    controllers = {z: ZoneController(z, _default_gains(), cfg) for z in ZONE_IDS}
    bounds = _default_bounds()
    amb = AmbientState(24.0, 45.0)
    dry = _make_meas_dict(M=20.0)

    # Zone 1 exhausts pulse via MARL shadow tracking; Zone 2 stays idle; Zone 3 runs 1 pulse step
    controllers["zone_1"].track_shadow(dry, bounds, (0, 0, 0, 1), amb)
    controllers["zone_1"].track_shadow(dry, bounds, (0, 0, 0, 1), amb)
    controllers["zone_3"].track_shadow(dry, bounds, (0, 0, 0, 1), amb)

    assert controllers["zone_1"].step(dry, bounds, amb).action[3] == 0  # zone_1 locked in soak
    assert controllers["zone_2"].step(dry, bounds, amb).action[3] == 1  # zone_2 starts fresh pulse
    assert controllers["zone_3"].step(dry, bounds, amb).action[3] == 1  # zone_3 uses 2nd step of pulse


def test_act06_simultaneous_actuator_requests_and_switch_back_within_limits():
    cfg = ActuatorConfig(min_on_steps=2, min_off_steps=2, pwm_period_heat=4, pwm_period_mist=4, water_max_pulse_steps=2, water_soak_steps=4)
    zc = ZoneController("zone_1", _default_gains(), cfg)
    bounds = _default_bounds()
    amb = AmbientState(15.0, 35.0)

    # Simultaneous low T, low H, low M -> heat, mist, and water requested together (fan off)
    cold_dry = _make_meas_dict(T=17.0, H=50.0, M=25.0)
    out = zc.step(cold_dry, bounds, amb)
    assert out.action in [(1, 0, 1, 1), (1, 0, 1, 0), (1, 0, 0, 1), (0, 0, 1, 1)]
    assert out.action == (1, 0, 1, 1)

    # Switch to MARL tracking and back: dwell timers stay valid and outputs remain binary {0, 1}
    zc.track_shadow(cold_dry, bounds, (1, 0, 1, 0), amb)
    out_back = zc.step(cold_dry, bounds, amb)
    assert all(bit in (0, 1) for bit in out_back.action)


def test_zone_controller_validates_zone_id_missing_keys_and_actuator_names():
    from control.actuator_mapper import validate_actuator_name

    with pytest.raises(ValueError):
        validate_actuator_name("light")

    with pytest.raises(ValueError):
        ZoneController("", _default_gains())

    incomplete_gains = {"T": _default_gains()["T"], "H": _default_gains()["H"]}
    with pytest.raises(ValueError):
        ZoneController("zone_1", incomplete_gains)

    zc = ZoneController("zone_1", _default_gains())
    with pytest.raises(ValueError):
        zc.step({"T": _meas("T", 22.0)}, _default_bounds(), AmbientState(24.0, 45.0))


def test_zone_controller_dt_forwarding_and_shared_config_isolation():
    cfg = ActuatorConfig(min_on_steps=2, min_off_steps=2)
    gains = _default_gains()
    z1 = ZoneController("zone_1", gains, cfg)
    z2 = ZoneController("zone_2", gains, cfg)

    # Mutating original config after construction must not affect instantiated zones
    cfg.min_on_steps = 10
    assert z1.mapper.cfg.min_on_steps == 2
    assert z2.mapper.cfg.min_on_steps == 2

    out = z1.step(_make_meas_dict(T=18.0), _default_bounds(), AmbientState(24.0, 45.0), dt=2.0)
    assert out.u_T > 0.0


def test_arbiter_dehumidification_conflict_and_cold_dehum_override():
    # Case 1: H wants dehumidification while T is below optimal and harmed by cool fan
    cfg = ActuatorConfig(min_on_steps=1, min_off_steps=1, allow_cold_dehumidification=False)
    arbiter = ActuatorArbiter(cfg)
    bT = _bounds("T", 16.0, 21.0, 26.0, 32.0)
    bH = _bounds("H", 45.0, 60.0, 75.0, 85.0)
    req = MappedRequests(
        heat_cmd=1,
        mist_cmd=0,
        water_cmd=0,
        cool_demand=0.0,
        heat_assist_demand=0.8,
        dehum_demand=0.9,
        hum_assist_demand=0.0,
        water_soak_blocked=False,
    )
    # H is critical (88 > 85) while T is inside acceptable (20 > 16) -> H critical priority turns fan ON
    res_h = arbiter.arbitrate(req, _meas("T", 20.0), _meas("H", 88.0), bT, bH, AmbientState(15.0, 40.0))
    assert res_h.action[1] == 1
    assert res_h.action[0] == 0  # heat blocked by cooling fan

    # Case 2: allow_cold_dehumidification=True permits simultaneous heat + fan when H critical
    arb2 = ActuatorArbiter(ActuatorConfig(min_on_steps=1, min_off_steps=1, allow_cold_dehumidification=True))
    res_cold_dehum = arb2.arbitrate(req, _meas("T", 20.0), _meas("H", 88.0), bT, bH, AmbientState(15.0, 40.0))
    assert res_cold_dehum.action[1] == 1
    assert res_cold_dehum.action[0] == 1
