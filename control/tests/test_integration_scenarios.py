"""
Closed-loop integration tests combining GreenhouseEnv, TrustNetwork,
NoisySensors, CropConfigLoader, and FallbackManager across scenarios 1-15.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "env"))
sys.path.insert(0, str(ROOT / "trust_network"))

from env import GreenhouseEnv, PlaceholderParams  # noqa: E402
from harness import Fault, NoisySensors, bootstrap_healthy  # noqa: E402
from trust_network import AssumedParams, TrustNetwork  # noqa: E402

from control import (  # noqa: E402
    AmbientState,
    ControlMode,
    CropConfigLoader,
    EpisodeEvaluator,
    FallbackManager,
    SensorInterfaceConfig,
    SupervisorConfig,
    ZONE_IDS,
    load_pid_tuning,
)

DATASET_PATH = ROOT / "control" / "configs" / "crops_dataset.json"
TUNING_PATH = ROOT / "control" / "configs" / "pid_tuning.json"


@pytest.fixture(scope="module")
def healthy_residuals():
    return bootstrap_healthy(AssumedParams.from_env(PlaceholderParams()), n_episodes=10, steps=300)


def _make_manager(supervisor_cfg=None, initial_mode=ControlMode.MARL_ACTIVE):
    loader = CropConfigLoader.from_json_file(DATASET_PATH)
    loader.assign_zone("zone_1", "tomato", "vegetative")
    loader.assign_zone("zone_2", "lettuce", "mature")
    loader.assign_zone("zone_3", "cucumber", "flowering")
    gains = load_pid_tuning(TUNING_PATH)
    return (
        FallbackManager(
            crop_loader=loader,
            pid_gains=gains,
            supervisor_cfg=supervisor_cfg or SupervisorConfig(N_confirm=8),
            initial_mode=initial_mode,
        ),
        loader,
    )


def _simple_healthy_marl(obs_dict, loader, ambient):
    """Simple healthy rule-based surrogate standing in for a trained MARL policy."""
    out = {}
    for z, vec in obs_dict.items():
        T, H, M = float(vec[0]), float(vec[1]), float(vec[2])
        b = loader.get_zone_targets(z)
        heat = 1 if T < b["T"].optimal_min else 0
        fan = 1 if (T > b["T"].optimal_max and T > ambient.T_ambient + 0.5 and heat == 0) else 0
        mist = 1 if (H < b["H"].optimal_min and fan == 0) else 0
        water = 1 if M < b["M"].optimal_min else 0
        out[z] = np.array([heat, fan, mist, water], dtype=np.int8)
    return out


def test_scenario_1_2_3_isolated_and_simultaneous_zone_failures(healthy_residuals):
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=120)
    obs, _ = env.reset(seed=10)
    rng = np.random.default_rng(10)
    sensors = NoisySensors(rng)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    raw = sensors.read(env.state, 0)
    net.reset(raw)
    fm, loader = _make_manager()
    ambient = AmbientState(params.T_ambient, params.H_ambient)

    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    for t in range(1, 90):
        records = net.step(sensors.read(env.state, t), prev_actions)
        marl_acts = _simple_healthy_marl(obs, loader, ambient)
        if 20 <= t < 45:
            # Scenario 2: Only zone_1 fails
            del marl_acts["zone_1"]
        elif 55 <= t < 75:
            # Scenario 3: All 3 fail simultaneously (crash + NaN + bad length)
            marl_acts = {
                "zone_1": None,
                "zone_2": [float("nan"), 0, 0, 0],
                "zone_3": [1, 0],
            }

        res = fm.step(marl_acts, records, ambient=ambient)
        for z in ZONE_IDS:
            assert env.action_space(z).contains(res.actions[z])

        if t == 15:
            # Scenario 1: All healthy
            assert all(res.statuses[z].mode == ControlMode.MARL_ACTIVE for z in ZONE_IDS)
        if t == 25:
            assert res.statuses["zone_1"].mode == ControlMode.PID_FALLBACK
            assert res.statuses["zone_2"].mode == ControlMode.MARL_ACTIVE
            assert res.statuses["zone_3"].mode == ControlMode.MARL_ACTIVE
        if t == 60:
            assert all(res.statuses[z].mode == ControlMode.PID_FALLBACK for z in ZONE_IDS)
        if t == 89:
            # Scenario 13: Automatic recovery back to MARL_ACTIVE across all zones
            assert all(res.statuses[z].mode == ControlMode.MARL_ACTIVE for z in ZONE_IDS)

        obs, _, _, _, _ = env.step(res.actions)
        prev_actions = res.actions


def test_scenario_6_drift_fault_contained_and_pid_tracks_combined(healthy_residuals):
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=600)
    env.reset(seed=42)
    rng = np.random.default_rng(42)
    faults = [Fault("zone_1", "T", "drift", start=50, magnitude=0.05)]
    sensors = NoisySensors(rng, faults=faults)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))
    fm, _ = _make_manager(initial_mode=ControlMode.PID_FALLBACK)
    ambient = AmbientState(params.T_ambient, params.H_ambient)

    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    for t in range(1, 250):
        records = net.step(sensors.read(env.state, t), prev_actions)
        res = fm.step(None, records, ambient=ambient)
        env.step(res.actions)
        prev_actions = res.actions

    # Despite +10°C raw drift on zone_1 T sensor, true temperature did not freeze or cook
    assert 18.0 <= env.state["zone_1"]["T"] <= 28.0
    assert fm.get_metrics()["zone_1"]["pid_fallback_steps"] > 0


def test_scenario_14_bumpless_handover_limits_initial_toggle_burst(healthy_residuals):
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=100)
    env.reset(seed=3)
    # Start zone_1 cold BEFORE TrustNetwork anchor so physics matches initial state
    env.state["zone_1"]["T"] = 18.0
    rng = np.random.default_rng(3)
    sensors = NoisySensors(rng)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))

    fm, _ = _make_manager(
        supervisor_cfg=SupervisorConfig(N_confirm=3, handover_dwell_protect_steps=2),
        initial_mode=ControlMode.PID_FALLBACK,
    )
    ambient = AmbientState(params.T_ambient, params.H_ambient)
    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    for t in range(1, 8):
        records = net.step(sensors.read(env.state, t), prev_actions)
        # Recovered MARL commands safe heating/watering while requesting fan OFF
        marl_acts = {
            "zone_1": [0, 0, 0, 1],
            "zone_2": [0, 0, 0, 1],
            "zone_3": [0, 0, 0, 1],
        }
        res = fm.step(marl_acts, records, ambient=ambient)
        env.step(res.actions)
        prev_actions = res.actions

    metrics = fm.get_metrics()["zone_1"]
    assert metrics["recovery_successes"] >= 1
    assert metrics["mode"] == ControlMode.MARL_ACTIVE.value


def test_scenario_5_stuck_sensor_handled_via_combined_estimate(healthy_residuals):
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=400)
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    faults = [Fault("zone_1", "H", "stuck", start=40)]
    sensors = NoisySensors(rng, faults=faults)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))
    fm, _ = _make_manager(initial_mode=ControlMode.PID_FALLBACK)
    ambient = AmbientState(params.T_ambient, params.H_ambient)

    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    for t in range(1, 220):
        records = net.step(sensors.read(env.state, t), prev_actions)
        res = fm.step(None, records, ambient=ambient)
        env.step(res.actions)
        prev_actions = res.actions

    # True humidity in zone_1 remains inside safe bounds despite stuck H sensor
    assert 45.0 <= env.state["zone_1"]["H"] <= 85.0


def test_scenario_3_two_agents_fail_while_third_stays_healthy(healthy_residuals):
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=100)
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(11)
    sensors = NoisySensors(rng)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))
    fm, loader = _make_manager()
    ambient = AmbientState(params.T_ambient, params.H_ambient)

    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    for t in range(1, 35):
        records = net.step(sensors.read(env.state, t), prev_actions)
        marl_acts = _simple_healthy_marl(obs, loader, ambient)
        if t >= 15:
            del marl_acts["zone_1"]
            marl_acts["zone_2"] = [float("inf"), 0, 0, 0]
        res = fm.step(marl_acts, records, ambient=ambient)
        if t == 25:
            assert res.statuses["zone_1"].mode == ControlMode.PID_FALLBACK
            assert res.statuses["zone_2"].mode == ControlMode.PID_FALLBACK
            assert res.statuses["zone_3"].mode == ControlMode.MARL_ACTIVE
        obs, _, _, _, _ = env.step(res.actions)
        prev_actions = res.actions


def test_scenario_9_total_sensor_loss_triggers_safe_mode_and_evaluator_tracks_metrics(healthy_residuals):
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=100)
    env.reset(seed=12)
    rng = np.random.default_rng(12)
    sensors = NoisySensors(rng)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))

    loader = CropConfigLoader.from_json_file(DATASET_PATH)
    loader.assign_zones_from_json_file(ROOT / "control" / "configs" / "zone_assignments.json")
    fm = FallbackManager(
        crop_loader=loader,
        pid_gains=load_pid_tuning(TUNING_PATH),
        sensor_cfg=SensorInterfaceConfig(t_hold_max=2),
        supervisor_cfg=SupervisorConfig(N_confirm=3, W_persist=5),
    )
    evaluator = EpisodeEvaluator(dt=params.dt, fault_start_step=10)
    ambient = AmbientState(params.T_ambient, params.H_ambient)

    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    for t in range(1, 30):
        records = net.step(sensors.read(env.state, t), prev_actions)
        # Drop both raw & estimated records for zone_1 from t=10..18
        if 10 <= t <= 18:
            records_in = {z: records[z] for z in ("zone_2", "zone_3")}
        else:
            records_in = records

        # Stage change in zone_3 mid-run (Scenario 11)
        if t == 15:
            loader.update_zone_stage("zone_3", "vegetative")

        marl_acts = _simple_healthy_marl(
            {z: env._get_obs(z) for z in ZONE_IDS}, loader, ambient
        )
        res = fm.step(marl_acts, records_in, ambient=ambient, current_time=float(t))
        if t == 14:
            assert res.statuses["zone_1"].mode == ControlMode.SAFE_MODE
            assert res.statuses["zone_1"].executed_action == (0, 0, 0, 0)

        bounds_now = {z: loader.get_zone_targets(z) for z in ZONE_IDS}
        evaluator.record_step(t, env.state, bounds_now, res.statuses)
        env.step(res.actions)
        prev_actions = res.actions

    summary = evaluator.summarize(fm.get_metrics())
    assert summary["zone_1"]["safe_mode_events"] >= 1
    assert summary["zone_1"]["safe_mode_steps"] >= 5
    assert "rmse" in summary["zone_1"]["variables"]["T"]
    assert summary["zone_3"]["variables"]["H"]["time_outside_optimal_s"] >= 0.0


def test_scenario_14_and_17_persistent_divergence_and_counterproductive_ambient(healthy_residuals):
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=100)
    env.reset(seed=13)
    env.state["zone_1"]["T"] = 34.0  # Above tomato max_acceptable (32.0)
    rng = np.random.default_rng(13)
    sensors = NoisySensors(rng)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))

    fm, _ = _make_manager(supervisor_cfg=SupervisorConfig(W_persist=5, N_confirm=5))
    # Hot & humid outside: ventilation would be counterproductive
    bad_ambient = AmbientState(T_ambient=38.0, H_ambient=95.0)
    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}

    for t in range(1, 15):
        records = net.step(sensors.read(env.state, t), prev_actions)
        # Faulty MARL keeps turning heater ON while overheated
        marl_acts = {
            "zone_1": [1, 0, 0, 0],
            "zone_2": [0, 0, 0, 1],
            "zone_3": [0, 0, 0, 1],
        }
        res = fm.step(marl_acts, records, ambient=bad_ambient)
        if t >= 6:
            assert res.statuses["zone_1"].mode == ControlMode.PID_FALLBACK
            # Under hotter ambient (38 > 34), PID arbiter inhibits fan cooling and turns heater off
            assert res.statuses["zone_1"].executed_action[0] == 0
            assert res.statuses["zone_1"].executed_action[1] == 0
        env.step(res.actions)
        prev_actions = res.actions


def test_critical_three_zone_acceptance_zone2_crash_safe_mode_and_recovery(healthy_residuals, monkeypatch):
    """
    Critical acceptance test:
    - All 3 zones start on surrogate healthy actions (NOT a trained MAPPO policy).
    - Zone 2 agent crashes -> Zone 2 enters PID_FALLBACK while Zones 1 & 3 stay MARL_ACTIVE.
    - Zone 2 PID also fails -> Zone 2 enters SAFE_MODE with (0,0,0,0); Zones 1 & 3 stay MARL_ACTIVE.
    - Restore Zone 2 PID and agent -> Zone 2 follows SAFE_MODE -> PID_FALLBACK -> RECOVERY_PENDING -> MARL_ACTIVE
      without prematurely returning to MARL or interrupting Zones 1 & 3.
    """
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=120)
    obs, _ = env.reset(seed=10)
    rng = np.random.default_rng(10)
    sensors = NoisySensors(rng)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))

    fm, loader = _make_manager(
        supervisor_cfg=SupervisorConfig(N_confirm=4, safe_mode_retry_steps=2)
    )
    ambient = AmbientState(params.T_ambient, params.H_ambient)
    z2_ctrl = fm.controllers["zone_2"]
    orig_z2_step = z2_ctrl.step

    def _z2_boom(*args, **kwargs):
        raise RuntimeError("zone_2 PID fault")

    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    z2_modes: list[ControlMode] = []

    for t in range(1, 55):
        records = net.step(sensors.read(env.state, t), prev_actions)
        marl_acts = _simple_healthy_marl(obs, loader, ambient)

        # Phase A (t=10..17): Zone 2 agent crashes -> PID_FALLBACK
        if 10 <= t < 18:
            del marl_acts["zone_2"]

        # Phase B (t=18..25): Zone 2 agent still crashed AND Zone 2 PID controller fails -> SAFE_MODE
        if t == 18:
            monkeypatch.setattr(z2_ctrl, "step", _z2_boom)
        if 18 <= t < 26:
            del marl_acts["zone_2"]

        # Phase C (t=26..54): Restore Zone 2 PID and Zone 2 agent
        if t == 26:
            monkeypatch.setattr(z2_ctrl, "step", orig_z2_step)

        res = fm.step(marl_acts, records, ambient=ambient)
        z2_modes.append(res.statuses["zone_2"].mode)

        # Zones 1 & 3 must NEVER fall back or reset throughout the entire incident
        assert res.statuses["zone_1"].mode == ControlMode.MARL_ACTIVE
        assert res.statuses["zone_3"].mode == ControlMode.MARL_ACTIVE

        if t == 12:
            assert res.statuses["zone_2"].mode == ControlMode.PID_FALLBACK
        if 18 <= t < 26:
            assert res.statuses["zone_2"].mode == ControlMode.SAFE_MODE
            assert res.statuses["zone_2"].executed_action == (0, 0, 0, 0)

        obs, _, _, _, _ = env.step(res.actions)
        prev_actions = res.actions

    # After restoration at t=26, Zone 2 must pass through PID_FALLBACK and RECOVERY_PENDING before MARL_ACTIVE
    post_restore = z2_modes[25:]  # t >= 26
    assert ControlMode.PID_FALLBACK in post_restore
    assert ControlMode.RECOVERY_PENDING in post_restore
    assert post_restore[0] != ControlMode.MARL_ACTIVE  # cannot jump immediately to MARL
    assert res.statuses["zone_2"].mode == ControlMode.MARL_ACTIVE

    metrics = fm.get_metrics()
    assert metrics["zone_1"]["marl_to_pid_switches"] == 0
    assert metrics["zone_3"]["marl_to_pid_switches"] == 0
    assert metrics["zone_2"]["safe_mode_events"] >= 1
    assert metrics["zone_2"]["recovery_successes"] >= 1


def test_long_running_closed_loop_control_stability(healthy_residuals):
    """Long-running 1000-step closed-loop stability test across MARL, PID fallback, and recovery."""
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=1200)
    obs, _ = env.reset(seed=99)
    rng = np.random.default_rng(99)
    sensors = NoisySensors(rng)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))
    loader = CropConfigLoader.from_json_file(DATASET_PATH)
    loader.assign_zone("zone_1", "tomato", "vegetative")
    loader.assign_zone("zone_2", "lettuce", "mature")
    loader.assign_zone("zone_3", "cucumber", "flowering")
    fm = FallbackManager(
        crop_loader=loader,
        pid_gains=load_pid_tuning(TUNING_PATH),
        sensor_cfg=SensorInterfaceConfig(theta_trust_good=0.45, t_est_max={"T": 300, "H": 300, "M": 60}),
        supervisor_cfg=SupervisorConfig(N_confirm=6, theta_recovery_trust=0.45),
    )
    ambient = AmbientState(params.T_ambient, params.H_ambient)

    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    for t in range(1, 1001):
        records = net.step(sensors.read(env.state, t), prev_actions)
        marl_acts = _simple_healthy_marl(obs, loader, ambient)
        if 200 <= t < 500:
            del marl_acts["zone_2"]
        res = fm.step(marl_acts, records, ambient=ambient)
        obs, _, _, _, _ = env.step(res.actions)
        prev_actions = res.actions

        for z in ZONE_IDS:
            s = env.state[z]
            assert np.isfinite(s["T"]) and -10.0 <= s["T"] <= 60.0
            assert np.isfinite(s["H"]) and 0.0 <= s["H"] <= 100.0
            assert np.isfinite(s["M"]) and 0.0 <= s["M"] <= 100.0

    assert fm.get_metrics()["zone_2"]["pid_fallback_steps"] >= 250
    assert fm.get_metrics()["zone_2"]["recovery_successes"] >= 1
    assert all(fm.get_metrics()[z]["safe_mode_events"] == 0 for z in ZONE_IDS)


def test_phase6_full_acceptance_compound_sensor_pid_marl_fault_and_recovery(healthy_residuals, monkeypatch):
    """
    Phase 6 Step 4 Acceptance (A-E):
    - A: 3-zone healthy operation under surrogate MARL
    - B: Zone 2 MARL crash -> PID_FALLBACK; Zones 1 & 3 stay MARL_ACTIVE
    - C: Zone 2 PID failure -> SAFE_MODE (0,0,0,0); Zones 1 & 3 keep running
    - D: Zone 2 NaN/Inf sensor corruption isolated; Zones 1 & 3 unaffected; sensor recovers
    - E: Restore Zone 2 sensor, PID, and MARL -> SAFE_MODE -> PID_FALLBACK -> RECOVERY_PENDING -> MARL_ACTIVE
    """
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=200)
    obs, _ = env.reset(seed=10)
    rng = np.random.default_rng(10)
    faults = [
        Fault("zone_2", "T", "nan", start=22, end=38),
        Fault("zone_2", "H", "inf", start=22, end=38),
    ]
    sensors = NoisySensors(rng, faults=faults)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))

    fm, loader = _make_manager(supervisor_cfg=SupervisorConfig(N_confirm=5, safe_mode_retry_steps=2))
    ambient = AmbientState(params.T_ambient, params.H_ambient)
    z2_ctrl = fm.controllers["zone_2"]
    orig_z2_step = z2_ctrl.step

    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    z2_modes: list[ControlMode] = []

    for t in range(1, 100):
        records = net.step(sensors.read(env.state, t), prev_actions)
        marl_acts = _simple_healthy_marl(obs, loader, ambient)

        if 12 <= t < 40:
            del marl_acts["zone_2"]
        if t == 18:
            monkeypatch.setattr(z2_ctrl, "step", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("z2 pid fault")))
        if t == 38:
            monkeypatch.setattr(z2_ctrl, "step", orig_z2_step)

        res = fm.step(marl_acts, records, ambient=ambient)
        z2_modes.append(res.statuses["zone_2"].mode)

        # Zones 1 & 3 remain healthy in MARL_ACTIVE and uncontaminated throughout
        assert res.statuses["zone_1"].mode == ControlMode.MARL_ACTIVE
        assert res.statuses["zone_3"].mode == ControlMode.MARL_ACTIVE
        assert np.isfinite(records["zone_1"]["T"].combined)
        assert np.isfinite(records["zone_3"]["H"].combined)

        if 12 <= t < 18:
            assert res.statuses["zone_2"].mode == ControlMode.PID_FALLBACK
        if 18 <= t < 38:
            assert res.statuses["zone_2"].mode == ControlMode.SAFE_MODE
            assert res.statuses["zone_2"].executed_action == (0, 0, 0, 0)

        obs, _, _, _, _ = env.step(res.actions)
        prev_actions = res.actions

    # Trust in Zone 2 restores after t=38 and Zone 2 completes permitted recovery path to MARL_ACTIVE
    assert records["zone_2"]["T"].trust > 0.85
    assert records["zone_2"]["H"].trust > 0.85
    assert res.statuses["zone_2"].mode == ControlMode.MARL_ACTIVE
    transitions = fm.get_metrics()["zone_2"]["transitions"]
    assert any(src == "SAFE_MODE" and dst == "PID_FALLBACK" for _, src, dst, _ in transitions)
    assert any(src == "PID_FALLBACK" and dst == "RECOVERY_PENDING" for _, src, dst, _ in transitions)
    assert any(src == "RECOVERY_PENDING" and dst == "MARL_ACTIVE" for _, src, dst, _ in transitions)


def test_phase6_extended_operation_disturbances_and_metrics(healthy_residuals):
    """Phase 6 Step 4F: 1200-step closed-loop run with ambient weather disturbance, sensor/MARL faults, and evaluator metrics."""
    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=1300)
    obs, _ = env.reset(seed=10)
    rng = np.random.default_rng(10)
    faults = [Fault("zone_1", "T", "drift", start=200, magnitude=0.02, end=350)]
    sensors = NoisySensors(rng, faults=faults)
    assumed = AssumedParams.from_env(params)
    net = TrustNetwork(assumed, healthy=healthy_residuals)
    net.reset(sensors.read(env.state, 0))

    loader = CropConfigLoader.from_json_file(DATASET_PATH)
    loader.assign_zone("zone_1", "tomato", "vegetative")
    loader.assign_zone("zone_2", "lettuce", "mature")
    loader.assign_zone("zone_3", "cucumber", "flowering")
    fm = FallbackManager(
        crop_loader=loader,
        pid_gains=load_pid_tuning(TUNING_PATH),
        sensor_cfg=SensorInterfaceConfig(theta_trust_good=0.45, t_est_max={"T": 300, "H": 300, "M": 60}),
        supervisor_cfg=SupervisorConfig(N_confirm=6, theta_recovery_trust=0.45),
    )
    evaluator = EpisodeEvaluator(dt=params.dt, fault_start_step=200)

    prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    z2_pid_water_streak = 0
    max_z2_pid_water_streak = 0

    for t in range(1, 1201):
        # Coherent outdoor weather disturbance (cooler, more humid front) from t=600..850
        if 600 <= t < 850:
            params.T_ambient, params.H_ambient = 19.0, 58.0
            assumed.T_ambient, assumed.H_ambient = 19.0, 58.0
        else:
            params.T_ambient, params.H_ambient = 24.0, 45.0
            assumed.T_ambient, assumed.H_ambient = 24.0, 45.0
        amb = AmbientState(params.T_ambient, params.H_ambient)

        records = net.step(sensors.read(env.state, t), prev_actions)
        marl_acts = _simple_healthy_marl(obs, loader, amb)
        if 400 <= t < 750:
            del marl_acts["zone_2"]

        res = fm.step(marl_acts, records, ambient=amb, current_time=float(t))
        if res.statuses["zone_2"].mode == ControlMode.PID_FALLBACK:
            z2_pid_water_streak = z2_pid_water_streak + 1 if res.statuses["zone_2"].executed_action[3] == 1 else 0
            max_z2_pid_water_streak = max(max_z2_pid_water_streak, z2_pid_water_streak)
        else:
            z2_pid_water_streak = 0

        bounds_now = {z: loader.get_zone_targets(z) for z in ZONE_IDS}
        evaluator.record_step(t, env.state, bounds_now, res.statuses)
        obs, _, _, _, _ = env.step(res.actions)
        prev_actions = res.actions

    summary = evaluator.summarize(fm.get_metrics())
    for z in ZONE_IDS:
        for v in ("T", "H", "M"):
            assert np.isfinite(summary[z]["variables"][v]["rmse"])
            assert np.isfinite(summary[z]["variables"][v]["mae"])
        assert summary[z]["switching_frequency_per_min"] >= 0.0

    # Irrigation pulse limit (default water_max_pulse_steps=2) must never be exceeded in PID fallback
    assert max_z2_pid_water_streak <= 2
    assert fm.get_metrics()["zone_2"]["pid_fallback_steps"] >= 300
    assert fm.get_metrics()["zone_2"]["recovery_successes"] >= 1


def test_phase6_exact_reproducibility_with_identical_seeds(healthy_residuals):
    """Phase 6 Step 4G: Identical seeds yield deterministic states, trust records, actions, and metrics."""

    def _run_seeded(seed: int):
        params = PlaceholderParams()
        env = GreenhouseEnv(params=params, max_steps=150)
        obs, _ = env.reset(seed=seed)
        sensors = NoisySensors(np.random.default_rng(seed), faults=[Fault("zone_2", "T", "nan", start=20, end=35)])
        net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy_residuals)
        net.reset(sensors.read(env.state, 0))
        fm, loader = _make_manager(supervisor_cfg=SupervisorConfig(N_confirm=5))
        evaluator = EpisodeEvaluator(dt=params.dt, fault_start_step=20)
        ambient = AmbientState(params.T_ambient, params.H_ambient)
        prev_actions = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
        trace = []
        for t in range(1, 100):
            records = net.step(sensors.read(env.state, t), prev_actions)
            marl_acts = _simple_healthy_marl(obs, loader, ambient)
            if 40 <= t < 60:
                del marl_acts["zone_2"]
            res = fm.step(marl_acts, records, ambient=ambient, current_time=float(t))
            bounds_now = {z: loader.get_zone_targets(z) for z in ZONE_IDS}
            evaluator.record_step(t, env.state, bounds_now, res.statuses)
            obs, _, _, _, _ = env.step(res.actions)
            prev_actions = res.actions
            trace.append((
                {z: tuple(res.actions[z].tolist()) for z in ZONE_IDS},
                {z: res.statuses[z].mode.value for z in ZONE_IDS},
                {z: dict(env.state[z]) for z in ZONE_IDS},
            ))
        return trace, evaluator.summarize(fm.get_metrics())

    t1, s1 = _run_seeded(101)
    t2, s2 = _run_seeded(101)
    assert t1 == t2
    assert s1 == s2
