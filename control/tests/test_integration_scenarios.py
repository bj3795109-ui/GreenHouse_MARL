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
    return bootstrap_healthy(AssumedParams.from_env(PlaceholderParams()), n_episodes=5, steps=300)


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
