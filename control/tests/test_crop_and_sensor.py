"""Unit tests for CropConfigLoader and SensorInterface (scenarios 7, 8, 9 + NaN safety)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from control.crop_config_loader import CropConfigError, CropConfigLoader, load_pid_tuning
from control.sensor_interface import SensorInterface, SensorInterfaceConfig
from control.types import MeasurementQuality


DATASET_PATH = Path(__file__).resolve().parent.parent / "configs" / "crops_dataset.json"
ZONE_ASSIGNMENTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "zone_assignments.json"
TUNING_PATH = Path(__file__).resolve().parent.parent / "configs" / "pid_tuning.json"


@dataclass
class _MockTrustRecord:
    raw: float
    combined: float
    belief: float
    trust: float
    conf_cz: float | None
    conf_act: float


def test_load_external_crop_dataset_and_per_zone_assignment():
    loader = CropConfigLoader.from_json_file(DATASET_PATH)
    loader.assign_zone("zone_1", "tomato", "vegetative")
    loader.assign_zone("zone_2", "lettuce", "mature")
    loader.assign_zone("zone_3", "cucumber", "flowering")

    t1 = loader.get_zone_targets("zone_1")
    t2 = loader.get_zone_targets("zone_2")
    t3 = loader.get_zone_targets("zone_3")
    assert t1["T"].optimal_min == 21.0
    assert t2["T"].optimal_max == 20.0
    assert t3["T"].optimal_max == 28.0


def test_missing_crop_config_strict_vs_safe_fallback():
    loader_strict = CropConfigLoader.from_json_file(DATASET_PATH, allow_safe_default=False)
    with pytest.raises(CropConfigError):
        loader_strict.get_zone_targets("zone_1")

    loader_safe = CropConfigLoader.from_json_file(DATASET_PATH, allow_safe_default=True)
    bounds = loader_safe.get_zone_targets("zone_1")
    assert bounds["T"].is_fallback is True
    assert bounds["M"].is_fallback is True


def test_growth_stage_change_does_not_alter_pid_gains():
    loader = CropConfigLoader.from_json_file(DATASET_PATH)
    loader.assign_zone("zone_1", "tomato", "seedling")
    gains_before = load_pid_tuning(TUNING_PATH)

    loader.update_zone_stage("zone_1", "flowering")
    targets_after = loader.get_zone_targets("zone_1")
    gains_after = load_pid_tuning(TUNING_PATH)

    assert targets_after["T"].stage == "flowering"
    assert gains_before == gains_after


def test_incompatible_soil_moisture_unit_is_rejected():
    bad_dataset = {
        "schema_version": "1.0",
        "crops": {
            "tomato": {
                "stages": {
                    "vegetative": {
                        "T": {"unit": "degC", "min_acceptable": 16, "optimal_min": 21, "optimal_max": 26, "max_acceptable": 32},
                        "H": {"unit": "RH_percent", "min_acceptable": 45, "optimal_min": 60, "optimal_max": 75, "max_acceptable": 85},
                        "M": {"unit": "VWC_percent", "min_acceptable": 20, "optimal_min": 30, "optimal_max": 40, "max_acceptable": 55},
                    }
                }
            }
        },
    }
    with pytest.raises(CropConfigError, match="Incompatible unit for M"):
        CropConfigLoader(bad_dataset)


def test_nan_raw_does_not_poison_selected_value():
    si = SensorInterface(SensorInterfaceConfig(theta_trust_good=0.6, theta_est_min=0.15))
    rec = _MockTrustRecord(
        raw=float("nan"),
        combined=23.5,
        belief=float("nan"),  # mimics trust*NaN + (1-trust)*combined
        trust=0.2,
        conf_cz=0.8,
        conf_act=0.5,
    )
    m = si.from_trust_record("zone_1", "T", rec)
    assert m.quality == MeasurementQuality.ESTIMATE_ONLY
    assert m.value == pytest.approx(23.5)
    assert np.isfinite(m.value)


def test_low_trust_and_low_confidence_escalates_hold_then_failsafe():
    si = SensorInterface(SensorInterfaceConfig(theta_trust_good=0.6, theta_est_min=0.3, t_hold_max=3))
    rec = _MockTrustRecord(
        raw=25.0,
        combined=24.0,
        belief=24.1,
        trust=0.1,
        conf_cz=0.1,
        conf_act=0.1,
    )
    qualities = [si.from_trust_record("zone_1", "T", rec).quality for _ in range(4)]
    assert qualities == [
        MeasurementQuality.HOLD_TRANSIENT,
        MeasurementQuality.HOLD_TRANSIENT,
        MeasurementQuality.HOLD_TRANSIENT,
        MeasurementQuality.UNRELIABLE_FAILSAFE,
    ]


def test_open_loop_moisture_cap_escalates_when_raw_untrusted():
    si = SensorInterface(SensorInterfaceConfig(theta_trust_good=0.6, theta_est_min=0.15, t_hold_max=2, t_est_max={"T": 100, "H": 100, "M": 2}))
    rec = _MockTrustRecord(raw=90.0, combined=42.0, belief=46.8, trust=0.1, conf_cz=None, conf_act=0.5)
    q1 = si.from_trust_record("zone_1", "M", rec).quality
    q2 = si.from_trust_record("zone_1", "M", rec).quality
    q3 = si.from_trust_record("zone_1", "M", rec).quality
    assert q1 == MeasurementQuality.ESTIMATE_ONLY
    assert q2 == MeasurementQuality.ESTIMATE_ONLY
    assert q3 == MeasurementQuality.HOLD_TRANSIENT


def test_assign_zones_from_json_file():
    loader = CropConfigLoader.from_json_file(DATASET_PATH)
    loader.assign_zones_from_json_file(ZONE_ASSIGNMENTS_PATH)
    assert loader.get_zone_targets("zone_1")["T"].crop_id == "tomato"
    assert loader.get_zone_targets("zone_2")["T"].crop_id == "lettuce"
    assert loader.get_zone_targets("zone_3")["T"].crop_id == "cucumber"


def test_missing_zone_or_variable_in_trust_records_does_not_keyerror():
    si = SensorInterface(SensorInterfaceConfig(t_hold_max=2))
    partial = {"zone_1": {"T": _MockTrustRecord(22.0, 22.0, 22.0, 1.0, 0.8, 0.5)}}
    res = si.from_trust_records(partial)
    assert res["zone_1"]["T"].quality == MeasurementQuality.TRUSTED
    assert res["zone_1"]["H"].quality == MeasurementQuality.HOLD_TRANSIENT
    assert res["zone_2"]["M"].quality == MeasurementQuality.HOLD_TRANSIENT


def test_staleness_detection_rejects_old_or_frozen_timestamps():
    si = SensorInterface(SensorInterfaceConfig(max_staleness_s=5.0, max_identical_timestamps=2))
    raw = {"zone_1": {"T": 23.0, "H": 60.0, "M": 45.0}}
    # Fresh at t=10 with ts=9
    m_fresh = si.from_raw_dict(raw, timestamps={"zone_1": {"T": 9.0}}, current_time=10.0)["zone_1"]["T"]
    assert m_fresh.quality == MeasurementQuality.TRUSTED
    assert not m_fresh.stale
    # Same ts=9 repeated at t=11 (2nd identical ts) -> still allowed; 3rd identical -> stale!
    si.from_raw_dict(raw, timestamps={"zone_1": {"T": 9.0}}, current_time=11.0)
    m_frozen = si.from_raw_dict(raw, timestamps={"zone_1": {"T": 9.0}}, current_time=12.0)["zone_1"]["T"]
    assert m_frozen.stale is True
    assert m_frozen.quality == MeasurementQuality.HOLD_TRANSIENT
    # Old timestamp (current_time - ts > 5.0) -> stale
    m_old = si.from_raw_dict(raw, timestamps={"zone_1": {"T": 10.0}}, current_time=20.0)["zone_1"]["T"]
    assert m_old.stale is True


def test_trust_network_to_sensor_interface_nan_and_recovery_integration():
    """End-to-end check of TrustNetwork -> SensorInterface across NaN fault and recovery."""
    import sys

    tn_dir = str(Path(__file__).resolve().parent.parent.parent / "trust_network")
    env_dir = str(Path(__file__).resolve().parent.parent.parent / "env")
    if tn_dir not in sys.path:
        sys.path.insert(0, tn_dir)
    if env_dir not in sys.path:
        sys.path.insert(0, env_dir)

    from env import GreenhouseEnv, PlaceholderParams
    from harness import Fault, NoisySensors, bootstrap_healthy
    from trust_network import AssumedParams, TrustNetwork, ZONE_IDS

    params = PlaceholderParams()
    healthy = bootstrap_healthy(AssumedParams.from_env(params), n_episodes=3, steps=200)
    env = GreenhouseEnv(params=params, max_steps=250)
    env.reset(seed=21)
    rng = np.random.default_rng(21)
    faults = [Fault("zone_2", "T", "nan", start=20, end=70)]
    sensors = NoisySensors(rng, faults=faults)
    net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy)
    net.reset(sensors.read(env.state, 0))

    si = SensorInterface(SensorInterfaceConfig(theta_trust_good=0.60, theta_est_min=0.15, t_hold_max=3))
    acts = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}

    qualities_z2 = []
    for t in range(1, 140):
        env.step(acts)
        records = net.step(sensors.read(env.state, t), acts)
        meas = si.from_trust_records(records, current_time=float(t))
        qualities_z2.append(meas["zone_2"]["T"].quality)
        assert np.isfinite(meas["zone_2"]["T"].value)
        assert meas["zone_1"]["T"].quality == MeasurementQuality.TRUSTED
        assert meas["zone_3"]["T"].quality == MeasurementQuality.TRUSTED

    # During NaN fault (t=20..69 -> idx 19..68), zone_2 switches immediately to ESTIMATE_ONLY
    assert all(q == MeasurementQuality.ESTIMATE_ONLY for q in qualities_z2[19:69])
    # After recovery, zone_2 returns to TRUSTED once EMA trust climbs back >= theta_trust_good
    assert qualities_z2[-1] == MeasurementQuality.TRUSTED


def test_trust07_all_sensors_unreliable_escalates_to_failsafe_in_sensor_interface():
    import sys

    tn_dir = str(Path(__file__).resolve().parent.parent.parent / "trust_network")
    env_dir = str(Path(__file__).resolve().parent.parent.parent / "env")
    if tn_dir not in sys.path:
        sys.path.insert(0, tn_dir)
    if env_dir not in sys.path:
        sys.path.insert(0, env_dir)

    from env import GreenhouseEnv, PlaceholderParams
    from harness import Fault, NoisySensors
    from trust_network import AssumedParams, TrustNetwork, ZONE_IDS

    params = PlaceholderParams()
    env = GreenhouseEnv(params=params, max_steps=50)
    env.reset(seed=30)
    faults = [
        Fault(z, v, "nan", start=1)
        for z in ZONE_IDS
        for v in ("T", "H", "M")
    ]
    sensors = NoisySensors(np.random.default_rng(30), faults=faults)
    net = TrustNetwork(AssumedParams.from_env(params))
    net.reset(sensors.read(env.state, 0))

    si = SensorInterface(SensorInterfaceConfig(t_hold_max=2, t_est_max={"T": 4, "H": 4, "M": 3}))
    acts = {z: np.array([0, 0, 0, 0], dtype=np.int8) for z in ZONE_IDS}
    for t in range(1, 10):
        env.step(acts)
        records = net.step(sensors.read(env.state, t), acts)
        meas = si.from_trust_records(records, current_time=float(t))

    for z in ZONE_IDS:
        for v in ("T", "H", "M"):
            assert meas[z][v].quality == MeasurementQuality.UNRELIABLE_FAILSAFE
            assert meas[z][v].value is None


def test_from_env_observations_handles_valid_missing_and_nan_vectors():
    si = SensorInterface(SensorInterfaceConfig(t_hold_max=2))
    obs = {
        "zone_1": np.array([22.5, 65.0, 45.0, 23.0, 60.0, 40.0, 24.0, 55.0, 50.0], dtype=np.float32),
        "zone_2": np.array([float("nan"), 60.0, 40.0], dtype=np.float32),
        # zone_3 omitted -> treated as missing
    }
    meas = si.from_env_observations(obs)
    assert meas["zone_1"]["T"].quality == MeasurementQuality.TRUSTED
    assert meas["zone_1"]["T"].value == pytest.approx(22.5)
    assert meas["zone_2"]["T"].quality == MeasurementQuality.HOLD_TRANSIENT
    assert meas["zone_2"]["H"].quality == MeasurementQuality.TRUSTED
    assert meas["zone_3"]["M"].quality == MeasurementQuality.HOLD_TRANSIENT
