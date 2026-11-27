import math

import numpy as np
import pytest

from fault_injection.fault_config import FaultConfig
from fault_injection.fault_injector import FaultInjector
from fault_injection.fault_lifecycle import FaultLifecycleManager
from fault_injection.fault_logger import FaultLogger
from fault_injection.fault_scheduler import FaultScheduler
from fault_injection.fault_types import FaultCategory


def fault(**kw):
    base = dict(fault_id="f", category=FaultCategory.SENSOR, fault_type="bias", zone_id=0,
                target="temperature", start_step=3, duration=10, magnitude=1.0)
    base.update(kw)
    return FaultConfig(**base)


def lifecycle(f):
    sched, log = FaultScheduler([f]), FaultLogger()
    return sched, log, FaultLifecycleManager(sched, FaultInjector(seed=1), log)


# ---------------- lifecycle ----------------

def test_fault_deactivated_exactly_at_start_never_activates():
    f = fault()
    sched, log, lc = lifecycle(f)
    for t in range(3):
        lc.update(t)
    sched.deactivate_fault("f", 3)
    for t in range(3, 8):
        assert f not in lc.update(t)
    assert log.get_events() == []


def test_fault_deactivated_before_start_never_logs():
    f = fault()
    sched, log, lc = lifecycle(f)
    sched.deactivate_fault("f", 1)
    for t in range(0, 8):
        lc.update(t)
    assert log.get_events() == []


def test_fault_deactivated_after_start_logs_both_events():
    f = fault()
    sched, log, lc = lifecycle(f)
    for t in range(3, 5):
        lc.update(t)
    sched.deactivate_fault("f", 5)
    for t in range(5, 8):
        lc.update(t)
    assert [(e["event"], e["step"]) for e in log.get_events()] == [("activated", 3), ("deactivated", 5)]
    assert log.get_events()[1]["reason"] == "manual"


# ---------------- FaultConfig ----------------

@pytest.mark.parametrize("kw", [
    dict(magnitude=float("nan")), dict(magnitude=float("inf")), dict(magnitude="1"),
    dict(magnitude=True), dict(probability=float("nan")), dict(probability=True),
    dict(zone_id=True), dict(zone_id=1.0), dict(start_step=True), dict(start_step=2.5),
    dict(duration=True), dict(duration=2.5), dict(fault_id="   "), dict(fault_id=7),
    dict(category="thermal"), dict(auto_recover="yes"), dict(auto_recover=1),
    dict(fault_type="noise", magnitude=-0.1),
])
def test_invalid_configs_rejected(kw):
    with pytest.raises(ValueError):
        fault(**kw)


def test_negative_bias_and_drift_are_legitimate():
    assert fault(magnitude=-2.0).magnitude == -2.0
    assert fault(fault_type="drift", magnitude=-0.05).magnitude == -0.05


def test_plain_string_category_is_coerced_so_logging_works():
    f = fault(category="sensor")
    assert f.category is FaultCategory.SENSOR
    _, log, lc = lifecycle(f)
    for t in range(3, 6):
        lc.update(t)
    assert log.get_events()[0]["category"] == "sensor"


def test_numpy_numbers_accepted_and_normalised_to_python_types():
    f = fault(start_step=np.int64(4), duration=np.int32(2), zone_id=np.int8(1),
              magnitude=np.float32(1.5), probability=np.float64(0.25))
    assert type(f.start_step) is int and type(f.duration) is int and type(f.zone_id) is int
    assert type(f.magnitude) is float and f.probability == 0.25 and math.isfinite(f.magnitude)
