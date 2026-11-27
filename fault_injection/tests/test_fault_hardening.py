import pytest

from fault_injection.fault_config import FaultConfig
from fault_injection.fault_injector import FaultInjector
from fault_injection.fault_lifecycle import FaultLifecycleManager
from fault_injection.fault_scheduler import FaultScheduler
from fault_injection.fault_types import FaultCategory


def bias(fid="b", start=3, dur=4, auto=True, mag=1.0):
    return FaultConfig(fid, FaultCategory.SENSOR, "bias", zone_id=0, target="temperature",
                       start_step=start, duration=dur, magnitude=mag, auto_recover=auto)


@pytest.mark.parametrize("start,dur,auto", [(0, 3, True), (3, 4, True), (5, None, True),
                                            (2, 3, False), (7, 1, True)])
def test_scheduler_injector_and_predicate_agree_on_every_step(start, dur, auto):
    f = bias(start=start, dur=dur, auto=auto)
    sched = FaultScheduler([f])
    inj = FaultInjector(seed=1)
    for step in range(30):
        active = f.is_active_at(step)
        assert (f in sched.get_active_faults(step)) == active
        faulted = inj.apply_sensor_faults(20.0, [f], step, 0, "temperature") != 20.0
        assert faulted == active
        acted = inj.apply_actuator_faults(0, 0, [], step, 0, "fan")
        assert acted == 0


def test_lifecycle_expiry_matches_end_step():
    f = bias(start=3, dur=4)
    assert f.end_step == 7
    assert bias(auto=False).end_step is None and bias(dur=None).end_step is None


def test_injector_does_not_clip_physical_limits_wrapper_sensor_model_owns_that():
    drift = FaultConfig("d", FaultCategory.SENSOR, "drift", zone_id=0, target="temperature",
                        start_step=1, magnitude=1.0)
    assert FaultInjector(seed=1).apply_sensor_faults(20.0, [drift], 500, 0, "temperature") == 519.0   # elapsed = 500 - start_step 1
    assert not hasattr(FaultInjector(seed=1), "sensor_ranges")
