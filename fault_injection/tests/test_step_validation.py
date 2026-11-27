import numpy as np
import pytest

from fault_injection.fault_config import FaultConfig
from fault_injection.fault_injector import FaultInjector
from fault_injection.fault_lifecycle import FaultLifecycleManager
from fault_injection.fault_logger import FaultLogger
from fault_injection.fault_scheduler import FaultScheduler
from fault_injection.fault_types import FaultCategory

F = FaultConfig("f", FaultCategory.SENSOR, "bias", zone_id=0, target="temperature",
                start_step=2, magnitude=1.0)
A = FaultConfig("a", FaultCategory.ACTUATOR, "stuck_on", zone_id=0, target="fan", start_step=2)

BAD = [True, 1.5, -1, "3", None, np.True_]


def make():
    sched = FaultScheduler([F, A])
    return sched, FaultInjector(seed=1), FaultLifecycleManager(sched, FaultInjector(seed=1), FaultLogger())


@pytest.mark.parametrize("bad", BAD)
def test_every_public_step_argument_is_validated(bad):
    sched, inj, lc = make()
    with pytest.raises(ValueError):
        sched.get_active_faults(bad)
    with pytest.raises(ValueError):
        inj.apply_sensor_faults(20.0, [F], bad, 0, "temperature")
    with pytest.raises(ValueError):
        inj.apply_actuator_faults(0, 0, [A], bad, 0, "fan")
    with pytest.raises(ValueError):
        lc.update(bad)
    if bad is not None:                       # None means "use the current step" for deactivation
        with pytest.raises(ValueError):
            sched.deactivate_fault("f", bad)


def test_numpy_integer_steps_are_fine():
    sched, inj, lc = make()
    assert F in sched.get_active_faults(np.int64(3))
    assert inj.apply_sensor_faults(20.0, [F], np.int32(3), 0, "temperature") == 21.0
    lc.update(np.int64(3))
    sched.deactivate_fault("f")               # None still allowed
