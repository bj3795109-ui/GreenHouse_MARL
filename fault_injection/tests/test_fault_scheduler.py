import pytest

from fault_injection.fault_config import FaultConfig
from fault_injection.fault_scheduler import FaultScheduler
from fault_injection.fault_types import FaultCategory


def create_fault(
    fault_id="fault_1",
    start_step=5,
    duration=3,
    auto_recover=True
):
    return FaultConfig(
        fault_id=fault_id,
        category=FaultCategory.SENSOR,
        fault_type="bias",
        target="temperature",
        zone_id=0,
        start_step=start_step,
        duration=duration,
        magnitude=5.0,
        auto_recover=auto_recover
    )


def test_fault_starts_at_configured_step():
    scheduler = FaultScheduler([create_fault()])

    assert scheduler.get_active_faults(4) == []
    assert len(scheduler.get_active_faults(5)) == 1


def test_automatic_fault_recovery():
    scheduler = FaultScheduler([create_fault()])

    assert len(scheduler.get_active_faults(5)) == 1
    assert len(scheduler.get_active_faults(6)) == 1
    assert len(scheduler.get_active_faults(7)) == 1

    assert scheduler.get_active_faults(8) == []


def test_fault_without_auto_recovery():
    fault = create_fault(auto_recover=False)
    scheduler = FaultScheduler([fault])

    assert len(scheduler.get_active_faults(5)) == 1
    assert len(scheduler.get_active_faults(20)) == 1

    scheduler.deactivate_fault("fault_1")

    assert scheduler.get_active_faults(21) == []


def test_fault_without_duration():
    fault = create_fault(duration=None)
    scheduler = FaultScheduler([fault])

    assert len(scheduler.get_active_faults(5)) == 1
    assert len(scheduler.get_active_faults(1000)) == 1

    scheduler.deactivate_fault("fault_1")

    assert scheduler.get_active_faults(1001) == []


def test_manual_deactivation_before_start():
    scheduler = FaultScheduler([create_fault()])

    scheduler.deactivate_fault("fault_1")

    assert scheduler.get_active_faults(5) == []


def test_unknown_fault_deactivation():
    scheduler = FaultScheduler([create_fault()])

    with pytest.raises(ValueError):
        scheduler.deactivate_fault("unknown_fault")


def test_negative_timestep():
    scheduler = FaultScheduler([create_fault()])

    with pytest.raises(ValueError):
        scheduler.get_active_faults(-1)


def test_duplicate_fault_ids():
    fault1 = create_fault(fault_id="duplicate")
    fault2 = create_fault(fault_id="duplicate")

    with pytest.raises(ValueError):
        FaultScheduler([fault1, fault2])


def test_scheduled_faults_returns_copy():
    scheduler = FaultScheduler([create_fault()])

    faults = scheduler.get_scheduled_faults()
    faults.clear()

    assert len(scheduler.get_scheduled_faults()) == 1