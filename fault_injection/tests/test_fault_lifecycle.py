from fault_injection.fault_config import FaultConfig
from fault_injection.fault_types import FaultCategory
from fault_injection.fault_scheduler import FaultScheduler
from fault_injection.fault_injector import FaultInjector
from fault_injection.fault_logger import FaultLogger
from fault_injection.fault_lifecycle import FaultLifecycleManager


def create_sensor_fault(
    fault_id="test_fault",
    start_step=0,
    duration=None,
    auto_recover=True
):
    return FaultConfig(
        fault_id=fault_id,
        category=FaultCategory.SENSOR,
        fault_type="stuck",
        target="temperature",
        zone_id=0,
        start_step=start_step,
        duration=duration,
        auto_recover=auto_recover
    )


def create_manager(faults):
    scheduler = FaultScheduler(faults)
    injector = FaultInjector(seed=42)
    logger = FaultLogger()

    manager = FaultLifecycleManager(
        scheduler,
        injector,
        logger
    )

    return manager, scheduler, injector, logger


def test_fault_activation_is_logged():
    fault = create_sensor_fault(start_step=2)

    manager, _, _, logger = create_manager([fault])

    manager.update(1)
    manager.update(2)

    events = logger.get_events()

    assert len(events) == 1
    assert events[0]["event"] == "activated"
    assert events[0]["fault_id"] == "test_fault"
    assert events[0]["step"] == 2


def test_automatic_recovery_is_logged():
    fault = create_sensor_fault(
        start_step=1,
        duration=2,
        auto_recover=True
    )

    manager, _, _, logger = create_manager([fault])

    manager.update(1)
    manager.update(2)
    manager.update(3)

    events = logger.get_events()

    assert len(events) == 2
    assert events[1]["event"] == "deactivated"
    assert events[1]["reason"] == "automatic_recovery"
    assert events[1]["step"] == 3


def test_manual_deactivation_is_logged():
    fault = create_sensor_fault(
        auto_recover=False
    )

    manager, scheduler, _, logger = create_manager([fault])

    manager.update(0)

    scheduler.deactivate_fault("test_fault")

    manager.update(1)

    events = logger.get_events()

    assert len(events) == 2
    assert events[1]["event"] == "deactivated"
    assert events[1]["reason"] == "manual"


def test_deactivation_clears_stuck_values():
    fault = create_sensor_fault()

    manager, scheduler, injector, _ = create_manager([fault])

    injector.stuck_values[("test_fault", 0, "temperature")] = 25.0

    manager.update(0)

    scheduler.deactivate_fault("test_fault")

    manager.update(1)

    assert ("test_fault", 0, "temperature") not in injector.stuck_values


def test_repeated_update_does_not_duplicate_events():
    fault = create_sensor_fault()

    manager, _, _, logger = create_manager([fault])

    manager.update(0)
    manager.update(0)
    manager.update(0)

    events = logger.get_events()

    assert len(events) == 1
    assert events[0]["event"] == "activated"


def test_inactive_fault_does_not_generate_events():
    fault = create_sensor_fault(start_step=5)

    manager, _, _, logger = create_manager([fault])

    manager.update(0)
    manager.update(1)
    manager.update(2)

    assert logger.get_events() == []


def test_current_step_cannot_move_backwards():
    fault = create_sensor_fault()

    manager, _, _, _ = create_manager([fault])

    manager.update(5)

    try:
        manager.update(4)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    
def test_fault_lifecycle_when_entire_fault_is_missed():
    fault = create_sensor_fault(
        start_step=2,
        duration=3,
        auto_recover=True
    )

    manager, _, _, logger = create_manager([fault])

    manager.update(1)
    manager.update(6)

    events = logger.get_events()

    assert len(events) == 2
    assert events[0]["event"] == "activated"
    assert events[0]["step"] == 2

    assert events[1]["event"] == "deactivated"
    assert events[1]["step"] == 5
    assert events[1]["reason"] == "automatic_recovery"
    
    
def test_skipped_timestep_without_auto_recovery():
    fault = create_sensor_fault(
        start_step=2,
        duration=3,
        auto_recover=False
    )

    manager, _, _, logger = create_manager([fault])

    manager.update(1)
    active = manager.update(6)

    events = logger.get_events()

    assert len(events) == 1
    assert events[0]["event"] == "activated"
    assert events[0]["step"] == 2

    assert len(active) == 1
    
def test_manual_deactivation_between_observations():
    fault = create_sensor_fault(start_step=0)

    manager, scheduler, _, logger = create_manager([fault])

    manager.update(1)

    scheduler.deactivate_fault(
        "test_fault",
        current_step=4
    )

    manager.update(6)

    events = logger.get_events()

    assert len(events) == 2
    assert events[1]["event"] == "deactivated"
    assert events[1]["step"] == 4
    assert events[1]["reason"] == "manual"
    
def test_multiple_faults_with_different_expiration_steps():
    fault_a = create_sensor_fault(
        fault_id="fault_a",
        start_step=1,
        duration=3
    )

    fault_b = create_sensor_fault(
        fault_id="fault_b",
        start_step=1,
        duration=6
    )

    manager, _, _, logger = create_manager(
        [fault_a, fault_b]
    )

    manager.update(1)
    active = manager.update(4)

    events = logger.get_events()

    assert len(events) == 3

    assert sum(
        e["event"] == "activated" for e in events
    ) == 2

    assert sum(
        e["event"] == "deactivated" for e in events
    ) == 1

    assert len(active) == 1
    assert active[0].fault_id == "fault_b"
    
def test_manual_deactivation_before_fault_activation():
    fault = create_sensor_fault(start_step=5)

    manager, scheduler, _, logger = create_manager([fault])

    manager.update(1)

    scheduler.deactivate_fault(
        "test_fault",
        current_step=3
    )

    active = manager.update(6)

    assert active == []
    assert logger.get_events() == []
    
def test_manual_deactivation_takes_precedence_over_later_expiration():
    fault = create_sensor_fault(
        start_step=1,
        duration=5,
        auto_recover=True
    )

    manager, scheduler, _, logger = create_manager([fault])

    manager.update(2)

    scheduler.deactivate_fault(
        "test_fault",
        current_step=4
    )

    manager.update(8)

    events = logger.get_events()

    assert len(events) == 2
    assert events[1]["event"] == "deactivated"
    assert events[1]["step"] == 4
    assert events[1]["reason"] == "manual"
    
def test_automatic_recovery_precedes_later_manual_deactivation():
    fault = create_sensor_fault(
        start_step=1,
        duration=5,
        auto_recover=True
    )

    manager, scheduler, _, logger = create_manager([fault])

    manager.update(2)

    scheduler.deactivate_fault(
        "test_fault",
        current_step=7
    )

    manager.update(8)

    events = logger.get_events()

    assert len(events) == 2
    assert events[1]["event"] == "deactivated"
    assert events[1]["step"] == 6
    assert events[1]["reason"] == "automatic_recovery"