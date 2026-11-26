import pytest

from fault_injection.fault_config import FaultConfig
from fault_injection.fault_logger import FaultLogger
from fault_injection.fault_types import FaultCategory


def create_fault():
    return FaultConfig(
        fault_id="sensor_bias_1",
        category=FaultCategory.SENSOR,
        fault_type="bias",
        zone_id=0,
        target="temperature",
        start_step=10,
        duration=20,
        magnitude=5.0
    )


def test_log_activation():
    logger = FaultLogger()
    fault = create_fault()

    logger.log_activation(fault, 10)

    events = logger.get_events()

    assert len(events) == 1
    assert events[0]["fault_id"] == "sensor_bias_1"
    assert events[0]["event"] == "activated"
    assert events[0]["step"] == 10


def test_log_deactivation():
    logger = FaultLogger()
    fault = create_fault()

    logger.log_deactivation(
        fault,
        current_step=30,
        reason="automatic_recovery"
    )

    events = logger.get_events()

    assert len(events) == 1
    assert events[0]["event"] == "deactivated"
    assert events[0]["reason"] == "automatic_recovery"


def test_event_contains_fault_metadata():
    logger = FaultLogger()
    fault = create_fault()

    logger.log_activation(fault, 10)

    event = logger.get_events()[0]

    assert event["category"] == "sensor"
    assert event["fault_type"] == "bias"
    assert event["zone_id"] == 0
    assert event["target"] == "temperature"


def test_activation_and_deactivation_timeline():
    logger = FaultLogger()
    fault = create_fault()

    logger.log_activation(fault, 10)
    logger.log_deactivation(fault, 30, "automatic_recovery")

    events = logger.get_events()

    assert len(events) == 2
    assert events[0]["step"] == 10
    assert events[1]["step"] == 30


def test_multiple_fault_events():
    logger = FaultLogger()

    fault1 = create_fault()
    fault2 = create_fault()
    fault2.fault_id = "sensor_noise_1"
    fault2.fault_type = "noise"

    logger.log_activation(fault1, 10)
    logger.log_activation(fault2, 15)

    assert len(logger.get_events()) == 2


def test_negative_timestep_rejected():
    logger = FaultLogger()
    fault = create_fault()

    with pytest.raises(ValueError):
        logger.log_activation(fault, -1)


def test_clear_events():
    logger = FaultLogger()
    fault = create_fault()

    logger.log_activation(fault, 10)
    logger.clear()

    assert logger.get_events() == []


def test_get_events_returns_independent_records():
    logger = FaultLogger()
    fault = create_fault()

    logger.log_activation(fault, 10)

    events = logger.get_events()
    events.clear()

    assert len(logger.get_events()) == 1