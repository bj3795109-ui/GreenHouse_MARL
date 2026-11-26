import pytest

from fault_injection.fault_config import FaultConfig
from fault_injection.fault_types import FaultCategory


def create_actuator_fault(
    fault_type="stuck_on",
    target="heater"
):
    return FaultConfig(
        fault_id="actuator_fault_1",
        category=FaultCategory.ACTUATOR,
        fault_type=fault_type,
        zone_id=0,
        target=target,
        start_step=10,
        duration=20
    )


@pytest.mark.parametrize(
    "fault_type",
    [
        "stuck_on",
        "stuck_off",
        "failure_to_activate",
        "failure_to_deactivate"
    ]
)
def test_valid_actuator_fault_types(fault_type):
    fault = create_actuator_fault(fault_type=fault_type)

    assert fault.fault_type == fault_type


@pytest.mark.parametrize(
    "target",
    [
        "heater",
        "fan",
        "mister",
        "irrigation_pump"
    ]
)
def test_valid_actuator_targets(target):
    fault = create_actuator_fault(target=target)

    assert fault.target == target


def test_invalid_actuator_fault_type():
    with pytest.raises(ValueError):
        create_actuator_fault(fault_type="invalid_fault")


def test_invalid_actuator_target():
    with pytest.raises(ValueError):
        create_actuator_fault(target="light")


def test_sensor_fault_configuration_still_works():
    fault = FaultConfig(
        fault_id="sensor_1",
        category=FaultCategory.SENSOR,
        fault_type="bias",
        zone_id=0,
        target="temperature"
    )

    assert fault.category == FaultCategory.SENSOR