import pytest

from fault_injection.fault_config import FaultConfig
from fault_injection.fault_injector import FaultInjector
from fault_injection.fault_types import FaultCategory


def create_fault(
    fault_type="stuck_on",
    zone_id=0,
    target="heater",
    start_step=5,
    duration=10,
    auto_recover=True
):
    return FaultConfig(
        fault_id="actuator_fault_1",
        category=FaultCategory.ACTUATOR,
        fault_type=fault_type,
        zone_id=zone_id,
        target=target,
        start_step=start_step,
        duration=duration,
        auto_recover=auto_recover
    )


def apply(injector, requested, previous, fault, step=5, zone=0, target="heater"):
    return injector.apply_actuator_faults(
        requested_action=requested,
        previous_action=previous,
        faults=[fault],
        current_step=step,
        zone_id=zone,
        target=target
    )


@pytest.mark.parametrize("requested", [0, 1])
def test_stuck_on(requested):
    injector = FaultInjector(seed=42)
    fault = create_fault("stuck_on")

    assert apply(injector, requested, 0, fault) == 1


@pytest.mark.parametrize("requested", [0, 1])
def test_stuck_off(requested):
    injector = FaultInjector(seed=42)
    fault = create_fault("stuck_off")

    assert apply(injector, requested, 1, fault) == 0


def test_failure_to_activate():
    injector = FaultInjector()
    fault = create_fault("failure_to_activate")

    assert apply(injector, 1, 0, fault) == 0


def test_failure_to_activate_when_already_off():
    injector = FaultInjector()
    fault = create_fault("failure_to_activate")

    assert apply(injector, 0, 0, fault) == 0


def test_failure_to_deactivate():
    injector = FaultInjector()
    fault = create_fault("failure_to_deactivate")

    assert apply(injector, 0, 1, fault) == 1


def test_failure_to_deactivate_when_already_off():
    injector = FaultInjector()
    fault = create_fault("failure_to_deactivate")

    assert apply(injector, 0, 0, fault) == 0


def test_no_fault_returns_requested_action():
    injector = FaultInjector()

    result = injector.apply_actuator_faults(
        requested_action=1,
        previous_action=0,
        faults=[],
        current_step=5,
        zone_id=0,
        target="heater"
    )

    assert result == 1


def test_fault_not_yet_active():
    injector = FaultInjector()
    fault = create_fault(start_step=10)

    assert apply(injector, 1, 0, fault, step=5) == 1


def test_fault_automatically_recovers():
    injector = FaultInjector()
    fault = create_fault(
        fault_type="stuck_on",
        start_step=5,
        duration=3,
        auto_recover=True
    )

    assert apply(injector, 0, 0, fault, step=7) == 1
    assert apply(injector, 0, 0, fault, step=8) == 0


def test_fault_without_auto_recovery():
    injector = FaultInjector()
    fault = create_fault(
        fault_type="stuck_on",
        start_step=5,
        duration=3,
        auto_recover=False
    )

    assert apply(injector, 0, 0, fault, step=20) == 1


def test_zone_independence():
    injector = FaultInjector()
    fault = create_fault(zone_id=0)

    assert apply(injector, 0, 0, fault, zone=1) == 0


def test_overlapping_faults_compose_in_list_order():
    injector = FaultInjector()

    fault1 = create_fault("stuck_on")
    fault2 = create_fault("stuck_off")
    fault2.fault_id = "actuator_fault_2"

    def run(faults):
        return injector.apply_actuator_faults(
            requested_action=0,
            previous_action=0,
            faults=faults,
            current_step=5,
            zone_id=0,
            target="heater"
        )

    assert run([fault1, fault2]) == 0   # last fault in the list has the final say
    assert run([fault2, fault1]) == 1


@pytest.mark.parametrize(
    "requested,previous",
    [
        (2, 0),
        (0, 2),
        (-1, 0),
        (0, -1)
    ]
)
def test_invalid_binary_actions(requested, previous):
    injector = FaultInjector()

    with pytest.raises(ValueError):
        injector.apply_actuator_faults(
            requested_action=requested,
            previous_action=previous,
            faults=[],
            current_step=5,
            zone_id=0,
            target="heater"
        )


def test_invalid_target():
    injector = FaultInjector()

    with pytest.raises(ValueError):
        injector.apply_actuator_faults(
            requested_action=1,
            previous_action=0,
            faults=[],
            current_step=5,
            zone_id=0,
            target="light"
        )


def test_negative_timestep():
    injector = FaultInjector()

    with pytest.raises(ValueError):
        injector.apply_actuator_faults(
            requested_action=1,
            previous_action=0,
            faults=[],
            current_step=-1,
            zone_id=0,
            target="heater"
        )