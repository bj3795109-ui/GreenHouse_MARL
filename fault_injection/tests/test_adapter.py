import pytest

from fault_injection.adapter import (
    to_injector_zone,
    to_canonical_zone,
    to_injector_variable,
    to_canonical_variable,
    to_injector_actuator,
    to_canonical_actuator,
)


@pytest.mark.parametrize(
    "canonical, injector",
    [
        ("zone_1", 0),
        ("zone_2", 1),
        ("zone_3", 2),
    ],
)
def test_zone_mapping(canonical, injector):
    assert to_injector_zone(canonical) == injector
    assert to_canonical_zone(injector) == canonical


@pytest.mark.parametrize(
    "canonical, injector",
    [
        ("T", "temperature"),
        ("H", "humidity"),
        ("M", "soil_moisture"),
    ],
)
def test_variable_mapping(canonical, injector):
    assert to_injector_variable(canonical) == injector
    assert to_canonical_variable(injector) == canonical


@pytest.mark.parametrize(
    "canonical, injector",
    [
        ("heat", "heater"),
        ("fan", "fan"),
        ("mist", "mister"),
        ("water", "irrigation_pump"),
    ],
)
def test_actuator_mapping(canonical, injector):
    assert to_injector_actuator(canonical) == injector
    assert to_canonical_actuator(injector) == canonical
    
@pytest.mark.parametrize(
    "function, value",
    [
        (to_injector_zone, "zone_4"),
        (to_injector_zone, 1),
        (to_canonical_zone, 3),
        (to_canonical_zone, True),
        (to_canonical_zone, 1.0),
        (to_injector_variable, "X"),
        (to_injector_variable, None),
        (to_canonical_variable, "pressure"),
        (to_canonical_variable, None),
        (to_injector_actuator, "light"),
        (to_injector_actuator, None),
        (to_canonical_actuator, "pump"),
        (to_canonical_actuator, None),
    ],
)
def test_invalid_adapter_inputs(function, value):
    with pytest.raises(ValueError):
        function(value)