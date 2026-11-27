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

# ---------------- boundary guards (added with the adapter hardening) ----------------
import numpy as np

from control.types import ACTUATOR_NAMES, VARS, ZONE_IDS
from fault_injection import adapter
from fault_injection.fault_config import FaultConfig
from fault_injection.fault_types import FaultCategory


def test_vocabulary_matches_control_types():
    assert set(adapter._ZONE_TO_INJECTOR) == set(ZONE_IDS)
    assert set(adapter._VARIABLE_TO_INJECTOR) == set(VARS)
    assert set(adapter._ACTUATOR_TO_INJECTOR) == set(ACTUATOR_NAMES)


def test_every_adapter_output_is_accepted_by_fault_config():
    for z in ZONE_IDS:
        zi = adapter.to_injector_zone(z)
        for v in VARS:
            FaultConfig("s", FaultCategory.SENSOR, "bias", zone_id=zi,
                        target=adapter.to_injector_variable(v), magnitude=1.0)
        for a in ACTUATOR_NAMES:
            FaultConfig("a", FaultCategory.ACTUATOR, "stuck_on", zone_id=zi,
                        target=adapter.to_injector_actuator(a))


def test_action_vector_order_is_heat_fan_mist_water():
    assert ACTUATOR_NAMES == ("heat", "fan", "mist", "water")
    for i, name in enumerate(ACTUATOR_NAMES):
        assert adapter.actuator_name_at(i) == name
        assert adapter.actuator_index_of(name) == i


@pytest.mark.parametrize("bad", [-1, 4, True, 1.0, None, "1"])
def test_action_index_rejects_invalid(bad):
    with pytest.raises(ValueError):
        adapter.actuator_name_at(bad)


def test_actuator_index_rejects_unknown_name():
    with pytest.raises(ValueError):
        adapter.actuator_index_of("heater")      # injector vocabulary, not canonical


def test_numpy_integers_accepted_but_not_numpy_bool():
    assert adapter.to_canonical_zone(np.int64(1)) == "zone_2"
    assert adapter.actuator_name_at(np.int8(3)) == "water"
    with pytest.raises(ValueError):
        adapter.to_canonical_zone(np.True_)
