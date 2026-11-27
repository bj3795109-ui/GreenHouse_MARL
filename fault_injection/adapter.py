"""
Naming adapter between the greenhouse environment and the fault injector.

Canonical greenhouse vocabulary (owned by control.types):
    Zones: zone_1, zone_2, zone_3
    Variables: T, H, M
    Actuators: heat, fan, mist, water      (action-vector order = ACTUATOR_NAMES)

Fault injector vocabulary:
    Zones: 0, 1, 2
    Variables: temperature, humidity, soil_moisture
    Actuators: heater, fan, mister, irrigation_pump

This is the only module that should reference both vocabularies. The maps below are
checked against control.types at import time, so a vocabulary change fails loudly here
instead of silently mis-targeting faults.
"""

import numbers

from control.types import ACTUATOR_NAMES, VARS, ZONE_IDS

_ZONE_TO_INJECTOR = {"zone_1": 0, "zone_2": 1, "zone_3": 2}
_VARIABLE_TO_INJECTOR = {"T": "temperature", "H": "humidity", "M": "soil_moisture"}
_ACTUATOR_TO_INJECTOR = {
    "heat": "heater",
    "fan": "fan",
    "mist": "mister",
    "water": "irrigation_pump",
}

_ZONE_FROM_INJECTOR = {v: k for k, v in _ZONE_TO_INJECTOR.items()}
_VARIABLE_FROM_INJECTOR = {v: k for k, v in _VARIABLE_TO_INJECTOR.items()}
_ACTUATOR_FROM_INJECTOR = {v: k for k, v in _ACTUATOR_TO_INJECTOR.items()}

if (
    set(_ZONE_TO_INJECTOR) != set(ZONE_IDS)
    or set(_VARIABLE_TO_INJECTOR) != set(VARS)
    or set(_ACTUATOR_TO_INJECTOR) != set(ACTUATOR_NAMES)
):
    raise RuntimeError("fault_injection.adapter is out of sync with control.types")


def _lookup(table: dict, key, kind: str):
    if not isinstance(key, str):
        raise ValueError(f"Invalid {kind}: {key!r}")
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"Unknown {kind}: {key!r}") from None


# ---- zones ----------------------------------------------------------------

def to_injector_zone(zone_id: str) -> int:
    """Canonical zone id -> injector zone index."""
    return _lookup(_ZONE_TO_INJECTOR, zone_id, "greenhouse zone")


def to_canonical_zone(zone_index: int) -> str:
    """Injector zone index -> canonical zone id. Accepts numpy integers, never bool."""
    if isinstance(zone_index, bool) or not isinstance(zone_index, numbers.Integral):
        raise ValueError(f"Invalid injector zone index: {zone_index!r}")
    try:
        return _ZONE_FROM_INJECTOR[int(zone_index)]
    except KeyError:
        raise ValueError(f"Unknown injector zone index: {zone_index!r}") from None


# ---- sensor variables -----------------------------------------------------

def to_injector_variable(variable: str) -> str:
    """Canonical sensor variable -> injector target."""
    return _lookup(_VARIABLE_TO_INJECTOR, variable, "greenhouse variable")


def to_canonical_variable(variable: str) -> str:
    """Injector sensor target -> canonical variable."""
    return _lookup(_VARIABLE_FROM_INJECTOR, variable, "injector variable")


# ---- actuators ------------------------------------------------------------

def to_injector_actuator(actuator: str) -> str:
    """Canonical actuator name -> injector target."""
    return _lookup(_ACTUATOR_TO_INJECTOR, actuator, "greenhouse actuator")


def to_canonical_actuator(actuator: str) -> str:
    """Injector actuator target -> canonical name."""
    return _lookup(_ACTUATOR_FROM_INJECTOR, actuator, "injector actuator")


def actuator_name_at(index: int) -> str:
    """Position in the 4-element action vector -> canonical actuator name."""
    if isinstance(index, bool) or not isinstance(index, numbers.Integral):
        raise ValueError(f"Invalid action index: {index!r}")
    if not 0 <= int(index) < len(ACTUATOR_NAMES):
        raise ValueError(f"Action index out of range: {index!r}")
    return ACTUATOR_NAMES[int(index)]


def actuator_index_of(actuator: str) -> int:
    """Canonical actuator name -> position in the 4-element action vector."""
    if not isinstance(actuator, str) or actuator not in ACTUATOR_NAMES:
        raise ValueError(f"Unknown greenhouse actuator: {actuator!r}")
    return ACTUATOR_NAMES.index(actuator)
