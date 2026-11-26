```python
"""
Naming adapter between the greenhouse environment and fault injector.

Canonical greenhouse vocabulary:
    Zones: zone_1, zone_2, zone_3
    Variables: T, H, M
    Actuators: heat, fan, mist, water

Fault injector vocabulary:
    Zones: 0, 1, 2
    Variables: temperature, humidity, soil_moisture
    Actuators: heater, fan, mister, irrigation_pump

This module is the only place where both naming systems
should be referenced.
"""


# --------------------------------------------------
# ZONE MAPPING
# --------------------------------------------------

_ZONE_TO_INJECTOR = {
    "zone_1": 0,
    "zone_2": 1,
    "zone_3": 2,
}

_ZONE_FROM_INJECTOR = {
    value: key
    for key, value in _ZONE_TO_INJECTOR.items()
}


def to_injector_zone(zone_id: str) -> int:
    """Convert canonical zone ID to injector zone index."""

    try:
        return _ZONE_TO_INJECTOR[zone_id]
    except KeyError:
        raise ValueError(f"Unknown greenhouse zone: {zone_id}")


def to_canonical_zone(zone_index: int) -> str:
    """Convert injector zone index to canonical zone ID."""

    try:
        return _ZONE_FROM_INJECTOR[zone_index]
    except KeyError:
        raise ValueError(f"Unknown injector zone index: {zone_index}")


# --------------------------------------------------
# SENSOR VARIABLE MAPPING
# --------------------------------------------------

_VARIABLE_TO_INJECTOR = {
    "T": "temperature",
    "H": "humidity",
    "M": "soil_moisture",
}

_VARIABLE_FROM_INJECTOR = {
    value: key
    for key, value in _VARIABLE_TO_INJECTOR.items()
}


def to_injector_variable(variable: str) -> str:
    """Convert canonical sensor variable to injector target."""

    try:
        return _VARIABLE_TO_INJECTOR[variable]
    except KeyError:
        raise ValueError(f"Unknown greenhouse variable: {variable}")


def to_canonical_variable(variable: str) -> str:
    """Convert injector sensor target to canonical variable."""

    try:
        return _VARIABLE_FROM_INJECTOR[variable]
    except KeyError:
        raise ValueError(f"Unknown injector variable: {variable}")


# --------------------------------------------------
# ACTUATOR MAPPING
# --------------------------------------------------

_ACTUATOR_TO_INJECTOR = {
    "heat": "heater",
    "fan": "fan",
    "mist": "mister",
    "water": "irrigation_pump",
}

_ACTUATOR_FROM_INJECTOR = {
    value: key
    for key, value in _ACTUATOR_TO_INJECTOR.items()
}


def to_injector_actuator(actuator: str) -> str:
    """Convert canonical actuator name to injector target."""

    try:
        return _ACTUATOR_TO_INJECTOR[actuator]
    except KeyError:
        raise ValueError(f"Unknown greenhouse actuator: {actuator}")


def to_canonical_actuator(actuator: str) -> str:
    """Convert injector actuator target to canonical name."""

    try:
        return _ACTUATOR_FROM_INJECTOR[actuator]
    except KeyError:
        raise ValueError(f"Unknown injector actuator: {actuator}")
```