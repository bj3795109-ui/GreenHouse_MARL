"""
Scenario-level validation for a list of FaultConfig.

FaultConfig validates one fault in isolation. Some problems only exist across faults or
across the wrapper's timeline, and would otherwise surface as a crash mid-episode or,
worse, as silently wrong results. Call validate_scenario() once, when a scenario is loaded.
"""

import math
from typing import Iterable

from .fault_config import FaultConfig
from .fault_types import FaultCategory


class ScenarioError(ValueError):
    """Carries every problem found in the scenario, not just the first."""


# Faults whose result depends on composition order. Overlapping one of these with a
# different fault type on the same sensor/actuator is ambiguous, so it is rejected.
_ORDER_SENSITIVE = {
    FaultCategory.SENSOR: {"stuck"},
    FaultCategory.ACTUATOR: {"stuck_on", "stuck_off"},
}


def _span(f: FaultConfig):
    end = f.end_step
    return f.start_step, (math.inf if end is None else end)


def _same_slot(a: FaultConfig, b: FaultConfig) -> bool:
    if a.category != b.category or a.target != b.target:
        return False
    return a.zone_id is None or b.zone_id is None or a.zone_id == b.zone_id


def _overlap(a: FaultConfig, b: FaultConfig) -> bool:
    (s1, e1), (s2, e2) = _span(a), _span(b)
    return s1 < e2 and s2 < e1


def validate_scenario(faults: Iterable[FaultConfig], min_start_step: int = 1) -> None:
    """
    Raise ScenarioError listing every problem. Checks:
      * unique fault_id (lifecycle and logging are keyed by it)
      * start_step >= min_start_step (default 1: step 0 is the initial observation, and a
        faulted first reading cannot be detected or even anchored by the TrustNetwork)
      * only implemented categories (others would be silently ignored)
      * no order-dependent overlaps (see _ORDER_SENSITIVE)
    """
    faults = list(faults)
    problems: list[str] = []
    seen: set[str] = set()

    for f in faults:
        if f.fault_id in seen:
            problems.append(f"duplicate fault_id '{f.fault_id}'")
        seen.add(f.fault_id)

        if f.category not in (FaultCategory.SENSOR, FaultCategory.ACTUATOR):
            problems.append(
                f"'{f.fault_id}': category '{getattr(f.category, 'value', f.category)}' "
                "has no implementation and would be silently ignored"
            )
        if f.start_step < min_start_step:
            problems.append(
                f"'{f.fault_id}': start_step {f.start_step} < {min_start_step}"
            )

    for i, a in enumerate(faults):
        for b in faults[i + 1:]:
            if not (_same_slot(a, b) and _overlap(a, b)):
                continue
            sensitive = _ORDER_SENSITIVE.get(a.category, set())
            if a.fault_type != b.fault_type and (
                a.fault_type in sensitive or b.fault_type in sensitive
            ):
                problems.append(
                    f"'{a.fault_id}' ({a.fault_type}) and '{b.fault_id}' ({b.fault_type}) "
                    f"overlap on the same {a.category.value} with order-dependent results; "
                    "separate them in time or drop one"
                )

    if problems:
        raise ScenarioError("invalid fault scenario:\n  - " + "\n  - ".join(problems))
