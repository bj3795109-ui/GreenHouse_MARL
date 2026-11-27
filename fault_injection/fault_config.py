import math
import numbers
from dataclasses import dataclass
from typing import Optional

from .fault_types import (
    FaultCategory,
    SensorFaultType,
    ActuatorFaultType
)


def _as_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _as_finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def check_step(value, name: str = "current_step") -> int:
    """Simulation steps are non-negative integers (numpy ints fine; bool and floats are not)."""
    step = _as_int(value, name)
    if step < 0:
        raise ValueError(f"{name} cannot be negative")
    return step


@dataclass
class FaultConfig:
    """
    Configuration for an individual fault injection event.
    """

    fault_id: str
    category: FaultCategory
    fault_type: str

    zone_id: Optional[int] = None
    target: Optional[str] = None

    start_step: int = 0
    duration: Optional[int] = None

    magnitude: float = 0.0
    auto_recover: bool = True
    probability: float = 0.5

    def __post_init__(self):
        # Type hygiene first: scenarios loaded from JSON/YAML arrive as plain str/int/float.
        if not isinstance(self.fault_id, str):
            raise ValueError("fault_id must be a string")
        if not self.fault_id.strip():
            raise ValueError("fault_id cannot be empty")

        try:
            self.category = FaultCategory(self.category)   # str -> enum; logger needs .value
        except ValueError:
            raise ValueError(f"Unsupported fault category: {self.category!r}") from None

        self.start_step = _as_int(self.start_step, "start_step")
        if self.duration is not None:
            self.duration = _as_int(self.duration, "duration")
        if self.zone_id is not None:
            self.zone_id = _as_int(self.zone_id, "zone_id")
        # Negative magnitudes are legitimate (negative bias/drift); non-finite never is.
        self.magnitude = _as_finite(self.magnitude, "magnitude")
        self.probability = _as_finite(self.probability, "probability")
        if not isinstance(self.auto_recover, bool):
            raise ValueError("auto_recover must be a bool")


        if self.start_step < 0:
            raise ValueError("start_step cannot be negative")

        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be positive")

        if self.zone_id is not None and self.zone_id not in (0, 1, 2):
            raise ValueError("zone_id must be between 0 and 2")

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")

        if self.category == FaultCategory.SENSOR:
            if self.fault_type not in [f.value for f in SensorFaultType]:
                raise ValueError(
                    f"Unsupported sensor fault type: {self.fault_type}"
                )

            if self.target not in (
                "temperature",
                "humidity",
                "soil_moisture"
            ):
                raise ValueError("Invalid sensor target")

            if self.fault_type == SensorFaultType.NOISE.value and self.magnitude < 0.0:
                raise ValueError("noise magnitude is a standard deviation and must be >= 0")

        elif self.category == FaultCategory.ACTUATOR:
            if self.fault_type not in [f.value for f in ActuatorFaultType]:
                raise ValueError(
                    f"Unsupported actuator fault type: {self.fault_type}"
                )

            if self.target not in (
                "heater",
                "fan",
                "mister",
                "irrigation_pump"
            ):
                raise ValueError("Invalid actuator target")

    @property
    def end_step(self) -> Optional[int]:
        """First step at which an auto-recovering fault is no longer active (None = open-ended)."""
        if self.auto_recover and self.duration is not None:
            return self.start_step + self.duration
        return None

    def is_active_at(self, step: int) -> bool:
        """Pure timing predicate. Ignores manual deactivation, which the lifecycle owns."""
        if step < self.start_step:
            return False
        end = self.end_step
        return end is None or step < end