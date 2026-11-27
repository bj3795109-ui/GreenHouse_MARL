from dataclasses import dataclass
from typing import Optional

from .fault_types import (
    FaultCategory,
    SensorFaultType,
    ActuatorFaultType
)


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
        if not self.fault_id:
            raise ValueError("fault_id cannot be empty")

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