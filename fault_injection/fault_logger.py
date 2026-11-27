from dataclasses import asdict, dataclass
from typing import Optional

from .fault_config import FaultConfig


@dataclass
class FaultEvent:
    fault_id: str
    category: str
    fault_type: str
    zone_id: Optional[int]
    target: Optional[str]
    step: int
    event: str
    reason: Optional[str] = None


class FaultLogger:
    """
    Records fault activation and deactivation events
    during greenhouse simulation.
    """

    def __init__(self):
        self.events: list[FaultEvent] = []

    def log_activation(
        self,
        fault: FaultConfig,
        current_step: int
    ):
        self._log_event(
            fault=fault,
            current_step=current_step,
            event="activated"
        )

    def log_deactivation(
        self,
        fault: FaultConfig,
        current_step: int,
        reason: str
    ):
        self._log_event(
            fault=fault,
            current_step=current_step,
            event="deactivated",
            reason=reason
        )

    def _log_event(
        self,
        fault: FaultConfig,
        current_step: int,
        event: str,
        reason: Optional[str] = None
    ):
        if current_step < 0:
            raise ValueError("current_step cannot be negative")

        record = FaultEvent(
            fault_id=fault.fault_id,
            category=fault.category.value,
            fault_type=fault.fault_type,
            zone_id=fault.zone_id,
            target=fault.target,
            step=current_step,
            event=event,
            reason=reason
        )

        self.events.append(record)

    def get_events(self) -> list[dict]:
        return [
            asdict(event)
            for event in self.events
        ]

    def clear(self):
        self.events.clear()