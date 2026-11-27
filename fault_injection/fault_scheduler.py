from .fault_config import FaultConfig, check_step
from .scenario import validate_scenario


class FaultScheduler:
    """
    Schedules and manages fault activation and recovery.

    Responsibilities:
    - Track configured faults.
    - Determine active faults at each simulation step.
    - Handle automatic fault recovery.
    - Support manual fault deactivation.
    - Record manual deactivation timestamps.
    """

    def __init__(self, faults: list[FaultConfig], min_start_step: int = 0):
        """
        Scenario validation is mandatory (a scheduler can no longer be built from a scenario
        with unimplemented categories or order-dependent overlaps). The integration wrapper
        must pass min_start_step=1: step 0 is the initial observation and cannot be faulted.
        """

        self.faults = list(faults)

        self.deactivated_faults = set()

        # Record the simulation step of manual deactivation.
        self.deactivation_steps = {}

        fault_ids = [fault.fault_id for fault in self.faults]

        if len(fault_ids) != len(set(fault_ids)):
            raise ValueError("Duplicate fault IDs are not allowed")

        validate_scenario(self.faults, min_start_step=min_start_step)

    def get_active_faults(
        self,
        current_step: int
    ) -> list[FaultConfig]:

        current_step = check_step(current_step)

        active_faults = []

        for fault in self.faults:

            if fault.fault_id in self.deactivated_faults:
                continue

            if not fault.is_active_at(current_step):
                continue

            active_faults.append(fault)

        return active_faults

    def deactivate_fault(
        self,
        fault_id: str,
        current_step: int | None = None
    ):

        known_fault_ids = {
            fault.fault_id for fault in self.faults
        }

        if fault_id not in known_fault_ids:
            raise ValueError(f"Unknown fault ID: {fault_id}")

        if current_step is not None:
            current_step = check_step(current_step)

        self.deactivated_faults.add(fault_id)

        self.deactivation_steps[fault_id] = current_step

    def get_scheduled_faults(self) -> list[FaultConfig]:

        return list(self.faults)