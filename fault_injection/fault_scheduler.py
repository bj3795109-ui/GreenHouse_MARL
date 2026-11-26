from .fault_config import FaultConfig


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

    def __init__(self, faults: list[FaultConfig]):

        self.faults = list(faults)

        self.deactivated_faults = set()

        # Record the simulation step of manual deactivation.
        self.deactivation_steps = {}

        fault_ids = [fault.fault_id for fault in self.faults]

        if len(fault_ids) != len(set(fault_ids)):
            raise ValueError("Duplicate fault IDs are not allowed")

    def get_active_faults(
        self,
        current_step: int
    ) -> list[FaultConfig]:

        if current_step < 0:
            raise ValueError("current_step cannot be negative")

        active_faults = []

        for fault in self.faults:

            if fault.fault_id in self.deactivated_faults:
                continue

            if current_step < fault.start_step:
                continue

            if fault.auto_recover and fault.duration is not None:

                expiration_step = (
                    fault.start_step + fault.duration
                )

                if current_step >= expiration_step:
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

        if current_step is not None and current_step < 0:
            raise ValueError("current_step cannot be negative")

        self.deactivated_faults.add(fault_id)

        self.deactivation_steps[fault_id] = current_step

    def get_scheduled_faults(self) -> list[FaultConfig]:

        return list(self.faults)