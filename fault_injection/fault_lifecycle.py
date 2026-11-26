class FaultLifecycleManager:
    """
    Coordinates fault scheduling, injection cleanup,
    and lifecycle event logging.

    Reconstructs lifecycle transitions even when simulation
    updates skip over activation or deactivation steps.
    """

    def __init__(self, scheduler, injector, logger):

        self.scheduler = scheduler
        self.injector = injector
        self.logger = logger

        self.last_step = -1

        self.logged_activations = set()
        self.logged_deactivations = set()

    def update(self, current_step: int):
        """
        Synchronize fault lifecycle state.

        Returns:
            list[FaultConfig]: Currently active faults.
        """

        if current_step < 0:
            raise ValueError("current_step cannot be negative")

        if current_step < self.last_step:
            raise ValueError(
                "current_step cannot move backwards"
            )

        for fault in self.scheduler.get_scheduled_faults():

            fault_id = fault.fault_id

            # Already processed completely.
            if fault_id in self.logged_deactivations:
                continue

            # Determine whether manual deactivation occurred.
            manually_deactivated = (
                fault_id in self.scheduler.deactivated_faults
            )

            manual_step = None

            if manually_deactivated:
                manual_step = (
                    self.scheduler.deactivation_steps.get(fault_id)
                )

                # Legacy calls without an explicit step.
                if manual_step is None:
                    manual_step = current_step

            # Determine automatic expiration.
            expiration_step = None

            if fault.auto_recover and fault.duration is not None:
                expiration_step = (
                    fault.start_step + fault.duration
                )

            # Determine the actual termination event.
            end_step = None
            reason = None

            if manual_step is not None:
                end_step = manual_step
                reason = "manual"

            if expiration_step is not None:

                if (
                    end_step is None
                    or expiration_step < end_step
                ):
                    end_step = expiration_step
                    reason = "automatic_recovery"

            # A fault manually deactivated before its scheduled
            # start never became active.
            if end_step is not None and end_step < fault.start_step:
                continue

            # Detect activation.
            if (
                fault.start_step <= current_step
                and fault_id not in self.logged_activations
            ):

                self.logger.log_activation(
                    fault=fault,
                    current_step=fault.start_step
                )

                self.logged_activations.add(fault_id)

            # Detect deactivation.
            if (
                end_step is not None
                and end_step <= current_step
                and fault_id in self.logged_activations
            ):

                self.injector.reset_fault(fault_id)

                self.logger.log_deactivation(
                    fault=fault,
                    current_step=end_step,
                    reason=reason
                )

                self.logged_deactivations.add(fault_id)

        self.last_step = current_step

        return self.scheduler.get_active_faults(current_step)