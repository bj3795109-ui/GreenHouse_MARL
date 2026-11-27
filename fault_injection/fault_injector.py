import random

from .fault_types import (
    FaultCategory,
    SensorFaultType,
    ActuatorFaultType
)
from .fault_config import FaultConfig, check_step


class FaultInjector:
    """
    Injects configured faults into greenhouse sensors and actuators.

    Sensor faults:
    - Bias
    - Drift
    - Stuck
    - Noise
    - Spike
    - Dropout
    - Intermittent

    Actuator faults:
    - Stuck ON
    - Stuck OFF
    - Failure to activate
    - Failure to deactivate

    Supports multiple simultaneous faults across different
    zones and components.
    """

    def __init__(self, seed=None):
        # One independent RNG stream per (fault, zone, target): a fault's random
        # realization must not depend on which other faults exist or get queried.
        self._base_seed = (
            seed if seed is not None else random.SystemRandom().getrandbits(64)
        )

        # Stores frozen sensor readings independently.
        # Key: (fault_id, zone_id, target)
        self.stuck_values: dict[tuple, float] = {}

    def _rng_for(self, key: tuple, step: int) -> random.Random:
        # Stateless: the draw depends only on (seed, fault, zone, target, step), so
        # skipped steps, repeated queries, or a different policy's call pattern
        # cannot shift a fault's realization.
        return random.Random(f"{self._base_seed}:{key}:{step}")

    # --------------------------------------------------
    # SENSOR FAULT INJECTION
    # --------------------------------------------------

    def apply_sensor_fault(
        self,
        reading: float,
        fault: FaultConfig,
        current_step: int,
        zone_id: int,
        target: str
    ) -> float | None:
        """
        Apply a single sensor fault to a reading.
        """

        if fault.category != FaultCategory.SENSOR:
            raise ValueError("Expected a sensor fault")

        if (
            fault.zone_id is not None
            and fault.zone_id != zone_id
        ) or fault.target != target:
            return reading

        stuck_key = (fault.fault_id, zone_id, target)

        if not fault.is_active_at(current_step):
            if current_step >= fault.start_step:   # expired, not merely pending
                self.stuck_values.pop(stuck_key, None)
            return reading

        # Bias
        if fault.fault_type == SensorFaultType.BIAS.value:
            return reading + fault.magnitude

        # Drift
        elif fault.fault_type == SensorFaultType.DRIFT.value:
            elapsed_steps = current_step - fault.start_step
            return reading + (fault.magnitude * elapsed_steps)

        # Stuck sensor
        elif fault.fault_type == SensorFaultType.STUCK.value:
            if stuck_key not in self.stuck_values:
                self.stuck_values[stuck_key] = reading

            return self.stuck_values[stuck_key]

        # Random Gaussian noise
        elif fault.fault_type == SensorFaultType.NOISE.value:
            noise = self._rng_for(stuck_key, current_step).gauss(0, fault.magnitude)
            return reading + noise

        # Sudden abnormal measurement
        elif fault.fault_type == SensorFaultType.SPIKE.value:
            # Transient impulse on the activation step only. A sustained offset is BIAS.
            if current_step == fault.start_step:
                return reading + fault.magnitude
            return reading

        # Missing sensor reading
        elif fault.fault_type == SensorFaultType.DROPOUT.value:
            return None

        # Probabilistic intermittent fault
        elif fault.fault_type == SensorFaultType.INTERMITTENT.value:
            if self._rng_for(stuck_key, current_step).random() < fault.probability:
                return reading + fault.magnitude

            return reading

        return reading

    def apply_sensor_faults(
        self,
        reading: float,
        faults: list[FaultConfig],
        current_step: int,
        zone_id: int,
        target: str
    ) -> float | None:
        """
        Apply all relevant scheduled sensor faults.

        Overlapping active faults are composed in list order (each fault
        transforms the output of the previous one); a dropout short-circuits
        the chain and returns None.

        The result is NOT clipped to physical sensor limits. Saturation applies to
        every reading (faulted or not), so it belongs to the wrapper's sensor model,
        applied after fault injection.
        """

        current_step = check_step(current_step)

        matching_faults = []

        for fault in faults:

            if fault.category != FaultCategory.SENSOR:
                continue

            if (
                fault.zone_id is not None
                and fault.zone_id != zone_id
            ):
                continue

            if fault.target != target:
                continue

            if not fault.is_active_at(current_step):
                if (current_step >= fault.start_step
                        and fault.fault_type == SensorFaultType.STUCK.value):
                    self.stuck_values.pop((fault.fault_id, zone_id, target), None)
                continue

            matching_faults.append(fault)

        value = reading
        for fault in matching_faults:
            if value is None:
                break
            value = self.apply_sensor_fault(
                value, fault, current_step, zone_id, target
            )
        return value

    # --------------------------------------------------
    # ACTUATOR FAULT INJECTION
    # --------------------------------------------------

    def apply_actuator_faults(
        self,
        requested_action: int,
        previous_action: int,
        faults: list[FaultConfig],
        current_step: int,
        zone_id: int,
        target: str
    ) -> int:
        """
        Apply an active actuator fault.

        requested_action:
            Controller-requested binary action.

        previous_action:
            Previous actual actuator state.
        """

        if requested_action not in (0, 1):
            raise ValueError("requested_action must be 0 or 1")

        if previous_action not in (0, 1):
            raise ValueError("previous_action must be 0 or 1")

        valid_targets = (
            "heater",
            "fan",
            "mister",
            "irrigation_pump"
        )

        if target not in valid_targets:
            raise ValueError("Invalid actuator target")

        current_step = check_step(current_step)

        matching_faults = []

        for fault in faults:

            if fault.category != FaultCategory.ACTUATOR:
                continue

            if (
                fault.zone_id is not None
                and fault.zone_id != zone_id
            ):
                continue

            if fault.target != target:
                continue

            if not fault.is_active_at(current_step):
                continue

            matching_faults.append(fault)

        action = requested_action
        for fault in matching_faults:
            action = self._apply_one_actuator_fault(
                fault, action, previous_action
            )
        return action

    @staticmethod
    def _apply_one_actuator_fault(
        fault: FaultConfig, requested_action: int, previous_action: int
    ) -> int:
        ft = fault.fault_type

        if ft == ActuatorFaultType.STUCK_ON.value:
            return 1

        if ft == ActuatorFaultType.STUCK_OFF.value:
            return 0

        # Cannot switch ON from OFF; an actuator that is already running still
        # follows the request (distinguishes this from STUCK_OFF).
        if ft == ActuatorFaultType.FAILURE_TO_ACTIVATE.value:
            if requested_action == 1 and previous_action == 0:
                return 0
            return requested_action

        # Cannot switch OFF from ON; an idle actuator still follows the request.
        if ft == ActuatorFaultType.FAILURE_TO_DEACTIVATE.value:
            if requested_action == 0 and previous_action == 1:
                return 1
            return requested_action

        return requested_action

    # --------------------------------------------------
    # FAULT STATE CLEANUP
    # --------------------------------------------------

    def reset_fault(self, fault_id: str):
        """
        Remove stored state associated with a fault.

        Should be called when a fault is manually
        deactivated or automatically recovered.
        """

        matching_keys = [
            key
            for key in self.stuck_values
            if key[0] == fault_id
        ]

        for key in matching_keys:
            del self.stuck_values[key]