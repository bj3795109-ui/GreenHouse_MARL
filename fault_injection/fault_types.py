from enum import Enum


class FaultCategory(str, Enum):
    """
    Broad categories of faults supported by the framework.
    """

    SENSOR = "sensor"
    ACTUATOR = "actuator"
    COMMUNICATION = "communication"
    AGENT = "agent"


class SensorFaultType(str, Enum):
    """
    Types of faults that can affect greenhouse sensor readings.
    """

    BIAS = "bias"
    DRIFT = "drift"
    STUCK = "stuck"
    NOISE = "noise"
    SPIKE = "spike"
    DROPOUT = "dropout"
    INTERMITTENT = "intermittent"
    
class ActuatorFaultType(str, Enum):
    STUCK_ON = "stuck_on"
    STUCK_OFF = "stuck_off"
    FAILURE_TO_ACTIVATE = "failure_to_activate"
    FAILURE_TO_DEACTIVATE = "failure_to_deactivate"