"""
Shared types and constants for the per-zone PID fallback control system.

This module has zero dependencies on PettingZoo, MARL policies, or
TrustNetwork internals so every control submodule can remain loosely coupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

ZONE_IDS: tuple[str, ...] = ("zone_1", "zone_2", "zone_3")
VARS: tuple[str, ...] = ("T", "H", "M")
ACTUATOR_NAMES: tuple[str, ...] = ("heat", "fan", "mist", "water")


class MeasurementQuality(str, Enum):
    """Quality classification produced by SensorInterface."""

    TRUSTED = "TRUSTED"
    ESTIMATE_ONLY = "ESTIMATE_ONLY"
    HOLD_TRANSIENT = "HOLD_TRANSIENT"
    UNRELIABLE_FAILSAFE = "UNRELIABLE_FAILSAFE"


class ControlMode(str, Enum):
    """Per-zone supervisor state machine modes."""

    MARL_ACTIVE = "MARL_ACTIVE"
    PID_FALLBACK = "PID_FALLBACK"
    RECOVERY_PENDING = "RECOVERY_PENDING"
    SAFE_MODE = "SAFE_MODE"


@dataclass(frozen=True)
class SensorMeasurement:
    """Normalized per-variable sensor reading consumed by PID and Supervisor."""

    var: str
    value: Optional[float]
    raw: Optional[float]
    combined: Optional[float]
    belief: Optional[float]
    trust: float
    confidence: float
    quality: MeasurementQuality
    valid_raw: bool
    valid_est: bool
    open_loop_steps: int = 0
    hold_steps: int = 0
    timestamp: Optional[float] = None
    stale: bool = False


@dataclass(frozen=True)
class AmbientState:
    """Ambient conditions used for ventilation effectiveness evaluation."""

    T_ambient: Optional[float] = None
    H_ambient: Optional[float] = None


@dataclass(frozen=True)
class CropBounds:
    """Target and safety envelope for one environmental variable in one zone."""

    crop_id: str
    stage: str
    var: str
    unit: str
    min_acceptable: float
    optimal_min: float
    optimal_max: float
    max_acceptable: float
    is_fallback: bool = False

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.optimal_min + self.optimal_max)


@dataclass
class PIDGains:
    """Tuning parameters for one PID loop (kept strictly separate from crop targets)."""

    Kp: float
    Ki: float
    Kd: float = 0.0
    tau_f: float = 2.0
    T_t: float = 10.0
    Ts: float = 1.0
    dt: float = 1.0
    u_min: float = -1.0
    u_max: float = 1.0
    unidirectional_positive: bool = False


@dataclass
class ActuatorConfig:
    """Mapping, PWM, dwell, and arbitration parameters for one zone."""

    pwm_period_heat: int = 20
    pwm_period_mist: int = 10
    min_on_steps: int = 2
    min_off_steps: int = 2
    min_duty_cutoff: float = 0.05
    fan_on_thresh: float = 0.20
    fan_off_thresh: float = 0.08
    water_max_pulse_steps: int = 2
    water_soak_steps: int = 10
    ambient_margin_T: float = 0.5
    ambient_margin_H: float = 1.5
    arbiter_hysteresis: float = 0.05
    allow_cold_dehumidification: bool = False
    # Tie-break bias (in normalized-excursion units) given to temperature over humidity
    # when the shared fan is contested. 0.0 makes the arbiter symmetric.
    temp_priority_bias: float = 0.15


@dataclass(frozen=True)
class ArbitrationBlockFlags:
    """Feedback from ActuatorArbiter/Mapper to PID anti-windup."""

    block_T_pos: bool = False
    block_T_neg: bool = False
    block_H_pos: bool = False
    block_H_neg: bool = False
    block_M_pos: bool = False


@dataclass(frozen=True)
class ZoneStatus:
    """Telemetry snapshot for one zone at one control step."""

    zone_id: str
    mode: ControlMode
    executed_action: tuple[int, int, int, int]
    pid_action: tuple[int, int, int, int]
    marl_valid: bool
    marl_failure_reason: Optional[str]
    recovery_streak: int
    transitioned_from: Optional[ControlMode] = None
    transition_reason: Optional[str] = None
    arbiter_reason: str = ""
    measurement_qualities: dict[str, MeasurementQuality] = field(default_factory=dict)
