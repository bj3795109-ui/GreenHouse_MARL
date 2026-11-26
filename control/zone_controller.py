"""
Per-zone controller composing three independent PID loops (T, H, M),
ActuatorMapper, and ActuatorArbiter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional

from control.actuator_arbiter import ActuatorArbiter, ArbitrationResult, _validate_optional_finite
from control.actuator_mapper import ActuatorMapper, validate_binary_action
from control.pid_controller import PIDController
from control.types import (
    ActuatorConfig,
    AmbientState,
    ArbitrationBlockFlags,
    CropBounds,
    PIDGains,
    SensorMeasurement,
    VARS,
)


@dataclass(frozen=True)
class ZoneControlOutput:
    """Result of one ZoneController tick."""

    action: tuple[int, int, int, int]
    u_T: float
    u_H: float
    u_M: float
    arbiter_reason: str


class ZoneController:
    """Independent 3-loop PID + Mapping + Arbitration controller for one zone."""

    def __init__(
        self,
        zone_id: str,
        gains: Mapping[str, PIDGains],
        actuator_cfg: ActuatorConfig | None = None,
    ):
        if not isinstance(zone_id, str) or not zone_id.strip():
            raise ValueError(f"zone_id must be a non-empty string, got {zone_id!r}")
        for v in VARS:
            if v not in gains:
                raise ValueError(f"Missing PIDGains for variable '{v}' in zone '{zone_id}'")

        self.zone_id = zone_id
        self.pids: dict[str, PIDController] = {
            v: PIDController(v, replace(gains[v])) for v in VARS
        }
        cfg = replace(actuator_cfg) if actuator_cfg is not None else ActuatorConfig()
        self.mapper = ActuatorMapper(cfg)
        self.arbiter = ActuatorArbiter(cfg)
        self._last_blocks = ArbitrationBlockFlags()

    def reset(self) -> None:
        for pid in self.pids.values():
            pid.reset()
        self.mapper.reset()
        self.arbiter.reset()
        self._last_blocks = ArbitrationBlockFlags()

    def _validate_inputs(
        self,
        measurements: Mapping[str, SensorMeasurement],
        bounds: Mapping[str, CropBounds],
        ambient: AmbientState,
    ) -> tuple[Optional[float], Optional[float]]:
        for v in VARS:
            if v not in measurements:
                raise ValueError(f"Zone '{self.zone_id}' missing measurement for '{v}'")
            if v not in bounds:
                raise ValueError(f"Zone '{self.zone_id}' missing crop bounds for '{v}'")
        if not isinstance(ambient, AmbientState):
            raise TypeError(f"Expected AmbientState, got {type(ambient).__name__}")
        amb_T = _validate_optional_finite(ambient.T_ambient, "ambient.T_ambient")
        amb_H = _validate_optional_finite(ambient.H_ambient, "ambient.H_ambient")
        return amb_T, amb_H

    def step(
        self,
        measurements: Mapping[str, SensorMeasurement],
        bounds: Mapping[str, CropBounds],
        ambient: AmbientState,
        dt: Optional[float] = None,
    ) -> ZoneControlOutput:
        """Run one active PID + Mapper + Arbiter control step."""
        self._validate_inputs(measurements, bounds, ambient)
        b = self._last_blocks
        u_T = self.pids["T"].step(
            measurements["T"],
            bounds["T"],
            block_positive=b.block_T_pos,
            block_negative=b.block_T_neg,
            dt=dt,
        )
        u_H = self.pids["H"].step(
            measurements["H"],
            bounds["H"],
            block_positive=b.block_H_pos,
            block_negative=b.block_H_neg,
            dt=dt,
        )
        u_M = self.pids["M"].step(
            measurements["M"],
            bounds["M"],
            block_positive=b.block_M_pos,
            block_negative=False,
            dt=dt,
        )

        reqs = self.mapper.map_outputs(u_T, u_H, u_M)
        arb: ArbitrationResult = self.arbiter.arbitrate(
            reqs,
            measurements["T"],
            measurements["H"],
            bounds["T"],
            bounds["H"],
            ambient,
        )
        self._last_blocks = arb.block_flags
        return ZoneControlOutput(
            action=arb.action,
            u_T=u_T,
            u_H=u_H,
            u_M=u_M,
            arbiter_reason=arb.reason,
        )

    def track_shadow(
        self,
        measurements: Mapping[str, SensorMeasurement],
        bounds: Mapping[str, CropBounds],
        executed_action: tuple[int, int, int, int],
        ambient: AmbientState,
    ) -> None:
        """
        Keep PID internal states and actuator dwell timers synchronized with
        MARL/SAFE_MODE executed actions for bumpless handover.
        """
        amb_T, amb_H = self._validate_inputs(measurements, bounds, ambient)
        heat, fan, mist, water = validate_binary_action(executed_action)

        T_val = _validate_optional_finite(measurements["T"].value, "measurements['T'].value")
        H_val = _validate_optional_finite(measurements["H"].value, "measurements['H'].value")
        _validate_optional_finite(measurements["M"].value, "measurements['M'].value")

        self.mapper.sync_executed((heat, fan, mist, water))
        self.arbiter.sync_fan(fan)

        # Infer effective signed effort applied to each variable by MARL
        eff_T = 1.0 if heat else 0.0
        if fan and not heat and T_val is not None and amb_T is not None:
            if T_val > amb_T:
                eff_T = -1.0

        eff_H = 1.0 if mist else 0.0
        if fan and not mist and H_val is not None and amb_H is not None:
            if H_val > amb_H:
                eff_H = -1.0

        eff_M = 1.0 if water else 0.0

        self.pids["T"].track_shadow(measurements["T"], bounds["T"], eff_T)
        self.pids["H"].track_shadow(measurements["H"], bounds["H"], eff_H)
        self.pids["M"].track_shadow(measurements["M"], bounds["M"], eff_M)
        self._last_blocks = ArbitrationBlockFlags()
