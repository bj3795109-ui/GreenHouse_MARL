"""
Per-zone controller composing three independent PID loops (T, H, M),
ActuatorMapper, and ActuatorArbiter.
"""

from __future__ import annotations

from dataclasses import dataclass

from control.actuator_arbiter import ActuatorArbiter, ArbitrationResult
from control.actuator_mapper import ActuatorMapper
from control.pid_controller import PIDController
from control.types import (
    ActuatorConfig,
    AmbientState,
    ArbitrationBlockFlags,
    CropBounds,
    PIDGains,
    SensorMeasurement,
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
        gains: dict[str, PIDGains],
        actuator_cfg: ActuatorConfig | None = None,
    ):
        self.zone_id = zone_id
        self.pids: dict[str, PIDController] = {
            "T": PIDController("T", gains["T"]),
            "H": PIDController("H", gains["H"]),
            "M": PIDController("M", gains["M"]),
        }
        cfg = actuator_cfg or ActuatorConfig()
        self.mapper = ActuatorMapper(cfg)
        self.arbiter = ActuatorArbiter(cfg)
        self._last_blocks = ArbitrationBlockFlags()

    def reset(self) -> None:
        for pid in self.pids.values():
            pid.reset()
        self.mapper.reset()
        self.arbiter.reset()
        self._last_blocks = ArbitrationBlockFlags()

    def step(
        self,
        measurements: dict[str, SensorMeasurement],
        bounds: dict[str, CropBounds],
        ambient: AmbientState,
    ) -> ZoneControlOutput:
        """Run one active PID + Mapper + Arbiter control step."""
        b = self._last_blocks
        u_T = self.pids["T"].step(
            measurements["T"], bounds["T"], block_positive=b.block_T_pos, block_negative=b.block_T_neg
        )
        u_H = self.pids["H"].step(
            measurements["H"], bounds["H"], block_positive=b.block_H_pos, block_negative=b.block_H_neg
        )
        u_M = self.pids["M"].step(
            measurements["M"], bounds["M"], block_positive=b.block_M_pos, block_negative=False
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
        measurements: dict[str, SensorMeasurement],
        bounds: dict[str, CropBounds],
        executed_action: tuple[int, int, int, int],
        ambient: AmbientState,
    ) -> None:
        """
        Keep PID internal states and actuator dwell timers synchronized with
        MARL/SAFE_MODE executed actions for bumpless handover.
        """
        heat, fan, mist, water = executed_action
        self.mapper.sync_executed(executed_action)
        self.arbiter.sync_fan(fan)

        # Infer effective signed effort applied to each variable by MARL
        T_val = measurements["T"].value
        H_val = measurements["H"].value

        eff_T = 1.0 if heat else 0.0
        if fan and not heat and T_val is not None and ambient.T_ambient is not None:
            if T_val > ambient.T_ambient:
                eff_T = -1.0

        eff_H = 1.0 if mist else 0.0
        if fan and not mist and H_val is not None and ambient.H_ambient is not None:
            if H_val > ambient.H_ambient:
                eff_H = -1.0

        eff_M = 1.0 if water else 0.0

        self.pids["T"].track_shadow(measurements["T"], bounds["T"], eff_T)
        self.pids["H"].track_shadow(measurements["H"], bounds["H"], eff_H)
        self.pids["M"].track_shadow(measurements["M"], bounds["M"], eff_M)
        self._last_blocks = ArbitrationBlockFlags()
