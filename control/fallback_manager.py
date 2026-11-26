"""
Top-level multi-zone orchestrator coordinating per-zone Supervisors,
ZoneControllers, CropConfigLoader, and SensorInterface around GreenhouseEnv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from control.crop_config_loader import CropConfigLoader
from control.sensor_interface import SensorInterface, SensorInterfaceConfig
from control.supervisor import SupervisorConfig, ZoneSupervisor
from control.types import (
    ActuatorConfig,
    AmbientState,
    ControlMode,
    PIDGains,
    SensorMeasurement,
    ZONE_IDS,
    ZoneStatus,
)
from control.zone_controller import ZoneController


@dataclass(frozen=True)
class FallbackStepResult:
    """Output returned by FallbackManager.step()."""

    actions: dict[str, np.ndarray]
    statuses: dict[str, ZoneStatus]
    measurements: dict[str, dict[str, SensorMeasurement]]


class FallbackManager:
    """
    Coordinates three independent zone supervisors and PID fallback controllers.

    Ensures final actions strictly conform to GreenhouseEnv's `MultiBinary(4)`
    action space and enables isolated per-zone or simultaneous multi-zone fallback.
    """

    def __init__(
        self,
        crop_loader: CropConfigLoader,
        pid_gains: Mapping[str, Mapping[str, PIDGains]],
        actuator_cfg: Optional[ActuatorConfig] = None,
        sensor_cfg: Optional[SensorInterfaceConfig] = None,
        supervisor_cfg: Optional[SupervisorConfig] = None,
        initial_mode: ControlMode = ControlMode.MARL_ACTIVE,
    ):
        self.crop_loader = crop_loader
        self.sensor_interface = SensorInterface(sensor_cfg)
        self.controllers: dict[str, ZoneController] = {}
        self.supervisors: dict[str, ZoneSupervisor] = {}

        for z in ZONE_IDS:
            zc = ZoneController(z, dict(pid_gains[z]), actuator_cfg=actuator_cfg)
            sup = ZoneSupervisor(z, zc, cfg=supervisor_cfg, initial_mode=initial_mode)
            self.controllers[z] = zc
            self.supervisors[z] = sup

    def reset(self) -> None:
        self.sensor_interface.reset()
        for sup in self.supervisors.values():
            sup.reset()

    def step(
        self,
        marl_actions: Optional[Mapping[str, Any]],
        sensor_inputs: Optional[Mapping[str, Any]],
        ambient: Optional[AmbientState] = None,
        agent_errors: Optional[Mapping[str, bool]] = None,
        inputs_are_trust_records: bool = True,
        current_time: Optional[float] = None,
    ) -> FallbackStepResult:
        """
        Execute one supervised control cycle across all 3 greenhouse zones.

        Parameters
        ----------
        marl_actions : Mapping[str, Any] | None
            Proposed actions keyed by zone_id. Missing keys or malformed payloads
            trigger per-zone fallback cleanly.
        sensor_inputs : Mapping[str, Any] | None
            Either TrustNetwork records (`inputs_are_trust_records=True`) or
            GreenhouseEnv observation vectors (`inputs_are_trust_records=False`).
        ambient : AmbientState | None
            Ambient (T_ambient, H_ambient) used for ventilation arbitration.
        agent_errors : Mapping[str, bool] | None
            Optional per-zone flag indicating an upstream agent crash or timeout.
        current_time : float | None
            Optional simulation/wall-clock timestamp for staleness validation.
        """
        amb = ambient or AmbientState()
        if inputs_are_trust_records:
            meas_by_zone = self.sensor_interface.from_trust_records(
                sensor_inputs, current_time=current_time
            )
        else:
            meas_by_zone = self.sensor_interface.from_env_observations(
                sensor_inputs, current_time=current_time
            )

        final_actions: dict[str, np.ndarray] = {}
        statuses: dict[str, ZoneStatus] = {}
        marl_map = marl_actions or {}
        err_map = agent_errors or {}

        for z in ZONE_IDS:
            bounds = self.crop_loader.get_zone_targets(z)
            raw_act = marl_map.get(z, None)
            failed = bool(err_map.get(z, False)) or (z not in marl_map)

            status = self.supervisors[z].step(
                raw_marl_action=raw_act,
                measurements=meas_by_zone[z],
                bounds=bounds,
                ambient=amb,
                agent_failed=failed,
            )
            statuses[z] = status
            final_actions[z] = np.array(status.executed_action, dtype=np.int8)

        return FallbackStepResult(
            actions=final_actions,
            statuses=statuses,
            measurements=meas_by_zone,
        )

    def get_metrics(self) -> dict[str, dict[str, Any]]:
        """Return per-zone fallback, safety, switching, and recovery metrics."""
        out: dict[str, dict[str, Any]] = {}
        for z, sup in self.supervisors.items():
            m = sup.metrics
            total = max(1, m.total_steps)
            out[z] = {
                "mode": sup.mode.value,
                "total_steps": m.total_steps,
                "marl_active_steps": m.marl_active_steps,
                "pid_fallback_steps": m.pid_fallback_steps,
                "recovery_pending_steps": m.recovery_pending_steps,
                "safe_mode_steps": m.safe_mode_steps,
                "safe_mode_events": m.safe_mode_events,
                "fallback_fraction": (m.pid_fallback_steps + m.safe_mode_steps) / total,
                "unsafe_actions_intercepted": m.unsafe_actions_intercepted,
                "marl_to_pid_switches": m.marl_to_pid_switches,
                "recovery_attempts": m.recovery_attempts,
                "recovery_successes": m.recovery_successes,
                "recovery_success_rate": (
                    m.recovery_successes / m.recovery_attempts if m.recovery_attempts > 0 else 1.0
                ),
                "actuator_toggles": m.actuator_toggles,
                "transitions": list(m.transition_log),
            }
        return out
