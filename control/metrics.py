"""
Episode evaluation and controller comparison metrics aligned to pdireadme.md §11.3.

Tracks all 13 metrics:
- T, H, M tracking error (RMSE, MAE, IAE outside optimal range)
- Time outside crop target ranges (optimal and acceptable)
- Overshoot & settling time
- Time to enter fallback & time to recover
- Unsafe action count
- Actuator switching frequency
- Time spent in fallback
- Recovery success rate
- Safe-mode event count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from control.types import ControlMode, CropBounds, VARS, ZONE_IDS, ZoneStatus


@dataclass
class _ZoneEvalAccumulator:
    steps: int = 0
    errors: dict[str, list[float]] = field(default_factory=lambda: {v: [] for v in VARS})
    values: dict[str, list[float]] = field(default_factory=lambda: {v: [] for v in VARS})
    outside_optimal_steps: dict[str, int] = field(default_factory=lambda: {v: 0 for v in VARS})
    outside_acceptable_steps: dict[str, int] = field(default_factory=lambda: {v: 0 for v in VARS})
    max_overshoot: dict[str, float] = field(default_factory=lambda: {v: 0.0 for v in VARS})
    last_out_of_opt_step: dict[str, int] = field(default_factory=lambda: {v: -1 for v in VARS})
    first_fallback_step: Optional[int] = None
    fallback_entry_step: Optional[int] = None
    recovery_durations_steps: list[int] = field(default_factory=list)
    prev_action: Optional[tuple[int, int, int, int]] = None
    actuator_toggles: int = 0


class EpisodeEvaluator:
    """Collects per-step state, crop bounds, and supervisor statuses to compute §11.3 metrics."""

    def __init__(self, dt: float = 1.0, fault_start_step: Optional[int] = None):
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.dt = float(dt)
        self.fault_start_step = fault_start_step
        self._acc: dict[str, _ZoneEvalAccumulator] = {z: _ZoneEvalAccumulator() for z in ZONE_IDS}

    def record_step(
        self,
        step_idx: int,
        true_state: Mapping[str, Mapping[str, float]],
        bounds_by_zone: Mapping[str, Mapping[str, CropBounds]],
        statuses: Mapping[str, ZoneStatus],
    ) -> None:
        for z in ZONE_IDS:
            acc = self._acc[z]
            acc.steps += 1
            st = statuses[z]

            if acc.prev_action is not None:
                acc.actuator_toggles += sum(
                    1 for a, b in zip(acc.prev_action, st.executed_action) if a != b
                )
            acc.prev_action = st.executed_action

            if st.transitioned_from == ControlMode.MARL_ACTIVE and st.mode == ControlMode.PID_FALLBACK:
                if acc.first_fallback_step is None:
                    acc.first_fallback_step = step_idx
                acc.fallback_entry_step = step_idx
            elif st.mode == ControlMode.PID_FALLBACK and acc.fallback_entry_step is None:
                acc.fallback_entry_step = step_idx
            elif st.transitioned_from == ControlMode.RECOVERY_PENDING and st.mode == ControlMode.MARL_ACTIVE:
                if acc.fallback_entry_step is not None:
                    acc.recovery_durations_steps.append(step_idx - acc.fallback_entry_step)
                    acc.fallback_entry_step = None

            z_state = true_state[z]
            z_bounds = bounds_by_zone[z]
            for v in VARS:
                val = float(z_state[v])
                b = z_bounds[v]
                acc.values[v].append(val)

                if val < b.optimal_min:
                    err = b.optimal_min - val
                    acc.outside_optimal_steps[v] += 1
                    acc.last_out_of_opt_step[v] = acc.steps - 1
                    acc.max_overshoot[v] = max(acc.max_overshoot[v], err)
                elif val > b.optimal_max:
                    err = val - b.optimal_max
                    acc.outside_optimal_steps[v] += 1
                    acc.last_out_of_opt_step[v] = acc.steps - 1
                    acc.max_overshoot[v] = max(acc.max_overshoot[v], err)
                else:
                    err = 0.0

                acc.errors[v].append(err)
                if val < b.min_acceptable or val > b.max_acceptable:
                    acc.outside_acceptable_steps[v] += 1

    def summarize(self, manager_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for z in ZONE_IDS:
            acc = self._acc[z]
            mm = manager_metrics[z]
            dur_min = max(1e-9, (acc.steps * self.dt) / 60.0)

            var_metrics: dict[str, dict[str, Any]] = {}
            for v in VARS:
                errs = np.asarray(acc.errors[v], dtype=float)
                last_bad = acc.last_out_of_opt_step[v]
                settled_step = 0 if last_bad < 0 else (last_bad + 1 if last_bad + 1 < acc.steps else None)
                var_metrics[v] = {
                    "rmse": float(np.sqrt(np.mean(errs ** 2))) if len(errs) else 0.0,
                    "mae": float(np.mean(errs)) if len(errs) else 0.0,
                    "iae": float(np.sum(errs) * self.dt) if len(errs) else 0.0,
                    "max_overshoot": acc.max_overshoot[v],
                    "time_outside_optimal_s": acc.outside_optimal_steps[v] * self.dt,
                    "time_outside_acceptable_s": acc.outside_acceptable_steps[v] * self.dt,
                    "settling_time_s": (settled_step * self.dt) if settled_step is not None else None,
                }

            t_enter_fb = None
            if self.fault_start_step is not None and acc.first_fallback_step is not None:
                t_enter_fb = max(0.0, (acc.first_fallback_step - self.fault_start_step) * self.dt)

            mean_rec = (
                float(np.mean(acc.recovery_durations_steps) * self.dt)
                if acc.recovery_durations_steps
                else None
            )

            out[z] = {
                "variables": var_metrics,
                "time_to_enter_fallback_s": t_enter_fb,
                "mean_time_to_recover_s": mean_rec,
                "unsafe_actions_intercepted": mm["unsafe_actions_intercepted"],
                "actuator_toggles": acc.actuator_toggles,
                "switching_frequency_per_min": acc.actuator_toggles / dur_min,
                "time_in_fallback_s": mm["pid_fallback_steps"] * self.dt,
                "fallback_fraction": mm["fallback_fraction"],
                "recovery_success_rate": mm["recovery_success_rate"],
                "safe_mode_events": mm.get("safe_mode_events", 0),
                "safe_mode_steps": mm["safe_mode_steps"],
            }
        return out
