"""
Continuous PID output to binary actuator request mapping with software PWM,
minimum on/off dwell timers, and irrigation pulse/soak lockout.
"""

from __future__ import annotations

from dataclasses import dataclass

from control.types import ACTUATOR_NAMES, ActuatorConfig


@dataclass(frozen=True)
class MappedRequests:
    """Intermediate requests before shared-fan arbitration."""

    heat_cmd: int
    mist_cmd: int
    water_cmd: int
    cool_demand: float
    heat_assist_demand: float
    dehum_demand: float
    hum_assist_demand: float
    water_soak_blocked: bool


class DwellTracker:
    """Enforces minimum ON and minimum OFF durations to prevent relay chatter."""

    def __init__(self, min_on_steps: int, min_off_steps: int):
        self.min_on_steps = max(1, int(min_on_steps))
        self.min_off_steps = max(1, int(min_off_steps))
        self.state: int = 0
        self.steps_in_state: int = self.min_off_steps

    def reset(self) -> None:
        self.state = 0
        self.steps_in_state = self.min_off_steps

    def sync_external(self, actual_state: int) -> None:
        """Keep dwell timer synchronized when MARL or SAFE_MODE drives the actuator."""
        s = 1 if actual_state else 0
        if s == self.state:
            self.steps_in_state += 1
        else:
            self.state = s
            self.steps_in_state = 1

    def apply(self, desired_state: int) -> int:
        desired = 1 if desired_state else 0
        if desired == self.state:
            self.steps_in_state += 1
            return self.state

        if self.state == 1 and self.steps_in_state < self.min_on_steps:
            self.steps_in_state += 1
            return 1
        if self.state == 0 and self.steps_in_state < self.min_off_steps:
            self.steps_in_state += 1
            return 0

        self.state = desired
        self.steps_in_state = 1
        return self.state

    def can_switch_to(self, desired_state: int) -> bool:
        desired = 1 if desired_state else 0
        if desired == self.state:
            return True
        if self.state == 1:
            return self.steps_in_state >= self.min_on_steps
        return self.steps_in_state >= self.min_off_steps


class ActuatorMapper:
    """
    Converts normalized continuous PID outputs (u_T, u_H, u_M) into:
    - Time-proportioned (PWM) + dwell-protected binary commands for heat & mist
    - Pulse-limited + soak-locked binary command for water
    - Continuous scored ventilation demands for ActuatorArbiter
    """

    def __init__(self, cfg: ActuatorConfig | None = None):
        self.cfg = cfg or ActuatorConfig()
        self.dwell: dict[str, DwellTracker] = {
            name: DwellTracker(self.cfg.min_on_steps, self.cfg.min_off_steps)
            for name in ACTUATOR_NAMES
        }
        self._pwm_phase: dict[str, int] = {"heat": 0, "mist": 0}
        self._water_pulse_steps: int = 0
        self._water_soak_remaining: int = 0

    def reset(self) -> None:
        for d in self.dwell.values():
            d.reset()
        self._pwm_phase = {"heat": 0, "mist": 0}
        self._water_pulse_steps = 0
        self._water_soak_remaining = 0

    def sync_executed(self, action: tuple[int, int, int, int]) -> None:
        """Synchronize dwell timers with the actual executed action (for bumpless handover)."""
        for name, val in zip(ACTUATOR_NAMES, action):
            self.dwell[name].sync_external(val)
        if action[3] == 0 and self._water_pulse_steps > 0:
            self._water_pulse_steps = 0
            self._water_soak_remaining = self.cfg.water_soak_steps
        elif self._water_soak_remaining > 0 and action[3] == 0:
            self._water_soak_remaining -= 1

    def _pwm_decision(self, name: str, demand: float, period: int) -> int:
        if demand < self.cfg.min_duty_cutoff:
            self._pwm_phase[name] = (self._pwm_phase[name] + 1) % max(1, period)
            return 0
        period = max(1, int(period))
        phase = self._pwm_phase[name] % period
        self._pwm_phase[name] = (phase + 1) % period
        on_steps = round(min(1.0, max(0.0, demand)) * period)
        return 1 if phase < on_steps else 0

    def map_outputs(self, u_T: float, u_H: float, u_M: float) -> MappedRequests:
        c = self.cfg

        heat_demand = max(0.0, float(u_T))
        cool_demand = max(0.0, -float(u_T))
        mist_demand = max(0.0, float(u_H))
        dehum_demand = max(0.0, -float(u_H))
        water_demand = max(0.0, float(u_M))

        raw_heat = self._pwm_decision("heat", heat_demand, c.pwm_period_heat)
        raw_mist = self._pwm_decision("mist", mist_demand, c.pwm_period_mist)

        heat_cmd = self.dwell["heat"].apply(raw_heat)
        mist_cmd = self.dwell["mist"].apply(raw_mist)

        # Water: pulse-bounded + soak lockout to handle soil infiltration lag
        soak_blocked = False
        if self._water_soak_remaining > 0:
            raw_water = 0
            self._water_soak_remaining -= 1
            if water_demand >= c.min_duty_cutoff:
                soak_blocked = True
        elif water_demand >= c.min_duty_cutoff:
            if self._water_pulse_steps < c.water_max_pulse_steps:
                raw_water = 1
                self._water_pulse_steps += 1
            else:
                raw_water = 0
                self._water_pulse_steps = 0
                self._water_soak_remaining = max(0, c.water_soak_steps - 1)
                soak_blocked = True
        else:
            raw_water = 0
            self._water_pulse_steps = 0

        # Water bypasses min_on_steps if max_pulse is smaller than min_on_steps,
        # so respect explicit pulse bound cleanly while preventing sub-min_off bounce.
        if raw_water == 1 and self.dwell["water"].state == 0 and not self.dwell["water"].can_switch_to(1):
            water_cmd = 0
            soak_blocked = True
        else:
            self.dwell["water"].sync_external(raw_water)
            water_cmd = raw_water

        return MappedRequests(
            heat_cmd=heat_cmd,
            mist_cmd=mist_cmd,
            water_cmd=water_cmd,
            cool_demand=cool_demand,
            heat_assist_demand=heat_demand,
            dehum_demand=dehum_demand,
            hum_assist_demand=mist_demand,
            water_soak_blocked=soak_blocked,
        )
