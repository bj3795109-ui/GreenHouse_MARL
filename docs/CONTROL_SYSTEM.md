# Control System Specification & Implementation

This document details the mathematical formulation and runtime behavior of the PID fallback controllers, crop configuration, actuator mapping, shared-fan arbitration, and software interlocks.

---

## 1. Control Loops (`T`, `H`, `M`)

Each zone runs three independent discrete-time loops inside `ZoneController` ([control/zone_controller.py](../control/zone_controller.py)):

| Variable | Symbol | Loop Type | Configured Baseline Gains ([pid_tuning.json](../control/configs/pid_tuning.json)) | Actuators |
|---|---|---|---|---|
| Temperature | `T` | PID | `Kp=0.40, Ki=0.002, Kd=2.0, tau_f=5.0, T_t=20.0, u in [-1, 1]` | Heater (`+`), Fan (`-` when `T > T_ambient`) |
| Relative Humidity | `H` | PI | `Kp=0.12, Ki=0.005, Kd=0.0, tau_f=2.0, T_t=10.0, u in [-1, 1]` | Mister (`+`), Fan (`-` when `H > H_ambient`) |
| Soil Moisture | `M` | Unidirectional PI | `Kp=0.10, Ki=0.004, Kd=0.0, tau_f=2.0, T_t=10.0, u in [0, 1], unidirectional_positive=True` | Water pump (`+` only) |

Soil moisture uses **PI (`Kd = 0.0`)** because infiltration and decay dynamics are slower and delayed; derivative action on noisy/step moisture readings would amplify pump chatter.

---

## 2. Discrete PID Equations ([control/pid_controller.py](../control/pid_controller.py))

### Optimal-Band Deadband Error
Against crop bounds $[y_{\text{opt,min}}, y_{\text{opt,max}}]$:

$$
e_k = \begin{cases}
y_{\text{opt,min}} - y_k & \text{if } y_k < y_{\text{opt,min}} \\
0 & \text{if } y_{\text{opt,min}} \le y_k \le y_{\text{opt,max}} \\
0 & \text{if } y_k > y_{\text{opt,max}} \text{ and unidirectional\_positive} \\
y_{\text{opt,max}} - y_k & \text{if } y_k > y_{\text{opt,max}} \text{ otherwise}
\end{cases}
$$

### Control Law (with Effective Timestep $\Delta t_{\text{eff}}$)
- **Proportional**: $P_k = K_p e_k$
- **Filtered Derivative-on-Measurement** (zero derivative kick on crop/stage target shifts):
  $$D_k = \frac{\tau_f}{\tau_f + \Delta t_{\text{eff}}} D_{k-1} - \frac{K_d}{\tau_f + \Delta t_{\text{eff}}} (y_k - y_{k-1})$$
- **Provisional Integral**: $I_{\text{cand}} = I_{k-1} + K_i \Delta t_{\text{eff}} e_k$
- **Saturation**: $v_k = P_k + I_{\text{cand}} + D_k$, $u_k = \text{clip}(v_k, u_{\text{low}}, u_{\max})$, where $u_{\text{low}} = \max(0, u_{\min})$ when `unidirectional_positive=True`.
- **Back-Calculation & External-Block Anti-Windup**:
  $$I_k = I_{\text{cand}} + \frac{\Delta t_{\text{eff}}}{T_t}(u_k - v_k)$$
  If `block_positive` and $e_k > 0$, upward growth of $I_k$ is clamped to $I_{k-1}$; if `block_negative` and $e_k < 0$, downward growth is clamped to $I_{k-1}$.
- **Unidirectional Excess Unwinding**: When `unidirectional_positive=True` and $y_k > y_{\text{opt,max}}$, output is immediately $0.0$ and stored positive integral unwinds smoothly:
  $$I_k = \text{clip}\!\left(I_{k-1} e^{-\Delta t_{\text{eff}}/T_t} + K_i \Delta t_{\text{eff}}(y_{\text{opt,max}} - y_k),\ u_{\text{low}},\ u_{\max}\right)$$

---

## 3. Variable Timestep & Multi-Rate Sampling

`PIDController.step(..., dt=None)` and `ZoneController.step(..., dt=None)` accept an optional finite positive `dt` (seconds):
- When `Ts == gains.dt` (single-rate mode, default `1.0 s`): every step updates immediately using $\Delta t_{\text{eff}} = \text{dt}$ (or `gains.dt` if `dt is None`).
- When `Ts > gains.dt` (multi-rate sampling gate): steps accumulate `elapsed_since_update += dt` and hold `last_output` until `elapsed_since_update >= Ts`, preventing over-integration.
- `track_shadow()` resets `elapsed_since_update = 0.0` and `has_updated = False` so PID takes over immediately upon MARL->PID handover.

---

## 4. Crop Target Configuration ([control/crop_config_loader.py](../control/crop_config_loader.py))

- Canonical units: `T: degC` (with explicit `degF -> degC` conversion), `H: RH_percent`, `M: relative_percent`. Incompatible soil moisture units (e.g., `VWC_percent`) are explicitly rejected with `CropConfigError`.
- Target updates (`update_zone_stage`, `assign_zone`) alter `CropBounds` only and **never** overwrite `PIDGains`.
- When `set_targets()` shifts the optimal band across the previous band opposite to stored integral sign, stale integral is zeroed.

---

## 5. Actuator Mapping & Irrigation Pulse/Soak ([control/actuator_mapper.py](../control/actuator_mapper.py))

1. **Input validation**: `u_T, u_H, u_M` must be finite, non-boolean floats in `[-1.0, 1.0]`.
2. **Time-proportioned PWM**: Positive `u_T` (`heat`) and `u_H` (`mist`) drive software PWM (`pwm_period_heat=20`, `pwm_period_mist=10`) above `min_duty_cutoff=0.05`, passed through `DwellTracker` (`min_on_steps=2`, `min_off_steps=2`).
3. **Irrigation pulse & soak lockout**:
   - Pump (`water`) runs at most `water_max_pulse_steps` (default `2`) consecutive steps, then enforces `water_soak_steps` (default `10`) OFF steps (`water_soak_blocked=True`, triggering PID `block_M_pos`).
   - `sync_executed()` tracks MARL `water=1` steps into `_water_pulse_steps` and `water=0` steps into `_water_soak_remaining`, preventing PID from starting a fresh irrigation pulse immediately after MARL exhausted the pulse budget.

---

## 6. Shared-Fan Arbitration & Mutual Exclusion ([control/actuator_arbiter.py](../control/actuator_arbiter.py))

`ActuatorArbiter.arbitrate()` validates finite `MappedRequests`, `CropBounds`, `SensorMeasurement`, and `AmbientState`, then:
1. Evaluates fan utility $(+1, 0, -1)$ for `T` and `H` relative to `(T_ambient, H_ambient)` with margins (`ambient_margin_T=0.5`, `ambient_margin_H=1.5`) and hysteresis (`fan_on_thresh=0.20`, `fan_off_thresh=0.08`). If outside conditions would move a variable the wrong way (or ambient is `None`), cooling/dehumidifying fan demand is inhibited and `block_T_neg` / `block_H_neg` are set.
2. Resolves `T` vs. `H` conflicts (`+1` vs. `-1`) by checking acceptability breaches (`_outside_acceptable`), over-temperature priority, normalized excursion severity, and `arbiter_hysteresis=0.05`.
3. Enforces `fan_dwell` minimum on/off steps and intra-zone mutual exclusion:
   - Active cooling fan blocks `heat_cmd` (`block_T_pos=True`) unless `allow_cold_dehumidification=True` and `H` is outside acceptable bounds.
   - Active drying fan blocks `mist_cmd` (`block_H_pos=True`).

---

## 7. Software Safety Interlocks ([control/supervisor.py](../control/supervisor.py))

As a last line before returning `(heat, fan, mist, water)`:
- Malformed or non-finite actions are clamped to `(0, 0, 0, 0)`.
- Non-finite `T`, `H`, or `M` readings force `heat = 0`, `mist = 0`, or `water = 0`, respectively.
- `T > T_max_hard` (`45.0 °C`) forces `heat = 0`.
- Continuous runtime watchdogs clamp `heat = 0` beyond `t_max_continuous_heat` (`120` steps) and `water = 0` beyond `t_max_continuous_water` (`20` steps).
- **Hardware boundary**: These software checks complement—and never replace—independent ESP32 firmware safety interlocks and local microcontroller PID fallback on PC communication loss.
