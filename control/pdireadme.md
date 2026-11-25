# Greenhouse MARL — Per-Zone PID Fallback Controller

**Status:** Architecture and design specification  
**Scope:** Crop-aware, per-zone PID fallback for a three-zone greenhouse MARL system  
**Implementation status:** Proposed; not yet fully implemented or validated.

---

## 1. Overview

This document describes the design of a PID fallback controller for our multi-agent greenhouse control system.

The greenhouse normally operates using Multi-Agent Reinforcement Learning (MARL), where each greenhouse zone has its own agent responsible for controlling environmental conditions.

If an agent fails, produces invalid actions, or becomes unreliable, its corresponding zone automatically switches to a local PID fallback controller.

The fallback mechanism is independent for each zone.

Our intended control strategy:

- MARL is the primary controller.
- Each zone has its own PID fallback controller.
- A supervisor monitors agent and sensor health.
- A zone switches to PID when its MARL agent becomes unreliable.
- The system automatically returns to MARL after successful recovery validation.
- Crop requirements are loaded dynamically from a dataset.
- Different zones can grow different crops at different growth stages.

**Important:** PID is a recovery mechanism, not a replacement for MARL.

---

## 2. Existing Project Architecture

The current greenhouse environment uses PettingZoo's `ParallelEnv`.

It consists of three interconnected zones:

- `zone_1`
- `zone_2`
- `zone_3`

Each zone has its own agent and environmental state.

### 2.1 Environmental variables

Each zone currently models three variables:

| Variable | Description |
|---|---|
| T | Temperature |
| H | Relative humidity |
| M | Soil moisture |

Each zone therefore requires three independent control loops:

1. Temperature control.
2. Humidity control.
3. Soil moisture control.

Across three zones, we will have **nine logical PID controllers**.

### 2.2 Actuators

The current environment provides four binary actuators per zone.

| Index | Actuator | Function |
|---|---|---|
| 0 | Heater | Increases temperature |
| 1 | Fan | Ventilates the zone |
| 2 | Mister | Increases humidity |
| 3 | Water pump | Increases soil moisture |

The existing action space is:

```python
spaces.MultiBinary(4)
```

The action vector is:

```python
[heat, fan, mist, water]
```

Each action is either `0` or `1`.

### 2.3 Observations

Each zone receives nine observation values:

- Its own temperature, humidity, and soil moisture.
- Temperature, humidity, and soil moisture of its two neighboring zones.

The PID implementation should not depend directly on PettingZoo internals. An adapter should translate observations and actions between the environment and the control system.

### 2.4 Existing physics limitations

The current environment uses simplified physics and provisional parameters.

Before tuning PID gains, we must review:

- Temperature dynamics.
- Humidity dynamics.
- Soil moisture dynamics.
- Ambient temperature and humidity.
- Inter-zone coupling.
- Euler integration and simulation timestep.
- State clipping.
- Actuator behavior.

Important concerns:

**Soil moisture:** The current equation may change moisture too quickly and requires timescale validation.

**Fan behavior:** Ventilation moves conditions toward ambient conditions. It can cool or warm the greenhouse and can dry or humidify it.

**Inter-zone coupling:** Neighboring zones influence one another, affecting the system's dynamics.

**Timestep:** PID calculations depend on the actual elapsed simulation time.

**Measurement units:** The exact physical meaning of `M` must be established before connecting crop moisture targets.

PID gains should not be selected arbitrarily against an unvalidated simulation model.

---

## 3. Main PID Architecture

### 3.1 High-level architecture

```text
                  CROP DATASET
                       |
                       v
               Crop Config Loader
                       |
                       v
             Per-Zone Crop Targets
                       |
                       v
              ZONE SUPERVISOR
                       |
          +------------+------------+
          |            |            |
        Zone 1       Zone 2       Zone 3
          |            |            |
       MARL/PID     MARL/PID     MARL/PID
          |            |            |
          v            v            v
      Temperature   Temperature   Temperature
         PID           PID           PID

      Humidity      Humidity      Humidity
         PID           PID           PID

      Moisture      Moisture      Moisture
         PID           PID           PID
          |            |            |
          +------------+------------+
                       |
                       v
                ACTUATOR MAPPER
                       |
                       v
                ACTUATOR ARBITER
                       |
                       v
                FINAL ACTIONS
                       |
                       v
             GREENHOUSE ENVIRONMENT
                       |
                       v
                 NEW OBSERVATIONS
                       |
                       +---- Feedback Loop

 Sensor Readings
       |
       v
 Sensor Trust Network
       |
       v
 Sensor Interface
       |
       +---- Validated Measurements
```

### 3.2 Core design principles

- Every zone has independent control.
- PID and MARL remain separate components.
- Crop requirements are configuration-driven.
- Sensor measurements must be validated.
- Shared actuators require arbitration.
- Controller transitions must be controlled.
- Failure in one zone must not unnecessarily affect other zones.

---

## 4. Independent PID Controllers

Each zone will contain three PID control loops.

```text
ZoneController
    |
    +-- Temperature PID
    |
    +-- Humidity PID
    |
    +-- Soil Moisture PID
```

For all three zones:

```text
zone_1:
    TemperaturePID
    HumidityPID
    SoilMoisturePID

zone_2:
    TemperaturePID
    HumidityPID
    SoilMoisturePID

zone_3:
    TemperaturePID
    HumidityPID
    SoilMoisturePID
```

Each controller maintains:

- Proportional gain (`Kp`).
- Integral gain (`Ki`).
- Derivative gain (`Kd`).
- Setpoint or target range.
- Current error.
- Integral accumulator.
- Previous measurement or error.
- Output limits.
- Anti-windup mechanism.
- Sampling interval.

### 4.1 PID mathematical model

The basic control error is:

\[
e(t)=r(t)-y(t)
\]

Where:

- \(r(t)\) = target value.
- \(y(t)\) = measured value.
- \(e(t)\) = error.

The PID output is:

\[
u(t)=K_p e(t)+K_i\int e(t)dt+K_d\frac{de(t)}{dt}
\]

The three components:

**Proportional:** Responds to the current error.

**Integral:** Corrects accumulated error over time.

**Derivative:** Responds to the rate of change of error.

### 4.2 Discrete implementation

For a discrete simulation:

\[
I_k=I_{k-1}+e_k\Delta t
\]

\[
D_k=\frac{e_k-e_{k-1}}{\Delta t}
\]

\[
u_k=K_p e_k+K_i I_k+K_dD_k
\]

The final implementation must include:

- Output saturation.
- Integral anti-windup.
- Appropriate derivative filtering.
- Measurement validation.
- Correct timestep handling.

Derivative-on-measurement should also be considered to reduce derivative kick when setpoints change.

### 4.3 PID versus PI

Not every environmental variable necessarily requires a full PID controller.

Potential approaches:

| Variable | Candidate controller |
|---|---|
| Temperature | PI or PID |
| Humidity | PI or PID |
| Soil moisture | PI or slower irrigation control |

The final choice must be based on the dynamics of the simulated greenhouse.

Soil moisture may respond slowly and irrigation has delayed effects. A full PID controller may not be the best choice.

### 4.4 Crop target bands

Crop requirements should provide acceptable ranges rather than forcing every variable toward an arbitrary fixed number.

The controller should consider:

- Minimum acceptable value.
- Optimal operating range.
- Maximum acceptable value.
- Deadband behavior.

The target selection strategy must avoid unnecessary actuator switching when the current environmental condition is already acceptable.

---

## 5. Crop-Aware Configuration

Different greenhouse zones may grow different crops and operate at different growth stages.

Example configuration:

```json
{
  "zones": {
    "zone_1": {
      "crop_id": "tomato",
      "growth_stage": "vegetative"
    },
    "zone_2": {
      "crop_id": "lettuce",
      "growth_stage": "mature"
    },
    "zone_3": {
      "crop_id": "cucumber",
      "growth_stage": "flowering"
    }
  }
}
```

These are illustrative assignments.

Crop assignments must remain configurable.

### 5.1 Crop dataset

The crop dataset should contain:

- Crop ID.
- Crop name.
- Growth stage.
- Temperature requirements.
- Humidity requirements.
- Soil moisture requirements.
- Minimum and maximum acceptable values.
- Optimal ranges.
- Measurement units.
- Data source or metadata.

Illustrative schema:

```json
{
  "crop_id": "example_crop",
  "growth_stages": {
    "vegetative": {
      "temperature": {
        "min": null,
        "optimal": [null, null],
        "max": null,
        "unit": "degC"
      },
      "humidity": {
        "min": null,
        "optimal": [null, null],
        "max": null,
        "unit": "%RH"
      },
      "soil_moisture": {
        "min": null,
        "optimal": [null, null],
        "max": null,
        "unit": "UNCONFIRMED"
      }
    }
  }
}
```

The `null` values are placeholders, not actual agricultural recommendations.

### 5.2 Soil moisture measurement

The current environment represents soil moisture as `M`, but its measurement basis has not yet been established.

Possible measurement types:

- Volumetric water content (VWC).
- Sensor-specific moisture percentage.
- Relative moisture ranges.
- Substrate-specific measurements.

These are not interchangeable.

Before integrating crop data, we must establish the exact meaning and unit of `M`.

### 5.3 Configuration requirements

The system must handle:

- Missing crop profiles.
- Missing growth stages.
- Invalid dataset values.
- Unsupported measurement units.
- Runtime growth-stage changes.
- Crop reassignment.

Crop requirements and PID gains must remain separate.

Changing the crop should update the control target, not automatically change PID gains.

---

## 6. Sensor Trust Network Integration

A separate Sensor Trust Network is being developed.

Based on the supplied documents, it provides sensor estimation and trust-related information.

Its conceptual outputs include:

- Raw measurements.
- Estimated values.
- Confidence values.
- Combined estimates.
- Residuals.
- Trust scores.
- Belief values.

It handles temperature, humidity, and soil moisture.

The Trust Network currently acts as a separate estimation component rather than directly controlling the environment.

### 6.1 Proposed integration

PID should not blindly consume raw sensor readings.

Instead, create a sensor interface that normalizes Trust Network outputs.

Illustrative measurement object:

```python
measurement = {
    "zone_id": "zone_1",
    "variable": "temperature",
    "value": 24.5,
    "source": "trusted_estimate",
    "valid": True,
    "confidence": None,
    "trust": None,
    "timestamp": None
}
```

The exact schema must be aligned with the actual Trust Network implementation.

### 6.2 Measurement validation

The sensor interface should determine whether:

1. Raw measurement is valid.
2. Estimated measurement is available.
3. Confidence and trust values meet configured criteria.
4. Measurement is stale.
5. Measurement is missing or non-finite.

If no trustworthy measurement exists, PID cannot reliably determine the true environmental state.

The system must then enter a separately defined safe behavior or safe mode.

Do not assume confidence scores are calibrated probabilities.

Do not invent universal trust thresholds.

### 6.3 Existing Trust Network limitations

The supplied documents describe a prototype with heuristic confidence calculations and limited fault coverage.

The current architecture should not claim complete fault tolerance or hardware-level safety.

PID should communicate with the Trust Network through an adapter so that both systems remain independently testable.

---

## 7. Actuator Mapping and Arbitration

PID controllers normally produce continuous outputs.

Our environment currently accepts binary actuator commands.

Therefore, we need a mapping layer.

### 7.1 Actuator mapping

| PID loop | Available actuators |
|---|---|
| Temperature | Heater, Fan |
| Humidity | Mister, Fan |
| Soil moisture | Water pump |

The mapping layer converts controller outputs into actuator requests.

### 7.2 Shared fan problem

The fan is shared between temperature and humidity control.

This creates potential conflicts.

For example:

- Temperature PID requests ventilation.
- Humidity PID requests ventilation.
- One loop requests ventilation while the other does not.

A dedicated actuator arbiter must resolve these requests.

### 7.3 Ambient-condition awareness

The fan moves greenhouse conditions toward ambient conditions.

Its effect depends on the difference between internal and external conditions.

For temperature:

- Lower ambient temperature may make ventilation cool the greenhouse.
- Higher ambient temperature may make ventilation heat the greenhouse.

For humidity:

- Lower ambient humidity may make ventilation dry the greenhouse.
- Higher ambient humidity may make ventilation humidify the greenhouse.

Therefore, simply activating the fan whenever temperature or humidity is high is not a reliable strategy.

The actuator arbiter must consider:

- Internal temperature and humidity.
- Ambient temperature and humidity.
- Crop target ranges.
- Desired correction direction.
- Other actuator requests.
- Safety constraints.

### 7.4 Binary control and hysteresis

Use hysteresis and deadbands to prevent rapid switching.

Also consider:

- Minimum actuator on-time.
- Minimum actuator off-time.
- Maximum continuous runtime.
- Output saturation.
- Conflicting actuator combinations.
- Deterministic arbitration.

The thresholds must be configurable and tested against the actual simulation timestep.

---

## 8. MARL-PID Supervisor

The supervisor decides which controller operates each zone.

Each zone has an independent state machine.

### 8.1 Proposed states

| State | Description |
|---|---|
| `MARL_ACTIVE` | MARL controls the zone |
| `PID_FALLBACK` | PID controls the zone |
| `RECOVERY_PENDING` | MARL agent is being validated |
| `SAFE_MODE` | No trustworthy control or measurement path is available |

### 8.2 MARL to PID transition

Possible failure conditions:

- Agent crash.
- Missing action.
- Invalid action shape.
- NaN or infinite action values.
- Unsafe action.
- Unavailable observation.
- Unreliable sensor measurements.
- Persistent poor control performance, if a defensible detector exists.

Failure conditions must be explicit and testable.

### 8.3 PID fallback operation

When a zone enters `PID_FALLBACK`:

1. Obtain validated sensor measurements.
2. Load that zone's crop profile.
3. Update its temperature PID.
4. Update its humidity PID.
5. Update its soil moisture PID.
6. Convert outputs into actuator requests.
7. Resolve shared actuator conflicts.
8. Validate the final action.
9. Send the action to the environment.

Other healthy zones remain under MARL.

### 8.4 Automatic MARL recovery

Recovery must be automatic but should not happen after only one valid action.

The recovery process should validate:

- Agent availability.
- Consecutive valid actions.
- Reliable observations.
- Safety checks.
- Absence of active failure conditions.
- Stable handover conditions.

Once these criteria are satisfied, the supervisor transitions through `RECOVERY_PENDING` and transfers control back to MARL.

### 8.5 Bumpless transfer

Switching controllers must not create sudden actuator changes.

The implementation should consider:

- Comparing the current PID output with the proposed MARL action.
- Controlled handover.
- PID integral-state handling.
- Resetting or reconciling inactive controller state.
- Validating the first action after recovery.

### 8.6 Independent zone operation

Example:

```text
zone_1 -> PID_FALLBACK
zone_2 -> MARL_ACTIVE
zone_3 -> MARL_ACTIVE
```

If every agent fails:

```text
zone_1 -> PID_FALLBACK
zone_2 -> PID_FALLBACK
zone_3 -> PID_FALLBACK
```

Each zone must transition independently.

---

## 9. Proposed Software Architecture

Recommended directory structure:

```text
project/
│
├── env/
│   └── env.py
│
├── control/
│   ├── __init__.py
│   ├── pid_controller.py
│   ├── zone_controller.py
│   ├── actuator_mapper.py
│   ├── actuator_arbiter.py
│   ├── supervisor.py
│   ├── fallback_manager.py
│   ├── crop_config_loader.py
│   └── sensor_interface.py
│
├── config/
│   ├── crops.json
│   └── zone_assignments.json
│
├── tests/
│   ├── test_pid_controller.py
│   ├── test_zone_controller.py
│   ├── test_actuator_mapper.py
│   ├── test_actuator_arbiter.py
│   ├── test_supervisor.py
│   ├── test_crop_config_loader.py
│   └── test_fallback_integration.py
│
└── README.md
```

### 9.1 Module responsibilities

| Module | Responsibility |
|---|---|
| `pid_controller.py` | Generic PID/PI mathematics, output limits, anti-windup |
| `zone_controller.py` | Manages three PID loops for one zone |
| `actuator_mapper.py` | Converts continuous PID outputs into actuator requests |
| `actuator_arbiter.py` | Resolves shared fan and actuator conflicts |
| `supervisor.py` | Manages per-zone MARL/PID state transitions |
| `fallback_manager.py` | Coordinates fallback controllers across zones |
| `crop_config_loader.py` | Loads and validates crop requirements |
| `sensor_interface.py` | Normalizes sensor and Trust Network measurements |

Each module should have a clearly defined interface and should be independently testable.

---

## 10. Runtime Data Flow

For every control step:

1. The environment generates new observations.
2. The sensor interface obtains raw or estimated measurements.
3. Measurement validity and trust are checked.
4. The supervisor evaluates each MARL agent's health.
5. The supervisor selects MARL, PID, recovery validation, or safe mode for each zone.
6. If PID is selected, the zone controller loads the crop target profile.
7. Three PID loops calculate their control outputs.
8. The actuator mapper converts outputs into actuator requests.
9. The actuator arbiter resolves fan conflicts.
10. The final action is validated.
11. The action is sent to the greenhouse environment.
12. The system records the result for monitoring and evaluation.

The supervisor must own the final per-zone controller selection.

---

## 11. Testing Strategy

Testing should happen incrementally.

### 11.1 Unit testing

| Component | Tests |
|---|---|
| PID | Error direction, output limits, anti-windup, reset |
| Crop loader | Valid profiles, missing profiles, invalid ranges |
| Sensor interface | Valid readings, stale readings, missing readings |
| Zone controller | Independent temperature, humidity, moisture loops |
| Actuator mapper | Binary conversion, hysteresis, deadbands |
| Actuator arbiter | Conflicting fan requests and ambient conditions |
| Supervisor | State transitions and recovery validation |
| Fallback manager | One-zone and multiple-zone failures |

### 11.2 Integration testing

Test the following scenarios:

1. All MARL agents work normally.
2. Only `zone_1` fails.
3. Two agents fail.
4. All agents fail.
5. An agent produces an invalid action.
6. A sensor becomes stuck.
7. A sensor develops bias or drift.
8. The Trust Network reports low confidence.
9. Both raw and estimated measurements become unavailable.
10. A crop configuration is missing.
11. A growth stage changes during operation.
12. Temperature and humidity controllers request conflicting fan behavior.
13. PID output reaches its limits.
14. PID fallback activates.
15. MARL automatically recovers.
16. Handover causes no uncontrolled actuator jump.
17. Ambient conditions make ventilation counterproductive.

### 11.3 Evaluation metrics

Record:

- Temperature tracking error.
- Humidity tracking error.
- Soil moisture tracking error.
- Time outside crop target ranges.
- Overshoot.
- Settling time.
- Time to enter fallback.
- Time to recover.
- Unsafe action count.
- Actuator switching frequency.
- Time spent in fallback.
- Recovery success rate.
- Safe-mode events.

Compare MARL and PID fallback under equivalent simulation conditions, including disturbances and fault scenarios.

---

## 12. Implementation Roadmap

Implementation should be incremental. Each phase must be tested before proceeding.

### Phase 0 — Environment and interface validation

- Inspect the existing `env/env.py`.
- Confirm units of T/H/M.
- Confirm the actual simulation timestep.
- Review actuator behavior.
- Validate the physical equations.
- Inspect the Trust Network output schema.
- Define normalized measurement and action interfaces.

**Completion criteria:** Inputs, outputs, units, and plant assumptions are documented.

### Phase 1 — Generic PID controller

- Implement a reusable PID/PI class.
- Add output limits.
- Add anti-windup.
- Add timestep handling.
- Add reset functionality.
- Write unit tests.

**Completion criteria:** The standalone controller passes mathematical and behavioral tests.

### Phase 2 — Crop configuration

- Define the crop dataset schema.
- Implement the crop configuration loader.
- Add configuration validation.
- Support different crop assignments per zone.
- Handle growth-stage changes.

**Completion criteria:** Every zone can load validated crop targets.

### Phase 3 — Zone controller

- Compose temperature, humidity, and moisture loops.
- Implement target-band behavior.
- Support runtime target changes.
- Ensure independent controller states.

**Completion criteria:** Each zone can independently calculate its three control outputs.

### Phase 4 — Actuator mapping and arbitration

- Implement continuous-to-binary conversion.
- Add hysteresis.
- Add deadbands.
- Implement shared fan arbitration.
- Add final action validation.

**Completion criteria:** The controller produces valid, deterministic environment actions.

### Phase 5 — Trust Network integration

- Implement the sensor adapter.
- Normalize Trust Network records.
- Validate measurements.
- Add configurable trust policies.
- Define behavior when no valid measurement exists.

**Completion criteria:** PID consumes validated sensor values without depending on Trust Network internals.

### Phase 6 — Supervisor

- Implement the per-zone state machine.
- Add failure detection.
- Add PID fallback transitions.
- Add recovery validation.
- Implement bumpless handover.

**Completion criteria:** Each zone can independently switch between MARL and PID.

### Phase 7 — Environment integration

- Integrate the fallback manager with the simulation runner.
- Preserve the existing MARL execution path.
- Validate final actions.
- Run end-to-end fault scenarios.

**Completion criteria:** MARL remains primary and PID fallback works independently for each zone.

### Phase 8 — Benchmarking and tuning

- Validate the greenhouse physics.
- Tune PID gains through repeatable experiments.
- Test disturbances and sensor faults.
- Compare PID and MARL performance.
- Document results and limitations.

**Completion criteria:** Simulation results support the chosen control configuration.

---

## 13. Open Engineering Questions

These details must be confirmed before final implementation.

| Question | Importance |
|---|---|
| What exactly does `M` represent? | Soil moisture targets require matching measurement units. |
| What is the actual control timestep? | PID integral and derivative calculations depend on elapsed time. |
| Are ambient conditions global or zone-specific? | Fan arbitration depends on ambient conditions. |
| What are the actual actuator delays? | PID tuning and switching behavior depend on them. |
| What fields does the Trust Network return? | The sensor adapter must match its real interface. |
| How are MARL agent failures detected? | The supervisor needs reliable failure signals. |
| What is the safe state for each actuator? | Safe behavior must be explicitly defined. |
| Which crop dataset will be used? | Target ranges and units must be traceable. |

Missing information must be identified, not invented.

---

## 14. Definition of Done

The PID fallback system is considered implemented in simulation when:

- Each zone has three independent control loops.
- Different crops can be assigned to different zones.
- Crop targets are loaded from validated configuration data.
- Sensor measurements are validated before use.
- Continuous PID outputs are mapped to binary actuator actions.
- Shared fan arbitration is deterministic.
- One failed MARL agent triggers fallback only for its own zone.
- Multiple zones can enter fallback simultaneously.
- MARL recovery is automatic but validated.
- Controller handover does not cause uncontrolled actuator changes.
- All actions conform to the existing environment action space.
- Unit and integration tests pass.
- Fault scenarios and limitations are documented.

Passing simulation tests does not certify real-world greenhouse safety.

---

## 15. Reference Documents

This design is based on our project discussions and the following supplied documents:

- `TRUST_NETWORK_SPEC_DIFFERENCES.docx`
- `TRUST_NETWORK_IMPLEMENTATION_GUIDE.docx`
- The latest `env/env.py` implementation shared during the design discussion.

The Trust Network documents describe a separate sensor estimation and trust-scoring prototype. They do not establish that PID integration, MARL switching, hardware interlocks, or complete fault tolerance have already been implemented.

---

## Final Design Principle

**MARL remains the primary controller. PID is an independent, crop-aware fallback for each greenhouse zone.**

A supervisor monitors agent health and sensor reliability, switches individual zones to PID when necessary, and automatically restores MARL after validated recovery.

The system must remain modular, configurable, independently testable, and compatible with the existing greenhouse simulation.