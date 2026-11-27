# GreenHouse MARL — Three-Zone Fault-Tolerant Greenhouse Control

A three-zone greenhouse environmental control testbed combining a PettingZoo `ParallelEnv` physics simulator, a model-based sensor `TrustNetwork`, crop-aware per-zone PID fallback controllers, ambient-aware actuator arbitration, and a per-zone MARL/PID supervisory state machine.

> **Implementation status (Phase 6 Baseline):** The physics simulator, Trust Network, PID fallback loops, actuator mapper/arbiter, `SensorInterface`, and per-zone `ZoneSupervisor` / `FallbackManager` are implemented and verified (see [docs/TESTING.md](docs/TESTING.md) for the current suite and coverage snapshot). **A trained MAPPO policy is not implemented yet**; integration tests use a deterministic rule-based surrogate (`_simple_healthy_marl`) to validate MARL-to-PID fallback, `SAFE_MODE`, and recovery mechanics.

> **Checkpoint status:** the fault-injection framework (`fault_injection/`) is implemented and tested as a **standalone subsystem**. It is **not yet integrated** with `GreenhouseEnv`, the Trust Network or the controllers (the Trust Network harness still uses its legacy minimal `Fault`/`NoisySensors`). The integration wrapper, fault-aware closed-loop experiments and MAPPO training are **not implemented**.

> **Current model scope:** three controlled variables (temperature `T`, humidity `H`, soil moisture `M`) and four binary actuators per zone (`heat`, `fan`, `mist`, `water`). Light (`L`) and CO₂ are **deferred, not dropped** (see Future Roadmap).

---

## Problem Statement

Multi-agent reinforcement learning (MARL) controllers for coupled multi-zone greenhouses can fail during deployment due to policy crashes, malformed/NaN actions, sensor corruption, or persistent boundary divergence. This repository establishes a **validated, fault-isolated supervisory control and PID fallback architecture** so that:

- A single zone's agent crash or unsafe action switches **only that zone** to local PID fallback while healthy zones continue undisturbed.
- Corrupted (`NaN`/`Inf`/stuck/drifting) sensor readings are isolated by the `TrustNetwork` and `SensorInterface` before reaching control loops or contaminating neighboring zones.
- Simultaneous controller or sensor failure transitions only the affected zone to `SAFE_MODE` `(0, 0, 0, 0)`.

---

## System Architecture

```text
GreenhouseEnv (T, H, M) ─▶ RawReading ─▶ TrustNetwork ─▶ SensorInterface ─▶ ZoneSupervisor
                                                                                │
       ┌────────────────────────────────────────────────────────────────────────┴────────────────────────────┐
       ▼ (MARL_ACTIVE)                                                                                       ▼ (PID_FALLBACK / RECOVERY_PENDING)
  External MARL Action ──▶ validate_marl_action & safety checks                                        ZoneController (T, H, M PIDs)
       │                                                                                                     │
       ▼                                                                                                     ▼
  Handover Dwell Protection & Shadow Tracking                                                  ActuatorMapper ─▶ ActuatorArbiter
       │                                                                                                     │
       └──────────────────────────────────────────▶ Software Interlocks ◀────────────────────────────────────┘
                                                             │
                                                             ▼
                                                MultiBinary(4) [heat, fan, mist, water]
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed component diagrams and data flow.

---

## Key Features

- **Three coupled greenhouse zones** (`zone_1`, `zone_2`, `zone_3`) with thermal (`T`), relative humidity (`H`), and soil moisture (`M`) dynamics.
- **Model-based `TrustNetwork`** enforcing the raw-in / trust-gates-out invariant, isolating `NaN`/`Inf`/missing readings before fusion and neighbor coupling.
- **9 independent PID/PI control loops** (3 variables × 3 zones) with deadbands, filtered derivative-on-measurement, back-calculation anti-windup, variable `dt` support, unidirectional moisture integral unwinding, and bumpless shadow tracking.
- **Crop-aware configuration** decoupling horticultural target bands (`crops_dataset.json`, `zone_assignments.json`) from PID tuning (`pid_tuning.json`).
- **Actuator mapping & ambient-aware arbitration** with software PWM, minimum on/off dwell timers, irrigation pulse/soak limits, and shared-fan conflict resolution.
- **Per-zone `ZoneSupervisor` FSM** (`MARL_ACTIVE`, `PID_FALLBACK`, `RECOVERY_PENDING`, `SAFE_MODE`) with explicit `VALID_TRANSITIONS` validation, `NaN`-safe software interlocks, and exception isolation.

---

## Technology Stack

- **Python** `3.14.2` (author's Windows environments); the 2026-11-27 checkpoint was re-run on Linux / Python 3.12.3 (see [docs/TESTING.md](docs/TESTING.md))
- **Simulation & RL Interfaces**: `pettingzoo >= 1.27.0`, `gymnasium >= 1.4.0`
- **Numerics**: `numpy >= 2.5.5`
- **Testing & Coverage**: `pytest >= 9.2.0`, `pytest-cov >= 7.1.0`

---

## Repository Structure

```text
GreenHouse_MARL/
├── env/
│   ├── env.py                        # Three-zone GreenhouseEnv (PettingZoo ParallelEnv) & PlaceholderParams
│   ├── test_env.py                   # PettingZoo API, Euler stability, step-response, random-policy tests
│   └── readme.md                     # Physics-pass scope and calibration notes
├── trust_network/
│   ├── trust_network.py              # RawReading/CorrectedValue guards, estimators, HealthyResiduals, TrustNetwork
│   ├── harness.py                    # NoisySensors, legacy minimal Fault (to be replaced by fault_injection/), seeded rollout(), bootstrap_healthy()
│   └── test_trust_network.py         # Trust invariant, fault detection/isolation, NaN/Inf, recovery tests
├── fault_injection/                  # Standalone fault framework (NOT yet wired into env/control)
│   ├── fault_config.py               # FaultConfig: validated single-fault definition, is_active_at() timing predicate
│   ├── fault_types.py                # Sensor/actuator fault enums
│   ├── fault_injector.py             # Sensor/actuator transforms, per-(fault,zone,target,step) deterministic RNG
│   ├── fault_scheduler.py            # Active-fault selection; validates the scenario on construction
│   ├── fault_lifecycle.py            # One update per step: activation/deactivation events
│   ├── fault_logger.py               # Ground-truth event log
│   ├── scenario.py                   # validate_scenario(): ids, start_step, categories, order-dependent overlaps
│   ├── adapter.py                    # Greenhouse <-> injector naming (checked against control.types at import)
│   └── tests/                        # Unit, composition, lifecycle, config-hardening, scenario and adapter tests
├── control/
│   ├── __init__.py                   # Public control package exports
│   ├── types.py                      # ControlMode, MeasurementQuality, SensorMeasurement, CropBounds, PIDGains, etc.
│   ├── pid_controller.py             # Discrete PIDController with deadband, anti-windup, variable dt, lifecycle
│   ├── zone_controller.py            # Per-zone 3-loop PID + ActuatorMapper + ActuatorArbiter composition
│   ├── actuator_mapper.py            # PWM, DwellTracker, irrigation pulse/soak, action/output validation
│   ├── actuator_arbiter.py           # Ambient-aware shared-fan arbitration and mutual exclusion
│   ├── crop_config_loader.py         # CropConfigLoader and load_pid_tuning()
│   ├── sensor_interface.py           # TrustRecord/observation validation and quality tiering
│   ├── supervisor.py                 # ZoneSupervisor FSM, VALID_TRANSITIONS, validate_marl_action, interlocks
│   ├── fallback_manager.py           # Multi-zone orchestrator with per-zone exception isolation
│   ├── metrics.py                    # EpisodeEvaluator tracking 13 control & fallback metrics
│   ├── pdireadme.md                  # Original PID architecture design specification
│   ├── configs/
│   │   ├── crops_dataset.json        # Crop/stage target envelopes (tomato, lettuce, cucumber)
│   │   ├── pid_tuning.json           # Baseline PID/PI gains (separate from crop targets)
│   │   └── zone_assignments.json     # Per-zone crop and growth-stage assignments
│   └── tests/
│       ├── test_pid_unit.py          # PID-01..PID-12 and lifecycle/validation unit tests
│       ├── test_zone_and_actuator.py # ZONE-01..06, ACT-01..06, irrigation handover tests
│       ├── test_crop_and_sensor.py   # Crop config, unit checks, SensorInterface & TrustNetwork integration
│       ├── test_arbiter_and_supervisor.py # FB-07/09/10, NaN interlocks, metrics lifecycle tests
│       └── test_integration_scenarios.py  # Closed-loop 3-zone acceptance, stress, reproducibility tests
├── docs/
│   ├── ARCHITECTURE.md               # End-to-end architecture & Mermaid diagrams
│   ├── CONTROL_SYSTEM.md             # PID math, mapping, arbitration, interlocks, variable dt
│   ├── FAULT_TOLERANCE.md            # Trust isolation, FSM, SAFE_MODE, recovery rules
│   ├── TESTING.md                    # Test environment, commands, coverage, reproducibility
│   ├── MAPPO_READINESS.md            # CTDE verification & MAPPO implementation checklist
│   ├── PHASE_6_COMPLETION_REPORT.md  # Final Phase 6 audit report & status table
│   └── *.docx                        # Original project specification documents
├── requirements.txt
└── README.md
```

---

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional for coverage reports:
pip install pytest-cov
```

---

## Configuration

- **Crop targets** ([control/configs/crops_dataset.json](control/configs/crops_dataset.json)): Defines `min_acceptable`, `optimal_min`, `optimal_max`, `max_acceptable`, and units (`degC`, `RH_percent`, `relative_percent`) per crop and growth stage.
- **Zone assignments** ([control/configs/zone_assignments.json](control/configs/zone_assignments.json)): Maps `zone_1`, `zone_2`, `zone_3` to `(crop_id, growth_stage)`.
- **PID tuning** ([control/configs/pid_tuning.json](control/configs/pid_tuning.json)): Configures `Kp`, `Ki`, `Kd`, `tau_f`, `T_t`, `Ts`, `dt`, `u_min`, `u_max`, `unidirectional_positive` independently from crop profiles.

---

## Running the Simulator & Control Loop

```python
import sys
sys.path[:0] = ["env", "trust_network"]

from env import GreenhouseEnv, PlaceholderParams
from harness import NoisySensors, bootstrap_healthy
from trust_network import AssumedParams, TrustNetwork
from control import (
    AmbientState, CropConfigLoader, FallbackManager, ZONE_IDS, load_pid_tuning
)

params = PlaceholderParams()
healthy = bootstrap_healthy(AssumedParams.from_env(params), n_episodes=10, steps=300)
env = GreenhouseEnv(params=params, max_steps=300)
obs, _ = env.reset(seed=10)

sensors = NoisySensors(rng=__import__("numpy").random.default_rng(10))
net = TrustNetwork(AssumedParams.from_env(params), healthy=healthy)
net.reset(sensors.read(env.state, 0))

loader = CropConfigLoader.from_json_file("control/configs/crops_dataset.json")
loader.assign_zones_from_json_file("control/configs/zone_assignments.json")
fm = FallbackManager(loader, load_pid_tuning("control/configs/pid_tuning.json"))
ambient = AmbientState(params.T_ambient, params.H_ambient)
```

---

## Running Tests

```powershell
# Full regression suite
.venv\Scripts\python.exe -m pytest -v

# Full suite with branch coverage
.venv\Scripts\python.exe -m pytest --cov=control --cov=env --cov=trust_network --cov=fault_injection --cov-branch --cov-report=term-missing --cov-report=html:htmlcov

# Focused suites
.venv\Scripts\python.exe -m pytest control/tests/test_pid_unit.py -v
.venv\Scripts\python.exe -m pytest control/tests/test_zone_and_actuator.py -v
.venv\Scripts\python.exe -m pytest trust_network/test_trust_network.py -v
.venv\Scripts\python.exe -m pytest control/tests/test_arbiter_and_supervisor.py -v
.venv\Scripts\python.exe -m pytest control/tests/test_integration_scenarios.py -v
.venv\Scripts\python.exe -m pytest fault_injection -v
```

---

## Current Implementation Status & Known Limitations

- **Fault injection is not integrated**: `fault_injection/` has no callers outside its own tests. Planned design: `GreenhouseEnv` stays pure physics; a wrapper owns the sensor model (including physical sensor limits, applied after fault injection), sensor/actuator fault injection, the fault lifecycle (one update per step) and evaluator-only ground truth. The wrapper must build `FaultScheduler(..., min_start_step=1)`.
- **No trained MAPPO policy yet**: All MARL integration tests use `_simple_healthy_marl`. See [docs/MAPPO_READINESS.md](docs/MAPPO_READINESS.md).
- **Placeholder physics & tuning**: `PlaceholderParams` in [env/env.py](env/env.py) and gains in [control/configs/pid_tuning.json](control/configs/pid_tuning.json) are provisional; `M` uses a `relative_percent` placeholder scale (`0–100`).
- **Omitted V1.1 variables**: Light (`L`) and CO₂ dynamics are not yet implemented in `env/env.py` or `TrustNetwork`.
- **Trust Network placeholders**: `TrustConfig.conf_act` is a fixed `0.5` placeholder; `TrustRecord` has no timestamp field by default.
- **Host-side safety vs. ESP32 firmware**: Software interlocks and `SAFE_MODE` in `control/supervisor.py` complement—but do not replace—independent ESP32 microcontroller interlocks and local firmware PID on PC communication loss.

---

## Future Roadmap

1. Implement reward shaping, CTDE global state (`env.state()`), and a fault-aware observation wrapper around `GreenhouseEnv` (the env itself stays pure physics).
2. Train and evaluate single-agent RL and CTDE MAPPO baselines against the PID fallback controller.
3. Calibrate `PlaceholderParams`, `AssumedParams`, `conf_act`, and PID gains against hardware measurements.
4. Add light (`L`) dynamics (`u_light` 5th actuator), an integration wrapper that routes `fault_injection/` through the Trust Network and controllers, and V1.2 explainability dashboard integration.
