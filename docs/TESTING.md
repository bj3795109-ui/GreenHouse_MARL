# Testing & Verification Guide

## 1. Test Environment

- **Python**: `3.14.2` (Phase 6 baseline: project `.venv` and an isolated clean virtual environment on Windows) and `3.12.3` (2026-11-27 checkpoint, Linux; numpy 2.5.4 against the pinned 2.5.5, gymnasium 1.4.0, pettingzoo 1.27.0, pytest 9.2.0). These are separate verified environments, not one.
- **Dependencies** ([requirements.txt](../requirements.txt)): `numpy==2.5.5`, `gymnasium==1.4.0`, `pettingzoo==1.27.0`, `pytest==9.2.0` (plus `pytest-cov==7.1.0` / `coverage==7.17.5` for coverage audits)
- **No temporary stubs or undocumented setup**: All tests run directly from the repository root with `pytest`.

---

## 2. Test Commands & Current Suite

Checkpoint measured on Linux / Python 3.12.3 (2026-11-27). Counts change as tests are added: run `python -m pytest --collect-only -q` for the live number rather than trusting this table.

| Command | Scope |
|---|---|
| `python -m pytest -q` | Whole repository: **352 passed** (two consecutive runs, ~10 s without coverage) |
| `python -m pytest --cov=control --cov=env --cov=trust_network --cov=fault_injection --cov-branch --cov-report=term-missing -q` | Line + branch coverage (**93%** total at the 345-test checkpoint, see section 4) |

| Area | Files | Tests |
|---|---|---:|
| `control/tests/` | arbiter_and_supervisor 17, crop_and_sensor 13, integration_scenarios 12, pid_deadband 7, pid_numeric_policy 15, pid_unit 54, zone_and_actuator 27 | 145 |
| `env/` | test_env 6 | 6 |
| `trust_network/` | test_trust_network 26, test_dropout 4 | 30 |
| `fault_injection/tests/` | actuator_config 11, actuator_injector 20, adapter 34, fault_composition 9, fault_hardening 7, fault_injector 15, fault_lifecycle 14, fault_logger 8, fault_scheduler 9, lifecycle_and_config_hardening 24, scenario 13, step_validation 7 | 171 |

**What these tests do and do not show.** They show component behaviour and, for `control/`, closed-loop fallback mechanics with the legacy harness faults. They do **not** show end-to-end fault tolerance with the new `fault_injection/` framework: it is not yet wired into the environment, Trust Network or controllers.

---

## 3. Acceptance Scenarios Covered

1. **Three-zone healthy baseline (`A`)** & **Single-zone MARL failure (`B`)**: `test_critical_three_zone_acceptance_zone2_crash_safe_mode_and_recovery` & `test_phase6_full_acceptance_compound_sensor_pid_marl_fault_and_recovery` verify Zone 2 switches to `PID_FALLBACK` while Zones 1 and 3 stay in `MARL_ACTIVE`.
2. **Nested PID failure (`C`)**: Zone 2 `ZoneController.step()` exception forces `SAFE_MODE` `(0, 0, 0, 0)` while Zones 1 and 3 continue uninterrupted.
3. **Sensor corruption (`D`)**: `NaN` `T` and `Inf` `H` in Zone 2 are isolated by `TrustNetwork` and `SensorInterface`, keeping Zones 1 and 3 finite and operational, and recovering trust via EMA once valid readings resume.
4. **Permitted recovery (`E`)**: Verified `SAFE_MODE -> PID_FALLBACK -> RECOVERY_PENDING -> MARL_ACTIVE` without bypassing intermediate states or premature handover.
5. **Extended closed-loop disturbance (`F`)**: `test_phase6_extended_operation_disturbances_and_metrics` runs 1,200 steps with outdoor ambient shifts (`24°C/45%` -> `19°C/58%`), `zone_1` `T` drift, and `zone_2` MARL outage, verifying finite `EpisodeEvaluator` tracking errors (`RMSE`, `MAE`), switching frequency, and `water_max_pulse_steps <= 2`.
6. **Reproducibility (`G`)**: `test_phase6_exact_reproducibility_with_identical_seeds` confirms identical trajectories, modes, actions, and summaries across two independent runs with `seed=101`.

---

## 4. Measured Code Coverage Summary

Overall measured coverage: **93%** (`5,042` statements, `229` missed; `1,244` branches, `162` partial) over `control/`, `env/`, `trust_network/` and `fault_injection/`, including their test files (2026-11-27, Linux / Python 3.12.3), measured at the 345-test checkpoint, before `test_step_validation.py` was added; re-run the coverage command for current figures.

| Module | Statements | Miss | Branches | BrPart | Cover |
|---|---:|---:|---:|---:|---:|
| `control/__init__.py` | 11 | 0 | 0 | 0 | **100%** |
| `control/sensor_interface.py` | 107 | 2 | 30 | 0 | **99%** |
| `control/types.py` | 99 | 1 | 0 | 0 | **99%** |
| `env/env.py` | 109 | 2 | 18 | 2 | **97%** |
| `control/pid_controller.py` | 239 | 5 | 82 | 6 | **97%** |
| `control/metrics.py` | 82 | 2 | 30 | 3 | **96%** |
| `control/zone_controller.py` | 75 | 2 | 24 | 3 | **95%** |
| `trust_network/harness.py` | 72 | 2 | 32 | 3 | **95%** |
| `trust_network/trust_network.py` | 280 | 19 | 76 | 13 | **91%** |
| `control/fallback_manager.py` | 72 | 4 | 12 | 1 | **92%** |
| `control/actuator_mapper.py` | 189 | 22 | 78 | 16 | **86%** |
| `control/actuator_arbiter.py` | 159 | 25 | 68 | 13 | **81%** |
| `control/supervisor.py` | 502 | 89 | 246 | 53 | **80%** |
| `control/crop_config_loader.py` | 150 | 27 | 76 | 22 | **77%** |
| `fault_injection/scenario.py` | 37 | 0 | 20 | 0 | **100%** |
| `fault_injection/fault_logger.py` | 29 | 0 | 2 | 0 | **100%** |
| `fault_injection/fault_lifecycle.py` | 44 | 1 | 24 | 1 | **97%** |
| `fault_injection/adapter.py` | 44 | 1 | 12 | 1 | **96%** |
| `fault_injection/fault_scheduler.py` | 32 | 1 | 14 | 1 | **96%** |
| `fault_injection/fault_config.py` | 76 | 6 | 42 | 6 | **90%** |
| `fault_injection/fault_injector.py` | 109 | 10 | 80 | 10 | **89%** |

### Uncovered Critical / Notable Branches
- `control/supervisor.py`: `SupervisorConfig.__post_init__` invalid-field error branches (`60–105`), `transition_to` non-`ControlMode` guard (`207–211`), malformed container checks (`260–310`), `N_confirm == 1` immediate confirmation (`645–650`), and PID-failure-inside-`RECOVERY_PENDING`-abort branches (`693–725`).
- `control/actuator_arbiter.py`: neutral/deadband conflict branches (`217–231`, `238–247`) when `T` vs. `H` severity margin flips near `arbiter_hysteresis`.
- `control/actuator_mapper.py`: `validate_actuator_config` invalid-field error branches (`66–91`).
- `control/crop_config_loader.py`: malformed JSON schema error paths (`156–233`) and `degF -> degC` conversion branch (`76`).

---

## 5. Reproducibility & Nondeterminism Notes

- **Deterministic paths**: When `GreenhouseEnv.reset(seed=...)`, `NoisySensors(np.random.default_rng(seed))`, and `env.action_space(z).seed(seed)` (added in `harness.py`) are seeded identically, closed-loop execution is deterministic.
- **Sources of nondeterminism**: Calling `action_space(z).sample()` in custom scripts without seeding `env.action_space`, or asynchronous hardware/wall-clock timestamps (`current_time=time.time()`) on physical deployment.
- **Important caveat**: All MARL integration tests evaluate the `_simple_healthy_marl` surrogate, **not** a trained MAPPO neural policy.
