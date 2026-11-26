# Testing & Verification Guide

## 1. Test Environment

- **Python**: `3.14.2` (verified in both project `.venv` and an isolated clean virtual environment)
- **Dependencies** ([requirements.txt](../requirements.txt)): `numpy==2.5.5`, `gymnasium==1.4.0`, `pettingzoo==1.27.0`, `pytest==9.2.0` (plus `pytest-cov==7.1.0` / `coverage==7.17.5` for coverage audits)
- **No temporary stubs or undocumented setup**: All tests run directly from the repository root with `pytest`.

---

## 2. Test Commands & Categories

| Suite | Command | Scope | Measured Result |
|---|---|---|---|
| Full Clean-Venv Regression | `clean_venv\Scripts\python.exe -m pytest -v -W default` | All 7 test files | **155 passed in 8.10s**, 0 warnings |
| Full Coverage Run | `.venv\Scripts\python.exe -m pytest --cov=control --cov=env --cov=trust_network --cov-branch --cov-report=term-missing --cov-report=html:htmlcov -q` | Line + branch coverage (`htmlcov/`) | **155 passed in 9.71s**, **92%** total |
| PID Unit Tests | `.venv\Scripts\python.exe -m pytest control/tests/test_pid_unit.py -q` | `PID-01`..`PID-12`, windup, variable `dt`, lifecycle | **54 passed in 0.23s** |
| Zone & Actuator Tests | `.venv\Scripts\python.exe -m pytest control/tests/test_zone_and_actuator.py -q` | `ZONE-01`..`06`, `ACT-01`..`06`, irrigation handover | **27 passed in 0.28s** |
| Trust Network Tests | `.venv\Scripts\python.exe -m pytest trust_network/test_trust_network.py -q` | `TRUST-01`..`07`, `NaN`/`Inf` isolation, recovery | **26 passed in 4.10s** |
| Arbiter & Supervisor Tests | `.venv\Scripts\python.exe -m pytest control/tests/test_arbiter_and_supervisor.py -q` | `FB-07/09/10`, `NaN` interlocks, metrics lifecycle | **17 passed in 0.20s** |
| Integration & Acceptance | `.venv\Scripts\python.exe -m pytest control/tests/test_integration_scenarios.py -q` | 3-zone crash/safe-mode/recovery, 1200-step disturbance, reproducibility | **12 passed in 2.49s** |
| Crop & Sensor Tests | `.venv\Scripts\python.exe -m pytest control/tests/test_crop_and_sensor.py -q` | Crop schema/units, `SensorInterface` & `TrustNetwork` bridge | **13 tests** (verified in full suite) |
| Environment Physics Tests | `.venv\Scripts\python.exe -m pytest env/test_env.py -q` | PettingZoo API, Euler stability, step response | **6 tests** (verified in full suite) |

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

Overall measured coverage: **92%** (`3,779` statements, `204` missed; `998` branches, `141` partial). HTML report generated in `htmlcov/`.

| Module | Statements | Miss | Branches | BrPart | Cover |
|---|---:|---:|---:|---:|---:|
| `control/__init__.py` | 11 | 0 | 0 | 0 | **100%** |
| `control/sensor_interface.py` | 107 | 2 | 30 | 0 | **99%** |
| `control/types.py` | 98 | 1 | 0 | 0 | **99%** |
| `env/env.py` | 109 | 2 | 18 | 2 | **97%** |
| `control/metrics.py` | 82 | 2 | 30 | 3 | **96%** |
| `control/pid_controller.py` | 231 | 6 | 80 | 6 | **96%** |
| `control/zone_controller.py` | 75 | 2 | 24 | 3 | **95%** |
| `trust_network/harness.py` | 67 | 2 | 32 | 3 | **95%** |
| `trust_network/trust_network.py` | 255 | 13 | 66 | 11 | **93%** |
| `control/fallback_manager.py` | 72 | 4 | 12 | 1 | **92%** |
| `control/actuator_mapper.py` | 189 | 22 | 78 | 16 | **86%** |
| `control/actuator_arbiter.py` | 159 | 25 | 68 | 13 | **81%** |
| `control/supervisor.py` | 502 | 89 | 246 | 53 | **80%** |
| `control/crop_config_loader.py` | 150 | 27 | 76 | 22 | **77%** |

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
