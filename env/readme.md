# Greenhouse MARL Environment — Physics-Only Pass

## Status: passes all checks below. Ready for the next build-order step.

## Scope of this pass
Implements V1.1 Fault-Tolerance Spec §2.1–2.4 only:
- Thermal dynamics (§2.1)
- Humidity dynamics (§2.2)
- Soil moisture dynamics (§2.4)

**Deliberately NOT included yet** (per V1.1 §8 build order — added after this
core physics loop is validated, not before):
- Light dynamics (§2.5) and the 5th actuator (`u_light`)
- CO2 dynamics (§2.3)
- Trust network (§4) — no raw/estimate/trust triplet yet, `step()` returns
  raw state only
- Reward shaping (§7) — `step()` returns `0.0` for every agent, always
- Fault injection, safety interlocks (§0)

## What's been verified
Run `pytest test_env.py -v` — 6 tests, all passing:
1. **PettingZoo `ParallelEnv` API compliance** (`parallel_api_test`)
2. **V1.3 Numerical Stability Verification** — per-zone Euler stability
   bounds computed and checked at construction time for T, H, and M
   independently; env raises `ValueError` rather than silently running
   unstable if `dt` is too large for the given parameters
3. **Zero-action decay** — all three zones drift toward ambient/zero with
   no actuation
4. **Single-heater step response** — actuated zone rises well above
   ambient; neighbors rise less, confirming the "strong local behavior,
   weak physical coupling" property the 3-zone design targets
5. **Random-policy sanity check** (20 episodes × 1000 steps, random
   actions) — no NaN/Inf, all values stay in physically plausible
   ranges, temperature genuinely moves rather than sitting static

## Known placeholder-scale issue, already fixed
Original `C_humidity = 1.0` gave humidity a ~9s settling time versus
temperature's ~3000s+ — a 2-order-of-magnitude mismatch from a
placeholder guess, not real physics. Bumped to `C_humidity = 40.0` so
humidity settles on the order of minutes instead of seconds. **Still a
guess** — every constant in `PlaceholderParams` needs real calibration
per V1.1 §11 before any output here is quantitatively trustworthy.

## Files
- `env.py` — the environment (`GreenhouseEnv`, `PlaceholderParams`,
  stability-bound helpers)
- `test_env.py` — the test suite described above

## Next steps (not started)
Per V1.1 §8 build order:
1. Implement §4 trust network as a separate, unit-testable module
2. Fault-injection framework (single-fault, then multi-fault)
3. PID baseline
4. Single-agent RL baseline
5. MAPPO/CTDE training on the full multi-zone environment

Also still open: adding light (§2.5) and CO2 (§2.3) dynamics, both
deferred from this pass on purpose.