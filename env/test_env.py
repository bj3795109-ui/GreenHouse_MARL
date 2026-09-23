"""
Test suite for the V1.1 §2.1-2.4 physics-only environment.

Covers:
  - PettingZoo API compliance
  - V1.3 Numerical Stability Verification checklist items:
      * per-zone stability bounds hold for T, H, and M
      * zero-action decay test
      * single-heater step-response test
  - Random-policy sanity check (§8 build order, step 2 gate):
      * no NaN/Inf over many episodes
      * values stay in physically plausible ranges
      * the simulation actually moves (isn't static)

Run with: pytest test_env.py -v
"""

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from env import (
    GreenhouseEnv,
    PlaceholderParams,
    ZONE_IDS,
    stability_bound_thermal,
    stability_bound_humidity,
    stability_bound_moisture,
)


def snap(env):
    return {z: (env.state[z]["T"], env.state[z]["H"], env.state[z]["M"]) for z in ZONE_IDS}


# ---------------------------------------------------------------------------
# API compliance
# ---------------------------------------------------------------------------

def test_parallel_api_compliance():
    env = GreenhouseEnv(max_steps=100)
    parallel_api_test(env, num_cycles=100)


# ---------------------------------------------------------------------------
# V1.3 Numerical Stability Verification
# ---------------------------------------------------------------------------

def test_stability_bounds_hold_for_default_params():
    """dt must be strictly less than every zone's bound for T, H, and M."""
    p = PlaceholderParams()
    for zid in ZONE_IDS:
        assert p.dt < stability_bound_thermal(p, zid), f"{zid} thermal bound violated"
        assert p.dt < stability_bound_humidity(p, zid), f"{zid} humidity bound violated"
    assert p.dt < stability_bound_moisture(p), "moisture bound violated"


def test_env_construction_raises_if_dt_too_large():
    """Env should refuse to construct rather than silently run unstable."""
    bad_params = PlaceholderParams()
    bad_params.dt = 10000.0  # absurdly large, must exceed some bound
    with pytest.raises(ValueError):
        GreenhouseEnv(params=bad_params)


# ---------------------------------------------------------------------------
# Zero-action decay — everything should settle toward ambient / zero
# ---------------------------------------------------------------------------

def test_zero_action_decay_toward_ambient():
    env = GreenhouseEnv(max_steps=3000)
    env.reset(seed=1)
    actions = {zid: [0, 0, 0, 0] for zid in list(env.agents)}
    for _ in range(3000):
        env.step(actions)
        if not env.agents:
            break

    p = PlaceholderParams()
    for zid in ZONE_IDS:
        s = env.state[zid]
        # Not required to fully converge in 3000s given the thermal time
        # constant (~3000s+), but should be moving clearly toward ambient,
        # not away from it or diverging.
        assert abs(s["T"] - p.T_ambient) < 5.0, f"{zid} T did not approach ambient: {s['T']}"
        assert abs(s["H"] - p.H_ambient) < 2.0, f"{zid} H did not approach ambient: {s['H']}"
        assert s["M"] < 1.0, f"{zid} M did not decay to ~0 with no irrigation: {s['M']}"


# ---------------------------------------------------------------------------
# Single-heater step response — actuated zone rises more than neighbors
# ---------------------------------------------------------------------------

def test_heater_step_response_shape():
    env = GreenhouseEnv(max_steps=1500)
    env.reset(seed=1)
    for _ in range(1500):
        actions = {"zone_1": [1, 0, 0, 0], "zone_2": [0, 0, 0, 0], "zone_3": [0, 0, 0, 0]}
        env.step(actions)
        if not env.agents:
            break

    s = env.state
    # Actuated zone should have risen well above ambient.
    assert s["zone_1"]["T"] > 30.0, f"heated zone did not rise as expected: {s['zone_1']['T']}"
    # Neighbors should show SOME coupled rise, but much less than the actuated zone —
    # this is the "strong local, weak coupling" behavior the physical design targets.
    rise_1 = s["zone_1"]["T"] - PlaceholderParams.T_ambient
    rise_2 = s["zone_2"]["T"] - PlaceholderParams.T_ambient
    rise_3 = s["zone_3"]["T"] - PlaceholderParams.T_ambient
    assert 0 < rise_2 < rise_1, "zone_2 should warm some, but far less than zone_1"
    assert 0 < rise_3 < rise_1, "zone_3 should warm some, but far less than zone_1"


# ---------------------------------------------------------------------------
# Random-policy sanity check (§8 build order, step 2 gate)
# ---------------------------------------------------------------------------

def test_random_policy_no_nan_or_blowup():
    env = GreenhouseEnv(max_steps=1000)
    n_episodes = 20
    all_T = []

    for ep in range(n_episodes):
        env.reset(seed=ep)
        for _ in range(1000):
            actions = {zid: env.action_space(zid).sample() for zid in env.agents}
            env.step(actions)
            for zid in ZONE_IDS:
                T, H, M = env.state[zid]["T"], env.state[zid]["H"], env.state[zid]["M"]
                assert np.isfinite(T) and np.isfinite(H) and np.isfinite(M), (
                    f"non-finite value at episode {ep}, zone {zid}: T={T} H={H} M={M}"
                )
                # Loose physical plausibility bounds — not tight calibration
                # checks, just "hasn't blown up to something absurd."
                assert -20.0 < T < 100.0, f"T out of plausible range: {T}"
                assert 0.0 <= H <= 100.0, f"H out of bounds: {H}"
                assert 0.0 <= M <= 100.0, f"M out of bounds: {M}"
                all_T.append(T)
            if not env.agents:
                break

    # Confirm the sim is actually dynamic, not sitting static the whole time.
    assert np.std(all_T) > 0.1, "temperature barely moved across 20 random-policy episodes"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))