"""
Tests for the V1.1 §4 trust network.  Run from the project root:
    python -m pytest trust_network/test_trust_network.py -v
"""

import numpy as np
import pytest

from harness import (
    Fault, rollout, bootstrap_healthy, make_net, PlaceholderParams,
)
from trust_network import (
    AssumedParams, CorrectedValue, RawReading, HealthyResiduals, TrustConfig,
    ZONE_IDS, VARS, estimate_cz,
)


@pytest.fixture(scope="module")
def healthy():
    return bootstrap_healthy(AssumedParams.from_env(PlaceholderParams()),
                             n_episodes=10, steps=400)


def mean_trust(log, z, v, start=100):
    return float(np.mean([r[z][v].trust for r in log[start:]]))


# --- invariant: raw-in / trust-gates-out -----------------------------------

def test_cross_zone_estimate_rejects_corrected_values():
    p = AssumedParams.from_env(PlaceholderParams())
    u = dict(heat=False, fan=False, mist=False, water=False)
    good = [RawReading(24.0), RawReading(25.0)]
    estimate_cz("T", 24.5, u, good, p)  # fine
    with pytest.raises(TypeError):
        estimate_cz("T", 24.5, u, [CorrectedValue(24.0), RawReading(25.0)], p)


# --- assumed params are separate from the env's truth ----------------------

def test_calibration_error_only_scales_named_params_and_does_not_touch_env():
    env_p = PlaceholderParams()
    k_before = env_p.k_T
    a = AssumedParams.from_env(env_p, {"k_T": 0.5})
    assert a.k_T == pytest.approx(k_before * 1.5)
    assert a.k_H == pytest.approx(env_p.k_H)
    assert env_p.k_T == k_before
    with pytest.raises(KeyError):
        AssumedParams.from_env(env_p, {"not_a_param": 0.1})


# --- structure --------------------------------------------------------------

def test_moisture_has_no_cross_zone_source():
    log = rollout(make_net(), steps=20, seed=0)
    for z in ZONE_IDS:
        rec = log[-1][z]["M"]
        assert rec.est_cz is None and rec.conf_cz is None
        assert rec.combined == rec.est_act
        assert log[-1][z]["T"].est_cz is not None


# --- healthy residual distribution ------------------------------------------

def test_match_score_shape(healthy):
    for v in VARS:
        arr = healthy._sorted[v]
        assert healthy.match_score(v, float(np.median(arr)), 0.05) == 1.0
        assert healthy.match_score(v, float(arr.max()) * 2, 0.05) == 0.0


def test_literal_ecdf_would_leave_healthy_trust_near_half():
    """Documents WHY match_score divides by tail_ref (spec patch): 1-ECDF is ~uniform."""
    rng = np.random.default_rng(0)
    h = HealthyResiduals({"T": np.abs(rng.normal(0, 0.1, 20000))})
    fresh = np.abs(rng.normal(0, 0.1, 5000))
    literal = np.mean([h.survival("T", x) for x in fresh])
    patched = np.mean([h.match_score("T", x, 0.05) for x in fresh])
    assert 0.4 < literal < 0.6
    assert patched > 0.9


# --- behavior ---------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_healthy_run_keeps_trust_high(healthy, seed):
    log = rollout(make_net(healthy=healthy), steps=500, seed=seed)
    for z in ZONE_IDS:
        for v in VARS:
            assert mean_trust(log, z, v) > 0.85, (seed, z, v)


@pytest.mark.parametrize("seed", [1, 4, 7])
def test_drift_fault_is_detected_and_contained(healthy, seed):
    faults = [Fault("zone_1", "T", "drift", start=200, magnitude=0.01)]
    base = rollout(make_net(healthy=healthy), steps=700, seed=seed)
    flt = rollout(make_net(healthy=healthy), steps=700, seed=seed, faults=faults)

    detect = next((t for t, r in enumerate(flt, 1) if r["zone_1"]["T"].trust < 0.5), None)
    assert detect is not None and detect - 200 < 100, "drift not detected within 100 steps"

    # Containment, measured against each zone's own no-fault baseline (same seed).
    # Thresholds set ~2x the worst value seen over 10 seeds (mean 0.031, max 0.119),
    # not tuned to one seed.
    for z in ("zone_2", "zone_3"):
        d = np.array([abs(b[z]["T"].trust - f[z]["T"].trust) for b, f in zip(base[200:], flt[200:])])
        assert d.mean() < 0.06, f"{z}: mean trust shift {d.mean():.3f}"
        assert d.max() < 0.25, f"{z}: peak trust shift {d.max():.3f}"


def test_stuck_fault_is_detected(healthy):
    faults = [Fault("zone_1", "H", "stuck", start=200)]
    log = rollout(make_net(healthy=healthy), steps=800, seed=7, faults=faults)
    assert log[-1]["zone_1"]["H"].trust < 0.1


def test_coupling_miscalibration_is_tolerated_under_current_placeholders(healthy):
    """k_T/k_H assumed 50% high, healthy distribution fit on the PERFECT model."""
    log = rollout(make_net({"k_T": 0.5, "k_H": 0.5}, healthy=healthy), steps=500, seed=3)
    for z in ZONE_IDS:
        assert mean_trust(log, z, "T") > 0.8


def test_heater_model_error_is_visible_in_trust(healthy):
    """A wrong heater model (Q_heat +40%) must cost trust vs. the correct model."""
    good = rollout(make_net(healthy=healthy), steps=500, seed=3)
    bad = rollout(make_net({"Q_heat": 0.4}, healthy=healthy), steps=500, seed=3)
    g = np.mean([mean_trust(good, z, "T") for z in ZONE_IDS])
    b = np.mean([mean_trust(bad, z, "T") for z in ZONE_IDS])
    assert b < g


# --- belief = trust*raw + (1-trust)*combined --------------------------------

def test_belief_formula_and_no_feedback_into_estimators(healthy):
    net = make_net(healthy=healthy)
    log = rollout(net, steps=200, seed=2)
    for rec in log[::20]:
        for z in ZONE_IDS:
            for v in VARS:
                r = rec[z][v]
                assert r.belief == pytest.approx(r.trust * r.raw + (1 - r.trust) * r.combined)
                assert r.raw_weight == r.trust
    # estimator anchors are the fused estimate, never the belief
    for z in ZONE_IDS:
        for v in VARS:
            assert net.est[z][v] == log[-1][z][v].combined


def test_belief_ignores_a_faulty_sensor(healthy):
    faults = [Fault("zone_1", "T", "drift", start=200, magnitude=0.01)]
    log, truth = rollout(make_net(healthy=healthy), steps=700, seed=7,
                         faults=faults, return_truth=True)
    err = lambda key: np.mean([abs(getattr(r["zone_1"]["T"], key) - t["zone_1"]["T"])
                               for r, t in zip(log[400:], truth[400:])])
    assert err("belief") < 0.1 * err("raw")


def test_belief_is_not_worse_than_combined_when_model_is_wrong(healthy):
    """Q_heat +40%: trust drops, so belief leans on the biased model. It must at
    least not be worse than combined. (Known limit: it can't tell model error
    from sensor error.)"""
    e = {"belief": [], "combined": []}
    for s in range(3):
        lg, t = rollout(make_net({"Q_heat": 0.4}, healthy=healthy), steps=500, seed=s,
                        return_truth=True)
        for k in e:
            e[k].append(np.mean([abs(getattr(r[z]["T"], k) - tt[z]["T"])
                                 for r, tt in zip(lg[100:], t[100:]) for z in ZONE_IDS]))
    assert np.mean(e["belief"]) <= np.mean(e["combined"]) * 1.05


# --- Phase 4: NaN/Inf isolation, multi-fault, recovery (TRUST-01..TRUST-07) -

def test_trust01_valid_readings_across_all_three_zones(healthy):
    log = rollout(make_net(healthy=healthy), steps=120, seed=5)
    last = log[-1]
    for z in ZONE_IDS:
        for v in VARS:
            r = last[z][v]
            assert np.isfinite(r.raw)
            assert np.isfinite(r.combined)
            assert np.isfinite(r.belief)
            assert r.trust > 0.85
            assert r.raw_weight == pytest.approx(r.trust)
            if v in ("T", "H"):
                assert r.conf_cz is not None and np.isfinite(r.conf_cz)


def test_trust02_nan_temperature_in_zone2_rejected_and_isolated(healthy):
    p = AssumedParams.from_env(PlaceholderParams())
    u = dict(heat=False, fan=False, mist=False, water=False)
    with pytest.raises(ValueError):
        estimate_cz("T", 24.5, u, [RawReading(float("nan")), RawReading(25.0)], p)

    faults = [Fault("zone_2", "T", "nan", start=50, end=100)]
    base = rollout(make_net(healthy=healthy), steps=120, seed=2)
    log = rollout(make_net(healthy=healthy), steps=120, seed=2, faults=faults)

    # rollout() logs t=1..steps at indices 0..steps-1, so active [50, 100) is log[49:99]
    for b, r in zip(base[49:99], log[49:99]):
        z2 = r["zone_2"]["T"]
        assert np.isnan(z2.raw)
        assert z2.raw_weight == 0.0
        assert z2.match == 0.0
        assert np.isfinite(z2.est_act)
        assert np.isfinite(z2.combined)
        assert np.isfinite(z2.belief)
        assert z2.belief == pytest.approx(z2.combined)

        # Zones 1 and 3 must remain completely uncontaminated
        for z_ok in ("zone_1", "zone_3"):
            rok = r[z_ok]["T"]
            assert np.isfinite(rok.combined)
            assert np.isfinite(rok.belief)
            assert rok.conf_cz is not None and np.isfinite(rok.conf_cz)
            assert abs(rok.trust - b[z_ok]["T"].trust) < 0.15

    assert log[98]["zone_2"]["T"].trust < 0.05


def test_infinite_humidity_in_zone1_rejected_and_isolated(healthy):
    faults = [Fault("zone_1", "H", "inf", start=40, magnitude=1.0, end=90)]
    base = rollout(make_net(healthy=healthy), steps=110, seed=3)
    log = rollout(make_net(healthy=healthy), steps=110, seed=3, faults=faults)

    for b, r in zip(base[39:89], log[39:89]):
        z1 = r["zone_1"]["H"]
        assert np.isinf(z1.raw)
        assert z1.raw_weight == 0.0
        assert z1.match == 0.0
        assert np.isfinite(z1.combined)
        assert np.isfinite(z1.belief)
        assert z1.belief == pytest.approx(z1.combined)

        for z_ok in ("zone_2", "zone_3"):
            rok = r[z_ok]["H"]
            assert np.isfinite(rok.combined)
            assert np.isfinite(rok.belief)
            assert rok.conf_cz is not None and np.isfinite(rok.conf_cz)
            assert abs(rok.trust - b[z_ok]["H"].trust) < 0.15


def test_multiple_invalid_variables_in_one_zone_keeps_valid_var_and_other_zones_healthy(healthy):
    faults = [
        Fault("zone_2", "T", "nan", start=30, end=80),
        Fault("zone_2", "H", "inf", start=30, magnitude=-1.0, end=80),
    ]
    base = rollout(make_net(healthy=healthy), steps=100, seed=4)
    log = rollout(make_net(healthy=healthy), steps=100, seed=4, faults=faults)

    for b, r in zip(base[29:79], log[29:79]):
        # Invalid T and H in zone_2 stay finite in combined/belief
        assert np.isfinite(r["zone_2"]["T"].combined)
        assert np.isfinite(r["zone_2"]["H"].combined)
        assert r["zone_2"]["T"].raw_weight == 0.0
        assert r["zone_2"]["H"].raw_weight == 0.0
        # Valid M in zone_2 continues operating normally
        assert np.isfinite(r["zone_2"]["M"].raw)
        assert r["zone_2"]["M"].trust == pytest.approx(b["zone_2"]["M"].trust)
        # Zones 1 and 3 unaffected
        for z_ok in ("zone_1", "zone_3"):
            for v in VARS:
                assert np.isfinite(r[z_ok][v].combined)
                assert abs(r[z_ok][v].trust - b[z_ok][v].trust) < 0.15


def test_all_sensors_invalid_in_one_zone_isolated_from_healthy_zones(healthy):
    faults = [
        Fault("zone_2", "T", "nan", start=30, end=90),
        Fault("zone_2", "H", "inf", start=30, end=90),
        Fault("zone_2", "M", "nan", start=30, end=90),
    ]
    base = rollout(make_net(healthy=healthy), steps=100, seed=6)
    log = rollout(make_net(healthy=healthy), steps=100, seed=6, faults=faults)

    for b, r in zip(base[29:89], log[29:89]):
        for v in VARS:
            z2 = r["zone_2"][v]
            assert z2.raw_weight == 0.0
            assert z2.match == 0.0
            assert np.isfinite(z2.combined)
            assert np.isfinite(z2.belief)
        for z_ok in ("zone_1", "zone_3"):
            for v in VARS:
                rok = r[z_ok][v]
                assert np.isfinite(rok.combined)
                assert np.isfinite(rok.belief)
                assert abs(rok.trust - b[z_ok][v].trust) < 0.15

    for v in VARS:
        assert log[88]["zone_2"][v].trust < 0.05


def test_trust03_sudden_jump_spike_degrades_trust(healthy):
    faults = [Fault("zone_1", "T", "spike", start=50, magnitude=15.0, end=80)]
    log = rollout(make_net(healthy=healthy), steps=100, seed=8, faults=faults)
    assert log[48]["zone_1"]["T"].trust > 0.8
    assert log[78]["zone_1"]["T"].match == 0.0
    assert log[78]["zone_1"]["T"].trust < 0.1


def test_trust04_missing_or_dropped_readings_handled_predictably(healthy):
    net = make_net(healthy=healthy)
    base = {z: {v: RawReading(24.0 if v == "T" else 50.0) for v in VARS} for z in ZONE_IDS}
    net.reset(base)
    acts = {z: [0, 0, 0, 0] for z in ZONE_IDS}

    # Zone 2 drops all transmissions (missing keys & RawReading(None))
    partial = {
        "zone_1": base["zone_1"],
        "zone_2": {"T": RawReading(None)},
        "zone_3": base["zone_3"],
    }
    rec = net.step(partial, acts)
    for v in VARS:
        assert rec["zone_2"][v].raw_weight == 0.0
        assert rec["zone_2"][v].match == 0.0
        assert np.isfinite(rec["zone_2"][v].combined)
        assert np.isfinite(rec["zone_2"][v].belief)
    assert np.isfinite(rec["zone_1"]["T"].combined)
    assert np.isfinite(rec["zone_3"]["T"].combined)


def test_trust06_gradual_recovery_after_valid_readings_resume(healthy):
    faults = [Fault("zone_2", "T", "nan", start=40, end=100)]
    log = rollout(make_net(healthy=healthy), steps=200, seed=9, faults=faults)

    trust_at_end_of_fault = log[98]["zone_2"]["T"].trust   # t=99 (last fault step)
    trust_first_valid = log[99]["zone_2"]["T"].trust       # t=100 (first valid step)
    trust_final = log[-1]["zone_2"]["T"].trust

    assert trust_at_end_of_fault < 0.05
    # Must recover gradually via EMA, not jump straight to 1.0 on the first valid sample
    assert trust_first_valid < 0.3
    assert trust_first_valid > trust_at_end_of_fault
    assert trust_final > 0.85


def test_trust07_all_sensors_invalid_at_cold_start_and_cold_recovery():
    net = make_net()
    bad_init = {z: {v: RawReading(float("nan")) for v in VARS} for z in ZONE_IDS}
    net.reset(bad_init)
    acts = {z: [0, 0, 0, 0] for z in ZONE_IDS}

    # Without any valid reading or prior anchor, TrustNetwork refuses to fabricate plausible values
    r1 = net.step(bad_init, acts)
    for z in ZONE_IDS:
        for v in VARS:
            assert np.isnan(r1[z][v].combined)
            assert np.isnan(r1[z][v].belief)
            assert r1[z][v].trust == pytest.approx(net.cfg.trust_floor)
            assert r1[z][v].raw_weight == 0.0

    # Once valid readings arrive, cold-anchor cleanly and begin gradual trust recovery
    good = {z: {v: RawReading(24.0 if v == "T" else 50.0) for v in VARS} for z in ZONE_IDS}
    r2 = net.step(good, acts)
    for z in ZONE_IDS:
        for v in VARS:
            assert np.isfinite(r2[z][v].combined)
            assert np.isfinite(r2[z][v].belief)
            assert net.cfg.trust_floor < r2[z][v].trust < 0.5