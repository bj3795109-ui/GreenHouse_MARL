import math
import pytest
from trust_network import RawReading, VARS, ZONE_IDS
from harness import make_net

IDLE = {z: (0, 0, 0, 0) for z in ZONE_IDS}


def _raw(missing=()):
    return {z: {v: RawReading(None if (z, v) in missing else val)
                for v, val in zip(VARS, (22.0, 60.0, 40.0))} for z in ZONE_IDS}


def test_missing_reading_does_not_crash_and_holds_trust():
    net = make_net(); net.reset(_raw())
    r = net.step(_raw({("zone_1", "T")}), IDLE)["zone_1"]["T"]
    assert r.raw is None and math.isnan(r.residual)
    assert r.trust == 1.0
    assert r.belief == r.combined


def test_estimate_keeps_running_through_outage():
    net = make_net(); net.reset(_raw())
    for _ in range(20):
        rec = net.step(_raw({("zone_1", "T")}), IDLE)
    assert math.isfinite(rec["zone_1"]["T"].combined)
    assert math.isfinite(rec["zone_2"]["T"].combined)


def test_no_usable_neighbors_falls_back_to_actuator_history():
    net = make_net(); net.reset(_raw())
    rec = net.step(_raw({(z, "T") for z in ZONE_IDS}), IDLE)["zone_1"]["T"]
    assert rec.est_cz is None and rec.combined == rec.est_act


def test_cold_start_requires_valid_readings():
    with pytest.raises(ValueError):
        make_net().reset(_raw({("zone_2", "H")}))
