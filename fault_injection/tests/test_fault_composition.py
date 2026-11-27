from fault_injection.fault_config import FaultConfig
from fault_injection.fault_injector import FaultInjector
from fault_injection.fault_types import FaultCategory


def sf(fid, ftype, zone=0, target="temperature", **kw):
    return FaultConfig(fid, FaultCategory.SENSOR, ftype, zone_id=zone, target=target, **kw)


def af(fid, ftype):
    return FaultConfig(fid, FaultCategory.ACTUATOR, ftype, zone_id=0, target="fan")


def test_overlapping_sensor_faults_compose_in_order():
    inj = FaultInjector(seed=1)
    out = inj.apply_sensor_faults(
        20.0, [sf("b1", "bias", magnitude=2.0), sf("b2", "bias", magnitude=3.0)], 0, 0, "temperature")
    assert out == 25.0


def test_dropout_short_circuits_regardless_of_order():
    inj = FaultInjector(seed=1)
    b, d = sf("b", "bias", magnitude=2.0), sf("d", "dropout")
    assert inj.apply_sensor_faults(20.0, [b, d], 0, 0, "temperature") is None
    assert inj.apply_sensor_faults(20.0, [d, b], 0, 0, "temperature") is None


def test_overlapping_actuator_faults_compose():
    inj = FaultInjector()
    out = inj.apply_actuator_faults(
        0, 1, [af("a", "failure_to_deactivate"), af("b", "stuck_off")], 0, 0, "fan")
    assert out == 0   # first holds it on, second forces it off


def test_failure_to_activate_differs_from_stuck_off():
    inj = FaultInjector()
    f = af("f", "failure_to_activate")
    g = lambda req, prev: inj.apply_actuator_faults(req, prev, [f], 0, 0, "fan")
    assert g(1, 0) == 0          # cannot turn on
    assert g(1, 1) == 1          # already running keeps following the request
    assert g(0, 1) == 0 and g(0, 0) == 0


def test_failure_to_deactivate_is_symmetric():
    inj = FaultInjector()
    f = af("f", "failure_to_deactivate")
    g = lambda req, prev: inj.apply_actuator_faults(req, prev, [f], 0, 0, "fan")
    assert g(0, 1) == 1          # cannot turn off
    assert g(0, 0) == 0 and g(1, 0) == 1 and g(1, 1) == 1


def test_spike_is_a_single_step_impulse_unlike_bias():
    inj = FaultInjector()
    f = sf("sp", "spike", magnitude=5.0, start_step=3, duration=10)
    vals = [inj.apply_sensor_faults(20.0, [f], t, 0, "temperature") for t in range(8)]
    assert vals == [20.0, 20.0, 20.0, 25.0, 20.0, 20.0, 20.0, 20.0]


def test_noise_stream_independent_of_unrelated_faults():
    nz = sf("n", "noise", magnitude=1.0)
    other = sf("o", "intermittent", zone=1, target="humidity", magnitude=1.0, probability=0.5)
    a, b = FaultInjector(seed=7), FaultInjector(seed=7)
    ra = [a.apply_sensor_faults(20.0, [nz, other], t, 0, "temperature") for t in range(6)]
    rb = []
    for t in range(6):
        b.apply_sensor_faults(20.0, [nz, other], t, 1, "humidity")   # extra, unrelated draws
        rb.append(b.apply_sensor_faults(20.0, [nz, other], t, 0, "temperature"))
    assert ra == rb


def test_seed_controls_reproducibility():
    nz = sf("n", "noise", magnitude=1.0)
    run = lambda s: [FaultInjector(seed=s).apply_sensor_faults(20.0, [nz], 0, 0, "temperature")]
    assert run(3) == run(3) and run(3) != run(4)


def test_draws_depend_on_step_not_call_order():
    nz = sf("n", "noise", magnitude=1.0)
    a, b = FaultInjector(seed=5), FaultInjector(seed=5)
    full = {t: a.apply_sensor_faults(20.0, [nz], t, 0, "temperature") for t in range(10)}
    sparse = {t: b.apply_sensor_faults(20.0, [nz], t, 0, "temperature") for t in (1, 4, 8)}
    assert all(sparse[t] == full[t] for t in sparse)          # skipped steps don't shift draws
    assert a.apply_sensor_faults(20.0, [nz], 4, 0, "temperature") == full[4]   # re-query is stable
