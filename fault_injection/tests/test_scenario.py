import pytest

from fault_injection.fault_config import FaultConfig
from fault_injection.fault_types import FaultCategory
from fault_injection.scenario import ScenarioError, validate_scenario


def sf(fid, ftype="bias", zone=0, target="temperature", start=5, dur=10, **kw):
    return FaultConfig(fid, FaultCategory.SENSOR, ftype, zone_id=zone, target=target,
                       start_step=start, duration=dur, magnitude=1.0, **kw)


def af(fid, ftype, zone=0, target="fan", start=5, dur=10):
    return FaultConfig(fid, FaultCategory.ACTUATOR, ftype, zone_id=zone, target=target,
                       start_step=start, duration=dur)


def test_clean_scenario_passes():
    validate_scenario([sf("a"), sf("b", zone=1), af("c", "stuck_on")])


def test_start_step_zero_rejected_by_default_but_configurable():
    with pytest.raises(ScenarioError, match="start_step 0"):
        validate_scenario([sf("a", start=0)])
    validate_scenario([sf("a", start=0)], min_start_step=0)


def test_duplicate_ids_rejected():
    with pytest.raises(ScenarioError, match="duplicate"):
        validate_scenario([sf("a"), sf("a", zone=2)])


def test_unimplemented_category_rejected():
    comm = FaultConfig("c", FaultCategory.COMMUNICATION, "x", start_step=3)
    with pytest.raises(ScenarioError, match="no implementation"):
        validate_scenario([comm])


def test_stuck_overlapping_other_sensor_fault_is_ambiguous():
    with pytest.raises(ScenarioError, match="order-dependent"):
        validate_scenario([sf("a", "stuck"), sf("b", "bias", start=8)])


def test_stuck_sequential_or_other_zone_is_fine():
    validate_scenario([sf("a", "stuck", dur=3), sf("b", "bias", start=8)])       # no overlap in time
    validate_scenario([sf("a", "stuck"), sf("b", "bias", zone=1)])               # different zone


def test_zone_none_overlaps_every_zone():
    with pytest.raises(ScenarioError):
        validate_scenario([sf("a", "stuck", zone=None), sf("b", "noise", zone=2)])


def test_additive_sensor_faults_may_overlap():
    validate_scenario([sf("a", "bias"), sf("b", "noise"), sf("c", "drift"), sf("d", "dropout")])


def test_conflicting_stuck_actuator_faults_rejected_but_activation_pair_ok():
    with pytest.raises(ScenarioError):
        validate_scenario([af("a", "stuck_on"), af("b", "stuck_off")])
    validate_scenario([af("a", "failure_to_activate"), af("b", "failure_to_deactivate")])


def test_open_ended_fault_overlaps_later_faults():
    forever = sf("a", "stuck", dur=None, auto_recover=False)
    with pytest.raises(ScenarioError):
        validate_scenario([forever, sf("b", "bias", start=500)])


def test_all_problems_reported_together():
    with pytest.raises(ScenarioError) as e:
        validate_scenario([sf("a", start=0), sf("a", "stuck"), sf("b", "bias", start=6)])
    msg = str(e.value)
    assert "duplicate" in msg and "start_step 0" in msg and "order-dependent" in msg


# ---------------- scheduler enforces validation ----------------
from fault_injection.fault_scheduler import FaultScheduler


def test_scheduler_rejects_invalid_scenarios_without_a_separate_call():
    comm = FaultConfig("c", FaultCategory.COMMUNICATION, "x", start_step=3)
    with pytest.raises(ScenarioError):
        FaultScheduler([comm])
    with pytest.raises(ScenarioError):
        FaultScheduler([sf("a", "stuck"), sf("b", "bias", start=8)])


def test_wrapper_start_step_rule_is_a_scheduler_argument():
    FaultScheduler([sf("a", start=0)])                          # generic use: allowed
    with pytest.raises(ScenarioError, match="start_step 0"):
        FaultScheduler([sf("a", start=0)], min_start_step=1)    # what the wrapper must pass
