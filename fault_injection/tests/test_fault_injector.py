import pytest

from fault_injection.fault_config import FaultConfig
from fault_injection.fault_types import FaultCategory
from fault_injection.fault_injector import FaultInjector


def create_sensor_fault(
    fault_id,
    fault_type,
    zone_id=0,
    target="temperature",
    magnitude=0,
    start_step=0,
    duration=None,
    probability=0.5
):
    return FaultConfig(
        fault_id=fault_id,
        category=FaultCategory.SENSOR,
        fault_type=fault_type,
        zone_id=zone_id,
        target=target,
        magnitude=magnitude,
        start_step=start_step,
        duration=duration,
        probability=probability
    )


# --------------------------------------------------
# 1. BIAS FAULT
# --------------------------------------------------

def test_bias_fault():
    injector = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "bias_1", "bias", magnitude=5
    )

    result = injector.apply_sensor_fault(
        25, fault, 0, 0, "temperature"
    )

    assert result == 30


# --------------------------------------------------
# 2. DRIFT FAULT
# --------------------------------------------------

def test_drift_fault():
    injector = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "drift_1",
        "drift",
        magnitude=0.2,
        start_step=10
    )

    result = injector.apply_sensor_fault(
        25, fault, 15, 0, "temperature"
    )

    assert result == pytest.approx(26.0)


# --------------------------------------------------
# 3. STUCK FAULT
# --------------------------------------------------

def test_stuck_fault():
    injector = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "stuck_1",
        "stuck",
        start_step=2,
        duration=3
    )

    results = []

    for step in range(7):
        reading = 25 + step

        result = injector.apply_sensor_fault(
            reading, fault, step, 0, "temperature"
        )

        results.append(result)

    assert results == [25, 26, 27, 27, 27, 30, 31]


# --------------------------------------------------
# 4. NOISE FAULT
# --------------------------------------------------

def test_noise_fault():
    injector1 = FaultInjector(seed=42)
    injector2 = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "noise_1",
        "noise",
        magnitude=0.5
    )

    result1 = injector1.apply_sensor_fault(
        25, fault, 0, 0, "temperature"
    )

    result2 = injector2.apply_sensor_fault(
        25, fault, 0, 0, "temperature"
    )

    assert result1 == result2
    assert result1 != 25


# --------------------------------------------------
# 5. SPIKE FAULT
# --------------------------------------------------

def test_spike_fault():
    injector = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "spike_1",
        "spike",
        magnitude=15
    )

    result = injector.apply_sensor_fault(
        25, fault, 0, 0, "temperature"
    )

    assert result == 40


# --------------------------------------------------
# 6. DROPOUT FAULT
# --------------------------------------------------

def test_dropout_fault():
    injector = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "dropout_1", "dropout"
    )

    result = injector.apply_sensor_fault(
        25, fault, 0, 0, "temperature"
    )

    assert result is None


# --------------------------------------------------
# 7. INTERMITTENT FAULT
# --------------------------------------------------

def test_intermittent_fault():
    injector = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "intermittent_1",
        "intermittent",
        magnitude=10,
        probability=1.0
    )

    result = injector.apply_sensor_fault(
        25, fault, 0, 0, "temperature"
    )

    assert result == 35


def test_intermittent_fault_probability_zero():
    injector = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "intermittent_2",
        "intermittent",
        magnitude=10,
        probability=0.0
    )

    result = injector.apply_sensor_fault(
        25, fault, 0, 0, "temperature"
    )

    assert result == 25


# --------------------------------------------------
# 8. ZONE INDEPENDENCE
# --------------------------------------------------

def test_stuck_fault_zone_independence():
    injector = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "multi_zone_stuck",
        "stuck",
        zone_id=None
    )

    result0 = injector.apply_sensor_fault(
        25, fault, 0, 0, "temperature"
    )

    result1 = injector.apply_sensor_fault(
        35, fault, 0, 1, "temperature"
    )

    result2 = injector.apply_sensor_fault(
        45, fault, 0, 2, "temperature"
    )

    assert result0 == 25
    assert result1 == 35
    assert result2 == 45


# --------------------------------------------------
# 9. MULTIPLE SIMULTANEOUS FAULTS
# --------------------------------------------------

def test_multiple_faults_across_zones_and_sensors():
    injector = FaultInjector(seed=42)

    faults = [
        create_sensor_fault(
            "bias_1",
            "bias",
            zone_id=0,
            target="temperature",
            magnitude=5
        ),

        create_sensor_fault(
            "noise_1",
            "noise",
            zone_id=1,
            target="humidity",
            magnitude=0.5
        ),

        create_sensor_fault(
            "dropout_1",
            "dropout",
            zone_id=2,
            target="soil_moisture"
        )
    ]

    temperature = injector.apply_sensor_faults(
        25, faults, 0, 0, "temperature"
    )

    humidity = injector.apply_sensor_faults(
        60, faults, 0, 1, "humidity"
    )

    soil_moisture = injector.apply_sensor_faults(
        40, faults, 0, 2, "soil_moisture"
    )

    assert temperature == 30
    assert humidity != 60
    assert soil_moisture is None


# --------------------------------------------------
# 10. CONFLICTING FAULTS
# --------------------------------------------------

def test_overlapping_sensor_faults_stack():
    injector = FaultInjector(seed=42)

    faults = [
        create_sensor_fault(
            "bias_1", "bias", magnitude=5
        ),

        create_sensor_fault(
            "spike_1", "spike", magnitude=10
        )
    ]

    # bias (+5) then spike impulse (+10) on its start step
    assert injector.apply_sensor_faults(
        25, faults, 0, 0, "temperature"
    ) == 40


# --------------------------------------------------
# 11. NO MATCHING FAULT
# --------------------------------------------------

def test_no_matching_fault():
    injector = FaultInjector(seed=42)

    fault = create_sensor_fault(
        "bias_1",
        "bias",
        zone_id=1,
        target="humidity",
        magnitude=10
    )

    result = injector.apply_sensor_faults(
        25, [fault], 0, 0, "temperature"
    )

    assert result == 25
    
def test_sensor_fault_persists_without_auto_recovery():
    injector = FaultInjector()

    fault = FaultConfig(
        fault_id="persistent_bias",
        category=FaultCategory.SENSOR,
        fault_type="bias",
        target="temperature",
        zone_id=0,
        start_step=5,
        duration=3,
        magnitude=5.0,
        auto_recover=False
    )

    result = injector.apply_sensor_faults(
        reading=25.0,
        faults=[fault],
        current_step=20,
        zone_id=0,
        target="temperature"
    )

    assert result == 30.0


def test_sensor_fault_expires_with_auto_recovery():
    injector = FaultInjector()

    fault = FaultConfig(
        fault_id="temporary_bias",
        category=FaultCategory.SENSOR,
        fault_type="bias",
        target="temperature",
        zone_id=0,
        start_step=5,
        duration=3,
        magnitude=5.0,
        auto_recover=True
    )

    result = injector.apply_sensor_faults(
        reading=25.0,
        faults=[fault],
        current_step=8,
        zone_id=0,
        target="temperature"
    )

    assert result == 25.0


def test_individual_sensor_fault_respects_auto_recovery():
    injector = FaultInjector()

    fault = FaultConfig(
        fault_id="persistent_bias",
        category=FaultCategory.SENSOR,
        fault_type="bias",
        target="temperature",
        zone_id=0,
        start_step=5,
        duration=3,
        magnitude=5.0,
        auto_recover=False
    )

    result = injector.apply_sensor_fault(
        reading=25.0,
        fault=fault,
        current_step=20,
        zone_id=0,
        target="temperature"
    )

    assert result == 30.0