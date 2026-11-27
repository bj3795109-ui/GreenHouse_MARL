"""Numeric-boundary and decay-law policy for the PID."""
import math

import numpy as np
import pytest

from control.pid_controller import PIDController
from control.types import CropBounds, MeasurementQuality, PIDGains, SensorMeasurement

B = CropBounds("c", "s", "T", "C", 10.0, 18.0, 26.0, 34.0)
G = PIDGains(Kp=0.4, Ki=0.002, Kd=2.0, tau_f=5.0, T_t=20.0)


def m(v):
    return SensorMeasurement("T", v, v, v, v, 1.0, 1.0, MeasurementQuality.TRUSTED, True, True)


@pytest.mark.parametrize("v", [np.float32(22.0), np.float64(22.0), np.int64(22), np.int32(22)])
def test_numpy_scalars_are_valid_measurements(v):
    u = PIDController("T", G).step(m(v), B)
    assert isinstance(u, float) and math.isfinite(u)


@pytest.mark.parametrize("u", [np.int8(1), np.float32(1.0), np.float64(0.5), 1, 0.5])
def test_shadow_tracking_accepts_numpy_effort(u):
    PIDController("T", G).track_shadow(m(17.0), B, u)


@pytest.mark.parametrize("v", [True, np.True_, float("nan"), float("inf")])
def test_bool_and_nonfinite_measurements_still_rejected(v):
    with pytest.raises(ValueError):
        PIDController("T", G).step(m(v), B)


def test_in_band_leak_is_exact_exponential():
    pid = PIDController("T", G)
    pid.state.integral = 0.6
    for _ in range(7):
        pid.step(m(22.0), B)
    assert pid.state.integral == pytest.approx(0.6 * math.exp(-7 * G.dt / G.T_t))


def test_in_band_leak_is_timestep_invariant():
    a, b = PIDController("T", G), PIDController("T", G)
    a.state.integral = b.state.integral = 0.6
    a.step(m(22.0), B, dt=2.0)
    b.step(m(22.0), B)
    b.step(m(22.0), B)
    assert a.state.integral == pytest.approx(b.state.integral)
