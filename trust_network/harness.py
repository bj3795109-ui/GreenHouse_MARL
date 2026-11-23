"""
Test harness for the trust network: noisy sensors, minimal fault injection,
rollouts, and healthy-residual bootstrapping.

The env has no sensor noise (it returns exact state), so noise is added here.
The fault injector here is deliberately minimal (stuck / drift / bias) — the
full fault-injection framework (V1.1 §8 step 4) is a separate later module.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "env"))
from env import GreenhouseEnv, PlaceholderParams  # noqa: E402

from trust_network import (  # noqa: E402
    ZONE_IDS, VARS, RawReading, AssumedParams, HealthyResiduals,
    TrustConfig, TrustNetwork,
)

SENSOR_SIGMA = {"T": 0.1, "H": 0.5, "M": 0.5}


@dataclass
class Fault:
    zone: str
    var: str
    kind: str          # "stuck" | "drift" | "bias"
    start: int
    magnitude: float = 0.0   # drift: units per step; bias: constant offset


class NoisySensors:
    def __init__(self, rng, sigma=None, faults=None):
        self.rng, self.sigma, self.faults = rng, sigma or SENSOR_SIGMA, faults or []
        self._stuck = {}

    def read(self, state, t):
        out = {}
        for z in ZONE_IDS:
            out[z] = {}
            for v in VARS:
                val = state[z][v] + self.rng.normal(0, self.sigma[v])
                for f in self.faults:
                    if f.zone == z and f.var == v and t >= f.start:
                        if f.kind == "stuck":
                            val = self._stuck.setdefault((z, v), val)
                        elif f.kind == "drift":
                            val += f.magnitude * (t - f.start)
                        elif f.kind == "bias":
                            val += f.magnitude
                out[z][v] = RawReading(float(val))
        return out


def rollout(net, steps=600, seed=0, faults=None, env_params=None, return_truth=False):
    """Run one episode with random actions. Returns list of per-step record dicts."""
    env = GreenhouseEnv(params=env_params, max_steps=steps + 1)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    sensors = NoisySensors(rng, faults=faults)
    net.reset(sensors.read(env.state, 0))
    log, truth = [], []
    for t in range(1, steps + 1):
        actions = {z: env.action_space(z).sample() for z in ZONE_IDS}
        env.step(actions)
        log.append(net.step(sensors.read(env.state, t), actions))
        truth.append({z: dict(env.state[z]) for z in ZONE_IDS})
    return (log, truth) if return_truth else log


def bootstrap_healthy(assumed, n_episodes=30, steps=600, seed0=100):
    """Fit healthy-residual distribution from fault-free runs through the same pipeline."""
    net = TrustNetwork(assumed, healthy=None)
    samples = {v: [] for v in VARS}
    for ep in range(n_episodes):
        for rec in rollout(net, steps=steps, seed=seed0 + ep):
            for z in ZONE_IDS:
                for v in VARS:
                    samples[v].append(rec[z][v].residual)
    return HealthyResiduals(samples)


def make_net(calibration_error=None, healthy=None, cfg=None):
    assumed = AssumedParams.from_env(PlaceholderParams(), calibration_error)
    return TrustNetwork(assumed, healthy, cfg or TrustConfig())