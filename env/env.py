"""
Greenhouse MARL environment — V1.1 §2.1-2.4 physics only.

Scope for this pass (per V1.1 §8 build order, step 1):
  - Thermal (§2.1), Humidity (§2.2), Soil Moisture (§2.4)
  - NOT included yet: Light (§2.5), CO2 (§2.3) — deferred until this
    core physics loop is validated with a random policy.
  - NOT included yet: trust network, fault injection, reward shaping.
    This is physics-only, to be sanity-checked before anything else
    is layered on top.

3 zones, fully-coupled (each zone shares a vent with both others).
"""

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

N_ZONES = 3
ZONE_IDS = [f"zone_{i+1}" for i in range(N_ZONES)]
NEIGHBORS = {  # each zone's neighbor list (all-pairs, 3-zone pie layout)
    "zone_1": ["zone_2", "zone_3"],
    "zone_2": ["zone_1", "zone_3"],
    "zone_3": ["zone_1", "zone_2"],
}


class PlaceholderParams:
    """
    Every value here is a [PLACEHOLDER] per V1.1 §9/§11 — physically
    plausible for a small tabletop enclosure, NOT calibrated.
    Do not trust simulator output quantitatively until these are
    replaced with measured values.
    """

    # --- Thermal (§2.1) ---
    Q_heat = 40.0          # W, heater output when on (guess: small ceramic/PTC heater)
    C_thermal = 2500.0     # J/°C, effective air+contents capacitance of one small zone
    h_loss = 0.8           # W/°C, passive loss coefficient
    h_vent = 3.5           # W/°C, forced-vent loss coefficient (> h_loss)
    k_T = 0.6              # W/°C, thermal coupling coefficient between adjacent zones
    T_ambient = 24.0       # °C

    # --- Humidity (§2.2) ---
    W_mist = 8.0            # %/s-equivalent contribution, mister output when on
    C_humidity = 40.0        # capacitance-style scaling for humidity balance.
                              # NOTE: originally 1.0, which gave a ~9s settling
                              # time vs. temperature's ~3000s+ — a 2-order-of-
                              # magnitude mismatch that was a placeholder-scale
                              # artifact, not intended physics. Bumped so H
                              # settles on the order of minutes instead of
                              # seconds; still a guess, still needs calibration.
    g_loss = 0.05           # passive humidity loss coefficient
    g_vent = 0.25           # forced-vent humidity loss coefficient (> g_loss)
    k_H = 0.03               # humidity coupling coefficient between adjacent zones
    H_ambient = 45.0         # %

    # --- Soil moisture (§2.4) ---
    alpha = 6.0              # moisture gain per unit irrigation time
    beta = 0.02               # lumped decay constant

    # --- Simulation ---
    dt = 1.0                  # seconds per step — WILL need revisiting once
                               # the numerical-stability check (V1.3 checklist) is run


def stability_bound_thermal(p: PlaceholderParams, zone_id: str) -> float:
    """
    Per-zone Euler stability bound for the thermal state, per the V1.3
    Numerical Stability Verification checklist item. Includes this
    zone's own loss/vent terms AND its coupling to every neighbor —
    a zone with more/stronger neighbors needs a smaller dt.
    Returns the bound in seconds; dt must be strictly less than this.
    """
    lam = p.h_loss + p.h_vent + len(NEIGHBORS[zone_id]) * p.k_T
    return 2.0 * p.C_thermal / lam


def stability_bound_humidity(p: PlaceholderParams, zone_id: str) -> float:
    """Same check, for the humidity state — separate params, separate bound."""
    lam = p.g_loss + p.g_vent + len(NEIGHBORS[zone_id]) * p.k_H
    return 2.0 * p.C_humidity / lam


def stability_bound_moisture(p: PlaceholderParams) -> float:
    """
    Soil moisture (§2.4) has no coupling term, so this bound is the
    same for every zone — just the decay constant beta.
    """
    return 2.0 / p.beta


class GreenhouseEnv(ParallelEnv):
    metadata = {"name": "greenhouse_v0"}

    def __init__(self, params: PlaceholderParams = None, max_steps: int = 500):
        self.params = params or PlaceholderParams()
        self.max_steps = max_steps
        self.possible_agents = list(ZONE_IDS)

        # Sanity-check dt against every zone's stability bound for all
        # three coupled state variables at init — fail loudly rather
        # than silently producing garbage trajectories.
        for zid in ZONE_IDS:
            tb = stability_bound_thermal(self.params, zid)
            if self.params.dt >= tb:
                raise ValueError(
                    f"dt={self.params.dt}s violates thermal Euler stability "
                    f"bound for {zid} (bound={tb:.3f}s). Reduce dt or adjust "
                    f"h_loss/h_vent/k_T/C_thermal."
                )
            hb = stability_bound_humidity(self.params, zid)
            if self.params.dt >= hb:
                raise ValueError(
                    f"dt={self.params.dt}s violates humidity Euler stability "
                    f"bound for {zid} (bound={hb:.3f}s). Reduce dt or adjust "
                    f"g_loss/g_vent/k_H/C_humidity."
                )
        mb = stability_bound_moisture(self.params)
        if self.params.dt >= mb:
            raise ValueError(
                f"dt={self.params.dt}s violates soil-moisture Euler stability "
                f"bound (bound={mb:.3f}s). Reduce dt or adjust beta."
            )

        # Action space per zone: heater, fan, mister, water — 4 binary
        # actuators (light excluded from this pass; see module docstring)
        self._action_space = spaces.MultiBinary(4)  # [heat, fan, mist, water]

        # Observation per zone: own [T, H, M] + both neighbors' [T, H, M]
        # raw values (no trust/estimate layer yet — that's §4, not this pass)
        obs_dim = 3 + 3 * len(NEIGHBORS[ZONE_IDS[0]])
        self._observation_space = spaces.Box(
            low=-100.0, high=200.0, shape=(obs_dim,), dtype=np.float32
        )

    def action_space(self, agent):
        return self._action_space

    def observation_space(self, agent):
        return self._observation_space

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self.step_count = 0
        rng = np.random.default_rng(seed)

        # Initialize each zone near a plausible baseline, with a little
        # jitter so the 3 zones aren't perfectly identical at t=0.
        self.state = {
            zid: {
                "T": self.params.T_ambient + rng.uniform(-1.0, 1.0),
                "H": self.params.H_ambient + rng.uniform(-3.0, 3.0),
                "M": 40.0 + rng.uniform(-5.0, 5.0),
            }
            for zid in ZONE_IDS
        }

        obs = {zid: self._get_obs(zid) for zid in self.agents}
        infos = {zid: {} for zid in self.agents}
        return obs, infos

    def _get_obs(self, zid):
        s = self.state[zid]
        own = [s["T"], s["H"], s["M"]]
        neighbor_vals = []
        for nid in NEIGHBORS[zid]:
            ns = self.state[nid]
            neighbor_vals += [ns["T"], ns["H"], ns["M"]]
        return np.array(own + neighbor_vals, dtype=np.float32)

    def step(self, actions):
        p = self.params
        dt = p.dt

        # Unpack actions per zone
        act = {}
        for zid in self.agents:
            heat, fan, mist, water = actions[zid]
            act[zid] = dict(heat=bool(heat), fan=bool(fan), mist=bool(mist), water=bool(water))

        # Compute derivatives for ALL zones from the CURRENT state before
        # updating any of them — simultaneous update, not sequential
        # (sequential would let zone order bias the coupling terms).
        dT = {}
        dH = {}
        dM = {}
        for zid in self.agents:
            s = self.state[zid]
            a = act[zid]

            # --- Thermal, §2.1 ---
            Q_heat = p.Q_heat if a["heat"] else 0.0
            Q_loss = p.h_loss * (s["T"] - p.T_ambient)
            Q_vent = (p.h_vent * (s["T"] - p.T_ambient)) if a["fan"] else 0.0
            coupling_T = sum(p.k_T * (s["T"] - self.state[nid]["T"]) for nid in NEIGHBORS[zid])
            dT[zid] = (Q_heat - Q_loss - coupling_T - Q_vent) / p.C_thermal

            # --- Humidity, §2.2 ---
            W_mist = p.W_mist if a["mist"] else 0.0
            W_loss = p.g_loss * (s["H"] - p.H_ambient)
            W_vent = (p.g_vent * (s["H"] - p.H_ambient)) if a["fan"] else 0.0
            coupling_H = sum(p.k_H * (s["H"] - self.state[nid]["H"]) for nid in NEIGHBORS[zid])
            dH[zid] = (W_mist - W_loss - coupling_H - W_vent) / p.C_humidity

            # --- Soil moisture, §2.4 — no coupling term ---
            water_added = 1.0 if a["water"] else 0.0
            dM[zid] = p.alpha * water_added - p.beta * s["M"]

        # Apply Euler update
        for zid in self.agents:
            self.state[zid]["T"] += dT[zid] * dt
            self.state[zid]["H"] += dH[zid] * dt
            self.state[zid]["H"] = float(np.clip(self.state[zid]["H"], 0.0, 100.0))
            self.state[zid]["M"] += dM[zid] * dt
            self.state[zid]["M"] = float(np.clip(self.state[zid]["M"], 0.0, 100.0))

        self.step_count += 1
        obs = {zid: self._get_obs(zid) for zid in self.agents}
        rewards = {zid: 0.0 for zid in self.agents}  # no reward shaping yet — physics pass only
        terminations = {zid: False for zid in self.agents}
        truncations = {zid: self.step_count >= self.max_steps for zid in self.agents}
        infos = {zid: {} for zid in self.agents}

        if all(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos