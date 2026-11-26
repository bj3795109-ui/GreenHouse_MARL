# MAPPO Readiness Assessment (CTDE Verification)

**Status:** Pre-MAPPO readiness audit complete. **No MAPPO networks, reward functions, or training loops are implemented yet.**

---

## 1. CTDE Compatibility Assessment (`env/env.py`)

Intended paradigm: **Centralized Training with Decentralized Execution (CTDE)** across three zone agents (`zone_1`, `zone_2`, `zone_3`).

- **Decentralized Actors (Execution)**:
  - `GreenhouseEnv` implements PettingZoo `ParallelEnv`, returning synchronized per-zone observations `{zone_id: np.ndarray}` and accepting `{zone_id: action}` for all 3 zones simultaneously.
  - Each zone's `ZoneSupervisor` / `FallbackManager` already treats each zone's `marl_actions[zone_id]` independently, so decentralized per-zone actor inference maps directly onto `FallbackManager.step()`.
- **Centralized Critic (Training)**:
  - **Gap**: `GreenhouseEnv` does **not** currently implement `ParallelEnv.state()` (nor a global state space) returning the joint multi-zone state vector required by a centralized value function $V(s_t)$.

### Observation & Action Space Suitability for PPO

| Aspect | Current Implementation (`env/env.py`) | Readiness for PPO / CTDE |
|---|---|---|
| **Action Space** | `spaces.MultiBinary(4)` (`[heat, fan, mist, water]`) | **Suitable** (parameterize actor head as 4 independent Bernoulli logits; sum log-probs across bits). **Caution**: when `FallbackManager` overrides/clamps an action during training or rollout, PPO must store raw sampled actions + log-probs separately or treat safety clamping as environment dynamics. |
| **Observation Space** | `Box(-100, 200, shape=(9,), float32)`: own `[T, H, M]` + 2 neighbors' raw `[T, H, M]` | **Partially suitable (missing context & normalization)**. Lacks Trust Network `combined`/`trust` (or `belief`/`trust`), crop target bands `[optimal_min, optimal_max]` (needed if policies handle varying crops/stages), neighbor actuator states, and feature scaling/normalization. |
| **Reward Interface** | Hardcoded `rewards = {zid: 0.0}` | **Not ready** (`§7` reward function missing). |

---

## 2. Subsystem Readiness Checklist

| Component | Current Status | Missing Work for MAPPO |
|---|---|---|
| **Observation interface** | Raw 9-dim `[T, H, M]` vector in `env/env.py` | Add observation wrapper/adapter exposing normalized `[belief/combined, trust]` (or raw/estimate/trust triplets), crop target bounds (`CropBounds`), ambient conditions, and optionally neighbor previous actions. |
| **Action interface** | `MultiBinary(4)` (`heat, fan, mist, water`) | Compatible with multi-Bernoulli PPO actor; add optional `u_light` 5th actuator later if V1.1 scope requires `L`. |
| **Reward interface** | Returns `0.0` always | Implement `§7` multi-objective reward (crop optimal/acceptable band adherence, energy/water cost, switching penalty) and verify against PID baseline. |
| **Environment compatibility** | PettingZoo `ParallelEnv` passing `parallel_api_test` | Implement `env.state()` / `state_space` for centralized critic; expose `TrustNetwork` / `NoisySensors` wrapper cleanly without `sys.path` hacks (add package `__init__.py` or `pyproject.toml`). |
| **Centralized critic** | Missing | Implement MLP/GRU centralized critic $V_\phi(s_{\text{global}})$ taking joint observations + crop targets + ambient state. |
| **Actor network** | Missing | Implement decentralized actor $\pi_\theta(a_i \mid o_i)$ (parameter-shared across zones with optional zone/crop embedding, or per-zone actors) outputting 4 Bernoulli logits. |
| **Rollout collection** | Missing | Implement vectorized or parallel-episode rollout collector recording `(obs, global_state, actions, log_probs, rewards, dones, values)` per step. |
| **GAE** | Missing | Implement Generalized Advantage Estimation ($\gamma, \lambda$) with proper truncation vs. termination bootstrapping (`terminations` vs. `truncations`). |
| **PPO clipped objective** | Missing | Implement clipped surrogate policy loss, value loss (optional clipping), entropy bonus, and advantage normalization. |
| **Optimizer & training loop** | Missing | Add PyTorch dependency, Adam optimizer, learning-rate annealing, gradient clipping (`max_grad_norm`), and deterministic seeding. |
| **Checkpointing** | Missing | Save/load actor/critic weights, observation normalization stats (`RunningMeanStd`), and hyperparameters. |
| **Policy inference** | Missing (`_simple_healthy_marl` surrogate only) | Build inference adapter converting normalized observations -> `MultiBinary(4)` actions fed into `FallbackManager.step()`, with deterministic evaluation mode (`argmax` / threshold `0.5`). |
| **Multi-agent synchronization** | `ParallelEnv` synchronous step works | Ensure consistent agent ordering (`ZONE_IDS`) and clean handling if one zone enters `PID_FALLBACK` during evaluation. |
| **Evaluation & benchmarking** | `EpisodeEvaluator` (`control/metrics.py`) ready | Compare trained MAPPO vs. PID fallback baseline vs. MAPPO+PID hybrid under identical seeded fault-injection suites (`§11.3`). |

---

## 3. Recommended Next Implementation Sequence

1. **Packaging & Environment Wrapper**: Add `pyproject.toml` / `__init__.py` cleanly, implement `env.state()` for CTDE, and build a fault-aware observation + reward wrapper (`TrustNetwork` + `CropConfigLoader` + normalized features).
2. **Reward & Single-Agent RL Sanity Baseline**: Validate reward shaping with a single-zone PPO policy and benchmark against `PIDController` using `EpisodeEvaluator`.
3. **CTDE MAPPO Training Pipeline**: Implement PyTorch actor-critic, rollout buffer, GAE, PPO clipped updates, checkpointing, and deterministic evaluation.
4. **Closed-Loop Hybrid Evaluation**: Plug trained MAPPO inference into `FallbackManager` (replacing `_simple_healthy_marl`) and evaluate single/multi-zone crash, sensor fault, and recovery benchmarks.
