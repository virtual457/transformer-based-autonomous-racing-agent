# Assetto Corsa Deep RL Racing Agent — Project Report

> **Maintained document.** Updated after every significant architectural, reward, or training change.
> Last updated: 2026-04-03

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Model Architecture](#3-model-architecture)
4. [SAC Algorithm](#4-sac-algorithm)
5. [Replay Buffer](#5-replay-buffer)
6. [Training Loop](#6-training-loop)
7. [Reward System — Evolution and Rationale](#7-reward-system--evolution-and-rationale)
8. [Behavioural Cloning Pipeline](#8-behavioural-cloning-pipeline)
9. [Variants Tried](#9-variants-tried)
10. [Problems Faced and How They Were Solved](#10-problems-faced-and-how-they-were-solved)
11. [Checkpointing](#11-checkpointing)
12. [Infrastructure and Tooling](#12-infrastructure-and-tooling)

---

## 1. Project Overview

**Goal:** Train a deep reinforcement learning agent to drive a car around Monza in Assetto Corsa (AC), matching or exceeding the performance of a human reference lap.

**Car / Track:** `ks_mazda_miata` on `monza` (5,758 m). Hotlap mode, automatic gearbox, ABS/TC on maximum.

**Approach:** Soft Actor-Critic (SAC) with a real-time gym environment built on the `assetto_corsa_gym` library. The agent receives 125-dimensional observations and outputs 3-dimensional continuous actions (steer, throttle, brake) in `[-1, 1]`.

**Key constraint:** The environment runs on a live Windows gaming PC — Assetto Corsa is a real game, not a simulator wrapper. All training happens in real time at 25 Hz, with the agent communicating with the AC plugin over a local TCP/UDP socket.

---

## 2. System Architecture

```
┌─────────────────────────────────────┐
│         train_sac.py / finetune_sac │  ← entry point
├─────────────────────────────────────┤
│           SACAgent (agent.py)       │  ← phase loop, latency overlap
│  collect_phase()  │  train_phase()  │
├───────────────────┴─────────────────┤
│           OurEnv (our_env.py)       │  ← reward, telemetry, obs
├─────────────────────────────────────┤
│     AssettoCorsaEnv (ac_env.py)     │  ← AC plugin socket, vJoy
├─────────────────────────────────────┤
│    Assetto Corsa (acs.exe)          │  ← real game, physics at 25 Hz
└─────────────────────────────────────┘
```

**Action pipeline:** Policy outputs tanh-squashed `[-1,1]` → `vjoy.py` maps with `steer = action * 2 - 1` → `car_control.py` sends: `steer_axis = steer + 1` (→ `[0,2]`), `acc_axis = (pedal+1)/2` (→ `[0,1]`), `brake_axis = (brake+1)/2` (→ `[0,1]`).

**AC lifecycle:** `ac_lifecycle.py` manages launch/kill of `acs.exe` via Content Manager. With `--manage-ac`, AC is launched before each collect phase and killed after. A 15-second settle delay after launch ensures the plugin is ready before the agent begins pinging.

---

## 3. Model Architecture

### GaussianPolicy

Tanh-squashed diagonal Gaussian, implemented in `gym/sac/network.py`.

```
obs (125,) → Linear(125→256) → ReLU
           → Linear(256→256) → ReLU
           → Linear(256→256) → ReLU
           → Linear(256→6)
                ├── mean  (3,)
                └── log_std (3,)   clamped to [-20, +2]
```

A reparametrised sample is drawn from `N(mean, exp(log_std))`, squashed through `tanh`, and the log-probability is corrected:

```
log π(a|s) = log N(x|μ,σ) − Σ log(1 − tanh²(x) + ε),   ε = 1e-6
```

**Parameter count:** 165,382 (~2 MB at float32).

**Why 3 hidden layers of width 256:** Standard SAC baseline (Haarnoja et al. 2018 used `[256,256]`). A third layer was added to handle the 125-dimensional observation — substantially larger than standard MuJoCo benchmarks (17–60 dims). The model has more than enough capacity for the task; increasing size beyond this yields no benefit.

### Observation Space (125 dims)

| Block | Dims | Description |
|---|---|---|
| Telemetry | 11 | speed, gap, LastFF, RPM, accelX/Y, gear, angularVel_y, localVel_x/y |
| Ray sensors | 11 | 11-ray fan cast ÷ 200 m |
| Out-of-track | 1 | binary OOT flag |
| Curvature look-ahead | 12 | 300 m ahead, 12-point downsampled spline curvature |
| Past actions | 9 | 3 steps × (steer, pedal, brake) |
| Current absolute actions | 3 | current steer, pedal, brake |
| Previous observations | 78 | 3 × prev_obs (with `add_previous_obs_to_state=True`) |

### Action Space (3 dims)

`[steer, throttle, brake]` all in `[-1, 1]` policy space. Delta-rate limiting applied before vJoy: steer 900 deg/s, pedal/brake 1200 per step.

### TwinQNetwork

Two independent Q-networks, input `cat([obs, action], dim=128)` → `[256,256,256]` → scalar. Min(Q1, Q2) provides the clipped double-Q trick (Fujimoto et al. 2018) to prevent overestimation bias.

**Parameter count per Q-network:** 164,865. Total online: **495,112 parameters**.

---

## 4. SAC Algorithm

Implemented in `gym/sac/sac.py`.

### Update Order (one step)

1. Sample batch from replay buffer
2. Bellman target with entropy regularisation:
   ```
   y = r + γ(1−done) × (min(Q1_t, Q2_t)(s', a') − α × log π(a'|s'))
   ```
3. Update twin Q: minimise `MSE(Q1,y) + MSE(Q2,y)`
4. Update policy: minimise `α × log π(a|s) − min(Q1,Q2)(s,a)`
5. Update temperature: minimise `−log_α × (log π(a|s) + H_target)`
6. Soft-update targets: `θ_t ← τ θ + (1−τ) θ_t`

Gradient clipping at `max_norm=1.0` on all networks (added to fix training instability — see §10).

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| gamma (γ) | 0.992 | Higher than 0.99 default. At 25 Hz, half-life ≈ 85 steps ≈ 3.4 s — appropriate for a racing task where progress compounds over seconds |
| lr | 3e-4 | Standard SAC default (Haarnoja et al.). Same for policy, Q, alpha optimizers |
| tau (τ) | 0.005 | Standard Polyak rate — stable TD targets while tracking policy improvement |
| batch_size | 256 | Doubled from default 128 to stabilize gradient estimates at small buffer sizes |
| target_entropy | -3.0 | `-action_dim` heuristic (Haarnoja et al.). Allows stochasticity early, permits near-deterministic convergence |
| alpha cap | 1.0 | `log_alpha` clamped ≤ 0 after each update → `alpha ≤ 1.0`. Prevents runaway entropy |

**Observation clamping:** `obs.clamp(-3.0, 3.0)` applied in both `select_action()` and `update_from_batch()` to guard against out-of-range channels (LastFF, curvature) that caused gradient spikes early in training.

### Latency Overlap Trick

```python
obs = next_obs                  # frame N has arrived
action = select_action(obs)     # infer (~0.1 ms GPU)
env.set_actions(action)         # fire to AC immediately, non-blocking
replay_buffer.push(...)         # CPU work — overlaps with AC's 40 ms tick
time.sleep(until 9ms elapsed)   # 9 ms action pacing floor
env.step(action=None)           # wait for tick completion
```

**Why it matters:** Without this, the action sent to AC would be computed from a state 40 ms old (one full stale frame). With the overlap, inference fires first, AC processes physics while bookkeeping runs, and effective latency collapses to ~0.2 ms.

**9 ms pacing (`STEP_INTERVAL_S = 0.009`):** Enforces a minimum inter-action interval. Without it, inference time varied 1–8 ms causing variable-length transitions that violated the fixed time-step assumption of the discount factor γ. Present in both `agent.py` and `finetune_sac.py`.

---

## 5. Replay Buffer

Implemented in `gym/sac/replay_buffer.py`.

### Base ReplayBuffer

Pre-allocated FIFO ring buffer. Each transition: `(obs, action, reward, next_obs, done)` = 255 float32s = **~1 KB**. Uniform random sampling. Returns CUDA tensors.

### DualReplayBuffer

Two independent `ReplayBuffer` instances: `positive_buffer` (reward > 0) and `negative_buffer` (reward ≤ 0). Sampling draws `batch_size//2` from each (128 each), concatenates, shuffles. Graceful fallback when one side is sparse.

**Why positive/negative split:** Early training produces almost entirely negative-reward transitions. A flat FIFO buffer would overwhelm the Q-network with bad states, causing Q-value collapse. The 50/50 forced mix ensures positive-reward transitions (once they appear) receive equal gradient signal, accelerating learning of the correct behavior manifold.

**Capacity (SAC_TargetSpeed):** 500,000 per sub-buffer = 1M total = ~1 GB RAM. Safe maximum for an 8 GB RAM system running AC simultaneously.

**Persistence:** Both sub-buffers serialized to `buffer_pos.npz` + `buffer_neg.npz` at every checkpoint. Auto-loaded on restart.

### Bucketed Buffer (finetune_sac.py)

10 reward-stratified buckets with boundaries `[-100, -0.8, -0.6, ..., 0.8, 1.001]`, 50k capacity each. Uniform sampling across all non-empty buckets. Ensures crash transitions and high-reward transitions receive equal gradient signal despite different natural frequencies.

---

## 6. Training Loop

### Phase Structure

**Collect phase:**
- Run `episodes_per_phase = 10` episodes with current SAC policy
- Each episode: 150–250 warmup steps at full throttle (discarded) then policy control
- Latency overlap trick + 9 ms pacing active from step 1
- Stationary-car crash detection: if displacement < 0.5 m over 10 frames → terminate episode

**Train phase:**
- Gradient steps = `min(steps_collected, 20_000)` (1:1 ratio with new experience)
- Buffer NOT cleared between phases — rolling FIFO window
- Checkpoint every 5 phases

**Gradient steps rationale:** 1:1 ratio is standard SAC. Training the entire buffer every phase would over-fit to stale transitions from earlier (worse) policies. The 20k cap prevents runaway compute on large collect phases.

---

## 7. Reward System — Evolution and Rationale

### Generation 1 (original)

Unbounded weighted sum with a lagged speed multiplier:
```
total = Σ weight_i × (negate_i ? -raw_i : raw_i)
multiplier = prev_episode_mean_speed − 40.0 + 1.0   clamped [-3, 3]
if total ≥ 0: total *= multiplier
else:         total *= abs(multiplier)
```

Output typically in `[-5, +1]`. No bounds enforced. The crash component had weight 10.0 — it dominated all other signals. The speed multiplier could flip sign on positive rewards, creating a training blocker.

### Generation 2 (2026-03-28)

Bounded `[-1, 1]` redesign. All component formulas normalised, negation baked in, speed multiplier removed. `GapDeltaReward` added briefly. `REWARD_SCALE` and `negate` flags removed.

### Generation 3 (2026-03-30 → current)

`GapDeltaReward` removed (exploitable — see §10). Replaced by a gap-delta cap. `TargetSpeedReward` replaced `SpeedReward` on 2026-04-02.

---

### Active Reward Components

#### ProgressReward (`r_progress`, weight 0.30)
```
delta = telem.lap_dist − prev_lap_dist   (with lap wrap-around)
r_progress = clamp(delta / 0.111 − 1.0, −1, 1)
```
Cap = 0.111 m/frame ≈ 2.775 m/s at 25 Hz. Standstill = −1.0. At cap speed = 0.0. Above cap → +1.0.

#### TargetSpeedReward (`r_speed`, weight 0.20)
```
r_speed = clamp(1.0 − 2.0 × |speed_ms − target_speed_ms| / target_speed_ms, −1, 1)
```
| speed vs. target | r_speed |
|---|---|
| speed == target | +1.0 |
| speed == 1.5× target | −1.0 |
| speed == 0.5× target | 0.0 (neutral) |

Symmetric: both overspeed and underspeed are penalized. Fallback to Generation-2 `SpeedReward` formula when `target_speed_ms ≤ 0.0` (ref_lap not loaded).

**Why this replaced SpeedReward:** `SpeedReward` was monotonically increasing with speed — the policy had zero incentive to ever brake. A car blasting into the first Monza chicane at 47 m/s got rewarded more than one braking to the correct 19.4 m/s. `TargetSpeedReward` creates a genuine braking incentive by making the corner apex speed the optimal target.

#### GapReward (`r_gap_abs`, weight 0.35)
```
r_gap_abs = 1.0 − 2.0 × clamp(|gap_m| / 10.0, 0, 1)
```
On racing line = +1.0. At 5 m = 0.0. At 10 m+ = −1.0. Highest weight — tracking the racing line is the primary learning signal.

#### SmoothnessReward (`r_smoothness`, weight 0.15)
```
delta_norm = ||action − prev_action||₂
r_smoothness = 1.0 − 2.0 × clamp(delta_norm / 2.0, 0, 1)
```
Zero delta = +1.0. Zero action jitter is rewarded; oscillatory steering punished. Weight tripled from 0.05 → 0.15 on 2026-04-01 after observing rapid oscillatory steering in early training.

#### CrashReward (backwards-compat stub only)
Never evaluated by active `CompositeReward.compute()`. Crash is handled as a hard override (see below).

#### GapDeltaReward (removed 2026-03-30)
Class retained in `components.py` for backwards compatibility. Log keys `r_gap_delta` / `c_gap_delta` always 0.0. Never instantiated by any active factory method.

---

### CompositeReward Architecture

**Crash short-circuit:** If `telem.out_of_track`: immediately return `total = −1.0`. No component evaluated.

**Weighted sum:**
```
total = 0.30×r_progress + 0.20×r_speed + 0.35×r_gap_abs + 0.15×r_smoothness
```
Followed by `clamp(total, −1, 1)`.

**Gap-delta cap:**
```python
gap_delta = abs(telem.gap_m) − prev_gap_m
if gap_delta > 0 and abs(telem.gap_m) > 0.3:       # dead zone = 0.3 m
    cap = −(abs(telem.gap_m) / 10.0)               # range [−1, 0]
    total = min(total, cap)
```
Cap only reduces total, never increases it — cannot be gamed for positive reward.

**No speed multiplier.** `multiplier = 1.0` and `reward_scale = 1.0` appear in log rows for schema backwards compatibility only.

**Bounded `[-1, 1]` invariant:** Every step returns exactly one float in `[-1, 1]`. On crash: floor −1.0. On perfect step: ceiling +1.0.

---

### Target Speed Generation

Generated by `AICLONE/generate_target_speed.py` from human demo data.

**Method:**
1. Load `collectDataAI/data/ks_mazda_miata/monza/20260402_175809/*.parquet` (18,572 frames, out-of-track excluded)
2. Build `cKDTree` over `(pos_x, pos_y)` of all demo frames
3. For each of 3,750 racing line waypoints: find K=10 nearest demo frames → take **median** speed (median chosen over mean to suppress braking transient outliers)
4. Apply `uniform_filter1d(window=150)` with circular wrap-around (~150 m smoothing — removes noise, preserves corner braking gradients)

**Monza speed profile (ks_mazda_miata):**

| Statistic | Value |
|---|---|
| Minimum | 19.4 m/s (69.8 km/h) — first chicane (Variante del Rettifilo) |
| Maximum | 47.3 m/s (170.3 km/h) — pit straight |
| Mean | 36.9 m/s (132.8 km/h) |

**Runtime injection:** `our_env.py` populates `telem.target_speed_ms` at both `step()` and `collect_episode()` sites via `ref_lap.get_target_speed_value(telem.lap_dist)`, conditional on `use_target_speed: true` in `config.yml`.

---

### TelemetryFrame Fields

| Field | Type | Source | Notes |
|---|---|---|---|
| `speed_ms` | float | `ac_state["speed"]` | m/s |
| `lap_dist` | float | `ac_state["LapDist"]` | metres along lap |
| `gap_m` | float | `ac_state["gap"]` | signed lateral gap from racing line |
| `n_tyres_out` | int | `ac_state["numberOfTyresOut"]` | 0–4 |
| `out_of_track` | bool | set by `ac_env` | True when `n_tyres_out > 2` |
| `lap_time_s` | float | optional | defaults 0.0 |
| `best_lap_s` | float | optional | defaults 0.0 |
| `target_speed_ms` | float | injected by `our_env.py` | defaults 0.0; populated when `use_target_speed=True` |
| `raw` | dict | pass-through | full AC state dict for logging |

---

## 8. Behavioural Cloning Pipeline

### Architecture

`AICLONE/pretrain_actor.py` trains the same `GaussianPolicy` network used by SAC, via supervised MSE loss on `(obs, action)` pairs. No Q-networks. Checkpoint format: `{"policy": state_dict}` — intentionally a subset of the full SAC checkpoint, loadable directly via `sac.policy.load_state_dict(ckpt["policy"])`.

### Action Space Extraction (Critical Fix)

The correct extraction from raw AC state (derived from `car_control.py` mapping):
```python
steer = float(np.clip((steerAngle / 302.4 + 1.0) / 2.0, 0.0, 1.0))
pedal = float(accStatus)     # direct — car_control does: acc_axis = (pedal+1)/2
brake = float(brakeStatus)   # direct — same mapping
```

**Why this matters:** The old formula used `steer = steerAngle / 302.4` which gives 0.0 for straight-ahead. But SAC expects 0.5 for straight-ahead (since `map_action = action * 2 − 1`, so `0.5 * 2 − 1 = 0.0` = centered). Training on the wrong formula caused the BC model to always command full-left steering on a straight road. This was confirmed by inspecting `buffer_pos.npz` where SAC stored steer mean ≈ 0.501 for straight driving.

### Data Sources

| Dataset | Steps | Notes |
|---|---|---|
| HuggingFace `dasgringuen/assettoCorsaGym` | 37,374 | Multi-track, cross-car |
| Human demo (monza, 22 sessions) | 159,255 | User's own laps |
| Monza-only filtered | 57,490 | 3 tracks: monza, imola, ks_silverstone |

---

## 9. Variants Tried

| Name | Description | Status |
|---|---|---|
| `150_jitter` | SAC from scratch on Monza, original SpeedReward | ✅ Baseline, drives but weak at high-speed braking |
| `150_bc` | BC pre-train (all 3 tracks) → SAC. Had action space bug (going-left). | ❌ Abandoned due to steer bug |
| `150_bc_monza` | BC pre-train (monza only, fixed steer formula) → merged into SAC | ✅ Better init, merged critics from 150_jitter |
| `150_demo` | Offline SAC fine-tune on demo data with reward=1.0 | ❌ Catastrophic forgetting |
| `SAC_TargetSpeed` | Warm-start from 150_jitter, TargetSpeedReward, alpha reset | 🔄 Current active run |

### Model Merging (150_bc_monza)

Critics (twin_q, twin_q_target) and optimizers loaded from `150_jitter` (RL-trained). Policy loaded from `150_bc_monza/best.pt` (BC-trained). Rationale: BC policy knows how to drive; RL critics know the value landscape of Monza. Merging gives the best of both without re-learning from scratch.

---

## 10. Problems Faced and How They Were Solved

### 1. BC Model Going Left (Fixed 2026-04-02)

**Symptom:** BC-trained model consistently steered hard left immediately.

**Root cause:** Wrong steer extraction formula. `steer = steerAngle / 302.4` gives 0.0 for straight. SAC expects 0.5 for straight (because downstream mapping `action * 2 − 1` converts 0.5 → 0.0 = centered). Training on 0.0 taught the model that 0.0 = straight, but at inference `0.0 * 2 − 1 = −1.0` = full left.

**Fix:** `steer = (steerAngle / 302.4 + 1.0) / 2.0` — confirmed by checking `buffer_pos.npz` where stored SAC steer mean ≈ 0.501 for straight driving.

### 2. Training Instability / Gradient Spikes (ISSUE-013, Fixed 2026-03-29)

**Symptom:** Q-loss spiking to >1000, then NaN propagation in policy loss.

**Root cause:** Out-of-range observation channels. `LastFF` divided by 1.0 (max ~5). Curvature channels divided by 0.1 (range ~[-10, 10] in tight corners). These produced gradient spikes in the Q-network.

**Fix:** `obs.clamp(-3.0, 3.0)` in both `select_action()` and `update_from_batch()`. Gradient clipping at `max_norm=1.0` added as secondary defense.

### 3. GapDeltaReward Exploit (Fixed 2026-03-30)

**Symptom:** Policy learned to snake across the track rather than following the racing line.

**Root cause:** `GapDeltaReward` rewarded lateral improvement per step. Policy discovered: drift wide → incur one-time gap-absolute penalty → aggressively correct back → accumulate large positive delta rewards. Net delta signal from correction exceeded gap-absolute penalty from drift. Oscillation became profitable.

**Fix:** Removed from weighted sum. Replaced with a gap-delta cap (hard ceiling, can only reduce total reward, cannot be gamed for positive reward).

### 4. SpeedReward No Braking Incentive (Fixed 2026-04-02)

**Symptom:** Model drove at maximum speed into corners, never braked appropriately.

**Root cause:** `SpeedReward` was monotonically increasing with speed. More speed = more reward, always, everywhere on track.

**Fix:** `TargetSpeedReward` with per-corner target speed derived from human demo data. First time the policy has a genuine incentive to slow down before a corner.

### 5. Speed Multiplier Sign-Flip Bug (Fixed 2026-03-28)

**Symptom:** After an episode with mean speed < 39 m/s, multiplier went negative and flipped positive rewards to negative — punishing correct behavior.

**Root cause:** `multiplier = prev_speed − 40 + 1`. If prev_speed < 39, multiplier < 0. Code path: `if total ≥ 0: total *= multiplier` silently inverted good rewards.

**Fix:** Removed speed multiplier entirely in Generation-2 redesign.

### 6. Catastrophic Forgetting from Offline Fine-tuning (150_demo, 2026-04-01)

**Symptom:** Model forgot how to drive within a small number of RL gradient steps after BC init.

**Root cause:** BC pre-training initializes the policy near human behavior. SAC fine-tuning immediately fires gradients computed from low-quality online transitions (the new RL data is poor because the policy just started). The SAC gradient overwrites BC weights, the policy gets worse, RL data gets worse — feedback loop.

**Current status:** Unmitigated. Future options: (a) mix offline BC data into replay buffer from start, (b) supervised-RL hybrid loss, (c) smaller LR for first N fine-tuning steps. Abandoned in favor of TargetSpeedReward approach.

### 7. Single-Episode Val Split Crash (Fixed 2026-04-02)

**Symptom:** `pretrain_actor.py` crashed when dataset had only 1 episode with `val_frac=0.2`.

**Root cause:** `max(1, int(1 * 0.2)) = 1` took the whole dataset as val, leaving 0 train steps.

**Fix:** `n_val = int(E * val_fraction) if val_fraction > 0 else 0` + skip val loop when val dataset is empty.

### 8. Reward Always Negative Early Training (Gen 1)

**Root cause:** Crash component weight 10.0 dominated all other signals. Progress signal produced values ~0.00035/step — swamped by crash penalty. Speed multiplier initialized neutrally but fell below 0 after first slow episode, compressing all rewards.

**Fix:** Bounded `[-1, 1]` redesign, crash → hard override, speed multiplier removed.

---

## 11. Checkpointing

### SAC Checkpoint Format (`sac.save()`)

```python
{
    "config":           {obs_dim, action_dim, hidden_units, lr, gamma, tau, target_entropy},
    "policy":           policy.state_dict(),
    "twin_q":           twin_q.state_dict(),
    "twin_q_target":    twin_q_target.state_dict(),
    "log_alpha":        log_alpha.detach().cpu(),   # scalar tensor
    "policy_optimizer": policy_optimizer.state_dict(),
    "q_optimizer":      q_optimizer.state_dict(),
    "alpha_optimizer":  alpha_optimizer.state_dict(),
}
```

Config stored alongside weights for `SAC.from_checkpoint()` — reconstruct architecture without external config. Optimizer states saved for true resumption (Adam maintains moment estimates; losing them causes initial large gradient step on restart).

### Warm-start Design (SAC_TargetSpeed ← 150_jitter)

Load policy + critics + optimizers from `150_jitter`. Then reset alpha:
```python
sac.log_alpha.data.fill_(0.0)   # alpha = exp(0) = 1.0
sac.alpha_optimizer = Adam([sac.log_alpha], lr=3e-4)
```

**Why reset alpha:** Old alpha was calibrated to the old reward scale. `TargetSpeedReward` changes the reward magnitude distribution. Inheriting a miscalibrated alpha causes either premature entropy collapse or excessive exploration. Reset to 1.0 lets it re-calibrate over 5k–10k gradient steps under the new reward — a small cost vs. the benefit of keeping trained weights.

**Why init from 150_jitter:** The existing policy has learned track-following, throttle control, and crash avoidance — non-trivial priors that take ~2–4 hours to acquire from random init. The reward change is orthogonal to these priors. Warm-starting saves that exploration time.

---

## 12. Infrastructure and Tooling

### Key Files

| File | Responsibility |
|---|---|
| `gym/sac/train_sac.py` | Original SAC training entry point |
| `trained_models/SAC_TargetSpeed/train_sac.py` | TargetSpeedReward variant entry point |
| `gym/sac/finetune_sac.py` | Fine-tuning entry point (offline/online) |
| `gym/sac/agent.py` | SACAgent: phase loop, latency overlap, checkpointing |
| `gym/sac/sac.py` | SAC algorithm: update, save, load |
| `gym/sac/network.py` | GaussianPolicy, TwinQNetwork |
| `gym/sac/replay_buffer.py` | ReplayBuffer, DualReplayBuffer |
| `gym/our_env.py` | Reward pipeline, obs assembly, telemetry |
| `gym/rewards/components.py` | Individual reward components |
| `gym/rewards/composite.py` | CompositeReward, factory methods |
| `gym/telemetry/base.py` | TelemetryFrame dataclass |
| `gym/telemetry/ac.py` | ACTelemetry parser |
| `gym/ac_lifecycle.py` | AC launch/kill lifecycle management |
| `AICLONE/pretrain_actor.py` | BC pre-training of GaussianPolicy |
| `AICLONE/preprocess_parquet.py` | Demo data preprocessing |
| `AICLONE/finetune_on_demo.py` | Offline SAC fine-tuning on demo data |
| `AICLONE/generate_target_speed.py` | Build target_speed column from demo parquet |

### Run Commands

```bash
# Start SAC_TargetSpeed from 150_jitter warm-start
..\AssetoCorsa\Scripts\python.exe trained_models/SAC_TargetSpeed/train_sac.py \
    --manage-ac --init-from trained_models/SAC/150_jitter/model.pt

# Resume SAC_TargetSpeed (latest.pt exists)
..\AssetoCorsa\Scripts\python.exe trained_models/SAC_TargetSpeed/train_sac.py --manage-ac

# BC pre-train from 150_jitter warm-start
..\AssetoCorsa\Scripts\python.exe AICLONE/pretrain_actor.py \
    --init-from-sac AICLONE/checkpoints/150_jitter_base.pt \
    --out-dir AICLONE/checkpoints/150_bc --epochs 20 --lr 1e-4

# Generate target_speed for monza racing line
..\AssetoCorsa\Scripts\python.exe AICLONE/generate_target_speed.py
```
