# Project Proposal — SUBMITTED ✅

**File:** `C:/Users/chand/Downloads/racing_proposal_final (4).pdf`
**Title:** *Deep Reinforcement Learning for Autonomous Racing Using Telemetry Data*
**Author:** Chandan Gowda Keelara Shivanna (solo, Group 9)
**Submitted:** long back; latest file 2026-04-20 13:58

---

## What the proposal committed to

| Area | Proposal commitment |
|---|---|
| **Algorithm** | Twin Delayed DDPG (TD3) |
| **State** | ~20 telemetry features (speed, RPM, gear, accel, track position, distance from center, turn info) |
| **Control rate** | 60 Hz |
| **Action space** | 3D continuous: steer [−1,1], throttle [0,1], brake [0,1] |
| **Reward components** | progress + speed + track-center penalty + smoothness + crash |
| **Tracks** | Monza, Spa-Francorchamps, Silverstone |
| **Tricks** | Curriculum learning; optional imitation-learning (BC) pre-training |
| **Baselines** | Random, rule-based, imitation-learning |
| **Data** | Self-generated, 500–1000 episodes/track, 50–100 hours |

### Submitted references (LOCKED for lit survey)
1. Lillicrap et al. 2015 — *Continuous Control with Deep RL* (DDPG)
2. Fujimoto et al. 2018 — *Addressing Function Approximation Error in Actor-Critic Methods* (TD3)
3. Fuchs et al. 2020 — *Super-Human Performance in Gran Turismo Sport using DRL*
4. Kendall et al. 2019 — *Learning to Drive in a Day*
5. Riedmiller et al. 2018 — *Learning by Playing* (sim-to-real with proprioceptive sensors)
6. Hester et al. 2018 — *Deep Q-learning from Demonstrations* (DQfD)

---

## ⚠️ Proposal vs. implementation drift

The submitted proposal does **not** match the implemented system. The final report must acknowledge and justify these deviations:

| Area | Proposal | Implemented | Justification to write |
|---|---|---|---|
| Algorithm | **TD3** | **SAC** (Soft Actor-Critic) | SAC's max-entropy objective handles exploration in a real-time, hard-to-reset sim better than TD3's explicit noise; automatic α tuning reduces hyper-param search |
| Encoder | MLP on 20 features | **Transformer, 4L × 4H, d=256, window=75** | Raw telemetry is noisy; a 3 s window + attention captures the temporal structure (braking cue → turn-in) that a per-frame MLP misses |
| Token dim | ~20 features | **50 dims** (includes track-distance ray sensors) | Richer spatial signal |
| Control rate | 60 Hz | **25 Hz** | AC plugin tick + Python network round-trip pins us near 25 Hz; 60 Hz was aspirational |
| Tracks | 3 (Monza, Spa, Silverstone) | 1 (whatever `config.yml` is pointed at) | Time budget; single-track generalisation ablation deferred |
| Imitation learning | Optional BC pre-train | **Not used** (AICLONE preprocess exists but v2_final fine-tunes a learned policy) | SAC replay + vector-Q made BC pre-train unnecessary; still worth mentioning as future work |
| Baselines | Random / rule-based / IL | Scalar-reward Transformer SAC vs. Vector-Q v2 | The in-project ablation (scalar Q vs. vector Q) is a stronger comparison than random/rule-based |
| **NEW** (not in proposal) | — | **Vector Q-head**, **per-action α**, **intent-based reward decomposition**, **6-channel memmap replay** | These are the innovation points — emphasise in slides/report |

---

## How to frame the drift in the final report
1. Open Methods with: "*We originally proposed TD3 as our algorithm. During implementation we adopted SAC and extended it with a vector-valued Q-function because…*"
2. Keep the proposal's reward taxonomy (progress/speed/center/smoothness/crash) — you implemented exactly these five terms, just decomposed per action channel.
3. Grade impact should be small: the grading rubric cares about **soundness + originality of method** and **completeness of experiments**, not strict adherence to the proposal.
