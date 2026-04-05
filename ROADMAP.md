# Project Roadmap — Assetto Corsa Deep RL Racing Agent

**Deadline:** Report due April 22, 2026 | Presentation April 23, 2026
**Started:** March 21, 2026
**Hardware:** NVIDIA RTX 4060 (8GB VRAM)
**Track:** Silverstone | **Car:** dallara_f317

---

## Team

| Agent | Role | Owns |
|---|---|---|
| `team-lead` | Planning, delegation, unblocking | This file, ISSUES.md, agent memory |
| `systems-engineer` | Infrastructure, venv, AC plugin, comms | Plugin files, venv, ports |
| `vjoy-engineer` | vJoy device config, AC controller binding, live controls | `vjoy.py`, `car_control.py`, `Vjoy.ini` |
| `env-engineer` | Environment design, reward, observation space | `ac_env.py`, `config.yml` |
| `ml-engineer` | Algorithm, network architecture, all variants | `network.py`, `sac.py`, `agent.py` |
| `data-analyst` | Training analysis, ablation study, plots | WandB outputs, `results/` |
| `technical-writer` | Report, presentation, literature survey | `Docs/` |

**Agent Protocol:** Every agent reads ROADMAP.md + ISSUES.md at session start. team-lead assigns via these files. Agents update them when tasks complete or are blocked.

---

## Variant Plan

| Variant | Description | Status | Owner |
|---|---|---|---|
| A | SAC + MLP baseline | ⬜ NOT STARTED | `ml-engineer` |
| B | SAC + MLP + imitation pre-training | ⬜ NOT STARTED | `ml-engineer` |
| C | SAC + MLP + 16-step flat history | ⬜ NOT STARTED | `ml-engineer` + `env-engineer` |
| D | SAC + Transformer encoder + 16-step history | ⬜ NOT STARTED | `ml-engineer` |
| E | SAC + CNN + screen capture | ⬜ STRETCH GOAL | `ml-engineer` |

---

## Phase 1 — Environment Setup
**Owner:** `systems-engineer`
**Status:** 🔴 BLOCKED — vJoy inputs not correctly configured (ISSUE-008)
**Must complete before:** any training can begin

| # | Task | Status | Notes |
|---|---|---|---|
| 1.1 | Verify Assetto Corsa installed & launches | ✅ DONE | D:\SteamLibrary\steamapps\common\assettocorsa |
| 1.2 | Install vJoy, verify 3 axes on Device 1 | 🔴 BLOCKED | v2.1.9 installed but vJoy inputs are NOT correctly configured — ISSUE-008. Must verify: vJoyConf axes range 0–32768, PGUID matches, AC loads Vjoy.ini as active profile |
| 1.3 | Copy AC plugin to `apps\python\sensors_par\` | ✅ DONE | Copied to D:\SteamLibrary\...\apps\python\sensors_par |
| 1.4 | Copy `windows-libs` to `system\x64\` | ✅ DONE | DLLs\ and Lib\ copied to system\x64 |
| 1.5 | Copy `Vjoy.ini` to savedsetups | ✅ DONE | Vjoy.ini + WASD.ini copied to C:\Users\chand\Documents\Assetto Corsa\cfg\controllers\savedsetups |
| 1.6 | Configure AC: FPS=50, sensor_par ON, vJoy loaded | 🔴 BLOCKED | FPS=50 set, sensor_par enabled — but vJoy not confirmed as active AC input device. Depends on ISSUE-008 resolution. Must confirm: Vjoy.ini loaded as active profile, vJoy Device 1 bound in Controls screen |
| 1.7 | Set all driving assists correctly | ✅ DONE | All assists configured |
| 1.8 | Install Custom Shaders Patch (CSP) | ✅ DONE | Installed via Content Manager |
| 1.9 | Fix Python venv — install 9 missing packages | ✅ DONE | gym==0.21.0 patched for Python 3.12; all packages installed |
| 1.10 | Verify CUDA: `torch.cuda.is_available() == True` | ✅ DONE | torch 2.6.0+cu124, CUDA: True on RTX 4060 |
| 1.11 | Verify all imports succeed | ✅ DONE | All imports pass |
| 1.12 | Smoke test: AC hotlap + port 2345 live | ✅ DONE | Port 2345 bound, acs.exe confirmed |
| 1.13 | Run `test_client.ipynb` — live telemetry | ✅ DONE | 100 steps collected, speed/steer plots non-zero — but controls may not have been applied (vJoy issue) |

---

## Phase 2 — Reward Function & Environment
**Owner:** `env-engineer`
**Status:** ⬜ NOT STARTED
**Depends on:** Phase 1 complete

| # | Task | Status | Notes |
|---|---|---|---|
| 2.1 | Rewrite `ac_env.py:get_reward()` — 5-component reward | ⬜ TODO | progress, speed, gap, smoothness, crash |
| 2.2 | Add `reward_weights` block to `config.yml` | ⬜ TODO | Each component separately configurable |
| 2.3 | Log each reward component separately to WandB | ⬜ TODO | Needed for ablation analysis |
| 2.4 | Make `PAST_ACTIONS_WINDOW` configurable via config | ⬜ TODO | For Variant C (16-step history) |
| 2.5 | Validate reward function on short test run | ⬜ TODO | All 5 components visible in WandB |

---

## Phase 3 — Variant A: SAC Baseline
**Owner:** `ml-engineer`
**Status:** ⬜ NOT STARTED
**Depends on:** Phase 2 complete
**Gates:** Variants B, C, D

| # | Task | Status | Notes |
|---|---|---|---|
| 3.1 | `wandb login`, tag run as `variant-a` | ⬜ TODO | |
| 3.2 | Run `python train.py --algo sac` | ⬜ TODO | Silverstone, dallara_f317 |
| 3.3 | Confirm reward trends positive by step 50k | ⬜ TODO | If flat, escalate to team-lead |
| 3.4 | Run to 200k+ steps, save best checkpoint | ⬜ TODO | ~2.2 hours wall time |
| 3.5 | Flag results to `data-analyst` | ⬜ TODO | |

---

## Phase 4 — Variant B: Imitation Pre-training
**Owner:** `ml-engineer`
**Status:** ⬜ NOT STARTED
**Depends on:** Phase 3 complete (Variant A stable)

| # | Task | Status | Notes |
|---|---|---|---|
| 4.1 | Verify human demo data exists in `ac_offline_train_paths.yml` | ⬜ TODO | |
| 4.2 | Run with `load_offline_data: True`, `pre_train: True` | ⬜ TODO | Tag as `variant-b` in WandB |
| 4.3 | Confirm `EnsembleBuffer` seeds correctly before online steps | ⬜ TODO | |
| 4.4 | Run to 200k+ steps, compare to Variant A | ⬜ TODO | |
| 4.5 | Flag results to `data-analyst` | ⬜ TODO | |

---

## Phase 5 — Variant C: 16-Step Flat History
**Owner:** `env-engineer` + `ml-engineer`
**Status:** ⬜ NOT STARTED
**Depends on:** Phase 3 complete

| # | Task | Status | Notes |
|---|---|---|---|
| 5.1 | Add `obs_history_steps: 16` to `config.yml` | ⬜ TODO | `env-engineer` |
| 5.2 | Make `PAST_ACTIONS_WINDOW` read from config | ⬜ TODO | `env-engineer` in `ac_env.py` |
| 5.3 | Verify obs dimension expands correctly | ⬜ TODO | `env-engineer` |
| 5.4 | Run with flat 16-step history, tag `variant-c` | ⬜ TODO | `ml-engineer` |
| 5.5 | Flag results to `data-analyst` | ⬜ TODO | |

---

## Phase 6 — Variant D: Transformer Encoder
**Owner:** `ml-engineer`
**Status:** ⬜ NOT STARTED
**Depends on:** Phase 5 complete (Variant C stable)
**This is the main architectural contribution**

| # | Task | Status | Notes |
|---|---|---|---|
| 6.1 | Add `TransformerEncoderNetwork` class to `network.py` | ⬜ TODO | d_model=64, 2 layers, 4 heads |
| 6.2 | Add `use_transformer: False` flag to `config.yml` | ⬜ TODO | |
| 6.3 | Wire Transformer into `GaussianPolicy` and `TwinnedStateActionFunction` | ⬜ TODO | |
| 6.4 | Verify VRAM stays under 7GB on RTX 4060 | ⬜ TODO | |
| 6.5 | Run with Transformer, tag `variant-d` | ⬜ TODO | |
| 6.6 | Extract attention maps for visualization | ⬜ TODO | Which timesteps attended to at corners? |
| 6.7 | Flag results to `data-analyst` | ⬜ TODO | |

---

## Phase 7 — Analysis & Ablation Study
**Owner:** `data-analyst`
**Status:** ⬜ NOT STARTED
**Depends on:** At least Variants A + B complete; ideally all 4

| # | Task | Status | Notes |
|---|---|---|---|
| 7.1 | Pull all variant runs from WandB | ⬜ TODO | |
| 7.2 | Produce `results/ablation_table.csv` | ⬜ TODO | Convergence step, best lap, mean reward |
| 7.3 | Produce `results/reward_curves.png` | ⬜ TODO | All variants, same axes |
| 7.4 | Compute sample efficiency ratios | ⬜ TODO | Convergence step of A / variant |
| 7.5 | Write interpretation: what does C vs D show? | ⬜ TODO | The architectural claim |
| 7.6 | Hand off to `technical-writer` | ⬜ TODO | |

---

## Phase 8 — Report & Presentation
**Owner:** `technical-writer`
**Status:** ⬜ NOT STARTED
**Can start Methods section now; Results needs Phase 7**

| # | Task | Status | Notes |
|---|---|---|---|
| 8.1 | Draft Introduction section | ⬜ TODO | Can start now |
| 8.2 | Draft Related Work (4-6 papers, 3 clusters) | ⬜ TODO | Can start now |
| 8.3 | Draft Methods section (env, reward, SAC, Transformer) | ⬜ TODO | After Phase 2 |
| 8.4 | Draft Results section | ⬜ TODO | After Phase 7 |
| 8.5 | Draft Discussion + Conclusion | ⬜ TODO | After Phase 7 |
| 8.6 | Format to NeurIPS template | ⬜ TODO | |
| 8.7 | Build 10-slide presentation outline | ⬜ TODO | |
| 8.8 | Final proofread + submission | ⬜ TODO | Due April 22 |

---

## Phase 9 — Stretch Goal: Variant E (CNN + Visual)
**Owner:** `ml-engineer`
**Status:** ⬜ STRETCH — only if Phases 3-6 done with 2+ weeks to spare

| # | Task | Status | Notes |
|---|---|---|---|
| 9.1 | Enable screen capture (`screen_capture_enable: True`) | ⬜ TODO | Dual-buffer already implemented |
| 9.2 | Add CNN encoder to `network.py` | ⬜ TODO | Embed image → concat with telemetry |
| 9.3 | Reduce replay buffer to 200k steps (VRAM) | ⬜ TODO | |
| 9.4 | Run Variant E, tag `variant-e` | ⬜ TODO | |

---

## Priority Order (right now)

```
🔴 PRIORITY 1 — Phase 1 (setup) — BLOCKING EVERYTHING — vJoy ISSUE-008 must be resolved first
                 systems-engineer must fix: vJoyConf axes, PGUID, AC active profile, live controls test
🟡 PRIORITY 2 — Phase 2 (reward function) — on hold until Phase 1 fully unblocked
🟡 PRIORITY 3 — Phase 8 (report intro + related work) — can start NOW, no blockers
🔴 HOLD — Phase 3 (Variant A) — blocked, do not start until Phase 1 + 2 confirmed
🔴 HOLD — Phases 4,5,6 (Variants B,C,D) — blocked downstream of Phase 3
🔴 HOLD — Phase 7 (analysis) — blocked downstream
⚪ PRIORITY 7 — Phase 9 (Variant E) — stretch only
```

---

## Status Legend

| Symbol | Meaning |
|---|---|
| ⬜ TODO | Not started |
| 🔄 IN PROGRESS | Currently being worked on |
| ✅ DONE | Complete and verified |
| 🔴 BLOCKED | Cannot proceed, needs unblocking |
| ⚠️ AT RISK | Running behind, needs attention |

---

---

## Open Issues

See [ISSUES.md](ISSUES.md) for full details.

| ID | Summary | Priority | Blocks |
|---|---|---|---|
| ISSUE-008 | vJoy input configuration incorrect | CRITICAL | Phase 1 (1.2, 1.6) → ALL phases |
| ISSUE-001 | Demo data files missing (data/ empty) | CRITICAL | Phase 4 (Variant B) |
| ~~ISSUE-002~~ | ~~Spa-Francorchamps not in track config~~ | ~~HIGH~~ | ✅ Resolved — Silverstone everywhere |
| ISSUE-003 | `PAST_ACTIONS_WINDOW` hardcoded | HIGH | Phase 5 (Variant C) |
| ISSUE-004 | Variant A instability has no early-warning protocol | HIGH | Phases 4,5,6,7 |
| ISSUE-005 | Transformer Variant D no VRAM budget plan | MEDIUM | Phase 6 |
| ISSUE-006 | Team-lead invoked as wrong agent type | MEDIUM | Workflow |

---

*Last updated: 2026-03-22 by team-lead — Phase 1 reopened: ISSUE-008 vJoy inputs not correctly configured*
