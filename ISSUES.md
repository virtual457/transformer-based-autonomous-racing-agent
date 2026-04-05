# Issue Tracker — Assetto Corsa Deep RL Project

**Format:** Each issue has a phase, priority, status, and what it blocks.
**Priority logic:** CRITICAL = blocks multiple phases | HIGH = blocks one phase | MEDIUM = workaround exists | LOW = cosmetic/minor

---

## Open Issues

---

### ISSUE-011 — BC model turns left on straight (3 root causes identified)
**Status:** ✅ FULLY RESOLVED (2026-03-24) — retrained; checkpoint at checkpoints/bc_monza_v2/best.pt
**Phase:** Phase 4 (Variant B)
**Priority:** HIGH
**Blocks:** Phase 4 (Variant B imitation pre-training); any live BC evaluation

**Problem:**
The BC model at `checkpoints/bc_monza_full/best.pt` turns left on Monza main straight.

**Root Cause Analysis (2026-03-24 by ml-engineer):**

**ROOT CAUSE 1 — CONFIRMED: Absent history at inference causes LEFT bias**
At speed >= 20 m/s with no action history (channels 38-49 all zero), the model outputs
steer_SAC = -0.043 to -0.048 regardless of gap. This maps to policy steer ~0.476 (LEFT).
Cause: the model uses past-action channels (38-49) to condition steering. At inference
start-of-episode (history empty), all 12 history slots are zero. The model never saw
this exact combination during training (training data always starts mid-lap at 41 m/s
where history was naturally populated). The OOD input of zeros-with-high-speed
consistently biases the model toward a small left turn.

**ROOT CAUSE 2 — CONFIRMED: steerAngle normalization mismatch in obs channels 38-40, 47**
- `preprocess_bc.py` normalises steerAngle in obs by `302.4` (read from steer_map.csv)
- `ac_env.py` normalises steerAngle in obs by `450` (hardcoded `obs_channels_info['steerAngle'] = 450`)
- Training obs steer = live obs steer × 1.489
- Effect on straight (steer ≈ 0): ~-0.003 shift in predicted steer (negligible)
- Effect in corners where steerAngle > 100°: ~0.15 shift in steer channel → model sees
  wrong past-action signal → corner tracking errors compound

**ROOT CAUSE 3 — CONFIRMED: Low-speed data contamination**
1,500 of 48,373 training steps (3%) are at speed < 5 m/s with steer std = 0.65
(erratic driver repositioning/pitting). These poison the model's low-speed predictions
with high variance. At a dead-stop start in AC, model is in this contaminated regime.

**Root Causes NOT found:**
- No global left-steer bias in the training actions (mean = +0.007, near zero)
- No wrong sign convention in the NeuralPolicy → VJoyControl → set_controls chain
- No mismatch in pedal/brake encoding (both use accStatus round-trip correctly)
- No wrong steer_max in action targets (both preprocess_bc.py and DataLoader use 302.4)

**Fixes (in priority order):**

**Fix 1 — HIGH priority (addresses ROOT CAUSE 1): Warm-start history at inference**
IMPLEMENTED (2026-03-24 ml-engineer).
- `gym/preprocess_bc.py` `build_obs()`: when `len(history) < PAST_ACTIONS_WINDOW`,
  missing past-action slots (channels 38-46) are now filled with the *current state's
  own action* instead of zeros. Missing prev-basic-obs slots (channels 50-124) are
  filled with the current state's basic obs. New data must be regenerated from pkl
  files before retraining.
- `assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaEnv/ac_env.py` `get_obs()`: same
  warm-start applied symmetrically — missing history slots filled with current state's
  action / basic obs instead of zeros. Also preserves `actions_diff` using last real
  history entry when available (previously always zeroed in the short-history branch).
Training and inference are now consistent: the model will see the same representation
at episode start that it was trained on.

**Fix 2 — HIGH priority (addresses ROOT CAUSE 2): Fix steer scale in preprocess_bc.py**
IMPLEMENTED (2026-03-24 ml-engineer).
`OBS_CHANNEL_SCALES['steerAngle']` changed from `302.4` to `450` in `preprocess_bc.py`
line 120. Now matches `ac_env.py obs_channels_info['steerAngle'] = 450`.
.npz regenerated. `steer_obs_scale: 450.0` confirmed in metadata.

**Fix 3 — MEDIUM priority (addresses ROOT CAUSE 3): Filter low-speed training data**
IMPLEMENTED (2026-03-24 ml-engineer).
`process_files()` now accepts `min_speed_ms=5.0`; steps below threshold have history
advanced but are NOT added to the dataset.  Added as `--min-speed` CLI arg (default 5.0).
1,021 low-speed steps removed from KC_Sim_Racer episode (adrianremonda episode had 0).
New dataset: 37,374 steps (was 48,373; -1,021 low-speed + large out-lap recategorisation).
Speed floor confirmed: min=5.02 m/s in regenerated .npz.

**Immediate workaround (no retraining):**
Pass `invert_steer=False` and run with `--obs-warmup-steps 3` if that flag is added.
Or use `NeuralPolicy(invert_steer=True)` to confirm/deny if the issue is a sign flip
(if invert_steer=True makes it drive straight, the issue is a sign flip elsewhere).

**Files changed (all 3 fixes done):**
- `gym/preprocess_bc.py` `build_obs()` — warm-start history padding (Fix 1, DONE)
- `assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaEnv/ac_env.py` `get_obs()` — warm-start history padding (Fix 1, DONE)
- `gym/preprocess_bc.py` line 120 — steer scale 302.4 → 450 (Fix 2, DONE)
- `gym/preprocess_bc.py` `process_files()` — min_speed_ms=5.0 filter + --min-speed CLI arg (Fix 3, DONE)
- `data/processed/monza_ks_mazda_miata_mlp.npz` — regenerated with all fixes applied
- `data/processed/monza_ks_mazda_miata_mlp.json` — updated metadata (steer_obs_scale=450, min_speed_ms=5.0)

**Action required:** DONE. BC model retrained 2026-03-24. Best checkpoint at checkpoints/bc_monza_v2/best.pt (best val=0.018358 at epoch 12). Ready for live evaluation.

---

### ISSUE-010 — Vjoy.ini active profile shows INPUT_METHOD=JOYSTICK instead of WHEEL
**Status:** 🟡 OPEN
**Phase:** Phase 1
**Priority:** LOW
**Blocks:** Nothing currently — vJoy controls confirmed working in JOYSTICK mode

**Problem:**
`preflight.py` detects `INPUT_METHOD=JOYSTICK` in the active `Vjoy.ini` profile. ISSUE-008 fix specified WHEEL mode, but controls are verified working with JOYSTICK mode (car drives, FullThrottlePolicy reaches 31 m/s, crashes at Monza chicane as expected).

**Resolution when prioritized:**
Open AC Content Manager → Controls → Load the savedsetups `Vjoy.ini` (with WHEEL) → Save as active profile. Verify preflight shows `[OK] WHEEL`.

---

### ISSUE-009 — vJoy axes snap to wrong state on process exit causing runaway throttle
**Status:** 🔴 OPEN
**Phase:** Phase 1
**Priority:** HIGH
**Blocks:** Reproducible data collection (car accelerates after every collection run)

**Problem:**
After `test_our_env.py` exits, the car accelerates at max throttle even though `_neutralize_vjoy()` runs successfully (log shows both axis set and relinquish). Two runs of `_neutralize_vjoy()` are seen (one from `env.close()`, one from atexit), both claim `"no runaway throttle"`, but the car still accelerates.

**Root cause hypothesis:**
Live `Vjoy.ini` still has `INPUT_METHOD=JOYSTICK` (not WHEEL). In JOYSTICK/DirectInput mode, AC reads axes in the range [-32767, +32767] centered at 0. When vJoy device is relinquished, axes snap to 0 = **center = 50% throttle** in AC's calibration. In WHEEL mode, axes are [0, 32767] unidirectional so 0 = no throttle.

ISSUE-008 fix (INPUT_METHOD=WHEEL) may not have been saved to the active profile in AC Content Manager — the savedsetups file may have been updated but not loaded as active.

**Evidence:**
- `_neutralize_vjoy()` logs show correct execution: `wAxisX=16384 wAxisY=0 wAxisZ=0`
- Car still accelerates post-exit → 0 on throttle axis = half throttle in JOYSTICK mode
- Live `C:\Users\chand\Documents\Assetto Corsa\cfg\controllers\Vjoy.ini` still shows `INPUT_METHOD=JOYSTICK`

**Resolution needed:**
1. Open AC Content Manager → Controls → Load `Vjoy.ini` (the savedsetups version that has WHEEL mode)
2. Click Save so it becomes the active profile
3. Verify `C:\Users\chand\Documents\Assetto Corsa\cfg\controllers\Vjoy.ini` now shows `INPUT_METHOD=WHEEL`
4. Re-run `test_our_env.py` to confirm car stays still after exit

Assign to: **vjoy-engineer** when prioritized.

---

### ISSUE-001 — Demo data files are missing
**Status:** 🔴 OPEN
**Phase:** Pre-Phase 4 (must resolve before Variant B training)
**Priority:** CRITICAL
**Blocks:** Phase 4 (Variant B: imitation pre-training), Phase 7 (ablation table — B is a required variant)

**Problem:**
`ac_offline_train_paths.yml` references human lap recordings by ID (e.g. `20240229_HC`) but the `data/` directory is completely empty. No `.pkl` or `.parquet` demo files exist locally. `Agent.load_pre_train_data()` will fail immediately.

**Resolution needed:**
Locate where the original demo files are stored (original repo, external drive, cloud). Download and place them in `data/` or update `ac_offline_train_paths.yml` with correct paths. Assign to: **systems-engineer** to locate + **env-engineer** to verify `data_loader.py` can parse them.

---

### ISSUE-002 — Spa-Francorchamps not configured
**Status:** ✅ RESOLVED (2026-03-22)
**Decision:** Use Silverstone everywhere — training, evaluation, and report. Already fully configured in `config.yaml`. Spa dropped.

---

### ISSUE-003 — `PAST_ACTIONS_WINDOW` is hardcoded
**Status:** 🔴 OPEN
**Phase:** Phase 2 (env-engineer task 2.4)
**Priority:** HIGH
**Blocks:** Phase 5 (Variant C: 16-step flat history cannot run until this is configurable)

**Problem:**
`ac_env.py` line ~33 defines `PAST_ACTIONS_WINDOW = 3` as a module-level constant. Variant C needs this to be 16. It must be read from `config.yml` instead.

**Resolution needed:**
`env-engineer` must make `PAST_ACTIONS_WINDOW` a config parameter during Phase 2 (task 2.4). This is already on the ROADMAP — just needs to be tracked as a hard blocker.

---

### ISSUE-004 — Variant A instability has no early-warning protocol
**Status:** 🟡 OPEN
**Phase:** Phase 3
**Priority:** HIGH
**Blocks:** Phases 4, 5, 6, 7 — all variants depend on a healthy Variant A

**Problem:**
If Variant A reward is flat at 50k steps, we are already ~2 hours into a dead run. There is no defined escalation path: who diagnoses it, what they check, how fast they re-run.

**Resolution needed:**
Before launching Variant A, define the go/no-go criteria and diagnosis checklist. Assign to: **ml-engineer** to define, **team-lead** to approve before training starts.

---

### ISSUE-005 — Transformer (Variant D) has no VRAM budget plan
**Status:** 🟡 OPEN
**Phase:** Phase 6
**Priority:** MEDIUM
**Blocks:** Phase 6 (Variant D may OOM on RTX 4060 8GB)

**Problem:**
Transformer encoder with d_model=64, 2 layers, 4 heads + replay buffer + SAC networks — total VRAM footprint is unknown. RTX 4060 has 8GB. If it OOMs, Phase 6 is blocked.

**Resolution needed:**
`ml-engineer` must run a VRAM dry-run before committing to full Variant D training. If >7GB, reduce replay buffer size or d_model. Track peak VRAM with `nvidia-smi dmon` or `torch.cuda.max_memory_allocated()`.

---

### ISSUE-006 — Team-lead agent invoked as wrong type (no color shown)
**Status:** 🟠 OPEN (process fix)
**Phase:** All phases (workflow issue)
**Priority:** MEDIUM
**Blocks:** Correct agent activation UI; agent color visibility

**Problem:**
The team-lead was invoked using `subagent_type: general-purpose` instead of `subagent_type: team-lead`. The custom agent definition in `.claude/agents/team-lead.md` (blue color, project memory, specialized persona) was not used.

**Resolution needed:**
Always invoke custom agents by their slug (`team-lead`, `env-engineer`, `ml-engineer`, etc.) not `general-purpose`. Fix in all future invocations.

---

### ISSUE-007 — Architecture doc lists Python 3.9 but venv is Python 3.12
**Status:** ✅ RESOLVED
**Phase:** Phase 1 (resolved)
**Priority:** N/A

**Problem:** Memory doc said Python 3.9. Actual venv is Python 3.12.7.
**Resolution:** `gym==0.21.0` was patched for Python 3.12 compatibility during Phase 1 setup. All imports verified. Closed.

---

---

### ISSUE-008 — vJoy input configuration incorrect
**Status:** ✅ RESOLVED (2026-03-22)
**Phase:** Phase 1 (tasks 1.2 and 1.6)
**Priority:** CRITICAL (was)
**Blocked:** Phase 1 (tasks 1.2 and 1.6), and therefore ALL subsequent phases (2 through 8)

**Resolution (2026-03-22):**
Fixed by vjoy-engineer. Root cause was the AC controller profile configuration, not the axis range or PGUID. Three settings were required:
- `INPUT_METHOD=WHEEL` — tells AC to treat vJoy Device 1 as a wheel/joystick input rather than keyboard
- `GAS_ON_AXIS=1` — maps throttle to a dedicated axis instead of sharing with steering
- `BRAKE_ON_AXIS=1` — maps brake to a dedicated axis instead of sharing with steering

With these three values set in the active AC controller profile, vJoy Device 1 correctly receives steer, throttle, and brake control signals from the Python client. Verified by vjoy-engineer with live axis movement in AC calibration screen.

**Original problem (for reference):**
vJoy Device 1 axes were confirmed installed but inputs were not correctly configured for AC to receive control signals from the Python client. Three suspected root causes were investigated:

1. Axis value range mismatch (car_control.py SCALE=16384, range [0,32768])
2. AC controller binding (PGUID in Vjoy.ini)
3. INPUT_METHOD/axis assignment in active controller profile (this was the actual cause)

---

## Resolved Issues

| ID | Summary | Resolved In | Date |
|---|---|---|---|
| ISSUE-013 | Unnormalized obs channels (LastFF, curvature) could destabilize networks | sac.py — clamp(-3,3) on obs/next_obs in update() and select_action() | 2026-03-29 |
| ISSUE-008 | vJoy input configuration incorrect | Phase 1 — INPUT_METHOD=WHEEL, GAS_ON_AXIS=1, BRAKE_ON_AXIS=1 | 2026-03-22 |
| ISSUE-007 | Python 3.9 vs 3.12 mismatch | Phase 1 | 2026-03-21 |
| ISSUE-002 | Spa-Francorchamps not in track config | Dropped — Silverstone everywhere | 2026-03-22 |

---

## Issue Priority Legend

| Priority | Meaning |
|---|---|
| CRITICAL | Blocks 2+ phases or the entire ablation study |
| HIGH | Blocks exactly one phase; no workaround |
| MEDIUM | Has a workaround or workaround being evaluated |
| LOW | Does not block work; cosmetic or minor |

---

*Last updated: 2026-03-29 by ml-engineer — ISSUE-013 RESOLVED: obs clamp(-3,3) added to sac.py to guard against out-of-range channels (LastFF/1.0, curvature/0.1).*
