# Final Report — Requirements

## Format
- **NeurIPS template** (LaTeX preferred; Word accepted if it mimics LaTeX style closely — significant deviation = point deductions)
- **6–8 pages** excluding references

## NO late days allowed. Strict final-grade deadline.

## Required sections (exact order)

1. **Introduction** — problem definition + motivation
   - If using any open-source code, **disclose at end of introduction** (failing = point deduction)
2. **Related Work** — literature survey, **≥6 papers**
3. **Methods**
   - Overview of method
   - Intuition on why it beats the baseline
   - Details of models + algorithms developed
4. **Experiments**
   - Testbed description + questions your experiments answer
   - Details of experiments + results
5. **Conclusion** — discussion + future work

## Grading breakdown

| %  | Criterion |
|----|-----------|
| 10 | Intro + literature survey |
| 30 | Proposed method (soundness + originality) |
| 30 | Correctness, completeness, difficulty of experiments + figures |
| 10 | Empirical analysis of results + methods |
| 20 | Quality of writing |

## Bonus
Up to 5 most innovative projects → **+5 extra points per member**.

## Open-source disclosures (MUST appear in intro)
- `assetto_corsa_gym` (Remonda et al. — environment + plugin)
- PyTorch, NumPy (standard libs — may not require disclosure, but safer to list)
- `mss`, `opencv-python` (demo video capture)
- Any pretrained weights from `AICLONE/` dataset if used

## Experimental narrative (use this as the Methods → Experiments backbone)

Present the work as an **algorithmic evolution**, each step motivated by a limitation of the previous:

1. **Proposed baseline: TD3** (Fujimoto et al. 2018) — cited as the approach the proposal promised. Not implemented; we switched to SAC before any training runs. Justify the swap in Methods (see reasons below).

2. **Variant A — Flat SAC** (`gym/sac/`)
   Standard SAC with MLP encoder on telemetry. Purpose: establish that a max-entropy agent can learn *anything* on Assetto Corsa.
   Limitation → single-frame state cannot distinguish "entering a turn" from "leaving a turn" at the same position.

3. **Variant B — Transformer SAC, scalar reward** (`gym/transformer_sac/`)
   Adds a 4-layer, 4-head Transformer encoder over a 75-frame × 50-dim token window (3 s @ 25 Hz). Single scalar reward, single scalar Q, single scalar α.
   Limitation → the three actuators (steer, throttle, brake) fight for the same scalar reward; throttle and brake compete over one speed-tracking term.

4. **Variant C — Vector-Q Transformer SAC** (`gym/transformer_sac_vectorq_v2/` + `_v2_final/`)
   Decomposes Q, α, and reward per action channel. Adds a 6-channel memmap replay buffer with stratified pos/neg sampling per channel. **Our final proposed method.**

**Why this framing is honest and strong:**
- Each step is a real implemented folder with real training runs.
- TD3 is cited (as proposed + motivating literature) but not claimed as a run.
- The scalar→vector progression is a clean story with a concrete architectural novelty per stage.

### Why SAC replaced TD3 (write this in Methods)
- Entropy-regularised exploration prevents the always-straight / always-throttle collapse TD3's deterministic policy would fall into under noisy telemetry.
- Auto-tuned α removes a hyperparameter search we cannot afford with 10-min collection phases.
- Per-action α (extended in Variant C) gives each actuator its own exploration temperature — impossible with TD3.
- TD3's policy-smoothing + delayed-update tricks are hard-coded fixes for a deterministic policy; SAC gets stability from the stochastic policy itself.

## ⚠️ Proposal drift to justify
The submitted proposal committed to **TD3 + MLP on 20 telemetry features at 60 Hz across 3 tracks**.
The implementation is **SAC + Transformer + Vector-Q at 25 Hz on 1 track**.
See [01_PROPOSAL.md](01_PROPOSAL.md) "Proposal vs. implementation drift" for the justification table.
**Open the Methods section with an explicit acknowledgement** of these deviations and the reasons.

## Locked lit-survey papers (from submitted proposal)
These 6 are already committed; add papers below but do not silently drop any:
1. Lillicrap 2015 (DDPG), 2. Fujimoto 2018 (TD3), 3. Fuchs 2020 (GT Sport),
4. Kendall 2019 (Drive in a Day), 5. Riedmiller 2018, 6. Hester 2018 (DQfD)

## Suggested additional references (to reach ≥6 required for report AND to cover the actual method)
- Haarnoja et al. 2018 — *Soft Actor-Critic* (to justify algorithm swap from TD3)
- Vaswani et al. 2017 — *Attention Is All You Need* (Transformer encoder)
- Van Seijen et al. 2017 — *Hybrid Reward Architecture* (vector-valued reward decomposition — key novelty justification)
- Remonda et al. 2024 — *Assetto Corsa Gym* benchmark paper (our environment)

## Figures / tables to produce
- Training curves: total Q-loss, per-channel Q-loss, per-channel entropy, per-channel α
- Reward curves: scalar baseline vs. vector variant (mean + std across seeds if possible)
- Ablation table: baseline vs. each delta (vector Q only / per-action α only / memmap buffer only / full stack)
- Episode trajectory heatmap (position) — before vs. after fine-tune
- Action distribution histograms — detect degenerate always-straight / always-full-throttle policies
