# Project Presentation — Requirements

## 🔥 DEADLINE: Wed 2026-04-22, 6:00 pm (STRICT)
Upload `.pptx` or `.pdf` to Canvas. Late = no presentation grade.

## My slot
- **Group 9** (Chandan, solo)
- Presentation order position: **27 of 30** (order list: 27 1 29 24 12 25 14 8 3 21 6 23 7 13 28 2 22 16 17 15 19 4 20 30 10 26 **9** 18 5 11)

## Time limit
- **3 minutes per group** — strict.
- Every 10 s over = cumulative **25% grade penalty**.
- Too short / unclear = also penalised. Practice before presenting.

## Required slide template (4 slides)

| # | Content |
|---|---------|
| 1 | Title + group info |
| 2 | Problem setting: model **input** and **output** |
| 3 | Baseline + delta (what you added on top) |
| 4 | Empirical results, especially vs. baseline |

## Bonus
Up to 5 most innovative projects → **+5 extra credits per member**.

## My content draft

### Slide 1 — Title
- "Vector-Q SAC with Transformer Encoders for Autonomous Racing in Assetto Corsa"
- Chandan Gowda Keelara Shivanna, Group 9

### Slide 2 — Problem setting
- **Input**: rolling 75-frame window of 50-dim telemetry tokens (3 s @ 25 Hz)
- **Output**: continuous 3-D action in [−1, 1] → (steer, throttle, brake) via vJoy
- **Env**: real Assetto Corsa over custom Python ↔ plugin bridge

### Slide 3 — Baseline + delta
- **Baseline**: Scalar-reward Transformer SAC (`transformer_sac/`)
- **Delta**:
  - **Vector Q-head**: Q = [Q_steer, Q_throttle, Q_brake]
  - **Per-action α**: independent entropy tuning per channel
  - **Intent-based reward decomposition**: each action channel earns its own reward (throttle rewarded when below target speed, brake when above, etc.)
  - **6-channel memmap replay buffer**: stratified sampling across pos/neg per channel, disk-backed for 100 K+ windows per channel
  - **ChunkedPrefetcher**: double-buffered background loader → GPU never waits on disk

### Slide 4 — Empirical results
- Baseline (scalar) vs. vector-Q variant:
  - Mean episode reward
  - % positive-reward frames
  - Off-track rate
  - Per-channel entropy/alpha evolution
  - Training wall-clock (thanks to memmap + prefetcher)
- Show reward curves + one per-channel entropy plot.
