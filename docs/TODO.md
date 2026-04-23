# Course Project — TODO

Full requirement details live in:
`gym/transformer_sac_vectorq_v2_final/docs/`
- [01_PROPOSAL.md](gym/transformer_sac_vectorq_v2_final/docs/01_PROPOSAL.md)
- [02_PRESENTATION.md](gym/transformer_sac_vectorq_v2_final/docs/02_PRESENTATION.md)
- [03_FINAL_REPORT.md](gym/transformer_sac_vectorq_v2_final/docs/03_FINAL_REPORT.md)
- [04_CODE_SUBMISSION.md](gym/transformer_sac_vectorq_v2_final/docs/04_CODE_SUBMISSION.md)

Project: **Vector-Q SAC with Transformer Encoders for Autonomous Racing in Assetto Corsa**
Group: **9 — Chandan Gowda Keelara Shivanna (solo)**
Presentation slot: **27 of 30**

---

## 🔥 URGENT — Wed 2026-04-22 6:00 pm (strict)

### Presentation slides (4 slides, 3-min max)
- [ ] Slide 1 — title + group info
- [ ] Slide 2 — problem setting (input/output)
- [ ] Slide 3 — baseline + deltas (vector Q, per-action α, intent reward, memmap buffer)
- [ ] Slide 4 — results: baseline vs. vector-Q (reward curves, off-track %, entropy/α)
- [ ] Practice timing (≤3 min; 25% penalty per 10 s over)
- [ ] Upload `.pptx` or `.pdf` to Canvas before 6 pm

---

## Experiments needed before the final report

### Training runs
- [ ] Confirm baseline (`transformer_sac/`) has a completed run on the same track
- [ ] Fine-tune v2_final for N epochs offline (`train.py --epochs 3`)
- [ ] Collect evaluation episodes with `collect.py` (no gradient updates) post-training
- [ ] Save final `latest.pt` as the "submission checkpoint"

### Demo artefacts
- [ ] Run `demo.py --manage-ac` until crash — record video + telemetry
- [ ] Install `mss` + `opencv-python` if not already: `pip install mss opencv-python`
- [ ] Keep one clean demo run as the showcase

### Plots / tables
- [ ] Training curves (auto by `train.py`) — total Q-loss, per-channel Q, policy loss, entropy × 3, α × 3
- [ ] Ablation table: baseline vs. vector-Q vs. + per-action α vs. + memmap buffer
- [ ] Reward-per-frame histograms (baseline vs. v2_final)
- [ ] Trajectory heatmap — pre vs. post fine-tune
- [ ] Action-distribution histograms (detect degenerate policies)

---

## Final report (6–8 pages, NeurIPS template)
**Proposal was submitted on TD3 + MLP + 60 Hz + 3 tracks; we built SAC + Transformer + Vector-Q + 25 Hz + 1 track.**
Report MUST acknowledge + justify this drift at the top of Methods. See [docs/01_PROPOSAL.md](gym/transformer_sac_vectorq_v2_final/docs/01_PROPOSAL.md) for the drift table.

- [ ] Section 1: Introduction + motivation (end with open-source disclosure)
- [ ] Section 2: Related Work — cover all 6 submitted references + add the papers covering what was actually built
- [ ] Section 3: Methods — OPEN with deviation acknowledgement, then overview → intuition → details
- [ ] Section 4: Experiments (testbed + questions + results)
- [ ] Section 5: Conclusion + future work (mention: imitation-learning pre-train, multi-track generalisation — both deferred from proposal)
- [ ] Proofread for NeurIPS formatting (no significant template deviation)

### Literature survey
**Locked (from submitted proposal — do NOT drop):**
- [ ] Lillicrap et al. 2015 — Continuous Control with Deep RL (DDPG)
- [ ] Fujimoto et al. 2018 — TD3 *(cite; explain why we switched to SAC)*
- [ ] Fuchs et al. 2020 — Gran Turismo Sport DRL
- [ ] Kendall et al. 2019 — Learning to Drive in a Day
- [ ] Riedmiller et al. 2018 — Learning by Playing
- [ ] Hester et al. 2018 — DQfD

**Must-add (for the method we actually built):**
- [ ] Haarnoja et al. 2018 — SAC (justifies algorithm choice)
- [ ] Vaswani et al. 2017 — Transformer (encoder)
- [ ] Van Seijen et al. 2017 — Hybrid Reward Architecture (key motivation for vector-Q)
- [ ] Remonda et al. 2024 — Assetto Corsa Gym (environment disclosure)

---

## Code submission
- [ ] Clean repo snapshot (`.zip`, exclude 17 GB buffers, venv, AC install)
- [ ] Host `latest.pt` on OneDrive / Google Drive → public link
- [ ] Root README: install + run `train.py` / `collect.py` / `demo.py`
- [ ] Verify demo script runs end-to-end on clean machine
- [ ] Open-source disclosures at end of intro section

---

## Innovation angle (chase the +5 bonus)
Points to emphasise as novel:
1. **Vector-Q decomposition** on a *real-time 25 Hz sim-racing task* (most vector-reward work is on Atari / gridworlds)
2. **Intent-based reward decomposition** — rewards the *action* that would be correct at each speed regime, not the resulting speed
3. **Memmap-backed stratified replay** — 6 × 100 K windows across pos/neg per channel, scales to 100 M+ transitions with minimal RAM
4. **Overlap trick** — inference + physics tick run concurrently → 25 Hz hard-real-time loop with GPU inference
