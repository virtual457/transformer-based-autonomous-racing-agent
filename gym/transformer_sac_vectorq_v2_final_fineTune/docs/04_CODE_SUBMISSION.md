# Code Submission — Requirements

- Submit code + a **simple demo script** showcasing how it works.
- Minimum bar: run inference on a couple of demo inputs using the trained model.
- Grades for this part are **integrated into the final report grade**.

## If weights are large
Host them on OneDrive / Google Drive and share the link in the code submission.
Our checkpoint is small (~several MB) but the **memmap buffers are ~17 GB** — those should **not** be submitted; host the checkpoint + a small sample buffer slice.

## What to submit (concrete plan)
1. **Repo snapshot** — `.zip` of the cleaned repo (exclude buffers, AC install, venv)
2. **Checkpoint** — `checkpoints/latest.pt` (include directly or via drive link)
3. **Demo script** — `demo.py` already written — showcases the trained model driving.
4. **README** pointing to:
   - How to install (venv, requirements, AC plugin install)
   - How to run `train.py`, `collect.py`, `demo.py`
   - Link to hosted weights / sample buffer if applicable

## Pre-submission checklist
- [ ] All 3 scripts (`train.py`, `collect.py`, `demo.py`) run end-to-end on a clean machine
- [ ] README lists exact run commands
- [ ] Pretrained `latest.pt` hosted with public share link
- [ ] Open-source disclosure line in intro of final report
- [ ] No credentials, tokens, or absolute Windows paths leaking into submitted code
