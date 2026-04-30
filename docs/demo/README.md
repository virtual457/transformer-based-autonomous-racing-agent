# Vector-Q Transformer SAC: demo submission

This folder is a self-contained inference demo for the course-project submission. No Assetto Corsa, no live simulator, no training, just load the trained checkpoint and run it on pre-collected observation windows.

## How to run

Open `inference_demo.ipynb` in Jupyter, Colab, or VS Code and run every cell top-to-bottom.

- On a **local Jupyter kernel**: the notebook reads the checkpoint and sample data from this folder directly (the `fetch()` helper short-circuits when the files already exist locally).
- On **Google Colab**: the same `fetch()` helper downloads the files from the GitHub repo if this zip was not uploaded. If you upload this `demo/` folder to the Colab runtime first, it will skip the download and read locally.

Dependencies are standard: `torch`, `numpy`, `matplotlib`. Colab already has them; locally a stock `pip install torch numpy matplotlib` is enough.

## What you will see

| section | cells | produces |
|---|---|---|
| Video embed | 2 | clickable GIF that opens the full-lap demo on YouTube (`https://youtu.be/ZNJG0orcfXg`) |
| §1–§4 | setup | loads the 6.58 M-parameter Vector-Q Transformer SAC checkpoint and rebuilds the policy + twin-critic networks |
| §5–§6 | inference | runs the policy + per-channel critic on 216 real observation windows drawn from the six stratified replay sub-buffers |
| §7 | plot | per-channel Q histograms at $\pi(s)$, one per critic head |
| §8 | text | single-window inspection (stored action, predicted action, policy std, per-channel Q) |
| §9 | plot | speed + brake-zone track map over a recorded full lap on Monza |

The `§9` plot is the most visual result: the left panel shows speed along the lap (yellow on straights, dark blue at corner apexes), and the right panel overlays red segments on the grey track exactly where the policy applied brake, matching the five to six named Monza corners.

## Files in this folder

| file | size | purpose |
|---|---|---|
| `inference_demo.ipynb` | ~210 KB | notebook, run this |
| `inference_demo.py`    | ~8 KB   | equivalent `# %%`-delimited Python script |
| `inference_ckpt.pt`    | 26 MB   | trained weights (policy encoder, critic encoder, policy head, twin-Q head). Stripped of optimizer state, inference only |
| `sample_data.npz`      | 1.9 MB  | 216 real observation windows (75 frames × 50-dim tokens) sampled from the stratified replay buffer |
| `aiclone_human.npz`    | 2.1 MB  | 18,572 frames of a human-driven Monza lap, included for reference; not used in the final notebook cells |
| `lap_trajectory.npz`   | 344 KB  | recorded telemetry from a full-lap policy evaluation (world x/y, speed, action channels). Source for §9 |

`inference_ckpt.pt` is 26 MB so it ships inside this zip. If a future checkpoint grows past the zip-size budget, it can also be fetched from the repo's `demo/` folder on GitHub via `urllib.request.urlretrieve` (the `fetch()` helper in §2 already supports this path).

## Repository and full report

Full project repo: <https://github.com/virtual457/transformer-based-autonomous-racing-agent>

Full NeurIPS-format report: `paper/main.pdf` in the repo (or `report.md` under `docs/`).

## Course submission notes for the grader

- The model is trained against a **live Assetto Corsa simulator** at 25 Hz via a Python plugin stack, so training cannot be reproduced from this zip alone, and is not expected to be.
- This demo therefore reproduces only the **inference path**: given a trained checkpoint and saved observation windows, show that the network reconstructs correctly, the per-channel Q-values are meaningful (non-collapsed), and the recorded lap exhibits the learned behaviour (accelerate on straights, brake before corners).
- The notebook prints intermediate tensors at every stage so the computation is auditable cell-by-cell.
