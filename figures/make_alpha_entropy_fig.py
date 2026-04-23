"""
Generate the per-channel alpha + entropy figure for the paper.
Inputs:  4 tune_tail history.json files (chained training phases).
Output:  figures/per_channel_alpha.png
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "gym", "transformer_sac_vectorq_v2_final_fineTune", "outputs",
))

HIST_PATHS = [
    os.path.join(HISTORY_ROOT, "tune_tail_20260421_213948", "history.json"),
    os.path.join(HISTORY_ROOT, "tune_tail_20260421_224235", "history.json"),
    os.path.join(HISTORY_ROOT, "tune_tail_20260421_230354", "history.json"),
    os.path.join(HISTORY_ROOT, "tune_tail_20260421_231337", "history.json"),
]

alpha_pc    = []
entropy_pc  = []
run_offsets = [0]  # step index where each new run begins

for p in HIST_PATHS:
    with open(p) as f:
        h = json.load(f)
    a = np.asarray(h["alpha_per_action"])      # (T, 3)
    e = np.asarray(h["entropy_per_action"])    # (T, 3)
    alpha_pc.append(a)
    entropy_pc.append(e)
    run_offsets.append(run_offsets[-1] + len(a))

alpha   = np.concatenate(alpha_pc,   axis=0)
entropy = np.concatenate(entropy_pc, axis=0)
T       = alpha.shape[0]
step    = np.arange(T)

def smooth(x, k=50):
    if len(x) < k:
        return x
    c = np.convolve(x, np.ones(k) / k, mode="valid")
    pad = np.full(len(x) - len(c), c[0])
    return np.concatenate([pad, c])

channel_names = [r"steer ($\alpha_{\mathrm{s}}$)",
                 r"throttle ($\alpha_{\mathrm{th}}$)",
                 r"brake ($\alpha_{\mathrm{br}}$)"]
ent_names     = [r"steer ($H_{\mathrm{s}}$)",
                 r"throttle ($H_{\mathrm{th}}$)",
                 r"brake ($H_{\mathrm{br}}$)"]
colors = ["tab:blue", "tab:orange", "tab:red"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2), constrained_layout=True)

for i in range(3):
    ax1.plot(step, smooth(alpha[:, i]),   color=colors[i], lw=1.8, label=channel_names[i])
ax1.axhline(np.exp(-5.0), color="gray", ls=":", lw=1.0, alpha=0.8, label=r"$\alpha$ clamp floor $=e^{-5}$")
ax1.set_xlabel("gradient step")
ax1.set_ylabel(r"per-channel temperature $\alpha_c$")
ax1.set_title(r"(a) $\alpha_c$ auto-tuning per channel")
ax1.legend(fontsize=8, loc="center right", frameon=True)
ax1.grid(True, ls=":", alpha=0.4)
ax1.set_xlim(0, T)

for i in range(3):
    ax2.plot(step, smooth(entropy[:, i]), color=colors[i], lw=1.8, label=ent_names[i])
ax2.set_xlabel("gradient step")
ax2.set_ylabel(r"per-channel policy entropy $H_c$")
ax2.set_title(r"(b) Per-channel policy entropy")
ax2.legend(fontsize=8, loc="upper right", frameon=True)
ax2.grid(True, ls=":", alpha=0.4)
ax2.set_xlim(0, T)

out = os.path.abspath(os.path.join(os.path.dirname(__file__), "per_channel_alpha.png"))
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  ({T} steps across {len(HIST_PATHS)} phases)")
