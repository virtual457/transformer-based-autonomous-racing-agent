"""
Grouped bar chart of per-channel reward component weights used by
vector_reward.py. Makes the "same components, different channel weighting"
claim in Section 3.1 visually concrete.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Weights copied from gym/.../vector_reward.py --------------------------
W_STEER = {
    "gap":     0.30,
    "gap_dir": 0.25,
    "yaw":     0.25,
    "smooth":  0.20,
}
W_THROTTLE = {
    "speed":  0.30,
    "gap":    0.40,
    "yaw":    0.20,
    "smooth": 0.10,
}
W_BRAKE = {
    "speed":  0.30,
    "gap":    0.40,
    "yaw":    0.20,
    "smooth": 0.10,
}

components = ["speed", "gap", "gap_dir", "yaw", "smooth"]
labels     = ["speed", "gap", "gap_dir", "yaw", "smooth"]

steer    = [W_STEER.get(c, 0.0)    for c in components]
throttle = [W_THROTTLE.get(c, 0.0) for c in components]
brake    = [W_BRAKE.get(c, 0.0)    for c in components]

# ---- Bar chart -------------------------------------------------------------
x = np.arange(len(components))
w = 0.26

fig, ax = plt.subplots(figsize=(7.5, 3.4), constrained_layout=True)

bars_s = ax.bar(x - w, steer,    w, label=r"steer",    color="tab:blue")
bars_t = ax.bar(x,     throttle, w, label=r"throttle", color="tab:orange")
bars_b = ax.bar(x + w, brake,    w, label=r"brake",    color="tab:red")

for bars in (bars_s, bars_t, bars_b):
    for b in bars:
        v = b.get_height()
        if v > 0:
            ax.annotate(f"{v:.2f}",
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

ax.set_ylabel("weight in per-channel reward")
ax.set_xlabel("scalar reward component")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.55)
ax.legend(loc="upper right", frameon=True, fontsize=9)
ax.grid(True, axis="y", ls=":", alpha=0.4)
ax.set_title("Per-channel reward component weights", fontsize=11)

out = os.path.abspath(os.path.join(os.path.dirname(__file__), "reward_weights.png"))
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
