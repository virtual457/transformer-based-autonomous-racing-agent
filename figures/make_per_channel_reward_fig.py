"""
Per-channel reward trace along a full Monza lap.

Reconstructs the per-channel reward vector (r_steer, r_throttle, r_brake)
frame-by-frame from a logged evaluation episode using the production
vector_reward.compute_vector_reward function. Target speed is looked up
from the Monza racing_line.csv via nearest-neighbor on (world_x, world_y).

The figure shows that the three channel rewards move independently along
the lap: braking into a corner drives r_brake up while r_throttle drops,
and the sign of r_steer depends on which side of the line the car is on.
A scalar joint reward would average these into a single signal.

Output: figures/per_channel_reward.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "gym"))
from transformer_sac_vectorq_v2_final_fineTune.vector_reward import (
    compute_vector_reward,
)

TELEM = os.path.join(
    ROOT, "gym", "transformer_sac_vectorq_v2_final_fineTune",
    "demo_runs", "20260421_185418", "ep01", "telemetry.csv",
)
RACING = os.path.join(
    ROOT, "assetto_corsa_gym", "assetto_corsa_gym",
    "AssettoCorsaConfigs", "tracks", "monza-racing_line.csv",
)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   "per_channel_reward.png"))

# ── Load telemetry ────────────────────────────────────────────────────────────
df = pd.read_csv(TELEM)
N = len(df)
print(f"loaded {N} frames from {os.path.basename(os.path.dirname(TELEM))}")

# ── Lookup target_speed via nearest-neighbor on racing line ───────────────────
rl = pd.read_csv(RACING)
rl_xy = rl[["pos_x", "pos_y"]].values
rl_sp = rl["target_speed"].values

xy = df[["world_x", "world_y"]].values
# Vectorised nearest-neighbor: for each telem point find closest racing-line point.
# Chunk to keep memory reasonable.
target_speed = np.empty(N, dtype=np.float32)
chunk = 2000
for i in range(0, N, chunk):
    j = min(i + chunk, N)
    d2 = ((xy[i:j, None, :] - rl_xy[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(axis=1)
    target_speed[i:j] = rl_sp[idx]

# ── Per-frame vector reward ──────────────────────────────────────────────────
steer_act = df["action_steer"].values
thr_act   = df["action_throttle"].values
brk_act   = df["action_brake"].values
speed_ms  = df["speed"].values
gap_m     = df["gap"].values

# Previous action (shifted); first frame uses same action as prev.
prev_steer = np.concatenate([[steer_act[0]], steer_act[:-1]])
prev_thr   = np.concatenate([[thr_act[0]],   thr_act[:-1]])
prev_brk   = np.concatenate([[brk_act[0]],   brk_act[:-1]])

oot = df["out_of_track"].values.astype(bool) if "out_of_track" in df.columns \
    else np.zeros(N, dtype=bool)

r_s = np.empty(N, dtype=np.float32)
r_t = np.empty(N, dtype=np.float32)
r_b = np.empty(N, dtype=np.float32)

for i in range(N):
    comps = {
        "r_gap_abs": float(df["r_gap_abs"].iat[i]),
        "r_yaw":     float(df["r_yaw"].iat[i]),
        "r_progress": float(df["r_progress"].iat[i]),
    }
    mets = {
        "speed_ms":        float(speed_ms[i]),
        "gap_m":           float(gap_m[i]),
        "target_speed_ms": float(target_speed[i]),
    }
    action = np.array([steer_act[i], thr_act[i], brk_act[i]], dtype=np.float32)
    prev   = np.array([prev_steer[i], prev_thr[i], prev_brk[i]], dtype=np.float32)
    v = compute_vector_reward(comps, mets, action, prev, out_of_track=bool(oot[i]))
    r_s[i], r_t[i], r_b[i] = v[0], v[1], v[2]

print(f"channel means:  steer={r_s.mean():+.3f}  throttle={r_t.mean():+.3f}  brake={r_b.mean():+.3f}")
print(f"channel std:    steer={r_s.std():.3f}   throttle={r_t.std():.3f}   brake={r_b.std():.3f}")
# Pairwise correlation — low correlation = channels move independently.
corr_st = np.corrcoef(r_s, r_t)[0, 1]
corr_sb = np.corrcoef(r_s, r_b)[0, 1]
corr_tb = np.corrcoef(r_t, r_b)[0, 1]
print(f"pairwise correlation: steer/throttle={corr_st:.2f} steer/brake={corr_sb:.2f} throttle/brake={corr_tb:.2f}")


def smooth(x, k=25):
    if len(x) < k:
        return x
    c = np.convolve(x, np.ones(k) / k, mode="valid")
    pad = np.full(len(x) - len(c), c[0])
    return np.concatenate([pad, c])


# ── Figure: full-lap trace + one-corner zoom ─────────────────────────────────
time_s = np.arange(N) / 25.0  # 25 Hz control rate

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.6),
                               constrained_layout=True,
                               gridspec_kw={"height_ratios": [1.2, 1.0]})

# (a) full lap
ax1.plot(time_s, smooth(r_s), color="tab:blue",   lw=1.3, label=r"$r^{\mathrm{s}}$ (steer)")
ax1.plot(time_s, smooth(r_t), color="tab:orange", lw=1.3, label=r"$r^{\mathrm{th}}$ (throttle)")
ax1.plot(time_s, smooth(r_b), color="tab:red",    lw=1.3, label=r"$r^{\mathrm{br}}$ (brake)")
ax1.axhline(0, color="gray", lw=0.6, ls=":")
ax1.set_xlabel("lap time (s)")
ax1.set_ylabel("per-channel reward")
ax1.set_title(r"(a) Per-channel reward along a full Monza lap"
              f"  (corr steer/throttle = {corr_st:+.2f},"
              f"  steer/brake = {corr_sb:+.2f},"
              f"  throttle/brake = {corr_tb:+.2f})",
              fontsize=10)
ax1.legend(loc="lower right", fontsize=9, ncol=3, frameon=True)
ax1.grid(True, ls=":", alpha=0.35)
ax1.set_xlim(time_s[0], time_s[-1])

# (b) one corner — find the most informative braking zone.
# Pick the frame range where brake action is high and sustained.
brake_norm = (brk_act + 1.0) * 0.5
# Smoothed brake to find zones.
b_sm = smooth(brake_norm, 15)
# Find first sustained brake event (>0.4) at least 40 frames long.
above = b_sm > 0.4
diff = np.diff(above.astype(int))
starts = np.where(diff == 1)[0]
ends   = np.where(diff == -1)[0]
if len(starts) and len(ends):
    # Pad a bit on either side for context.
    for s, e in zip(starts, ends):
        if e - s > 40:
            lo = max(0, s - 80)
            hi = min(N, e + 120)
            break
    else:
        lo, hi = N // 3, N // 3 + 400
else:
    lo, hi = N // 3, N // 3 + 400

ax2.plot(time_s[lo:hi], smooth(r_s, 8)[lo:hi], color="tab:blue",   lw=1.5, label=r"$r^{\mathrm{s}}$")
ax2.plot(time_s[lo:hi], smooth(r_t, 8)[lo:hi], color="tab:orange", lw=1.5, label=r"$r^{\mathrm{th}}$")
ax2.plot(time_s[lo:hi], smooth(r_b, 8)[lo:hi], color="tab:red",    lw=1.5, label=r"$r^{\mathrm{br}}$")
# Shade the braking zone (from the detected start/end above).
try:
    ax2.axvspan(time_s[max(lo, s)], time_s[min(hi - 1, e)],
                color="tab:red", alpha=0.08, label="braking zone")
except Exception:
    pass
ax2.axhline(0, color="gray", lw=0.6, ls=":")
ax2.set_xlabel("lap time (s)")
ax2.set_ylabel("per-channel reward")
ax2.set_title("(b) Zoomed-in braking zone — the three channels diverge",
              fontsize=10)
ax2.legend(loc="lower right", fontsize=9, ncol=4, frameon=True)
ax2.grid(True, ls=":", alpha=0.35)
ax2.set_xlim(time_s[lo], time_s[hi - 1])

plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
