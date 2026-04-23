"""
Monza policy-behaviour map: speed and brake application along the
Vector-Q Transformer SAC full-lap evaluation episode.

Left panel:  trajectory coloured by per-frame speed (m/s).
Right panel: trajectory coloured by per-frame brake action, highlighting
             where the policy applies braking.

Output: figures/trajectory.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OURS_CSV = os.path.join(ROOT, "gym", "transformer_sac_vectorq_v2_final_fineTune",
                       "demo_runs", "20260421_185418", "ep01", "telemetry.csv")
RACING_LINE = os.path.join(ROOT, "assetto_corsa_gym", "assetto_corsa_gym",
                           "AssettoCorsaConfigs", "tracks", "monza-racing_line.csv")
out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "trajectory.png"))

# ---- Load data -------------------------------------------------------------
df = pd.read_csv(OURS_CSV,
                 usecols=["world_x", "world_y", "speed", "action_brake"])
x = df["world_x"].values
y = df["world_y"].values
speed  = df["speed"].values
# action_brake is raw tanh output in [-1, 1]; -1 = off, +1 = full brake.
# Remap to [0, 1] for cleaner visualisation (0 = off, 1 = full).
brake_norm = (df["action_brake"].values + 1.0) * 0.5
brake_norm = np.clip(brake_norm, 0.0, 1.0)

# Racing-line reference (faint outline)
rl_x = rl_y = None
try:
    df_rl = pd.read_csv(RACING_LINE)
    xcands = [c for c in df_rl.columns if c.lower() in ("position.x", "x", "pos_x", "world_position_x", "pos.x")]
    ycands = [c for c in df_rl.columns if c.lower() in ("position.y", "y", "pos_y", "world_position_y", "pos.y")]
    if xcands and ycands:
        rl_x = df_rl[xcands[0]].values
        rl_y = df_rl[ycands[0]].values
except Exception:
    pass

# ---- Helper: draw a line coloured by a per-segment scalar ------------------
def colored_line(ax, x, y, c, cmap, vmin, vmax, lw=2.6):
    pts  = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap, norm=Normalize(vmin, vmax),
                          linewidth=lw, capstyle="round")
    lc.set_array(c[:-1])
    ax.add_collection(lc)
    return lc

# ---- Figure ----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

for ax, title in zip((ax1, ax2),
                      ("(a) Speed along the lap",
                       "(b) Brake application along the lap")):
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel(r"world $x$ (m)")
    ax.set_ylabel(r"world $y$ (m)")
    ax.grid(True, ls=":", alpha=0.35)
    if rl_x is not None:
        ax.plot(rl_x, rl_y, color="#cfcfcf", lw=0.9, ls="-", alpha=0.7, zorder=1)

# -- speed panel
lc_s = colored_line(ax1, x, y, speed, cmap="viridis",
                    vmin=float(np.percentile(speed, 2)),
                    vmax=float(np.percentile(speed, 98)))
ax1.autoscale_view()
cb1 = fig.colorbar(lc_s, ax=ax1, fraction=0.046, pad=0.02)
cb1.set_label("speed (m/s)")

# -- brake panel: gray base trajectory + red overlay on active braking.
ax2.plot(x, y, color="#8a8a8a", lw=1.4, alpha=0.9, zorder=2)
brake_mask = brake_norm > 0.15
if brake_mask.any():
    xb = np.where(brake_mask, x, np.nan)
    yb = np.where(brake_mask, y, np.nan)
    lc_b = colored_line(ax2, xb, yb, brake_norm,
                        cmap="Reds", vmin=0.15, vmax=1.0, lw=3.4)
    ax2.autoscale_view()
    cb2 = fig.colorbar(lc_b, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label("brake action (light = soft, dark = full)")
else:
    ax2.autoscale_view()

plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"wrote {out_path}  (frames: {len(x)}, speed max: {speed.max():.1f} m/s, "
      f"% frames with brake > 0.3: {100*(brake_norm>0.3).mean():.1f}%)")
