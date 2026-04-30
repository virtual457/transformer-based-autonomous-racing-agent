"""
plot_aiclone_dataset.py — Diagnostic plots for the processed AIClone npz.

Source:  AICLONE/data/monza_miata/aiclone_dataset.npz
Output:  AICLONE/data/monza_miata/plots/*.png

Usage:
    .\\AssetoCorsa\\Scripts\\python.exe AICLONE/plot_aiclone_dataset.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_THIS_DIR, "data", "monza_miata")
NPZ_PATH   = os.path.join(DATA_DIR, "aiclone_dataset.npz")
META_PATH  = os.path.join(DATA_DIR, "aiclone_dataset.json")
OUT_DIR    = os.path.join(DATA_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Matches AICLONE/preprocess_parquet.py
OBS_ENABLED_CHANNELS = [
    "speed", "gap", "LastFF", "RPM", "accelX", "accelY",
    "actualGear", "angular_velocity_y",
    "local_velocity_x", "local_velocity_y",
    "SlipAngle_fl", "SlipAngle_fr", "SlipAngle_rl", "SlipAngle_rr",
]
RAY_LABELS = ["ray_r90","ray_r67","ray_r45","ray_r22","ray_r11",
              "ray_fwd","ray_l11","ray_l22","ray_l45","ray_l67","ray_l90"]
BASIC_LABELS = OBS_ENABLED_CHANNELS + RAY_LABELS   # 25
# idx 25 = out_of_track
# idx 26..37 = curvature lookahead (12)
# idx 38..46 = past actions (3 steps × 3)
ACTION_NAMES = ["steer", "throttle", "brake"]


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    print(f"Loading {NPZ_PATH}")
    d = np.load(NPZ_PATH)
    obs        = d["obs"].astype(np.float32)
    actions    = d["actions"].astype(np.float32)
    ep_ends    = d["episode_ends"]
    N, D       = obs.shape
    E          = len(ep_ends)
    print(f"  N={N}  obs_dim={D}  episodes={E}")

    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        print(f"  meta: {meta}")

    t = np.arange(N)

    # ── 1. Action histograms ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, name in enumerate(ACTION_NAMES):
        axes[i].hist(actions[:, i], bins=80, color="steelblue", edgecolor="k", alpha=0.8)
        axes[i].set_title(f"{name}   mean={actions[:,i].mean():.3f}  std={actions[:,i].std():.3f}")
        axes[i].set_xlabel(name); axes[i].set_ylabel("count")
        axes[i].axvline(0, color="k", ls=":", lw=0.8)
    fig.suptitle("AIClone Monza — action distributions")
    savefig(fig, "01_action_histograms.png")

    # ── 2. Action time-series ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for i, name in enumerate(ACTION_NAMES):
        axes[i].plot(t, actions[:, i], lw=0.5, color="steelblue")
        axes[i].set_ylabel(name); axes[i].grid(alpha=0.3)
        axes[i].axhline(0, color="k", ls=":", lw=0.5)
    for ep_end in ep_ends[:-1]:
        for ax in axes: ax.axvline(ep_end, color="red", ls="--", lw=0.5)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"AIClone Monza — actions over time  (N={N}, episodes={E})")
    savefig(fig, "02_action_timeseries.png")

    # ── 3. Basic obs time-series (speed + key channels) ───────────────────────
    key_channels = ["speed", "RPM", "local_velocity_x", "local_velocity_y",
                    "accelX", "accelY", "angular_velocity_y", "actualGear"]
    fig, axes = plt.subplots(len(key_channels), 1, figsize=(14, 12), sharex=True)
    for ax, ch in zip(axes, key_channels):
        idx = OBS_ENABLED_CHANNELS.index(ch)
        axes_data = obs[:, idx]
        ax.plot(t, axes_data, lw=0.5, color="darkgreen")
        ax.set_ylabel(ch); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle("AIClone Monza — key scaled obs channels over time")
    savefig(fig, "03_obs_timeseries.png")

    # ── 4. Ray sensors heatmap ────────────────────────────────────────────────
    ray_start = len(OBS_ENABLED_CHANNELS)
    rays = obs[:, ray_start:ray_start + 11]   # (N, 11)
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(rays.T, aspect="auto", cmap="viridis",
                   extent=[0, N, 10.5, -0.5], interpolation="nearest")
    ax.set_yticks(range(11)); ax.set_yticklabels(RAY_LABELS)
    ax.set_xlabel("step"); ax.set_title("AIClone Monza — 11 ray sensors (scaled 0..1, 1=far)")
    fig.colorbar(im, ax=ax, label="ray distance (normalized)")
    savefig(fig, "04_ray_sensors_heatmap.png")

    # ── 5. Out-of-track indicator ─────────────────────────────────────────────
    oot_idx = 25
    if D > oot_idx:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(t, obs[:, oot_idx], lw=0.6, color="red")
        ax.set_xlabel("step"); ax.set_ylabel("out_of_track")
        frac = (obs[:, oot_idx] > 0.5).mean() * 100
        ax.set_title(f"AIClone Monza — out_of_track flag  ({frac:.2f}% of frames)")
        ax.grid(alpha=0.3)
        savefig(fig, "05_out_of_track.png")

    # ── 6. Joint action scatter ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [(0,1,"steer","throttle"),(0,2,"steer","brake"),(1,2,"throttle","brake")]
    for ax, (i, j, a, b) in zip(axes, pairs):
        ax.scatter(actions[:, i], actions[:, j], s=1, alpha=0.15, color="navy")
        ax.set_xlabel(a); ax.set_ylabel(b); ax.grid(alpha=0.3)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    fig.suptitle("AIClone Monza — action joint scatter")
    savefig(fig, "06_action_joint_scatter.png")

    # ── 7. Curvature lookahead heatmap ────────────────────────────────────────
    if D >= 38:
        curv = obs[:, 26:38]   # (N, 12)
        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(curv.T, aspect="auto", cmap="coolwarm",
                       extent=[0, N, 11.5, -0.5], interpolation="nearest",
                       vmin=-np.abs(curv).max(), vmax=np.abs(curv).max())
        ax.set_xlabel("step"); ax.set_ylabel("lookahead bin (near→far)")
        ax.set_title("AIClone Monza — 12-bin curvature lookahead")
        fig.colorbar(im, ax=ax, label="curvature (normalized)")
        savefig(fig, "07_curvature_lookahead.png")

    # ── 8. Summary stats ──────────────────────────────────────────────────────
    stats = {
        "n_frames":   int(N),
        "n_episodes": int(E),
        "obs_dim":    int(D),
        "episode_ends": ep_ends.tolist(),
        "action_stats": {
            name: {
                "min":  float(actions[:, i].min()),
                "max":  float(actions[:, i].max()),
                "mean": float(actions[:, i].mean()),
                "std":  float(actions[:, i].std()),
                "abs_gt_0.05_frac": float((np.abs(actions[:, i]) > 0.05).mean()),
            } for i, name in enumerate(ACTION_NAMES)
        },
        "speed_stats_scaled": {
            "min":  float(obs[:, 0].min()),
            "max":  float(obs[:, 0].max()),
            "mean": float(obs[:, 0].mean()),
        },
        "meta": meta,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {os.path.join(OUT_DIR, 'summary.json')}")

    print("\nAll plots written to:", OUT_DIR)


if __name__ == "__main__":
    main()
