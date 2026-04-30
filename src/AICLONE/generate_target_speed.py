"""
generate_target_speed.py — Add target_speed column to a racing line CSV.

Reads human demo parquet files (speed_ms, pos_x, pos_y per frame),
matches each racing-line waypoint to its nearest demo frames by XY position,
takes the median speed, smooths it, and writes target_speed into the CSV.

Usage
-----
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/generate_target_speed.py

Output
------
    Overwrites the racing line CSV in-place (backs up original to .bak first).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ── Defaults ──────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent

DEFAULT_PARQUET_GLOB = str(
    _REPO / "collectDataAI/data/ks_mazda_miata/monza/20260402_175809/*.parquet"
)
DEFAULT_RACING_LINE = str(
    _REPO / "assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaConfigs/tracks/monza-racing_line.csv"
)

# Smoothing window (number of waypoints).  Monza racing line has ~5700 points
# so 150 points ≈ ~150m of smoothing — removes sensor jitter, keeps corners sharp.
SMOOTH_WINDOW = 150

# For each racing-line waypoint, use the N closest demo frames to compute median speed.
K_NEAREST = 10


def main():
    parser = argparse.ArgumentParser(description="Generate target_speed for racing line CSV")
    parser.add_argument("--parquet-glob", default=DEFAULT_PARQUET_GLOB)
    parser.add_argument("--racing-line",  default=DEFAULT_RACING_LINE)
    parser.add_argument("--smooth",       type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--k",            type=int, default=K_NEAREST)
    args = parser.parse_args()

    # ── 1. Load demo frames ───────────────────────────────────────────────────
    import glob as _glob
    files = _glob.glob(args.parquet_glob)
    if not files:
        raise FileNotFoundError(f"No parquet files found: {args.parquet_glob}")
    print(f"Loading {len(files)} parquet file(s) ...")
    demo = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(demo):,} demo frames loaded")

    # Drop out-of-track frames — don't learn target speed from crashed states
    if "out_of_track" in demo.columns:
        demo = demo[demo["out_of_track"] == 0]
        print(f"  {len(demo):,} frames after dropping out-of-track")

    demo_xy  = demo[["pos_x", "pos_y"]].values.astype(np.float32)   # (N, 2)
    demo_spd = demo["speed_ms"].values.astype(np.float32)            # (N,)

    # ── 2. Load racing line ───────────────────────────────────────────────────
    rl = pd.read_csv(args.racing_line)
    print(f"Racing line: {len(rl)} waypoints  columns={list(rl.columns)}")

    rl_xy = rl[["pos_x", "pos_y"]].values.astype(np.float32)   # (M, 2)
    M = len(rl_xy)

    # ── 3. For each waypoint, find K nearest demo frames → median speed ───────
    print(f"Computing target speed (K={args.k} nearest) ...")

    # Build a simple KD-tree for fast lookup
    from scipy.spatial import cKDTree
    tree = cKDTree(demo_xy)

    _, idx = tree.query(rl_xy, k=args.k)      # (M, K)
    speeds = demo_spd[idx]                     # (M, K)
    target_speed = np.median(speeds, axis=1)   # (M,)

    # ── 4. Smooth with circular wrap (track is a loop) ────────────────────────
    # Duplicate edges to handle wrap-around before filtering
    w = args.smooth
    padded = np.concatenate([target_speed[-w:], target_speed, target_speed[:w]])
    smoothed_padded = uniform_filter1d(padded, size=w)
    target_speed_smooth = smoothed_padded[w:w + M]

    print(f"  min={target_speed_smooth.min():.1f} m/s  "
          f"max={target_speed_smooth.max():.1f} m/s  "
          f"mean={target_speed_smooth.mean():.1f} m/s")

    # ── 5. Write back to CSV ──────────────────────────────────────────────────
    bak = args.racing_line + ".bak"
    shutil.copy2(args.racing_line, bak)
    print(f"Backed up original to {bak}")

    rl["target_speed"] = target_speed_smooth.astype(np.float32)
    rl.to_csv(args.racing_line, index=False)
    print(f"Written: {args.racing_line}")
    print("Done. Re-run training with use_target_speed: true in config.yml.")


if __name__ == "__main__":
    main()
