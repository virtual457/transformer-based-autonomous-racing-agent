"""
preprocess_parquet.py — Convert raw collectDataAI parquet files to a
                        training-ready .npz for BC actor pre-training.

Reads every episode parquet under collectDataAI/data/ and builds:
    obs     (N, 125) float32 — normalised observation vector (matches ac_env.get_obs())
    actions (N, 3)   float32 — SAC-space actions in [-1, 1]
                               [steer, pedal, brake]
    episode_ends (E,) int64  — index of last step in each episode

125-dim obs layout (mirrors preprocess_bc.py exactly):
    [0:14]   14 telemetry channels (speed, gap, LastFF, RPM, ...)
    [14:25]  11 ray sensors / 200 m
    [25]      1 out_of_track flag
    [26:38]  12 curvature look-ahead (~300 m ahead on racing line)
    [38:47]   9 past actions  (3 steps × 3: steer, pedal, brake)
    [47:50]   3 current action (steer, pedal, brake)
    [50:125] 75 previous basic obs (3 steps × 25)

Action normalisation (SAC policy space [0, 1]):
    steer  = clip((steer_deg / STEER_MAX + 1) / 2, 0, 1)   0.5 = straight
    pedal  = clip(accStatus, 0, 1)                           direct throttle
    brake  = clip(brakeStatus, 0, 1)                         direct brake  (0=none)

Usage
-----
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/preprocess_parquet.py
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/preprocess_parquet.py --output-dir AICLONE/data
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/preprocess_parquet.py --data-dir collectDataAI/data --output-dir AICLONE/data
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
_ASSETTO_GYM = _REPO / "assetto_corsa_gym" / "assetto_corsa_gym"
_GYM         = _REPO / "gym"
_AC_CONFIGS  = _ASSETTO_GYM / "AssettoCorsaConfigs"

for _p in [str(_GYM), str(_ASSETTO_GYM)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AssettoCorsaEnv.reference_lap import ReferenceLap   # noqa: E402
from AssettoCorsaEnv.brake_map     import BrakeMap       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Obs constants (must match preprocess_bc.py / ac_env.py exactly) ───────────
PAST_ACTIONS_WINDOW         = 3
CURV_LOOK_AHEAD_DISTANCE    = 300.0
CURV_LOOK_AHEAD_VECTOR_SIZE = 12
CURV_NORMALIZATION_CONSTANT = 0.1
MAX_RAY_LEN                 = 200.0
TOP_SPEED_MS                = 80.0
NUM_SENSORS                 = 11

OBS_ENABLED_CHANNELS = [
    "speed", "gap", "LastFF", "RPM", "accelX", "accelY",
    "actualGear", "angular_velocity_y",
    "local_velocity_x", "local_velocity_y",
    "SlipAngle_fl", "SlipAngle_fr", "SlipAngle_rl", "SlipAngle_rr",
]

OBS_CHANNEL_SCALES = {
    "speed":              TOP_SPEED_MS,
    "gap":                10.0,
    "LastFF":             1.0,
    "RPM":                10000.0,
    "accelX":             5.0,
    "accelY":             5.0,
    "actualGear":         8.0,
    "angular_velocity_y": np.pi,
    "local_velocity_x":   TOP_SPEED_MS,
    "local_velocity_y":   20.0,
    "SlipAngle_fl":       25.0,
    "SlipAngle_fr":       25.0,
    "SlipAngle_rl":       25.0,
    "SlipAngle_rr":       25.0,
    "steerAngle":         450.0,
}

BASIC_DIM = len(OBS_ENABLED_CHANNELS) + NUM_SENSORS   # 25
OBS_DIM = (
    BASIC_DIM                           # 25
    + 1                                 # out_of_track
    + CURV_LOOK_AHEAD_VECTOR_SIZE       # 12
    + PAST_ACTIONS_WINDOW * 3           # 9
    + 3                                 # current action
    + PAST_ACTIONS_WINDOW * BASIC_DIM   # 75
)  # = 125

# ── Action constants ───────────────────────────────────────────────────────────
STEER_MAX = 302.4   # degrees — from steer_map.csv (x=1.0 → y=302.4)

# ── Track → racing-line CSV mapping ───────────────────────────────────────────
# Key: track name as it appears in episode_meta.json
# Value: racing-line CSV filename under AssettoCorsaConfigs/tracks/
TRACK_RACING_LINE = {
    "monza":          "monza-racing_line.csv",
    "imola":          "imola-racing_line.csv",
    "ks_silverstone": "ks_silverstone-gp-racing_line.csv",
    "spa":            "spa-racing_line.csv",
    "mugello":        "mugello-racing_line.csv",
    "ks_laguna_seca": "ks_laguna_seca-racing_line.csv",
    "ks_barcelona":   "ks_barcelona-layout_gp-racing_line.csv",
    "ks_red_bull_ring": "ks_red_bull_ring-layout_gp-racing_line.csv",
    "monaco":         "monaco-racing_line.csv",
}

RAY_LABELS = [
    "ray_r90", "ray_r67", "ray_r45", "ray_r22", "ray_r11",
    "ray_fwd",
    "ray_l11", "ray_l22", "ray_l45", "ray_l67", "ray_l90",
]


# ── Parquet row → state dict ───────────────────────────────────────────────────

def row_to_state(row) -> dict:
    """
    Convert a parquet row to a state dict with the same keys that
    preprocess_bc.py's build_obs() and extract_action() expect.
    """
    sensors = [float(row[lbl]) if not pd.isna(row[lbl]) else MAX_RAY_LEN
               for lbl in RAY_LABELS]
    return {
        "speed":                    float(row["speed_ms"]),
        "gap":                      float(row["gap"]),
        "LastFF":                   float(row["last_ff"]),
        "RPM":                      float(row["rpm"]),
        "accelX":                   float(row["accel_x"]),
        "accelY":                   float(row["accel_y"]),
        "actualGear":               float(row["gear"]),
        "angular_velocity_y":       float(row["yaw_rate"]),
        "local_velocity_x":         float(row["vel_x"]),
        "local_velocity_y":         float(row["vel_y"]),
        "SlipAngle_fl":             float(row["sa_fl"]),
        "SlipAngle_fr":             float(row["sa_fr"]),
        "SlipAngle_rl":             float(row["sa_rl"]),
        "SlipAngle_rr":             float(row["sa_rr"]),
        "out_of_track":             bool(row["out_of_track"]),
        "LapDist":                  float(row["lap_dist"]),
        "NormalizedSplinePosition": float(row["nsp"]),
        "LapCount":                 int(row["lap_count"]),
        "steerAngle":               float(row["steer_deg"]),
        "accStatus":                float(row["throttle"]),
        "brakeStatus":              float(row["brake"]),
        "sensors":                  sensors,
    }


# ── Obs builder (mirrors preprocess_bc.py build_obs exactly) ──────────────────

def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _basic_obs(state: dict, channel_scales: np.ndarray) -> np.ndarray:
    obs = np.array([state[ch] for ch in OBS_ENABLED_CHANNELS], dtype=np.float32)
    obs /= channel_scales
    sensors = np.array(state["sensors"], dtype=np.float32) / MAX_RAY_LEN
    return np.hstack([obs, sensors])   # (25,)


def build_obs(state: dict, history: list, ref_lap: ReferenceLap,
              channel_scales: np.ndarray) -> np.ndarray:
    """Build the 125-dim obs vector for one timestep."""
    # 1. Basic obs (25)
    obs = _basic_obs(state, channel_scales)

    # 2. out_of_track (1)
    obs = np.hstack([obs, float(state["out_of_track"])])

    # 3. Curvature look-ahead (12)
    try:
        curv = ref_lap.get_curvature_segment(
            float(state["LapDist"]),
            CURV_LOOK_AHEAD_DISTANCE,
            CURV_LOOK_AHEAD_VECTOR_SIZE,
        )
        if isinstance(curv, tuple):
            curv = curv[0]
        curv = np.array(curv, dtype=np.float32) / CURV_NORMALIZATION_CONSTANT
    except Exception:
        curv = np.zeros(CURV_LOOK_AHEAD_VECTOR_SIZE, dtype=np.float32)
    obs = np.hstack([obs, curv])

    # 4. Past actions (9) — warm-start with current action if history is short
    steer_scale = OBS_CHANNEL_SCALES["steerAngle"]
    cur_steer = state["steerAngle"] / steer_scale
    cur_pedal = state["accStatus"]
    cur_brake = state["brakeStatus"]

    if len(history) < PAST_ACTIONS_WINDOW:
        n_missing = PAST_ACTIONS_WINDOW - len(history)
        steer_slots = [cur_steer] * n_missing + [h["steerAngle"] / steer_scale for h in history]
        pedal_slots = [cur_pedal] * n_missing + [h["accStatus"]               for h in history]
        brake_slots = [cur_brake] * n_missing + [h["brakeStatus"]             for h in history]
        obs = np.hstack([obs,
                         np.array(steer_slots, dtype=np.float32),
                         np.array(pedal_slots, dtype=np.float32),
                         np.array(brake_slots, dtype=np.float32)])
    else:
        past = history[-PAST_ACTIONS_WINDOW:]
        obs = np.hstack([obs,
                         [h["steerAngle"] / steer_scale for h in past],
                         [h["accStatus"]               for h in past],
                         [h["brakeStatus"]             for h in past]])

    # 5. Current action (3)
    obs = np.hstack([obs, cur_steer, cur_pedal, cur_brake])

    # 6. Previous basic obs (3 × 25 = 75)
    cur_basic = _basic_obs(state, channel_scales)
    if len(history) < PAST_ACTIONS_WINDOW:
        n_missing = PAST_ACTIONS_WINDOW - len(history)
        filler = [cur_basic] * n_missing + [_basic_obs(h, channel_scales) for h in history]
        prev = np.hstack(filler)
    else:
        prev = np.hstack([_basic_obs(h, channel_scales)
                          for h in history[-PAST_ACTIONS_WINDOW:]])
    obs = np.hstack([obs, prev])

    return obs.astype(np.float32)


# ── Action extraction (SAC policy space [0, 1]) ────────────────────────────────
#
# How the SAC action becomes an AC control (via our_env → VJoyControl → car_control):
#
#   map_action (vjoy.py):    ac_cmd  = sac * 2 - 1          [-1, 1]
#   car_control.py steer:    axis    = ac_cmd + 1            [0, 2]
#   car_control.py acc/brake: axis   = (ac_cmd + 1) / 2     [0, 1]
#
# Inverting:
#   steer  → steerAngle = (sac * 2) * STEER_MAX
#           → sac = (steerAngle / STEER_MAX + 1) / 2        [0, 1]  0.5 = straight
#   pedal  → accStatus  = sac  (clamped to [0,1])
#           → sac = accStatus                                [0, 1]
#   brake  → brakeStatus = sac  (clamped to [0,1])
#           → sac = brakeStatus                              [0, 1]  0 = no brake
#
# NOTE: brake_map is NOT used here. It was the old BC convention.  The SAC
# expects brake targets in [0,1] where 0 = no brake and 1 = full brake.

def extract_action(state: dict, brake_map: BrakeMap) -> np.ndarray:  # brake_map kept for API compat
    """
    Convert raw AC telemetry to SAC policy space [0, 1].

        steer  = clip((steerAngle / STEER_MAX + 1) / 2, 0, 1)
                 0 = full left, 0.5 = straight, 1 = full right
        pedal  = clip(accStatus, 0, 1)      — throttle [0,1] directly
        brake  = clip(brakeStatus, 0, 1)    — brake [0,1] directly  (0 = none)
    """
    steer = float(np.clip((state["steerAngle"] / STEER_MAX + 1.0) / 2.0, 0.0, 1.0))
    pedal = float(_clamp(state["accStatus"],   0.0, 1.0))
    brake = float(_clamp(state["brakeStatus"], 0.0, 1.0))
    return np.array([steer, pedal, brake], dtype=np.float32)


# ── Per-episode processor ──────────────────────────────────────────────────────

def process_episode(parquet_path: Path, ref_lap: ReferenceLap,
                    brake_map: BrakeMap, channel_scales: np.ndarray,
                    min_speed_ms: float = 5.0):
    """
    Process one parquet episode → (obs array, actions array).

    Returns
    -------
    obs     np.ndarray (N, 125)
    actions np.ndarray (N, 3)
    """
    df = pd.read_parquet(parquet_path)
    logger.info(f"  {parquet_path.name}: {len(df)} frames")

    all_obs, all_act = [], []
    history = []

    for _, row in df.iterrows():
        state = row_to_state(row)

        # Skip low-speed frames (pitting / repositioning)
        if float(state["speed"]) < min_speed_ms:
            history.append(state)
            continue

        obs = build_obs(state, history, ref_lap, channel_scales)
        act = extract_action(state, brake_map)

        all_obs.append(obs)
        all_act.append(act)
        history.append(state)

    if not all_obs:
        return None, None

    return (
        np.stack(all_obs, axis=0).astype(np.float32),
        np.stack(all_act, axis=0).astype(np.float32),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess collectDataAI parquet files → .npz for BC pre-training"
    )
    parser.add_argument("--data-dir",   default="collectDataAI/data",
                        help="Root of collectDataAI/data tree (default: collectDataAI/data)")
    parser.add_argument("--output-dir", default="AICLONE/data",
                        help="Where to write the .npz and .json (default: AICLONE/data)")
    parser.add_argument("--min-speed",  type=float, default=5.0,
                        help="Skip frames below this speed in m/s (default: 5.0)")
    args = parser.parse_args()

    data_dir   = _REPO / args.data_dir
    output_dir = _REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    brake_map_path = str(_AC_CONFIGS / "cars" / "ks_mazda_miata" / "brake_map.csv")
    brake_map = BrakeMap.load(brake_map_path)
    logger.info(f"Brake map loaded: {brake_map_path}")

    channel_scales = np.array(
        [OBS_CHANNEL_SCALES[ch] for ch in OBS_ENABLED_CHANNELS], dtype=np.float32
    )

    # ── Discover all parquet files grouped by track ───────────────────────────
    # Structure: data/{car}/{track}/{session}/episode_NNNN.parquet
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    logger.info(f"Found {len(parquet_files)} parquet file(s)")

    all_obs, all_act, episode_ends = [], [], []
    step_cursor = 0

    for pq_path in parquet_files:
        # Infer track from path: data/{car}/{track}/{session}/episode_*.parquet
        track = pq_path.parent.parent.name

        racing_line_file = TRACK_RACING_LINE.get(track)
        if racing_line_file is None:
            logger.warning(f"No racing-line entry for track '{track}' — skipping {pq_path.name}")
            continue

        racing_line_path = _AC_CONFIGS / "tracks" / racing_line_file
        if not racing_line_path.exists():
            logger.warning(f"Racing-line CSV not found: {racing_line_path} — skipping")
            continue

        ref_lap = ReferenceLap(str(racing_line_path), use_target_speed=False)
        logger.info(f"Processing [{track}] {pq_path.name}")

        obs, act = process_episode(
            pq_path, ref_lap, brake_map, channel_scales,
            min_speed_ms=args.min_speed,
        )
        if obs is None:
            logger.warning(f"  No valid steps — skipping")
            continue

        all_obs.append(obs)
        all_act.append(act)
        step_cursor += len(obs)
        episode_ends.append(step_cursor - 1)
        logger.info(f"  → {len(obs)} steps added")

    if not all_obs:
        raise ValueError("No valid steps found across all parquet files.")

    obs_all = np.concatenate(all_obs, axis=0)
    act_all = np.concatenate(all_act, axis=0)
    ep_ends = np.array(episode_ends, dtype=np.int64)

    assert obs_all.shape[1] == OBS_DIM, (
        f"Obs dim mismatch: got {obs_all.shape[1]}, expected {OBS_DIM}"
    )

    logger.info(f"Total steps   : {len(obs_all)}")
    logger.info(f"Episodes      : {len(ep_ends)}")
    logger.info(f"obs shape     : {obs_all.shape}")
    logger.info(f"actions shape : {act_all.shape}")
    logger.info(f"Steer range   : [{act_all[:,0].min():.3f}, {act_all[:,0].max():.3f}]")
    logger.info(f"Pedal range   : [{act_all[:,1].min():.3f}, {act_all[:,1].max():.3f}]")
    logger.info(f"Brake range   : [{act_all[:,2].min():.3f}, {act_all[:,2].max():.3f}]")

    # ── Save ─────────────────────────────────────────────────────────────────
    npz_path  = output_dir / "aiclone_dataset.npz"
    json_path = output_dir / "aiclone_dataset.json"

    np.savez(str(npz_path), obs=obs_all, actions=act_all, episode_ends=ep_ends)
    logger.info(f"Saved: {npz_path}")

    meta = {
        "obs_dim":       int(obs_all.shape[1]),
        "action_dim":    3,
        "n_steps":       int(len(obs_all)),
        "n_episodes":    int(len(ep_ends)),
        "steer_max_deg": STEER_MAX,
        "min_speed_ms":  float(args.min_speed),
        "tracks":        list({pq.parent.parent.name for pq in parquet_files}),
        "created":       datetime.utcnow().isoformat(),
    }
    with open(str(json_path), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved: {json_path}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
