"""
preprocess_bc.py — One-time preprocessing of raw pkl trajectory files into
                   (obs, actions) numpy arrays ready for BC training.

Reads the raw pkl state dicts, reconstructs the 125-dim obs vector exactly
as ac_env.get_obs() would, and extracts normalised [steer, pedal, brake]
actions. Saves a single .npz + .json metadata file per run.

The result is self-contained — train_bc.py reads only the .npz and never
touches a pkl file again.

Usage
-----
    # From project root:
    .\\AssetoCorsa\\Scripts\\python.exe gym/preprocess_bc.py \\
        --data \\
            data/monza/ks_mazda_miata/20240229_HC/laps/monza_ks_mazda_miata_adrianremonda_stint_1.pkl \\
            data/monza/ks_mazda_miata/20240313_OP/laps/monza_ks_mazda_miata_KC_Sim_Racer_stint_1.pkl \\
        --output-dir data/processed/

Output
------
    data/processed/monza_ks_mazda_miata_mlp.npz   — arrays
    data/processed/monza_ks_mazda_miata_mlp.json  — metadata

.npz arrays
-----------
    obs           (N, 125)  float32  — full obs vector, ready for DataLoader
    actions       (N, 3)    float32  — [steer, pedal, brake] in [-1, 1] SAC space
    episode_ends  (E,)      int64    — index of last step in each episode
                                       (used by train_bc.py for episode-level splits)
    lap_counts    (N,)      int32    — LapCount per step (useful for filtering out-laps)
    out_of_track  (N,)      bool     — out-of-track flag per step
    speeds        (N,)      float32  — speed m/s per step (sanity check)

# TODO: track and car are hardcoded to monza / ks_mazda_miata.
#       When adding a second track or car, extract track/car from the pkl
#       path convention data/{track}/{car}/{session}/laps/*.pkl and resolve
#       config files dynamically from AssettoCorsaConfigs/.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent.parent
GYM_DIR       = REPO_ROOT / "gym"
ASSETTO_GYM   = REPO_ROOT / "assetto_corsa_gym" / "assetto_corsa_gym"
AC_CONFIGS    = ASSETTO_GYM / "AssettoCorsaConfigs"

for p in [str(GYM_DIR), str(ASSETTO_GYM)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from AssettoCorsaEnv.reference_lap import ReferenceLap   # noqa: E402
from AssettoCorsaEnv.brake_map     import BrakeMap       # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded config — monza / ks_mazda_miata
# TODO: make this dynamic when adding more tracks/cars (see module docstring)
# ---------------------------------------------------------------------------
TRACK = "monza"
CAR   = "ks_mazda_miata"

REF_LAP_FILE   = str(AC_CONFIGS / "tracks" / "monza-racing_line.csv")
BRAKE_MAP_FILE = str(AC_CONFIGS / "cars"   / "ks_mazda_miata" / "brake_map.csv")
STEER_MAP_FILE = str(AC_CONFIGS / "cars"   / "ks_mazda_miata" / "steer_map.csv")

# ---------------------------------------------------------------------------
# Constants — must match ac_env.py exactly
# ---------------------------------------------------------------------------
PAST_ACTIONS_WINDOW        = 3
CURV_LOOK_AHEAD_DISTANCE   = 300.0
CURV_LOOK_AHEAD_VECTOR_SIZE = 12
CURV_NORMALIZATION_CONSTANT = 0.1
MAX_RAY_LEN                = 200.0
TOP_SPEED_MS               = 80.0
NUM_SENSORS                = 11

OBS_ENABLED_CHANNELS = [
    "speed", "gap", "LastFF", "RPM", "accelX", "accelY",
    "actualGear", "angular_velocity_y",
    "local_velocity_x", "local_velocity_y",
    "SlipAngle_fl", "SlipAngle_fr", "SlipAngle_rl", "SlipAngle_rr",
]

OBS_CHANNEL_SCALES = {
    "speed":             TOP_SPEED_MS,
    "gap":               10.0,
    "LastFF":            1.0,
    "RPM":               10000.0,
    "accelX":            5.0,
    "accelY":            5.0,
    "actualGear":        8.0,
    "angular_velocity_y": np.pi,
    "local_velocity_x":  TOP_SPEED_MS,
    "local_velocity_y":  20.0,
    "SlipAngle_fl":      25.0,
    "SlipAngle_fr":      25.0,
    "SlipAngle_rl":      25.0,
    "SlipAngle_rr":      25.0,
    "steerAngle":        450,      # must match ac_env.py obs_channels_info['steerAngle']
}

# 125 = 14 + 11 + 1 + 12 + 9 + 3 + 75
BASIC_DIM = len(OBS_ENABLED_CHANNELS) + NUM_SENSORS   # 25
OBS_DIM   = (
    BASIC_DIM                            # 14 telemetry + 11 sensors = 25
    + 1                                  # out_of_track
    + CURV_LOOK_AHEAD_VECTOR_SIZE        # 12 curvature
    + PAST_ACTIONS_WINDOW * 3            # 9  past actions
    + 3                                  # 3  current action
    + PAST_ACTIONS_WINDOW * BASIC_DIM    # 75 prev basic obs
)  # = 125


# ---------------------------------------------------------------------------
# Obs reconstruction — mirrors ac_env.get_obs() exactly
# ---------------------------------------------------------------------------

def _basic_obs(state: dict, scales: np.ndarray) -> np.ndarray:
    """14 telemetry channels + 11 ray sensors, normalised."""
    obs = np.array([state[ch] for ch in OBS_ENABLED_CHANNELS], dtype=np.float32)
    obs /= scales
    sensors = np.array(state["sensors"], dtype=np.float32) / MAX_RAY_LEN
    return np.hstack([obs, sensors])   # (25,)


def build_obs(state: dict, history: List[dict], ref_lap: ReferenceLap,
              channel_scales: np.ndarray) -> Optional[np.ndarray]:
    """
    Reconstruct the full 125-dim obs vector for one timestep.

    Returns None and logs a warning if any required field is missing.
    """
    required = list(OBS_ENABLED_CHANNELS) + [
        "out_of_track", "LapDist", "sensors",
        "steerAngle", "accStatus", "brakeStatus",
    ]
    for field in required:
        if field not in state:
            logger.warning(f"Missing field '{field}' — skipping step")
            return None

    # 1. Basic obs
    obs = _basic_obs(state, channel_scales)                             # (25,)

    # 2. out_of_track flag
    obs = np.hstack([obs, float(state["out_of_track"])])                # (26,)

    # 3. Curvature look-ahead
    try:
        curv = ref_lap.get_curvature_segment(
            float(state["LapDist"]),
            CURV_LOOK_AHEAD_DISTANCE,
            CURV_LOOK_AHEAD_VECTOR_SIZE,
        )
        if isinstance(curv, tuple):
            curv = curv[0]
        curv = np.array(curv, dtype=np.float32) / CURV_NORMALIZATION_CONSTANT
    except Exception as e:
        logger.warning(f"Curvature failed at LapDist={state['LapDist']:.1f}: {e}")
        curv = np.zeros(CURV_LOOK_AHEAD_VECTOR_SIZE, dtype=np.float32)
    obs = np.hstack([obs, curv])                                        # (38,)

    # 4. Past actions
    # When history is shorter than PAST_ACTIONS_WINDOW, fill the missing slots
    # by repeating the current state's own action.  This matches the warm-start
    # behaviour at inference time (see ac_env.get_obs fix) so the model never
    # sees the OOD all-zeros pattern that causes the left-steer bias (ISSUE-011
    # Fix 1).
    steer_scale = OBS_CHANNEL_SCALES["steerAngle"]
    cur_steer  = state["steerAngle"] / steer_scale
    cur_pedal  = state["accStatus"]
    cur_brake  = state["brakeStatus"]

    if len(history) < PAST_ACTIONS_WINDOW:
        n_missing = PAST_ACTIONS_WINDOW - len(history)
        # Build the full PAST_ACTIONS_WINDOW-length lists by prepending the
        # current action for every missing slot, then appending real history.
        steer_slots = [cur_steer] * n_missing + [h["steerAngle"] / steer_scale for h in history]
        pedal_slots = [cur_pedal] * n_missing + [h["accStatus"]               for h in history]
        brake_slots = [cur_brake] * n_missing + [h["brakeStatus"]             for h in history]
        obs = np.hstack([obs,
                         np.array(steer_slots, dtype=np.float32),
                         np.array(pedal_slots, dtype=np.float32),
                         np.array(brake_slots, dtype=np.float32)])
    else:
        past = history[-PAST_ACTIONS_WINDOW:]
        obs = np.hstack([
            obs,
            [h["steerAngle"] / steer_scale for h in past],
            [h["accStatus"]               for h in past],
            [h["brakeStatus"]             for h in past],
        ])                                                              # (47,)

    # 5. Current action
    obs = np.hstack([
        obs,
        cur_steer,
        cur_pedal,
        cur_brake,
    ])                                                                  # (50,)

    # 6. Previous basic obs (3 × 25 = 75)
    # Same warm-start: fill missing history slots with current basic obs.
    cur_basic = _basic_obs(state, channel_scales)
    if len(history) < PAST_ACTIONS_WINDOW:
        n_missing = PAST_ACTIONS_WINDOW - len(history)
        filler_blocks = [cur_basic] * n_missing + [
            _basic_obs(h, channel_scales) for h in history
        ]
        prev = np.hstack(filler_blocks)
    else:
        prev = np.hstack([_basic_obs(h, channel_scales)
                          for h in history[-PAST_ACTIONS_WINDOW:]])
    obs = np.hstack([obs, prev])                                        # (125,)

    return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# Action extraction — mirrors DataLoader.get_actions_from_state()
# ---------------------------------------------------------------------------

def extract_action(state: dict, brake_map: BrakeMap, steer_max: float) -> np.ndarray:
    """[steer, pedal, brake] all in [-1, 1] SAC space.

    Steer is clipped to [-1, 1]: human drivers occasionally exceed steer_max
    (e.g. parking manoeuvres, spins). The network outputs tanh ∈ [-1, 1] so
    targets outside that range cause gradient saturation during BC training.
    """
    steer = float(np.clip(state["steerAngle"] / steer_max, -1.0, 1.0))
    pedal = (state["accStatus"] - 0.5) * 2.0
    brake = float(brake_map.get_x(state["brakeStatus"]))
    return np.array([steer, pedal, brake], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_files(
    pkl_paths:    List[str],
    ref_lap:      ReferenceLap,
    brake_map:    BrakeMap,
    steer_max:    float,
    skip_outlap:  bool = True,
    min_speed_ms: float = 5.0,
):
    """
    Process a list of pkl files and return stacked arrays + episode metadata.

    Parameters
    ----------
    min_speed_ms : float
        Skip any step where speed < min_speed_ms (m/s).  These are typically
        pitting or repositioning steps with erratic steering (ROOT CAUSE 3 in
        ISSUE-011).  History still advances through skipped steps so context
        is not broken.  Default 5.0 m/s.

    Returns
    -------
    obs          np.ndarray (N, 125)
    actions      np.ndarray (N, 3)
    episode_ends np.ndarray (E,)   — index of last step in each episode
    lap_counts   np.ndarray (N,)
    out_of_track np.ndarray (N,)
    speeds       np.ndarray (N,)
    """
    channel_scales = np.array(
        [OBS_CHANNEL_SCALES[ch] for ch in OBS_ENABLED_CHANNELS], dtype=np.float32
    )

    all_obs, all_act, all_laps, all_oot, all_spd = [], [], [], [], []
    episode_ends: List[int] = []
    skipped_steps = 0
    step_cursor   = 0

    for pkl_path in pkl_paths:
        logger.info(f"Processing {Path(pkl_path).name} ...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        states = data["states"]

        history: List[dict] = []
        ep_steps = 0
        skipped_outlap = 0
        skipped_lowspeed = 0

        for state in states:
            if skip_outlap and int(state.get("LapCount", 0)) == 0:
                history.append(state)   # keep history flowing across out-lap
                skipped_outlap += 1
                continue

            # Fix 3 (ISSUE-011 ROOT CAUSE 3): skip low-speed steps with erratic
            # steering (pitting / repositioning).  History still advances so the
            # next in-range step has correct context.
            if min_speed_ms > 0.0 and float(state.get("speed", 0.0)) < min_speed_ms:
                history.append(state)
                skipped_lowspeed += 1
                continue

            obs = build_obs(state, history, ref_lap, channel_scales)
            if obs is None:
                skipped_steps += 1
                history.append(state)
                continue

            action = extract_action(state, brake_map, steer_max)

            all_obs.append(obs)
            all_act.append(action)
            all_laps.append(int(state.get("LapCount", 0)))
            all_oot.append(bool(state.get("out_of_track", False)))
            all_spd.append(float(state.get("speed", 0.0)))

            history.append(state)
            ep_steps += 1

        if ep_steps > 0:
            step_cursor += ep_steps
            episode_ends.append(step_cursor - 1)

        logger.info(
            f"  {ep_steps} steps added  "
            f"(out-lap skipped={skipped_outlap}  "
            f"low-speed skipped={skipped_lowspeed}  "
            f"bad_fields skipped={skipped_steps})"
        )

    if not all_obs:
        raise ValueError("No valid steps found in any pkl file.")

    return (
        np.stack(all_obs,  axis=0).astype(np.float32),
        np.stack(all_act,  axis=0).astype(np.float32),
        np.array(episode_ends, dtype=np.int64),
        np.array(all_laps,     dtype=np.int32),
        np.array(all_oot,      dtype=bool),
        np.array(all_spd,      dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess AC pkl files → (obs, actions) .npz for BC training.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example:
  python gym/preprocess_bc.py \\
      --data \\
          data/monza/ks_mazda_miata/20240229_HC/laps/monza_ks_mazda_miata_adrianremonda_stint_1.pkl \\
          data/monza/ks_mazda_miata/20240313_OP/laps/monza_ks_mazda_miata_KC_Sim_Racer_stint_1.pkl \\
      --output-dir data/processed/
        """,
    )
    parser.add_argument(
        "--data", nargs="+", required=True, metavar="PKL",
        help="One or more paths to .pkl trajectory files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Directory to write the .npz and .json files.",
    )
    parser.add_argument(
        "--skip-outlap", action="store_true", default=True,
        help="Skip LapCount==0 steps (out-lap, cold tyres). On by default.",
    )
    parser.add_argument(
        "--keep-outlap", dest="skip_outlap", action="store_false",
        help="Include out-lap steps in the dataset.",
    )
    parser.add_argument(
        "--min-speed", type=float, default=5.0, metavar="M/S",
        help=(
            "Skip steps where speed < MIN_SPEED (m/s). "
            "Removes pitting/repositioning steps with erratic steering "
            "(ISSUE-011 Fix 3). Default: 5.0. Set to 0 to disable."
        ),
    )
    args = parser.parse_args()

    # Validate inputs
    pkl_paths = []
    for p in args.data:
        p = str(Path(p).resolve())
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
        pkl_paths.append(p)

    logger.info(f"Input files: {len(pkl_paths)}")
    for p in pkl_paths:
        logger.info(f"  {p}")
    logger.info(f"Track: {TRACK}  Car: {CAR}")
    logger.info(f"Ref lap  : {REF_LAP_FILE}")
    logger.info(f"Brake map: {BRAKE_MAP_FILE}")
    logger.info(f"Steer map: {STEER_MAP_FILE}")

    # Load config files
    import pandas as pd
    steer_max = float(pd.read_csv(STEER_MAP_FILE).values[1, 1])
    logger.info(f"steer_max = {steer_max:.1f} deg")

    brake_map = BrakeMap.load(BRAKE_MAP_FILE)
    ref_lap   = ReferenceLap(REF_LAP_FILE, use_target_speed=False)

    # Process
    logger.info(
        f"Building obs and actions "
        f"(skip_outlap={args.skip_outlap}, min_speed={args.min_speed} m/s) ..."
    )
    obs, actions, episode_ends, lap_counts, out_of_track, speeds = process_files(
        pkl_paths, ref_lap, brake_map, steer_max,
        skip_outlap=args.skip_outlap,
        min_speed_ms=args.min_speed,
    )

    # Sanity check obs dim
    assert obs.shape[1] == OBS_DIM, (
        f"Obs dim mismatch: got {obs.shape[1]}, expected {OBS_DIM}. "
        f"Check constants match ac_env.py."
    )

    logger.info(f"Total steps     : {len(obs)}")
    logger.info(f"Episodes        : {len(episode_ends)}")
    logger.info(f"obs shape       : {obs.shape}")
    logger.info(f"actions shape   : {actions.shape}")
    logger.info(f"Speed range     : {speeds.min():.1f} – {speeds.max():.1f} m/s")
    logger.info(f"Out-of-track    : {out_of_track.sum()} steps ({100*out_of_track.mean():.1f}%)")

    # Save
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = f"{TRACK}_{CAR}_mlp"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"

    np.savez(
        str(npz_path),
        obs=obs,
        actions=actions,
        episode_ends=episode_ends,
        lap_counts=lap_counts,
        out_of_track=out_of_track,
        speeds=speeds,
    )
    logger.info(f"Saved: {npz_path}")

    meta = {
        "track":               TRACK,
        "car":                 CAR,
        "model_type":          "mlp",
        "obs_dim":             int(obs.shape[1]),
        "action_dim":          int(actions.shape[1]),
        "past_actions_window": PAST_ACTIONS_WINDOW,
        "n_steps":             int(len(obs)),
        "n_episodes":          int(len(episode_ends)),
        "steer_max":           float(steer_max),
        "steer_obs_scale":     float(OBS_CHANNEL_SCALES["steerAngle"]),
        "min_speed_ms":        float(args.min_speed),
        "source_pkl_files":    pkl_paths,
        "ref_lap":             REF_LAP_FILE,
        "brake_map":           BRAKE_MAP_FILE,
        "created":             datetime.utcnow().isoformat(),
    }
    with open(str(json_path), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved: {json_path}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
