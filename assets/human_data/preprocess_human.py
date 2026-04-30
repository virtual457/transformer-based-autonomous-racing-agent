"""
preprocess_human.py — Preprocess human-collected pkl episodes into (obs, actions)
                      numpy arrays ready for BC training.

Differences from gym/preprocess_bc.py:
  1. Dynamic --track / --car  (no hardcoded monza/ks_mazda_miata).
  2. Auto-discovers raw pkls from  human_data/raw/{track}_{car}/*.pkl
  3. Trims the LAST --trim-tail-seconds (default 7 s) from every episode
     using the per-frame timestamp_ac field — these are the crash frames
     that would poison the training signal.
  4. Config files (brake map, steer map, ref lap) are resolved from
     AssettoCorsaConfigs/ with graceful fallbacks and CLI overrides.

Frame stacking (identical to preprocess_bc.py):
  obs (125-dim) = 14 telemetry
                + 11 ray sensors
                + 1  out_of_track
                + 12 curvature look-ahead
                + 9  past actions  (3 frames × 3 channels)
                + 3  current action
                + 75 prev basic obs (3 frames × 25)
  At episode start the history is warm-started by repeating the first frame.

Usage:
    python human_data/preprocess_human.py --track monza --car lotus_exos_125
    python human_data/preprocess_human.py --track monza --car lotus_exos_125 \\
        --steer-max 318 --trim-tail-seconds 7

Output:
    human_data/processed/monza_lotus_exos_125_mlp.npz
    human_data/processed/monza_lotus_exos_125_mlp.json
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

# ── Path bootstrap ─────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
GYM_DIR     = REPO_ROOT / "gym"
ASSETTO_GYM = REPO_ROOT / "assetto_corsa_gym" / "assetto_corsa_gym"
AC_CONFIGS  = ASSETTO_GYM / "AssettoCorsaConfigs"
HUMAN_DATA  = REPO_ROOT / "human_data"

for _p in [str(GYM_DIR), str(ASSETTO_GYM)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AssettoCorsaEnv.reference_lap import ReferenceLap  # noqa: E402
from AssettoCorsaEnv.brake_map     import BrakeMap      # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants — must match ac_env.py exactly ───────────────────────────────────
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
    "steerAngle":         450,
}

BASIC_DIM = len(OBS_ENABLED_CHANNELS) + NUM_SENSORS   # 25
OBS_DIM   = (
    BASIC_DIM
    + 1
    + CURV_LOOK_AHEAD_VECTOR_SIZE
    + PAST_ACTIONS_WINDOW * 3
    + 3
    + PAST_ACTIONS_WINDOW * BASIC_DIM
)  # = 125


# ── Config file resolution ─────────────────────────────────────────────────────

def _resolve_config_files(track: str, car: str,
                           brake_map_override: str = None,
                           steer_map_override: str = None,
                           ref_lap_override:   str = None):
    """
    Resolve brake map, steer map, and reference lap paths.

    Priority: CLI override → car-specific configs → ks_mazda_miata fallback.
    """
    car_dir    = AC_CONFIGS / "cars" / car
    mazda_dir  = AC_CONFIGS / "cars" / "ks_mazda_miata"
    tracks_dir = AC_CONFIGS / "tracks"

    # ── Brake map ─────────────────────────────────────────────────────────────
    if brake_map_override:
        brake_map_path = brake_map_override
    elif (car_dir / "brake_map.csv").exists():
        brake_map_path = str(car_dir / "brake_map.csv")
    else:
        brake_map_path = str(mazda_dir / "brake_map.csv")
        logger.warning(
            f"No brake_map.csv for '{car}' — falling back to ks_mazda_miata. "
            f"Pass --brake-map to override."
        )

    # ── Steer map ─────────────────────────────────────────────────────────────
    if steer_map_override:
        steer_map_path = steer_map_override
    elif (car_dir / "steer_map.csv").exists():
        steer_map_path = str(car_dir / "steer_map.csv")
    else:
        steer_map_path = str(mazda_dir / "steer_map.csv")
        logger.warning(
            f"No steer_map.csv for '{car}' — falling back to ks_mazda_miata. "
            f"Pass --steer-map or --steer-max to override."
        )

    # ── Reference lap ─────────────────────────────────────────────────────────
    if ref_lap_override:
        ref_lap_path = ref_lap_override
    else:
        # Try exact match first, then any file containing the track name
        candidates = sorted(tracks_dir.glob(f"{track}*.csv"))
        if candidates:
            ref_lap_path = str(candidates[0])
            if len(candidates) > 1:
                logger.info(f"Multiple ref-lap files for '{track}', using: {candidates[0].name}")
        else:
            raise FileNotFoundError(
                f"No reference lap CSV found for track '{track}' in {tracks_dir}. "
                f"Pass --ref-lap to specify manually."
            )

    return brake_map_path, steer_map_path, ref_lap_path


# ── Obs reconstruction ─────────────────────────────────────────────────────────

def _basic_obs(state: dict, scales: np.ndarray) -> np.ndarray:
    obs     = np.array([state[ch] for ch in OBS_ENABLED_CHANNELS], dtype=np.float32)
    obs    /= scales
    sensors = np.array(state["sensors"], dtype=np.float32) / MAX_RAY_LEN
    return np.hstack([obs, sensors])  # (25,)


def build_obs(state: dict, history: List[dict],
              ref_lap: ReferenceLap, channel_scales: np.ndarray) -> Optional[np.ndarray]:
    required = list(OBS_ENABLED_CHANNELS) + [
        "out_of_track", "LapDist", "sensors",
        "steerAngle", "accStatus", "brakeStatus",
    ]
    for field in required:
        if field not in state:
            logger.warning(f"Missing field '{field}' — skipping step")
            return None

    # 1. Basic obs (25)
    obs = _basic_obs(state, channel_scales)

    # 2. out_of_track flag (1)
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
    except Exception as e:
        logger.warning(f"Curvature failed at LapDist={state['LapDist']:.1f}: {e}")
        curv = np.zeros(CURV_LOOK_AHEAD_VECTOR_SIZE, dtype=np.float32)
    obs = np.hstack([obs, curv])  # (38,)

    # 4. Past actions — warm-start if history shorter than window (9)
    steer_scale = OBS_CHANNEL_SCALES["steerAngle"]
    cur_steer   = state["steerAngle"] / steer_scale
    cur_pedal   = state["accStatus"]
    cur_brake   = state["brakeStatus"]

    if len(history) < PAST_ACTIONS_WINDOW:
        n_miss = PAST_ACTIONS_WINDOW - len(history)
        steer_slots = [cur_steer] * n_miss + [h["steerAngle"] / steer_scale for h in history]
        pedal_slots = [cur_pedal] * n_miss + [h["accStatus"]               for h in history]
        brake_slots = [cur_brake] * n_miss + [h["brakeStatus"]             for h in history]
        obs = np.hstack([obs,
                         np.array(steer_slots, dtype=np.float32),
                         np.array(pedal_slots, dtype=np.float32),
                         np.array(brake_slots, dtype=np.float32)])
    else:
        past = history[-PAST_ACTIONS_WINDOW:]
        obs  = np.hstack([
            obs,
            [h["steerAngle"] / steer_scale for h in past],
            [h["accStatus"]               for h in past],
            [h["brakeStatus"]             for h in past],
        ])  # (47,)

    # 5. Current action (3)
    obs = np.hstack([obs, cur_steer, cur_pedal, cur_brake])  # (50,)

    # 6. Previous basic obs — warm-start if needed (75)
    cur_basic = _basic_obs(state, channel_scales)
    if len(history) < PAST_ACTIONS_WINDOW:
        n_miss  = PAST_ACTIONS_WINDOW - len(history)
        blocks  = [cur_basic] * n_miss + [_basic_obs(h, channel_scales) for h in history]
        prev    = np.hstack(blocks)
    else:
        prev = np.hstack([_basic_obs(h, channel_scales)
                          for h in history[-PAST_ACTIONS_WINDOW:]])
    obs = np.hstack([obs, prev])  # (125,)

    return obs.astype(np.float32)


# ── Action extraction ──────────────────────────────────────────────────────────

def extract_action(state: dict, brake_map: BrakeMap, steer_max: float) -> np.ndarray:
    steer = float(np.clip(state["steerAngle"] / steer_max, -1.0, 1.0))
    pedal = (state["accStatus"] - 0.5) * 2.0
    brake = float(brake_map.get_x(state["brakeStatus"]))
    return np.array([steer, pedal, brake], dtype=np.float32)


# ── Tail trim ──────────────────────────────────────────────────────────────────

def trim_tail(states: List[dict], trim_seconds: float) -> List[dict]:
    """
    Remove the last `trim_seconds` of an episode using timestamp_ac.

    If timestamp_ac is absent, falls back to trimming a fixed count
    assuming 33 Hz physics.
    """
    if not states:
        return states

    if "timestamp_ac" in states[0]:
        cutoff = states[-1]["timestamp_ac"] - trim_seconds
        trimmed = [s for s in states if s["timestamp_ac"] <= cutoff]
        n_removed = len(states) - len(trimmed)
        logger.debug(f"  Tail trim: removed {n_removed} frames ({trim_seconds:.1f} s) via timestamp_ac")
        return trimmed
    else:
        # Fallback: assume 33 Hz
        n_remove = int(trim_seconds * 33)
        trimmed  = states[:-n_remove] if len(states) > n_remove else []
        logger.warning(
            f"  timestamp_ac missing — trimming last {n_remove} frames "
            f"(assumed 33 Hz). Provide timestamp_ac in states for accuracy."
        )
        return trimmed


# ── Main processing ────────────────────────────────────────────────────────────

def process_files(
    pkl_paths:     List[str],
    ref_lap:       ReferenceLap,
    brake_map:     BrakeMap,
    steer_max:     float,
    trim_seconds:  float = 7.0,
    skip_outlap:   bool  = False,
    min_speed_ms:  float = 5.0,
):
    """
    Process list of pkl files → stacked arrays + episode metadata.

    Returns
    -------
    obs, actions, episode_ends, lap_counts, out_of_track, speeds
    """
    channel_scales = np.array(
        [OBS_CHANNEL_SCALES[ch] for ch in OBS_ENABLED_CHANNELS], dtype=np.float32
    )

    all_obs, all_act, all_laps, all_oot, all_spd = [], [], [], [], []
    episode_ends: List[int] = []
    step_cursor = 0

    for pkl_path in sorted(pkl_paths):
        logger.info(f"Processing {Path(pkl_path).name} ...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        states = data["states"]
        raw_len = len(states)

        # ── Trim crash tail ────────────────────────────────────────────────
        if trim_seconds > 0:
            states = trim_tail(states, trim_seconds)

        logger.info(f"  {raw_len} raw frames → {len(states)} after {trim_seconds:.1f}s tail trim")

        if not states:
            logger.warning("  Episode too short after trim — skipping.")
            continue

        history:           List[dict] = []
        ep_steps           = 0
        skipped_outlap     = 0
        skipped_lowspeed   = 0
        skipped_badfields  = 0

        for state in states:
            # Out-lap filter (human episodes usually start with LapCount==0
            # if the teleport puts you on the out-lap)
            if skip_outlap and int(state.get("LapCount", 0)) == 0:
                history.append(state)
                skipped_outlap += 1
                continue

            # Low-speed filter (erratic steer at near-standstill)
            if min_speed_ms > 0.0 and float(state.get("speed", 0.0)) < min_speed_ms:
                history.append(state)
                skipped_lowspeed += 1
                continue

            obs = build_obs(state, history, ref_lap, channel_scales)
            if obs is None:
                skipped_badfields += 1
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
                f"  ✅ {ep_steps} steps added "
                f"(skipped: outlap={skipped_outlap} "
                f"lowspeed={skipped_lowspeed} "
                f"badfields={skipped_badfields})"
            )
        else:
            logger.warning("  ⚠️  No valid steps after filtering — skipping episode.")

    if not all_obs:
        raise ValueError("No valid steps found across all pkl files.")

    return (
        np.stack(all_obs, axis=0).astype(np.float32),
        np.stack(all_act, axis=0).astype(np.float32),
        np.array(episode_ends, dtype=np.int64),
        np.array(all_laps,     dtype=np.int32),
        np.array(all_oot,      dtype=bool),
        np.array(all_spd,      dtype=np.float32),
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess human-collected AC pkls → (obs, actions) .npz",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  python human_data/preprocess_human.py --track monza --car lotus_exos_125\n"
            "  python human_data/preprocess_human.py --track monza --car lotus_exos_125\n"
            "      --steer-max 318 --trim-tail-seconds 5\n"
        ),
    )
    # ── Required ────────────────────────────────────────────────────────────
    parser.add_argument("--track", required=True, help="Track name, e.g. monza")
    parser.add_argument("--car",   required=True, help="Car name,   e.g. lotus_exos_125")

    # ── Tail trim ────────────────────────────────────────────────────────────
    parser.add_argument("--trim-tail-seconds", type=float, default=7.0, metavar="S",
                        help="Seconds to strip from the end of each episode (default: 7.0)")

    # ── Config overrides ─────────────────────────────────────────────────────
    parser.add_argument("--brake-map",  type=str, default=None,
                        help="Path to brake_map.csv (auto-resolved if omitted)")
    parser.add_argument("--steer-map",  type=str, default=None,
                        help="Path to steer_map.csv (auto-resolved if omitted)")
    parser.add_argument("--steer-max",  type=float, default=None,
                        help="Max steer angle in degrees (overrides steer_map.csv)")
    parser.add_argument("--ref-lap",    type=str, default=None,
                        help="Path to reference lap CSV (auto-resolved if omitted)")

    # ── Data / output ─────────────────────────────────────────────────────────
    parser.add_argument("--input-dir",  type=str, default=None,
                        help="Directory of pkl files (default: human_data/raw/{track}_{car})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save .npz/.json (default: human_data/processed)")
    parser.add_argument("--data", nargs="+", default=None, metavar="PKL",
                        help="Explicit pkl file list (overrides --input-dir)")

    # ── Filtering ─────────────────────────────────────────────────────────────
    parser.add_argument("--min-speed",    type=float, default=5.0, metavar="M/S",
                        help="Skip steps below this speed in m/s (default: 5.0)")
    parser.add_argument("--skip-outlap",  action="store_true", default=False,
                        help="Skip LapCount==0 steps (out-lap). Off by default for human data.")

    args = parser.parse_args()

    # ── Resolve input pkls ────────────────────────────────────────────────────
    if args.data:
        pkl_paths = [str(Path(p).resolve()) for p in args.data]
    else:
        input_dir = Path(args.input_dir) if args.input_dir else (
            HUMAN_DATA / "raw" / f"{args.track}_{args.car}"
        )
        pkl_paths = sorted(str(p) for p in input_dir.glob("*.pkl"))
        if not pkl_paths:
            raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    logger.info(f"Found {len(pkl_paths)} episode pkl file(s)")
    for p in pkl_paths:
        logger.info(f"  {p}")

    # ── Resolve config files ───────────────────────────────────────────────────
    brake_map_path, steer_map_path, ref_lap_path = _resolve_config_files(
        track=args.track, car=args.car,
        brake_map_override=args.brake_map,
        steer_map_override=args.steer_map,
        ref_lap_override=args.ref_lap,
    )
    logger.info(f"Ref lap  : {ref_lap_path}")
    logger.info(f"Brake map: {brake_map_path}")
    logger.info(f"Steer map: {steer_map_path}")

    # ── Steer max ─────────────────────────────────────────────────────────────
    if args.steer_max is not None:
        steer_max = args.steer_max
        logger.info(f"steer_max = {steer_max:.1f} deg  (from --steer-max)")
    else:
        import pandas as pd
        steer_max = float(pd.read_csv(steer_map_path).values[1, 1])
        logger.info(f"steer_max = {steer_max:.1f} deg  (from steer map)")

    # ── Load config objects ────────────────────────────────────────────────────
    brake_map = BrakeMap.load(brake_map_path)
    ref_lap   = ReferenceLap(ref_lap_path, use_target_speed=False)

    # ── Process ───────────────────────────────────────────────────────────────
    logger.info(
        f"Processing | trim_tail={args.trim_tail_seconds}s | "
        f"min_speed={args.min_speed} m/s | "
        f"skip_outlap={args.skip_outlap}"
    )
    obs, actions, episode_ends, lap_counts, out_of_track, speeds = process_files(
        pkl_paths      = pkl_paths,
        ref_lap        = ref_lap,
        brake_map      = brake_map,
        steer_max      = steer_max,
        trim_seconds   = args.trim_tail_seconds,
        skip_outlap    = args.skip_outlap,
        min_speed_ms   = args.min_speed,
    )

    # ── Sanity check ──────────────────────────────────────────────────────────
    assert obs.shape[1] == OBS_DIM, (
        f"Obs dim mismatch: got {obs.shape[1]}, expected {OBS_DIM}. "
        f"Check constants match ac_env.py."
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info(f"Track / Car       : {args.track} / {args.car}")
    logger.info(f"Episodes          : {len(episode_ends)}")
    logger.info(f"Total steps       : {len(obs)}")
    logger.info(f"obs shape         : {obs.shape}")
    logger.info(f"actions shape     : {actions.shape}")
    logger.info(f"Speed range       : {speeds.min():.1f} – {speeds.max():.1f} m/s")
    logger.info(f"Mean speed        : {speeds.mean():.1f} m/s")
    logger.info(f"Out-of-track steps: {out_of_track.sum()} ({100*out_of_track.mean():.1f}%)")

    # Steer distribution
    steers = actions[:, 0]
    logger.info(f"Steer range       : {steers.min():.3f} – {steers.max():.3f} (SAC space [-1,1])")
    logger.info(f"Steer mean/std    : {steers.mean():.4f} / {steers.std():.4f}")
    near_zero = np.abs(steers) < 0.05
    logger.info(f"Near-zero steer   : {near_zero.sum()} steps ({100*near_zero.mean():.1f}%)")
    logger.info("─" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else HUMAN_DATA / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = f"{args.track}_{args.car}_mlp"
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
    logger.info(f"Saved npz : {npz_path}")

    meta = {
        "track":               args.track,
        "car":                 args.car,
        "model_type":          "mlp",
        "obs_dim":             int(obs.shape[1]),
        "action_dim":          int(actions.shape[1]),
        "past_actions_window": PAST_ACTIONS_WINDOW,
        "n_steps":             int(len(obs)),
        "n_episodes":          int(len(episode_ends)),
        "steer_max":           float(steer_max),
        "steer_obs_scale":     float(OBS_CHANNEL_SCALES["steerAngle"]),
        "trim_tail_seconds":   args.trim_tail_seconds,
        "min_speed_ms":        args.min_speed,
        "skip_outlap":         args.skip_outlap,
        "ref_lap":             ref_lap_path,
        "brake_map":           brake_map_path,
        "steer_map":           steer_map_path,
        "source_pkl_files":    pkl_paths,
        "created":             datetime.utcnow().isoformat(),
    }
    with open(str(json_path), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved json: {json_path}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
