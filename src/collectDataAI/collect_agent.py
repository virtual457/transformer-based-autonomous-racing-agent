"""
collect_agent.py — Real-time AC data collection agent.

Connects to Assetto Corsa as an observer and records full telemetry + controls
each physics frame. Designed for capturing AI or policy driving for BC training.

── Manual mode (default) ────────────────────────────────────────────────────────
    Q  — start recording a new episode
    E  — stop recording and save the episode
    Ctrl+C — quit (saves any in-progress episode first)

── Loop mode (--loop) ───────────────────────────────────────────────────────────
    Automatically cycles through a list of tracks.
    For each track: launches AC, records N episodes (one lap each), saves, next track.
    Requires --manage-ac.

Output:
    collectDataAI/data/{car}/{track}/{session_timestamp}/episode_NNNN.parquet
    collectDataAI/data/{car}/{track}/{session_timestamp}/episode_NNNN_meta.json

Each row in the parquet file is one physics frame (25 Hz) and contains:
    - frame index
    - steer_deg   : raw steering angle in degrees (steerAngle, NOT normalised)
    - throttle    : raw throttle pedal [0, 1]  (accStatus)
    - brake       : raw brake pedal [0, 1]     (brakeStatus)
    - full telemetry  : speed_ms, last_ff, rpm, gear, nsp, lap_dist, gap,
                        pos_x/y, yaw, accel_x/y, vel_x/y, yaw_rate, sa_fl/fr/rl/rr
    - ray sensors : 11 ray-cast wall distances in metres (ray_r90 … ray_l90, max 200 m)

    All values are stored RAW. Normalisation happens in the preprocessor.

Usage:
    # Manual mode
    ..\\AssetoCorsa\\Scripts\\python.exe collectDataAI/collect_agent.py
    ..\\AssetoCorsa\\Scripts\\python.exe collectDataAI/collect_agent.py --manage-ac

    # Loop mode — all lappable tracks, 5 episodes each
    ..\\AssetoCorsa\\Scripts\\python.exe collectDataAI/collect_agent.py --loop --manage-ac

    # Loop mode — specific tracks, 10 episodes each
    ..\\AssetoCorsa\\Scripts\\python.exe collectDataAI/collect_agent.py --loop --manage-ac --tracks monza spa imola --episodes-per-track 10
"""

import sys
import os
import time
import msvcrt
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path

# ── Path bootstrap (mirrors watch_sensors.py) ─────────────────────────────────
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "assetto_corsa_gym"))
sys.path.insert(0, str(_root / "assetto_corsa_gym" / "assetto_corsa_gym"))
sys.path.insert(0, str(_root / "assetto_corsa_gym" / "algorithm" / "discor"))
sys.path.insert(0, str(_root / "gym"))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("collect_agent")

# ── ANSI helpers ──────────────────────────────────────────────────────────────
CLEAR  = "\033[2J\033[H"
BOLD   = "\033[1m"
RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
WHITE  = "\033[97m"

RAY_LABELS = ["ray_r90","ray_r67","ray_r45","ray_r22","ray_r11",
              "ray_fwd","ray_l11","ray_l22","ray_l45","ray_l67","ray_l90"]

# ── Track catalogue ───────────────────────────────────────────────────────────
# Each entry: (race_ini_track, config_track_layout, cfg_track_name)
#   race_ini_track     → TRACK field in race.ini  (ac_lifecycle.TARGET_TRACK)
#   config_track_layout→ CONFIG_TRACK in race.ini (ac_lifecycle.TARGET_LAYOUT), "" = single layout
#   cfg_track_name     → AssettoCorsa.track in config.yml (what make_ac_env expects)
#
# For multi-layout tracks the cfg name uses AC's hyphenated convention: "track-layout".
# Single-layout tracks use the bare folder name.

TRACK_CATALOGUE = [
    # (race_ini_track,  layout,  cfg_name)
    ("monza",           "",      "monza"),
    ("imola",           "",      "imola"),
    ("ks_silverstone",  "gp",    "ks_silverstone-gp"),
]

# Quick lookup by race_ini_track name
_CATALOGUE_BY_NAME = {entry[0]: entry for entry in TRACK_CATALOGUE}


def get_track_entry(track_name):
    """
    Return (race_ini_track, layout, cfg_name) for a track name.
    Accepts either the race_ini_track name or the cfg_name (hyphenated).
    Falls back gracefully if not in catalogue.
    """
    if track_name in _CATALOGUE_BY_NAME:
        return _CATALOGUE_BY_NAME[track_name]
    # Try cfg_name lookup
    for entry in TRACK_CATALOGUE:
        if entry[2] == track_name:
            return entry
    # Unknown track — no layout, use name as-is
    logger.warning(f"Track '{track_name}' not in catalogue — assuming single layout.")
    return (track_name, "", track_name)


def get_all_track_names():
    """Return list of all race_ini_track names from the catalogue."""
    return [entry[0] for entry in TRACK_CATALOGUE]


def _bar(val, lo, hi, width=20):
    frac = max(0.0, min(1.0, (val - lo) / (hi - lo))) if hi != lo else 0.0
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Live display ──────────────────────────────────────────────────────────────

def render(state, recording, ep_idx, ep_frames, session_episodes, session_frames):
    speed    = float(state.get("speed",                    0.0)) * 3.6
    steer    = float(state.get("steerAngle",               0.0))
    acc      = float(state.get("accStatus",                0.0))
    brake    = float(state.get("brakeStatus",              0.0))
    nsp      = float(state.get("NormalizedSplinePosition", 0.0))
    gear     = int(  state.get("actualGear",               0))
    lap_t    = float(state.get("currentTime",              0.0))
    last_lap = float(state.get("lastLap",                  0.0))
    lap_cnt  = int(  state.get("LapCount",                 0))

    if recording:
        rec_banner = (
            f"{RED}{BOLD}● REC{RESET}  "
            f"ep={ep_idx:04d}  frames={ep_frames:6,d}"
        )
    else:
        rec_banner = f"{DIM}○ IDLE — press Q to start recording{RESET}"

    sensors = state.get("sensors", None)
    if sensors is not None:
        min_wall = float(np.min(sensors))
        wall_c   = RED if min_wall < 5.0 else (YELLOW if min_wall < 15.0 else GREEN)
        wall_str = f"{wall_c}{min_wall:5.1f} m{RESET}"
    else:
        wall_str = f"{DIM}n/a{RESET}"

    lines = [
        f"{BOLD}{CYAN}══ Collect Agent ══{RESET}  {rec_banner}",
        "",
        f"  Speed      {GREEN}{speed:6.1f} km/h{RESET}   "
        f"Gear {gear}   Lap #{lap_cnt}   NSP {nsp:.4f}",
        f"  Steer      {steer:+7.1f}°  {_bar(steer, -450, 450)}",
        f"  Throttle   {acc:6.3f}       {_bar(acc, 0, 1)}",
        f"  Brake      {brake:6.3f}       {_bar(brake, 0, 1)}",
        f"  Lap time   {lap_t:6.2f} s    Last: {last_lap:.2f} s",
        f"  Wall dist  {wall_str}  (nearest ray)",
        "",
        f"  {BOLD}Session:{RESET}  {session_episodes} episodes saved   "
        f"{session_frames:,} frames total",
        "",
        f"  {GREEN}[Q]{RESET} Start   "
        f"{YELLOW}[E]{RESET} Stop & save   "
        f"{DIM}[Ctrl+C] Quit{RESET}",
    ]
    return "\n".join(lines)


# ── Data persistence ──────────────────────────────────────────────────────────

def _state_to_row(state, frame_idx):
    """
    Convert one expanded state dict to a flat dict for a DataFrame row.

    All values are stored RAW — no normalisation, no remapping.
    Normalisation happens later in the preprocessor.

    Field audit (source → column, unit):
        steerAngle          → steer_deg      degrees, raw AC value
        accStatus           → throttle       [0, 1], raw pedal position
        brakeStatus         → brake          [0, 1], raw pedal position
        speed               → speed_ms       m/s
        LastFF              → last_ff        force feedback, raw AC value
        RPM                 → rpm            rev/min
        actualGear          → gear           integer {0..6}
        NormalizedSplinePosition → nsp       [0, 1], fraction of lap
        LapDist             → lap_dist       metres from lap start
        currentTime         → lap_time_s     seconds, elapsed in current lap
        LapCount            → lap_count      integer, increments each lap
        gap                 → gap            metres, signed (+ = right of line)
        world_position_x    → pos_x          metres, world X
        world_position_y    → pos_y          metres, world Y
        yaw                 → yaw            radians [-π, π]
        accelX              → accel_x        G (lateral accel)
        accelY              → accel_y        G (longitudinal accel)
        local_velocity_x    → vel_x          m/s, longitudinal (car frame)
        local_velocity_y    → vel_y          m/s, lateral (car frame)
        angular_velocity_y  → yaw_rate       rad/s
        SlipAngle_fl/fr/rl/rr → sa_fl/fr/rl/rr  degrees
        out_of_track        → out_of_track   bool
        numberOfTyresOut    → tyres_out      integer {0..4}
        sensors[0..10]      → ray_*          metres, wall distance (max 200 m)
    """
    row = {
        "frame": frame_idx,
        # ── Driver inputs (raw AC values, no normalisation) ───────────────
        "steer_deg": float(state.get("steerAngle",  0.0)),   # degrees
        "throttle":  float(state.get("accStatus",   0.0)),   # [0, 1]
        "brake":     float(state.get("brakeStatus", 0.0)),   # [0, 1]
        # ── Speed / drivetrain ────────────────────────────────────────────
        "speed_ms":  float(state.get("speed",      0.0)),    # m/s
        "last_ff":   float(state.get("LastFF",     0.0)),    # force feedback
        "rpm":       float(state.get("RPM",        0.0)),    # rev/min
        "gear":      int(  state.get("actualGear", 0)),      # {0..6}
        # ── Track position ────────────────────────────────────────────────
        "nsp":        float(state.get("NormalizedSplinePosition", 0.0)),  # [0, 1]
        "lap_dist":   float(state.get("LapDist",     0.0)),  # metres
        "lap_time_s": float(state.get("currentTime", 0.0)),  # seconds
        "lap_count":  int(  state.get("LapCount",    0)),    # integer
        "gap":        float(state.get("gap",         0.0)),  # metres, signed
        # ── World pose ───────────────────────────────────────────────────
        "pos_x": float(state.get("world_position_x", 0.0)),  # metres
        "pos_y": float(state.get("world_position_y", 0.0)),  # metres
        "yaw":   float(state.get("yaw",              0.0)),  # radians [-π, π]
        # ── Dynamics ─────────────────────────────────────────────────────
        "accel_x":  float(state.get("accelX",             0.0)),  # G
        "accel_y":  float(state.get("accelY",             0.0)),  # G
        "vel_x":    float(state.get("local_velocity_x",   0.0)),  # m/s
        "vel_y":    float(state.get("local_velocity_y",   0.0)),  # m/s
        "yaw_rate": float(state.get("angular_velocity_y", 0.0)),  # rad/s
        # ── Tyre slip angles ──────────────────────────────────────────────
        "sa_fl": float(state.get("SlipAngle_fl", 0.0)),  # degrees
        "sa_fr": float(state.get("SlipAngle_fr", 0.0)),  # degrees
        "sa_rl": float(state.get("SlipAngle_rl", 0.0)),  # degrees
        "sa_rr": float(state.get("SlipAngle_rr", 0.0)),  # degrees
        # ── Track limits ─────────────────────────────────────────────────
        "out_of_track": bool(state.get("out_of_track",     False)),  # bool
        "tyres_out":    int( state.get("numberOfTyresOut", 0)),      # {0..4}
    }

    # 11 ray-cast wall sensors
    sensors = state.get("sensors", None)
    if sensors is not None:
        for lbl, v in zip(RAY_LABELS, sensors):
            row[lbl] = float(v)
    else:
        for lbl in RAY_LABELS:
            row[lbl] = float("nan")

    return row


def save_episode(frames, car, track, session_ts, ep_idx, output_root):
    """
    Persist one episode as a parquet file + JSON metadata.

    Parameters
    ----------
    frames      : list of expanded state dicts (one per physics frame)
    car         : car identifier string (e.g. 'ks_mazda_miata')
    track       : track identifier string (e.g. 'monza')
    session_ts  : session timestamp string (YYYYmmdd_HHMMSS)
    ep_idx      : episode number within this session
    output_root : folder that contains the 'data/' sub-tree

    Returns
    -------
    Path to the saved .parquet file.
    """
    folder = Path(output_root) / "data" / car / track / session_ts
    folder.mkdir(parents=True, exist_ok=True)

    rows = [_state_to_row(s, i) for i, s in enumerate(frames)]
    df   = pd.DataFrame(rows)

    parquet_path = folder / f"episode_{ep_idx:04d}.parquet"
    df.to_parquet(parquet_path, index=False)

    # Metadata
    lap_counts = df["lap_count"].values
    completed_laps = int(lap_counts[-1] - lap_counts[0]) if len(lap_counts) > 1 else 0
    meta = {
        "car":             car,
        "track":           track,
        "session":         session_ts,
        "episode":         ep_idx,
        "frames":          len(frames),
        "saved_at":        _ts(),
        "max_speed_kmh":   round(float(df["speed_ms"].max()) * 3.6, 1),
        "completed_laps":  completed_laps,
        "has_sensors":     not df["ray_fwd"].isna().all(),
    }
    meta_path = folder / f"episode_{ep_idx:04d}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return parquet_path


# ── Environment helpers ───────────────────────────────────────────────────────

def drain_buffer(ac_env, drain_timeout_s=0.005):
    """Discard stale UDP packets that queued during make_ac_env init."""
    from AssettoCorsaEnv.ac_client import MAX_MSG_SIZE
    sock = ac_env.client.socket
    if sock is None:
        return 0
    orig = sock.gettimeout()
    sock.settimeout(drain_timeout_s)
    n = 0
    try:
        while True:
            sock.recvfrom(MAX_MSG_SIZE)
            n += 1
    except Exception:
        pass
    sock.settimeout(orig)
    return n


# ── Loop mode ─────────────────────────────────────────────────────────────────

def _run_loop(tracks, episodes_per_track, max_frames_per_ep, config_path):
    """
    Automatically cycle through a list of tracks and collect episodes on each.

    For each track:
      1. Patch ac_lifecycle.TARGET_TRACK and cfg.AssettoCorsa.track
      2. Call full_cycle() to (re)launch AC on that track
      3. Connect via make_ac_env
      4. Record `episodes_per_track` episodes, ending each episode when:
           - lap_count increases (lap completed), OR
           - max_frames_per_ep frames recorded (crash/timeout fallback)
      5. Save all episodes, close env, move to next track

    Prints a full summary at the end.
    """
    import ac_lifecycle as _acl
    from ac_lifecycle import full_cycle
    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    from omegaconf import OmegaConf

    cfg          = OmegaConf.load(config_path)
    car          = _acl.TARGET_CAR
    session_ts   = _ts()
    output_root  = Path(__file__).parent
    work_dir     = str(_root / "outputs" / "collect_agent")
    os.makedirs(work_dir, exist_ok=True)

    total_episodes = 0
    total_frames   = 0
    summary        = []   # list of (track, episodes, frames)

    print(f"\n{'═'*60}")
    print(f"  LOOP MODE — {len(tracks)} tracks × {episodes_per_track} episodes each")
    print(f"  Tracks: {', '.join(tracks)}")
    print(f"  Max frames/episode: {max_frames_per_ep:,}  (~{max_frames_per_ep/25:.0f}s)")
    print(f"{'═'*60}\n")

    for track_idx, track_key in enumerate(tracks):
        race_track, layout, cfg_track = get_track_entry(track_key)

        print(f"\n{'─'*60}")
        print(f"  Track {track_idx+1}/{len(tracks)}: {race_track}"
              + (f"  layout={layout}" if layout else ""))
        print(f"{'─'*60}")

        # ── 1. Patch track in ac_lifecycle and config ─────────────────────────
        _acl.TARGET_TRACK  = race_track
        _acl.TARGET_LAYOUT = layout
        OmegaConf.update(cfg, "AssettoCorsa.track", cfg_track)

        # ── 2. Launch AC on this track ────────────────────────────────────────
        try:
            logger.info(f"Launching AC: track={race_track}  layout={layout!r}  cfg={cfg_track}")
            full_cycle(max_retries=3)
            logger.info("AC is live.")
        except Exception as e:
            logger.error(f"full_cycle failed for '{race_track}': {e} — skipping.")
            summary.append((race_track, 0, 0, "LAUNCH FAILED"))
            continue

        # ── 3. Connect env ────────────────────────────────────────────────────
        # Disable the 120s episode timeout so the env never auto-terminates
        # while we're waiting for manual Q/E input.
        OmegaConf.update(cfg, "AssettoCorsa.max_episode_py_time", 99999.0)
        OmegaConf.update(cfg, "AssettoCorsa.max_laps_number", None)
        try:
            ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
            ac_env.reset()
            n_drained = drain_buffer(ac_env)
            logger.info(f"Connected. Drained {n_drained} stale packets.")
        except Exception as e:
            logger.error(f"make_ac_env failed for '{cfg_track}': {e} — skipping.")
            summary.append((race_track, 0, 0, "CONNECT FAILED"))
            continue

        # ── 4. Collect episodes (manual Q/E control per episode) ─────────────
        track_episodes = 0
        track_frames   = 0

        for ep in range(episodes_per_track):
            ep_frames_buf = []
            recording     = False

            print(f"\n  [{race_track}]  episode {ep+1}/{episodes_per_track}")
            print(f"  {GREEN}[Q]{RESET} Start recording   {YELLOW}[E]{RESET} Stop & save   {RED}[S]{RESET} Skip track\n")

            try:
                while True:
                    raw      = ac_env.client.step_sim()
                    state, _ = ac_env.expand_state(raw)

                    # Keyboard
                    if msvcrt.kbhit():
                        key = msvcrt.getch().lower()
                        if key == b'q' and not recording:
                            recording = True
                            print(f"  {RED}● REC started{RESET}")
                        elif key == b'e' and recording:
                            recording = False
                            break   # save and move on
                        elif key == b's':
                            recording = False
                            ep_frames_buf = []
                            print(f"  Skipping track.")
                            break

                    if recording:
                        ep_frames_buf.append(dict(state))

                    # Live one-liner status
                    speed = float(state.get("speed", 0)) * 3.6
                    nsp   = float(state.get("NormalizedSplinePosition", 0))
                    rec_indicator = f"{RED}● REC {len(ep_frames_buf):5d} frames{RESET}" if recording else f"{DIM}○ idle{RESET}"
                    sys.stdout.write(
                        f"\r  {rec_indicator}   {speed:5.1f} km/h   NSP {nsp:.3f}   "
                    )
                    sys.stdout.flush()

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.warning(f"Frame read error on {race_track} ep {ep}: {e} — episode ended.")

            # Save episode
            print()  # newline after inline status
            n_frames = len(ep_frames_buf)
            if n_frames > 0:
                path = save_episode(
                    ep_frames_buf, car, race_track,
                    session_ts, total_episodes, output_root,
                )
                track_episodes += 1
                track_frames   += n_frames
                total_episodes += 1
                total_frames   += n_frames
                print(f"  Saved: {n_frames} frames → {path.name}")
            else:
                print(f"  Nothing recorded — skipping save.")

        summary.append((race_track, track_episodes, track_frames, "OK"))
        logger.info(f"Track '{race_track}' done: {track_episodes} episodes, {track_frames:,} frames")

        # ── 5. Confirm before killing AC for next track ───────────────────────
        next_idx = tracks.index(track_key) + 1 if track_key in tracks else track_idx + 1
        if next_idx < len(tracks):
            next_race, next_layout, _ = get_track_entry(tracks[next_idx])
            next_label = next_race + (f" ({next_layout})" if next_layout else "")
            print(f"\n  {BOLD}Track done:{RESET} {race_track}  |  "
                  f"{track_episodes} episodes  {track_frames:,} frames")
            print(f"  Next track: {CYAN}{next_label}{RESET}")
            print(f"  {GREEN}[Enter]{RESET} Launch next track   "
                  f"{YELLOW}[S + Enter]{RESET} Skip next   "
                  f"{RED}[Q + Enter]{RESET} Quit loop\n")

            # Drain any leftover keyboard input before reading
            while msvcrt.kbhit():
                msvcrt.getch()

            choice = input("  > ").strip().lower()
            if choice == 'q':
                print("  Quitting loop.")
                try:
                    ac_env.close()
                except Exception:
                    pass
                break
            elif choice == 's':
                print(f"  Skipping {next_label}.")
                # Mark the skipped track in summary
                _, _, skip_cfg = get_track_entry(tracks[next_idx])
                summary.append((next_race, 0, 0, "SKIPPED"))
                tracks = tracks[:next_idx] + tracks[next_idx+1:]  # remove from list

        # Close env — AC will be killed by full_cycle() on next iteration
        try:
            ac_env.close()
        except Exception:
            pass

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  LOOP COMPLETE — Session: {session_ts}")
    print(f"{'═'*60}")
    print(f"  {'Track':<30} {'Episodes':>9} {'Frames':>10}  Status")
    print(f"  {'─'*30} {'─'*9} {'─'*10}  {'─'*14}")
    for track, eps, frms, status in summary:
        print(f"  {track:<30} {eps:>9} {frms:>10,}  {status}")
    print(f"  {'─'*30} {'─'*9} {'─'*10}")
    print(f"  {'TOTAL':<30} {total_episodes:>9} {total_frames:>10,}")
    print(f"{'═'*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AC data collection agent — Q to record, E to save"
    )
    parser.add_argument(
        "--manage-ac", action="store_true",
        help="Auto-launch Assetto Corsa via Content Manager before collecting",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(_root / "assetto_corsa_gym" / "config.yml"),
        help="Path to config.yml",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip vJoy / AC connection preflight checks",
    )
    # ── Loop mode args ────────────────────────────────────────────────────────
    parser.add_argument(
        "--loop", action="store_true",
        help="Auto-loop through tracks (requires --manage-ac)",
    )
    parser.add_argument(
        "--tracks", nargs="+", default=None, metavar="TRACK",
        help="Tracks to collect from in loop mode (default: all installed lappable tracks). "
             "Example: --tracks monza spa imola",
    )
    parser.add_argument(
        "--episodes-per-track", type=int, default=5, metavar="N",
        help="Number of episodes to collect per track in loop mode (default: 5)",
    )
    parser.add_argument(
        "--max-frames-per-episode", type=int, default=3000, metavar="N",
        help="Max frames per episode before auto-ending (default: 3000 = ~2 min at 25 Hz)",
    )
    args = parser.parse_args()

    if args.loop and not args.manage_ac:
        parser.error("--loop requires --manage-ac (script must launch AC per track)")

    # ── Loop mode ─────────────────────────────────────────────────────────────
    if args.loop:
        tracks = args.tracks if args.tracks else get_all_track_names()
        _run_loop(
            tracks               = tracks,
            episodes_per_track   = args.episodes_per_track,
            max_frames_per_ep    = args.max_frames_per_episode,
            config_path          = args.config,
        )
        return

    # ── 1. Launch AC if requested ─────────────────────────────────────────────
    if args.manage_ac:
        from ac_lifecycle import full_cycle
        logger.info("--manage-ac: launching Assetto Corsa via Content Manager...")
        full_cycle(max_retries=3)
        logger.info("AC is live and on track.")
    elif not args.skip_preflight:
        try:
            from preflight import run_preflight
            run_preflight(warn_only=True)
        except Exception as e:
            logger.warning(f"Preflight: {e}")

    # ── 2. Config & car/track names ───────────────────────────────────────────
    cfg = OmegaConf.load(args.config)

    try:
        from ac_lifecycle import TARGET_CAR, TARGET_TRACK
        car   = TARGET_CAR
        track = TARGET_TRACK
    except (ImportError, AttributeError):
        car   = "unknown_car"
        track = "unknown_track"
        logger.warning("Could not read car/track from ac_lifecycle — using 'unknown'")

    logger.info(f"Car: {car}   Track: {track}")

    # ── 3. Connect to AC ──────────────────────────────────────────────────────
    work_dir = str(_root / "outputs" / "collect_agent")
    os.makedirs(work_dir, exist_ok=True)

    logger.info("Connecting to Assetto Corsa... (AC must be running with plugin loaded)")
    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    logger.info(f"Connected. obs_dim={ac_env.state_dim}")

    ac_env.reset()

    logger.info("Draining stale UDP buffer...")
    n_drained = drain_buffer(ac_env)
    logger.info(f"Drained {n_drained} stale packets. Now live at 25 Hz.")

    # ── 4. Session state ──────────────────────────────────────────────────────
    session_ts       = _ts()
    output_root      = Path(__file__).parent
    recording        = False
    ep_idx           = 0
    ep_frames_buf    = []   # list[dict] — expanded state dicts for current episode
    session_episodes = 0
    session_frames   = 0

    print(f"\n  Session: {session_ts}")
    print(f"  Output:  {output_root / 'data' / car / track / session_ts}\n")
    logger.info("Ready. Q=start  E=stop&save  Ctrl+C=quit")

    # ── 5. Main recording loop ────────────────────────────────────────────────
    try:
        while True:
            # Read one physics frame from AC (~40 ms cadence)
            raw            = ac_env.client.step_sim()
            state, _       = ac_env.expand_state(raw)

            # Non-blocking keyboard check (Windows msvcrt)
            if msvcrt.kbhit():
                key = msvcrt.getch().lower()

                if key == b"q":
                    if not recording:
                        recording     = True
                        ep_frames_buf = []
                        logger.info(f"Recording started — episode {ep_idx:04d}")
                    else:
                        logger.info("Already recording (press E to stop)")

                elif key == b"e":
                    if recording:
                        recording = False
                        n_frames  = len(ep_frames_buf)
                        if n_frames > 0:
                            path = save_episode(
                                ep_frames_buf, car, track,
                                session_ts, ep_idx, output_root,
                            )
                            session_episodes += 1
                            session_frames   += n_frames
                            logger.info(
                                f"Saved episode {ep_idx:04d}: "
                                f"{n_frames} frames → {path.name}"
                            )
                            ep_idx += 1
                        else:
                            logger.warning("Empty episode — nothing to save.")
                        ep_frames_buf = []
                    else:
                        logger.info("Not currently recording (press Q to start)")

            # Accumulate frame
            if recording:
                ep_frames_buf.append(dict(state))

            # Refresh live display
            panel = render(
                state, recording, ep_idx,
                len(ep_frames_buf), session_episodes, session_frames,
            )
            sys.stdout.write(CLEAR + panel + "\n")
            sys.stdout.flush()

    except KeyboardInterrupt:
        # Auto-save any in-progress episode on Ctrl+C
        if recording and ep_frames_buf:
            path = save_episode(
                ep_frames_buf, car, track,
                session_ts, ep_idx, output_root,
            )
            session_frames   += len(ep_frames_buf)
            session_episodes += 1
            logger.info(
                f"Ctrl+C — auto-saved episode {ep_idx:04d}: "
                f"{len(ep_frames_buf)} frames → {path.name}"
            )

        print(
            f"\n  Session complete: {session_episodes} episodes, "
            f"{session_frames:,} frames total."
        )

    finally:
        try:
            ac_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
