"""
collect_human.py — Passive human-drive data collector for Assetto Corsa.

Listens to the AC physics stream while a human drives.  Recording starts only
when ALL of the following are true simultaneously:
  1. numberOfTyresOut == 0   (all four tyres on track)
  2. accStatus >= threshold  (driver applies throttle)

Recording ends when numberOfTyresOut >= oot_threshold (car goes off track).
The episode is saved as a .pkl that preprocess_bc.py can read directly.

State machine:
    WAITING ──(tyresOut==0 AND throttle)──► RECORDING ──(tyresOut >= N)──► SAVE
       ▲                                                                       │
       └───────────────────────────────────────────────────────────────────────┘

Usage:
    .\\AssetoCorsa\\Scripts\\python.exe human_data/collect_human.py \\
        --track monza --car ks_mazda_miata

Options:
    --output-dir DIR          Override save location (default: human_data/raw/{track}_{car})
    --oot-threshold N         Tyres off track to end episode (default: 4)
    --throttle-threshold F    Minimum accStatus to arm recording (default: 0.05)
    --min-steps N             Discard episodes shorter than N steps (default: 50)
    --skip-preflight          Skip vJoy / AC preflight checks

Output:
    human_data/raw/{track}_{car}/
        {timestamp}_{track}_{car}_{episode:03d}.pkl

Each .pkl format: {"states": [...], "track": ..., "car": ..., "timestamp": ..., "n_steps": N}
Feed directly into preprocess_bc.py:
    .\\AssetoCorsa\\Scripts\\python.exe gym/preprocess_bc.py \\
        --data human_data/raw/monza_ks_mazda_miata/*.pkl \\
        --output-dir data/processed/
"""

import sys
import os
import time
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path

# ── Path bootstrap (mirrors tests/watch_sensors.py) ──────────────────────────
_root = Path(__file__).resolve().parent.parent
for _p in [
    str(_root / "assetto_corsa_gym"),
    str(_root / "assetto_corsa_gym" / "assetto_corsa_gym"),
    str(_root / "assetto_corsa_gym" / "algorithm" / "discor"),
    str(_root / "gym"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("collect_human")

# ── ANSI helpers ──────────────────────────────────────────────────────────────
CLEAR  = "\033[2J\033[H"
BOLD   = "\033[1m"
RESET  = "\033[0m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
WHITE  = "\033[97m"

# ── State machine labels ───────────────────────────────────────────────────────
WAITING   = "WAITING"
RECORDING = "RECORDING"


# ── Display ───────────────────────────────────────────────────────────────────

def _bar(val, lo, hi, width=24):
    frac = max(0.0, min(1.0, (val - lo) / (hi - lo))) if hi != lo else 0.0
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def render_waiting(state, tyres_ok, throttle_ok, episodes_saved,
                   throttle_threshold):
    tyres_out = int(state.get("numberOfTyresOut", 0))
    throttle  = float(state.get("accStatus", 0.0))
    speed_kh  = float(state.get("speed", 0.0)) * 3.6
    nsp       = float(state.get("NormalizedSplinePosition", 0.0))
    lap_cnt   = int(state.get("LapCount", 0))

    lines = [
        f"{BOLD}{CYAN}══ Human Data Collector  [WAITING]  saved={episodes_saved} ══{RESET}",
        "",
        f"  Speed      {WHITE}{speed_kh:6.1f} km/h{RESET}",
        f"  Lap        {WHITE}{lap_cnt}{RESET}   NSP={nsp:.4f}",
        "",
        f"  Tyres out  {GREEN if tyres_ok  else RED}{tyres_out}/4{RESET}",
        f"  Throttle   {GREEN if throttle_ok else DIM}{throttle:.3f}{RESET}   "
        f"(threshold={throttle_threshold})",
        "",
        "  Waiting for:",
        f"    {'✅' if tyres_ok     else '⏳'} All tyres on track  (numberOfTyresOut == 0)",
        f"    {'✅' if throttle_ok  else '⏳'} Throttle applied    (accStatus >= {throttle_threshold})",
        "",
        f"{DIM}Ctrl+C to quit{RESET}",
    ]
    return "\n".join(lines)


def render_recording(state, step, episode_num, oot_threshold):
    speed_kh  = float(state.get("speed", 0.0)) * 3.6
    lap_dist  = float(state.get("LapDist", 0.0))
    lap_cnt   = int(state.get("LapCount", 0))
    tyres_out = int(state.get("numberOfTyresOut", 0))
    steer     = float(state.get("steerAngle", 0.0))
    throttle  = float(state.get("accStatus", 0.0))
    brake     = float(state.get("brakeStatus", 0.0))

    lines = [
        f"{BOLD}{RED}● REC{RESET}  "
        f"{BOLD}{CYAN}Episode #{episode_num}   step={step:5d}{RESET}",
        "",
        f"  Speed      {WHITE}{speed_kh:6.1f} km/h{RESET}   {_bar(speed_kh, 0, 250)}",
        f"  LapDist    {WHITE}{lap_dist:8.1f} m{RESET}   Lap#{lap_cnt}",
        f"  Tyres out  {RED if tyres_out > 0 else GREEN}{tyres_out}/4{RESET}"
        f"   (end at {oot_threshold})",
        "",
        f"  Steer      {YELLOW}{steer:+8.1f} deg{RESET}   {_bar(steer, -450, 450)}",
        f"  Throttle   {GREEN}{throttle:.3f}{RESET}   {_bar(throttle, 0, 1)}",
        f"  Brake      {RED if brake > 0.05 else DIM}{brake:.3f}{RESET}   "
        f"{_bar(brake, 0, 1)}",
        "",
        f"{DIM}Ctrl+C to save current episode and quit{RESET}",
    ]
    return "\n".join(lines)


# ── UDP drain ─────────────────────────────────────────────────────────────────

def drain_udp_buffer(ac_env, timeout_s=0.005):
    """Discard packets that piled up while make_ac_env() was loading assets."""
    sock = ac_env.client.socket
    if sock is None:
        return 0
    orig = sock.gettimeout()
    sock.settimeout(timeout_s)
    from AssettoCorsaEnv.ac_client import MAX_MSG_SIZE
    n = 0
    try:
        while True:
            sock.recvfrom(MAX_MSG_SIZE)
            n += 1
    except Exception:
        pass
    sock.settimeout(orig)
    return n


# ── Save ──────────────────────────────────────────────────────────────────────

def save_episode(states, track, car, episode_num, output_dir):
    """
    Save a list of state dicts as a .pkl file.

    Format matches what preprocess_bc.py expects:
        data["states"]  — list of state dicts, one per physics tick
    """
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{track}_{car}_{episode_num:03d}.pkl"
    path  = Path(output_dir) / fname
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "states":    states,
        "track":     track,
        "car":       car,
        "timestamp": ts,
        "n_steps":   len(states),
    }
    with open(str(path), "wb") as f:
        pickle.dump(payload, f)
    return str(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Passive human-drive data collector for AC.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  .\\AssetoCorsa\\Scripts\\python.exe human_data/collect_human.py\n"
            "      --track monza --car ks_mazda_miata\n"
        ),
    )
    parser.add_argument("--track",  required=True,
                        help="Track name, e.g. monza")
    parser.add_argument("--car",    required=True,
                        help="Car name, e.g. ks_mazda_miata")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Save directory (default: human_data/raw/{track}_{car})")
    parser.add_argument("--oot-threshold", type=int, default=4, metavar="N",
                        help="Tyres off track that end the episode (default: 4)")
    parser.add_argument("--throttle-threshold", type=float, default=0.05,
                        help="Min accStatus to arm recording (default: 0.05)")
    parser.add_argument("--min-steps", type=int, default=50,
                        help="Discard episodes shorter than N steps (default: 50)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip vJoy / AC preflight checks")
    parser.add_argument("--config", type=str,
                        default=str(_root / "assetto_corsa_gym" / "config.yml"),
                        help="Path to config.yml")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            _root / "human_data" / "raw" / f"{args.track}_{args.car}"
        )

    # ── Preflight ──────────────────────────────────────────────────────────
    if not args.skip_preflight:
        try:
            from preflight import run_preflight
            run_preflight(warn_only=True)
        except Exception as e:
            print(f"Preflight warning: {e}")

    # ── Build ACEnv ────────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    from AssettoCorsaEnv.assettoCorsa import make_ac_env

    cfg      = OmegaConf.load(args.config)
    # Override config with CLI args so make_ac_env validates against the right car/track
    cfg.car   = args.car
    cfg.track = args.track
    work_dir = str(_root / "outputs" / "collect_human")
    os.makedirs(work_dir, exist_ok=True)

    print("Connecting to Assetto Corsa ...")
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    print(f"Connected. obs_dim={ac_env.state_dim}")
    ac_env.reset()

    print("Draining stale UDP buffer ...")
    n_drained = drain_udp_buffer(ac_env)
    print(f"Drained {n_drained} stale packets. Now live.\n")

    print(f"Output dir       : {args.output_dir}")
    print(f"OOT threshold    : {args.oot_threshold} tyres")
    print(f"Throttle arm     : accStatus >= {args.throttle_threshold}")
    print(f"Min steps to save: {args.min_steps}")
    print("\nReady — put the car fully on track, then apply throttle to start.\n")

    # ── State machine ──────────────────────────────────────────────────────
    mode            = WAITING
    current_episode = []
    episode_count   = 0     # number of episodes saved so far

    def _flush_current(reason=""):
        """Save current_episode if long enough, print result."""
        nonlocal episode_count, current_episode
        n = len(current_episode)
        if n >= args.min_steps:
            path = save_episode(
                current_episode, args.track, args.car,
                episode_count + 1, args.output_dir,
            )
            episode_count += 1
            print(f"\n{GREEN}Saved: {path}  ({n} steps){RESET}")
        else:
            if n > 0:
                print(f"\n{YELLOW}Discarded short episode "
                      f"({n} steps < min {args.min_steps}){reason}{RESET}")
        current_episode = []

    try:
        while True:
            # ── Read next physics packet ───────────────────────────────────
            raw           = ac_env.client.step_sim()
            state, _      = ac_env.expand_state(raw)
            ac_env.state  = state

            tyres_out   = int(state.get("numberOfTyresOut", 0))
            throttle    = float(state.get("accStatus", 0.0))
            tyres_ok    = tyres_out == 0
            throttle_ok = throttle >= args.throttle_threshold

            # ── WAITING ───────────────────────────────────────────────────
            if mode == WAITING:
                panel = render_waiting(
                    state, tyres_ok, throttle_ok,
                    episode_count, args.throttle_threshold,
                )
                sys.stdout.write(CLEAR + panel + "\n")
                sys.stdout.flush()

                if tyres_ok and throttle_ok:
                    mode            = RECORDING
                    current_episode = [dict(state)]
                    sys.stdout.write(
                        f"\n{BOLD}{RED}● Recording started — Episode "
                        f"#{episode_count + 1}{RESET}\n"
                    )
                    sys.stdout.flush()

            # ── RECORDING ─────────────────────────────────────────────────
            elif mode == RECORDING:
                current_episode.append(dict(state))
                step = len(current_episode)

                panel = render_recording(
                    state, step, episode_count + 1, args.oot_threshold,
                )
                sys.stdout.write(CLEAR + panel + "\n")
                sys.stdout.flush()

                if tyres_out >= args.oot_threshold:
                    # ── Confirm save / discard ────────────────────────────
                    sys.stdout.write(CLEAR)
                    sys.stdout.flush()
                    print(f"{BOLD}Episode #{episode_count + 1} ended — {step} steps{RESET}\n")
                    if step < args.min_steps:
                        print(f"{YELLOW}Too short ({step} < {args.min_steps} min steps) — auto-discarding.{RESET}")
                    else:
                        try:
                            answer = input(
                                f"  Save this episode? [{GREEN}Y{RESET}/{RED}n{RESET}]: "
                            ).strip().lower()
                        except EOFError:
                            answer = "y"
                        if answer in ("", "y", "yes"):
                            _flush_current()
                        else:
                            print(f"{YELLOW}Discarded.{RESET}")
                            current_episode = []
                    mode = WAITING
                    print(f"\n{DIM}Returning to WAITING — get back on track and apply throttle...{RESET}\n")

    except KeyboardInterrupt:
        print(f"\n\nCtrl+C received.")
        _flush_current(reason=" — interrupted")
        print(f"\nDone. {episode_count} episode(s) saved to:\n  {args.output_dir}")

    finally:
        try:
            ac_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
