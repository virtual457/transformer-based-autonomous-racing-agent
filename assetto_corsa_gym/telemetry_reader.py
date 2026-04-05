"""
Telemetry Reader — Live AC telemetry display in terminal.
Shows key channels in real-time as they stream from the simulator at 25Hz.

Usage (with venv activated, AC running in any session):
    python telemetry_reader.py
    python telemetry_reader.py --channels speed RPM steerAngle accStatus brakeStatus
    python telemetry_reader.py --all      (print every channel)

Press Ctrl+C to stop.
"""

import sys
import os
import time
import argparse

sys.path.extend([os.path.abspath('./assetto_corsa_gym')])

from omegaconf import OmegaConf
import AssettoCorsaEnv.assettoCorsa as assettoCorsa

# ── Default channels to display ──────────────────────────────────────────────
DEFAULT_CHANNELS = [
    "speed",
    "RPM",
    "actualGear",
    "steerAngle",
    "accStatus",
    "brakeStatus",
    "NormalizedSplinePosition",   # track progress 0→1
    "numberOfTyresOut",           # 0=on track, >0=off track
    "LapCount",
    "local_velocity_x",
    "local_velocity_y",
    "angular_velocity_y",
    "accelX",
    "accelY",
    "SlipAngle_fl",
    "SlipAngle_fr",
    "SlipAngle_rl",
    "SlipAngle_rr",
]

def clear_line():
    sys.stdout.write("\033[F\033[K" )

def print_telemetry(state, channels, step, t_start):
    print(f"\n{'='*55}")

    for ch in channels:
        val = state.get(ch, "N/A")
        if isinstance(val, float):
            print(f"  {ch:<35s} {val:>10.4f}")
        else:
            print(f"  {ch:<35s} {str(val):>10s}")

    print(f"{'='*55}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS,
                        help="Channels to display")
    parser.add_argument("--all", action="store_true",
                        help="Display all available channels")
    parser.add_argument("--rate", type=int, default=5,
                        help="Print every N steps (default: 5 = ~5Hz display)")
    args = parser.parse_args()

    config = OmegaConf.load("config.yml")
    print("Connecting to Assetto Corsa (make sure AC is running in a session)...")
    client = assettoCorsa.make_client_only(config.AssettoCorsa)
    client.setup_connection()  # establishes UDP link and gets first state
    print("Connected! Streaming telemetry. Press Ctrl+C to stop.\n")

    # Get first state to discover all channels
    state = dict(client.state)

    if args.all:
        channels = sorted(state.keys())
        print(f"Available channels ({len(channels)} total):")
        for c in channels:
            print(f"  {c}")
        print()
    else:
        channels = args.channels

    step = 0
    t_start = time.time()

    try:
        while True:
            client.get_servers_input()
            state = dict(client.state)
            step += 1

            if step % args.rate == 0:
                # Move cursor up to overwrite previous block
                if step > args.rate:
                    lines = len(channels) + 3
                    sys.stdout.write(f"\033[{lines}A")
                print_telemetry(state, channels, step, t_start)

            # Reply to keep the connection alive
            client.controls.set_controls(steer=0, acc=0, brake=0)
            client.respond_to_server()

    except KeyboardInterrupt:
        print(f"\n\nStopped after {step} steps ({time.time()-t_start:.1f}s).")
        client.close()


if __name__ == "__main__":
    main()
