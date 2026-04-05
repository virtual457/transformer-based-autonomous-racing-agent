"""
Input Sender — Send manual controls to Assetto Corsa via vJoy.
Lets you test that the control pipeline works before training.

Usage (venv activated, AC running in any session):
    python input_sender.py

Controls (keyboard):
    W        — full throttle
    S        — full brake
    A        — steer left
    D        — steer right
    SPACE    — reset car position
    R        — release all (coast)
    Q        — quit

Steer/throttle/brake values are in range [-1, 1].
Displays live telemetry feedback (speed, gear, steer) after each command.
"""

import sys
import os
import time

sys.path.extend([os.path.abspath('./assetto_corsa_gym')])

from omegaconf import OmegaConf
import AssettoCorsaEnv.assettoCorsa as assettoCorsa

STEER_STEP = 0.2   # how much to increment steer per keypress
STEER_MAX  = 1.0

def print_state(state, steer, acc, brake):
    speed   = state.get("speed", 0)
    rpm     = state.get("RPM", 0)
    gear    = state.get("actualGear", 0)
    oot     = state.get("numberOfTyresOut", 0)
    lap_pos = state.get("NormalizedSplinePosition", 0)

    print(
        f"\r  Speed: {float(speed):6.1f} m/s  |  RPM: {float(rpm):6.0f}  |  Gear: {gear}  |"
        f"  OOT: {oot}  |  Pos: {float(lap_pos):.3f}"
        f"  ||  CMD → steer: {steer:+.2f}  acc: {acc:+.2f}  brake: {brake:+.2f}   ",
        end="", flush=True
    )

def send_and_read(client, steer, acc, brake):
    client.controls.set_controls(steer=steer, acc=acc, brake=brake)
    client.respond_to_server()
    state = client.step_sim()
    return state

def main():
    config = OmegaConf.load("config.yml")
    print("Connecting to Assetto Corsa...")
    client = assettoCorsa.make_client_only(config.AssettoCorsa)
    client.setup_connection()
    print("Connected!\n")
    print("Commands: W=throttle  S=brake  A=left  D=right  SPACE=reset  R=release  Q=quit")
    print("─" * 75)

    steer = 0.0
    acc   = 0.0
    brake = 0.0

    # Windows keyboard input (no Enter required)
    import msvcrt

    try:
        while True:
            # Non-blocking key check
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()

                if key == "q":
                    break
                elif key == "g":
                    # manual gear up (use this once at start to engage gear 1)
                    client.controls.set_controls(steer=steer, acc=acc, brake=brake,
                                                 enable_gear_shift=True, shift_up=True)
                    client.respond_to_server()
                    time.sleep(0.05)
                    client.controls.set_controls(steer=steer, acc=acc, brake=brake,
                                                 enable_gear_shift=True, shift_up=False)
                    client.respond_to_server()
                    print("\n  Shifted up!")
                    continue
                elif key == "w":
                    acc   = 1.0
                    brake = 0.0
                elif key == "s":
                    brake = 1.0
                    acc   = 0.0
                elif key == "a":
                    steer = max(-STEER_MAX, steer - STEER_STEP)
                elif key == "d":
                    steer = min(STEER_MAX, steer + STEER_STEP)
                elif key == " ":
                    print("\n  Resetting car...")
                    client.simulation_management.send_reset()
                    time.sleep(1.5)
                    client.setup_connection()
                    steer, acc, brake = 0.0, 0.0, 0.0
                    print("  Car reset. Ready.")
                    continue
                elif key == "r":
                    steer = 0.0
                    acc   = 0.0
                    brake = 0.0

            state = send_and_read(client, steer, acc, brake)
            print_state(state, steer, acc, brake)
            time.sleep(0.04)  # ~25Hz

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nReleasing controls and disconnecting...")
        client.controls.set_controls(steer=0, acc=0, brake=0)
        client.respond_to_server()
        client.close()
        print("Done.")

if __name__ == "__main__":
    main()
