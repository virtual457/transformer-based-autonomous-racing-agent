"""
controls_checker.py
-------------------
Interactive controls tester. Pick a channel, give a value, holds it in-game
until you press Q, then loops back to ask again.

Usage:
    cd assetto_corsa_gym
    python ../gym/policies/controls_checker.py

Channels:
    throttle  [0.0 – 1.0]
    brake     [0.0 – 1.0]
    steer     [-1.0 – 1.0]  (-1 = full left, 0 = straight, +1 = full right)
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assetto_corsa_gym')))

from vjoy_input import JoystickSimulator

CHANNELS = {
    "throttle": (0.0, 1.0),
    "brake":    (0.0, 1.0),
    "steer":    (-1.0, 1.0),
}

LOOP_HZ = 25
DT      = 1.0 / LOOP_HZ


def _bar(value, lo, hi, width=30):
    ratio  = (value - lo) / (hi - lo)
    filled = int(ratio * width)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def ask_channel():
    print("\nWhat do you want to test?")
    for ch in CHANNELS:
        lo, hi = CHANNELS[ch]
        print(f"  {ch}  [{lo} – {hi}]")
    while True:
        choice = input(">> ").strip().lower()
        if choice in CHANNELS:
            return choice
        print(f"  Unknown channel. Choose from: {', '.join(CHANNELS)}")


def ask_value(channel):
    lo, hi = CHANNELS[channel]
    while True:
        raw = input(f"Value for {channel} [{lo} to {hi}]: ").strip()
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"  Out of range. Enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Not a number. Try again.")


def hold(sim, channel, value):
    """Send the chosen channel at the given value until Q is pressed."""
    import keyboard

    print(f"\n  Holding {channel} = {value}  |  Press Q to stop\n")

    running = True

    def on_q(_):
        nonlocal running
        running = False

    keyboard.on_press_key("q", on_q)

    try:
        while running:
            t0 = time.perf_counter()

            kwargs = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
            kwargs[channel] = value
            sim.send(**kwargs)

            bar = _bar(value, *CHANNELS[channel])
            print(f"\r  {channel:<8} {bar}  {value:+.3f}   ", end="", flush=True)

            elapsed = time.perf_counter() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)
    finally:
        keyboard.unhook_all()
        sim.reset()
        print("\n  Released — controls reset to neutral.")


def main():
    print("=" * 50)
    print("  AC Controls Checker")
    print("=" * 50)
    print("  Opens vJoy and holds a control value in-game.")
    print("  Press Q at any time to release and re-prompt.")

    try:
        with JoystickSimulator() as sim:
            print("  vJoy acquired.\n")
            while True:
                channel = ask_channel()
                value   = ask_value(channel)
                hold(sim, channel, value)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
