"""
vjoy_input.find_axes
---------------------
Interactive diagnostic script to identify which vJoy axis maps to steer,
throttle, and brake in AC.

Instructions:
1. Open Assetto Corsa -> Content Manager -> Settings -> Controls
   (or the in-game Controls calibration screen)
2. Run this script from the assetto_corsa_gym directory:
       python -m vjoy_input.find_axes
3. Hold a key (1-6) to drive that axis to 32768 (max).
   Release the key and the axis drops back to 0.
   All other axes stay at 0 while no key is held.
   Watch the axis bars in AC to see which one moves.
4. Press Q to quit.

Key -> axis mapping:
    1 -> wAxisX    (DirectInput axis X)
    2 -> wAxisY    (DirectInput axis Y)
    3 -> wAxisZ    (DirectInput axis Z)
    4 -> wAxisXRot (DirectInput axis Rx)
    5 -> wAxisYRot (DirectInput axis Ry)
    6 -> wAxisZRot (DirectInput axis Rz)
    7 -> (not exposed by VJoyDriver.update — cannot test)
    8 -> (not exposed by VJoyDriver.update — cannot test)

Behaviour:
    - Key held   -> that axis = 32768, all others = 0
    - Key released -> all axes = 0
    - Multiple keys held simultaneously -> all held axes = 32768
"""

import sys
import os
import time

import keyboard  # pip install keyboard

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vjoy_input.driver import VJoyDriver

DEVICE_ID = 1
MAX_VAL   = 32768
ZERO_VAL  = 0

# Poll interval in seconds.  20 ms gives smooth visual feedback in AC.
POLL_INTERVAL = 0.02

# Keys 1-6 map to axes supported by VJoyDriver.update().
# Keys 7-8 (wSlider / wDial) are NOT in update()'s signature and cannot
# be driven without modifying the driver — they are listed for reference only.
AXIS_MAP = [
    ("1", "wAxisX",    "wAxisX    (DirectInput X)"),
    ("2", "wAxisY",    "wAxisY    (DirectInput Y)"),
    ("3", "wAxisZ",    "wAxisZ    (DirectInput Z)"),
    ("4", "wAxisXRot", "wAxisXRot (DirectInput Rx)"),
    ("5", "wAxisYRot", "wAxisYRot (DirectInput Ry)"),
    ("6", "wAxisZRot", "wAxisZRot (DirectInput Rz)"),
]

# Fast lookup: key char -> (axis_kwarg_name, friendly_label)
KEY_TO_AXIS = {key: (name, label) for key, name, label in AXIS_MAP}

# All axis kwarg names in display order (for the live readout line).
ALL_AXIS_NAMES = [name for _, name, _ in AXIS_MAP]


def build_state() -> dict:
    """
    Return a dict of axis_kwarg -> value based on which keys are currently held.
    Any key in KEY_TO_AXIS that is held right now drives its axis to MAX_VAL.
    All other axes are ZERO_VAL.
    """
    state = {name: ZERO_VAL for name in ALL_AXIS_NAMES}
    for key_char, axis_name in ((k, n) for k, (n, _) in KEY_TO_AXIS.items()):
        if keyboard.is_pressed(key_char):
            state[axis_name] = MAX_VAL
    return state


def print_live(state: dict):
    """Overwrite the current terminal line with all axis values."""
    parts = []
    for name in ALL_AXIS_NAMES:
        val = state[name]
        # Mark axes that are at max with an asterisk so they stand out.
        marker = "*" if val == MAX_VAL else " "
        parts.append(f"{marker}{name}={val:5d}")
    line = "  ".join(parts)
    # \r returns to line start; pad to 100 chars to overwrite any leftover text.
    sys.stdout.write(f"\r{line:<100}")
    sys.stdout.flush()


def print_menu():
    print()
    print("=" * 64)
    print("  vJoy Interactive Axis Finder  (hold-to-drive edition)")
    print("=" * 64)
    print("  Hold a key to drive that axis to 32768.")
    print("  Release the key to return it to 0.")
    print("  Multiple keys can be held at the same time.")
    print()
    print("  Key  Axis")
    print("  ---  ----")
    for key, name, label in AXIS_MAP:
        print(f"   {key}   {label}")
    print("   7   wSlider  (not supported by driver -- skip)")
    print("   8   wDial    (not supported by driver -- skip)")
    print()
    print("   Q   Quit")
    print("-" * 64)
    print("  Open the AC Controls screen before testing.")
    print("  Held axis is marked with * in the live readout.")
    print("=" * 64)


def main():
    print_menu()

    with VJoyDriver(device_id=DEVICE_ID) as driver:
        if not driver.acquired:
            print("\n  ERROR: Could not acquire vJoy Device 1.")
            print("  Make sure vJoy is installed and the device is enabled.")
            return

        # Zero all axes at start.
        driver.update(**{name: ZERO_VAL for name in ALL_AXIS_NAMES})
        print("\n  All axes at 0.  Hold 1-6 to drive an axis.  Press Q to quit.\n")

        while True:
            # Quit check — keyboard.is_pressed works for held and tapped Q.
            if keyboard.is_pressed("q"):
                print("\n\n  Q pressed.  Zeroing all axes and exiting...")
                break

            state = build_state()
            driver.update(**state)
            print_live(state)

            time.sleep(POLL_INTERVAL)

        # Final zero on exit.
        driver.update(**{name: ZERO_VAL for name in ALL_AXIS_NAMES})

    print("\n" + "=" * 64)
    print("  Done.")
    print("  Use your observations to confirm the axis mapping:")
    print("    steer    -> which axis moved the steering bar?")
    print("    throttle -> which axis moved the throttle bar?")
    print("    brake    -> which axis moved the brake bar?")
    print("=" * 64)


if __name__ == "__main__":
    main()
