"""
vjoy_input.keyboard_simulator
------------------------------
Real-time keyboard -> vJoy bridge for Assetto Corsa.

Key bindings:
    W          throttle (hold = ramp up, release = decay back)
    S          brake    (hold = ramp up, release = decay back)
    A          steer left  (analog ramp)
    D          steer right (analog ramp)
    Space      instant full brake (handbrake)
    Q          shift down (tap)
    E          shift up   (tap)
    R          reset all inputs to neutral
    Esc        quit

Analog feel:
    Holding a key increases the axis at RAMP_RATE per second.
    Releasing a key decays the axis at DECAY_RATE per second back to neutral.

FIX NOTE:
    Uses key-down/key-up EVENT hooks (not is_pressed polling) so it works
    reliably even when Assetto Corsa has window focus.

Usage:
    cd assetto_corsa_gym
    python -m vjoy_input.keyboard_simulator
"""

import time
import sys
import os
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import keyboard
from vjoy_input import JoystickSimulator
from vjoy_input.controls import AXIS_MAX, AXIS_MID

# ------------------------------------------------------------------
# Tuning constants
# ------------------------------------------------------------------

LOOP_HZ = 50            # control update rate (matches AC plugin 50 Hz)
DT      = 1.0 / LOOP_HZ

STEER_RAMP_RATE     = 2.0   # full steer in 0.5 s
STEER_DECAY_RATE    = 3.0   # springs back faster
THROTTLE_RAMP_RATE  = 2.0   # full throttle in 0.5 s
THROTTLE_DECAY_RATE = 4.0
BRAKE_RAMP_RATE     = 3.0
BRAKE_DECAY_RATE    = 5.0
GEAR_HOLD_S         = 0.05  # seconds button is held for gear shift

# ------------------------------------------------------------------
# Key bindings
# ------------------------------------------------------------------

BINDINGS = {
    "throttle":    "w",
    "brake":       "s",
    "steer_left":  "a",
    "steer_right": "d",
    "shift_up":    "e",
    "shift_down":  "q",
    "handbrake":   "space",
    "reset":       "r",
    "quit":        "escape",
}

# ------------------------------------------------------------------
# Held-key tracker  (event-based, works when AC has focus)
# ------------------------------------------------------------------

class KeyTracker:
    """
    Tracks which keys are currently held by listening to key-down / key-up
    events from the keyboard hook. More reliable than is_pressed() when
    another application has window focus.
    """

    def __init__(self):
        self._held  = set()   # set of currently-held key names
        self._taps  = {}      # key -> pending tap count
        self._lock  = threading.Lock()

    def register(self, keys: list):
        """Register key-down/key-up hooks for the given key list."""
        keyboard.hook(self._on_event)

    def _on_event(self, event: keyboard.KeyboardEvent):
        name = event.name.lower() if event.name else ""
        with self._lock:
            if event.event_type == keyboard.KEY_DOWN:
                if name not in self._held:
                    self._held.add(name)
                    # count as a tap only on the leading edge
                    self._taps[name] = self._taps.get(name, 0) + 1
            elif event.event_type == keyboard.KEY_UP:
                self._held.discard(name)

    def is_held(self, key: str) -> bool:
        with self._lock:
            return key.lower() in self._held

    def consume_tap(self, key: str) -> bool:
        """Returns True once per leading-edge press of key."""
        with self._lock:
            if self._taps.get(key.lower(), 0) > 0:
                self._taps[key.lower()] -= 1
                return True
            return False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _clamp(val, lo=-1.0, hi=1.0):
    return max(lo, min(hi, val))


def _ramp(current, target, rate, dt):
    if current < target:
        return min(current + rate * dt, target)
    return max(current - rate * dt, target)


def _bar(value, lo=-1.0, hi=1.0, width=22):
    ratio  = (value - lo) / (hi - lo)
    filled = int(ratio * width)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _render(steer, throttle, brake, raw_axes, gear_event):
    lines = [
        "",
        "  WASD -> Assetto Corsa  (Esc to quit)",
        "  " + "-" * 50,
        f"  Steer    (A/D)  {_bar(steer)}  {steer:+.3f}",
        f"  Throttle (W)    {_bar(throttle)}  {throttle:+.3f}",
        f"  Brake    (S)    {_bar(brake)}  {brake:+.3f}",
        "  " + "-" * 50,
        f"  vJoy raw  X={raw_axes['wAxisX']:5d} (steer)  Z={raw_axes['wAxisZ']:5d} (gas/brake)",
        f"  Gear: {gear_event}",
        "  Space=handbrake  E=shift up  Q=shift down  R=reset",
    ]
    # Move cursor up N lines and overwrite (works in most terminals)
    n = len(lines)
    sys.stdout.write(f"\033[{n}A")   # cursor up
    for line in lines:
        sys.stdout.write(f"\r{line:<60}\n")
    sys.stdout.flush()


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def run(device_id: int = 1):
    tracker    = KeyTracker()
    tracker.register(list(BINDINGS.values()))

    steer    =  0.0
    throttle =  0.0
    brake    =  0.0
    gear_event = "none"
    running  = True
    raw_axes = {"wAxisX": AXIS_MID, "wAxisZ": AXIS_MID}

    def on_quit(e):
        nonlocal running
        running = False

    keyboard.on_press_key(BINDINGS["quit"], on_quit)

    print("\nOpening vJoy device...")
    try:
        with JoystickSimulator(device_id=device_id) as sim:
            print(f"vJoy device {device_id} acquired.")
            print("Hold W=throttle  S=brake  A=left  D=right")
            print("Tap  E=shift-up  Q=shift-down  Space=handbrake  R=reset\n")

            # Print blank lines that _render will overwrite
            print("\n" * 10)

            while running:
                t0 = time.perf_counter()

                # ---- Read key states (event-based) -------------------
                throttle_held  = tracker.is_held(BINDINGS["throttle"])
                brake_held     = tracker.is_held(BINDINGS["brake"])
                left_held      = tracker.is_held(BINDINGS["steer_left"])
                right_held     = tracker.is_held(BINDINGS["steer_right"])
                handbrake_held = tracker.is_held(BINDINGS["handbrake"])
                reset_now      = tracker.consume_tap(BINDINGS["reset"])
                shift_up_now   = tracker.consume_tap(BINDINGS["shift_up"])
                shift_down_now = tracker.consume_tap(BINDINGS["shift_down"])

                # ---- Reset -------------------------------------------
                if reset_now:
                    steer    =  0.0
                    throttle =  0.0
                    brake    =  0.0
                    sim.reset()
                    gear_event = "reset"

                # ---- Steer -------------------------------------------
                if left_held and not right_held:
                    steer = _ramp(steer, -1.0, STEER_RAMP_RATE, DT)
                elif right_held and not left_held:
                    steer = _ramp(steer,  1.0, STEER_RAMP_RATE, DT)
                else:
                    steer = _ramp(steer,  0.0, STEER_DECAY_RATE, DT)

                # ---- Throttle ----------------------------------------
                if throttle_held:
                    throttle = _ramp(throttle, 1.0, THROTTLE_RAMP_RATE, DT)
                else:
                    throttle = _ramp(throttle, 0.0, THROTTLE_DECAY_RATE, DT)

                # ---- Brake / Handbrake -------------------------------
                if handbrake_held:
                    brake = 1.0
                elif brake_held:
                    brake = _ramp(brake, 1.0, BRAKE_RAMP_RATE, DT)
                else:
                    brake = _ramp(brake, 0.0, BRAKE_DECAY_RATE, DT)

                # ---- Send to vJoy ------------------------------------
                raw_axes = sim.send(steer=steer, throttle=throttle, brake=brake)

                # ---- Gear shifts (tap) -------------------------------
                if shift_up_now:
                    sim.send_gear_shift(shift_up=True, hold_s=GEAR_HOLD_S)
                    gear_event = "shift UP (E)"
                elif shift_down_now:
                    sim.send_gear_shift(shift_down=True, hold_s=GEAR_HOLD_S)
                    gear_event = "shift DOWN (Q)"

                # ---- Display -----------------------------------------
                _render(steer, throttle, brake, raw_axes, gear_event)

                # ---- Timing ------------------------------------------
                elapsed = time.perf_counter() - t0
                sleep_t = DT - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.unhook_all()
        print("\n\nExited. vJoy device released.")


if __name__ == "__main__":
    run()
