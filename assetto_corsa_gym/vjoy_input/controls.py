"""
vjoy_input.controls
--------------------
Axis mapping and conversion logic.

Confirmed axis mapping (from find_axes.py diagnostic run):
    Axis 1 (wAxisX)  -> STEERING
    Axis 2 (wAxisY)  -> IGNORED (not used)
    Axis 3 (wAxisZ)  -> COMBINED GAS/BRAKE (single axis, shared pedals)

Steering:
    steer in [-1, +1]
    -1 = full left  ->  wAxisX = 0
     0 = center     ->  wAxisX = 16384
    +1 = full right ->  wAxisX = 32768
    Formula: int((steer + 1) * 16384)

Combined gas/brake (single axis):
    throttle in [0, 1], brake in [0, 1]
    W pressed (throttle > 0): axis goes LOW from center
    S pressed (brake    > 0): axis goes HIGH from center
    Neither:                  axis = center = 16384

    Formula: int(16384 + (brake - throttle) * 16384)

    throttle=0, brake=0  ->  16384  (neutral / center)
    throttle=1, brake=0  ->      0  (full throttle, LOW)
    throttle=0, brake=1  ->  32768  (full brake,    HIGH)

Neutral joystick state: steer=0.0, throttle=0.0, brake=0.0
    -> wAxisX = 16384, wAxisZ = 16384
"""

AXIS_MAX  = 32768
AXIS_MID  = 16384   # center value for both steer and combined axis
SCALE     = 16384   # half of AXIS_MAX

BUTTON_SHIFT_UP   = 0x00000001
BUTTON_SHIFT_DOWN = 0x00000002


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def steer_to_axis(steer_norm: float) -> int:
    """
    Normalized steer [-1, +1] -> vJoy wAxisX [0, 32768].

    -1 = full left, 0 = center, +1 = full right.
    """
    steer_norm = clamp(steer_norm, -1.0, 1.0)
    return int((steer_norm + 1.0) * SCALE)    # [-1,1] -> [0, 32768]


def combined_to_axis(throttle: float, brake: float) -> int:
    """
    Convert separate throttle [0, 1] and brake [0, 1] values to the
    single combined gas/brake axis (wAxisZ) [0, 32768].

    Axis is LOW when throttle is applied, HIGH when brake is applied,
    and center (16384) when neither is pressed.

    Formula: int(16384 + (brake - throttle) * 16384)

    throttle=0, brake=0  ->  16384  (neutral)
    throttle=1, brake=0  ->      0  (full throttle)
    throttle=0, brake=1  ->  32768  (full brake)
    """
    throttle = clamp(throttle, 0.0, 1.0)
    brake    = clamp(brake,    0.0, 1.0)
    return int(AXIS_MID + (brake - throttle) * SCALE)


def buttons_mask(shift_up: bool = False, shift_down: bool = False) -> int:
    """Build the 32-bit button bitmask for gear shift buttons."""
    mask = 0
    if shift_up:
        mask |= BUTTON_SHIFT_UP
    if shift_down:
        mask |= BUTTON_SHIFT_DOWN
    return mask


def encode(
    steer:      float = 0.0,
    throttle:   float = 0.0,
    brake:      float = 0.0,
    shift_up:   bool  = False,
    shift_down: bool  = False,
) -> dict:
    """
    Convert normalized agent controls to a dict of raw vJoy axis values.

    Confirmed vJoy Device 1 axis mapping (from find_axes.py diagnostic):
        wAxisX  -> Steering         (AC AXLE=0)
        wAxisY  -> Ignored
        wAxisZ  -> Combined gas/brake (AC AXLE=2, single shared axis)

    Args:
        steer:    [-1, +1]  left to right
        throttle: [ 0, +1]  0=released, 1=fully pressed
        brake:    [ 0, +1]  0=released, 1=fully pressed

    Returns:
        dict with keys: wAxisX, wAxisZ, lButtons
    """
    return {
        "wAxisX":  steer_to_axis(steer),
        "wAxisZ":  combined_to_axis(throttle, brake),
        "lButtons": buttons_mask(shift_up, shift_down),
    }


def neutral() -> dict:
    """
    Return the neutral/idle control state.

    Steer    = center = 16384
    Combined = center = 16384  (throttle=0, brake=0)
    """
    return encode(steer=0.0, throttle=0.0, brake=0.0)
