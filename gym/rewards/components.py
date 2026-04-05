"""
Reward components — one class per term in the 4-component bounded reward function.

Every component returns a value already normalised to [-1, 1].
Negation is baked into the formula where the component is a penalty
(gap_abs, smoothness).  Crash is handled as a hard override in
CompositeReward and does not appear in this file's weighted sum.

Component definitions
---------------------
r_progress  : clamp(delta / 0.111 - 1, -1, 1)
              Cap = 0.111 m/frame.  At 25 Hz and ~2.78 m/s this scores 0.
              At cap speed (2.775 m/s = 0.111 m / 0.04 s) this scores +1.

r_speed     : piecewise linear, neutral at 20 km/h (5.556 m/s).
              Below neutral: speed / 5.556 - 1  → [-1, 0]
              Above neutral: (speed - 5.556) / 47.222 → [0, 1]
              Cap = 190 km/h (52.778 m/s).  Result clamped to [-1, 1].

r_gap_abs   : 1.0 - 2.0 * clamp(abs(gap_m) / GAP_ABS_CAP_M, 0, 1)
              Perfect on-line = +1.0, 10 m or more off-line = -1.0.

r_smoothness: 1.0 - 2.0 * clamp(action_delta_norm / SMOOTHNESS_CAP, 0, 1)
              Perfectly smooth = +1, half-cap delta = 0, at/above cap = -1.
"""

import numpy as np

from rewards.base import BaseRewardComponent
from telemetry.base import TelemetryFrame


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

PROGRESS_CAP_M = 0.111   # metres per frame cap (≈ 2.775 m/s at 25 Hz)


class ProgressReward(BaseRewardComponent):
    """
    Forward progress along the track this step, normalised to [-1, 1].

    Formula: clamp(delta / PROGRESS_CAP_M - 1, -1, 1)
    Handles lap wrap-around (LapDist resets to 0 at line crossing).
    """

    name = "r_progress"

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0):
        delta = telem.lap_dist - prev_lap_dist
        if delta < -0.5 * track_length:   # lap wrap-around
            delta += track_length
        return _clamp(delta / PROGRESS_CAP_M - 1.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

SPEED_NEUTRAL_MS  = 5.556    # 20 km/h — below this scores negative
SPEED_CAP_MS      = 52.778   # 190 km/h — scores +1.0
_SPEED_UPPER_RANGE = SPEED_CAP_MS - SPEED_NEUTRAL_MS   # 47.222


class SpeedReward(BaseRewardComponent):
    """
    Piecewise linear speed reward, normalised to [-1, 1].

    speed < 5.556 m/s : speed / 5.556 - 1          → [-1,  0]
    speed >= 5.556 m/s: (speed - 5.556) / 47.222   → [ 0, +1]
    Clamped to [-1, 1].
    """

    name = "r_speed"

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0):
        s = telem.speed_ms
        if s < SPEED_NEUTRAL_MS:
            raw = s / SPEED_NEUTRAL_MS - 1.0
        else:
            raw = (s - SPEED_NEUTRAL_MS) / _SPEED_UPPER_RANGE
        return _clamp(raw, -1.0, 1.0)


class TargetSpeedReward(BaseRewardComponent):
    """
    Symmetric speed-matching reward: +1 at target, 0 at half-target, -1 at 0.

    Formula: clamp(1 - 2 * |speed - target| / target, -1, 1)
    Falls back to SpeedReward formula when target_speed_ms == 0.0
    (e.g. ref_lap not loaded or use_target_speed=False).
    """

    name = "r_speed"

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0):
        target = telem.target_speed_ms
        if target <= 0.0:
            # fallback: use SpeedReward formula
            s = telem.speed_ms
            if s < SPEED_NEUTRAL_MS:
                return _clamp(s / SPEED_NEUTRAL_MS - 1.0, -1.0, 1.0)
            return _clamp((s - SPEED_NEUTRAL_MS) / _SPEED_UPPER_RANGE, -1.0, 1.0)
        delta = abs(telem.speed_ms - target) / target
        return _clamp(1.0 - 2.0 * delta, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Gap absolute
# ---------------------------------------------------------------------------

GAP_ABS_CAP_M = 10.0  # 10 m off the racing line = full penalty (-1)


class GapReward(BaseRewardComponent):
    """
    Absolute lateral distance from the racing line, normalised to [-1, 1].
    Perfectly on-line = +1.0.  10 m or more off-line = -1.0.
    Linear mapping: 0 m → +1.0, 5 m → 0.0, 10 m → -1.0.

    Formula: 1.0 - 2.0 * clamp(abs(gap_m) / GAP_ABS_CAP_M, 0.0, 1.0)
    CompositeReward must NOT negate this component.
    """

    name = "r_gap_abs"

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0):
        return 1.0 - 2.0 * _clamp(abs(telem.gap_m) / GAP_ABS_CAP_M, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Smoothness
# ---------------------------------------------------------------------------

SMOOTHNESS_CAP = 2.0   # L2 norm of action delta that scores -1


class SmoothnessReward(BaseRewardComponent):
    """
    Smoothness reward, normalised to [-1, +1].
    Perfectly smooth (no change) = +1.0.  Half-cap delta = 0.0.  At/above cap = -1.0.

    Formula: 1.0 - 2.0 * clamp(action_delta_norm / SMOOTHNESS_CAP, 0, 1)
    Negation is baked in; CompositeReward must NOT negate this component.
    """

    name = "r_smoothness"

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0):
        delta_norm = float(np.linalg.norm(
            np.asarray(action, dtype=np.float32)
            - np.asarray(prev_action, dtype=np.float32),
            ord=2,
        ))
        return 1.0 - 2.0 * _clamp(delta_norm / SMOOTHNESS_CAP, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Crash  (kept for backwards compatibility; not used in weighted sum)
# ---------------------------------------------------------------------------

class CrashReward(BaseRewardComponent):
    """
    1.0 if the car is out-of-track (>= 3 tyres outside limits), else 0.0.

    This component is retained for backwards compatibility with from_weights()
    and from_config().  In the new bounded reward architecture it is NOT part
    of the weighted sum — CompositeReward short-circuits to -1.0 on crash
    before evaluating any component.  This class is never called by the new
    CompositeReward.compute() path.
    """

    name = "r_crash"

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0):
        return 1.0 if telem.out_of_track else 0.0
