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

r_speed     : delta-based target-tracking reward (SpeedDeltaReward).
              AT_TARGET_TOL = target * 0.05 (5% window).
              Both frames within tolerance → +1.0 (maintaining target).
              Otherwise: clamp((error_t - error_t1) / normaliser, -1, 1)
              where normaliser = max(target * 0.1, 1.0).
              Falls back to piecewise SpeedReward formula when target == 0.

r_gap_abs   : 1.0 - 2.0 * clamp(abs(gap_m) / GAP_ABS_CAP_M, 0, 1)
              Perfect on-line = +1.0, 4 m or more off-line = -1.0.

r_smoothness: 1.0 - 2.0 * clamp(action_delta_norm / SMOOTHNESS_CAP, 0, 1)
              Perfectly smooth = +1, half-cap delta = 0, at/above cap = -1.
"""

import math

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
                prev_gap_m: float = 0.0, prev_speed_ms: float = 0.0):
        delta = telem.lap_dist - prev_lap_dist
        if delta < -0.5 * track_length:   # lap wrap-around
            delta += track_length
        return _clamp(delta / PROGRESS_CAP_M - 1.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

SPEED_NEUTRAL_MS  = 16.667   # 60 km/h — below this scores negative
SPEED_CAP_MS      = 52.778   # 190 km/h — scores +1.0
_SPEED_UPPER_RANGE = SPEED_CAP_MS - SPEED_NEUTRAL_MS   # 36.111


class SpeedReward(BaseRewardComponent):
    """
    Piecewise linear speed reward, normalised to [-1, 1].

    speed < 5.556 m/s : speed / 5.556 - 1          → [-1,  0]
    speed >= 5.556 m/s: (speed - 5.556) / 47.222   → [ 0, +1]
    Clamped to [-1, 1].
    """

    name = "r_speed"

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0, prev_speed_ms: float = 0.0):
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
                prev_gap_m: float = 0.0, prev_speed_ms: float = 0.0):
        target = telem.target_speed_ms
        if target <= 0.0:
            # fallback: use SpeedReward formula
            s = telem.speed_ms
            if s < SPEED_NEUTRAL_MS:
                return _clamp(s / SPEED_NEUTRAL_MS - 1.0, -1.0, 1.0)
            return _clamp((s - SPEED_NEUTRAL_MS) / _SPEED_UPPER_RANGE, -1.0, 1.0)
        delta = abs(telem.speed_ms - target) / target
        return _clamp(1.0 - 2.0 * delta, -1.0, 1.0)


class SpeedDeltaReward(BaseRewardComponent):
    """
    Delta-based target-speed tracking reward, normalised to [-1, 1].

    Rewards progress toward the target speed rather than proximity to it.
    This gives a positive gradient for any action that reduces the error,
    regardless of the current magnitude — fixing the flat-gradient problem
    of TargetSpeedReward when the car is far from target.

    Call convention
    ---------------
    The caller (our_env.py) must pass prev_speed_ms = speed at frame t.
    telem.speed_ms is the speed at frame t+1 (next frame, already observed).
    telem.target_speed_ms is the current target (frame t).

    Formula
    -------
    AT_TARGET_TOL = target * 0.05          # 5% window
    error_t   = |prev_speed  - target|
    error_t1  = |speed_t1    - target|

    If both error_t and error_t1 < AT_TARGET_TOL:
        r = +1.0                           # maintaining target → full reward

    Else:
        delta      = error_t - error_t1   # positive = getting closer
        normaliser = max(target * 0.1, 1.0)
        r = clamp(delta / normaliser, -1.0, 1.0)

    Falls back to SpeedReward formula when target_speed_ms == 0.0.

    Behaviour summary
    -----------------
    Maintaining target speed      → +1.0
    Over target, braking          → positive (error shrinks)
    Over target, throttling more  → negative (error grows)
    Under target, throttling      → positive (error shrinks)
    Under target, braking         → negative (error grows)
    Delta == 0, not at target     → 0.0 (neutral, no gradient trick needed)
    """

    name = "r_speed"   # same key for log compatibility

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0, prev_speed_ms: float = 0.0):
        target = telem.target_speed_ms
        if target <= 0.0:
            # fallback: piecewise SpeedReward formula
            s = telem.speed_ms
            if s < SPEED_NEUTRAL_MS:
                return _clamp(s / SPEED_NEUTRAL_MS - 1.0, -1.0, 1.0)
            return _clamp((s - SPEED_NEUTRAL_MS) / _SPEED_UPPER_RANGE, -1.0, 1.0)

        error_t  = abs(prev_speed_ms  - target)
        error_t1 = abs(telem.speed_ms - target)

        normaliser = max(target * 0.1, 1.0)
        return _clamp((error_t - error_t1) / normaliser, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Gap absolute
# ---------------------------------------------------------------------------

GAP_ABS_CAP_M = 4.0   # 4 m off the racing line = full penalty (-1)  # was 10.0


class GapReward(BaseRewardComponent):
    """
    Absolute lateral distance from the racing line, normalised to [-1, 1].
    Perfectly on-line = +1.0.  4 m or more off-line = -1.0.
    Linear mapping: 0 m → +1.0, 2 m → 0.0, 4 m → -1.0.

    Formula: 1.0 - 2.0 * clamp(abs(gap_m) / GAP_ABS_CAP_M, 0.0, 1.0)
    CompositeReward must NOT negate this component.
    """

    name = "r_gap_abs"

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0, prev_speed_ms: float = 0.0):
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
                prev_gap_m: float = 0.0, prev_speed_ms: float = 0.0):
        delta_norm = float(np.linalg.norm(
            np.asarray(action, dtype=np.float32)
            - np.asarray(prev_action, dtype=np.float32),
            ord=2,
        ))
        return 1.0 - 2.0 * _clamp(delta_norm / SMOOTHNESS_CAP, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Yaw alignment
# ---------------------------------------------------------------------------

YAW_NEG_ONE_DEG = 6.0   # degrees — reward = -1 here (clamped below)
YAW_ZERO_DEG    = 3.0   # degrees — reward = 0 here (zero crossing)


class YawAlignmentReward(BaseRewardComponent):
    """
    Heading alignment reward, normalised to [-1, 1].

      0°  → +1.0  (perfectly aligned with racing line)
      3°  →  0.0  (zero crossing)
      6°  → -1.0  (clamped)

    Single linear slope: raw = 1.0 - yaw_abs / 3.0, clamped to [-1, 1].
    """

    name = "r_yaw"

    def compute(self, telem, action, prev_action, prev_lap_dist, track_length,
                prev_gap_m: float = 0.0, prev_speed_ms: float = 0.0):
        yaw_abs = abs(math.degrees(telem.yaw_error_rad))
        return _clamp(1.0 - yaw_abs / YAW_ZERO_DEG, -1.0, 1.0)


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
                prev_gap_m: float = 0.0, prev_speed_ms: float = 0.0):
        return 1.0 if telem.out_of_track else 0.0
