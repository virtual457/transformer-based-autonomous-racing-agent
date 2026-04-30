"""
vector_reward.py — Decompose scalar reward components into per-action reward vector.

Each action channel gets its own reward signal:
    r[0] (steer):    gap + gap_dir + yaw + smoothness_steer
    r[1] (throttle): speed_throttle + gap + yaw + smoothness_throttle
    r[2] (brake):    speed_brake + gap + yaw + smoothness_brake

Crash override: if out_of_track, all channels get -1.0.

Speed decomposition (pure intent-based):
    Four zones:
      At target (within 5%):        both channels +1.0 (maintain)
      Below target (>= 50%):        both channels +1.0 (near enough — any action OK)
      Below target (< 50%):         r_throttle = +throttle_action, r_brake = -brake_action
      Above target:                 r_throttle = -throttle_action, r_brake = +brake_action

    When dangerously slow (< 50% target), reward is proportional to action magnitude.
    When near target (50-95%), both throttle and brake get a free pass.

    No proximity/outcome blending — purely rewards correct intent.
"""

import numpy as np

# ── Per-channel weights (sum to 1.0 within each channel) ─────────────────────

W_STEER = {
    "gap":     0.30,
    "gap_dir": 0.25,
    "yaw":     0.25,
    "smooth":  0.20,
}

W_THROTTLE = {
    "speed":    0.30,
    "gap":      0.40,
    "yaw":      0.20,
    "smooth":   0.10,
}

W_BRAKE = {
    "speed":  0.30,
    "gap":    0.40,
    "yaw":    0.20,
    "smooth": 0.10,
}

# ── Per-action smoothness caps ────────────────────────────────────────────────
# Maximum per-action delta that scores -1.  L1 per dimension, not L2.
SMOOTH_CAP_STEER    = 0.3
SMOOTH_CAP_THROTTLE = 0.5
SMOOTH_CAP_BRAKE    = 0.5

# ── Speed target tolerance ───────────────────────────────────────────────────
AT_TARGET_TOL = 0.05  # 5% of target — within this band = "at target" → +1.0

# ── Steer directional gap deadzone ────────────────────────────────────────────
GAP_DIR_DEADZONE_M = 0.25  # within 0.25m of center — any steer direction is fine

# ── Fallback speed constants (when target_speed unavailable) ──────────────────
SPEED_NEUTRAL_MS = 16.667   # 60 km/h



def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _per_action_smoothness(
    action: np.ndarray,
    prev_action: np.ndarray,
) -> tuple:
    """Return (smooth_steer, smooth_throttle, smooth_brake) each in [-1, 1]."""
    d0 = abs(float(action[0] - prev_action[0]))
    d1 = abs(float(action[1] - prev_action[1]))
    d2 = abs(float(action[2] - prev_action[2]))
    s0 = 1.0 - 2.0 * _clamp(d0 / SMOOTH_CAP_STEER,    0.0, 1.0)
    s1 = 1.0 - 2.0 * _clamp(d1 / SMOOTH_CAP_THROTTLE,  0.0, 1.0)
    s2 = 1.0 - 2.0 * _clamp(d2 / SMOOTH_CAP_BRAKE,     0.0, 1.0)
    return s0, s1, s2


def _steer_gap_dir(gap_m: float, steer_action: float) -> float:
    """
    Directional steer reward: reward steering back toward the racing line.

    gap_m > 0  → car RIGHT of line → correct steer is LEFT  (negative)
    gap_m < 0  → car LEFT  of line → correct steer is RIGHT (positive)
    |gap_m| < deadzone → on the line, any steer is fine → +1.0

    Returns [-1, 1], proportional to steer magnitude.
    """
    if abs(gap_m) < GAP_DIR_DEADZONE_M:
        return 1.0
    sign = 1.0 if gap_m > 0.0 else -1.0
    return _clamp(-sign * steer_action, -1.0, 1.0)


def _speed_rewards_per_channel(
    speed_ms: float,
    target_speed_ms: float,
    throttle_action: float,
    brake_action: float,
) -> tuple:
    """
    Return (r_throttle_speed, r_brake_speed) each in [-1, 1].

    Pure intent-based: reward = direction × action_value.
    The reward is proportional to the action magnitude — half throttle when
    below target scores +0.5, full throttle scores +1.0.

    Three zones:
      At target (within 5%):  both channels +1.0 (maintain — keep doing whatever)
      Below target:           r_throttle = +throttle_action, r_brake = -brake_action
      Above target:           r_throttle = -throttle_action, r_brake = +brake_action

    Actions are in [-1, 1]: throttle −1=off +1=full, brake −1=off +1=full.

    Examples (below target):
      full throttle (+1.0)  → r_throttle = +1.0   (correct intent, full reward)
      half throttle (+0.5)  → r_throttle = +0.5   (correct intent, partial reward)
      no throttle   (-1.0)  → r_throttle = -1.0   (wrong intent, full penalty)
      no brake      (-1.0)  → r_brake    = +1.0   (correct — not braking)
      full brake    (+1.0)  → r_brake    = -1.0   (wrong — braking when too slow)

    Fallback when target_speed_ms <= 0: use SPEED_NEUTRAL_MS as target.
    """
    target = target_speed_ms if target_speed_ms > 0.0 else SPEED_NEUTRAL_MS

    # At target: both channels get +1.0 (maintain).
    if abs(speed_ms - target) < target * AT_TARGET_TOL:
        return 1.0, 1.0

    # Below target: throttle good, brake bad.
    # Above target: brake good, throttle bad.
    if speed_ms < target:
        if speed_ms < target * 0.80:
            r_throttle = throttle_action   # punish lifting off when slow
            r_brake = -brake_action        # punish braking when slow
        else:
            r_throttle = 1.0              # near target — lifting off is fine
            r_brake = 1.0                 # near target — braking is fine
    else:
        r_throttle = -throttle_action      # +1 at throttle off (-1), -1 at full throttle (+1)
        r_brake    = brake_action          # +1 at full brake, -1 at brake off

    return _clamp(r_throttle, -1.0, 1.0), _clamp(r_brake, -1.0, 1.0)


def compute_vector_reward_detailed(
    components: dict,
    metrics: dict,
    action: np.ndarray,
    prev_action: np.ndarray,
    out_of_track: bool = False,
) -> dict:
    """
    Same as compute_vector_reward but returns full per-channel component breakdown.

    Returns
    -------
    dict with keys:
        reward_vec : (3,) — [r_steer, r_throttle, r_brake]
        steer      : dict of weighted components
        throttle   : dict of weighted components
        brake      : dict of weighted components
    """
    if out_of_track:
        _crash_vec = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        _z = {"crash": -1.0}
        return {"reward_vec": _crash_vec, "steer": _z, "throttle": _z, "brake": _z}

    smooth_steer, smooth_throttle, smooth_brake = _per_action_smoothness(action, prev_action)

    speed_ms = metrics.get("speed_ms", 0.0)
    target_speed_ms = metrics.get("target_speed_ms", 0.0)
    r_throttle_speed, r_brake_speed = _speed_rewards_per_channel(
        speed_ms, target_speed_ms,
        throttle_action=float(action[1]),
        brake_action=float(action[2]),
    )

    r_gap      = components.get("r_gap_abs",  0.0)
    r_yaw      = components.get("r_yaw",      0.0)
    r_progress = components.get("r_progress", 0.0)

    gap_m = metrics.get("gap_m", 0.0)
    r_gap_dir = _steer_gap_dir(gap_m, float(action[0]))

    # Weighted components per channel.
    sc = {
        "gap":     W_STEER["gap"]     * r_gap,
        "gap_dir": W_STEER["gap_dir"] * r_gap_dir,
        "yaw":     W_STEER["yaw"]     * r_yaw,
        "smooth":  W_STEER["smooth"]  * smooth_steer,
    }
    tc = {
        "speed":    W_THROTTLE["speed"]    * r_throttle_speed,
        "gap":      W_THROTTLE["gap"]      * r_gap,
        "yaw":      W_THROTTLE["yaw"]      * r_yaw,
        "smooth":   W_THROTTLE["smooth"]   * smooth_throttle,
    }
    bc = {
        "speed":  W_BRAKE["speed"]  * r_brake_speed,
        "gap":    W_BRAKE["gap"]    * r_gap,
        "yaw":    W_BRAKE["yaw"]    * r_yaw,
        "smooth": W_BRAKE["smooth"] * smooth_brake,
    }

    r_steer    = sum(sc.values())
    r_throttle = sum(tc.values())
    r_brake    = sum(bc.values())

    reward_vec = np.clip(
        np.array([r_steer, r_throttle, r_brake], dtype=np.float32), -1.0, 1.0,
    )
    return {"reward_vec": reward_vec, "steer": sc, "throttle": tc, "brake": bc}


def compute_vector_reward(
    components: dict,
    metrics: dict,
    action: np.ndarray,
    prev_action: np.ndarray,
    out_of_track: bool = False,
) -> np.ndarray:
    """
    Decompose reward components into a per-action reward vector.

    Parameters
    ----------
    components : dict
        Keys: r_gap_abs, r_yaw, r_progress, r_speed, r_smoothness, r_crash.
        All values in [-1, 1].
    metrics : dict
        Keys: speed_ms, gap_m, yaw_error_deg, target_speed_ms (optional).
    action : np.ndarray, shape (3,)
        Current action [steer, throttle, brake].
    prev_action : np.ndarray, shape (3,)
        Previous action [steer, throttle, brake].
    out_of_track : bool
        Whether the car is out of track (crash).

    Returns
    -------
    np.ndarray, shape (3,) — [r_steer, r_throttle, r_brake], each in [-1, 1].
    """
    # ── Crash override ───────────────────────────────────────────────────────
    if out_of_track:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32)

    # ── Per-action smoothness ────────────────────────────────────────────────
    smooth_steer, smooth_throttle, smooth_brake = _per_action_smoothness(
        action, prev_action,
    )

    # ── Speed rewards (directional, action-modulated) ────────────────────────
    speed_ms = metrics.get("speed_ms", 0.0)
    target_speed_ms = metrics.get("target_speed_ms", 0.0)
    r_throttle_speed, r_brake_speed = _speed_rewards_per_channel(
        speed_ms, target_speed_ms,
        throttle_action=float(action[1]),
        brake_action=float(action[2]),
    )

    # ── Existing components ──────────────────────────────────────────────────
    r_gap      = components.get("r_gap_abs",    0.0)
    r_yaw      = components.get("r_yaw",        0.0)
    r_progress = components.get("r_progress",   0.0)

    # ── Directional steer gap ────────────────────────────────────────────────
    gap_m = metrics.get("gap_m", 0.0)
    r_gap_dir = _steer_gap_dir(gap_m, float(action[0]))

    # ── Steer channel ────────────────────────────────────────────────────────
    r_steer = (
        W_STEER["gap"]     * r_gap
        + W_STEER["gap_dir"] * r_gap_dir
        + W_STEER["yaw"]     * r_yaw
        + W_STEER["smooth"]  * smooth_steer
    )

    # ── Throttle channel ─────────────────────────────────────────────────────
    r_throttle = (
        W_THROTTLE["speed"]    * r_throttle_speed
        + W_THROTTLE["gap"]      * r_gap
        + W_THROTTLE["yaw"]      * r_yaw
        + W_THROTTLE["smooth"]   * smooth_throttle
    )

    # ── Brake channel ────────────────────────────────────────────────────────
    r_brake = (
        W_BRAKE["speed"]  * r_brake_speed
        + W_BRAKE["gap"]    * r_gap
        + W_BRAKE["yaw"]    * r_yaw
        + W_BRAKE["smooth"] * smooth_brake
    )

    return np.clip(
        np.array([r_steer, r_throttle, r_brake], dtype=np.float32),
        -1.0, 1.0,
    )
