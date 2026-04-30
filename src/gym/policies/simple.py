"""
Simple baseline policies — no learning, no model weights.

Useful for:
  - Smoke testing the environment pipeline
  - Generating baseline data (random exploration)
  - Verifying reward signal sanity
"""

import numpy as np
from policies.base import BasePolicy


class ZeroPolicy(BasePolicy):
    """
    Does nothing — no throttle, no brake, no steer.
    Car sits still. Useful only for pipeline smoke tests.
    """

    def select_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        # steering=0.5 (straight), throttle=0.0 (no gas), brake=0.0 (no brake)
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)


class FullThrottlePolicy(BasePolicy):
    """
    Full throttle, straight steering, no braking.
    Car drives straight — will crash at first corner.
    Useful to verify vJoy input is actually moving the car.
    """

    def select_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        # steering=0.5 (straight), throttle=1.0 (full gas), brake=0.0 (none)
        return np.array([0.5, 1.0, 0.0], dtype=np.float32)


class RandomPolicy(BasePolicy):
    """
    Uniformly random actions each step.
    Useful for collecting exploratory data and testing termination conditions.
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        # All three outputs in [0, 1]
        return self.rng.uniform(0.0, 1.0, size=(3,)).astype(np.float32)


class ConstantPolicy(BasePolicy):
    """
    Applies a fixed action every step.
    All values in [0, 1] policy space.

    Example:
        policy = ConstantPolicy(steering=0.5, throttle=0.5, brake=0.0)
    """

    def __init__(self, steering: float = 0.5, throttle: float = 0.0, brake: float = 0.0):
        self.action = np.array([steering, throttle, brake], dtype=np.float32)

    def select_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return self.action.copy()


class SpeedLimitedThrottlePolicy(BasePolicy):
    """
    Applies throttle until a target speed is reached, then coasts.
    Uses the speed from info dict (raw AC state).
    """

    def __init__(self, target_speed_ms: float = 20.0, throttle: float = 0.8):
        self.target_speed_ms = target_speed_ms
        self.throttle = throttle

    def select_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        speed = float(info.get("speed", 0.0))
        if speed < self.target_speed_ms:
            return np.array([0.5, self.throttle, 0.0], dtype=np.float32)
        else:
            return np.array([0.5, 0.0, 0.0], dtype=np.float32)  # coast
