"""
Base Policy interface.

Every policy must subclass BasePolicy and implement select_action().

Input:
    obs   - numpy array (obs_dim,) — the full observation vector from OurEnv
    info  - dict — the full info dict from the last step (contains raw AC state:
            speed, gap, LapDist, numberOfTyresOut, etc.)
            On the first step (after reset), info is an empty dict.

Output:
    numpy array (3,) — [steering, throttle, brake] in the POLICY space:
        steering: [0.0, 1.0]  — 0.0 = full left, 0.5 = straight, 1.0 = full right
        throttle: [0.0, 1.0]  — 0.0 = no gas,    1.0 = full gas
        brake:    [0.0, 1.0]  — 0.0 = no brake,   1.0 = full brake

    OurEnv automatically maps these to AC's internal [-1, 1] space.
    Policies never need to know about AC's internal representation.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):

    @abstractmethod
    def select_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Given the current observation and info dict, return an action.

        Parameters
        ----------
        obs  : np.ndarray, shape (obs_dim,)
        info : dict — raw AC state from the last env.step() or {} after reset

        Returns
        -------
        np.ndarray, shape (3,), dtype float32, values in [-1, 1]
            [steer, throttle, brake]
        """

    def reset(self):
        """Called at the start of each episode. Override if policy has state."""

    def __repr__(self):
        return self.__class__.__name__
