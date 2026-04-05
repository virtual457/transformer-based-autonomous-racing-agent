"""
BaseRewardComponent — interface every reward term must implement.

Each component returns a single raw float (always positive/unsigned).
Negation and weighting are applied by CompositeReward, not here.

compute() signature is fixed: callers pass everything that any component
might need, components pick what they use and ignore the rest.
"""

from abc import ABC, abstractmethod
import numpy as np

from telemetry.base import TelemetryFrame


class BaseRewardComponent(ABC):
    """
    One term in the reward function.

    Subclasses:
        - set `name` (used as dict key in the trajectory)
        - implement `compute()`

    compute() must be pure / stateless — no internal state between calls.
    """

    name: str = "unnamed"

    @abstractmethod
    def compute(
        self,
        telem: TelemetryFrame,
        action: np.ndarray,
        prev_action: np.ndarray,
        prev_lap_dist: float,
        track_length: float,
        prev_gap_m: float = 0.0,
    ) -> float:
        """
        Parameters
        ----------
        telem         : current step telemetry
        action        : current action in POLICY space [0,1]³
        prev_action   : previous step action in POLICY space [0,1]³
        prev_lap_dist : lap distance at the previous step (m)
        track_length  : full track length (m), for normalising progress
        prev_gap_m    : lateral gap from the previous step (m), for delta components

        Returns
        -------
        float — raw unsigned component value (sign handled by CompositeReward)
        """

    def __repr__(self) -> str:
        return self.__class__.__name__
