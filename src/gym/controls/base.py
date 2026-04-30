"""
BaseControl — interface for the policy-output → hardware bridge.

Policies output actions in [0,1]³ (steering, throttle, brake).
Controls map that to whatever the hardware needs and send it.

Two usage modes:
    1. map_action(policy_action) → ac_action
       Returns the AC-space action; caller passes it to env.step().
       This is the normal per-step path.

    2. execute_direct(ac_action)
       Sends AC-space action directly to hardware, bypassing env.step().
       Used by crash handler and neutralize (outside the step loop).
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseControl(ABC):

    @abstractmethod
    def map_action(self, policy_action: np.ndarray) -> np.ndarray:
        """
        Map policy space [0,1]³ → AC internal space [-1,1]³.

        Parameters
        ----------
        policy_action : np.ndarray shape (3,) values in [0,1]
            [steering, throttle, brake]

        Returns
        -------
        np.ndarray shape (3,) values in [-1,1]
            [steer, acc, brake] in AC space
        """

    @abstractmethod
    def execute_direct(self, ac_action: np.ndarray) -> None:
        """
        Send an AC-space action directly to hardware.

        Used when we need to send controls outside of env.step():
            - crash handler (brake for N seconds)
            - neutralize on exit

        Parameters
        ----------
        ac_action : np.ndarray shape (3,) values in [-1,1]
        """

    def neutralize(self) -> None:
        """Send the neutral (no input) command: steer=0, acc=-1, brake=-1."""
        self.execute_direct(np.array([0.0, -1.0, -1.0], dtype=np.float32))
