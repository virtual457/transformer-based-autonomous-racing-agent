"""
VJoyControl — maps policy actions to AC space and sends via vJoy.

The actual vJoy write in the per-step loop happens inside ac_env.step()
through DriverControls → Controls → setJoy(). We do NOT write to vJoy
directly on every step — we just return the mapped ac_action and the env
takes care of it.

execute_direct() IS a direct vJoy write — used by crash handler and
neutralize, which operate outside the env.step() loop.
"""

import numpy as np
import logging

from controls.base import BaseControl

logger = logging.getLogger(__name__)


class VJoyControl(BaseControl):
    """
    Policy-space [0,1]³ → AC-space [-1,1]³ mapping + direct send capability.

    Axis mapping formula:
        ac = policy * 2.0 - 1.0

        policy=0.0 → ac=-1.0  (min: no gas / no brake / full left)
        policy=0.5 → ac=0.0   (center: straight steer, half gas — avoid!)
        policy=1.0 → ac=+1.0  (max: full gas / full brake / full right)

    Parameters
    ----------
    ac_client : AssettoCorsaEnv.ac_client.Client
        The underlying AC client that holds the DriverControls object.
        Used only by execute_direct() and neutralize().
    """

    def __init__(self, ac_client):
        self._client = ac_client

    def map_action(self, policy_action: np.ndarray) -> np.ndarray:
        """[0,1]³ → [-1,1]³."""
        return (np.asarray(policy_action, dtype=np.float32) * 2.0 - 1.0)

    def execute_direct(self, ac_action: np.ndarray) -> None:
        """
        Bypass env.step() and write directly to vJoy.

        Used by:
            - handle_crash(): brake for N seconds after going off-track
            - _neutralize_vjoy(): zero all inputs on exit
        """
        controls = self._client.controls
        controls.set_controls(
            steer=float(ac_action[0]),
            acc=float(ac_action[1]),
            brake=float(ac_action[2]),
        )
        controls.apply_local_controls()

    def neutralize(self) -> None:
        """Neutral: steer=0, acc=-1, brake=-1 (no input in AC space)."""
        self.execute_direct(np.array([0.0, -1.0, -1.0], dtype=np.float32))
        logger.debug("VJoyControl neutralized")
