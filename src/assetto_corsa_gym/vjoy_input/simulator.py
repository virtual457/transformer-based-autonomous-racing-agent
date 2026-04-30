"""
vjoy_input.simulator
---------------------
High-level joystick simulator.  This is the main entry point for all
code that wants to send inputs to Assetto Corsa via vJoy.

Usage (RL agent or test script):

    from vjoy_input import JoystickSimulator

    sim = JoystickSimulator()
    sim.open()

    sim.send(steer=0.0, throttle=0.5, brake=0.0)   # straight, half throttle
    sim.send(steer=-0.3, throttle=0.8, brake=0.0)  # left turn, heavy throttle
    sim.send_gear_shift(shift_up=True)               # tap shift up

    sim.reset()   # return to neutral
    sim.close()

Or as a context manager:

    with JoystickSimulator() as sim:
        sim.send(steer=0.1, throttle=0.3, brake=0.0)
"""

import time
import logging

from vjoy_input.driver   import VJoyDriver
from vjoy_input.controls import encode, neutral

logger = logging.getLogger(__name__)


class JoystickSimulator:
    """
    High-level API for sending normalized inputs to Assetto Corsa via vJoy.

    All inputs use the normalized range expected by the RL agent:
        steer    : [-1, +1]   left to right
        throttle : [ 0, +1]   0=released, 1=fully pressed
        brake    : [ 0, +1]   0=released, 1=fully pressed

    Throttle and brake share a single combined vJoy axis (wAxisZ).
    Full throttle drives the axis LOW (0); full brake drives it HIGH (32768);
    neither leaves it at center (16384).

    Gear shift buttons are handled separately via send_gear_shift().
    """

    def __init__(self, device_id: int = 1):
        self._driver = VJoyDriver(device_id=device_id)
        self._last   = neutral()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> "JoystickSimulator":
        """Acquire the vJoy device. Must be called before send()."""
        ok = self._driver.open()
        if not ok:
            raise RuntimeError(
                f"Failed to acquire vJoy device {self._driver.device_id}. "
                "Is the vJoy driver installed and Device 1 free (not held by AC or another process)?"
            )
        logger.info(f"vJoy device {self._driver.device_id} acquired")
        self.reset()
        return self

    def close(self):
        """Return to neutral and release the vJoy device."""
        try:
            self.reset()
        finally:
            self._driver.close()
            logger.info(f"vJoy device {self._driver.device_id} released")

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Input sending
    # ------------------------------------------------------------------

    def send(
        self,
        steer:    float = 0.0,
        throttle: float = 0.0,
        brake:    float = 0.0,
    ) -> dict:
        """
        Send driving inputs to vJoy.

        Args:
            steer:    [-1, +1]  — negative=left, positive=right
            throttle: [ 0, +1]  — 0=released, 1=full throttle
            brake:    [ 0, +1]  — 0=released, 1=full brake

        Throttle and brake share a single combined axis (wAxisZ).
        Full throttle = axis LOW (0); full brake = axis HIGH (32768).

        Returns:
            dict of raw axis values sent (useful for logging/testing)
        """
        axes = encode(steer=steer, throttle=throttle, brake=brake)
        self._driver.update(**axes)
        self._last = axes
        return axes

    def send_gear_shift(
        self,
        shift_up:   bool = False,
        shift_down: bool = False,
        hold_s:     float = 0.05,
    ):
        """
        Tap a gear shift button.

        Sends the button press, waits hold_s seconds, then releases.
        Keeps the current steering/pedal state unchanged during the shift.
        """
        from vjoy_input.controls import buttons_mask

        press_axes = dict(self._last)
        press_axes["lButtons"] = buttons_mask(shift_up=shift_up, shift_down=shift_down)
        self._driver.update(**press_axes)

        time.sleep(hold_s)

        release_axes = dict(self._last)
        release_axes["lButtons"] = 0
        self._driver.update(**release_axes)

    def reset(self):
        """Send neutral state (wheels straight, all pedals released)."""
        axes = neutral()
        self._driver.update(**axes)
        self._last = axes

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._driver.acquired

    @property
    def last_axes(self) -> dict:
        """Raw vJoy axis values from the last send() call."""
        return dict(self._last)
