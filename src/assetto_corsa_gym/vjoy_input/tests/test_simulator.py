"""
test_simulator.py
-----------------
Integration tests for JoystickSimulator (full pipeline: normalize -> convert -> send to vJoy).
Requires the vJoy device to be available.

Run:
    cd assetto_corsa_gym
    python -m pytest vjoy_input/tests/test_simulator.py -v
    python -m pytest vjoy_input/tests/test_simulator.py -v -k "not device"  # offline only
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from vjoy_input import JoystickSimulator
from vjoy_input.controls import AXIS_MAX, AXIS_MID

DLL_PATH = "C:\\Program Files\\vJoy\\x64\\vJoyInterface.dll"


def dll_available() -> bool:
    return os.path.exists(DLL_PATH)


device = pytest.mark.skipif(
    not dll_available(),
    reason=f"vJoy DLL not found at {DLL_PATH}"
)


# ------------------------------------------------------------------
# Instantiation (offline — no device needed)
# ------------------------------------------------------------------

class TestSimulatorOffline:
    def test_instantiates(self):
        sim = JoystickSimulator(device_id=1)
        assert not sim.is_open

    def test_last_axes_before_open(self):
        sim = JoystickSimulator(device_id=1)
        # last_axes is pre-populated with neutral even before open
        axes = sim.last_axes
        assert "wAxisZ" in axes
        assert axes["wAxisZ"] == AXIS_MID  # neutral steer (Z axis)


# ------------------------------------------------------------------
# Lifecycle (device required)
# ------------------------------------------------------------------

class TestSimulatorLifecycle:
    @device
    def test_open_close(self):
        sim = JoystickSimulator()
        sim.open()
        assert sim.is_open
        sim.close()
        assert not sim.is_open

    @device
    def test_context_manager(self):
        with JoystickSimulator() as sim:
            assert sim.is_open
        assert not sim.is_open

    @device
    def test_open_returns_self(self):
        sim = JoystickSimulator()
        result = sim.open()
        assert result is sim
        sim.close()


# ------------------------------------------------------------------
# send() (device required)
# ------------------------------------------------------------------

class TestSimulatorSend:
    @device
    def test_send_neutral(self):
        with JoystickSimulator() as sim:
            axes = sim.send(steer=0.0, throttle=-1.0, brake=-1.0)
            assert axes["wAxisZ"]    == AXIS_MID  # Z = steer center
            assert axes["wAxisXRot"] == 0          # Rx = throttle released
            assert axes["wAxisYRot"] == 0          # Ry = brake released

    @device
    def test_send_full_throttle(self):
        with JoystickSimulator() as sim:
            axes = sim.send(steer=0.0, throttle=1.0, brake=-1.0)
            assert axes["wAxisXRot"] == AXIS_MAX   # Rx = throttle

    @device
    def test_send_full_brake(self):
        with JoystickSimulator() as sim:
            axes = sim.send(steer=0.0, throttle=-1.0, brake=1.0)
            assert axes["wAxisYRot"] == AXIS_MAX   # Ry = brake

    @device
    def test_send_steer_left(self):
        with JoystickSimulator() as sim:
            axes = sim.send(steer=-1.0)
            assert axes["wAxisZ"] == 0             # Z = steer full left

    @device
    def test_send_steer_right(self):
        with JoystickSimulator() as sim:
            axes = sim.send(steer=1.0)
            assert axes["wAxisZ"] == AXIS_MAX      # Z = steer full right

    @device
    def test_send_updates_last_axes(self):
        with JoystickSimulator() as sim:
            sim.send(steer=0.5, throttle=0.8, brake=-1.0)
            axes = sim.last_axes
            assert axes["wAxisZ"] > AXIS_MID      # right of center

    @device
    def test_send_sequence(self):
        """Simulate a short driving sequence without errors."""
        inputs = [
            (0.0,  0.5, -1.0),
            (0.1,  0.7, -1.0),
            (0.2,  0.8, -1.0),
            (0.0,  0.3,  0.3),   # lift and brake
            (-0.1, 0.5, -1.0),
            (0.0,  0.0, -1.0),
        ]
        with JoystickSimulator() as sim:
            for steer, throttle, brake in inputs:
                axes = sim.send(steer=steer, throttle=throttle, brake=brake)
                assert all(0 <= v <= AXIS_MAX for k, v in axes.items() if k != "lButtons")
                time.sleep(0.01)


# ------------------------------------------------------------------
# Gear shift (device required)
# ------------------------------------------------------------------

class TestGearShift:
    @device
    def test_shift_up(self):
        with JoystickSimulator() as sim:
            sim.send(steer=0.0, throttle=0.5, brake=-1.0)
            sim.send_gear_shift(shift_up=True, hold_s=0.03)
            # After release, last_axes should have lButtons=0
            assert sim.last_axes["lButtons"] == 0

    @device
    def test_shift_down(self):
        with JoystickSimulator() as sim:
            sim.send_gear_shift(shift_down=True, hold_s=0.03)
            assert sim.last_axes["lButtons"] == 0


# ------------------------------------------------------------------
# reset() (device required)
# ------------------------------------------------------------------

class TestReset:
    @device
    def test_reset_returns_to_neutral(self):
        with JoystickSimulator() as sim:
            sim.send(steer=1.0, throttle=1.0, brake=1.0)
            sim.reset()
            axes = sim.last_axes
            assert axes["wAxisZ"]    == AXIS_MID  # steer center
            assert axes["wAxisXRot"] == 0          # throttle released
            assert axes["wAxisYRot"] == 0          # brake released
            assert axes["lButtons"]  == 0


# ------------------------------------------------------------------
# Manual smoke test (run directly, not via pytest)
# ------------------------------------------------------------------

def run_smoke_test():
    """
    Visual smoke test: open a vJoy monitor (vJoyConf.exe or Joy.cpl)
    and watch the axes move through this sequence.
    """
    print("\n[smoke test] Opening vJoy device 1...")
    with JoystickSimulator() as sim:
        print("[smoke test] Sending: steer=0, throttle=0, brake=released")
        sim.send(steer=0.0, throttle=0.0, brake=-1.0)
        time.sleep(0.5)

        print("[smoke test] Steering left...")
        sim.send(steer=-1.0, throttle=0.3, brake=-1.0)
        time.sleep(0.5)

        print("[smoke test] Steering right...")
        sim.send(steer=1.0, throttle=0.3, brake=-1.0)
        time.sleep(0.5)

        print("[smoke test] Full throttle, straight...")
        sim.send(steer=0.0, throttle=1.0, brake=-1.0)
        time.sleep(0.5)

        print("[smoke test] Full brake...")
        sim.send(steer=0.0, throttle=-1.0, brake=1.0)
        time.sleep(0.5)

        print("[smoke test] Shift up...")
        sim.send(steer=0.0, throttle=0.5, brake=-1.0)
        sim.send_gear_shift(shift_up=True)
        time.sleep(0.5)

        print("[smoke test] Resetting to neutral...")
        sim.reset()
        time.sleep(0.3)

    print("[smoke test] DONE -- device released")


if __name__ == "__main__":
    run_smoke_test()
