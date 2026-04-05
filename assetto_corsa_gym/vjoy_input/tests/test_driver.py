"""
test_driver.py
--------------
Tests for the VJoyDriver (low-level DLL wrapper).
Requires the vJoy device to be available (Device 1, X/Y/Z axes configured).

Run:
    cd assetto_corsa_gym
    python -m pytest vjoy_input/tests/test_driver.py -v

Skip if no device:
    python -m pytest vjoy_input/tests/test_driver.py -v -m "not device"
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from vjoy_input.driver import VJoyDriver, BUTTON_SHIFT_UP, BUTTON_SHIFT_DOWN

DLL_PATH = "C:\\Program Files\\vJoy\\x64\\vJoyInterface.dll"


def dll_available() -> bool:
    return os.path.exists(DLL_PATH)


# Skip all device tests if DLL is not present
device = pytest.mark.skipif(
    not dll_available(),
    reason=f"vJoy DLL not found at {DLL_PATH}"
)


# ------------------------------------------------------------------
# DLL loading
# ------------------------------------------------------------------

class TestDLLLoading:
    def test_dll_exists(self):
        assert os.path.exists(DLL_PATH), (
            f"vJoy DLL missing: {DLL_PATH}\n"
            "Install vJoy from https://github.com/shauleiz/vJoy"
        )

    @device
    def test_driver_instantiates(self):
        driver = VJoyDriver(device_id=1)
        assert driver.device_id == 1
        assert not driver.acquired


# ------------------------------------------------------------------
# Acquire / release
# ------------------------------------------------------------------

class TestAcquireRelease:
    @device
    def test_open_and_close(self):
        driver = VJoyDriver(device_id=1)
        assert driver.open(), "Failed to acquire vJoy device 1"
        assert driver.acquired
        assert driver.close()
        assert not driver.acquired

    @device
    def test_context_manager(self):
        with VJoyDriver(device_id=1) as driver:
            assert driver.acquired
        assert not driver.acquired

    @device
    def test_double_close_safe(self):
        driver = VJoyDriver(device_id=1)
        driver.open()
        driver.close()
        driver.close()  # second close should not raise


# ------------------------------------------------------------------
# Update (send to device)
# ------------------------------------------------------------------

class TestUpdate:
    @device
    def test_update_neutral(self):
        with VJoyDriver(device_id=1) as driver:
            ok = driver.update(wAxisX=16384, wAxisY=0, wAxisZ=0, lButtons=0)
            assert ok, "UpdateVJD returned False"

    @device
    def test_update_full_steer_left(self):
        with VJoyDriver(device_id=1) as driver:
            ok = driver.update(wAxisX=0, wAxisY=0, wAxisZ=0)
            assert ok

    @device
    def test_update_full_steer_right(self):
        with VJoyDriver(device_id=1) as driver:
            ok = driver.update(wAxisX=32768, wAxisY=0, wAxisZ=0)
            assert ok

    @device
    def test_update_full_throttle(self):
        with VJoyDriver(device_id=1) as driver:
            ok = driver.update(wAxisX=16384, wAxisY=32768, wAxisZ=0)
            assert ok

    @device
    def test_update_full_brake(self):
        with VJoyDriver(device_id=1) as driver:
            ok = driver.update(wAxisX=16384, wAxisY=0, wAxisZ=32768)
            assert ok

    @device
    def test_update_shift_up_button(self):
        with VJoyDriver(device_id=1) as driver:
            ok = driver.update(wAxisX=16384, wAxisY=16384, wAxisZ=0, lButtons=BUTTON_SHIFT_UP)
            assert ok

    @device
    def test_reset(self):
        with VJoyDriver(device_id=1) as driver:
            driver.update(wAxisX=32768, wAxisY=32768, wAxisZ=32768)
            ok = driver.reset()
            assert ok

    @device
    def test_update_without_open_raises(self):
        driver = VJoyDriver(device_id=1)
        with pytest.raises(RuntimeError, match="not acquired"):
            driver.update(wAxisX=16384)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
