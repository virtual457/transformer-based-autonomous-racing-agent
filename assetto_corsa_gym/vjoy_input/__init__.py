"""
vjoy_input
----------
Modular vJoy input package for the Assetto Corsa RL project.

Public API:
    JoystickSimulator  -- high-level: send normalized inputs to AC via vJoy
    VJoyDriver         -- low-level: direct DLL wrapper
    controls           -- axis conversion utilities (encode, neutral, steer_to_axis, ...)

Quick start:
    from vjoy_input import JoystickSimulator

    with JoystickSimulator() as sim:
        sim.send(steer=0.0, throttle=0.5, brake=-1.0)
"""

from vjoy_input.simulator import JoystickSimulator
from vjoy_input.driver    import VJoyDriver
from vjoy_input            import controls

__all__ = ["JoystickSimulator", "VJoyDriver", "controls"]
