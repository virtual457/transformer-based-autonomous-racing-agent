"""
vjoy_input.driver
-----------------
Low-level wrapper around the vJoy Windows DLL (vJoyInterface.dll).

Mirrors the JOYSTICK_POSITION struct exactly:
    bDevice  : BYTE   -- 1-based device index
    axes     : LONG * 18
    lButtons : LONG   -- 32-bit button mask
    bHats*   : DWORD * 4

Axis value range accepted by UpdateVJD: 0 -- 32768

All higher-level axis scaling lives in controls.py, NOT here.
"""

import ctypes
import struct

DLL_PATH = "C:\\Program Files\\vJoy\\x64\\vJoyInterface.dll"

# Struct format matches _JOYSTICK_POSITION (see vjoy SDK header)
_JOY_FMT = "BlllllllllllllllllllIIII"

# Button bitmasks
BUTTON_SHIFT_UP   = 0x00000001  # button 1
BUTTON_SHIFT_DOWN = 0x00000002  # button 2


class VJoyDriver:
    """
    Thin wrapper over the vJoy DLL.  Responsible for:
      - Acquiring / releasing the vJoy device
      - Packing the JOYSTICK_POSITION struct
      - Calling UpdateVJD to push state to the virtual device

    Usage:
        driver = VJoyDriver(device_id=1)
        driver.open()
        driver.update(wAxisX=16384, wAxisY=32768, wAxisZ=0)
        driver.close()
    """

    def __init__(self, device_id: int = 1, dll_path: str = DLL_PATH):
        self.device_id = device_id
        self.dll_path  = dll_path
        self._dll      = None
        self.acquired  = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Load DLL and acquire the vJoy device. Returns True on success."""
        try:
            self._dll = ctypes.CDLL(self.dll_path)
        except OSError as e:
            raise RuntimeError(
                f"Cannot load vJoy DLL at '{self.dll_path}'. "
                f"Is vJoy installed? Error: {e}"
            )

        if self._dll.AcquireVJD(self.device_id):
            self.acquired = True
            return True
        return False

    def close(self) -> bool:
        """Release the vJoy device. Returns True on success."""
        if self._dll is None:
            return True
        if self._dll.RelinquishVJD(self.device_id):
            self.acquired = False
            return True
        return False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        wAxisZ:    int = 16384,
        wAxisXRot: int = 16384,
        wAxisYRot: int = 16384,
        wAxisZRot: int = 16384,
        lButtons:  int = 0,
        wThrottle: int = 16384,
        wRudder:   int = 16384,
        wAileron:  int = 16384,
        wAxisX:    int = 16384,
        wAxisY:    int = 16384,
    ) -> bool:
        """
        Push the given axis values to the vJoy device.

        Active axes on Device 1: Z, Rx, Ry, Rz, Slider, Dial  (no X or Y).
        DirectInput index -> vJoy field:
            0 = Z    (wAxisZ)    -> Steer
            1 = Rx   (wAxisXRot) -> Throttle
            2 = Ry   (wAxisYRot) -> Brake

        All axis values must be in [0, 32768].
        lButtons is a 32-bit bitmask (BUTTON_SHIFT_UP / BUTTON_SHIFT_DOWN).

        Returns True if UpdateVJD succeeded.
        """
        if not self.acquired:
            raise RuntimeError("vJoy device not acquired. Call open() first.")

        packed = self._pack(
            wAxisX=wAxisX, wAxisY=wAxisY, wAxisZ=wAxisZ,
            wAxisXRot=wAxisXRot, wAxisYRot=wAxisYRot, wAxisZRot=wAxisZRot,
            lButtons=lButtons, wThrottle=wThrottle,
            wRudder=wRudder, wAileron=wAileron,
        )
        return bool(self._dll.UpdateVJD(self.device_id, packed))

    def reset(self) -> bool:
        """Send all axes to center."""
        return self.update(wAxisZ=16384, wAxisXRot=16384, wAxisYRot=16384, lButtons=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pack(
        self,
        wThrottle=16384, wRudder=16384, wAileron=16384,
        wAxisX=16384,    wAxisY=16384,  wAxisZ=16384,
        wAxisXRot=16384, wAxisYRot=16384, wAxisZRot=16384,
        wSlider=16384,   wDial=16384,   wWheel=16384,
        wAxisVX=16384,   wAxisVY=16384, wAxisVZ=16384,
        wAxisVBRX=16384, wAxisVBRY=16384, wAxisVBRZ=16384,
        lButtons=0, bHats=0, bHatsEx1=0, bHatsEx2=0, bHatsEx3=0,
    ) -> bytes:
        return struct.pack(
            _JOY_FMT,
            self.device_id,
            wThrottle, wRudder, wAileron,
            wAxisX, wAxisY, wAxisZ,
            wAxisXRot, wAxisYRot, wAxisZRot,
            wSlider, wDial, wWheel,
            wAxisVX, wAxisVY, wAxisVZ,
            wAxisVBRX, wAxisVBRY, wAxisVBRZ,
            lButtons, bHats, bHatsEx1, bHatsEx2, bHatsEx3,
        )
