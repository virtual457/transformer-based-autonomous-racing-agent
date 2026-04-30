"""
test_ac_lifecycle.py — Test programmatic AC launch, session config, and shutdown.

Tests (run independently, each prints PASS/FAIL):
    1. write_session_config   — write race.ini + launcher.ini for monza/ks_mazda_miata
    2. kill_ac                — kill acs.exe + AssettoCorsa.exe if running
    3. launch_ac              — launch AC via Steam and wait until plugin port responds
    4. launch_ac_cm           — Content Manager URI launch, wait for plugin, dismiss
                                session info screen, confirm car on track
    5. dismiss_session_screen — standalone: send reset via TCP 2347, poll shared memory
                                until status=AC_LIVE and isInPit=0
    6. full_cycle             — write config → kill → CM launch → dismiss → on track

Usage:
    # Run all tests:
    .\\AssetoCorsa\\Scripts\\python.exe tests/test_ac_lifecycle.py

    # Run a specific test:
    .\\AssetoCorsa\\Scripts\\python.exe tests/test_ac_lifecycle.py --test kill_ac
    .\\AssetoCorsa\\Scripts\\python.exe tests/test_ac_lifecycle.py --test launch_ac_cm
    .\\AssetoCorsa\\Scripts\\python.exe tests/test_ac_lifecycle.py --test dismiss_session_screen
    .\\AssetoCorsa\\Scripts\\python.exe tests/test_ac_lifecycle.py --test write_session_config
    .\\AssetoCorsa\\Scripts\\python.exe tests/test_ac_lifecycle.py --test full_cycle

Notes:
    - launch_ac and full_cycle require Steam to be running.
    - AC takes 30-90 seconds to fully load; the wait_for_plugin test polls with a timeout.
    - kill_ac is safe to run even if AC is not running (no-op).
    - dismiss_session_screen uses AC shared memory (acpmf_graphics mmap) — no pywin32 needed.
"""

import argparse
import configparser
import ctypes
import json
import mmap
import socket
import subprocess
import sys
import time
import urllib.parse
from ctypes import c_int32, c_float, c_wchar
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

STEAM_EXE         = Path(r"C:\Program Files (x86)\Steam\steam.exe")
AC_APP_ID         = "244210"
AC_PROCESSES      = ["acs.exe", "AssettoCorsa.exe", "Content Manager.exe"]
CM_EXE            = Path(r"C:\Users\chand\Downloads\content-manager\Content Manager.exe")

AC_RACE_INI       = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\cfg\race.ini")
AC_LAUNCHER_INI   = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\cfg\launcher.ini")
AC_ASSISTS_INI    = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\cfg\assists.ini")

# Target session
TARGET_TRACK      = "monza"
TARGET_CAR        = "ks_mazda_miata"
TARGET_DRIVE_MODE = "hotlap"       # Hotlap/Time Attack — no AI, no race start
TARGET_TIME_OF_DAY = 54000         # Seconds from midnight (0=00:00, 43200=12:00, 54000=15:00, 64800=18:00)
                                   # CM QuickDrive SaveableData.Time uses SECONDS, not minutes.
                                   # 54000 = 15 * 3600 = 15:00 (afternoon).
                                   # sun_angle formula: 16.0 * (seconds - 46800) / 3600
                                   #   → 16 * (54000-46800)/3600 = 32  (matches race.ini [LIGHTING] sun_angle)

# ── Driving aids (written to assists.ini before every session) ─────────────────
# Field names come directly from cfg/assists.ini — do not rename them.
#
# AID_AUTO_GEARBOX / AID_AUTO_CLUTCH:
#   Written as AUTO_SHIFTER / AUTO_CLUTCH respectively.
#   0 = manual, 1 = automatic.
#
# AID_TC / AID_ABS:
#   Written as TRACTION_CONTROL / ABS.
#   0 = off, 1 = factory (permitted by car), 2 = maximum (forced on regardless of car).
#   Use 2 for maximum reliability during RL training.
#
# AID_STABILITY:
#   Written as STABILITY_CONTROL.  Integer 0–100 (percentage).
#
# AID_FUEL_RATE:
#   Written as FUEL_RATE.  0 = fuel consumption off, 1 = normal consumption.
#   Set 0 so the car never runs out of fuel mid-episode.
#
# AID_DAMAGE:
#   Written as DAMAGE.  0 = mechanical damage off, 1 = normal damage.
#   Set 0 so crash contacts do not degrade car performance during training.
AID_AUTO_GEARBOX = 1   # AUTO_SHIFTER:       0=manual, 1=automatic
AID_AUTO_CLUTCH  = 1   # AUTO_CLUTCH:        0=manual, 1=automatic
AID_TC           = 2   # TRACTION_CONTROL:   0=off, 1=factory, 2=max
AID_ABS          = 2   # ABS:                0=off, 1=factory, 2=max
AID_STABILITY    = 100 # STABILITY_CONTROL:  0-100 %
AID_FUEL_RATE    = 0   # FUEL_RATE:          0=off, 1=normal consumption
AID_DAMAGE       = 0   # DAMAGE:             0=off, 1=normal mechanical damage

# Plugin readiness
PLUGIN_HOST       = "127.0.0.1"
PLUGIN_EGO_PORT   = 2345           # ego_server (UDP)
PLUGIN_MGMT_PORT  = 2347           # simulation_management_server (TCP)
PLUGIN_READY_TIMEOUT = 120         # seconds to wait for plugin after launch
PLUGIN_POLL_INTERVAL = 3           # seconds between readiness polls

# AC shared memory status codes (from sim_info.py in the plugin)
AC_OFF    = 0   # simulation off / loading
AC_REPLAY = 1   # replay mode
AC_LIVE   = 2   # session running, car on track
AC_PAUSE  = 3   # paused

# Timeout and poll interval for dismissing the session info screen
DISMISS_TIMEOUT      = 30    # seconds to wait for car to leave pit after reset
DISMISS_POLL_INTERVAL = 1    # seconds between polls


# ── AC shared memory helpers ──────────────────────────────────────────────────
# These ctypes structures mirror SPageFileGraphic in sim_info.py.
# We only need the first few fields; the rest are declared as padding so the
# struct size and offsets stay correct.

class _SPageFileGraphic(ctypes.Structure):
    """Minimal mirror of AC's acpmf_graphics shared memory page."""
    _pack_ = 4
    _fields_ = [
        ("packetId",          c_int32),
        ("status",            c_int32),   # AC_OFF/REPLAY/LIVE/PAUSE
        ("session",           c_int32),
        ("currentTime",       c_wchar * 15),
        ("lastTime",          c_wchar * 15),
        ("bestTime",          c_wchar * 15),
        ("split",             c_wchar * 15),
        ("completedLaps",     c_int32),
        ("position",          c_int32),
        ("iCurrentTime",      c_int32),
        ("iLastTime",         c_int32),
        ("iBestTime",         c_int32),
        ("sessionTimeLeft",   c_float),
        ("distanceTraveled",  c_float),
        ("isInPit",           c_int32),   # 1 = car is in pit box
    ]


def read_ac_graphics() -> dict | None:
    """
    Open AC's acpmf_graphics named shared memory and return a dict with
    ``status`` (int, AC_OFF/REPLAY/LIVE/PAUSE) and ``isInPit`` (int, 0/1).

    Returns None if the shared memory is not yet available (AC not running
    or still loading).
    """
    try:
        mm = mmap.mmap(0, ctypes.sizeof(_SPageFileGraphic),
                       "acpmf_graphics", access=mmap.ACCESS_READ)
        gfx = _SPageFileGraphic.from_buffer_copy(mm)
        mm.close()
        return {"status": gfx.status, "isInPit": gfx.isInPit}
    except Exception:
        return None


def send_management_reset(timeout: float = 3.0) -> bool:
    """
    Send the ``reset`` command to the management server on TCP port 2347.
    This triggers ``ac.ext_resetCar()`` inside the AC plugin, which
    repositions the car to the start line and dismisses the session info
    overlay.

    Returns True if the command was sent successfully, False on error.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((PLUGIN_HOST, PLUGIN_MGMT_PORT))
            s.sendall(b"reset")
        return True
    except Exception as exc:
        _info(f"  send_management_reset failed: {exc}")
        return False


# ── Win32 keypress + mouse helpers (ctypes stdlib only, no pywin32) ───────────
#
# These functions find the AC window and send input events to dismiss the
# session info overlay that appears when a hotlap/practice session first loads.
#
# AC shows "HOTLAP 30'00" / Restart Session / Skip Session / Exit" on first
# load.  The overlay can be dismissed by:
#   - Pressing Enter (selects the focused button, usually "Skip Session")
#   - Pressing Space (same effect)
#   - The plugin's ac.ext_resetCar() call (sent via TCP 2347 "reset")
#   - A left mouse click outside the session info panel (right half of window)
#
# We try all mechanisms in order: TCP reset → keypresses → mouse click.

_user32 = ctypes.windll.user32

# Windows virtual key codes
_VK_RETURN = 0x0D   # Enter
_VK_SPACE  = 0x20   # Space
_VK_ESCAPE = 0x1B   # ESC

# WM_KEYDOWN / WM_KEYUP constants for PostMessageW
_WM_KEYDOWN = 0x0100
_WM_KEYUP   = 0x0101

# mouse_event flags for left button click (used with SetCursorPos + mouse_event)
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004

# ShowWindow commands
_SW_RESTORE = 9

# RECT structure for GetWindowRect
class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

# AC window titles to search for (AC uses different titles depending on version)
_AC_WINDOW_TITLES = [
    "Assetto Corsa",
    "AC2",
]


def find_ac_window() -> int:
    """
    Return the HWND of the AC rendering window, or 0 if not found.

    Tries each entry in _AC_WINDOW_TITLES with FindWindowW.  AC's main
    rendering window is typically titled "Assetto Corsa".
    """
    for title in _AC_WINDOW_TITLES:
        hwnd = _user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
    return 0


def _post_key(hwnd: int, vk: int) -> None:
    """
    Post WM_KEYDOWN + WM_KEYUP for the given virtual key to hwnd.

    Using PostMessageW avoids requiring the window to be in the foreground.
    The lParam for WM_KEYDOWN is 0x00000001 (repeat count=1, all other bits 0).
    The lParam for WM_KEYUP   is 0xC0000001 (transition + previous state bits).
    """
    _user32.PostMessageW(hwnd, _WM_KEYDOWN, vk, 0x00000001)
    time.sleep(0.05)
    _user32.PostMessageW(hwnd, _WM_KEYUP,   vk, 0xC0000001)


def send_keypress_to_ac(vk: int = _VK_RETURN) -> bool:
    """
    Send a keypress to the AC window using PostMessageW (no focus required).

    Returns True if the AC window was found and the key was posted, False
    if the AC window could not be located.
    """
    hwnd = find_ac_window()
    if not hwnd:
        _info("  send_keypress_to_ac: AC window not found (FindWindowW returned 0).")
        return False
    _info(f"  Sending VK=0x{vk:02X} to AC hwnd={hwnd:#010x} via PostMessageW ...")
    _post_key(hwnd, vk)
    return True


def send_mouse_click_to_ac() -> bool:
    """
    Bring the AC window to the foreground and send a left mouse click to the
    right half of the window (away from the session info panel which sits on
    the left side of the screen).

    Uses only ctypes.windll.user32 (stdlib — no pywin32, no pyautogui):
      1. FindWindowW(None, "Assetto Corsa") — locate the window
      2. GetWindowRect(hwnd, rect)          — get screen coordinates
      3. ShowWindow(hwnd, SW_RESTORE)       — unminimize if needed
      4. SetForegroundWindow(hwnd)          — bring to front
      5. SetCursorPos(x, y)                 — move cursor to right-half center
      6. mouse_event(LEFTDOWN | LEFTUP)     — fire a click at current cursor pos

    The click target is the horizontal midpoint between the window centre and
    the right edge, at the vertical centre — well clear of the left-side panel.

    Returns True if the window was found and the click was dispatched.
    Returns False if the AC window could not be located.
    """
    hwnd = find_ac_window()
    if not hwnd:
        _info("  send_mouse_click_to_ac: AC window not found (FindWindowW returned 0).")
        return False

    rect = _RECT()
    _user32.GetWindowRect(hwnd, ctypes.byref(rect))
    win_w = rect.right  - rect.left
    win_h = rect.bottom - rect.top

    # Click the steering wheel / Drive icon on the left sidebar.
    # Captured via capture_mouse_coords.py: frac=(0.047, 0.195)
    click_x = rect.left + int(win_w * 0.047)
    click_y = rect.top  + int(win_h * 0.195)

    _info(
        f"  send_mouse_click_to_ac: hwnd={hwnd:#010x}  "
        f"window=({rect.left},{rect.top})-({rect.right},{rect.bottom})  "
        f"click=({click_x},{click_y})"
    )

    # Unminimize and bring to foreground so mouse_event is delivered correctly
    _user32.ShowWindow(hwnd, _SW_RESTORE)
    _user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)   # wait for focus to settle before clicking

    _user32.SetCursorPos(click_x, click_y)
    for i in range(3):
        _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.05)
        _user32.mouse_event(_MOUSEEVENTF_LEFTUP,   0, 0, 0, 0)
        _info(f"  Click {i+1}/3 at ({click_x}, {click_y})")
        time.sleep(1.0)
    return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pass(msg: str):
    print(f"  [PASS] {msg}")

def _fail(msg: str):
    print(f"  [FAIL] {msg}")

def _info(msg: str):
    print(f"  [INFO] {msg}")


def is_ac_running() -> bool:
    """Return True if any AC_PROCESSES entry is in the process list (wmic, handles spaces)."""
    for name in AC_PROCESSES:
        result = subprocess.run(
            f"wmic process where \"name='{name}'\" get name",
            shell=True, capture_output=True, text=True
        )
        if name.lower() in result.stdout.lower():
            return True
    return False


def is_plugin_ready(port: int = PLUGIN_EGO_PORT, timeout: float = 2.0) -> bool:
    """
    Try to connect to the plugin ego server (UDP) or management server (TCP).
    Returns True if the port responds within timeout seconds.
    """
    if port == PLUGIN_EGO_PORT:
        # UDP — send connect, wait for any response
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            sock.sendto(b"connect", (PLUGIN_HOST, port))
            data, _ = sock.recvfrom(4096)
            sock.sendto(b"disconnect", (PLUGIN_HOST, port))
            sock.close()
            return len(data) > 0
        except Exception:
            return False
    else:
        # TCP — try to connect
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((PLUGIN_HOST, port))
            sock.close()
            return True
        except Exception:
            return False


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_dismiss_session_screen(timeout: float = DISMISS_TIMEOUT) -> bool:
    """
    Dismiss the AC session info / pit lane overlay and confirm the car is
    on track ready to drive.

    AC shows this overlay after the session loads — the car sits in the pit
    box with the door open before the driver presses "Skip Session" or
    "Restart Session".

    Strategy (three-mechanism, automatic fallback)
    -----------------------------------------------
    Attempt 1 — TCP reset (preferred):
        Send ``reset`` to management server (TCP 2347) → ``ac.ext_resetCar()``
        inside the plugin.  This is the cleanest path: no window focus needed,
        works headlessly.  Poll ``isInPit`` for up to 5s.

    Attempt 2 — Keypress fallback (if TCP reset alone doesn't move isInPit):
        Use ``ctypes.windll.user32.PostMessageW`` to post WM_KEYDOWN/WM_KEYUP
        to the AC window handle (found via ``FindWindowW``).  PostMessageW does
        NOT require the window to be in the foreground.  Key sequence tried:
          Enter → poll 3s → Space → poll 3s → ESC → poll 3s

        Enter typically selects the focused button ("Skip Session") on the
        overlay, which is equivalent to the user clicking it.

    Attempt 3 — Mouse click fallback (if keypresses are also ignored):
        AC can filter WM_KEYDOWN messages when it is not the foreground window.
        As a last resort, bring the window to the foreground via
        ``SetForegroundWindow`` + ``ShowWindow(SW_RESTORE)``, wait 0.5s for
        focus to settle, then move the cursor to the right half of the AC
        window (away from the session info panel on the left) using
        ``SetCursorPos`` and fire a click via ``mouse_event(LEFTDOWN|LEFTUP)``.
        Poll for 1s after the click.

    Up to ``DISMISS_RETRY_COUNT`` full retry cycles (TCP reset + keypresses +
    mouse click) are attempted before giving up.

    Detection
    ---------
    Reads AC's ``acpmf_graphics`` named shared memory via ``mmap`` + ``ctypes``
    (pure Python stdlib, no pywin32):

    * ``graphics.status``   — AC_LIVE (2) when the session is running
    * ``graphics.isInPit``  — 1 while in the pit box, 0 on track

    Returns True when ``status == AC_LIVE`` **and** ``isInPit == 0``.
    Returns False on timeout or if the management port is not reachable.
    """
    print("\n=== test_dismiss_session_screen ===")

    # ── Verify management port is reachable ───────────────────────────────────
    if not is_plugin_ready(PLUGIN_MGMT_PORT):
        _fail(
            f"Management server (TCP {PLUGIN_MGMT_PORT}) not responding. "
            "Run test_launch_ac_cm first."
        )
        return False

    # ── Read current shared memory state ─────────────────────────────────────
    gfx = read_ac_graphics()
    if gfx is None:
        _fail(
            "Cannot open acpmf_graphics shared memory. "
            "AC must be running and past the loading screen."
        )
        return False

    _info(
        f"Before dismiss: status={gfx['status']} "
        f"({'AC_LIVE' if gfx['status'] == AC_LIVE else 'OTHER'})  "
        f"isInPit={gfx['isInPit']}"
    )


    def _poll_on_track(poll_duration: float, label: str) -> bool:
        """Poll shared memory for poll_duration seconds. Return True if on track."""
        deadline = time.time() + poll_duration
        while time.time() < deadline:
            g = read_ac_graphics()
            elapsed_total = time.time() - t_start
            if g is None:
                _info(f"  {elapsed_total:.1f}s [{label}] — shared memory not readable")
                time.sleep(DISMISS_POLL_INTERVAL)
                continue
            _info(
                f"  {elapsed_total:.1f}s [{label}] — "
                f"status={g['status']}  isInPit={g['isInPit']}"
            )
            if g["status"] == AC_LIVE and g["isInPit"] == 0:
                return True
            time.sleep(DISMISS_POLL_INTERVAL)
        return False

    DISMISS_RETRY_COUNT  = 3   # full cycles before giving up
    TCP_POLL_SECS        = 5   # seconds to poll after TCP reset before trying keys
    KEY_POLL_SECS        = 3   # seconds to poll after each keypress

    t_start = time.time()

    for attempt in range(1, DISMISS_RETRY_COUNT + 1):
        elapsed = time.time() - t_start
        _info(f"Attempt {attempt}/{DISMISS_RETRY_COUNT} (elapsed={elapsed:.1f}s)")

        # ── Step A: TCP reset ─────────────────────────────────────────────────
        _info(f"  [A] Sending reset to TCP {PLUGIN_MGMT_PORT} ...")
        if send_management_reset():
            _info("  [A] Reset sent.")
        else:
            _info("  [A] TCP reset failed — continuing to mouse click.")

        # ── Step B: Mouse click — always run ─────────────────────────────────
        # Always simulate the Drive (steering wheel) icon click regardless of
        # TCP reset result. Clicks 3 times with 1s delay to ensure AC registers it.
        _info("  [B] Simulating Drive icon click (steering wheel, left sidebar) ...")
        send_mouse_click_to_ac()

        # ── Poll for on-track confirmation ────────────────────────────────────
        if _poll_on_track(TCP_POLL_SECS, "mouse-click"):
            elapsed = time.time() - t_start
            _pass(f"Car is on track after {elapsed:.1f}s (status=AC_LIVE, isInPit=0)")
            return True

        # ── Check total timeout ───────────────────────────────────────────────
        if time.time() - t_start >= timeout:
            break

    gfx = read_ac_graphics()
    _fail(
        f"Car did not leave the pit within {time.time() - t_start:.1f}s. "
        f"Last state: status={gfx['status'] if gfx else 'N/A'}  "
        f"isInPit={gfx['isInPit'] if gfx else 'N/A'}. "
        "Check that the AC plugin (sensors_par) is installed and loaded."
    )
    return False


def test_write_session_config():
    """
    Write race.ini, launcher.ini, and assists.ini to configure the session.

    race.ini    — track, car, time of day (sun_angle)
    launcher.ini — drive mode (hotlap/timeattack/practice)
    assists.ini  — all driving aids (auto gearbox, auto clutch, TC, ABS,
                   stability control, fuel rate, mechanical damage)

    Driving aids are NOT passed through the acmanager:// URI preset JSON.
    CM's SaveableData carries an AssistsData blob (separate serialised string)
    that is only applied when CM writes it to assists.ini before launching
    acs.exe.  Writing assists.ini directly here guarantees the values are
    set correctly regardless of which launch path is used (CM URI, direct
    acs.exe, or Steam applaunch).

    Reads back and verifies every written value.
    """
    print("\n=== test_write_session_config ===")

    # ── Write race.ini ────────────────────────────────────────────────────────
    if not AC_RACE_INI.exists():
        _fail(f"race.ini not found at {AC_RACE_INI}")
        return

    race_cfg = configparser.ConfigParser()
    race_cfg.read(str(AC_RACE_INI))

    if "RACE" not in race_cfg:
        race_cfg["RACE"] = {}

    race_cfg["RACE"]["TRACK"]        = TARGET_TRACK
    race_cfg["RACE"]["CONFIG_TRACK"] = ""
    race_cfg["RACE"]["MODEL"]        = TARGET_CAR

    # Time of day via sun_angle in [LIGHTING].
    # Formula (from CM source Game.Properties.cs GetSunAngle):
    #   sun_angle = 16.0 * (seconds_from_midnight - 46800) / 3600
    # Reference points: 13:00→0, 14:00→16, 15:00→32, 18:00→80, 08:00→-80
    _sun_angle = int(16.0 * (TARGET_TIME_OF_DAY - 46800) / 3600)
    if "LIGHTING" not in race_cfg:
        race_cfg["LIGHTING"] = {}
    race_cfg["LIGHTING"]["SUN_ANGLE"] = str(_sun_angle)

    with open(str(AC_RACE_INI), "w") as f:
        race_cfg.write(f)

    _target_hhmm = f"{TARGET_TIME_OF_DAY // 3600:02d}:{(TARGET_TIME_OF_DAY % 3600) // 60:02d}"
    _info(f"Written race.ini: TRACK={TARGET_TRACK}  MODEL={TARGET_CAR}  SUN_ANGLE={_sun_angle} ({_target_hhmm})")

    # ── Write launcher.ini ────────────────────────────────────────────────────
    if not AC_LAUNCHER_INI.exists():
        _fail(f"launcher.ini not found at {AC_LAUNCHER_INI}")
        return

    launcher_cfg = configparser.ConfigParser()
    launcher_cfg.read(str(AC_LAUNCHER_INI))

    if "SAVED" not in launcher_cfg:
        launcher_cfg["SAVED"] = {}

    launcher_cfg["SAVED"]["DRIVE"] = TARGET_DRIVE_MODE

    with open(str(AC_LAUNCHER_INI), "w") as f:
        launcher_cfg.write(f)
    _info(f"Written launcher.ini: DRIVE={TARGET_DRIVE_MODE}")

    # ── Write assists.ini ─────────────────────────────────────────────────────
    # AC reads assists.ini from cfg/ on session start.  We write every field we
    # care about explicitly; all other existing fields (IDEAL_LINE, AUTO_BLIP,
    # TYRE_WEAR, etc.) are preserved by reading the file first.
    if not AC_ASSISTS_INI.exists():
        _fail(f"assists.ini not found at {AC_ASSISTS_INI}")
        return

    # configparser lowercases keys by default — use RawConfigParser with
    # optionxform=str to preserve the original UPPER_CASE key names that AC expects.
    assists_cfg = configparser.RawConfigParser()
    assists_cfg.optionxform = str
    assists_cfg.read(str(AC_ASSISTS_INI))

    if "ASSISTS" not in assists_cfg:
        assists_cfg["ASSISTS"] = {}

    assists_cfg["ASSISTS"]["AUTO_SHIFTER"]       = str(AID_AUTO_GEARBOX)
    assists_cfg["ASSISTS"]["AUTO_CLUTCH"]        = str(AID_AUTO_CLUTCH)
    assists_cfg["ASSISTS"]["TRACTION_CONTROL"]   = str(AID_TC)
    assists_cfg["ASSISTS"]["ABS"]                = str(AID_ABS)
    assists_cfg["ASSISTS"]["STABILITY_CONTROL"]  = str(AID_STABILITY)
    assists_cfg["ASSISTS"]["FUEL_RATE"]          = str(AID_FUEL_RATE)
    assists_cfg["ASSISTS"]["DAMAGE"]             = str(AID_DAMAGE)

    with open(str(AC_ASSISTS_INI), "w") as f:
        assists_cfg.write(f)
    _info(
        f"Written assists.ini: AUTO_SHIFTER={AID_AUTO_GEARBOX}  AUTO_CLUTCH={AID_AUTO_CLUTCH}  "
        f"TRACTION_CONTROL={AID_TC}  ABS={AID_ABS}  STABILITY_CONTROL={AID_STABILITY}  "
        f"FUEL_RATE={AID_FUEL_RATE}  DAMAGE={AID_DAMAGE}"
    )

    # ── Verify by reading back ────────────────────────────────────────────────
    verify_race = configparser.ConfigParser()
    verify_race.read(str(AC_RACE_INI))
    verify_launcher = configparser.ConfigParser()
    verify_launcher.read(str(AC_LAUNCHER_INI))
    verify_assists = configparser.RawConfigParser()
    verify_assists.optionxform = str
    verify_assists.read(str(AC_ASSISTS_INI))

    ok = True
    if verify_race["RACE"]["track"] != TARGET_TRACK:
        _fail(f"race.ini TRACK mismatch: got {verify_race['RACE']['track']}")
        ok = False
    if verify_race["RACE"]["model"] != TARGET_CAR:
        _fail(f"race.ini MODEL mismatch: got {verify_race['RACE']['model']}")
        ok = False
    expected_sun_angle = str(int(16.0 * (TARGET_TIME_OF_DAY - 46800) / 3600))
    got_sun_angle = verify_race["LIGHTING"].get("sun_angle", "<missing>") if "LIGHTING" in verify_race else "<missing>"
    if got_sun_angle != expected_sun_angle:
        _fail(f"race.ini SUN_ANGLE mismatch: expected {expected_sun_angle}, got {got_sun_angle}")
        ok = False
    if verify_launcher["SAVED"]["drive"] != TARGET_DRIVE_MODE:
        _fail(f"launcher.ini DRIVE mismatch: got {verify_launcher['SAVED']['drive']}")
        ok = False

    # Verify every aid field we wrote
    _aid_checks = [
        ("AUTO_SHIFTER",      AID_AUTO_GEARBOX),
        ("AUTO_CLUTCH",       AID_AUTO_CLUTCH),
        ("TRACTION_CONTROL",  AID_TC),
        ("ABS",               AID_ABS),
        ("STABILITY_CONTROL", AID_STABILITY),
        ("FUEL_RATE",         AID_FUEL_RATE),
        ("DAMAGE",            AID_DAMAGE),
    ]
    for key, expected in _aid_checks:
        got = verify_assists.get("ASSISTS", key, fallback="<missing>")
        # Strip inline comments (assists.ini uses '; comment' on the same line)
        got_val = got.split(";")[0].strip()
        if got_val != str(expected):
            _fail(f"assists.ini {key} mismatch: expected {expected}, got {got_val!r}")
            ok = False

    if ok:
        _pass("race.ini, launcher.ini, and assists.ini written and verified correctly.")


def test_kill_ac():
    """
    Kill acs.exe and AssettoCorsa.exe if running, verify they are gone.
    Safe to run when AC is not running.
    """
    print("\n=== test_kill_ac ===")

    was_running = is_ac_running()
    _info(f"AC running before kill: {was_running}")

    for name in AC_PROCESSES:
        result = subprocess.run(
            f"wmic process where \"name='{name}'\" delete",
            shell=True, capture_output=True, text=True
        )
        combined = (result.stdout + result.stderr).lower()
        if "deleting instance" in combined or "instance deletion" in combined:
            _info(f"Killed {name}")
        else:
            _info(f"{name} was not running (no-op)")

    time.sleep(2)

    still_running = is_ac_running()
    if still_running:
        _fail("AC process still detected after kill.")
    else:
        _pass("AC processes confirmed gone (or were not running).")


def wait_for_acs_process(timeout: float = 60.0) -> bool:
    """
    Poll until acs.exe appears in the process list or timeout expires.
    acs.exe is the simulation process — it only spawns AFTER the user clicks
    'Drive' in the AC launcher UI (AssettoCorsa.exe).
    Returns True if acs.exe found, False on timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq acs.exe"],
            capture_output=True, text=True
        )
        if "acs.exe" in result.stdout.lower():
            return True
        time.sleep(2)
    return False


def test_launch_ac(wait_for_plugin: bool = True):
    """
    Launch AC via Steam -applaunch and optionally wait for the plugin to respond.

    IMPORTANT — two-stage launch:
      Stage 1: steam -applaunch → AssettoCorsa.exe (launcher UI) appears.
      Stage 2: User must click 'Drive' in the launcher → acs.exe spawns.
               Only after acs.exe is running and the session is loaded does
               the plugin start serving on port 2345.

    This test:
      - Sends the launch command
      - Waits for AssettoCorsa.exe to appear
      - Prompts the user to click Drive
      - Waits for acs.exe to appear
      - Polls the plugin port

    If running headlessly (no user), this test will hang at the Drive prompt.
    Full automation requires Content Manager's '--start' flag or similar.
    """
    print("\n=== test_launch_ac ===")

    if not STEAM_EXE.exists():
        _fail(f"Steam not found at {STEAM_EXE}")
        return

    if is_ac_running():
        _info("AC already running — skipping launch (kill first if you want a fresh test).")
        if wait_for_plugin:
            _info("Checking if plugin is already responding ...")
            if is_plugin_ready(PLUGIN_EGO_PORT):
                _pass(f"Plugin already responding on port {PLUGIN_EGO_PORT}.")
            else:
                _fail("AC running but plugin not responding — is a session loaded?")
        return

    # Stage 1: send launch command
    _info(f"Launching AC launcher via: {STEAM_EXE} -applaunch {AC_APP_ID}")
    subprocess.Popen([str(STEAM_EXE), "-applaunch", AC_APP_ID])

    # Wait for AssettoCorsa.exe (launcher UI)
    _info("Waiting for AssettoCorsa.exe (launcher UI) to appear ...")
    deadline = time.time() + 30
    while time.time() < deadline:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq AssettoCorsa.exe"],
            capture_output=True, text=True
        )
        if "assettoCorsa.exe" in result.stdout.lower() or "assetocorsa.exe" in result.stdout.lower():
            _pass("AssettoCorsa.exe (launcher) is running.")
            break
        time.sleep(2)
    else:
        _fail("AssettoCorsa.exe did not appear within 30s.")
        return

    if not wait_for_plugin:
        _pass("Launch command sent. acs.exe not yet running — user must click Drive.")
        return

    # Stage 2: wait for acs.exe (requires user to click Drive)
    _info("=" * 55)
    _info("ACTION REQUIRED: Click 'Drive' in the AC launcher UI.")
    _info("Waiting up to 120s for acs.exe to appear ...")
    _info("=" * 55)

    if not wait_for_acs_process(timeout=120):
        _fail("acs.exe did not appear within 120s — did you click Drive?")
        return
    _pass("acs.exe is running — session is loading ...")

    # Stage 3: poll plugin port
    _info(f"Waiting up to {PLUGIN_READY_TIMEOUT}s for plugin on port {PLUGIN_EGO_PORT} ...")
    deadline = time.time() + PLUGIN_READY_TIMEOUT
    while time.time() < deadline:
        elapsed = time.time() - (deadline - PLUGIN_READY_TIMEOUT)
        if is_plugin_ready(PLUGIN_EGO_PORT):
            _pass(f"Plugin ego port {PLUGIN_EGO_PORT} responded after {elapsed:.0f}s.")
            return
        _info(f"  {elapsed:.0f}s elapsed — plugin not ready yet, retrying in {PLUGIN_POLL_INTERVAL}s ...")
        time.sleep(PLUGIN_POLL_INTERVAL)

    _fail(f"Plugin did not respond within {PLUGIN_READY_TIMEOUT}s after acs.exe started.")


def test_launch_ac_direct(wait_for_plugin: bool = True):
    """
    Launch acs.exe directly — bypasses both the stock launcher and Content Manager.

    acs.exe reads cfg/race.ini on startup for track/car/session config.
    Requires:
      - Steam running (for authentication via steam_appid.txt in AC root)
      - race.ini already written with correct track/car (test_write_session_config)

    This is the cleanest fully-automated approach — no UI, no manual clicks.
    """
    print("\n=== test_launch_ac_direct ===")

    ACS_EXE = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\acs.exe")

    if not ACS_EXE.exists():
        _fail(f"acs.exe not found at {ACS_EXE}")
        return

    if is_ac_running():
        _info("AC already running — skipping launch (kill first for a fresh test).")
        if wait_for_plugin and is_plugin_ready(PLUGIN_EGO_PORT):
            _pass("Plugin already responding.")
        return

    _info(f"Launching acs.exe directly: {ACS_EXE}")
    _info("(reads cfg/race.ini for track/car — make sure write_session_config ran first)")
    subprocess.Popen([str(ACS_EXE)], cwd=str(ACS_EXE.parent))

    if not wait_for_plugin:
        _pass("acs.exe launched.")
        return

    # Wait for acs.exe to show up (it IS acs.exe so it's immediate)
    _info("Waiting for session to load and plugin to respond ...")
    _info(f"Polling port {PLUGIN_EGO_PORT} every {PLUGIN_POLL_INTERVAL}s ...")

    deadline = time.time() + PLUGIN_READY_TIMEOUT
    while time.time() < deadline:
        elapsed = time.time() - (deadline - PLUGIN_READY_TIMEOUT)
        if is_plugin_ready(PLUGIN_EGO_PORT):
            _pass(f"Plugin ego port {PLUGIN_EGO_PORT} responded after {elapsed:.0f}s.")
            return
        _info(f"  {elapsed:.0f}s elapsed — not ready yet ...")
        time.sleep(PLUGIN_POLL_INTERVAL)

    _fail(f"Plugin did not respond within {PLUGIN_READY_TIMEOUT}s.")


def test_launch_ac_cm(wait_for_plugin: bool = True):
    """
    Launch AC fully automatically via Content Manager's acmanager:// URI scheme.

    How it works
    ------------
    Content Manager registers itself as the handler for the 'acmanager://'
    protocol in the Windows registry:

        HKCR\\acmanager\\shell\\open\\command ->
            "C:\\...\\Content Manager.exe" "%1"

    Passing 'acmanager://race/quick?preset=<json>' as a CLI argument to CM.exe
    is equivalent to the OS dispatching the URI — CM parses it through its
    ArgumentsHandler and calls QuickDrive.RunAsync() directly, bypassing all
    manual UI interaction.

    The preset JSON structure (QuickDrive.SaveableData fields):
        Mode     — relative URI to the mode XAML page
        CarId    — AC car folder name, e.g. 'ks_mazda_miata'
        TrackId  — AC track folder name (with optional '/layout' suffix)
        Time     — time of day in minutes from midnight (e.g. 720 = 12:00)
        Temperature — ambient temperature in °C

    Mode paths (from QuickDrive.xaml.cs constants):
        /Pages/Drive/QuickDrive_TimeAttack.xaml
        /Pages/Drive/QuickDrive_Hotlap.xaml
        /Pages/Drive/QuickDrive_Practice.xaml
        /Pages/Drive/QuickDrive_Race.xaml

    Two-stage launch (same as stock launcher, but no manual click needed):
        Stage 1: CM.exe receives URI → calls QuickDrive.RunAsync() →
                 spawns AssettoCorsa.exe (launcher) → acs.exe starts automatically.
        Stage 2: acs.exe loads session → plugin port becomes active.

    Requires:
        - CM_EXE must exist (Content Manager installed)
        - Steam must be running (AC Steam auth)
        - AC not already running (or this is a no-op with a readiness check)
    """
    print("\n=== test_launch_ac_cm ===")

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    if not CM_EXE.exists():
        _fail(f"Content Manager not found at {CM_EXE}")
        _info("Install CM from https://acstuff.ru/app/ and update CM_EXE.")
        return

    if is_ac_running():
        _info("AC already running — skipping launch (kill first for a fresh test).")
        if wait_for_plugin:
            if is_plugin_ready(PLUGIN_EGO_PORT):
                _pass(f"Plugin already responding on port {PLUGIN_EGO_PORT}.")
            else:
                _fail("AC running but plugin not responding — is a session loaded?")
        return

    # ── Build acmanager://race/quick URI ───────────────────────────────────────
    # Mode URI maps TARGET_DRIVE_MODE to the CM QuickDrive page path.
    _MODE_MAP = {
        "timeattack": "/Pages/Drive/QuickDrive_TimeAttack.xaml",
        "hotlap":     "/Pages/Drive/QuickDrive_Hotlap.xaml",
        "practice":   "/Pages/Drive/QuickDrive_Practice.xaml",
        "race":       "/Pages/Drive/QuickDrive_Race.xaml",
        "drift":      "/Pages/Drive/QuickDrive_Drift.xaml",
        "drag":       "/Pages/Drive/QuickDrive_Drag.xaml",
    }
    mode_path = _MODE_MAP.get(TARGET_DRIVE_MODE.lower())
    if mode_path is None:
        _fail(
            f"Unknown drive mode '{TARGET_DRIVE_MODE}'. "
            f"Valid: {list(_MODE_MAP.keys())}"
        )
        return

    # AssistsData embedded in preset — requires CM "Load assists with preset"
    # (Settings → Drive) to be enabled. asc=True forces CM to apply it.
    assists_data = json.dumps({
        "AutoShifter":      True,
        "AutoClutch":       True,
        "Abs":              2,
        "TractionControl":  2,
        "StabilityControl": 100.0,
        "Damage":           0.0,
        "TyreWear":         0.0,
        "FuelConsumption":  0.0,
        "IdealLine":        True,
        "AutoBlip":         True,
        "AutoBrake":        False,
        "VisualDamage":     False,
        "TyreBlankets":     False,
        "SlipSteam":        1.0,
    }, separators=(",", ":"))

    preset_dict = {
        "Mode":        mode_path,
        "CarId":       TARGET_CAR,
        "TrackId":     TARGET_TRACK,
        "Time":        TARGET_TIME_OF_DAY,
        "Temperature": 26,
        "asc":         True,
        "AssistsData": assists_data,
    }
    preset_json    = json.dumps(preset_dict, separators=(",", ":"))
    preset_encoded = urllib.parse.quote(preset_json, safe="")
    uri            = f"acmanager://race/quick?preset={preset_encoded}"

    _info(f"Preset JSON : {preset_json}")
    _info(f"Launching   : {CM_EXE.name} <acmanager://race/quick?preset=...>")

    # ── Launch CM with the URI as its first argument ───────────────────────────
    # CM.exe accepts the URI directly (same contract as the registry handler).
    # We do NOT use shell=True or subprocess.run — Popen keeps this non-blocking.
    try:
        subprocess.Popen(
            [str(CM_EXE), uri],
            # Detach from our console so CM's own window management works.
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    except OSError as exc:
        _fail(f"Failed to launch Content Manager: {exc}")
        return

    if not wait_for_plugin:
        _pass("CM launched with race URI. acs.exe not yet confirmed — plugin check skipped.")
        return

    # ── Stage 1: wait for acs.exe to appear ───────────────────────────────────
    # CM → QuickDrive.RunAsync → Game.StartAsync → acs.exe spawns.
    # Allow up to 90s for CM to resolve assets and launch the session.
    _info("Waiting up to 90s for acs.exe to appear (CM is loading the session) ...")
    if not wait_for_acs_process(timeout=90.0):
        _fail(
            "acs.exe did not appear within 90s after CM was launched.\n"
            "  Possible causes:\n"
            "    - CM showed an error dialog (missing car/track asset)\n"
            "    - Steam is not running (AC needs Steam auth)\n"
            "    - CM opened its UI instead of auto-starting (preset format mismatch)\n"
            "  Check the CM window for error messages."
        )
        return
    _pass("acs.exe is running — session is loading ...")

    # ── Stage 2: poll plugin port ──────────────────────────────────────────────
    _info(f"Waiting up to {PLUGIN_READY_TIMEOUT}s for plugin on port {PLUGIN_EGO_PORT} ...")
    t_launch = time.time()
    deadline = t_launch + PLUGIN_READY_TIMEOUT
    plugin_ready = False
    while time.time() < deadline:
        elapsed = time.time() - t_launch
        if is_plugin_ready(PLUGIN_EGO_PORT):
            _info(
                f"Plugin ego port {PLUGIN_EGO_PORT} responded after {elapsed:.0f}s. "
                "Proceeding to dismiss session screen ..."
            )
            plugin_ready = True
            break
        _info(f"  {elapsed:.0f}s elapsed — plugin not ready yet, retrying in {PLUGIN_POLL_INTERVAL}s ...")
        time.sleep(PLUGIN_POLL_INTERVAL)

    if not plugin_ready:
        _fail(
            f"Plugin did not respond within {PLUGIN_READY_TIMEOUT}s after acs.exe started.\n"
            "  The AC plugin (acti) may not be installed or the session may still be loading."
        )
        return

    # ── Stage 3: dismiss session info screen, get car on track ────────────────
    # After AC loads, the session info overlay (pit lane screen) is shown.
    # Sending reset via TCP 2347 calls ac.ext_resetCar() which dismisses the
    # overlay and places the car at the start line.
    test_dismiss_session_screen()


def test_full_cycle():
    """
    Full sequence: write config → kill AC → launch AC → dismiss session screen.

    Steps:
        1. test_write_session_config  — write race.ini, launcher.ini, assists.ini
        2. test_kill_ac               — kill acs.exe + AssettoCorsa.exe
        3. sleep 3s                   — give OS time to release ports
        4. test_launch_ac_cm          — CM URI launch, wait for plugin, dismiss
                                        session info screen via TCP 2347 reset
                                        and confirm car is on track

    This is the sequence that runs before each training/collection phase.
    After this function returns PASS, the car is at the Monza start line,
    status=AC_LIVE, isInPit=0, and the RL env can call reset()/step() immediately.
    """
    print("\n=== test_full_cycle ===")

    t_start = time.time()

    # Step 1: Write session config
    _info("Step 1: Writing session config ...")
    test_write_session_config()

    # Step 2: Kill AC
    _info("Step 2: Killing AC ...")
    test_kill_ac()
    time.sleep(3)  # give OS time to release ports

    # Step 3: Launch via Content Manager (no manual click needed).
    # test_launch_ac_cm internally calls test_dismiss_session_screen after
    # the plugin port responds, so the car is on track when this returns.
    _info("Step 3: Launching AC via Content Manager, waiting for plugin, dismissing session screen ...")
    test_launch_ac_cm(wait_for_plugin=True)

    # Step 4: re-write assists.ini after session load
    # AC/CM overwrites assists.ini during startup — enforce our aid settings now.
    _info("Step 4: Re-writing assists.ini (post-launch override) ...")
    test_write_session_config()

    elapsed = time.time() - t_start
    _info(f"Full cycle completed in {elapsed:.1f}s")


# ── Entry point ───────────────────────────────────────────────────────────────

ALL_TESTS = {
    "write_session_config":    test_write_session_config,
    "kill_ac":                 test_kill_ac,
    "launch_ac":               test_launch_ac,              # stock Steam launcher (requires manual Drive click)
    "launch_ac_cm":            test_launch_ac_cm,           # CM URI launch + dismiss session screen
    "dismiss_session_screen":  test_dismiss_session_screen, # standalone: reset + wait for on-track
    "full_cycle":              test_full_cycle,              # uses CM internally
}


def main():
    parser = argparse.ArgumentParser(
        description="Test programmatic AC launch/kill/session-config.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--test", type=str, default=None,
        choices=list(ALL_TESTS.keys()),
        help=(
            "Run a specific test.  If omitted, runs all tests in order:\n"
            "  write_session_config → kill_ac → launch_ac → launch_ac_cm\n"
            "  → dismiss_session_screen → full_cycle"
        ),
    )
    args = parser.parse_args()

    if args.test:
        ALL_TESTS[args.test]()
    else:
        for name, fn in ALL_TESTS.items():
            fn()

    print()


if __name__ == "__main__":
    main()
