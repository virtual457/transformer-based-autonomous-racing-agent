"""
ac_lifecycle.py — Library module for programmatic Assetto Corsa lifecycle management.

Public API
----------
    write_session_config()     — write race.ini, launcher.ini, assists.ini
    kill_ac()                  — kill acs.exe / AssettoCorsa.exe / Content Manager.exe
    launch_ac_cm(wait_for_plugin=True) — CM URI launch; optionally wait for plugin port
    dismiss_session_screen()   — TCP reset + Drive icon click; poll until AC_LIVE + off pit
    full_cycle(max_retries=3)  — write_session_config → kill_ac → launch_ac_cm
                                  → dismiss_session_screen, retrying on failure
    is_ac_live()               — True if acpmf_graphics shows AC_LIVE and isInPit==0
    is_ac_crashed()            — True if acpmf_graphics is unavailable or status==AC_OFF

Extracted from tests/test_ac_lifecycle.py.  Uses only Python stdlib (no pywin32,
no pyautogui).
"""

import configparser
import ctypes
import json
import logging
import mmap
import socket
import subprocess
import time
import urllib.parse
from ctypes import c_float, c_int32, c_wchar
from pathlib import Path

logger = logging.getLogger("ac_lifecycle")

# ── Constants ──────────────────────────────────────────────────────────────────

STEAM_EXE         = Path(r"C:\Program Files (x86)\Steam\steam.exe")
AC_APP_ID         = "244210"
AC_PROCESSES      = ["acs.exe", "AssettoCorsa.exe", "Content Manager.exe"]
CM_EXE            = Path(r"C:\Users\chand\Downloads\content-manager\Content Manager.exe")

AC_RACE_INI       = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\cfg\race.ini")
AC_LAUNCHER_INI   = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\cfg\launcher.ini")
AC_ASSISTS_INI    = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\cfg\assists.ini")

# Target session
TARGET_TRACK       = "monza"
TARGET_LAYOUT      = ""          # Sub-layout for multi-layout tracks (e.g. "layout_gp"). "" = single-layout track.
TARGET_CAR         = "ks_mazda_miata"
TARGET_DRIVE_MODE  = "hotlap"   # Hotlap — no AI, no race start
TARGET_TIME_OF_DAY = 54000      # Seconds from midnight; 54000 = 15:00

# Driving aids
AID_AUTO_GEARBOX = 1    # AUTO_SHIFTER:       0=manual, 1=automatic
AID_AUTO_CLUTCH  = 1    # AUTO_CLUTCH:        0=manual, 1=automatic
AID_TC           = 2    # TRACTION_CONTROL:   0=off, 1=factory, 2=max
AID_ABS          = 2    # ABS:                0=off, 1=factory, 2=max
AID_STABILITY    = 100  # STABILITY_CONTROL:  0-100 %
AID_FUEL_RATE    = 0    # FUEL_RATE:          0=off, 1=normal consumption
AID_DAMAGE       = 0    # DAMAGE:             0=off, 1=normal mechanical damage

# Plugin ports
PLUGIN_HOST          = "127.0.0.1"
PLUGIN_EGO_PORT      = 2345     # ego_server (UDP)
PLUGIN_MGMT_PORT     = 2347     # simulation_management_server (TCP)
PLUGIN_READY_TIMEOUT = 600      # seconds to wait for plugin after launch
PLUGIN_POLL_INTERVAL = 3        # seconds between readiness polls

# AC shared memory status codes
AC_OFF    = 0
AC_REPLAY = 1
AC_LIVE   = 2
AC_PAUSE  = 3

# Dismiss timeouts
DISMISS_TIMEOUT       = 30
DISMISS_POLL_INTERVAL = 1


# ── AC shared memory ───────────────────────────────────────────────────────────

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


def _read_ac_graphics() -> dict | None:
    """
    Open AC's acpmf_graphics named shared memory and return
    {'status': int, 'isInPit': int}, or None if unavailable.
    """
    try:
        mm = mmap.mmap(0, ctypes.sizeof(_SPageFileGraphic),
                       "acpmf_graphics", access=mmap.ACCESS_READ)
        gfx = _SPageFileGraphic.from_buffer_copy(mm)
        mm.close()
        return {"status": gfx.status, "isInPit": gfx.isInPit}
    except Exception:
        return None


# ── Win32 mouse / keypress helpers ────────────────────────────────────────────

_user32 = ctypes.windll.user32

_VK_RETURN  = 0x0D
_VK_SPACE   = 0x20
_VK_ESCAPE  = 0x1B
_VK_CONTROL = 0x11
_VK_C       = 0x43

_WM_KEYDOWN = 0x0100
_WM_KEYUP   = 0x0101

_KEYEVENTF_KEYUP = 0x0002

_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004

_SW_RESTORE = 9


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left",   ctypes.c_long),
        ("top",    ctypes.c_long),
        ("right",  ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


_AC_WINDOW_TITLES = [
    "Assetto Corsa",
    "AC2",
]


def _find_ac_window() -> int:
    """Return HWND of the AC rendering window, or 0 if not found."""
    for title in _AC_WINDOW_TITLES:
        hwnd = _user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
    return 0


def _send_mouse_click_to_ac() -> bool:
    """
    Bring the AC window to the foreground and send three left-clicks to the
    Drive (steering wheel) icon on the left sidebar.

    Click target: frac=(0.047, 0.195) — confirmed via capture_mouse_coords.py.
    Returns True if the window was found and clicks were dispatched.
    """
    hwnd = _find_ac_window()
    if not hwnd:
        logger.warning("send_mouse_click_to_ac: AC window not found (FindWindowW returned 0).")
        return False

    rect = _RECT()
    _user32.GetWindowRect(hwnd, ctypes.byref(rect))
    win_w = rect.right  - rect.left
    win_h = rect.bottom - rect.top

    click_x = rect.left + int(win_w * 0.047)
    click_y = rect.top  + int(win_h * 0.195)

    logger.info(
        "send_mouse_click_to_ac: hwnd=%s  window=(%d,%d)-(%d,%d)  click=(%d,%d)",
        hex(hwnd), rect.left, rect.top, rect.right, rect.bottom, click_x, click_y,
    )

    _user32.ShowWindow(hwnd, _SW_RESTORE)
    _user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)

    _user32.SetCursorPos(click_x, click_y)
    for i in range(3):
        _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.05)
        _user32.mouse_event(_MOUSEEVENTF_LEFTUP,   0, 0, 0, 0)
        logger.info("  Click %d/3 at (%d, %d)", i + 1, click_x, click_y)
        time.sleep(1.0)
    return True


def _send_ctrl_c_to_ac() -> bool:
    """
    Bring the AC window to the foreground and send a Ctrl+C keypress.

    Used to toggle AI autopilot mode on/off.  Returns True if the window
    was found and the keypress was dispatched.
    """
    hwnd = _find_ac_window()
    if not hwnd:
        logger.warning("_send_ctrl_c_to_ac: AC window not found (FindWindowW returned 0).")
        return False

    logger.info("_send_ctrl_c_to_ac: hwnd=%s — sending Ctrl+C", hex(hwnd))
    _user32.ShowWindow(hwnd, _SW_RESTORE)
    _user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)

    _user32.keybd_event(_VK_CONTROL, 0, 0, 0)            # Ctrl down
    time.sleep(0.05)
    _user32.keybd_event(_VK_C, 0, 0, 0)                  # C down
    time.sleep(0.05)
    _user32.keybd_event(_VK_C, 0, _KEYEVENTF_KEYUP, 0)   # C up
    time.sleep(0.05)
    _user32.keybd_event(_VK_CONTROL, 0, _KEYEVENTF_KEYUP, 0)  # Ctrl up

    logger.info("_send_ctrl_c_to_ac: Ctrl+C dispatched.")
    return True


def _send_management_reset(timeout: float = 3.0) -> bool:
    """
    Send the 'reset' command to the management server on TCP port 2347.
    Returns True on success, False on error.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((PLUGIN_HOST, PLUGIN_MGMT_PORT))
            s.sendall(b"reset")
        return True
    except Exception as exc:
        logger.warning("send_management_reset failed: %s", exc)
        return False


def _is_plugin_ready(port: int = PLUGIN_EGO_PORT, timeout: float = 2.0) -> bool:
    """
    Return True if the plugin port is accepting connections.
    EGO port (2345) is UDP; management port (2347) is TCP.
    """
    if port == PLUGIN_EGO_PORT:
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
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((PLUGIN_HOST, port))
            sock.close()
            return True
        except Exception:
            return False


def _is_ac_running() -> bool:
    """Return True if any AC_PROCESSES entry is in the process list (wmic)."""
    for name in AC_PROCESSES:
        result = subprocess.run(
            f"wmic process where \"name='{name}'\" get name",
            shell=True, capture_output=True, text=True,
        )
        if name.lower() in result.stdout.lower():
            return True
    return False


def _wait_for_acs_process(timeout: float = 90.0) -> bool:
    """Poll until acs.exe appears in the process list or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq acs.exe"],
            capture_output=True, text=True,
        )
        if "acs.exe" in result.stdout.lower():
            return True
        time.sleep(2)
    return False


# ── Public API ─────────────────────────────────────────────────────────────────

def write_session_config() -> None:
    """
    Write race.ini, launcher.ini, and assists.ini to configure the AC session.

    race.ini     — track, car, time of day (sun_angle in [LIGHTING])
    launcher.ini — drive mode (hotlap / timeattack / practice)
    assists.ini  — all driving aids (auto gearbox, auto clutch, TC, ABS,
                   stability control, fuel rate, mechanical damage)

    Raises RuntimeError if any config file is missing or cannot be written.
    """
    # ── race.ini ──────────────────────────────────────────────────────────────
    if not AC_RACE_INI.exists():
        raise RuntimeError(f"race.ini not found at {AC_RACE_INI}")

    race_cfg = configparser.ConfigParser()
    race_cfg.read(str(AC_RACE_INI))

    if "RACE" not in race_cfg:
        race_cfg["RACE"] = {}

    race_cfg["RACE"]["TRACK"]        = TARGET_TRACK
    race_cfg["RACE"]["CONFIG_TRACK"] = TARGET_LAYOUT
    race_cfg["RACE"]["MODEL"]        = TARGET_CAR

    # sun_angle formula (from CM source Game.Properties.cs GetSunAngle):
    #   sun_angle = 16.0 * (seconds_from_midnight - 46800) / 3600
    _sun_angle = int(16.0 * (TARGET_TIME_OF_DAY - 46800) / 3600)
    if "LIGHTING" not in race_cfg:
        race_cfg["LIGHTING"] = {}
    race_cfg["LIGHTING"]["SUN_ANGLE"] = str(_sun_angle)

    try:
        with open(str(AC_RACE_INI), "w") as f:
            race_cfg.write(f)
    except OSError as exc:
        raise RuntimeError(f"Failed to write race.ini: {exc}") from exc

    _target_hhmm = f"{TARGET_TIME_OF_DAY // 3600:02d}:{(TARGET_TIME_OF_DAY % 3600) // 60:02d}"
    logger.info(
        "Written race.ini: TRACK=%s  MODEL=%s  SUN_ANGLE=%d (%s)",
        TARGET_TRACK, TARGET_CAR, _sun_angle, _target_hhmm,
    )

    # ── launcher.ini ─────────────────────────────────────────────────────────
    if not AC_LAUNCHER_INI.exists():
        raise RuntimeError(f"launcher.ini not found at {AC_LAUNCHER_INI}")

    launcher_cfg = configparser.ConfigParser()
    launcher_cfg.read(str(AC_LAUNCHER_INI))

    if "SAVED" not in launcher_cfg:
        launcher_cfg["SAVED"] = {}

    launcher_cfg["SAVED"]["DRIVE"] = TARGET_DRIVE_MODE

    try:
        with open(str(AC_LAUNCHER_INI), "w") as f:
            launcher_cfg.write(f)
    except OSError as exc:
        raise RuntimeError(f"Failed to write launcher.ini: {exc}") from exc

    logger.info("Written launcher.ini: DRIVE=%s", TARGET_DRIVE_MODE)

    # ── assists.ini ───────────────────────────────────────────────────────────
    if not AC_ASSISTS_INI.exists():
        raise RuntimeError(f"assists.ini not found at {AC_ASSISTS_INI}")

    # RawConfigParser with optionxform=str preserves UPPER_CASE keys that AC expects.
    assists_cfg = configparser.RawConfigParser()
    assists_cfg.optionxform = str
    assists_cfg.read(str(AC_ASSISTS_INI))

    if "ASSISTS" not in assists_cfg:
        assists_cfg["ASSISTS"] = {}

    assists_cfg["ASSISTS"]["AUTO_SHIFTER"]      = str(AID_AUTO_GEARBOX)
    assists_cfg["ASSISTS"]["AUTO_CLUTCH"]       = str(AID_AUTO_CLUTCH)
    assists_cfg["ASSISTS"]["TRACTION_CONTROL"]  = str(AID_TC)
    assists_cfg["ASSISTS"]["ABS"]               = str(AID_ABS)
    assists_cfg["ASSISTS"]["STABILITY_CONTROL"] = str(AID_STABILITY)
    assists_cfg["ASSISTS"]["FUEL_RATE"]         = str(AID_FUEL_RATE)
    assists_cfg["ASSISTS"]["DAMAGE"]            = str(AID_DAMAGE)

    try:
        with open(str(AC_ASSISTS_INI), "w") as f:
            assists_cfg.write(f)
    except OSError as exc:
        raise RuntimeError(f"Failed to write assists.ini: {exc}") from exc

    logger.info(
        "Written assists.ini: AUTO_SHIFTER=%d  AUTO_CLUTCH=%d  "
        "TRACTION_CONTROL=%d  ABS=%d  STABILITY_CONTROL=%d  "
        "FUEL_RATE=%d  DAMAGE=%d",
        AID_AUTO_GEARBOX, AID_AUTO_CLUTCH, AID_TC, AID_ABS,
        AID_STABILITY, AID_FUEL_RATE, AID_DAMAGE,
    )


def kill_ac() -> None:
    """
    Kill acs.exe, AssettoCorsa.exe, and Content Manager.exe via wmic.
    No-op if no AC processes are running.  Never raises.
    """
    for name in AC_PROCESSES:
        try:
            result = subprocess.run(
                f"wmic process where \"name='{name}'\" delete",
                shell=True, capture_output=True, text=True,
            )
            combined = (result.stdout + result.stderr).lower()
            if "deleting instance" in combined or "instance deletion" in combined:
                logger.info("Killed %s", name)
            else:
                logger.info("%s was not running (no-op)", name)
        except Exception as exc:
            logger.warning("kill_ac: wmic call for '%s' raised: %s", name, exc)

    time.sleep(2)


def launch_ac_cm(wait_for_plugin: bool = True) -> None:
    """
    Launch AC via Content Manager's acmanager:// URI scheme.

    CM.exe receives a 'acmanager://race/quick?preset=<json>' URI and calls
    QuickDrive.RunAsync() directly — no manual UI interaction needed.

    If wait_for_plugin is True, blocks until the plugin ego port (2345) responds
    or PLUGIN_READY_TIMEOUT seconds elapse.

    Raises RuntimeError if:
      - CM_EXE does not exist
      - acs.exe does not appear within 90s
      - the plugin port does not respond within PLUGIN_READY_TIMEOUT seconds
    """
    if not CM_EXE.exists():
        raise RuntimeError(
            f"Content Manager not found at {CM_EXE}. "
            "Install CM from https://acstuff.ru/app/ and update CM_EXE."
        )

    if _is_ac_running():
        logger.info("AC already running — skipping launch.")
        if wait_for_plugin:
            if _is_plugin_ready(PLUGIN_EGO_PORT):
                logger.info("Plugin already responding on port %d.", PLUGIN_EGO_PORT)
            else:
                raise RuntimeError(
                    "AC is running but plugin is not responding. "
                    "Is a session loaded?"
                )
        return

    # Build acmanager:// URI
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
        raise RuntimeError(
            f"Unknown drive mode '{TARGET_DRIVE_MODE}'. "
            f"Valid: {list(_MODE_MAP.keys())}"
        )

    # AssistsData embedded in preset — requires CM setting
    # "Load assists with preset" (Settings → Drive) to be enabled.
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
        "asc":         True,          # AssistsChanged — force CM to apply AssistsData
        "AssistsData": assists_data,
    }
    preset_json    = json.dumps(preset_dict, separators=(",", ":"))
    preset_encoded = urllib.parse.quote(preset_json, safe="")
    uri            = f"acmanager://race/quick?preset={preset_encoded}"

    logger.info("Preset JSON : %s", preset_json)
    logger.info("Launching   : %s <acmanager://race/quick?preset=...>", CM_EXE.name)

    try:
        subprocess.Popen(
            [str(CM_EXE), uri],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to launch Content Manager: {exc}") from exc

    if not wait_for_plugin:
        logger.info("CM launched with race URI. Plugin check skipped.")
        return

    # Stage 1: wait for acs.exe
    logger.info("Waiting up to 90s for acs.exe to appear ...")
    if not _wait_for_acs_process(timeout=90.0):
        raise RuntimeError(
            "acs.exe did not appear within 90s after CM was launched. "
            "Possible causes: CM showed an error dialog (missing car/track asset), "
            "Steam is not running, or CM opened its UI instead of auto-starting."
        )
    logger.info("acs.exe is running — session is loading ...")

    # Brief pause to let the session finish loading before pinging the plugin.
    # acs.exe appears quickly but the plugin socket isn't ready for ~10-15s.
    _LAUNCH_SETTLE_S = 15
    logger.info("Waiting %ds for session to settle before pinging plugin ...", _LAUNCH_SETTLE_S)
    time.sleep(_LAUNCH_SETTLE_S)

    # Stage 2: poll plugin port
    logger.info(
        "Waiting up to %ds for plugin on port %d ...",
        PLUGIN_READY_TIMEOUT, PLUGIN_EGO_PORT,
    )
    t_launch = time.time()
    deadline = t_launch + PLUGIN_READY_TIMEOUT
    while time.time() < deadline:
        elapsed = time.time() - t_launch
        if _is_plugin_ready(PLUGIN_EGO_PORT):
            logger.info(
                "Plugin ego port %d responded after %.0fs.",
                PLUGIN_EGO_PORT, elapsed,
            )
            return
        logger.info(
            "  %.0fs elapsed — plugin not ready yet, retrying in %ds ...",
            elapsed, PLUGIN_POLL_INTERVAL,
        )
        time.sleep(PLUGIN_POLL_INTERVAL)

    raise RuntimeError(
        f"Plugin did not respond within {PLUGIN_READY_TIMEOUT}s after acs.exe started. "
        "The AC plugin (acti) may not be installed or the session may still be loading."
    )


def dismiss_session_screen() -> bool:
    """
    Dismiss the AC session info / pit lane overlay and confirm the car is on track.

    Strategy (per attempt):
      A. Send TCP reset to management port 2347 → ac.ext_resetCar()
      B. Send three mouse clicks to the Drive (steering wheel) icon at frac=(0.047, 0.195)
      Poll for AC_LIVE + isInPit==0 after each attempt.

    Up to DISMISS_RETRY_COUNT (3) full cycles are attempted.

    Returns True when status==AC_LIVE and isInPit==0.
    Returns False on timeout.
    """
    if not _is_plugin_ready(PLUGIN_MGMT_PORT):
        logger.error(
            "Management server (TCP %d) not responding. "
            "Run launch_ac_cm first.",
            PLUGIN_MGMT_PORT,
        )
        return False

    gfx = _read_ac_graphics()
    if gfx is None:
        logger.error(
            "Cannot open acpmf_graphics shared memory. "
            "AC must be running and past the loading screen."
        )
        return False

    logger.info(
        "Before dismiss: status=%d (%s)  isInPit=%d",
        gfx["status"],
        "AC_LIVE" if gfx["status"] == AC_LIVE else "OTHER",
        gfx["isInPit"],
    )

    DISMISS_RETRY_COUNT = 3
    TCP_POLL_SECS       = 5

    def _poll_on_track(poll_duration: float, label: str) -> bool:
        deadline = time.time() + poll_duration
        while time.time() < deadline:
            g = _read_ac_graphics()
            elapsed_total = time.time() - t_start
            if g is None:
                logger.info("  %.1fs [%s] — shared memory not readable", elapsed_total, label)
                time.sleep(DISMISS_POLL_INTERVAL)
                continue
            logger.info(
                "  %.1fs [%s] — status=%d  isInPit=%d",
                elapsed_total, label, g["status"], g["isInPit"],
            )
            if g["status"] == AC_LIVE and g["isInPit"] == 0:
                return True
            time.sleep(DISMISS_POLL_INTERVAL)
        return False

    t_start = time.time()

    for attempt in range(1, DISMISS_RETRY_COUNT + 1):
        elapsed = time.time() - t_start
        logger.info("Attempt %d/%d (elapsed=%.1fs)", attempt, DISMISS_RETRY_COUNT, elapsed)

        # Step A: TCP reset
        logger.info("  [A] Sending reset to TCP %d ...", PLUGIN_MGMT_PORT)
        if _send_management_reset():
            logger.info("  [A] Reset sent.")
        else:
            logger.info("  [A] TCP reset failed — continuing to mouse click.")

        # Step B: Mouse click on Drive icon
        logger.info("  [B] Simulating Drive icon click (steering wheel, left sidebar) ...")
        _send_mouse_click_to_ac()

        if _poll_on_track(TCP_POLL_SECS, "mouse-click"):
            elapsed = time.time() - t_start
            logger.info("Car is on track after %.1fs (status=AC_LIVE, isInPit=0)", elapsed)
            return True

        if time.time() - t_start >= DISMISS_TIMEOUT:
            break

    gfx = _read_ac_graphics()
    logger.error(
        "Car did not leave the pit within %.1fs. "
        "Last state: status=%s  isInPit=%s. "
        "Check that the AC plugin (sensors_par) is installed and loaded.",
        time.time() - t_start,
        gfx["status"] if gfx else "N/A",
        gfx["isInPit"] if gfx else "N/A",
    )
    return False


def randomize_start_position(wait_s: float = 25.0) -> None:
    """
    Randomize the car's start position using AC's built-in AI autopilot.

    Sequence:
      1. Ctrl+C  — activate AI autopilot (AC built-in keybinding).
      2. Sleep wait_s seconds — AI drives to a random position on track.
      3. Ctrl+C  — deactivate AI autopilot.

    Call this BEFORE env.reset() each episode.  env.reset() will then read
    the current (randomized) position as the episode's starting observation
    without teleporting the car.

    Parameters
    ----------
    wait_s : float
        Seconds to let the AI drive.  Default: 25.
    """
    logger.info("randomize_start_position: Ctrl+C → AI on ...")
    if not _send_ctrl_c_to_ac():
        logger.warning(
            "randomize_start_position: AC window not found — skipping."
        )
        return
    logger.info("randomize_start_position: AI driving for %.0fs ...", wait_s)
    time.sleep(wait_s)
    logger.info("randomize_start_position: Ctrl+C → AI off ...")
    _send_ctrl_c_to_ac()
    logger.info("randomize_start_position: done — ready for env.reset().")


def full_cycle(max_retries: int = 3) -> None:
    """
    Full AC launch sequence: write_session_config → kill_ac → launch_ac_cm
    → dismiss_session_screen.

    Retries the entire sequence up to max_retries times on any failure.
    Raises RuntimeError after all retries are exhausted.

    When this function returns without raising, the car is at the start line
    with status=AC_LIVE, isInPit=0 and the RL environment can call reset()/step().
    """
    # Fast path: if AC is already live and plugin is responding, skip everything.
    if is_ac_live() and _is_plugin_ready(PLUGIN_EGO_PORT):
        logger.info("full_cycle: AC already live and plugin responding — skipping.")
        return

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        logger.info("full_cycle attempt %d/%d", attempt, max_retries)
        try:
            # Step 1: write config files
            logger.info("Step 1: Writing session config ...")
            write_session_config()

            # Step 2: kill any running AC processes
            logger.info("Step 2: Killing AC ...")
            kill_ac()
            time.sleep(3)  # give OS time to release ports

            # Step 3: launch via CM, wait for plugin
            logger.info(
                "Step 3: Launching AC via Content Manager, "
                "waiting for plugin, dismissing session screen ..."
            )
            launch_ac_cm(wait_for_plugin=True)

            # Step 4: dismiss session info screen
            logger.info("Step 4: Dismissing session screen ...")
            if not dismiss_session_screen():
                raise RuntimeError(
                    "dismiss_session_screen timed out — car did not reach AC_LIVE + isInPit=0."
                )

            # Step 5: re-write assists.ini after session load
            # AC/CM overwrites assists.ini during session startup — write again now
            # that the session is live to guarantee our aid settings are active.
            logger.info("Step 5: Re-writing assists.ini (post-launch override) ...")
            write_session_config()

            logger.info("full_cycle complete.")
            return

        except Exception as exc:
            last_exc = exc
            logger.error("full_cycle attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                logger.info("Retrying ...")
            else:
                logger.error("All %d attempts exhausted.", max_retries)

    raise RuntimeError(
        f"full_cycle failed after {max_retries} attempts. "
        f"Last error: {last_exc}"
    ) from last_exc


def is_ac_live() -> bool:
    """
    Return True if AC shared memory reports status=AC_LIVE(2) and isInPit=0.
    Returns False if shared memory is unavailable or the car is in the pit.
    """
    gfx = _read_ac_graphics()
    if gfx is None:
        return False
    return gfx["status"] == AC_LIVE and gfx["isInPit"] == 0


def is_ac_crashed() -> bool:
    """
    Return True if the AC shared memory is unavailable (AC not running / crashed)
    or if the status is AC_OFF(0).
    """
    gfx = _read_ac_graphics()
    if gfx is None:
        return True
    return gfx["status"] == AC_OFF
