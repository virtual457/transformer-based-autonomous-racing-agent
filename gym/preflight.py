"""
preflight.py — Pre-collection sanity checks.

Verifies every prerequisite before a data collection run starts.
Exits with non-zero if any check fails (unless --warn-only is passed).

Usage:
    python gym/preflight.py
    python gym/preflight.py --warn-only   # print failures but don't exit

Checks:
    1. vJoy DLL exists at the expected path
    2. vJoy Device 1 can be acquired (driver running)
    3. Vjoy.ini active profile has INPUT_METHOD=WHEEL
    4. AC process is running (acs.exe / AssettoCorsa.exe)
    5. AC plugin port 2345 is reachable (plugin loaded and session active)
    6. AC management port (sim_management) is reachable
"""

import os
import sys
import socket
import ctypes
import logging
import argparse
import subprocess

logger = logging.getLogger("preflight")

VJOY_DLL = r"C:\Program Files\vJoy\x64\vJoyInterface.dll"
VJOY_INI = r"C:\Users\chand\Documents\Assetto Corsa\cfg\controllers\Vjoy.ini"
AC_EGO_PORT = 2345
AC_MGMT_PORT = 2347          # simulation_management_server_port in config.yml
AC_HOST = "localhost"
AC_PROCESS_NAMES = {"acs.exe", "assettoCorsa.exe", "AssettoCorsa.exe"}


def _ok(msg: str):
    print(f"  [OK]  {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")


def _warn(msg: str):
    print(f"  [WARN] {msg}")


# -----------------------------------------------------------------------
# Individual checks
# -----------------------------------------------------------------------

def check_vjoy_dll() -> bool:
    if os.path.isfile(VJOY_DLL):
        _ok(f"vJoy DLL found: {VJOY_DLL}")
        return True
    _fail(f"vJoy DLL not found: {VJOY_DLL}")
    return False


def check_vjoy_device() -> bool:
    if not os.path.isfile(VJOY_DLL):
        _fail("vJoy DLL missing — cannot check device")
        return False
    try:
        dll = ctypes.CDLL(VJOY_DLL)
        acquired = dll.AcquireVJD(1)
        if acquired:
            dll.RelinquishVJD(1)
            _ok("vJoy Device 1: acquired and released OK")
            return True
        else:
            _fail("vJoy Device 1: AcquireVJD(1) returned False — is another process holding it?")
            return False
    except Exception as e:
        _fail(f"vJoy Device 1: DLL call failed — {e}")
        return False


def check_vjoy_ini() -> bool:
    # ISSUE-010: vJoy controls work in JOYSTICK mode despite WHEEL being preferred.
    # This check is demoted to a warning — it will never block collection.
    if not os.path.isfile(VJOY_INI):
        _warn(f"Vjoy.ini not found: {VJOY_INI} (ISSUE-010 — low priority)")
        return True
    with open(VJOY_INI) as f:
        content = f.read()
    for line in content.splitlines():
        if line.strip().startswith("INPUT_METHOD"):
            value = line.split("=", 1)[1].strip()
            if value == "WHEEL":
                _ok(f"Vjoy.ini INPUT_METHOD=WHEEL  ✓")
            else:
                _warn(
                    f"Vjoy.ini INPUT_METHOD={value}  (WHEEL preferred — ISSUE-010, low priority, "
                    f"controls working anyway)"
                )
            return True
    _warn("Vjoy.ini: INPUT_METHOD not found (ISSUE-010)")
    return True


def check_ac_process() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq acs.exe", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        for name in AC_PROCESS_NAMES:
            if any(name.lower() in l.lower() for l in lines):
                _ok(f"AC process running ({name})")
                return True
        _fail("AC process not found — start Assetto Corsa first")
        return False
    except Exception as e:
        _warn(f"Could not check AC process: {e}")
        return False


def check_port(port: int, label: str, timeout: float = 1.5) -> bool:
    """Try to receive one UDP packet on port (the plugin sends on every frame)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        # Send a probe; plugin replies when it sees a client
        sock.sendto(b"connect", (AC_HOST, port))
        try:
            data, _ = sock.recvfrom(65536)
            if data:
                _ok(f"{label} (port {port}): responded ({len(data)} bytes)")
                # Send disconnect so we don't leave a dangling session
                sock.sendto(b"disconnect", (AC_HOST, port))
                sock.close()
                return True
        except socket.timeout:
            pass
        sock.close()
        _fail(
            f"{label} (port {port}): no response — "
            "is AC running with the sensors_par plugin and a session active?"
        )
        return False
    except Exception as e:
        _fail(f"{label} (port {port}): error — {e}")
        return False


def check_mgmt_port(timeout: float = 2.0) -> bool:
    """TCP probe for the simulation management server (port 2347)."""
    try:
        with socket.create_connection((AC_HOST, AC_MGMT_PORT), timeout=timeout) as s:
            s.sendall(b"get_static_info")
            s.settimeout(timeout)
            data = s.recv(1048576)
            if data:
                _ok(f"AC management server (port {AC_MGMT_PORT}): responded")
                return True
        _fail(f"AC management server (port {AC_MGMT_PORT}): connected but no response")
        return False
    except ConnectionRefusedError:
        _fail(
            f"AC management server (port {AC_MGMT_PORT}): connection refused — "
            "is AC running with a session active (not in main menu)?"
        )
        return False
    except socket.timeout:
        _fail(
            f"AC management server (port {AC_MGMT_PORT}): timed out — "
            "session may not be started yet"
        )
        return False
    except Exception as e:
        _fail(f"AC management server (port {AC_MGMT_PORT}): {e}")
        return False


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

CHECKS = [
    ("vJoy DLL",            check_vjoy_dll),
    ("vJoy Device 1",       check_vjoy_device),
    ("Vjoy.ini WHEEL mode", check_vjoy_ini),
    ("AC process",          check_ac_process),
    ("AC ego port",         lambda: check_port(AC_EGO_PORT, "AC ego server")),
    ("AC mgmt port",        check_mgmt_port),
]


def run_preflight(warn_only: bool = False) -> bool:
    """
    Run all checks. Returns True if all pass.

    Parameters
    ----------
    warn_only : bool
        If True, print failures but do not raise SystemExit.

    Returns
    -------
    bool — True if all checks passed
    """
    print("\n=== Preflight checks ===")
    passed = []
    for name, fn in CHECKS:
        result = fn()
        passed.append(result)
    print()

    all_ok = all(passed)
    n_fail = sum(1 for p in passed if not p)

    if all_ok:
        print("All checks passed. Ready to collect.\n")
    else:
        print(f"{n_fail}/{len(CHECKS)} check(s) failed.")
        if not warn_only:
            print("Fix the issues above and re-run preflight.py.\n")
            sys.exit(1)
        print("Running in warn-only mode — continuing anyway.\n")

    return all_ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Pre-collection sanity checks")
    parser.add_argument("--warn-only", action="store_true",
                        help="Print failures but don't exit with error")
    args = parser.parse_args()
    run_preflight(warn_only=args.warn_only)
