"""
watch_controls.py — Lightweight AC controls monitor.

Connects directly to the AC plugin UDP server (no vJoy, no env setup)
and prints a live view of steer / throttle / brake as AC reports them.

Usage (AC must be running with sensors_par plugin loaded):
    .\\AssetoCorsa\\Scripts\\python.exe tests/watch_controls.py
    .\\AssetoCorsa\\Scripts\\python.exe tests/watch_controls.py --port 2345 --host localhost

Can run alongside collect.py — uses the "observe" handshake which never touches the control path.
"""

import socket
import time
import sys
import argparse

MAX_MSG_SIZE = 2 ** 18

# ANSI
CLEAR  = "\033[2J\033[H"
BOLD   = "\033[1m"
RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def bar(val, lo, hi, width=30):
    frac = max(0.0, min(1.0, (val - lo) / (hi - lo))) if hi != lo else 0.0
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def connect(host, port, timeout=10.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    deadline = time.perf_counter() + timeout
    print(f"Connecting to AC plugin at {host}:{port} ...")
    while time.perf_counter() < deadline:
        try:
            sock.sendto(b"observe", (host, port))
            data, addr = sock.recvfrom(MAX_MSG_SIZE)
            msg = data.decode()
            if msg == "identified":
                print(f"Connected as observer ({addr}).")
                sock.settimeout(2.0)
                return sock, addr
        except (socket.timeout, ConnectionResetError):
            # timeout  = no one listening yet, retry
            # ConnectionResetError (WinError 10054) = ICMP port unreachable,
            #   Windows UDP raises this when the port is closed — retry
            time.sleep(0.5)
            continue
    sock.close()
    raise TimeoutError(f"AC plugin did not respond on {host}:{port} within {timeout}s")


def recv_latest_state(sock):
    """Block until the next packet arrives, then drain any queued packets
    behind it so we always return the most recent state (never stale)."""
    # 1. Block until at least one packet is available
    sock.settimeout(2.0)
    data, _ = sock.recvfrom(MAX_MSG_SIZE)

    # 2. Drain the rest of the OS buffer with non-blocking reads
    sock.setblocking(False)
    try:
        while True:
            newer, _ = sock.recvfrom(MAX_MSG_SIZE)
            data = newer   # keep replacing until buffer is empty
    except BlockingIOError:
        pass   # buffer empty — data is now the latest packet
    finally:
        sock.settimeout(2.0)

    msg = data.decode()
    if msg in ("disconnect", "identified"):
        return None
    return eval(msg)   # AC plugin sends Python dict literals


def render(state, step, fps, steer_min, steer_max):
    steer  = float(state.get("steerAngle",  0.0))   # degrees, raw
    acc    = float(state.get("accStatus",   0.0))   # 0–1
    brake  = float(state.get("brakeStatus", 0.0))   # 0–1
    speed  = float(state.get("speed",       0.0)) * 3.6   # m/s → km/h
    gear   = int(  state.get("actualGear",  0))
    nsp    = float(state.get("NormalizedSplinePosition", 0.0))

    steer_c = RED if abs(steer) > 300 else (YELLOW if abs(steer) > 150 else GREEN)
    acc_c   = GREEN if acc > 0.05 else DIM
    brake_c = RED   if brake > 0.05 else DIM

    direction = f"{RED}◄ LEFT {RESET}" if steer < -5 else (f"{GREEN} RIGHT ►{RESET}" if steer > 5 else f"{DIM}CENTER{RESET}")

    lines = [
        f"{BOLD}{CYAN}══ AC Controls Monitor  step={step:6d}  fps={fps:5.1f} ══{RESET}",
        "",
        f"{BOLD}STEER{RESET}    {steer_c}{steer:+8.1f} deg (raw){RESET}   {direction}",
        f"         {bar(steer, -450, 450)}",
        f"         {DIM}session min={steer_min:+.1f} deg   max={steer_max:+.1f} deg   "
        f"observed range={abs(steer_min - steer_max):.1f} deg{RESET}",
        "",
        f"{BOLD}THROTTLE{RESET} {acc_c}{acc:8.3f} (raw 0–1){RESET}",
        f"         {bar(acc, 0, 1)}",
        "",
        f"{BOLD}BRAKE{RESET}    {brake_c}{brake:8.3f} (raw 0–1){RESET}",
        f"         {bar(brake, 0, 1)}",
        "",
        f"{DIM}speed={speed:.1f} km/h   gear={gear}   nsp={nsp:.4f}{RESET}",
        "",
        f"{DIM}Steer to full lock both ways to find physical max.  Ctrl+C to exit{RESET}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AC controls live monitor (no vJoy)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2345)
    args = parser.parse_args()

    sock, addr = connect(args.host, args.port)

    # Drain packets that queued up during the connect handshake
    print("Draining stale buffer...")
    sock.setblocking(False)
    drained = 0
    try:
        while True:
            sock.recvfrom(MAX_MSG_SIZE)
            drained += 1
    except BlockingIOError:
        pass
    sock.settimeout(2.0)
    print(f"Drained {drained} stale packets. Now live.")

    step       = 0
    fps        = 0.0
    t_last     = time.perf_counter()
    steer_min  =  9999.0
    steer_max  = -9999.0

    try:
        while True:
            state = recv_latest_state(sock)
            if state is None:
                print("AC disconnected.")
                break

            now    = time.perf_counter()
            dt     = now - t_last
            fps    = 1.0 / dt if dt > 0 else fps
            t_last = now
            step  += 1

            steer = float(state.get("steerAngle", 0.0))
            steer_min = min(steer_min, steer)
            steer_max = max(steer_max, steer)

            sys.stdout.write(CLEAR + render(state, step, fps, steer_min, steer_max) + "\n")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        try:
            sock.sendto(b"unobserve", (args.host, args.port))
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
