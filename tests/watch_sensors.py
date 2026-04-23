"""
watch_sensors.py — Real-time sensor/telemetry monitor for Assetto Corsa.

Connects to AC (same as collect.py), steps the simulation with zero actions,
and prints a live dashboard of every named sensor value each frame.

Usage (AC must be running):
    .\\AssetoCorsa\\Scripts\\python.exe tests/watch_sensors.py

Options:
    --skip-preflight    Skip vJoy/AC checks
    --hz N              Redraw rate cap (default: 25, matches AC physics rate)
    --raw               Also dump the raw state dict keys not shown in the panel

Press Ctrl+C to exit cleanly.
"""

import sys
import os
import time
import math
import argparse
import logging

# ── path bootstrap (mirrors test_our_env.py) ─────────────────────────────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_root, 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_root, 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.join(_root, 'gym'))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")

# ── ANSI helpers ──────────────────────────────────────────────────────────────
CLEAR  = "\033[2J\033[H"          # clear screen + cursor home
BOLD   = "\033[1m"
RESET  = "\033[0m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
WHITE  = "\033[97m"

def _color_value(val, lo, hi, warn_lo=None, warn_hi=None):
    """Return ANSI colour for a float value based on soft warn thresholds."""
    if warn_hi is not None and val >= warn_hi:
        return RED
    if warn_lo is not None and val <= warn_lo:
        return YELLOW
    return GREEN

def _bar(val, lo, hi, width=20):
    """ASCII progress bar for a value in [lo, hi]."""
    frac = max(0.0, min(1.0, (val - lo) / (hi - lo))) if hi != lo else 0.0
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"

# ── Display layout ────────────────────────────────────────────────────────────

def render(state: dict, step: int, fps: float, show_raw: bool):
    lines = []
    add = lines.append

    add(f"{BOLD}{CYAN}══ AC Sensor Monitor  step={step:5d}  fps={fps:4.1f} ══{RESET}")
    add("")

    # ── Car dynamics ──────────────────────────────────────────────────────────
    speed    = float(state.get("speed",             0.0))
    speed_kh = speed * 3.6
    gap      = float(state.get("gap",               0.0))
    accelX   = float(state.get("accelX",            0.0))
    accelY   = float(state.get("accelY",            0.0))
    rpm      = float(state.get("RPM",               0.0))
    gear     = int(  state.get("actualGear",        0))
    lap_dist = float(state.get("LapDist",           0.0))
    nsp      = float(state.get("NormalizedSplinePosition", 0.0))
    lap_t    = float(state.get("currentTime",       0.0))
    last_lap = float(state.get("lastLap",           0.0))
    lap_cnt  = int(  state.get("LapCount",          0))

    add(f"{BOLD}── CAR DYNAMICS ──────────────────────────────────────────────{RESET}")
    c = _color_value(speed_kh, 0, 200)
    target_speed_ms = float(state.get("_target_speed_ms", 0.0))
    target_speed_kh = target_speed_ms * 3.6
    add(f"  Speed        {c}{speed_kh:7.1f} km/h{RESET}  {speed:6.2f} m/s   {_bar(speed, 0, 80)}")
    tc = _color_value(abs(speed_kh - target_speed_kh), 0, 30, warn_hi=20.0)
    add(f"  Target speed {tc}{target_speed_kh:7.1f} km/h{RESET}  {target_speed_ms:6.2f} m/s   diff={speed_kh - target_speed_kh:+.1f} km/h")
    c = _color_value(abs(gap), 0, 5, warn_hi=3.0)
    add(f"  Gap (RL)     {c}{gap:+8.3f} m{RESET}    (+ = right of line)")
    add(f"  AccelX (lon) {YELLOW}{accelX:+7.2f} g{RESET}    (+ = braking)")
    add(f"  AccelY (lat) {YELLOW}{accelY:+7.2f} g{RESET}    (+ = left turn)")
    add(f"  RPM          {GREEN}{rpm:8.0f}{RESET}       {_bar(rpm, 0, 10000)}")
    add(f"  Gear         {WHITE}{gear:8d}{RESET}")
    add("")

    # ── Local velocity ────────────────────────────────────────────────────────
    vx = float(state.get("local_velocity_x", 0.0))
    vy = float(state.get("local_velocity_y", 0.0))
    wY = float(state.get("angular_velocity_y", 0.0))

    add(f"{BOLD}── LOCAL VELOCITY & YAW ──────────────────────────────────────{RESET}")
    add(f"  vel_x (fwd)  {GREEN}{vx:+8.2f} m/s{RESET}")
    c = _color_value(abs(vy), 0, 10, warn_hi=5.0)
    add(f"  vel_y (side) {c}{vy:+8.2f} m/s{RESET}  (sideslip indicator)")
    c = _color_value(abs(wY), 0, 3.14, warn_hi=2.0)
    add(f"  yaw rate     {c}{wY:+8.3f} rad/s{RESET}")
    add("")

    # ── Tyre slip angles ──────────────────────────────────────────────────────
    sa_fl = float(state.get("SlipAngle_fl", 0.0))
    sa_fr = float(state.get("SlipAngle_fr", 0.0))
    sa_rl = float(state.get("SlipAngle_rl", 0.0))
    sa_rr = float(state.get("SlipAngle_rr", 0.0))

    def slip_color(v):
        a = abs(v)
        if a > 12: return RED
        if a > 7:  return YELLOW
        return GREEN

    add(f"{BOLD}── TYRE SLIP ANGLES (deg) ────────────────────────────────────{RESET}")
    add(f"  FL {slip_color(sa_fl)}{sa_fl:+7.2f}°{RESET}   FR {slip_color(sa_fr)}{sa_fr:+7.2f}°{RESET}")
    add(f"  RL {slip_color(sa_rl)}{sa_rl:+7.2f}°{RESET}   RR {slip_color(sa_rr)}{sa_rr:+7.2f}°{RESET}")
    add("")

    # ── Ray sensors ───────────────────────────────────────────────────────────
    sensors = state.get("sensors", None)
    add(f"{BOLD}── RAY CAST WALL SENSORS (11 rays, 0°→180°) ──────────────────{RESET}")
    if sensors is not None:
        import numpy as np
        MAX_RAY_LEN = 150.0  # from sensors_ray_casting.py
        labels = ["R90", "R67", "R45", "R22", "R11", "FWD", "L11", "L22", "L45", "L67", "L90"]
        row = "  "
        for i, (lbl, d) in enumerate(zip(labels, sensors)):
            c = RED if d < 5.0 else (YELLOW if d < 15.0 else GREEN)
            row += f"{lbl}:{c}{d:5.1f}m{RESET}  "
            if i == 5:
                row += "\n  "
        add(row)
        dist_min = float(np.min(sensors))
        c = RED if dist_min < 5.0 else (YELLOW if dist_min < 15.0 else GREEN)
        add(f"  min wall dist {c}{dist_min:6.2f} m{RESET}")
    else:
        add("  (sensors not available)")
    add("")

    # ── Track position ────────────────────────────────────────────────────────
    wx = float(state.get("world_position_x", 0.0))
    wy = float(state.get("world_position_y", 0.0))
    yaw = float(state.get("yaw", 0.0))
    oot = state.get("out_of_track", False)
    n_oot = int(state.get("numberOfTyresOut", 0))

    add(f"{BOLD}── TRACK POSITION ────────────────────────────────────────────{RESET}")
    add(f"  LapDist      {WHITE}{lap_dist:8.1f} m{RESET}   NSP={nsp:.4f}   Lap#{lap_cnt}")
    add(f"  World XY     ({wx:9.2f}, {wy:9.2f})")
    add(f"  Yaw          {WHITE}{yaw:+8.3f} rad{RESET}  ({yaw * 57.296:+7.2f} deg)")
    yaw_err_deg = float(state.get("_yaw_error_deg", 0.0))
    yaw_c = RED if abs(yaw_err_deg) > 45 else (YELLOW if abs(yaw_err_deg) > 20 else GREEN)
    add(f"  Yaw error    {yaw_c}{yaw_err_deg:+8.2f} deg{RESET}  (car vs racing line heading)")
    oot_c = RED if oot else GREEN
    add(f"  Out of track {oot_c}{'YES  ⚠' if oot else 'no':6s}{RESET}  tyres_out={n_oot}/4")
    add("")

    # ── Lap timing ────────────────────────────────────────────────────────────
    add(f"{BOLD}── LAP TIMING ────────────────────────────────────────────────{RESET}")
    add(f"  Current lap  {WHITE}{lap_t:8.3f} s{RESET}")
    add(f"  Last lap     {WHITE}{last_lap:8.3f} s{RESET}")
    add("")

    # ── Controls feedback ─────────────────────────────────────────────────────
    steer  = float(state.get("steerAngle",  0.0))
    acc    = float(state.get("accStatus",   0.0))
    brake  = float(state.get("brakeStatus", 0.0))
    ff     = float(state.get("LastFF",      0.0))

    add(f"{BOLD}── CONTROLS FEEDBACK ─────────────────────────────────────────{RESET}")
    steer_c = YELLOW if abs(steer) > 200 else GREEN
    add(f"  Steer angle  {steer_c}{steer:+8.1f} deg{RESET}  {_bar(steer, -450, 450)}")
    add(f"  Throttle     {GREEN}{acc:8.3f}{RESET}         {_bar(acc, -1, 1)}")
    add(f"  Brake        {GREEN}{brake:8.3f}{RESET}         {_bar(brake, -1, 1)}")
    add(f"  LastFF       {DIM}{ff:+8.4f}{RESET}")
    add("")

    # ── Raw state dump (optional) ─────────────────────────────────────────────
    if show_raw:
        add(f"{BOLD}── RAW STATE KEYS ────────────────────────────────────────────{RESET}")
        skip = {"sensors", "rl_point"}   # array/noisy fields already shown above
        for k, v in sorted(state.items()):
            if k in skip:
                continue
            if isinstance(v, float):
                add(f"  {DIM}{k:<35s}{RESET} {v:.6g}")
            else:
                add(f"  {DIM}{k:<35s}{RESET} {v}")
        add("")

    # ── Timing budget ─────────────────────────────────────────────────────────
    wait_ms      = state.get("_timing_wait_ms",      0.0)
    compute_ms   = state.get("_timing_compute_ms",   0.0)
    remaining_ms = state.get("_timing_remaining_ms", 0.0)

    add(f"{BOLD}── TIMING (40ms budget per packet) ──────────────────────────{RESET}")
    add(f"  wait for UDP   {CYAN}{wait_ms:7.2f} ms{RESET}  (recvfrom blocked)")
    add(f"  expand_state   {GREEN}{compute_ms:7.2f} ms{RESET}  (ray-cast + gap)")
    rc = GREEN if remaining_ms > 10 else (YELLOW if remaining_ms > 0 else RED)
    add(f"  remaining      {rc}{remaining_ms:7.2f} ms{RESET}  ({'✅ safe' if remaining_ms > 10 else '⚠ tight' if remaining_ms > 0 else '❌ overrun'})")
    add("")

    add(f"{DIM}Ctrl+C to exit{RESET}")
    return "\n".join(lines)


# ── Socket drain ─────────────────────────────────────────────────────────────

def drain_udp_buffer(ac_env, drain_timeout_s=0.005):
    """
    Discard all stale UDP packets that queued up during slow AC init.

    Root cause: AC's plugin sends 25 packets/s from the moment it starts.
    make_ac_env() takes ~10s loading track/refline/grid files, so ~250 packets
    pile up in the OS socket buffer. Without draining, every recvfrom() returns
    instantly from the backlog and we're 10s behind real time.

    Fix: set a tiny timeout, read until it blocks (= buffer empty = real time).
    """
    sock = ac_env.client.socket
    if sock is None:
        return 0

    original_timeout = sock.gettimeout()
    sock.settimeout(drain_timeout_s)   # 5ms — real-time packet arrives every 40ms

    from AssettoCorsaEnv.ac_client import MAX_MSG_SIZE
    discarded = 0
    try:
        while True:
            sock.recvfrom(MAX_MSG_SIZE)   # discard
            discarded += 1
    except Exception:
        pass   # timeout = buffer empty, we're live

    sock.settimeout(original_timeout)
    return discarded


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time AC sensor monitor")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--hz",  type=float, default=25.0,
                        help="Target display refresh rate (default 25)")
    parser.add_argument("--raw", action="store_true",
                        help="Also print all raw state dict keys")
    parser.add_argument("--config", type=str,
                        default=os.path.join(_root, 'assetto_corsa_gym', 'config.yml'))
    args = parser.parse_args()

    if not args.skip_preflight:
        sys.path.insert(0, os.path.join(_root, 'gym'))
        try:
            from preflight import run_preflight
            run_preflight(warn_only=True)
        except Exception as e:
            print(f"Preflight warning: {e}")

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    from AssettoCorsaEnv.assettoCorsa import make_ac_env

    work_dir = os.path.join(_root, 'outputs', 'watch_sensors')
    os.makedirs(work_dir, exist_ok=True)

    print("Connecting to Assetto Corsa... (AC must be running)")
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    print(f"Connected. obs_dim={ac_env.state_dim}. Resetting...")

    ac_env.reset()

    print("Draining stale UDP buffer...")
    discarded = drain_udp_buffer(ac_env)
    print(f"Drained {discarded} stale packets. Now live.")

    frame_dt = 1.0 / args.hz

    step = 0
    t_last = time.perf_counter()
    fps = 0.0

    try:
        while True:
            t0 = time.perf_counter()

            # Read next UDP packet — blocks until AC sends (~40ms)
            t_recv = time.perf_counter()
            raw = ac_env.client.step_sim()
            t_after_recv = time.perf_counter()

            # Compute sensors from this packet's position/yaw
            ac_env.state, _ = ac_env.expand_state(raw)
            t_after_expand = time.perf_counter()

            state = ac_env.state

            # Target speed from racing line
            try:
                if getattr(ac_env.ref_lap, 'use_target_speed', False):
                    state["_target_speed_ms"] = float(
                        ac_env.ref_lap.get_target_speed_value(float(state.get("LapDist", 0.0)))
                    )
            except Exception:
                pass

            # Relative yaw: car heading vs racing-line heading at current LapDist
            try:
                import numpy as _np
                _lap_dist  = float(state.get("LapDist", 0.0))
                _car_yaw   = float(state.get("yaw", 0.0))
                _yaw_idx   = ac_env.ref_lap.channels_dist.index("yaw")
                _line_yaw  = float(_np.interp(
                    _lap_dist,
                    ac_env.ref_lap.distance_ch_dist,
                    ac_env.ref_lap.td[:, _yaw_idx],
                ))
                _yaw_err   = (_car_yaw - _line_yaw + math.pi) % (2 * math.pi) - math.pi
                state["_yaw_error_deg"] = math.degrees(_yaw_err)
            except Exception:
                state["_yaw_error_deg"] = 0.0

            wait_ms    = (t_after_recv   - t_recv)        * 1000
            compute_ms = (t_after_expand - t_after_recv)  * 1000
            budget_ms  = 40.0  # one AC tick window
            remaining_ms = budget_ms - wait_ms - compute_ms

            # FPS
            now = time.perf_counter()
            elapsed = now - t_last
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            t_last = now
            step += 1

            state["_timing_wait_ms"]      = wait_ms
            state["_timing_compute_ms"]   = compute_ms
            state["_timing_remaining_ms"] = remaining_ms

            # Render
            panel = render(state, step, fps, args.raw)
            sys.stdout.write(CLEAR + panel + "\n")
            sys.stdout.flush()

            # Sleep to cap redraw rate without overshooting
            used = time.perf_counter() - t0
            remaining = frame_dt - used
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nExiting watch_sensors.")
    finally:
        try:
            ac_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
