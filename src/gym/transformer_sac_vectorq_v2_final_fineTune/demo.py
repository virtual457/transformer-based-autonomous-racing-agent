"""
demo.py — Vector Q v2 FINAL: deterministic driving demo.

Loads the checkpoint, drives the car in Assetto Corsa in deterministic
mode (tanh(mean), no sampling), records a screen-capture video and a
per-frame telemetry CSV. Runs until the car crashes / goes off-track
or the user hits Ctrl+C.

Supports multi-episode benchmark runs via `--episodes N`. Each episode
gets its own subfolder `ep00/`, `ep01/`, … and an aggregate summary is
written at the run root.

Outputs saved under: gym/transformer_sac_vectorq_v2_final_fineTune/demo_runs/<timestamp>/
  - ep{NN}/video.mp4          (screen capture of AC window)
  - ep{NN}/telemetry.csv      (per-frame action + sensor data)
  - ep{NN}/summary.txt        (final stats for that episode)
  - aggregate_summary.txt     (mean/std/min/max across all episodes)

Requires: `mss` and `opencv-python` for video capture. If either is
missing, demo still runs and saves telemetry, but no video.

Run command:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2_final_fineTune/demo.py --manage-ac --episodes 5
"""

import sys
import os
import argparse
import logging
import time
import csv
import math
import threading
import collections
from datetime import datetime

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vectorq_v2_final.demo")

TOKEN_DIM   = 50
WINDOW_SIZE = 75
STATIONARY_FRAMES      = 60
STATIONARY_THRESHOLD_M = 0.5

TRANSFORMER_CONFIG = {
    "token_dim": TOKEN_DIM, "action_dim": 3, "window_size": WINDOW_SIZE,
    "d_model": 256, "n_heads": 4, "n_layers": 4, "ffn_dim": 1024,
    "policy_hidden": [256], "q_hidden": [256],
}
SAC_HYPERPARAMS = {
    "lr": 3e-4, "gamma": 0.992, "tau": 0.005, "target_entropy": -2.0,
}

CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
DEMO_DIR       = os.path.join(_THIS_DIR, "demo_runs")


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def build_env(config_path: str, work_dir: str):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    cfg.AssettoCorsa.enable_out_of_track_termination = False
    cfg.AssettoCorsa.add_previous_obs_to_state       = False
    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    os.makedirs(work_dir, exist_ok=True)
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    from our_env import OurEnv
    our_env_cfg = OmegaConf.create({"our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)})
    return OurEnv(ac_env, our_env_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Video recorder (background thread using mss + cv2)
# ──────────────────────────────────────────────────────────────────────────────

_AC_WINDOW_TITLES = [
    "Assetto Corsa",
    "AssettoCorsa",
    "assettocorsa",
]


def _find_ac_window_rect():
    """
    Locate the AC rendering window and return its client-area rect as
    a dict compatible with mss.grab: {"top","left","width","height"}.
    Returns None if the window is not found or not on a visible monitor.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = wintypes.HWND

    hwnd = 0
    for title in _AC_WINDOW_TITLES:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            break
    if not hwnd:
        return None

    client = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(client)):
        return None

    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None

    w = client.right - client.left
    h = client.bottom - client.top
    if w <= 0 or h <= 0:
        return None
    return {"top": int(pt.y), "left": int(pt.x), "width": int(w), "height": int(h)}


class VideoRecorder:
    """
    Background screen-capture recorder.

    If `window_title` matches the AC window, captures only the client area
    of that window (follows moves/resizes each frame). Otherwise falls back
    to the given `monitor_idx`. Gracefully no-ops if mss/cv2 are missing.
    """

    def __init__(self, out_path: str, fps: int = 30, monitor_idx: int = 1,
                 capture_window: bool = True):
        self.out_path = out_path
        self.fps = fps
        self.monitor_idx = monitor_idx
        self.capture_window = capture_window
        self._thread = None
        self._stop = threading.Event()
        self._available = False
        self._writer = None

        try:
            import mss
            import cv2
            self._mss_mod = mss
            self._cv2 = cv2
            self._available = True
        except ImportError as exc:
            logger.warning(
                f"Video capture disabled — missing dependency ({exc}). "
                f"Install with: pip install mss opencv-python"
            )

    def start(self) -> None:
        if not self._available:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Video recording started: {self.out_path}")

    def _resolve_region(self, sct):
        """Return the region dict to capture this frame."""
        if self.capture_window:
            rect = _find_ac_window_rect()
            if rect is not None:
                return rect
        monitors = sct.monitors
        idx = self.monitor_idx if self.monitor_idx < len(monitors) else 1
        return monitors[idx]

    def _run(self) -> None:
        cv2 = self._cv2
        with self._mss_mod.mss() as sct:
            # Resolve initial region to fix the output video size.
            region = self._resolve_region(sct)
            w, h = int(region["width"]), int(region["height"])
            mode = "ac-window" if self.capture_window and _find_ac_window_rect() else "monitor"
            logger.info(
                f"Video recorder: mode={mode}  size={w}x{h}  "
                f"top={region.get('top')}  left={region.get('left')}"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))
            frame_interval = 1.0 / self.fps
            next_t = time.perf_counter()
            while not self._stop.is_set():
                # Re-resolve each frame so the window can move; resize if needed.
                region = self._resolve_region(sct)
                img = np.array(sct.grab(region))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                self._writer.write(frame)
                next_t += frame_interval
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.perf_counter()
            self._writer.release()
            self._writer = None

    def stop(self) -> None:
        if not self._available:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info(f"Video recording stopped: {self.out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry logger
# ──────────────────────────────────────────────────────────────────────────────

_TELEMETRY_FIELDS = [
    "frame", "game_time_ms",
    "action_steer", "action_throttle", "action_brake",
    "speed", "gap", "yaw_error", "lap_dist",
    "world_x", "world_y",
    "r_progress", "r_speed", "r_gap_abs", "r_smoothness", "r_yaw", "r_crash",
    "reward_scalar", "out_of_track",
]


class TelemetryLogger:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=_TELEMETRY_FIELDS)
        self._w.writeheader()

    def log(self, frame: int, action: np.ndarray, info: dict,
            reward_scalar: float) -> None:
        comp = info.get("reward_components", {}) if isinstance(info, dict) else {}
        row = {
            "frame":            frame,
            "game_time_ms":     float(info.get("currentTime", 0.0)) if isinstance(info, dict) else 0.0,
            "action_steer":     float(action[0]),
            "action_throttle":  float(action[1]),
            "action_brake":     float(action[2]),
            "speed":            float(info.get("speed", info.get("speed_kmh", 0.0))) if isinstance(info, dict) else 0.0,
            "gap":              float(info.get("gap", 0.0)) if isinstance(info, dict) else 0.0,
            "yaw_error":        float(info.get("yaw_error", 0.0)) if isinstance(info, dict) else 0.0,
            "lap_dist":         float(info.get("LapDist", 0.0)) if isinstance(info, dict) else 0.0,
            "world_x":          float(info.get("world_position_x", 0.0)) if isinstance(info, dict) else 0.0,
            "world_y":          float(info.get("world_position_y", 0.0)) if isinstance(info, dict) else 0.0,
            "r_progress":       float(comp.get("r_progress",   0.0)),
            "r_speed":          float(comp.get("r_speed",      0.0)),
            "r_gap_abs":        float(comp.get("r_gap_abs",    0.0)),
            "r_smoothness":     float(comp.get("r_smoothness", 0.0)),
            "r_yaw":            float(comp.get("r_yaw",        0.0)),
            "r_crash":          float(comp.get("r_crash",      0.0)),
            "reward_scalar":    reward_scalar,
            "out_of_track":     int(bool(info.get("out_of_track", False))) if isinstance(info, dict) else 0,
        }
        self._w.writerow(row)

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vector Q v2 Final — deterministic demo drive"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'config.yml'),
    )
    parser.add_argument("--device",         type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--manage-ac",      action="store_true")
    parser.add_argument("--no-ai-drive",    action="store_true")
    parser.add_argument("--no-video",       action="store_true",
                        help="Disable screen-capture video.")
    parser.add_argument("--fps",            type=int, default=30)
    parser.add_argument("--monitor",        type=int, default=1,
                        help="mss monitor index (1 = primary), used as fallback.")
    parser.add_argument("--full-monitor",   action="store_true",
                        help="Record the whole monitor instead of just the AC window.")
    parser.add_argument("--max-frames",     type=int, default=0,
                        help="Hard cap on frames (0 = unlimited).")
    parser.add_argument("--stochastic",     action="store_true",
                        help="Sample actions instead of tanh(mean).")
    parser.add_argument("--episodes",       type=int, default=1,
                        help="Number of demo episodes to run back-to-back.")
    parser.add_argument("--oot-frames",     type=int, default=1,
                        help="Consecutive off-track frames that end an episode "
                             "(1 = strict demo, 100 = benchmark-style).")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    logger.info(f"Device: {device}  deterministic={not args.stochastic}")

    # ── Output folder ────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(DEMO_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Demo run: {run_dir}  episodes={args.episodes}")

    # ── AC launch ────────────────────────────────────────────────────────────
    if args.manage_ac:
        from ac_lifecycle import full_cycle
        full_cycle(max_retries=3)

    config_path = os.path.abspath(args.config)
    work_dir    = os.path.join(_THIS_DIR, '..', '..', 'outputs', 'transformer_sac_vectorq_v2_final_fineTune_demo')
    env = build_env(config_path, work_dir)

    # ── SAC (inference only) ─────────────────────────────────────────────────
    from transformer_sac_vectorq_v2_final_fineTune.sac import TransformerSAC

    sac = TransformerSAC(**TRANSFORMER_CONFIG, **SAC_HYPERPARAMS, device=device)
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.isfile(latest_ckpt):
        logger.error(f"No checkpoint at {latest_ckpt} — cannot demo. Abort.")
        env.close()
        return
    logger.info(f"Loading checkpoint: {latest_ckpt}")
    sac.load(latest_ckpt)

    # ── Drive N episodes ─────────────────────────────────────────────────────
    all_stats = []  # list of per-episode dicts for aggregate summary
    interrupted = False

    try:
        for ep in range(args.episodes):
            ep_dir = os.path.join(run_dir, f"ep{ep:02d}")
            os.makedirs(ep_dir, exist_ok=True)
            video_path     = os.path.join(ep_dir, "video.mp4")
            telemetry_path = os.path.join(ep_dir, "telemetry.csv")
            summary_path   = os.path.join(ep_dir, "summary.txt")
            logger.info(
                f"\n{'=' * 60}\n  EPISODE {ep + 1}/{args.episodes}  →  {ep_dir}\n"
                f"{'=' * 60}"
            )

            # Per-episode video + telemetry
            video = None
            if not args.no_video:
                video = VideoRecorder(
                    video_path, fps=args.fps, monitor_idx=args.monitor,
                    capture_window=not args.full_monitor,
                )
            telemetry = TelemetryLogger(telemetry_path)

            # Per-episode state
            frame           = 0
            ep_reward       = 0.0
            positive_frames = 0
            t_start         = time.perf_counter()
            pos_window: collections.deque = collections.deque(maxlen=STATIONARY_FRAMES)
            terminated_by   = None

            # Track-following / quality metrics (mirrors benchmark_backups.run_episode)
            speed_sum         = 0.0
            speed_max         = 0.0
            gap_abs_sum       = 0.0
            gap_max           = 0.0
            gap_count         = 0
            yaw_abs_sum       = 0.0
            yaw_count         = 0
            oot_frames        = 0
            consecutive_oot   = 0
            max_consec_oot    = 0
            prev_oot          = False
            recovery_count    = 0
            action_prev       = None
            sum_abs_delta     = np.zeros(3, dtype=np.float64)
            delta_count       = 0

            # Cumulative driven distance with lap-wrap handling
            initial_lap_dist  = None
            current_lap_start = 0.0
            current_lap_peak  = 0.0
            cumulative_prior  = 0.0
            laps_completed    = 0
            prev_lap_dist     = None
            lap_dist_end      = 0.0
            best_lap_s        = 0.0
            LAP_WRAP_DROP_THRESHOLD = 2000.0

            # Reward-component breakdown (summed across episode)
            rcomp_sums = {
                "r_progress":   0.0, "r_speed":    0.0, "r_gap_abs": 0.0,
                "r_smoothness": 0.0, "r_yaw":      0.0, "r_crash":   0.0,
            }

            try:
                if args.manage_ac and not args.no_ai_drive:
                    try:
                        from ac_lifecycle import randomize_start_position
                        randomize_start_position(wait_s=25.0)
                    except Exception as exc:
                        logger.warning(f"randomize_start_position failed: {exc}")

                obs, _info = env.reset()
                obs_deque: collections.deque = collections.deque(maxlen=WINDOW_SIZE)

                # Warmup: full throttle to fill deque
                warmup_action = np.array([0.0, 1.0, -1.0], dtype=np.float32)
                warmup_done = False
                for _ in range(WINDOW_SIZE):
                    obs, _r, wdone, _info = env.step(warmup_action)
                    env._read_latest_state()
                    obs_deque.append(obs[:TOKEN_DIM].astype(np.float32))
                    if wdone:
                        warmup_done = True
                        break

                if video is not None:
                    video.start()

                done = bool(warmup_done)
                while not done:
                    if args.max_frames and frame >= args.max_frames:
                        terminated_by = "max_frames"
                        break

                    # Build window
                    W = WINDOW_SIZE
                    available = list(obs_deque)
                    if len(available) == W:
                        window = np.stack(available, axis=0)
                    else:
                        pad = [available[0]] * (W - len(available))
                        window = np.stack(pad + available, axis=0)

                    # Inference (deterministic)
                    action, mean, _std = sac.select_action(
                        window, deterministic=not args.stochastic,
                    )

                    # Per-channel action smoothness
                    action_np = np.asarray(action, dtype=np.float64)
                    if action_prev is not None:
                        sum_abs_delta += np.abs(action_np - action_prev)
                        delta_count   += 1
                    action_prev = action_np

                    obs, reward, done, info = env.step(action)
                    env._read_latest_state()
                    obs_deque.append(obs[:TOKEN_DIM].astype(np.float32))

                    # Accumulate reward (scalar, for display)
                    r_scalar = (float(np.sum(reward)) if hasattr(reward, "__len__")
                                else float(reward))
                    ep_reward += r_scalar
                    if r_scalar > 0:
                        positive_frames += 1

                    telemetry.log(frame, action, info, r_scalar)
                    frame += 1

                    if isinstance(info, dict):
                        # Speed
                        sp = info.get("speed", info.get("speed_kmh"))
                        if sp is not None:
                            sp_f = float(sp)
                            speed_sum += sp_f
                            if sp_f > speed_max:
                                speed_max = sp_f

                        # LapDist + cumulative distance with lap-wrap
                        ld_raw = info.get("LapDist")
                        if ld_raw is not None:
                            lap_dist = float(ld_raw)
                            lap_dist_end = lap_dist
                            if initial_lap_dist is None:
                                initial_lap_dist  = lap_dist
                                current_lap_start = lap_dist
                                current_lap_peak  = lap_dist
                                prev_lap_dist     = lap_dist
                            else:
                                if (prev_lap_dist is not None and
                                        prev_lap_dist > 1000 and
                                        (prev_lap_dist - lap_dist) > LAP_WRAP_DROP_THRESHOLD):
                                    # Lap completed — wrap
                                    segment = prev_lap_dist - current_lap_start
                                    cumulative_prior += segment
                                    laps_completed  += 1
                                    current_lap_start = 0.0
                                    current_lap_peak  = lap_dist
                                if lap_dist > current_lap_peak:
                                    current_lap_peak = lap_dist
                                prev_lap_dist = lap_dist

                        # Gap to racing line (the "how well are we following the line" metric)
                        if "gap" in info:
                            g_abs = abs(float(info.get("gap", 0.0)))
                            gap_abs_sum += g_abs
                            if g_abs > gap_max:
                                gap_max = g_abs
                            gap_count   += 1

                        # Yaw error
                        if "yaw_error" in info:
                            yaw_abs_sum += abs(float(info.get("yaw_error", 0.0)))
                            yaw_count   += 1

                        # Best lap time
                        lap_t = float(info.get("LapTime", 0.0) or 0.0)
                        if lap_t > 1e-3 and (best_lap_s == 0.0 or lap_t < best_lap_s):
                            best_lap_s = lap_t

                        # Reward-component accumulation
                        comp = info.get("reward_components")
                        if isinstance(comp, dict):
                            for k in rcomp_sums:
                                if k in comp:
                                    rcomp_sums[k] += float(comp[k])

                        # OOT tracking (with configurable termination threshold)
                        is_oot_now = bool(info.get("out_of_track", False))
                        if is_oot_now:
                            oot_frames += 1
                            consecutive_oot += 1
                            if consecutive_oot > max_consec_oot:
                                max_consec_oot = consecutive_oot
                            if consecutive_oot >= args.oot_frames:
                                terminated_by = "out_of_track"
                                done = True
                        else:
                            consecutive_oot = 0
                        if (prev_oot and not is_oot_now
                                and terminated_by != "out_of_track"):
                            recovery_count += 1
                        prev_oot = is_oot_now

                        # Stationary termination
                        sx = info.get("world_position_x")
                        sy = info.get("world_position_y")
                        if sx is not None and sy is not None:
                            pos_window.append((float(sx), float(sy)))
                            if len(pos_window) == STATIONARY_FRAMES:
                                dx = pos_window[-1][0] - pos_window[0][0]
                                dy = pos_window[-1][1] - pos_window[0][1]
                                if math.sqrt(dx * dx + dy * dy) < STATIONARY_THRESHOLD_M:
                                    terminated_by = "stationary"
                                    done = True

                    if frame % 100 == 0:
                        elapsed = time.perf_counter() - t_start
                        logger.info(
                            f"ep{ep}  f={frame}  r={ep_reward:.1f}  "
                            f"r/f={ep_reward/max(frame,1):.3f}  "
                            f"pos%={100.0 * positive_frames / max(frame,1):.0f}  "
                            f"v={speed_sum/max(frame,1):.1f}/{speed_max:.1f}  "
                            f"gap={gap_abs_sum/max(gap_count,1):.2f}  "
                            f"t={elapsed:.0f}s"
                        )

                if terminated_by is None:
                    terminated_by = "env_done"

            finally:
                if video is not None:
                    video.stop()
                telemetry.close()

                elapsed = time.perf_counter() - t_start

                # Finalize derived stats
                current_segment = max(0.0, current_lap_peak - current_lap_start)
                total_dist      = cumulative_prior + current_segment
                pos_pct         = 100.0 * positive_frames / max(frame, 1)
                oot_pct         = 100.0 * oot_frames / max(frame, 1)
                reward_per_frame = ep_reward / max(frame, 1)
                speed_mean      = speed_sum / max(frame, 1)
                mean_abs_gap    = gap_abs_sum / max(gap_count, 1)
                mean_abs_yaw    = yaw_abs_sum / max(yaw_count, 1)
                if delta_count > 0:
                    smooth = sum_abs_delta / delta_count
                else:
                    smooth = np.zeros(3)
                smoothness_steer    = float(smooth[0])
                smoothness_throttle = float(smooth[1])
                smoothness_brake    = float(smooth[2])

                summary = (
                    f"Demo run: {run_id}  episode {ep + 1}/{args.episodes}\n"
                    f"Checkpoint: {latest_ckpt}\n"
                    f"Deterministic: {not args.stochastic}\n"
                    f"Terminated by: {terminated_by}\n"
                    f"Duration: {elapsed:.1f} s  |  Frames: {frame}\n"
                    f"\n"
                    f"── Reward ─────────────────────────────\n"
                    f"  total reward        : {ep_reward:.3f}\n"
                    f"  reward / frame      : {reward_per_frame:.4f}\n"
                    f"  positive-reward %   : {pos_pct:.1f}\n"
                    f"  components  r_progress={rcomp_sums['r_progress']:.2f}  "
                    f"r_speed={rcomp_sums['r_speed']:.2f}  "
                    f"r_gap_abs={rcomp_sums['r_gap_abs']:.2f}  "
                    f"r_smooth={rcomp_sums['r_smoothness']:.2f}  "
                    f"r_yaw={rcomp_sums['r_yaw']:.2f}  "
                    f"r_crash={rcomp_sums['r_crash']:.2f}\n"
                    f"\n"
                    f"── Progress ───────────────────────────\n"
                    f"  total driven dist   : {total_dist:.2f} m\n"
                    f"  laps completed      : {laps_completed}\n"
                    f"  final LapDist       : {lap_dist_end:.2f}\n"
                    f"  best lap time       : {best_lap_s:.3f} s\n"
                    f"\n"
                    f"── Track-following quality ────────────\n"
                    f"  mean |gap|          : {mean_abs_gap:.3f} m  (max {gap_max:.3f})\n"
                    f"  mean |yaw_error|    : {mean_abs_yaw:.4f} rad\n"
                    f"  smoothness steer    : {smoothness_steer:.4f}\n"
                    f"  smoothness throttle : {smoothness_throttle:.4f}\n"
                    f"  smoothness brake    : {smoothness_brake:.4f}\n"
                    f"\n"
                    f"── Speed ──────────────────────────────\n"
                    f"  mean speed          : {speed_mean:.2f}\n"
                    f"  max speed           : {speed_max:.2f}\n"
                    f"\n"
                    f"── OOT ────────────────────────────────\n"
                    f"  oot frames          : {oot_frames}  ({oot_pct:.1f}%)\n"
                    f"  max consec OOT      : {max_consec_oot}\n"
                    f"  recoveries          : {recovery_count}\n"
                    f"  OOT termination @   : {args.oot_frames} consec frames\n"
                    f"\n"
                    f"Video: {video_path if video and video._available else 'disabled'}\n"
                    f"Telemetry: {telemetry_path}\n"
                )
                with open(summary_path, "w", encoding="utf-8") as fp:
                    fp.write(summary)
                logger.info("\n" + summary)

                all_stats.append({
                    "episode":             ep,
                    "frames":              frame,
                    "duration_s":          elapsed,
                    "ep_reward":           ep_reward,
                    "reward_per_frame":    reward_per_frame,
                    "pos_pct":             pos_pct,
                    "total_dist":          total_dist,
                    "laps_completed":      laps_completed,
                    "lap_dist_end":        lap_dist_end,
                    "best_lap_s":          best_lap_s,
                    "mean_abs_gap":        mean_abs_gap,
                    "gap_max":             gap_max,
                    "mean_abs_yaw":        mean_abs_yaw,
                    "smoothness_steer":    smoothness_steer,
                    "smoothness_throttle": smoothness_throttle,
                    "smoothness_brake":    smoothness_brake,
                    "speed_mean":          speed_mean,
                    "speed_max":           speed_max,
                    "oot_frames":          oot_frames,
                    "oot_pct":             oot_pct,
                    "max_consec_oot":      max_consec_oot,
                    "recovery_count":      recovery_count,
                    "terminated_by":       terminated_by,
                    "r_progress":          rcomp_sums["r_progress"],
                    "r_speed":             rcomp_sums["r_speed"],
                    "r_gap_abs":           rcomp_sums["r_gap_abs"],
                    "r_smoothness":        rcomp_sums["r_smoothness"],
                    "r_yaw":               rcomp_sums["r_yaw"],
                    "r_crash":             rcomp_sums["r_crash"],
                })

    except KeyboardInterrupt:
        interrupted = True
        logger.info("Demo interrupted by user.")
    finally:
        # Aggregate summary across episodes
        if all_stats:
            import statistics as _stats
            def _mean(xs): return float(_stats.mean(xs)) if xs else 0.0
            def _stdev(xs): return float(_stats.stdev(xs)) if len(xs) > 1 else 0.0
            def _col(k): return [s[k] for s in all_stats]

            term_counts: dict = {}
            for s in all_stats:
                term_counts[s["terminated_by"]] = term_counts.get(s["terminated_by"], 0) + 1

            # Per-episode CSV (one row per episode, all raw stats)
            per_ep_csv = os.path.join(run_dir, "per_episode.csv")
            csv_fields = [
                "episode", "frames", "duration_s",
                "ep_reward", "reward_per_frame", "pos_pct",
                "total_dist", "laps_completed", "lap_dist_end", "best_lap_s",
                "mean_abs_gap", "gap_max", "mean_abs_yaw",
                "smoothness_steer", "smoothness_throttle", "smoothness_brake",
                "speed_mean", "speed_max",
                "oot_frames", "oot_pct", "max_consec_oot", "recovery_count",
                "terminated_by",
                "r_progress", "r_speed", "r_gap_abs",
                "r_smoothness", "r_yaw", "r_crash",
            ]
            with open(per_ep_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=csv_fields)
                w.writeheader()
                for s in all_stats:
                    w.writerow({k: s[k] for k in csv_fields})

            # Compact per-episode table for the txt summary
            header_line = (
                f"{'ep':>3} {'frames':>7} {'dur_s':>6} {'reward':>8} {'r/f':>6} "
                f"{'dist':>8} {'laps':>4} {'gap':>6} {'smo_s':>6} {'smo_t':>6} "
                f"{'smo_b':>6} {'v_mean':>6} {'v_max':>6} "
                f"{'oot%':>5} {'maxOOT':>6} {'rcov':>4} {'term':<14}"
            )
            rows = []
            for s in all_stats:
                rows.append(
                    f"{s['episode']:>3} {s['frames']:>7d} {s['duration_s']:>6.1f} "
                    f"{s['ep_reward']:>8.1f} {s['reward_per_frame']:>6.3f} "
                    f"{s['total_dist']:>8.1f} {s['laps_completed']:>4d} "
                    f"{s['mean_abs_gap']:>6.2f} "
                    f"{s['smoothness_steer']:>6.3f} {s['smoothness_throttle']:>6.3f} "
                    f"{s['smoothness_brake']:>6.3f} "
                    f"{s['speed_mean']:>6.2f} {s['speed_max']:>6.2f} "
                    f"{s['oot_pct']:>5.1f} {s['max_consec_oot']:>6d} "
                    f"{s['recovery_count']:>4d} {s['terminated_by']:<14}"
                )

            # Aggregate means ± stdev
            def _ms(key, fmt=".2f"):
                xs = _col(key)
                return f"{_mean(xs):{fmt}}±{_stdev(xs):{fmt}}"

            agg_block = (
                f"\n{'-' * 120}\n"
                f"MEAN ± STDEV over {len(all_stats)} episodes:\n"
                f"  reward              : {_ms('ep_reward')}   r/frame: {_ms('reward_per_frame', '.4f')}\n"
                f"  total driven dist   : {_ms('total_dist')}  m\n"
                f"  laps completed (sum): {sum(_col('laps_completed'))}\n"
                f"  best lap (min > 0)  : "
                f"{min([b for b in _col('best_lap_s') if b > 0] or [0.0]):.3f} s\n"
                f"  mean |gap|          : {_ms('mean_abs_gap', '.3f')} m   (episode max: {max(_col('gap_max')):.3f})\n"
                f"  mean |yaw_error|    : {_ms('mean_abs_yaw', '.4f')} rad\n"
                f"  smoothness steer    : {_ms('smoothness_steer', '.4f')}\n"
                f"  smoothness throttle : {_ms('smoothness_throttle', '.4f')}\n"
                f"  smoothness brake    : {_ms('smoothness_brake', '.4f')}\n"
                f"  speed mean          : {_ms('speed_mean', '.2f')}\n"
                f"  speed max           : {_ms('speed_max', '.2f')}\n"
                f"  positive-reward %   : {_ms('pos_pct', '.1f')}\n"
                f"  OOT %               : {_ms('oot_pct', '.2f')}\n"
                f"  max consec OOT      : {_ms('max_consec_oot', '.0f')}\n"
                f"  recoveries          : {_ms('recovery_count', '.1f')}\n"
                f"  frames              : {_ms('frames', '.0f')}\n"
                f"  duration_s          : {_ms('duration_s', '.1f')}\n"
                f"\n"
                f"min/max:  reward=[{min(_col('ep_reward')):.1f}, {max(_col('ep_reward')):.1f}]  "
                f"dist=[{min(_col('total_dist')):.1f}, {max(_col('total_dist')):.1f}]  "
                f"gap=[{min(_col('mean_abs_gap')):.3f}, {max(_col('mean_abs_gap')):.3f}]\n"
                f"terminations: {term_counts}\n"
                f"OOT-termination threshold: {args.oot_frames} consec frames\n"
            )

            header = (
                f"Demo run: {run_id}\n"
                f"Episodes completed: {len(all_stats)}/{args.episodes}"
                f"{' (interrupted)' if interrupted else ''}\n"
                f"Checkpoint: {latest_ckpt}\n"
                f"Deterministic: {not args.stochastic}\n\n"
                f"{header_line}\n"
            )

            aggregate_text = header + "\n".join(rows) + agg_block
            agg_path = os.path.join(run_dir, "aggregate_summary.txt")
            with open(agg_path, "w", encoding="utf-8") as fp:
                fp.write(aggregate_text)
            logger.info("\n" + aggregate_text)
            logger.info(f"Aggregate summary: {agg_path}")
            logger.info(f"Per-episode CSV:  {per_ep_csv}")

        env.close()


if __name__ == "__main__":
    main()
