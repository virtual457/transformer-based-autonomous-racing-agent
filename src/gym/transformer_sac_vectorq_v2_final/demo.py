"""
demo.py — Vector Q v2 FINAL: deterministic driving demo.

Loads the checkpoint, drives the car in Assetto Corsa in deterministic
mode (tanh(mean), no sampling), records a screen-capture video and a
per-frame telemetry CSV. Runs until the car crashes / goes off-track
or the user hits Ctrl+C.

Outputs saved under: gym/transformer_sac_vectorq_v2_final/demo_runs/<timestamp>/
  - video.mp4            (screen capture of AC window)
  - telemetry.csv        (per-frame action + sensor data)
  - summary.txt          (final stats)

Requires: `mss` and `opencv-python` for video capture. If either is
missing, demo still runs and saves telemetry, but no video.

Run command:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2_final/demo.py --manage-ac
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
    "lr": 3e-4, "gamma": 0.992, "tau": 0.005, "target_entropy": -1.5,
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
    args = parser.parse_args()

    device = _resolve_device(args.device)
    logger.info(f"Device: {device}  deterministic={not args.stochastic}")

    # ── Output folder ────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(DEMO_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    video_path     = os.path.join(run_dir, "video.mp4")
    telemetry_path = os.path.join(run_dir, "telemetry.csv")
    summary_path   = os.path.join(run_dir, "summary.txt")
    logger.info(f"Demo run: {run_dir}")

    # ── AC launch ────────────────────────────────────────────────────────────
    if args.manage_ac:
        from ac_lifecycle import full_cycle
        full_cycle(max_retries=3)

    config_path = os.path.abspath(args.config)
    work_dir    = os.path.join(_THIS_DIR, '..', '..', 'outputs', 'transformer_sac_vectorq_v2_final_demo')
    env = build_env(config_path, work_dir)

    # ── SAC (inference only) ─────────────────────────────────────────────────
    from transformer_sac_vectorq_v2_final.sac import TransformerSAC

    sac = TransformerSAC(**TRANSFORMER_CONFIG, **SAC_HYPERPARAMS, device=device)
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.isfile(latest_ckpt):
        logger.error(f"No checkpoint at {latest_ckpt} — cannot demo. Abort.")
        env.close()
        return
    logger.info(f"Loading checkpoint: {latest_ckpt}")
    sac.load(latest_ckpt)

    # ── Video recorder ───────────────────────────────────────────────────────
    video = None
    if not args.no_video:
        video = VideoRecorder(video_path, fps=args.fps, monitor_idx=args.monitor,
                              capture_window=not args.full_monitor)

    # ── Telemetry ────────────────────────────────────────────────────────────
    telemetry = TelemetryLogger(telemetry_path)

    # ── Drive ────────────────────────────────────────────────────────────────
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
        for _ in range(WINDOW_SIZE):
            obs, _r, done, _info = env.step(warmup_action)
            env._read_latest_state()
            obs_deque.append(obs[:TOKEN_DIM].astype(np.float32))
            if done:
                break

        if video is not None:
            video.start()

        frame          = 0
        ep_reward      = 0.0
        positive_frames = 0
        t_start        = time.perf_counter()
        pos_window: collections.deque = collections.deque(maxlen=STATIONARY_FRAMES)
        oot_consec     = 0
        terminated_by  = None

        done = False
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

            obs, reward, done, info = env.step(action)
            env._read_latest_state()
            obs_deque.append(obs[:TOKEN_DIM].astype(np.float32))

            # Accumulate reward (scalar, for display)
            r_scalar = float(np.sum(reward)) if hasattr(reward, "__len__") else float(reward)
            ep_reward += r_scalar
            if r_scalar > 0:
                positive_frames += 1

            telemetry.log(frame, action, info, r_scalar)
            frame += 1

            # Stationary / OOT termination
            if isinstance(info, dict):
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
                if info.get("out_of_track"):
                    oot_consec += 1
                    if oot_consec >= 1:
                        terminated_by = "out_of_track"
                        done = True
                else:
                    oot_consec = 0

            if frame % 100 == 0:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f"frame={frame}  reward={ep_reward:.2f}  "
                    f"pos%={100.0 * positive_frames / max(frame,1):.1f}  "
                    f"elapsed={elapsed:.1f}s"
                )

        if terminated_by is None:
            terminated_by = "env_done"

    except KeyboardInterrupt:
        terminated_by = "user_interrupt"
        logger.info("Demo interrupted by user.")
    finally:
        if video is not None:
            video.stop()
        telemetry.close()

        elapsed = time.perf_counter() - t_start if "t_start" in locals() else 0.0
        summary = (
            f"Demo run: {run_id}\n"
            f"Terminated by: {terminated_by}\n"
            f"Frames: {frame if 'frame' in locals() else 0}\n"
            f"Duration: {elapsed:.1f} s\n"
            f"Total reward (scalar sum): {ep_reward if 'ep_reward' in locals() else 0.0:.3f}\n"
            f"Positive-reward frame %: "
            f"{100.0 * positive_frames / max(frame,1) if 'frame' in locals() and frame > 0 else 0.0:.1f}\n"
            f"Checkpoint: {latest_ckpt}\n"
            f"Video: {video_path if video and video._available else 'disabled'}\n"
            f"Telemetry: {telemetry_path}\n"
        )
        with open(summary_path, "w") as fp:
            fp.write(summary)
        logger.info("\n" + summary)

        env.close()


if __name__ == "__main__":
    main()
