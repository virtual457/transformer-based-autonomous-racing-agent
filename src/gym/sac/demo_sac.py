"""
demo_sac.py — Baseline SAC (MLP) deterministic driving demo.

Loads a checkpoint produced by `gym/sac/train_sac.py` (or any SAC-format
checkpoint with the same schema), drives the car in Assetto Corsa in
deterministic mode (tanh(mean), no sampling), and records per-frame
telemetry + an optional screen-capture video.

This is the baseline comparison demo for the Vector-Q Transformer SAC
variant — mirrors the CLI surface and output format of
`transformer_sac_vectorq_v2_final_fineTune/demo.py` so the two runs
produce directly comparable aggregate summaries.

IMPORTANT — config.yml is NOT modified on disk.
The baseline SAC was trained with `use_target_speed=False` and
`add_previous_obs_to_state=True`. The project's current `config.yml`
has `use_target_speed=true` (used by the Vector-Q pipeline). This
script loads the config file into memory with OmegaConf.load() and
applies the required overrides to the in-memory copy only — the file
on disk is never rewritten.

Outputs saved under:
  trained_models/SAC/<ckpt_tag>/demo_runs/<timestamp>/
    ep{NN}/video.mp4          (screen capture of AC window)
    ep{NN}/telemetry.csv      (per-frame action + sensor data)
    ep{NN}/summary.txt        (final stats for that episode)
    aggregate_summary.txt     (mean ± stdev across episodes)
    per_episode.csv           (one row per episode, all raw stats)

Run commands:
    # Default: 5 episodes on the 150_demo checkpoint.
    .\\AssetoCorsa\\Scripts\\python.exe gym/sac/demo_sac.py --manage-ac --episodes 5

    # Pick a different checkpoint (path or known tag under trained_models/SAC/).
    .\\AssetoCorsa\\Scripts\\python.exe gym/sac/demo_sac.py --manage-ac \\
        --checkpoint trained_models/SAC/smooth_operator/model.pt

    # Strict OOT termination (1 consec frame) vs benchmark-style grace (100 consec).
    .\\AssetoCorsa\\Scripts\\python.exe gym/sac/demo_sac.py --manage-ac --oot-frames 100
"""

import sys
import os
import argparse
import logging
import time
import csv
import math
import collections
from datetime import datetime

import numpy as np

# ── Path setup — mirror train_sac.py ──────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_GYM_DIR    = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT    = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT, "assetto_corsa_gym"))
sys.path.insert(0, os.path.join(_PROJECT, "assetto_corsa_gym", "assetto_corsa_gym"))
sys.path.insert(0, os.path.join(_PROJECT, "assetto_corsa_gym", "algorithm", "discor"))
sys.path.insert(0, _GYM_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline_sac.demo")

# ── SAC training config (must match train_sac.py SAC_HYPERPARAMS) ─────────────
SAC_HYPERPARAMS = {
    "obs_dim":        125,
    "action_dim":     3,
    "hidden_units":   [256, 256, 256],
    "lr":             3e-4,
    "gamma":          0.992,
    "tau":            0.005,
    "target_entropy": -3.0,
}

STATIONARY_FRAMES      = 60
STATIONARY_THRESHOLD_M = 0.5

# Default checkpoint — the one you asked for on the comparison slide.
DEFAULT_CHECKPOINT = os.path.join(
    _PROJECT, "trained_models", "SAC", "150_demo", "model.pt"
)


# ──────────────────────────────────────────────────────────────────────────────
# Device + env
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def build_env(config_path: str, work_dir: str):
    """
    Build an OurEnv-wrapped AssettoCorsaEnv for the baseline SAC demo.

    The on-disk config.yml is loaded into memory only; no write-back occurs.
    Overrides applied (required for the baseline checkpoint's 125-dim input):
      - add_previous_obs_to_state      = True   (adds h-3/h-2/h-1 basic-obs frames)
      - use_target_speed               = False  (baseline was trained without it)
      - enable_out_of_track_termination = False (so we can control OOT via CLI)

    Returns
    -------
    OurEnv instance with state_dim == 125.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)

    # ── In-memory overrides — config.yml on disk stays untouched ─────────────
    # Reconstruct the env shape the baseline SAC checkpoints were trained
    # against (125-dim obs, no target-speed lookahead).  Action semantics
    # are left as config.yml default (absolute, use_relative_actions=False).
    cfg.AssettoCorsa.add_previous_obs_to_state       = True
    cfg.AssettoCorsa.use_target_speed                = False
    cfg.AssettoCorsa.enable_out_of_track_termination = False

    logger.info(
        "Env config overrides (in-memory only — config.yml on disk untouched): "
        "add_previous_obs_to_state=True, use_target_speed=False, "
        "enable_out_of_track_termination=False"
    )

    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    from our_env import OurEnv

    os.makedirs(work_dir, exist_ok=True)
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    if ac_env.state_dim != SAC_HYPERPARAMS["obs_dim"]:
        raise RuntimeError(
            f"Env obs_dim={ac_env.state_dim} does not match baseline SAC obs_dim="
            f"{SAC_HYPERPARAMS['obs_dim']}. Check config overrides."
        )
    logger.info(
        f"AssettoCorsaEnv built — obs_dim={ac_env.state_dim}  "
        f"action_dim={ac_env.action_dim}"
    )

    our_env_cfg = OmegaConf.create({
        "our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)
    })
    env = OurEnv(ac_env, our_env_cfg)
    logger.info("OurEnv ready.")
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — reuse the Vector-Q video recorder + telemetry classes so the
# output format is byte-identical to the Vector-Q demo (same CSV columns).
# The import is safe: the target module has no module-level side effects.
# ──────────────────────────────────────────────────────────────────────────────

from transformer_sac_vectorq_v2_final_fineTune.demo import (  # noqa: E402
    VideoRecorder, TelemetryLogger,
)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy-baseline observation adapter
# ──────────────────────────────────────────────────────────────────────────────
# The 150_* baseline checkpoints were trained when ac_env.get_obs() stored
# past_throttle / past_brake / current_throttle / current_brake in RAW [0, 1]
# telemetry space (accStatus, brakeStatus).  Current ac_env.py:743-770 applies
# `x * 2 - 1` to those slots, shifting them to [-1, 1].  Confirmed from the
# buffer: trained_models/SAC/150_jitter/buffer_pos.npz past_throttle has
# min=0, max=1, mean=0.825.
#
# We reverse the remap on 8 specific obs dimensions so the baseline policy
# sees its training distribution.  Steer slots (38-40, 47) used a different
# formula (steerAngle / scale) that has NOT changed, so they are left alone.
# Prev_obs block [50-124] contains 3× basic_obs (14 car-state channels + 11
# rays); accStatus/brakeStatus are NOT in obs_enabled_channels, so that block
# does not need fixing either.
_LEGACY_PEDAL_SLOTS = np.array(
    [41, 42, 43,  # past_throttle (t-3, t-2, t-1)
     44, 45, 46,  # past_brake    (t-3, t-2, t-1)
     48,          # current_throttle
     49],         # current_brake
    dtype=np.int64,
)


def fix_obs_for_legacy_baseline(obs: np.ndarray) -> np.ndarray:
    """
    Reverse the `x * 2 - 1` remap on past/current throttle & brake slots.
    Returns a NEW array; does not mutate the input.
    """
    out = np.asarray(obs, dtype=np.float32).copy()
    out[_LEGACY_PEDAL_SLOTS] = (out[_LEGACY_PEDAL_SLOTS] + 1.0) * 0.5
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline SAC (MLP) — deterministic demo drive"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(_PROJECT, "assetto_corsa_gym", "config.yml"),
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help=f"Path to .pt (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--device",       type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--manage-ac",    action="store_true")
    parser.add_argument("--no-ai-drive",  action="store_true")
    parser.add_argument("--no-video",     action="store_true")
    parser.add_argument("--fps",          type=int, default=30)
    parser.add_argument("--monitor",      type=int, default=1)
    parser.add_argument("--full-monitor", action="store_true")
    parser.add_argument("--max-frames",   type=int, default=0,
                        help="Hard cap on frames per episode (0 = unlimited).")
    parser.add_argument("--stochastic",   action="store_true",
                        help="Sample actions instead of tanh(mean).")
    parser.add_argument("--episodes",     type=int, default=5,
                        help="Number of demo episodes to run back-to-back.")
    parser.add_argument("--oot-frames",   type=int, default=1,
                        help="Consecutive off-track frames that end an episode "
                             "(1 = strict, 100 = benchmark-style).")
    parser.add_argument("--run-dir",      type=str, default=None,
                        help="Override output dir (default: <ckpt-dir>/demo_runs/<ts>/).")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    deterministic = not args.stochastic
    logger.info(f"Device: {device}  deterministic={deterministic}")

    # ── Validate checkpoint ──────────────────────────────────────────────────
    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 1
    ckpt_tag = os.path.basename(os.path.dirname(ckpt_path)) or "unknown_ckpt"

    # ── Output folder ────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
    else:
        run_dir = os.path.join(os.path.dirname(ckpt_path), "demo_runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(
        f"Demo run: {run_dir}  episodes={args.episodes}  "
        f"checkpoint_tag={ckpt_tag}"
    )

    # ── AC launch ────────────────────────────────────────────────────────────
    if args.manage_ac:
        from ac_lifecycle import full_cycle
        full_cycle(max_retries=3)

    # ── Env ──────────────────────────────────────────────────────────────────
    config_path = os.path.abspath(args.config)
    work_dir    = os.path.join(_PROJECT, "outputs", "baseline_sac_demo")
    env = build_env(config_path, work_dir)

    # ── SAC (inference only) ─────────────────────────────────────────────────
    from sac.sac import SAC

    sac = SAC(
        obs_dim=SAC_HYPERPARAMS["obs_dim"],
        action_dim=SAC_HYPERPARAMS["action_dim"],
        hidden_units=SAC_HYPERPARAMS["hidden_units"],
        lr=SAC_HYPERPARAMS["lr"],
        gamma=SAC_HYPERPARAMS["gamma"],
        tau=SAC_HYPERPARAMS["tau"],
        target_entropy=SAC_HYPERPARAMS["target_entropy"],
        device=device,
    )
    logger.info(f"Loading checkpoint: {ckpt_path}")
    sac.load(ckpt_path)

    # ── Drive N episodes ─────────────────────────────────────────────────────
    all_stats = []
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

            # Track-quality metrics (mirrors Vector-Q demo exactly)
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

            # Lap-wrap distance accounting
            initial_lap_dist  = None
            current_lap_start = 0.0
            current_lap_peak  = 0.0
            cumulative_prior  = 0.0
            laps_completed    = 0
            prev_lap_dist     = None
            lap_dist_end      = 0.0
            best_lap_s        = 0.0
            LAP_WRAP_DROP_THRESHOLD = 2000.0

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

                # Warmup: straight, full throttle, no brake.
                # AC control space is [-1, +1] where -1 = pedal fully up (off),
                # 0 = half-pressed, +1 = fully pressed (per VJoyControl.neutralize
                # which sets acc=-1, brake=-1 for "no input").  So:
                #   steer    = 0.0   straight (0-calibrated, confirmed by buffer)
                #   throttle = 1.0   full throttle
                #   brake    = -1.0  pedal off
                import random as _random
                _warmup_steps = _random.randint(150, 250)
                warmup_action = np.array([0.0, 1.0, -1.0], dtype=np.float32)
                logger.info(
                    f"  warmup: {_warmup_steps} steps action={warmup_action.tolist()}"
                )
                warmup_done = False
                for _ in range(_warmup_steps):
                    obs, _r, wdone, _info = env.step(warmup_action)
                    env._read_latest_state()
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

                    # ── Baseline SAC inference ───────────────────────────────
                    # obs is a flat 125-dim float32 array; SAC.select_action
                    # handles to-tensor, clamp(-3,3), and tanh(mean).
                    # Fix pedal slots back to [0,1] so the legacy policy sees
                    # its training-time convention (see fix_obs_for_legacy_baseline).
                    obs_for_model = fix_obs_for_legacy_baseline(obs)
                    action = sac.select_action(obs_for_model, deterministic=deterministic)
                    action = np.asarray(action, dtype=np.float32)

                    # Smoothness accounting
                    action64 = action.astype(np.float64)
                    if action_prev is not None:
                        sum_abs_delta += np.abs(action64 - action_prev)
                        delta_count   += 1
                    action_prev = action64

                    obs, reward, done, info = env.step(action)
                    env._read_latest_state()

                    r_scalar = (float(np.sum(reward)) if hasattr(reward, "__len__")
                                else float(reward))
                    ep_reward += r_scalar
                    if r_scalar > 0:
                        positive_frames += 1

                    telemetry.log(frame, action, info, r_scalar)
                    frame += 1

                    if isinstance(info, dict):
                        sp = info.get("speed", info.get("speed_kmh"))
                        if sp is not None:
                            sp_f = float(sp)
                            speed_sum += sp_f
                            if sp_f > speed_max:
                                speed_max = sp_f

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
                                    segment = prev_lap_dist - current_lap_start
                                    cumulative_prior += segment
                                    laps_completed  += 1
                                    current_lap_start = 0.0
                                    current_lap_peak  = lap_dist
                                if lap_dist > current_lap_peak:
                                    current_lap_peak = lap_dist
                                prev_lap_dist = lap_dist

                        if "gap" in info:
                            g_abs = abs(float(info.get("gap", 0.0)))
                            gap_abs_sum += g_abs
                            if g_abs > gap_max:
                                gap_max = g_abs
                            gap_count   += 1

                        if "yaw_error" in info:
                            yaw_abs_sum += abs(float(info.get("yaw_error", 0.0)))
                            yaw_count   += 1

                        lap_t = float(info.get("LapTime", 0.0) or 0.0)
                        if lap_t > 1e-3 and (best_lap_s == 0.0 or lap_t < best_lap_s):
                            best_lap_s = lap_t

                        comp = info.get("reward_components")
                        if isinstance(comp, dict):
                            for k in rcomp_sums:
                                if k in comp:
                                    rcomp_sums[k] += float(comp[k])

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
                    terminated_by = "env_done" if frame > 0 else "interrupted"

            finally:
                if video is not None:
                    video.stop()
                telemetry.close()
                # Guard a second time — if the try block exited via an
                # exception (e.g. KeyboardInterrupt) before the "if None"
                # check above ran, terminated_by is still None.
                if terminated_by is None:
                    terminated_by = "env_done" if frame > 0 else "interrupted"

                elapsed = time.perf_counter() - t_start
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
                    f"Model: baseline SAC (MLP, hidden=[256,256,256], obs_dim=125)\n"
                    f"Checkpoint: {ckpt_path}\n"
                    f"Deterministic: {deterministic}\n"
                    f"Terminated by: {terminated_by}\n"
                    f"Duration: {elapsed:.1f} s  |  Frames: {frame}\n"
                    f"\n"
                    f"-- Reward --\n"
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
                    f"-- Progress --\n"
                    f"  total driven dist   : {total_dist:.2f} m\n"
                    f"  laps completed      : {laps_completed}\n"
                    f"  final LapDist       : {lap_dist_end:.2f}\n"
                    f"  best lap time       : {best_lap_s:.3f} s\n"
                    f"\n"
                    f"-- Track-following quality --\n"
                    f"  mean |gap|          : {mean_abs_gap:.3f} m  (max {gap_max:.3f})\n"
                    f"  mean |yaw_error|    : {mean_abs_yaw:.4f} rad\n"
                    f"  smoothness steer    : {smoothness_steer:.4f}\n"
                    f"  smoothness throttle : {smoothness_throttle:.4f}\n"
                    f"  smoothness brake    : {smoothness_brake:.4f}\n"
                    f"\n"
                    f"-- Speed --\n"
                    f"  mean speed          : {speed_mean:.2f}\n"
                    f"  max speed           : {speed_max:.2f}\n"
                    f"\n"
                    f"-- OOT --\n"
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
        # Aggregate summary (same format as Vector-Q demo)
        if all_stats:
            import statistics as _stats
            def _mean(xs): return float(_stats.mean(xs)) if xs else 0.0
            def _stdev(xs): return float(_stats.stdev(xs)) if len(xs) > 1 else 0.0
            def _col(k): return [s[k] for s in all_stats]

            term_counts: dict = {}
            for s in all_stats:
                term_counts[s["terminated_by"]] = term_counts.get(s["terminated_by"], 0) + 1

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

            def _ms(key, fmt=".2f"):
                xs = _col(key)
                return f"{_mean(xs):{fmt}}+/-{_stdev(xs):{fmt}}"

            agg_block = (
                f"\n{'-' * 120}\n"
                f"MEAN +/- STDEV over {len(all_stats)} episodes:\n"
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
                f"Demo run: {run_id}  (BASELINE SAC — MLP)\n"
                f"Episodes completed: {len(all_stats)}/{args.episodes}"
                f"{' (interrupted)' if interrupted else ''}\n"
                f"Checkpoint: {ckpt_path}\n"
                f"Deterministic: {deterministic}\n\n"
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
