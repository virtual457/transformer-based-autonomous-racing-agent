"""
benchmark_backups.py — evaluate every saved checkpoint and rank them.

For each `latest.pt` found under the backups folder, this script drives
Assetto Corsa deterministically for N episodes per checkpoint and records
quality metrics independent of the reward function. Aggregates produce a
summary CSV and a ranked text table.

Primary ranking metric: reward_per_frame_mean (descending).
Other metrics reported per checkpoint (across N episodes, all mean unless noted):
    - total_dist       cumulative driven distance (handles lap wraps)
    - frames           survival time in simulator steps
    - mean_abs_gap     mean |lateral deviation| from racing line (meters)
    - smoothness_*     mean per-step |Δaction| per channel (steer/throttle/brake)
    - recovery_count   OOT excursions the policy recovered from
    - speed_mean       mean speed (m/s)
    - laps_completed   total laps completed across all episodes
    - crash/stat/done rates
    - ep_reward        raw sum (reward-shape-dependent, use r/f instead)

Termination rules (per episode):
    - crashed_oot   off-track for 100 consecutive frames. Policy is OOD.
    - stationary    has not moved 0.5m over the last 60 frames.
    - env_done      environment itself terminated.
    - max_frames    only if --max-frames > 0 as a safety cap.

Default --max-frames is 0 (unlimited).
AC is launched once and reused across all checkpoints.

Run:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2_final_fineTune/benchmark_backups.py --manage-ac --episodes 10

Options:
    --backups-dir <path>    folder containing <stamp>_cycle<N>/latest.pt subdirs
                            (default: trained_models/SAC_VectorQ_V2/cycle_backups)
    --include-current       also evaluate gym/transformer_sac_vectorq_v2_final_fineTune/checkpoints/latest.pt
    --episodes N            episodes per checkpoint (default 10)
    --max-frames N          safety cap per episode (default 0 = unlimited)
    --stochastic            sample actions instead of tanh(mean)
    --skip K                skip the first K checkpoints (for resuming)
"""

import argparse
import collections
import csv
import glob
import logging
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "assetto_corsa_gym"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "assetto_corsa_gym", "assetto_corsa_gym"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "assetto_corsa_gym", "algorithm", "discor"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "gym"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vectorq_fineTune.benchmark")

TOKEN_DIM              = 50
WINDOW_SIZE            = 75
STATIONARY_FRAMES      = 60
STATIONARY_THRESHOLD_M = 0.5
# If the car is off-track for this many *consecutive* frames, we consider the
# policy to have crashed (driven itself into the OOD region it was never trained
# to recover from). The episode is terminated and the stats are taken at the
# point of crash. A brief excursion that the policy recovers from is OK — the
# counter resets whenever we're back on-track.
OOT_CRASH_FRAMES       = 100

TRANSFORMER_CONFIG = {
    "token_dim":     TOKEN_DIM,
    "action_dim":    3,
    "window_size":   WINDOW_SIZE,
    "d_model":       256,
    "n_heads":       4,
    "n_layers":      4,
    "ffn_dim":       1024,
    "policy_hidden": [256],
    "q_hidden":      [256],
}
SAC_HYPERPARAMS = {
    "lr":             3e-4,
    "gamma":          0.992,
    "tau":            0.005,
    "target_entropy": -2.0,
}

_DEFAULT_BACKUPS_DIR = os.path.join(
    _REPO_ROOT, "trained_models", "SAC_VectorQ_V2", "cycle_backups"
)
_LIVE_CHECKPOINT = os.path.join(_THIS_DIR, "checkpoints", "latest.pt")
_RESULTS_ROOT    = os.path.join(_THIS_DIR, "benchmark_runs")


# ─────────────────────────────────────────────────────────────────────────────
# Env + SAC helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def _build_env(config_path: str, work_dir: str):
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode driver (deterministic, shared across all checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env, sac, *, deterministic: bool, max_frames: int) -> dict:
    """Run one deterministic episode. Returns stats dict."""
    obs, _info = env.reset()
    obs_deque: collections.deque = collections.deque(maxlen=WINDOW_SIZE)

    warmup = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    for _ in range(WINDOW_SIZE):
        obs, _r, done, _info = env.step(warmup)
        env._read_latest_state()
        obs_deque.append(obs[:TOKEN_DIM].astype(np.float32))
        if done:
            break

    frame            = 0
    ep_reward        = 0.0
    positive_frames  = 0
    oot_frames       = 0
    consecutive_oot  = 0
    max_consec_oot   = 0
    speed_sum        = 0.0
    speed_max        = 0.0
    lap_dist_last    = 0.0
    best_lap_s       = 0.0
    t0               = time.perf_counter()
    pos_window: collections.deque = collections.deque(maxlen=STATIONARY_FRAMES)
    terminated_by    = None
    done             = False

    # Cumulative total distance state (handles LapDist wrap at start/finish line)
    initial_lap_dist   = None
    current_lap_start  = 0.0
    current_lap_peak   = 0.0
    cumulative_prior   = 0.0
    laps_completed     = 0
    prev_lap_dist      = None
    LAP_WRAP_DROP_THRESHOLD = 2000.0

    # Mean abs gap to racing line
    gap_abs_sum   = 0.0
    gap_count     = 0

    # Per-channel action smoothness
    action_prev     = None
    sum_abs_delta   = np.zeros(3, dtype=np.float64)
    delta_count     = 0

    # Recovery count (OOT excursions that didn't crash)
    prev_oot       = False
    recovery_count = 0

    while not done:
        if max_frames and frame >= max_frames:
            terminated_by = "max_frames"
            break

        W = WINDOW_SIZE
        avail = list(obs_deque)
        if len(avail) == W:
            window = np.stack(avail, axis=0)
        else:
            pad = [avail[0]] * (W - len(avail))
            window = np.stack(pad + avail, axis=0)

        action, _mean, _std = sac.select_action(window, deterministic=deterministic)

        # Per-channel action smoothness
        if action_prev is not None:
            sum_abs_delta += np.abs(np.asarray(action, dtype=np.float64) - action_prev)
            delta_count   += 1
        action_prev = np.asarray(action, dtype=np.float64)

        obs, reward, done, info = env.step(action)
        env._read_latest_state()
        obs_deque.append(obs[:TOKEN_DIM].astype(np.float32))

        r_scalar = float(np.sum(reward)) if hasattr(reward, "__len__") else float(reward)
        ep_reward += r_scalar
        if r_scalar > 0:
            positive_frames += 1

        if isinstance(info, dict):
            speed_ms = float(info.get("speed", info.get("speed_kmh", 0.0)) or 0.0)
            speed_sum += speed_ms
            if speed_ms > speed_max:
                speed_max = speed_ms
            lap_dist = float(info.get("LapDist", lap_dist_last) or lap_dist_last)
            lap_dist_last = lap_dist

            # Cumulative total distance with lap wrap detection
            if initial_lap_dist is None:
                initial_lap_dist  = lap_dist
                current_lap_start = lap_dist
                current_lap_peak  = lap_dist
                prev_lap_dist     = lap_dist
            else:
                if prev_lap_dist is not None and prev_lap_dist > 1000 and (prev_lap_dist - lap_dist) > LAP_WRAP_DROP_THRESHOLD:
                    # wrap: completed a lap
                    segment = prev_lap_dist - current_lap_start
                    cumulative_prior += segment
                    laps_completed   += 1
                    current_lap_start = 0.0
                    current_lap_peak  = lap_dist
                if lap_dist > current_lap_peak:
                    current_lap_peak = lap_dist
                prev_lap_dist = lap_dist

            # Mean abs gap to racing line
            if "gap" in info:
                gap_abs_sum += abs(float(info.get("gap", 0.0)))
                gap_count   += 1

            lap_t = float(info.get("LapTime", 0.0) or 0.0)
            if lap_t > 1e-3:
                if best_lap_s == 0.0 or lap_t < best_lap_s:
                    best_lap_s = lap_t

            is_oot_now = bool(info.get("out_of_track", False))
            if is_oot_now:
                oot_frames += 1
                consecutive_oot += 1
                if consecutive_oot > max_consec_oot:
                    max_consec_oot = consecutive_oot
                if consecutive_oot >= OOT_CRASH_FRAMES:
                    terminated_by = "crashed_oot"
                    done = True
            else:
                consecutive_oot = 0

            # Recovery count: OOT → on-track transition. Don't count the final
            # terminated-by-crash state as a recovery.
            if prev_oot and not is_oot_now and terminated_by != "crashed_oot":
                recovery_count += 1
            prev_oot = is_oot_now

            sx, sy = info.get("world_position_x"), info.get("world_position_y")
            if sx is not None and sy is not None:
                pos_window.append((float(sx), float(sy)))
                if len(pos_window) == STATIONARY_FRAMES:
                    dx = pos_window[-1][0] - pos_window[0][0]
                    dy = pos_window[-1][1] - pos_window[0][1]
                    if math.sqrt(dx * dx + dy * dy) < STATIONARY_THRESHOLD_M:
                        terminated_by = "stationary"
                        done = True

        frame += 1

    if terminated_by is None:
        terminated_by = "env_done"

    elapsed = time.perf_counter() - t0

    # Finalize cumulative total distance (includes the in-progress segment)
    current_segment = max(0.0, current_lap_peak - current_lap_start)
    total_dist = cumulative_prior + current_segment

    # Finalize mean abs gap
    mean_abs_gap = gap_abs_sum / max(gap_count, 1)

    # Finalize per-channel action smoothness
    if delta_count > 0:
        smooth = sum_abs_delta / delta_count
    else:
        smooth = np.zeros(3)
    smoothness_steer    = float(smooth[0])
    smoothness_throttle = float(smooth[1])
    smoothness_brake    = float(smooth[2])

    return {
        "frames":               frame,
        "ep_reward":            ep_reward,
        "reward_per_frame":     ep_reward / max(frame, 1),
        "positive_frac":        positive_frames / max(frame, 1),
        "oot_frac":             oot_frames / max(frame, 1),
        "max_consec_oot":       max_consec_oot,
        "recovery_count":       recovery_count,
        "total_dist":           total_dist,
        "laps_completed":       laps_completed,
        "lap_dist_end":         lap_dist_last,   # final LapDist reading, for reference
        "mean_abs_gap":         mean_abs_gap,
        "smoothness_steer":     smoothness_steer,
        "smoothness_throttle":  smoothness_throttle,
        "smoothness_brake":     smoothness_brake,
        "best_lap_s":           best_lap_s,  # keep as-is even if always 0 for now
        "speed_mean":           speed_sum / max(frame, 1),
        "speed_max":            speed_max,
        "duration_s":           elapsed,
        "terminated_by":        terminated_by,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_checkpoints(backups_dir: str, include_current: bool) -> list[tuple[str, str]]:
    """Return list of (label, path) for checkpoints to evaluate, sorted by label."""
    pairs: list[tuple[str, str]] = []
    if os.path.isdir(backups_dir):
        for sub in sorted(os.listdir(backups_dir)):
            p = os.path.join(backups_dir, sub, "latest.pt")
            if os.path.isfile(p):
                pairs.append((sub, p))
    if include_current and os.path.isfile(_LIVE_CHECKPOINT):
        pairs.append(("_CURRENT_LIVE", _LIVE_CHECKPOINT))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rank backup checkpoints by driving quality.")
    parser.add_argument("--config", type=str,
        default=os.path.join(_REPO_ROOT, "assetto_corsa_gym", "config.yml"))
    parser.add_argument("--backups-dir",     type=str, default=_DEFAULT_BACKUPS_DIR)
    parser.add_argument("--include-current", action="store_true",
                        help="Also evaluate the live checkpoints/latest.pt of this variant.")
    parser.add_argument("--episodes",        type=int, default=10)
    parser.add_argument("--max-frames",      type=int, default=0,
                        help="Hard cap per episode (0 = unlimited, the default — "
                             "let episodes end on crash / stationary / env_done).")
    parser.add_argument("--device",          type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--manage-ac",       action="store_true")
    parser.add_argument("--stochastic",      action="store_true")
    parser.add_argument("--skip",            type=int, default=0,
                        help="Skip first K checkpoints (for resuming).")
    args = parser.parse_args()

    device = _resolve_device(args.device)

    # ── Discover checkpoints ─────────────────────────────────────────────────
    ckpts = discover_checkpoints(args.backups_dir, args.include_current)
    if not ckpts:
        logger.error(f"No checkpoints found under {args.backups_dir}"
                     f"{' or live path' if args.include_current else ''}. Abort.")
        return
    if args.skip > 0:
        ckpts = ckpts[args.skip:]

    logger.info(f"Will evaluate {len(ckpts)} checkpoint(s) with {args.episodes} episodes each:")
    for label, path in ckpts:
        logger.info(f"  - {label:40s}  {path}")

    # ── Output folder ────────────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(_RESULTS_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    per_ep_csv = os.path.join(run_dir, "per_episode.csv")
    summary_csv = os.path.join(run_dir, "summary.csv")
    ranking_txt = os.path.join(run_dir, "ranking.txt")
    logger.info(f"Results dir: {run_dir}")

    # ── AC + env (launched once) ─────────────────────────────────────────────
    if args.manage_ac:
        from ac_lifecycle import full_cycle
        logger.info("Launching AC ...")
        full_cycle(max_retries=3)

    work_dir = os.path.join(_REPO_ROOT, "outputs", "transformer_sac_vectorq_v2_final_fineTune_benchmark")
    env = _build_env(os.path.abspath(args.config), work_dir)

    # ── SAC (reused across checkpoints via .load) ────────────────────────────
    from transformer_sac_vectorq_v2_final_fineTune.sac import TransformerSAC
    sac = TransformerSAC(**TRANSFORMER_CONFIG, **SAC_HYPERPARAMS, device=device)

    # ── Per-episode CSV ──────────────────────────────────────────────────────
    per_ep_fields = [
        "label", "checkpoint", "episode",
        "frames", "ep_reward", "reward_per_frame",
        "positive_frac", "oot_frac", "max_consec_oot", "recovery_count",
        "total_dist", "laps_completed", "lap_dist_end",
        "mean_abs_gap",
        "smoothness_steer", "smoothness_throttle", "smoothness_brake",
        "best_lap_s", "speed_mean", "speed_max",
        "duration_s", "terminated_by",
    ]
    per_ep_f = open(per_ep_csv, "w", newline="")
    per_ep_w = csv.DictWriter(per_ep_f, fieldnames=per_ep_fields)
    per_ep_w.writeheader()

    summary_rows: list[dict] = []

    try:
        for ckpt_idx, (label, path) in enumerate(ckpts, start=1):
            logger.info("=" * 70)
            logger.info(f"CHECKPOINT [{ckpt_idx}/{len(ckpts)}]  {label}")
            logger.info("=" * 70)
            sac.load(path)

            ep_stats: list[dict] = []
            for ep in range(1, args.episodes + 1):
                logger.info(f"  Episode {ep}/{args.episodes} — driving ...")
                stats = run_episode(
                    env, sac,
                    deterministic=not args.stochastic,
                    max_frames=args.max_frames,
                )
                ep_stats.append(stats)
                row = dict(stats); row.update({"label": label, "checkpoint": path, "episode": ep})
                per_ep_w.writerow(row); per_ep_f.flush()
                logger.info(
                    f"  Episode {ep}: frames={stats['frames']}  "
                    f"total_dist={stats['total_dist']:.1f}  "
                    f"r/f={stats['reward_per_frame']:.4f}  "
                    f"gap={stats['mean_abs_gap']:.2f}  "
                    f"smooth=s{stats['smoothness_steer']:.3f}/t{stats['smoothness_throttle']:.3f}/b{stats['smoothness_brake']:.3f}  "
                    f"laps={stats['laps_completed']}  recov={stats['recovery_count']}  "
                    f"term={stats['terminated_by']}"
                )

            # ── Aggregate per-checkpoint stats ────────────────────────────
            arr     = lambda k: np.array([s[k] for s in ep_stats], dtype=np.float64)
            term_of = lambda tag: sum(1 for s in ep_stats if s["terminated_by"] == tag) / max(len(ep_stats), 1)
            summary = {
                "label":                      label,
                "checkpoint":                 path,
                "n_eps":                      len(ep_stats),
                "reward_per_frame_mean":      float(arr("reward_per_frame").mean()),
                "frames_mean":                float(arr("frames").mean()),
                "frames_max":                 float(arr("frames").max()),
                "total_dist_mean":            float(arr("total_dist").mean()),
                "total_dist_max":             float(arr("total_dist").max()),
                "laps_completed_total":       int(arr("laps_completed").sum()),
                "mean_abs_gap_mean":          float(arr("mean_abs_gap").mean()),
                "smoothness_steer_mean":      float(arr("smoothness_steer").mean()),
                "smoothness_throttle_mean":   float(arr("smoothness_throttle").mean()),
                "smoothness_brake_mean":      float(arr("smoothness_brake").mean()),
                "recovery_count_mean":        float(arr("recovery_count").mean()),
                "ep_reward_mean":             float(arr("ep_reward").mean()),
                "speed_mean_mean":            float(arr("speed_mean").mean()),
                "speed_max_mean":             float(arr("speed_max").mean()),
                "positive_frac_mean":         float(arr("positive_frac").mean()),
                "oot_frac_mean":              float(arr("oot_frac").mean()),
                "crash_rate":                 float(term_of("crashed_oot")),
                "stationary_rate":            float(term_of("stationary")),
                "env_done_rate":              float(term_of("env_done")),
                "max_frames_rate":            float(term_of("max_frames")),
                "best_lap_s_min":             float(np.min([s["best_lap_s"] for s in ep_stats if s["best_lap_s"] > 0] or [0.0])),
            }
            summary_rows.append(summary)
            logger.info(
                f"=> {label}: r/f={summary['reward_per_frame_mean']:.4f}  "
                f"frames={summary['frames_mean']:.0f}  "
                f"total_dist={summary['total_dist_mean']:.1f}  "
                f"gap={summary['mean_abs_gap_mean']:.2f}  "
                f"smooth=s{summary['smoothness_steer_mean']:.3f}/t{summary['smoothness_throttle_mean']:.3f}/b{summary['smoothness_brake_mean']:.3f}  "
                f"laps={summary['laps_completed_total']}  "
                f"crash%={100*summary['crash_rate']:.0f}"
            )

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user — writing partial results.")
    finally:
        per_ep_f.close()

        # ── Summary CSV ──────────────────────────────────────────────────
        if summary_rows:
            with open(summary_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)

            # ── Ranked by reward_per_frame_mean (descending) ─────────────
            ranked = sorted(summary_rows, key=lambda r: r["reward_per_frame_mean"], reverse=True)
            lines = [
                f"{'rank':>4}  {'label':40s}  "
                f"{'r/f':>7s}  {'frames':>7s}  {'total_dist':>10s}  "
                f"{'gap':>6s}  {'smoo_s':>7s}  {'smoo_t':>7s}  {'smoo_b':>7s}  "
                f"{'speed':>6s}  {'crash%':>6s}  {'laps':>5s}",
                "-" * 140,
            ]
            for i, r in enumerate(ranked, 1):
                lines.append(
                    f"{i:>4}  {r['label']:40s}  "
                    f"{r['reward_per_frame_mean']:>7.4f}  "
                    f"{r['frames_mean']:>7.0f}  "
                    f"{r['total_dist_mean']:>10.1f}  "
                    f"{r['mean_abs_gap_mean']:>6.2f}  "
                    f"{r['smoothness_steer_mean']:>7.4f}  "
                    f"{r['smoothness_throttle_mean']:>7.4f}  "
                    f"{r['smoothness_brake_mean']:>7.4f}  "
                    f"{r['speed_mean_mean']:>6.2f}  "
                    f"{100*r['crash_rate']:>6.0f}  "
                    f"{r['laps_completed_total']:>5d}"
                )
            ranking_str = "\n".join(lines)
            with open(ranking_txt, "w") as f:
                f.write(ranking_str + "\n")
            logger.info("\n" + ranking_str)

        logger.info(f"\nResults written to: {run_dir}")
        logger.info(f"  {per_ep_csv}")
        logger.info(f"  {summary_csv}")
        logger.info(f"  {ranking_txt}")

        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
