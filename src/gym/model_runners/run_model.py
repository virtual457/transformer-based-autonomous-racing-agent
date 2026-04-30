"""
run_model.py — Inference-only runner for a trained SAC model against Assetto Corsa.

Loads a saved checkpoint and runs the policy in deterministic mode (tanh(mean),
no sampling noise).  No training, no replay buffer, no gradient steps, no wandb.

Usage
-----
    # From the repo root:
    .\\AssetoCorsa\\Scripts\\python.exe gym/model_runners/run_model.py --model smooth_operator
    .\\AssetoCorsa\\Scripts\\python.exe gym/model_runners/run_model.py --model jittery --episodes 5
    .\\AssetoCorsa\\Scripts\\python.exe gym/model_runners/run_model.py --model smooth_operator --no-manage-ac

Checkpoint path resolved as:
    <repo_root>/trained_models/SAC/<model_name>/model.pt

AC lifecycle:
    By default (--manage-ac), full_cycle() is called once before episodes begin
    to write config files, kill any running AC instance, launch via CM, and wait
    for the plugin port.  Pass --no-manage-ac if AC is already running and you
    want to skip this step.

Latency-overlap trick:
    Identical to the collect_phase() in agent.py:
        1. select_action(obs)          — infer on current frame
        2. env.set_actions(action)     — fire to AC immediately (non-blocking)
        3. env._read_latest_state()    — CPU work overlaps with 40 ms physics tick
        4. env.step(action=None)       — block for next frame only

    This order must NOT be changed.  It ensures AC receives the action computed
    from the frame that just arrived, not a stale action from the previous frame.
"""

import sys
import os
import argparse
import logging
import time

# ── Path setup — mirrors train_sac.py exactly ─────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GYM_DIR  = os.path.join(_THIS_DIR, '..')          # gym/
_REPO_DIR = os.path.join(_GYM_DIR,  '..')          # repo root

sys.path.insert(0, os.path.join(_REPO_DIR, 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_REPO_DIR, 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_REPO_DIR, 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, _GYM_DIR)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("run_model")


# ── SAC architecture constants — must match the training run ──────────────────
# These are used only when the checkpoint's embedded config is incomplete.
# SAC.from_checkpoint() will always prefer the saved config values.
_FALLBACK_OBS_DIM    = 125
_FALLBACK_ACTION_DIM = 3
_FALLBACK_HIDDEN     = [256, 256, 256]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def _resolve_model_path(model_name: str) -> str:
    """
    Return the absolute path to <repo_root>/trained_models/SAC/<model_name>/model.pt.
    Raises FileNotFoundError early so the user gets a clear message.
    """
    path = os.path.abspath(
        os.path.join(_REPO_DIR, 'trained_models', 'SAC', model_name, 'model.pt')
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}\n"
            f"Available models under trained_models/SAC/:\n"
            + "\n".join(
                "  " + d for d in os.listdir(
                    os.path.join(_REPO_DIR, 'trained_models', 'SAC')
                )
                if os.path.isfile(
                    os.path.join(_REPO_DIR, 'trained_models', 'SAC', d, 'model.pt')
                )
            )
        )
    return path


def _build_env(config_path: str) -> "OurEnv":
    """Build AssettoCorsaEnv wrapped in OurEnv — identical to train_sac.py:build_env()."""
    from omegaconf import OmegaConf
    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    from our_env import OurEnv

    cfg = OmegaConf.load(config_path)
    assert hasattr(cfg, "AssettoCorsa"), (
        f"config.yml at {config_path} is missing 'AssettoCorsa' block"
    )
    assert hasattr(cfg, "OurEnv"), (
        f"config.yml at {config_path} is missing 'OurEnv' block"
    )

    # Disable the per-episode time limit so inference can run indefinitely.
    # Training uses 120s (~1 lap); for evaluation we want unlimited laps.
    cfg.AssettoCorsa.max_episode_py_time = 1_000_000.0
    logger.info("max_episode_py_time overridden to 1000000s (effectively unlimited)")

    work_dir = os.path.join(_REPO_DIR, 'outputs', 'run_model')
    os.makedirs(work_dir, exist_ok=True)

    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    logger.info(
        "AssettoCorsaEnv built — "
        "obs_dim=%d  action_dim=%d", ac_env.state_dim, ac_env.action_dim
    )

    our_env_cfg = OmegaConf.create({
        "our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)
    })
    env = OurEnv(ac_env, our_env_cfg)
    logger.info("OurEnv ready.")
    return env


def _print_episode_stats(
    ep: int,
    n_episodes: int,
    model_name: str,
    ep_steps: int,
    ep_reward: float,
    ep_positive_steps: int,
    total_steps: int,
    info: dict,
    mean_inference_ms: float,
    mean_interframe_ms: float,
) -> None:
    """Print a compact fixed-width table for one completed inference episode."""
    SEP = "+" + "-" * 26 + "+" + "-" * 16 + "+"

    def row(label: str, value: str) -> str:
        return f"| {label:<24} | {value:>14} |"

    lines = [
        SEP,
        row("Model", model_name),
        row("Episode", f"{ep} of {n_episodes}"),
        row("Steps (episode)", f"{ep_steps:,}"),
        row("Total steps (run)", f"{total_steps:,}"),
        row("Reward (episode)", f"{ep_reward:.4f}"),
        row(
            "Avg reward/step",
            f"{ep_reward / ep_steps:.6f}" if ep_steps > 0 else "N/A",
        ),
        row(
            "Positive reward steps",
            (
                f"{ep_positive_steps}/{ep_steps} "
                f"({100 * ep_positive_steps / ep_steps:.1f}%)"
                if ep_steps > 0 else "N/A"
            ),
        ),
    ]

    # Optional info fields — emit whatever the env provides
    for speed_key in ("avg_speed", "mean_speed"):
        if speed_key in info:
            lines.append(row("Mean speed (m/s)", f"{float(info[speed_key]):.2f}"))
            break

    if "lap_dist" in info:
        lines.append(row("Lap dist (m)", f"{float(info['lap_dist']):.1f}"))

    for lap_key in ("lap_time", "lap_time_s"):
        if lap_key in info:
            lines.append(row("Lap time (s)", f"{float(info[lap_key]):.3f}"))
            break

    for ot_key in ("off_track_count", "off_track"):
        if ot_key in info:
            lines.append(row("Off-track count", f"{int(info[ot_key]):,}"))
            break

    lines.append(row("Avg inference time", f"{mean_inference_ms:.1f} ms"))

    interframe_str = f"{mean_interframe_ms:.1f} ms"
    if mean_interframe_ms > 40.0:
        interframe_str += "  !!"
    lines.append(row("Avg inter-frame time", interframe_str))

    lines.append(SEP)
    print("\n".join(lines), flush=True)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, sac, ep: int, n_episodes: int, model_name: str, total_steps: int) -> dict:
    """
    Run one inference episode using the deterministic SAC policy.

    Follows the latency-overlap pattern from agent.py:collect_phase() exactly.
    No transitions are stored — the replay_buffer.push() call is simply omitted.

    Returns
    -------
    dict with keys: steps, reward, positive_steps
    """
    import numpy as np
    import random

    obs, _info = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    ep_positive_steps = 0
    done = False
    ep_final_info = {}
    inference_times: list = []
    interframe_times: list = []

    # ── Warmup: full throttle to match training (agent.py) ────────────────────
    # During training, agent.py runs 150-250 random steps of full throttle after
    # each reset so the policy never sees near-zero speeds.  Replicate here so
    # inference starts in the same speed regime the model was trained on.
    _warmup_steps = random.randint(150, 250)
    logger.info("Throttle warmup: %d steps (full throttle, steer centred)", _warmup_steps)
    for _ in range(_warmup_steps):
        obs, _, done, _ = env.step(np.array([0.5, 1.0, 0.0], dtype=np.float32))
        env._read_latest_state()
        if done:
            logger.warning("Episode ended during throttle warmup — skipping")
            return {"steps": 0, "reward": 0.0, "positive_steps": 0}

    # ── Stuck detection: if x/y position unchanged for 10 frames, end episode ──
    _STUCK_WINDOW = 10
    _STUCK_THRESHOLD = 0.5  # metres — if total movement < this over the window, car is stuck
    from collections import deque
    _pos_history = deque(maxlen=_STUCK_WINDOW)

    def _check_stuck(info_dict: dict) -> bool:
        """Return True if the car hasn't moved in the last _STUCK_WINDOW frames."""
        x = float(info_dict.get("world_position_x", 0.0))
        y = float(info_dict.get("world_position_y", 0.0))
        _pos_history.append((x, y))
        if len(_pos_history) < _STUCK_WINDOW:
            return False
        oldest = _pos_history[0]
        dx = x - oldest[0]
        dy = y - oldest[1]
        dist = (dx * dx + dy * dy) ** 0.5
        return dist < _STUCK_THRESHOLD

    # ── Step 0: blocking step to seed the overlap loop ────────────────────────
    # No previously-computed action exists yet, so we use the normal blocking path.
    _step0_ran = False
    if not done:
        _t0 = time.perf_counter()
        action = sac.select_action(obs, deterministic=True)
        inference_times.append(time.perf_counter() - _t0)
        prev_obs = obs
        prev_action = action
        next_obs, reward, done, step_info = env.step(action)
        env._read_latest_state()
        _step0_ran = True
        _t_frame = time.perf_counter()
        if isinstance(step_info, dict):
            if done:
                ep_final_info = step_info
            if _check_stuck(step_info):
                logger.warning("Car stuck after step 0 — ending episode")
                done = True

    # ── Steps 1+: latency-overlap loop ────────────────────────────────────────
    # Order is identical to agent.py — do NOT reorder:
    #   1. Advance obs to the arrived frame.
    #   2. select_action(obs)           infer on current frame
    #   3. env.set_actions(action)      fire to AC immediately (non-blocking)
    #   4. env._read_latest_state()     CPU work overlaps with AC tick
    #   5. Accumulate episode stats.
    #   6. env.step(action=None)        block only for the next frame
    while not done:
        obs = next_obs

        # 2. Infer deterministically on the arrived frame.
        _t_inf = time.perf_counter()
        action = sac.select_action(obs, deterministic=True)
        inference_times.append(time.perf_counter() - _t_inf)

        # 3. Fire action to AC immediately (non-blocking).
        env.set_actions(action)

        # 4. CPU work — overlaps with the AC physics tick.
        env._read_latest_state()

        # 5. Accumulate stats for the transition that just resolved
        #    (prev_obs -[prev_action]-> obs with reward from last step()).
        ep_reward += reward
        ep_steps += 1
        if reward > 0:
            ep_positive_steps += 1

        prev_obs = obs
        prev_action = action
        _t_next = time.perf_counter()
        interframe_times.append(_t_next - _t_frame)

        # 6. Block until the next AC frame arrives.
        next_obs, reward, done, step_info = env.step(action=None)
        _t_frame = time.perf_counter()
        if isinstance(step_info, dict):
            if done:
                ep_final_info = step_info
            if not done and _check_stuck(step_info):
                logger.warning(
                    "Car stuck (position unchanged for %d frames) — ending episode at step %d",
                    _STUCK_WINDOW, ep_steps,
                )
                done = True

    # Terminal transition (or sole step-0 transition if done triggered immediately).
    if _step0_ran:
        ep_reward += reward
        ep_steps += 1
        if reward > 0:
            ep_positive_steps += 1

    mean_inf_ms = float(sum(inference_times) / len(inference_times)) * 1000.0 if inference_times else 0.0
    mean_ift_ms = float(sum(interframe_times) / len(interframe_times)) * 1000.0 if interframe_times else 0.0

    _print_episode_stats(
        ep=ep,
        n_episodes=n_episodes,
        model_name=model_name,
        ep_steps=ep_steps,
        ep_reward=ep_reward,
        ep_positive_steps=ep_positive_steps,
        total_steps=total_steps + ep_steps,
        info=ep_final_info,
        mean_inference_ms=mean_inf_ms,
        mean_interframe_ms=mean_ift_ms,
    )
    logger.info(
        "ep=%d/%d  steps=%d  reward=%.3f  pos_steps=%d",
        ep, n_episodes, ep_steps, ep_reward, ep_positive_steps,
    )

    return {
        "steps":          ep_steps,
        "reward":         ep_reward,
        "positive_steps": ep_positive_steps,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAC inference runner — loads a trained model and runs episodes against AC."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Name of the model to load. "
            "Checkpoint is read from trained_models/SAC/<model>/model.pt"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            _REPO_DIR, 'assetto_corsa_gym', 'config.yml'
        ),
        help="Path to config.yml (default: assetto_corsa_gym/config.yml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device (default: auto — cuda if available, else cpu).",
    )
    parser.add_argument(
        "--manage-ac",
        action="store_true",
        default=True,
        help=(
            "Launch AC via full_cycle() before running episodes. "
            "Writes race.ini/assists.ini, kills existing AC, launches CM URI. "
            "Enabled by default. Use --no-manage-ac to skip."
        ),
    )
    parser.add_argument(
        "--no-manage-ac",
        dest="manage_ac",
        action="store_false",
        help="Skip AC lifecycle management — assume AC is already running.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 1. Resolve model checkpoint ───────────────────────────────────────────
    model_path = _resolve_model_path(args.model)
    logger.info("Model checkpoint: %s", model_path)

    # ── 2. Device ─────────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    logger.info("Device: %s", device)

    # ── 3. Load SAC from checkpoint ───────────────────────────────────────────
    # from_checkpoint() reads the embedded config so architecture is exact.
    from sac.sac import SAC

    logger.info("Loading SAC from checkpoint ...")
    sac = SAC.from_checkpoint(model_path, device=device)

    # Put policy in eval mode — disables dropout/batchnorm if any were added later.
    sac.policy.eval()
    logger.info("Policy loaded and set to eval mode.")

    # ── 4. Environment ────────────────────────────────────────────────────────
    config_path = os.path.abspath(args.config)
    logger.info("Config: %s", config_path)
    env = _build_env(config_path)

    # ── 5. AC lifecycle ───────────────────────────────────────────────────────
    if args.manage_ac:
        logger.info("Launching AC via full_cycle() ...")
        from ac_lifecycle import full_cycle
        full_cycle(max_retries=3)
        logger.info("AC is live — starting inference episodes.")
    else:
        logger.info("--no-manage-ac: assuming AC is already running.")

    # ── 6. Run episodes ───────────────────────────────────────────────────────
    logger.info(
        "Running %d inference episode(s) with model '%s' (deterministic=True).",
        args.episodes, args.model,
    )

    all_rewards = []
    all_steps = []
    total_steps = 0

    try:
        for ep in range(1, args.episodes + 1):
            result = run_episode(
                env=env,
                sac=sac,
                ep=ep,
                n_episodes=args.episodes,
                model_name=args.model,
                total_steps=total_steps,
            )
            all_rewards.append(result["reward"])
            all_steps.append(result["steps"])
            total_steps += result["steps"]

    except KeyboardInterrupt:
        logger.info("Interrupted by user after %d episode(s).", len(all_rewards))

    # ── 7. Summary ────────────────────────────────────────────────────────────
    if all_rewards:
        n = len(all_rewards)
        mean_reward = sum(all_rewards) / n
        mean_steps  = sum(all_steps) / n
        best_reward = max(all_rewards)
        worst_reward = min(all_rewards)
        print(
            f"\n=== Run summary — model={args.model}  episodes={n} ===\n"
            f"  mean_reward  : {mean_reward:.4f}\n"
            f"  best_reward  : {best_reward:.4f}\n"
            f"  worst_reward : {worst_reward:.4f}\n"
            f"  mean_ep_steps: {mean_steps:.0f}\n"
            f"  total_steps  : {total_steps:,}\n"
            f"=================================================",
            flush=True,
        )
    else:
        logger.warning("No episodes completed.")

    # ── 8. Cleanup ────────────────────────────────────────────────────────────
    logger.info("Closing environment.")
    env.close()

    if args.manage_ac:
        logger.info("Killing AC.")
        from ac_lifecycle import kill_ac
        kill_ac()
        logger.info("Done.")


if __name__ == "__main__":
    main()
