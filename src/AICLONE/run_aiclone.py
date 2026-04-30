"""
run_aiclone.py — Inference runner for a BC pre-trained AICLONE actor.

Loads a GaussianPolicy checkpoint produced by pretrain_actor.py and runs
the policy in deterministic mode (tanh(mean), no sampling noise) against
Assetto Corsa.  No Q-networks, no training, no replay buffer.

Checkpoint format (from pretrain_actor.py):
    {"policy": state_dict}   — actor weights only

Usage
-----
    # Best checkpoint (default)
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/run_aiclone.py

    # Specific checkpoint
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/run_aiclone.py \\
        --checkpoint AICLONE/checkpoints/final.pt

    # Multiple episodes, AC already running
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/run_aiclone.py \\
        --checkpoint AICLONE/checkpoints/best.pt \\
        --episodes 5 --no-manage-ac

Latency-overlap trick (same as run_model.py / agent.py):
    1. select_action(obs)      — infer on current frame
    2. env.set_actions(action) — fire to AC immediately (~0.1 ms)
    3. env._read_latest_state()— CPU work overlaps with 40 ms physics tick
    4. env.step(action=None)   — block only for the next frame
"""

import sys
import os
import argparse
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_AICLONE_DIR = Path(__file__).resolve().parent
_REPO_DIR    = _AICLONE_DIR.parent
_GYM_DIR     = _REPO_DIR / "gym"

sys.path.insert(0, str(_REPO_DIR / "assetto_corsa_gym"))
sys.path.insert(0, str(_REPO_DIR / "assetto_corsa_gym" / "assetto_corsa_gym"))
sys.path.insert(0, str(_REPO_DIR / "assetto_corsa_gym" / "algorithm" / "discor"))
sys.path.insert(0, str(_GYM_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("run_aiclone")

# ── Architecture constants — must match pretrain_actor.py ─────────────────────
OBS_DIM    = 125
ACTION_DIM = 3
HIDDEN     = [256, 256, 256]

# ── Stuck-detection constants (same as run_model.py) ──────────────────────────
_STUCK_WINDOW    = 10
_STUCK_THRESHOLD = 0.5   # metres — total movement over window


# ── Policy wrapper ────────────────────────────────────────────────────────────

class AIClonePolicy:
    """
    Thin wrapper around a loaded GaussianPolicy for deterministic inference.
    Mirrors the sac.select_action(obs, deterministic=True) interface.
    """

    def __init__(self, checkpoint_path: str, device: str):
        from sac.network import GaussianPolicy

        self.device = torch.device(device)
        self.policy = GaussianPolicy(
            state_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_units=HIDDEN,
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy.eval()
        logger.info(f"Loaded policy from: {checkpoint_path}")

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action — returns tanh(mean), no sampling noise."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _, _, mean_action = self.policy(obs_t)
        return mean_action.squeeze(0).cpu().numpy()


# ── Environment builder (identical to run_model.py) ───────────────────────────

def _build_env(config_path: str):
    from omegaconf import OmegaConf
    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    from our_env import OurEnv

    cfg = OmegaConf.load(config_path)
    cfg.AssettoCorsa.max_episode_py_time = 1_000_000.0

    work_dir = str(_REPO_DIR / "outputs" / "run_aiclone")
    os.makedirs(work_dir, exist_ok=True)

    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    our_env_cfg = OmegaConf.create({"our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)})
    env = OurEnv(ac_env, our_env_cfg)
    logger.info("OurEnv ready.")
    return env


# ── Stats printer (identical to run_model.py) ─────────────────────────────────

def _print_episode_stats(ep, n_episodes, ep_steps, ep_reward,
                         ep_positive_steps, total_steps,
                         mean_inference_ms, mean_interframe_ms):
    SEP = "+" + "-" * 26 + "+" + "-" * 16 + "+"

    def row(label, value):
        return f"| {label:<24} | {value:>14} |"

    lines = [
        SEP,
        row("Runner",              "AICLONE (BC actor)"),
        row("Episode",             f"{ep} of {n_episodes}"),
        row("Steps (episode)",     f"{ep_steps:,}"),
        row("Total steps (run)",   f"{total_steps:,}"),
        row("Reward (episode)",    f"{ep_reward:.4f}"),
        row("Avg reward/step",     f"{ep_reward/ep_steps:.6f}" if ep_steps else "N/A"),
        row("Positive reward steps",
            f"{ep_positive_steps}/{ep_steps} "
            f"({100*ep_positive_steps/ep_steps:.1f}%)" if ep_steps else "N/A"),
        row("Avg inference time",  f"{mean_inference_ms:.1f} ms"),
        row("Avg inter-frame time",
            f"{mean_interframe_ms:.1f} ms" + ("  !!" if mean_interframe_ms > 40.0 else "")),
        SEP,
    ]
    print("\n".join(lines), flush=True)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, policy: AIClonePolicy, ep: int, n_episodes: int,
                total_steps: int) -> dict:
    """
    Run one inference episode with the latency-overlap trick.
    Mirrors run_model.py:run_episode() but uses AIClonePolicy.select_action().
    """
    obs, _info = env.reset()
    ep_reward, ep_steps, ep_positive_steps = 0.0, 0, 0
    done = False
    inference_times, interframe_times = [], []
    pos_history = deque(maxlen=_STUCK_WINDOW)

    def _check_stuck(info_dict):
        x = float(info_dict.get("world_position_x", 0.0))
        y = float(info_dict.get("world_position_y", 0.0))
        pos_history.append((x, y))
        if len(pos_history) < _STUCK_WINDOW:
            return False
        oldest = pos_history[0]
        dist = ((x - oldest[0]) ** 2 + (y - oldest[1]) ** 2) ** 0.5
        return dist < _STUCK_THRESHOLD

    # ── Throttle warmup (same range as agent.py) ──────────────────────────────
    import random
    warmup_steps = random.randint(150, 250)
    logger.info("Throttle warmup: %d steps", warmup_steps)
    for _ in range(warmup_steps):
        obs, _, done, _ = env.step(np.array([0.5, 1.0, 0.0], dtype=np.float32))
        env._read_latest_state()
        if done:
            logger.warning("Episode ended during warmup — skipping")
            return {"steps": 0, "reward": 0.0, "positive_steps": 0}

    # ── Step 0: seed the overlap loop ─────────────────────────────────────────
    _t0 = time.perf_counter()
    action = policy.select_action(obs)
    inference_times.append(time.perf_counter() - _t0)
    next_obs, reward, done, step_info = env.step(action)
    env._read_latest_state()
    _t_frame = time.perf_counter()
    if isinstance(step_info, dict) and _check_stuck(step_info):
        done = True

    # ── Steps 1+: latency-overlap loop ────────────────────────────────────────
    while not done:
        obs = next_obs

        _t_inf = time.perf_counter()
        action = policy.select_action(obs)
        inference_times.append(time.perf_counter() - _t_inf)

        env.set_actions(action)
        env._read_latest_state()

        ep_reward += reward
        ep_steps  += 1
        if reward > 0:
            ep_positive_steps += 1

        _t_next = time.perf_counter()
        interframe_times.append(_t_next - _t_frame)

        next_obs, reward, done, step_info = env.step(action=None)
        _t_frame = time.perf_counter()

        if isinstance(step_info, dict) and not done and _check_stuck(step_info):
            logger.warning("Car stuck at step %d — ending episode", ep_steps)
            done = True

    # Terminal step
    ep_reward += reward
    ep_steps  += 1
    if reward > 0:
        ep_positive_steps += 1

    mean_inf_ms = sum(inference_times) / len(inference_times) * 1000.0 if inference_times else 0.0
    mean_ift_ms = sum(interframe_times) / len(interframe_times) * 1000.0 if interframe_times else 0.0

    _print_episode_stats(
        ep, n_episodes, ep_steps, ep_reward,
        ep_positive_steps, total_steps + ep_steps,
        mean_inf_ms, mean_ift_ms,
    )
    return {"steps": ep_steps, "reward": ep_reward, "positive_steps": ep_positive_steps}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AICLONE inference runner — BC pre-trained actor against AC"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(_AICLONE_DIR / "checkpoints" / "best.pt"),
        help="Path to pretrain_actor.py checkpoint (default: AICLONE/checkpoints/best.pt)",
    )
    parser.add_argument("--episodes",   type=int, default=5)
    parser.add_argument(
        "--config",
        default=str(_REPO_DIR / "assetto_corsa_gym" / "config.yml"),
    )
    parser.add_argument("--device",  default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--manage-ac",    action="store_true", default=True)
    parser.add_argument("--no-manage-ac", dest="manage_ac", action="store_false")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    logger.info("Device: %s", device)

    # ── Load policy ───────────────────────────────────────────────────────────
    ckpt_path = str(Path(args.checkpoint).resolve())
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    policy = AIClonePolicy(ckpt_path, device)

    # ── Build env ─────────────────────────────────────────────────────────────
    env = _build_env(str(Path(args.config).resolve()))

    # ── AC lifecycle ──────────────────────────────────────────────────────────
    if args.manage_ac:
        from ac_lifecycle import full_cycle
        logger.info("Launching AC via full_cycle() ...")
        full_cycle(max_retries=3)
        logger.info("AC is live.")
    else:
        logger.info("--no-manage-ac: assuming AC is already running.")

    # ── Run episodes ──────────────────────────────────────────────────────────
    all_rewards, all_steps = [], []
    total_steps = 0

    try:
        for ep in range(1, args.episodes + 1):
            result = run_episode(env, policy, ep, args.episodes, total_steps)
            all_rewards.append(result["reward"])
            all_steps.append(result["steps"])
            total_steps += result["steps"]
    except KeyboardInterrupt:
        logger.info("Interrupted after %d episode(s).", len(all_rewards))

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_rewards:
        n = len(all_rewards)
        print(
            f"\n=== AICLONE run summary  episodes={n} ===\n"
            f"  mean_reward  : {sum(all_rewards)/n:.4f}\n"
            f"  best_reward  : {max(all_rewards):.4f}\n"
            f"  worst_reward : {min(all_rewards):.4f}\n"
            f"  mean_steps   : {sum(all_steps)/n:.0f}\n"
            f"  total_steps  : {total_steps:,}\n"
            f"==========================================",
            flush=True,
        )

    env.close()
    if args.manage_ac:
        from ac_lifecycle import kill_ac
        kill_ac()


if __name__ == "__main__":
    main()
