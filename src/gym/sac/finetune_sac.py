"""
finetune_sac.py — Fine-tuning entry point for SAC on Assetto Corsa.

Loads an existing SAC checkpoint and continues training using a stratified
bucketed replay buffer.  Intended to run AFTER a baseline Variant A checkpoint
exists in checkpoints/sac_monza_v1/latest.pt (or a path supplied via
--source-checkpoint).

Run command:
    .\\AssetoCorsa\\Scripts\\python.exe gym/sac/finetune_sac.py
    .\\AssetoCorsa\\Scripts\\python.exe gym/sac/finetune_sac.py \\
        --source-checkpoint checkpoints/sac_monza_v1/latest.pt \\
        --checkpoint-dir   checkpoints/sac_finetune_v1

Differences from train_sac.py:
    1. No episode time limit — episodes run until natural termination only
       (crash detected by stationary-car check or out-of-track done signal).
    2. Bucketed replay buffer — 10 reward-stratified buckets replace the
       DualReplayBuffer.  Sampling is uniform across all 10 buckets so rare
       crash transitions and rare high-reward transitions each receive equal
       gradient signal.
    3. Main loop: COLLECT N episodes → DIVIDE into buckets → BALANCE buckets
       to equal size → TRAIN for n_grad_steps → repeat.
    4. Total buffer capacity: 10 buckets × 50,000 = 500,000 transitions.

Hyperparameters (held constant across all variants — do NOT tune mid-project):
    gamma:          0.992
    batch_size:     256
    lr:             3e-4
    hidden_units:   [256, 256, 256]
    tau:            0.005
    target_entropy: -3  (= -action_dim)
"""

import sys
import os
import argparse
import bisect
import logging
import collections
import math
import random
import time
from typing import List, Optional

import numpy as np

# ── Stationary-car crash detection constants ──────────────────────────────────
# Same values as train_sac.py — do not change.
_STATIONARY_FRAMES = 10
_STATIONARY_THRESHOLD_M = 0.5

# ── Fixed-interval action pacing ──────────────────────────────────────────────
# AC physics runs at 25 Hz (40 ms per tick).  After firing set_actions() we
# sleep the remainder of the 40 ms window before calling step(action=None).
# This ensures every transition in the replay buffer represents the same time
# delta, making the model's temporal reasoning consistent.
# Set to 0.0 to disable and run as fast as possible (legacy behaviour).
STEP_INTERVAL_S: float = 0.009   # 9 ms — ~2× the natural 4.5 ms inter-frame

# ── Path setup — identical to train_sac.py ────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("finetune_sac")

# ── SAC hyperparameters (held constant across all variants) ───────────────────
SAC_HYPERPARAMS = {
    "obs_dim":        125,
    "action_dim":     3,
    "hidden_units":   [256, 256, 256],
    "lr":             3e-4,
    "gamma":          0.992,
    "tau":            0.005,
    "target_entropy": -3.0,  # = -action_dim
}

# ── Bucketed replay buffer config ─────────────────────────────────────────────
# 10 buckets covering [-100, 1.001).  Bucket 0 catches both crashes (very
# negative rewards, e.g. -1.0) and any reward < -0.8.  Bucket 9 holds the
# best frames [0.8, 1.001).
#
# Using 11 edges → 10 half-open intervals [left, right).
BUCKET_BOUNDARIES = [-100.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.001]
N_BUCKETS = 10

# Each bucket is a deque(maxlen=50_000).  Total capacity = 500,000.
BUCKET_CAPACITY = 50_000

# Minimum frames required in EVERY bucket before any training occurs.
MIN_FRAMES_PER_BUCKET = 500

# ── Agent training config ─────────────────────────────────────────────────────
AGENT_CONFIG = {
    "episodes_per_round":    10,      # episodes to collect before dividing / balancing
    "train_steps_per_round": None,    # None = 1:1 with frames collected this round, capped at 6,000
    "batch_size":            256,
    "checkpoint_freq":       5,       # save every N rounds
    "log_interval":          100,     # log training stats every N gradient steps
    "warmup_steps":          37,      # legacy; not used in finetune loop
}


# ── Bucket helpers ────────────────────────────────────────────────────────────

def _bucket_index(reward: float) -> int:
    """Return the 0-based bucket index for a given reward value.

    Bucket i covers [BUCKET_BOUNDARIES[i], BUCKET_BOUNDARIES[i+1]).

    Uses bisect_right on BUCKET_BOUNDARIES[1:] to find the first boundary
    index strictly greater than reward, which gives the bucket directly.

    Examples:
        reward = -100.0 → bucket 0  ([-100, -0.8))
        reward =  -0.8  → bucket 1  ([-0.8, -0.6))
        reward =   0.0  → bucket 5  ([0.0,  0.2))
        reward =   0.85 → bucket 9  ([0.8,  1.001))
        reward =   1.5  → bucket 9  (clamped)

    Parameters
    ----------
    reward : float

    Returns
    -------
    int in [0, N_BUCKETS - 1]
    """
    # bisect_right returns the insertion point in BUCKET_BOUNDARIES for reward,
    # which equals the 1-based index of the last boundary <= reward.
    # Subtracting 1 gives the 0-based bucket index.
    # Clamp to [0, N_BUCKETS-1] for values outside the defined range.
    idx = bisect.bisect_right(BUCKET_BOUNDARIES, reward) - 1
    return max(0, min(idx, N_BUCKETS - 1))


def _make_buckets() -> List[collections.deque]:
    """Allocate N_BUCKETS deques, each with maxlen=BUCKET_CAPACITY."""
    return [collections.deque(maxlen=BUCKET_CAPACITY) for _ in range(N_BUCKETS)]


def _divide_into_buckets(
    staging: list,
    buckets: List[collections.deque],
) -> None:
    """Append each transition in staging to its reward bucket.

    Parameters
    ----------
    staging : list of (obs, action, reward, next_obs, done) tuples
    buckets : list of deques
    """
    for transition in staging:
        reward = transition[2]
        idx = _bucket_index(reward)
        buckets[idx].append(transition)


def _balance_buckets(buckets: List[collections.deque]) -> None:
    """Trim all buckets to the size of the smallest non-empty bucket.

    Oldest entries (left side of deque) are discarded until every bucket
    has the same number of transitions.  This ensures uniform per-bucket
    sampling is unbiased by bucket size disparities.

    If a bucket is empty the balancing is skipped (caller should have
    verified all buckets meet MIN_FRAMES_PER_BUCKET before calling this).
    """
    sizes = [len(b) for b in buckets]
    if min(sizes) == 0:
        return
    target = min(sizes)
    for bucket in buckets:
        while len(bucket) > target:
            bucket.popleft()


def _sample_batch(
    buckets: List[collections.deque],
    batch_size: int,
    device: str,
) -> dict:
    """Sample a balanced batch of batch_size transitions from all 10 buckets.

    Draws batch_size // N_BUCKETS transitions from each bucket (rounding
    down).  Any remainder frames are drawn from the first (batch_size %
    N_BUCKETS) buckets so the total is always exactly batch_size.

    Parameters
    ----------
    buckets : list of deques, each deque element is (obs, action, reward, next_obs, done)
    batch_size : int
    device : str

    Returns
    -------
    dict with keys obs, action, reward, next_obs, done — torch.Tensor on device
    """
    import torch

    base = batch_size // N_BUCKETS
    remainder = batch_size % N_BUCKETS

    obs_list, act_list, rew_list, nobs_list, done_list = [], [], [], [], []

    for i, bucket in enumerate(buckets):
        n = base + (1 if i < remainder else 0)
        samples = random.sample(list(bucket), n)
        for obs, action, reward, next_obs, done in samples:
            obs_list.append(obs)
            act_list.append(action)
            rew_list.append([float(reward)])
            nobs_list.append(next_obs)
            done_list.append([float(done)])

    # Shuffle so that bucket ordering does not create any gradient bias.
    perm = list(range(len(obs_list)))
    random.shuffle(perm)
    obs_list   = [obs_list[i]   for i in perm]
    act_list   = [act_list[i]   for i in perm]
    rew_list   = [rew_list[i]   for i in perm]
    nobs_list  = [nobs_list[i]  for i in perm]
    done_list  = [done_list[i]  for i in perm]

    def _t(lst):
        return torch.as_tensor(
            np.array(lst, dtype=np.float32), dtype=torch.float32
        ).to(device)

    return {
        "obs":      _t(obs_list),
        "action":   _t(act_list),
        "reward":   _t(rew_list),
        "next_obs": _t(nobs_list),
        "done":     _t(done_list),
    }


# ── Logging helpers ───────────────────────────────────────────────────────────

def _save_buffer(buckets: List[collections.deque], ckpt_dir: str) -> None:
    """Merge all 10 bucket deques into a single numpy array and save as buffer.npz."""
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    for bucket in buckets:
        for obs, action, reward, next_obs, done in bucket:
            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_obs.append(next_obs)
            all_dones.append(done)
    buf_path = os.path.join(ckpt_dir, "buffer.npz")
    np.savez_compressed(
        buf_path,
        obs=np.array(all_obs),
        action=np.array(all_actions),
        reward=np.array(all_rewards),
        next_obs=np.array(all_next_obs),
        done=np.array(all_dones),
    )
    total = sum(len(b) for b in buckets)
    logger.info(f"Buffer saved to {buf_path} — {total} transitions")


def _load_buffer(ckpt_dir: str, buckets: List[collections.deque]) -> None:
    """Load buffer.npz from ckpt_dir and divide transitions back into buckets."""
    buf_path = os.path.join(ckpt_dir, "buffer.npz")
    if not os.path.isfile(buf_path):
        return
    data = np.load(buf_path)
    staging = list(zip(data["obs"], data["action"], data["reward"], data["next_obs"], data["done"]))
    _divide_into_buckets(staging, buckets)
    logger.info(f"Buffer loaded from {buf_path} — {len(staging)} transitions")


def _bucket_summary(buckets: List[collections.deque]) -> str:
    """Return a compact one-line string showing each bucket's fill level."""
    parts = []
    for i, b in enumerate(buckets):
        lo = BUCKET_BOUNDARIES[i]
        hi = BUCKET_BOUNDARIES[i + 1]
        parts.append(f"B{i}[{lo:.1f},{hi:.2f})={len(b)}")
    return "  ".join(parts)


def _print_episode_table(
    round_num: int,
    ep: int,
    n_episodes: int,
    ep_steps: int,
    ep_reward: float,
    ep_positive_steps: int,
    total_env_steps: int,
    buckets: List[collections.deque],
    info: dict,
    mean_inference_ms: float = 0.0,
    mean_interframe_ms: float = 0.0,
    mean_response_latency_ms: float = 0.0,
    last_frame_game_time: float = 0.0,
    last_response_latency_ms: float = 0.0,
) -> None:
    """Print a compact fixed-width table for one completed episode."""
    SEP = "+" + "-" * 22 + "+" + "-" * 16 + "+"

    def row(label: str, value: str) -> str:
        return f"| {label:<20} | {value:>14} |"

    bucket_total = sum(len(b) for b in buckets)
    lines = [
        SEP,
        row("Round / Episode", f"{round_num} / {ep} of {n_episodes}"),
        row("Steps (episode)", f"{ep_steps:,}"),
        row("Total env steps", f"{total_env_steps:,}"),
        row("Reward (episode)", f"{ep_reward:.4f}"),
        row("Avg reward/frame",
            f"{ep_reward / ep_steps:.6f}" if ep_steps > 0 else "N/A"),
        row("Positive reward frames",
            f"{ep_positive_steps} / {ep_steps} "
            f"({100 * ep_positive_steps / ep_steps:.1f}%)"
            if ep_steps > 0 else "N/A"),
        row("Bucket total", f"{bucket_total:,}"),
    ]

    # Speed / lap info from env info dict.
    _speed_val: Optional[float] = None
    for _k in ("avg_speed", "mean_speed"):
        if _k in info:
            _speed_val = float(info[_k])
            break
    if _speed_val is None and "speed_multiplier" in info:
        _speed_val = float(info["speed_multiplier"]) - 1.0 + 40.0
    if _speed_val is not None:
        lines.append(row("Mean speed", f"{_speed_val:.2f}"))
    for _lap_k in ("lap_time", "lap_time_s"):
        if _lap_k in info:
            lines.append(row("Lap time", f"{float(info[_lap_k]):.3f}"))
            break
    for _ot_k in ("off_track_count", "off_track"):
        if _ot_k in info:
            lines.append(row("Off-track count", f"{int(info[_ot_k]):,}"))
            break

    lines.append(row("Avg inference time", f"{mean_inference_ms:.1f} ms"))
    _ift_str = f"{mean_interframe_ms:.1f} ms"
    if mean_interframe_ms > 40.0:
        _ift_str += " !!"
    lines.append(row("Avg inter-frame time", _ift_str))
    lines.append(row("Avg response latency", f"{mean_response_latency_ms:.1f} ms"))
    lines.append(SEP)
    print("\n".join(lines), flush=True)
    print(
        f"  [Timing] Last frame at {last_frame_game_time:.0f}ms (AC time), "
        f"action sent ~{last_frame_game_time + last_response_latency_ms:.0f}ms",
        flush=True,
    )


# ── Environment builder ───────────────────────────────────────────────────────

def build_env(config_path: str, work_dir: str):
    """Build the AssettoCorsaEnv, wrap in OurEnv, and disable the episode
    step-count limit so episodes terminate only on natural done signals.

    Returns
    -------
    OurEnv instance
    """
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

    os.makedirs(work_dir, exist_ok=True)
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)

    # Remove the episode time limit.  max_episode_py_time in config.yml
    # (default 120 s ≈ 3000 steps) is converted to max_episode_steps in
    # ModuleConfig.__init__ and stored on ac_env._max_episode_steps.
    # ac_env.py line 560 guards: `if self._max_episode_steps is not None:`
    # so setting it to None disables the cap entirely.
    if hasattr(ac_env, '_max_episode_steps'):
        old_limit = ac_env._max_episode_steps
        ac_env._max_episode_steps = None
        logger.info(
            f"Episode step limit disabled "
            f"(was {old_limit} steps = "
            f"{old_limit / cfg.AssettoCorsa.ego_sampling_freq:.0f}s). "
            f"Episodes now run until natural termination only."
        )
    else:
        logger.warning(
            "ac_env has no '_max_episode_steps' attribute — "
            "episode limit may still be active."
        )

    logger.info(
        f"AssettoCorsaEnv built — "
        f"obs_dim={ac_env.state_dim}  action_dim={ac_env.action_dim}"
    )

    our_env_cfg = OmegaConf.create({
        "our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)
    })
    env = OurEnv(ac_env, our_env_cfg)
    logger.info("OurEnv ready.")
    return env


# ── Device helper ─────────────────────────────────────────────────────────────

def _resolve_device(device_arg: str) -> str:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


# ── Collection helper ─────────────────────────────────────────────────────────

def collect_episodes(
    env,
    sac,
    n_episodes: int,
    round_num: int,
    total_env_steps_ref: list,   # mutable single-element list used as a ref
    buckets: List[collections.deque],
    manage_ac: bool = False,
) -> list:
    """Run n_episodes and return a flat staging list of transitions.

    Uses the latency-overlap trick (identical to agent.py collect_phase):
        obs = next_obs           # frame N arrived
        action = select_action() # infer on frame N
        env.set_actions(action)  # fire immediately (non-blocking)
        <CPU work overlaps with AC physics tick>
        env.step(action=None)    # wait for next frame

    Parameters
    ----------
    env : OurEnv
    sac : SAC
    n_episodes : int
    round_num : int
        Used only for logging.
    total_env_steps_ref : list
        Single-element list [int] that is incremented here so the caller's
        counter stays in sync.
    buckets : list of deques
        Passed only for reporting bucket fill levels in the episode table.
    manage_ac : bool
        Not used during collection (caller handles AC lifecycle).

    Returns
    -------
    list of (obs, action, reward, next_obs, done) tuples (numpy arrays / scalars)
    """
    staging: list = []

    _stat_frames    = _STATIONARY_FRAMES
    _stat_threshold = _STATIONARY_THRESHOLD_M

    for ep in range(n_episodes):
        obs, _info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_positive_steps = 0
        done = False
        ep_final_info = {}
        inference_times: list = []
        interframe_times: list = []
        response_latency_times: list = []
        last_frame_game_time: float = 0.0
        last_response_latency_ms: float = 0.0
        _pos_window: collections.deque = collections.deque(maxlen=_stat_frames)

        # ── Warmup: full throttle for a random number of steps ────────────────
        # Identical to agent.py — warmup frames are NOT stored.
        _warmup_steps = random.randint(150, 250)
        for _ in range(_warmup_steps):
            obs, _, done, _ = env.step(np.array([0.5, 1.0, 0.0], dtype=np.float32))
            env._read_latest_state()

        # ── Step 0: blocking step to seed the overlap loop ────────────────────
        _step0_ran = False
        if not done:
            _t_inf0_start = time.perf_counter()
            action = sac.select_action(obs, deterministic=False)
            _t_inf0_end = time.perf_counter()
            inference_times.append(_t_inf0_end - _t_inf0_start)
            prev_obs    = obs
            prev_action = action
            next_obs, reward, done, step_info = env.step(action)
            env._read_latest_state()
            _step0_ran = True
            _t_frame = time.perf_counter()

            if isinstance(step_info, dict):
                _sx = step_info.get("world_position_x")
                _sy = step_info.get("world_position_y")
                if _sx is not None and _sy is not None:
                    _pos_window.append((float(_sx), float(_sy)))
                    if len(_pos_window) == _stat_frames:
                        _dx = _pos_window[-1][0] - _pos_window[0][0]
                        _dy = _pos_window[-1][1] - _pos_window[0][1]
                        if math.sqrt(_dx * _dx + _dy * _dy) < _stat_threshold:
                            logger.info(
                                f"[Round {round_num}] ep={ep + 1} step=0 "
                                f"stationary crash — ending episode."
                            )
                            done = True
                if done:
                    ep_final_info = step_info
            else:
                if done:
                    ep_final_info = {}

        # ── Steps 1+: latency-overlap loop ────────────────────────────────────
        while not done:
            # 1. Advance to the arrived frame.
            obs = next_obs

            # 2. Infer on current frame — this is the action AC needs next.
            _t_inf_start = time.perf_counter()
            action = sac.select_action(obs, deterministic=False)
            _t_inf_end = time.perf_counter()
            inference_times.append(_t_inf_end - _t_inf_start)

            # 3. Fire to AC immediately (non-blocking).
            env.set_actions(action)
            t_action_sent = time.perf_counter()
            _resp_ms = (t_action_sent - _t_frame) * 1000.0
            response_latency_times.append(_resp_ms)
            last_response_latency_ms = _resp_ms

            # 4. CPU work — overlaps with the AC physics tick.
            #    Store the PREVIOUS transition (prev_obs →(prev_action)→ obs).
            env._read_latest_state()
            staging.append((
                prev_obs.copy(),
                prev_action.copy(),
                float(reward),
                obs.copy(),
                float(done),
            ))

            # Accumulate episode statistics.
            ep_reward += reward
            ep_steps += 1
            if reward > 0:
                ep_positive_steps += 1
            total_env_steps_ref[0] += 1

            # 5. Pace to fixed interval then wait for AC's next frame.
            #    Sleep the remainder of STEP_INTERVAL_S so every transition
            #    represents the same wall-clock delta regardless of inference speed.
            if STEP_INTERVAL_S > 0.0:
                _step_deadline = _t_frame + STEP_INTERVAL_S
                _remaining = _step_deadline - time.perf_counter()
                if _remaining > 0.0:
                    time.sleep(_remaining)

            prev_obs    = obs
            prev_action = action
            _t_next = time.perf_counter()
            interframe_times.append(_t_next - _t_frame)
            next_obs, reward, done, step_info = env.step(action=None)
            _t_frame = time.perf_counter()

            if isinstance(step_info, dict):
                last_frame_game_time = float(step_info.get("currentTime", 0.0))
                _sx = step_info.get("world_position_x")
                _sy = step_info.get("world_position_y")
                if _sx is not None and _sy is not None:
                    _pos_window.append((float(_sx), float(_sy)))
                    if len(_pos_window) == _stat_frames:
                        _dx = _pos_window[-1][0] - _pos_window[0][0]
                        _dy = _pos_window[-1][1] - _pos_window[0][1]
                        if math.sqrt(_dx * _dx + _dy * _dy) < _stat_threshold:
                            logger.info(
                                f"[Round {round_num}] ep={ep + 1} "
                                f"step={ep_steps + 1} "
                                f"stationary crash — ending episode."
                            )
                            done = True
                if done:
                    ep_final_info = step_info
            else:
                if done:
                    ep_final_info = {}

        # Push terminal transition (same two-case logic as agent.py).
        if _step0_ran:
            staging.append((
                prev_obs.copy(),
                prev_action.copy(),
                float(reward),
                next_obs.copy(),
                float(done),
            ))
            ep_reward += reward
            ep_steps += 1
            if reward > 0:
                ep_positive_steps += 1
            total_env_steps_ref[0] += 1

        _mean_inf_ms  = float(np.mean(inference_times))  * 1000.0 if inference_times  else 0.0
        _mean_ift_ms  = float(np.mean(interframe_times)) * 1000.0 if interframe_times else 0.0
        _mean_resp_ms = float(np.mean(response_latency_times))     if response_latency_times else 0.0

        _print_episode_table(
            round_num=round_num,
            ep=ep + 1,
            n_episodes=n_episodes,
            ep_steps=ep_steps,
            ep_reward=ep_reward,
            ep_positive_steps=ep_positive_steps,
            total_env_steps=total_env_steps_ref[0],
            buckets=buckets,
            info=ep_final_info,
            mean_inference_ms=_mean_inf_ms,
            mean_interframe_ms=_mean_ift_ms,
            mean_response_latency_ms=_mean_resp_ms,
            last_frame_game_time=last_frame_game_time,
            last_response_latency_ms=last_response_latency_ms,
        )
        logger.info(
            f"  ep={ep + 1}/{n_episodes}  steps={ep_steps}  "
            f"reward={ep_reward:.3f}  "
            f"bucket_sizes=[{', '.join(str(len(b)) for b in buckets)}]"
        )

    return staging


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAC fine-tuning with stratified bucketed replay buffer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            _THIS_DIR, '..', '..', 'assetto_corsa_gym', 'config.yml'
        ),
        help="Path to config.yml (default: assetto_corsa_gym/config.yml)",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=str,
        default=None,
        help=(
            "SAC checkpoint to load weights from.  "
            "Default: checkpoints/sac_monza_v1/latest.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory to save fine-tuning checkpoints.  "
            "Default: checkpoints/sac_finetune_v1"
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of COLLECT→BALANCE→TRAIN rounds (default: None = run forever)",
    )
    parser.add_argument(
        "--episodes-per-round",
        type=int,
        default=AGENT_CONFIG["episodes_per_round"],
        help=f"Episodes to collect per round (default: {AGENT_CONFIG['episodes_per_round']})",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help=(
            "Gradient steps per round.  "
            "Default: None = 1:1 with frames collected this round, capped at 6,000."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip AC connection preflight checks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device (default: auto)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--manage-ac",
        action="store_true",
        help=(
            "Launch AC via Content Manager before each collection round. "
            "Kills AC before the training phase to free VRAM."
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 1. Preflight ──────────────────────────────────────────────────────────
    if args.manage_ac:
        logger.info("--manage-ac active — preflight skipped.")
    elif not args.skip_preflight:
        try:
            from preflight import run_preflight
            run_preflight(warn_only=False)
        except ImportError:
            logger.warning(
                "preflight module not found — skipping.  "
                "Use --skip-preflight to suppress this warning."
            )
    else:
        logger.warning("Preflight skipped — ensure AC is running in a session.")

    # ── 2. Device ─────────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── 3. Environment ────────────────────────────────────────────────────────
    config_path = os.path.abspath(args.config)
    logger.info(f"Loading config: {config_path}")

    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'outputs', 'sac_finetune',
    )
    env = build_env(config_path, work_dir)

    # ── 4. Bucketed replay buffer ─────────────────────────────────────────────
    buckets = _make_buckets()
    logger.info(
        f"Bucketed replay buffer: {N_BUCKETS} buckets × {BUCKET_CAPACITY:,} capacity  "
        f"(total={N_BUCKETS * BUCKET_CAPACITY:,})  "
        f"min_frames_per_bucket={MIN_FRAMES_PER_BUCKET:,}  "
        f"boundaries={BUCKET_BOUNDARIES}"
    )

    # ── 5. SAC algorithm ──────────────────────────────────────────────────────
    from sac.sac import SAC

    ckpt_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'checkpoints', 'sac_finetune_v1',
    )
    os.makedirs(ckpt_dir, exist_ok=True)

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

    # Load from source checkpoint (base training run) if provided or default.
    src_ckpt = args.source_checkpoint
    if src_ckpt is None:
        src_ckpt = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', 'checkpoints', 'sac_monza_v1', 'latest.pt',
        )

    # If a fine-tuning checkpoint already exists (resume), prefer it over the
    # source.  Otherwise load from the source (initial weight transfer).
    resume_ckpt = os.path.join(os.path.abspath(ckpt_dir), "latest.pt")
    if os.path.isfile(resume_ckpt):
        logger.info(f"Resuming fine-tuning from {resume_ckpt}")
        sac.load(resume_ckpt)
    elif os.path.isfile(src_ckpt):
        logger.info(f"Loading base weights from {src_ckpt}")
        sac.load(src_ckpt)
    else:
        logger.warning(
            f"No checkpoint found at {resume_ckpt} or {src_ckpt}. "
            f"Starting fine-tuning from random weights."
        )

    # ── 6. Config resolution ──────────────────────────────────────────────────
    episodes_per_round  = args.episodes_per_round
    train_steps_cfg     = args.train_steps  # None = dynamic
    batch_size          = AGENT_CONFIG["batch_size"]
    ckpt_freq           = AGENT_CONFIG["checkpoint_freq"]
    log_interval        = AGENT_CONFIG["log_interval"]

    total_env_steps_ref  = [0]   # mutable single-element list used as a counter ref
    total_grad_steps     = 0

    # ── 6b. Buffer restore ────────────────────────────────────────────────────
    _load_buffer(os.path.abspath(ckpt_dir), buckets)

    # ── 7. Main loop: COLLECT → DIVIDE → BALANCE → TRAIN → repeat ─────────────
    logger.info(
        f"Starting SAC fine-tuning — "
        f"rounds={'inf' if args.rounds is None else args.rounds}  "
        f"device={device}  "
        f"episodes_per_round={episodes_per_round}  "
        f"batch_size={batch_size}  "
        f"gamma={SAC_HYPERPARAMS['gamma']}  "
        f"hidden={SAC_HYPERPARAMS['hidden_units']}  "
        f"target_entropy={SAC_HYPERPARAMS['target_entropy']}"
    )

    round_num = 0

    try:
        while args.rounds is None or round_num < args.rounds:
            logger.info(
                f"\n{'=' * 60}\n"
                f"  ROUND {round_num}  "
                f"(env_steps={total_env_steps_ref[0]}  "
                f"grad_steps={total_grad_steps})\n"
                f"{'=' * 60}"
            )

            # ── Launch AC before collection ────────────────────────────────────
            if args.manage_ac:
                try:
                    from ac_lifecycle import full_cycle
                except ImportError as exc:
                    raise RuntimeError(
                        "manage_ac=True but ac_lifecycle.py could not be imported. "
                        f"Original error: {exc}"
                    ) from exc
                logger.info(f"[Round {round_num}] Launching AC via full_cycle() ...")
                full_cycle(max_retries=3)
                logger.info(f"[Round {round_num}] AC is live — starting collection.")

            # ── COLLECT ────────────────────────────────────────────────────────
            logger.info(
                f"[Round {round_num}] COLLECT — {episodes_per_round} episodes  "
                f"bucket_sizes=[{', '.join(str(len(b)) for b in buckets)}]"
            )
            staging = collect_episodes(
                env=env,
                sac=sac,
                n_episodes=episodes_per_round,
                round_num=round_num,
                total_env_steps_ref=total_env_steps_ref,
                buckets=buckets,
                manage_ac=args.manage_ac,
            )
            logger.info(
                f"[Round {round_num}] COLLECT done — "
                f"staging={len(staging)} transitions  "
                f"total_env_steps={total_env_steps_ref[0]}"
            )

            # ── Kill AC before GPU work to free resources ──────────────────────
            if args.manage_ac:
                from ac_lifecycle import kill_ac
                logger.info(f"[Round {round_num}] Killing AC before training ...")
                kill_ac()
                logger.info(f"[Round {round_num}] AC killed.")

            # ── DIVIDE ─────────────────────────────────────────────────────────
            _divide_into_buckets(staging, buckets)
            logger.info(
                f"[Round {round_num}] DIVIDE — {_bucket_summary(buckets)}"
            )

            # ── WAIT / CHECK ───────────────────────────────────────────────────
            # Do not train until every bucket has at least MIN_FRAMES_PER_BUCKET
            # transitions.  Log which buckets are still under-filled.
            under_filled = [
                i for i, b in enumerate(buckets) if len(b) < MIN_FRAMES_PER_BUCKET
            ]
            if under_filled:
                logger.info(
                    f"[Round {round_num}] Waiting for buckets to fill — "
                    f"under-filled buckets (need {MIN_FRAMES_PER_BUCKET}): "
                    f"{under_filled}  |  "
                    f"sizes=[{', '.join(str(len(b)) for b in buckets)}]"
                )
                round_num += 1
                continue  # skip BALANCE + TRAIN

            # ── BALANCE ────────────────────────────────────────────────────────
            sizes_before = [len(b) for b in buckets]
            _balance_buckets(buckets)
            sizes_after  = [len(b) for b in buckets]
            dropped = [b - a for b, a in zip(sizes_before, sizes_after)]
            logger.info(
                f"[Round {round_num}] BALANCE — "
                f"target={sizes_after[0]}  "
                f"dropped_per_bucket={dropped}  "
                f"total_kept={sum(sizes_after):,}"
            )

            # ── TRAIN ──────────────────────────────────────────────────────────
            min_bucket_size = min(len(b) for b in buckets)
            # Guard: need at least batch_size // N_BUCKETS samples per bucket.
            min_needed = math.ceil(batch_size / N_BUCKETS)
            if min_bucket_size < min_needed:
                logger.warning(
                    f"[Round {round_num}] Smallest bucket has {min_bucket_size} frames "
                    f"but sampling needs {min_needed} per bucket — skipping TRAIN."
                )
                round_num += 1
                continue

            # Determine gradient steps: 1:1 with transitions staged this round,
            # capped at 6,000; or the configured override.
            if train_steps_cfg is not None:
                n_grad_steps = train_steps_cfg
                _ratio_note  = "fixed"
            else:
                n_grad_steps = min(len(staging), 6_000)
                _ratio_note  = "1:1 with staged frames"

            _total_buf = sum(len(b) for b in buckets)
            logger.info(
                f"[Round {round_num}] TRAIN — {n_grad_steps} gradient steps ({_ratio_note})  "
                f"batch_size={batch_size}  "
                f"bucket_total={_total_buf:,}"
            )

            q_losses, policy_losses, alphas, entropies = [], [], [], []
            t_train_start = time.perf_counter()

            for step in range(1, n_grad_steps + 1):
                batch = _sample_batch(buckets, batch_size, device)
                losses = sac.update_from_batch(batch)

                q_losses.append(losses["q_loss"])
                policy_losses.append(losses["policy_loss"])
                alphas.append(losses["alpha"])
                entropies.append(losses["entropy"])
                total_grad_steps += 1

                if step % log_interval == 0:
                    recent = slice(-log_interval, None)
                    elapsed = time.perf_counter() - t_train_start
                    logger.info(
                        f"  [Round {round_num}] step={step}/{n_grad_steps}  "
                        f"q_loss={float(np.mean(q_losses[recent])):.4f}  "
                        f"pi_loss={float(np.mean(policy_losses[recent])):.4f}  "
                        f"alpha={float(np.mean(alphas[recent])):.4f}  "
                        f"entropy={float(np.mean(entropies[recent])):.4f}  "
                        f"elapsed={elapsed:.1f}s  "
                        f"total_grad_steps={total_grad_steps}"
                    )

            elapsed_train = time.perf_counter() - t_train_start
            logger.info(
                f"[Round {round_num}] TRAIN done — "
                f"mean_q_loss={float(np.mean(q_losses)):.4f}  "
                f"mean_pi_loss={float(np.mean(policy_losses)):.4f}  "
                f"mean_alpha={float(np.mean(alphas)):.4f}  "
                f"mean_entropy={float(np.mean(entropies)):.4f}  "
                f"elapsed={elapsed_train:.1f}s"
            )

            # ── Phase summary ──────────────────────────────────────────────────
            logger.info(
                f"\n--- Round {round_num} summary ---\n"
                f"  Collected:  {len(staging)} transitions  "
                f"total_env_steps={total_env_steps_ref[0]}\n"
                f"  Buckets:    {_bucket_summary(buckets)}\n"
                f"  Training:   grad_steps={n_grad_steps}  "
                f"total_grad_steps={total_grad_steps}\n"
                f"---"
            )

            # ── Checkpoint ────────────────────────────────────────────────────
            if (round_num + 1) % ckpt_freq == 0:
                ckpt_path = os.path.join(os.path.abspath(ckpt_dir), "latest.pt")
                sac.save(ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}  (round={round_num})")
                _save_buffer(buckets, os.path.abspath(ckpt_dir))

            round_num += 1

    except KeyboardInterrupt:
        logger.info(
            f"\nFine-tuning stopped by user at round {round_num}  "
            f"(env_steps={total_env_steps_ref[0]}  "
            f"grad_steps={total_grad_steps})"
        )
        ckpt_path = os.path.join(os.path.abspath(ckpt_dir), "latest.pt")
        sac.save(ckpt_path)
        logger.info(f"Checkpoint saved on interrupt: {ckpt_path}")
        _save_buffer(buckets, os.path.abspath(ckpt_dir))

    # ── Cleanup ───────────────────────────────────────────────────────────────
    logger.info("Fine-tuning complete — closing environment.")
    env.close()


if __name__ == "__main__":
    main()
