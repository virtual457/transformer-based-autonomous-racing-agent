"""
tune_tail.py — fine-tune the saved checkpoint on *tail* data with a lower LR.

Idea
----
After collecting fresh Roggia rollouts (~10K frames, which push new windows
into all 6 channels based on per-channel reward sign), we want the model to
*bias* its next learning pass toward the new experience while still doing
gradient steps on a coherent multi-channel mix.

Three sampling strategies are provided:

1. ``all-tail``   — last N windows from each of the 6 channels (user's ask).
2. ``neg-tail``   — last N from NEG channels; FULL buffer from POS channels.
                    Rationale: Roggia failures live in the NEG channels
                    (brake/throttle/steer_neg). POS channels' "last N" is just
                    unrelated older good-driving windows, so we keep those
                    broad to preserve the general policy.
3. ``mix``        — 50/50 sampling per channel: half from tail N, half from
                    the full-buffer filled portion. Best of both — fresh data
                    gets over-represented without starving the model of prior
                    good examples.

The fine-tune uses a lower learning rate (default 1e-4 vs. 3e-4 during
initial training) so we nudge the policy rather than restart it.

Usage
-----
    # Backup + neg-tail (recommended): nudge the policy toward fresh Roggia
    # failures while keeping POS channels broad.
    python gym/transformer_sac_vectorq_v2_final_fineTune/tune_tail.py \\
        --strategy neg-tail --tail-n 20000 --lr 1e-4 --epochs 2

    # Exactly what the user asked for: last 20K per channel.
    python .../tune_tail.py --strategy all-tail --tail-n 20000 --lr 1e-4 --epochs 2

    # Dry run: see which indices would be used, no training.
    python .../tune_tail.py --strategy neg-tail --tail-n 20000 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch

# ── Path bootstrap (mirror train.py) ─────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vectorq_v2_final.tune_tail")

# ── Constants (match collect.py / train.py) ──────────────────────────────────
TOKEN_DIM   = 50
WINDOW_SIZE = 75

TRANSFORMER_CONFIG = {
    "token_dim": TOKEN_DIM, "action_dim": 3, "window_size": WINDOW_SIZE,
    "d_model": 256, "n_heads": 4, "n_layers": 4, "ffn_dim": 1024,
    "policy_hidden": [256], "q_hidden": [256],
}
SAC_STATIC = {"gamma": 0.992, "tau": 0.005, "target_entropy": -2.0}

BUFFER_CAPACITY = 100_000

CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
BUFFER_DIR     = os.path.join(CHECKPOINT_DIR, "buffers")
OUTPUTS_ROOT   = os.path.join(_THIS_DIR, "outputs")

CHANNELS = [
    "steer_pos",    "steer_neg",
    "throttle_pos", "throttle_neg",
    "brake_pos",    "brake_neg",
]
NEG_CHANNELS = ["steer_neg", "throttle_neg", "brake_neg"]
POS_CHANNELS = ["steer_pos", "throttle_pos", "brake_pos"]

# Default AI-clone dataset (relative to repo root).
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DEFAULT_AICLONE_PATH = os.path.join(
    _REPO_ROOT, "AICLONE", "data", "monza_miata", "aiclone_dataset.npz"
)


# ─────────────────────────────────────────────────────────────────────────────
# AI-clone data loader: build 75-frame windows with reward=1.0.
# ─────────────────────────────────────────────────────────────────────────────

def _load_ai_windows(path: str, window_size: int, token_dim: int) -> dict:
    """
    Load the AI-clone dataset and emit (N, W, D)-shape windows with the same
    schema the replay buffer uses.  Reward is fixed at 1.0 on all 3 channels
    (signals 'this is a maximally-valuable transition').

    Expected keys (we try several common names for robustness):
      obs:  (T, token_dim) float32  — per-frame observation token
      acts: (T, 3)         float32  — per-frame action (steer, throttle, brake)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"AI-clone dataset not found: {path}")

    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    logger.info(f"AI dataset keys: {keys}")

    # Resolve obs array (first match wins).
    obs_key = next((k for k in ("obs", "observations", "states", "tokens") if k in keys), None)
    if obs_key is None:
        raise ValueError(f"AI dataset has no obs key; found {keys}")
    obs_full = np.asarray(data[obs_key], dtype=np.float32)
    if obs_full.ndim != 2:
        raise ValueError(f"Expected obs to be 2D (T, D); got {obs_full.shape}")
    # Some datasets store 125-dim full obs; trim to the first `token_dim`.
    if obs_full.shape[1] < token_dim:
        raise ValueError(f"obs has {obs_full.shape[1]} dims, need >= {token_dim}")
    obs_full = obs_full[:, :token_dim]

    # Resolve action array.
    act_key = next((k for k in ("acts", "actions", "action") if k in keys), None)
    if act_key is None:
        raise ValueError(f"AI dataset has no action key; found {keys}")
    acts_full = np.asarray(data[act_key], dtype=np.float32)
    if acts_full.ndim != 2 or acts_full.shape[1] != 3:
        raise ValueError(f"Expected acts to be (T, 3); got {acts_full.shape}")
    if acts_full.shape[0] != obs_full.shape[0]:
        raise ValueError(f"obs/acts length mismatch: {obs_full.shape[0]} vs {acts_full.shape[0]}")

    T = obs_full.shape[0]
    N = T - window_size  # each window needs a valid next_obs one frame ahead
    if N <= 0:
        raise ValueError(f"AI data too short for window_size={window_size}: T={T}")

    # Vectorised sliding-window build (O(N) memory, single copy).
    # obs_windows[i]      = obs_full[i : i+W]          , i ∈ [0, N)
    # next_obs_windows[i] = obs_full[i+1 : i+1+W]
    # action[i]           = acts_full[i + W - 1]       (action taken at the last frame)
    from numpy.lib.stride_tricks import sliding_window_view
    all_windows = sliding_window_view(obs_full, window_shape=(window_size, token_dim)).squeeze(1)
    obs_windows      = np.ascontiguousarray(all_windows[:N])
    next_obs_windows = np.ascontiguousarray(all_windows[1:N + 1])
    actions          = np.ascontiguousarray(acts_full[window_size - 1 : window_size - 1 + N])
    rewards          = np.full((N, 3), 1.0, dtype=np.float32)
    dones            = np.zeros((N, 1), dtype=np.float32)

    logger.info(
        f"AI windows built: N={N}  obs={obs_windows.shape}  act={actions.shape}  "
        f"reward=1.0 (all 3 ch)  RAM≈{(obs_windows.nbytes*2 + actions.nbytes + rewards.nbytes + dones.nbytes)/1e6:.1f} MB"
    )
    return {
        "obs_seq":      obs_windows,
        "next_obs_seq": next_obs_windows,
        "action":       actions,
        "reward":       rewards,
        "done":         dones,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Index selection: pick the "last N" entries from a circular memmap buffer.
# ─────────────────────────────────────────────────────────────────────────────

def _tail_indices(size: int, ptr: int, capacity: int, n: int) -> np.ndarray:
    """
    Return the last ``n`` FIFO indices into a MemmapCircularBuffer.

    Cases
    -----
    • size < capacity  : buffer has not wrapped. Data lives at [0, size).
                         Newest is at size - 1.  Last N = [max(0, size-n), size).
    • size == capacity : buffer has wrapped. ``ptr`` is the next write slot
                         (= oldest entry). Newest is at (ptr - 1) % capacity.
                         Last N = [(ptr - n) % cap, ptr) modulo cap.
    """
    n = int(min(n, max(size, 0)))
    if n == 0:
        return np.empty(0, dtype=np.int64)

    if size < capacity:
        start = size - n  # size - n >= 0 because n <= size
        return np.arange(start, size, dtype=np.int64)

    # Wrapped / full.
    start = (ptr - n) % capacity
    return (np.arange(start, start + n) % capacity).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# RAM-backed per-channel slice + stratified sampler.
# ─────────────────────────────────────────────────────────────────────────────

class _ChannelSlice:
    """
    Holds an in-RAM copy of a subset of one channel buffer, plus an optional
    "recent" subset for weighted (mix) sampling.
    """

    def __init__(
        self,
        buf,
        indices: np.ndarray,
        recent_indices: np.ndarray | None = None,
        extra: dict | None = None,
    ):
        self.name = buf.name
        self.n_buf = int(len(indices))
        self.indices = indices

        # Bulk copy memmap → RAM. np.array() eagerly materialises.
        obs      = np.array(buf._obs[indices], dtype=np.float32)
        next_obs = np.array(buf._next_obs[indices], dtype=np.float32)
        action   = np.array(buf._action[indices], dtype=np.float32)
        reward   = np.array(buf._reward[indices], dtype=np.float32)
        done     = np.array(buf._done[indices], dtype=np.float32)

        # Optionally fold in extra (e.g. AI-clone) windows. Extras are
        # concatenated AFTER the buffer rows, so buffer-relative indices
        # (used by strategy=mix's "recent" subset) remain valid.
        self.n_extra = 0
        if extra is not None:
            for key, arr, expected in (
                ("obs_seq",      extra["obs_seq"],      obs.shape[1:]),
                ("next_obs_seq", extra["next_obs_seq"], next_obs.shape[1:]),
                ("action",       extra["action"],       action.shape[1:]),
                ("reward",       extra["reward"],       reward.shape[1:]),
                ("done",         extra["done"],         done.shape[1:]),
            ):
                if arr.shape[1:] != expected:
                    raise ValueError(
                        f"{self.name}: extra['{key}'] shape {arr.shape} "
                        f"does not match buffer shape (*, {expected})"
                    )
            obs      = np.concatenate([obs,      extra["obs_seq"].astype(np.float32, copy=False)],      axis=0)
            next_obs = np.concatenate([next_obs, extra["next_obs_seq"].astype(np.float32, copy=False)], axis=0)
            action   = np.concatenate([action,   extra["action"].astype(np.float32, copy=False)],       axis=0)
            reward   = np.concatenate([reward,   extra["reward"].astype(np.float32, copy=False)],       axis=0)
            done     = np.concatenate([done,     extra["done"].astype(np.float32, copy=False)],         axis=0)
            self.n_extra = int(extra["obs_seq"].shape[0])

        self.n = self.n_buf + self.n_extra

        self._obs      = obs
        self._next_obs = next_obs
        self._action   = action
        self._reward   = reward
        self._done     = done

        # Precompute positions inside our RAM block for the "recent" subset
        # used by strategy=mix.  Only the FIRST n_buf rows came from the
        # buffer, so recent_local indexes into those; extras live at the
        # tail of the pool and are sampled uniformly alongside.
        if recent_indices is not None and len(recent_indices) > 0:
            pos = {int(v): i for i, v in enumerate(indices)}
            recent_local = np.fromiter(
                (pos[int(v)] for v in recent_indices if int(v) in pos),
                dtype=np.int64,
            )
            self._recent_local = recent_local
        else:
            self._recent_local = None

    def ram_gb(self) -> float:
        return (self._obs.nbytes + self._next_obs.nbytes
                + self._action.nbytes + self._reward.nbytes
                + self._done.nbytes) / 1e9

    def sample(self, n: int, recent_frac: float = 0.0) -> dict:
        """Uniform sample of ``n`` rows, optionally biased toward ``recent``."""
        if recent_frac > 0.0 and self._recent_local is not None and len(self._recent_local) > 0:
            n_recent = int(round(recent_frac * n))
            n_rest   = n - n_recent
            idx_r    = self._recent_local[
                np.random.randint(0, len(self._recent_local), size=n_recent)
            ]
            idx_b    = np.random.randint(0, self.n, size=n_rest)
            idx      = np.concatenate([idx_r, idx_b])
        else:
            idx = np.random.randint(0, self.n, size=n)

        return {
            "obs_seq":      self._obs[idx],
            "next_obs_seq": self._next_obs[idx],
            "action":       self._action[idx],
            "reward":       self._reward[idx],
            "done":         self._done[idx],
        }


class TailSampler:
    """
    Stratified sampler over the 6 channels. Each channel holds a fixed RAM
    snapshot of the indices chosen by the selected strategy.
    """

    def __init__(
        self,
        replay_buffer,
        channel_indices: dict,
        channel_recent_indices: dict | None,
        batch_size: int,
        device: str,
        recent_frac: float = 0.0,
        channel_extras: dict | None = None,
    ):
        self.batch_size  = batch_size
        self.device      = device
        self.recent_frac = recent_frac

        self.channels: dict[str, _ChannelSlice] = {}
        total_ram = 0.0
        for name, idx in channel_indices.items():
            if len(idx) == 0:
                logger.warning(f"  channel {name}: EMPTY selection — skipping.")
                continue
            recent = None if channel_recent_indices is None else channel_recent_indices.get(name)
            extra  = None if channel_extras         is None else channel_extras.get(name)
            slc = _ChannelSlice(replay_buffer.buffers[name], idx, recent, extra=extra)
            self.channels[name] = slc
            total_ram += slc.ram_gb()
            recent_msg = (f" recent_subset={len(slc._recent_local) if slc._recent_local is not None else 0}"
                          if slc._recent_local is not None else "")
            extra_msg = f" +extra={slc.n_extra}" if slc.n_extra > 0 else ""
            logger.info(
                f"  channel {name}: {slc.n} windows in RAM (buf={slc.n_buf}{extra_msg}) "
                f"({slc.ram_gb():.2f} GB){recent_msg}"
            )

        self.n_channels = len(self.channels)
        if self.n_channels == 0:
            raise RuntimeError("TailSampler: no non-empty channels selected.")

        logger.info(
            f"TailSampler ready — {self.n_channels} channels  "
            f"{total_ram:.2f} GB RAM  recent_frac={recent_frac:.2f}"
        )

    def sample(self) -> dict:
        per_ch = self.batch_size // self.n_channels
        remainder = self.batch_size - per_ch * self.n_channels

        parts = []
        for i, (_, slc) in enumerate(self.channels.items()):
            n = per_ch + (1 if i < remainder else 0)
            if n > 0:
                parts.append(slc.sample(n, recent_frac=self.recent_frac))

        merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
        perm   = np.random.permutation(merged["obs_seq"].shape[0])
        for k in merged:
            merged[k] = merged[k][perm]

        return {
            k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
            for k, v in merged.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint backup.
# ─────────────────────────────────────────────────────────────────────────────

def _backup_checkpoint(ckpt_dir: str, tag: str) -> str | None:
    src = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.isfile(src):
        logger.warning(f"No latest.pt at {src} — nothing to back up.")
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_dir = os.path.join(ckpt_dir, f"{ts}_{tag}")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "latest.pt")
    shutil.copy2(src, dst)
    logger.info(f"Checkpoint backup: {dst}")
    return dst_dir


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate override (after sac.load() restored optimizer state dicts).
# ─────────────────────────────────────────────────────────────────────────────

def _override_lr(optimizer, lr: float, name: str) -> None:
    old = [g["lr"] for g in optimizer.param_groups]
    for g in optimizer.param_groups:
        g["lr"] = lr
    logger.info(f"  {name}: lr {old} → {lr:.1e}")


# ─────────────────────────────────────────────────────────────────────────────
# Device resolution.
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


# ─────────────────────────────────────────────────────────────────────────────
# Main.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vector-Q v2 Final — fine-tune on tail / mixed data with lower LR."
    )
    parser.add_argument(
        "--strategy", choices=["all-tail", "neg-tail", "mix"],
        default="neg-tail",
        help="all-tail: last N from all 6 channels. "
             "neg-tail: last N from NEG channels, full buffer from POS. "
             "mix: full buffer for all, but each batch over-samples recent N by --recent-frac.",
    )
    parser.add_argument("--tail-n",      type=int,   default=20_000,
                        help="How many most-recent windows define the 'tail'.")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Learning rate for fine-tuning (orig was 3e-4).")
    parser.add_argument("--epochs",      type=int,   default=2,
                        help="Passes over the selected windows.")
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--recent-frac", type=float, default=0.5,
                        help="(mix only) fraction of each batch drawn from the recent tail.")
    parser.add_argument("--device",      default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-backup",   action="store_true",
                        help="Skip auto-backup of the current checkpoint.")
    parser.add_argument("--save-tag",    default="post_tune_tail",
                        help="Filename tag for the extra saved copy.")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show which indices would be used, but do not train.")
    parser.add_argument("--inject-ai-data", action="store_true",
                        help="Mix AI-clone windows (reward=1.0) into the 3 POS channel pools. "
                             "Buffer is NOT modified; injection is in-RAM only.")
    parser.add_argument("--aiclone-path", default=DEFAULT_AICLONE_PATH,
                        help=f"Path to aiclone_dataset.npz (default: {DEFAULT_AICLONE_PATH}).")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    logger.info("=" * 64)
    logger.info(f"tune_tail — strategy={args.strategy}  tail_n={args.tail_n}  "
                f"lr={args.lr}  epochs={args.epochs}  bs={args.batch_size}  device={device}"
                + (f"  inject_ai_data=True" if args.inject_ai_data else ""))
    logger.info("=" * 64)

    # 1) Backup first (safe-by-default).
    if not args.no_backup and not args.dry_run:
        _backup_checkpoint(CHECKPOINT_DIR, "pre_tune_tail")

    # 2) Open the buffer read-only-by-convention (we never .push here).
    from transformer_sac_vectorq_v2_final_fineTune.replay_buffer import SixChannelMemmapBuffer

    replay_buffer = SixChannelMemmapBuffer(
        base_dir=os.path.abspath(BUFFER_DIR),
        capacity_per_buffer=BUFFER_CAPACITY,
        token_dim=TOKEN_DIM, action_dim=3,
        window_size=WINDOW_SIZE, reward_dim=3,
    )
    sizes = replay_buffer.sizes()
    logger.info(f"Buffer sizes: {sizes}  total={sum(sizes.values())}")

    # 3) Choose indices per channel.
    channel_indices: dict = {}
    channel_recent:  dict = {}   # only used for strategy=mix
    for name in CHANNELS:
        buf = replay_buffer.buffers[name]
        size, ptr, cap = buf._size, buf._ptr, buf.capacity

        if args.strategy == "all-tail":
            idx = _tail_indices(size, ptr, cap, args.tail_n)
            recent = None
        elif args.strategy == "neg-tail":
            if name in NEG_CHANNELS:
                idx = _tail_indices(size, ptr, cap, args.tail_n)
            else:
                idx = np.arange(size, dtype=np.int64)
            recent = None
        elif args.strategy == "mix":
            idx    = np.arange(size, dtype=np.int64)  # full filled portion
            recent = _tail_indices(size, ptr, cap, args.tail_n)
        else:
            raise ValueError(args.strategy)

        channel_indices[name] = idx
        if recent is not None:
            channel_recent[name] = recent

        tag = ""
        if args.strategy == "mix":
            tag = f"  recent_tail={len(recent)}"
        logger.info(
            f"  {name:<13}  size={size:>6} ptr={ptr:>6}  "
            f"selected={len(idx):>6}{tag}"
        )

    # 3b) Optionally load AI-clone windows and route them to POS channels.
    channel_extras: dict | None = None
    ai_window_count = 0
    if args.inject_ai_data:
        logger.info(f"Injecting AI-clone data (reward=1.0) from: {args.aiclone_path}")
        ai = _load_ai_windows(args.aiclone_path, WINDOW_SIZE, TOKEN_DIM)
        ai_window_count = int(ai["obs_seq"].shape[0])
        # Each of the 3 POS channels sees the SAME AI windows. This means
        # Q_steer, Q_throttle, Q_brake all get gradient signal from the AI
        # action/reward, because the stratified batch picks batch/6 per channel.
        channel_extras = {name: ai for name in POS_CHANNELS}
        for ch in POS_CHANNELS:
            logger.info(f"  POS channel {ch}: +{ai_window_count} AI windows will be mixed in.")

    if args.dry_run:
        logger.info("Dry run — exiting before sampler / model build.")
        return

    # 4) Build RAM sampler.
    sampler = TailSampler(
        replay_buffer,
        channel_indices,
        channel_recent if args.strategy == "mix" else None,
        batch_size=args.batch_size,
        device=device,
        recent_frac=args.recent_frac if args.strategy == "mix" else 0.0,
        channel_extras=channel_extras,
    )

    # 5) Build SAC at the new LR, then load weights + (old-LR) optimizer states.
    from transformer_sac_vectorq_v2_final_fineTune.sac import TransformerSAC

    sac = TransformerSAC(
        **TRANSFORMER_CONFIG,
        lr=args.lr,
        **SAC_STATIC,
        device=device,
    )
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    assert os.path.isfile(latest), f"No checkpoint at {latest}"
    sac.load(latest)

    # sac.load() restores the q/policy optimizer state dicts (carrying the
    # original 3e-4 lr baked into param_groups). Override them here.
    # alpha_optimizer was rebuilt fresh at self.lr (= args.lr) inside load().
    _override_lr(sac.q_optimizer,      args.lr, "q_optimizer")
    _override_lr(sac.policy_optimizer, args.lr, "policy_optimizer")
    _override_lr(sac.alpha_optimizer,  args.lr, "alpha_optimizer")

    # 6) Training loop.
    total_windows = sum(slc.n for slc in sampler.channels.values())
    steps_per_epoch = max(1, total_windows // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    logger.info(
        f"Training plan: {total_steps} gradient steps "
        f"({steps_per_epoch} steps/epoch × {args.epochs} epochs)"
    )

    history = {
        "q_loss":             [],
        "q_loss_per_channel": [],
        "pi_loss":            [],
        "alpha":              [],
        "alpha_per_action":   [],
        "entropy":            [],
        "entropy_per_action": [],
    }

    t_start = time.perf_counter()
    try:
        for step in range(1, total_steps + 1):
            batch  = sampler.sample()
            losses = sac.update_from_batch(batch)

            history["q_loss"].append(losses["q_loss"])
            history["q_loss_per_channel"].append(
                losses.get("q_loss_per_channel", [0, 0, 0])
            )
            history["pi_loss"].append(losses["policy_loss"])
            history["alpha"].append(losses["alpha"])
            history["alpha_per_action"].append(losses["alpha_per_action"])
            history["entropy"].append(losses["entropy"])
            history["entropy_per_action"].append(losses["entropy_per_action"])

            if step % args.log_interval == 0 or step == total_steps:
                recent = slice(-args.log_interval, None)
                qpc = np.mean(history["q_loss_per_channel"][recent], axis=0)
                epa = np.mean(history["entropy_per_action"][recent], axis=0)
                apa = np.mean(history["alpha_per_action"][recent], axis=0)
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f"step={step}/{total_steps}  "
                    f"q_loss={float(np.mean(history['q_loss'][recent])):.4f}  "
                    f"Qs=[{qpc[0]:.4f},{qpc[1]:.4f},{qpc[2]:.4f}]  "
                    f"pi={float(np.mean(history['pi_loss'][recent])):.4f}  "
                    f"H=[{epa[0]:.3f},{epa[1]:.3f},{epa[2]:.3f}]  "
                    f"A=[{apa[0]:.4f},{apa[1]:.4f},{apa[2]:.4f}]  "
                    f"elapsed={elapsed:.1f}s"
                )
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving partial progress.")

    elapsed = time.perf_counter() - t_start
    logger.info(f"Done — {len(history['q_loss'])} steps in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # 7) Save.  Overwrite latest.pt AND keep a tagged copy.
    sac.save(latest)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tagged = os.path.join(CHECKPOINT_DIR, f"{ts}_{args.save_tag}.pt")
    shutil.copy2(latest, tagged)
    logger.info(f"Saved: {latest}  and  {tagged}")

    # 8) Plot + write history.
    out_dir = os.path.join(OUTPUTS_ROOT, f"tune_tail_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    try:
        from transformer_sac_vectorq_v2_final_fineTune.train import _plot_training_curves
        _plot_training_curves(history, out_dir)
        logger.info(f"Plots written to {out_dir}")
    except Exception as exc:
        logger.warning(f"_plot_training_curves failed: {exc}")

    meta = {
        "strategy":    args.strategy,
        "tail_n":      args.tail_n,
        "lr":          args.lr,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "recent_frac": args.recent_frac,
        "total_steps": total_steps,
        "buffer_sizes": sizes,
        "selected_per_channel": {k: int(len(v)) for k, v in channel_indices.items()},
        "inject_ai_data":   bool(args.inject_ai_data),
        "aiclone_path":     args.aiclone_path if args.inject_ai_data else None,
        "ai_windows_count": ai_window_count,
        "ai_reward":        1.0 if args.inject_ai_data else None,
        "pool_per_channel": {name: int(slc.n) for name, slc in sampler.channels.items()},
        "elapsed_sec": elapsed,
        "timestamp":   ts,
        "saved_latest": latest,
        "saved_tagged": tagged,
    }
    with open(os.path.join(out_dir, "tune_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save raw history (for later comparison / ablation).
    hist_serialisable = {
        k: (v if not isinstance(v[0], (list, tuple)) else [list(x) for x in v])
        for k, v in history.items() if len(v) > 0
    }
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist_serialisable, f)

    logger.info(f"Run complete — artefacts: {out_dir}")


if __name__ == "__main__":
    main()
