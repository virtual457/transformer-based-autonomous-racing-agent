"""
load_aiclone_buffer.py — Seed the Vector Q DualWindowReplayBuffer with AIClone data.

Vector Q variant: rewards are stored as (reward_dim=3,) vectors.
AIClone demonstrations get uniform [0.9, 0.9, 0.9] reward per frame.

Run to reset the positive buffer back to clean AIClone data:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq/load_aiclone_buffer.py
"""

import os
import sys
import logging
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("load_aiclone_buffer_finetune")

# ── Config ─────────────────────────────────────────────────────────────────────
AICLONE_NPZ    = os.path.join(_THIS_DIR, "..", "..", "AIClone", "data", "aiclone_dataset.npz")
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
TOKEN_DIM      = 50
ACTION_DIM     = 3
REWARD_DIM     = 3
WINDOW_SIZE    = 75
CAPACITY       = 50_000
DEMO_REWARD    = np.array([0.9, 0.9, 0.9], dtype=np.float32)  # uniform vector reward


def remap_actions(actions_sac: np.ndarray) -> np.ndarray:
    """Actions are already in [-1, 1] SAC space — just ensure float32."""
    return np.asarray(actions_sac, dtype=np.float32)


def main():
    # ── Load aiclone data ─────────────────────────────────────────────────────
    npz_path = os.path.abspath(AICLONE_NPZ)
    logger.info(f"Loading AIClone dataset: {npz_path}")
    d = np.load(npz_path)

    obs_all     = d["obs"]            # (N, 125)
    actions_all = d["actions"]        # (N, 3) in [0, 1]
    ep_ends     = d["episode_ends"]   # (E,) last index of each episode

    N = len(obs_all)
    E = len(ep_ends)
    logger.info(f"Loaded {N} frames  {E} episodes  obs_dim={obs_all.shape[1]}")

    tokens_all  = obs_all[:, :TOKEN_DIM].astype(np.float32)
    actions_sac = remap_actions(actions_all)

    starts = np.concatenate([[0], ep_ends[:-1] + 1])
    ends   = ep_ends

    # ── Build buffer ──────────────────────────────────────────────────────────
    from transformer_sac_vectorq.replay_buffer import DualWindowReplayBuffer

    buffer = DualWindowReplayBuffer(
        capacity=CAPACITY,
        token_dim=TOKEN_DIM,
        action_dim=ACTION_DIM,
        window_size=WINDOW_SIZE,
        reward_dim=REWARD_DIM,
    )

    total_windows = 0
    for ep_idx in range(E):
        s = starts[ep_idx]
        e = ends[ep_idx] + 1
        ep_tokens  = tokens_all[s:e]
        ep_actions = actions_sac[s:e]
        ep_rewards = np.tile(DEMO_REWARD, (e - s, 1))  # (T, 3) vector rewards
        ep_dones   = np.zeros(e - s, dtype=np.float32)
        ep_dones[-1] = 1.0
        n = buffer.push_episode(ep_tokens, ep_actions, ep_rewards, ep_dones)
        total_windows += n
        logger.info(
            f"Episode {ep_idx + 1}/{E}:  frames={e - s}  windows_added={n}  "
            f"buffer_total={total_windows}"
        )

    logger.info(f"\nPositive buffer loaded — {len(buffer.positive_buffer)} windows")

    # ── Save positive buffer only — negative buffer left untouched ───────────
    os.makedirs(os.path.abspath(CHECKPOINT_DIR), exist_ok=True)
    pos_path = os.path.join(os.path.abspath(CHECKPOINT_DIR), "buffer_pos.npz")
    stem = pos_path[:-4]
    tmp_path = stem + "_tmp.npz"
    np.savez_compressed(
        stem + "_tmp",
        obs_seq      = buffer.positive_buffer._obs_seq[:len(buffer.positive_buffer)],
        next_obs_seq = buffer.positive_buffer._next_obs_seq[:len(buffer.positive_buffer)],
        action       = buffer.positive_buffer._action[:len(buffer.positive_buffer)],
        reward       = buffer.positive_buffer._reward[:len(buffer.positive_buffer)],
        done         = buffer.positive_buffer._done[:len(buffer.positive_buffer)],
        _ptr         = np.array(buffer.positive_buffer._ptr,  dtype=np.int64),
        _size        = np.array(buffer.positive_buffer._size, dtype=np.int64),
    )
    os.replace(tmp_path, pos_path)
    logger.info(f"Saved positive buffer to {pos_path}  (negative buffer untouched)")

    # ── Sanity check ──────────────────────────────────────────────────────────
    batch = buffer.sample(batch_size=256, device="cpu")
    r = batch['reward']   # (256, 3)
    logger.info(
        f"Sanity check — reward shape={list(r.shape)}  "
        f"r_steer={r[:,0].mean():.3f}  r_throttle={r[:,1].mean():.3f}  r_brake={r[:,2].mean():.3f}  "
        f"steer_action mean={batch['action'][:,0].mean():.3f}"
    )


if __name__ == "__main__":
    main()
