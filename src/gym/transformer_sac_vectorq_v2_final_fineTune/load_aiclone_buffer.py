"""
load_aiclone_buffer.py — Seed the 6-channel memmap buffer with AIClone data.

AIClone demonstrations get uniform [0.9, 0.9, 0.9] reward per frame,
so all 3 channels route to the positive side.

Run to populate the buffer with clean AIClone data:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2_final_fineTune/load_aiclone_buffer.py
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
logger = logging.getLogger("load_aiclone_buffer_v2")

# ── Config ─────────────────────────────────────────────────────────────────────
AICLONE_NPZ    = os.path.join(_THIS_DIR, "..", "..", "AICLONE", "data", "monza_miata", "aiclone_dataset.npz")
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
BUFFER_DIR     = os.path.join(CHECKPOINT_DIR, "buffers")
TOKEN_DIM      = 50
ACTION_DIM     = 3
REWARD_DIM     = 3
WINDOW_SIZE    = 75
CAPACITY       = 100_000
DEMO_REWARD    = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def remap_actions(actions_sac: np.ndarray) -> np.ndarray:
    return np.asarray(actions_sac, dtype=np.float32)


def main():
    npz_path = os.path.abspath(AICLONE_NPZ)
    logger.info(f"Loading AIClone dataset: {npz_path}")
    d = np.load(npz_path)

    obs_all     = d["obs"]
    actions_all = d["actions"]
    ep_ends     = d["episode_ends"]

    N = len(obs_all)
    E = len(ep_ends)
    logger.info(f"Loaded {N} frames  {E} episodes  obs_dim={obs_all.shape[1]}")

    tokens_all  = obs_all[:, :TOKEN_DIM].astype(np.float32)
    actions_sac = remap_actions(actions_all)

    starts = np.concatenate([[0], ep_ends[:-1] + 1])
    ends   = ep_ends

    # ── Build 6-channel memmap buffer ──────────────────────────────────────────
    from transformer_sac_vectorq_v2_final_fineTune.replay_buffer import SixChannelMemmapBuffer

    buffer_dir = os.path.abspath(BUFFER_DIR)
    logger.info(f"Creating 6-channel memmap buffer at: {buffer_dir}")
    buffer = SixChannelMemmapBuffer(
        base_dir=buffer_dir,
        capacity_per_buffer=CAPACITY,
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
        ep_rewards = np.tile(DEMO_REWARD, (e - s, 1))
        ep_dones   = np.zeros(e - s, dtype=np.float32)
        n = buffer.push_episode(ep_tokens, ep_actions, ep_rewards, ep_dones)
        total_windows += n
        logger.info(
            f"Episode {ep_idx + 1}/{E}:  frames={e - s}  windows_added={n}  "
            f"total_windows={total_windows}"
        )

    # Flush to disk.
    buffer.flush()
    sizes = buffer.sizes()
    logger.info(f"\nBuffer loaded and flushed — sizes: {sizes}")
    logger.info(f"Total entries across 6 buffers: {sum(sizes.values())}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    batch = buffer.sample(batch_size=256, device="cpu")
    r = batch['reward']
    logger.info(
        f"Sanity check — reward shape={list(r.shape)}  "
        f"r_steer={r[:,0].mean():.3f}  r_throttle={r[:,1].mean():.3f}  r_brake={r[:,2].mean():.3f}  "
        f"steer_action mean={batch['action'][:,0].mean():.3f}"
    )


if __name__ == "__main__":
    main()
