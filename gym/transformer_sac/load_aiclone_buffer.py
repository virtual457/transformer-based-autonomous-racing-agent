"""
load_aiclone_buffer.py — Seed the DualWindowReplayBuffer with AIClone demonstration data.

Loads AIClone/data/aiclone_dataset.npz, extracts 50-dim tokens (obs[:50]),
remaps actions from [0, 1] -> [-1, 1], assigns reward=0.9 to all transitions
(near-perfect demonstration reward), and pushes all episodes into the buffer.

All windows go into the positive sub-buffer (reward=0.9 > 0).
The negative sub-buffer stays empty — it fills naturally during RL collection.
The DualWindowReplayBuffer.sample() handles empty neg_buffer gracefully.

Run before starting training:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac/load_aiclone_buffer.py

The buffer is saved to checkpoints/sac_transformer_v1/buffer_{pos,neg}.npz
and will be auto-loaded by the agent on the next training run.
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
logger = logging.getLogger("load_aiclone_buffer")

# ── Config ─────────────────────────────────────────────────────────────────────
AICLONE_NPZ      = os.path.join(_THIS_DIR, "..", "..", "AIClone", "data", "aiclone_dataset.npz")
CHECKPOINT_DIR   = os.path.join(_THIS_DIR, "..", "..", "checkpoints", "sac_transformer_v1")
TOKEN_DIM        = 50
ACTION_DIM       = 3
WINDOW_SIZE      = 75
CAPACITY         = 50_000
DEMO_REWARD      = 0.9   # near-perfect reward — leaves headroom for RL to find reward=1.0


def remap_actions(actions_01: np.ndarray) -> np.ndarray:
    """
    Remap AIClone actions from [0, 1] to [-1, 1] (SAC action space).
    Formula: a_sac = a_clone * 2.0 - 1.0
    """
    return (actions_01 * 2.0 - 1.0).astype(np.float32)


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
    logger.info(f"Loaded {N} frames  {E} episodes  obs_dim={obs_all.shape[1]}  action_dim={actions_all.shape[1]}")

    # ── Extract tokens and remap actions ──────────────────────────────────────
    tokens_all  = obs_all[:, :TOKEN_DIM].astype(np.float32)   # (N, 50)
    actions_sac = remap_actions(actions_all)                   # (N, 3) in [-1, 1]

    logger.info(f"Token extraction: obs[:50]  shape={tokens_all.shape}")
    logger.info(
        f"Action remap [0,1]->[-1,1]:  "
        f"steer  mean={actions_sac[:,0].mean():.3f}  "
        f"throttle mean={actions_sac[:,1].mean():.3f}  "
        f"brake  mean={actions_sac[:,2].mean():.3f}"
    )

    # ── Build episode boundaries ───────────────────────────────────────────────
    starts = np.concatenate([[0], ep_ends[:-1] + 1])
    ends   = ep_ends   # inclusive last index of each episode

    # ── Build replay buffer ───────────────────────────────────────────────────
    from transformer_sac.replay_buffer import DualWindowReplayBuffer

    buffer = DualWindowReplayBuffer(
        capacity=CAPACITY,
        token_dim=TOKEN_DIM,
        action_dim=ACTION_DIM,
        window_size=WINDOW_SIZE,
    )

    total_windows = 0
    for ep_idx in range(E):
        s = starts[ep_idx]
        e = ends[ep_idx] + 1   # exclusive

        ep_tokens  = tokens_all[s:e]       # (T, 50)
        ep_actions = actions_sac[s:e]      # (T, 3)
        ep_rewards = np.full(e - s, DEMO_REWARD, dtype=np.float32)   # all 0.9
        ep_dones   = np.zeros(e - s, dtype=np.float32)
        ep_dones[-1] = 1.0   # mark last frame as terminal

        T = len(ep_tokens)
        n = buffer.push_episode(ep_tokens, ep_actions, ep_rewards, ep_dones)
        total_windows += n
        logger.info(
            f"Episode {ep_idx + 1}/{E}:  frames={T}  windows_added={n}  "
            f"buffer_total={total_windows}"
        )

    logger.info(
        f"\nBuffer loaded — total windows: {total_windows}"
        f"  pos={len(buffer.positive_buffer)}  neg={len(buffer.negative_buffer)}"
    )
    logger.info(f"All {total_windows} windows in positive buffer (reward={DEMO_REWARD})")
    logger.info("Negative buffer is empty — will fill during RL collection.")

    # ── Save buffer ───────────────────────────────────────────────────────────
    os.makedirs(os.path.abspath(CHECKPOINT_DIR), exist_ok=True)
    buf_base = os.path.join(os.path.abspath(CHECKPOINT_DIR), "buffer")
    buffer.save(buf_base + ".npz")
    logger.info(f"Buffer saved to {buf_base}_{{pos,neg}}.npz")

    # ── Sanity check — sample one batch ───────────────────────────────────────
    batch = buffer.sample(batch_size=256, device="cpu")
    logger.info(
        f"\nSanity check — sampled batch of 256:"
        f"\n  obs_seq      shape={batch['obs_seq'].shape}"
        f"\n  action       shape={batch['action'].shape}"
        f"\n  reward       min={batch['reward'].min():.3f}  max={batch['reward'].max():.3f}"
        f"\n  next_obs_seq shape={batch['next_obs_seq'].shape}"
        f"\n  steer        mean={batch['action'][:,0].mean():.3f}  std={batch['action'][:,0].std():.3f}"
        f"\n  throttle     mean={batch['action'][:,1].mean():.3f}  std={batch['action'][:,1].std():.3f}"
        f"\n  brake        mean={batch['action'][:,2].mean():.3f}  std={batch['action'][:,2].std():.3f}"
    )


if __name__ == "__main__":
    main()
