"""
pretrain_actor.py — BC pre-training of the SAC GaussianPolicy actor.

Reads the .npz produced by preprocess_parquet.py and trains the SAC
GaussianPolicy directly using supervised MSE loss on the mean action.
No Q-networks — actor only.

The saved checkpoint format is:
    {"policy": state_dict}

This is intentionally a subset of the full SAC checkpoint.  When loading
into SAC fine-tuning, pass --actor-pretrained to finetune_sac.py which
calls sac.policy.load_state_dict(checkpoint["policy"]).

Usage
-----
    # From scratch
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/pretrain_actor.py

    # Warm-start from 150_jitter (recommended — preserves RL knowledge)
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/pretrain_actor.py \\
        --init-from-sac AICLONE/checkpoints/150_jitter_base.pt \\
        --out-dir AICLONE/checkpoints/150_bc \\
        --epochs 20 --lr 1e-4

    # Full options
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/pretrain_actor.py \\
        --data     AICLONE/data/aiclone_dataset.npz \\
        --out-dir  AICLONE/checkpoints \\
        --epochs   50 --batch-size 256 --lr 3e-4

Output
------
    AICLONE/checkpoints/best.pt    — lowest val loss
    AICLONE/checkpoints/final.pt   — after last epoch
    AICLONE/checkpoints/log.csv    — per-epoch train/val losses
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_REPO    = Path(__file__).resolve().parent.parent
_GYM_DIR = _REPO / "gym"
for _p in [str(_GYM_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sac.network import GaussianPolicy   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OBS_DIM    = 125
ACTION_DIM = 3
HIDDEN     = [256, 256, 256]   # must match SAC hyperparameters


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_split(npz_path: str, val_fraction: float, seed: int):
    """
    Episode-level train/val split — same logic as train_bc.py.
    Prevents steps from the same lap appearing in both splits.
    """
    data = np.load(npz_path)
    obs          = data["obs"]           # (N, 125) float32
    actions      = data["actions"]       # (N, 3)   float32
    episode_ends = data["episode_ends"]  # (E,)     int64

    E   = len(episode_ends)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)
    n_val        = int(E * val_fraction) if val_fraction > 0 else 0
    val_ep_idx   = set(perm[:n_val].tolist())
    train_ep_idx = set(perm[n_val:].tolist())

    N           = len(obs)
    train_mask  = np.zeros(N, dtype=bool)
    val_mask    = np.zeros(N, dtype=bool)
    prev_end    = -1
    for i, end in enumerate(episode_ends):
        start = prev_end + 1
        if i in train_ep_idx:
            train_mask[start:end + 1] = True
        else:
            val_mask[start:end + 1] = True
        prev_end = end

    logger.info(
        f"Split: {train_mask.sum()} train steps ({E - n_val} eps) / "
        f"{val_mask.sum()} val steps ({n_val} eps)"
    )

    def make_ds(mask):
        return TensorDataset(
            torch.from_numpy(obs[mask]),
            torch.from_numpy(actions[mask]),
        )

    return make_ds(train_mask), make_ds(val_mask)


# ── Training ──────────────────────────────────────────────────────────────────

def train(policy, train_loader, val_loader, epochs, lr, out_dir, device):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = out_dir / "best.pt"
    final_ckpt = out_dir / "final.pt"
    csv_path   = out_dir / "log.csv"

    best_val = float("inf")

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "epoch",
            "train_loss", "train_steer", "train_pedal", "train_brake",
            "val_loss",   "val_steer",   "val_pedal",   "val_brake",
        ])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            # ── Train ────────────────────────────────────────────────────
            policy.train()
            tr_sum, tr_n = 0.0, 0
            tr_ch = [0.0, 0.0, 0.0]   # steer, pedal, brake

            for obs_b, act_b in train_loader:
                obs_b = obs_b.to(device)
                act_b = act_b.to(device)
                optimizer.zero_grad()
                _, _, mean = policy(obs_b)   # mean = tanh(mu) in [-1,1]
                loss = criterion(mean, act_b)
                loss.backward()
                optimizer.step()
                bs = obs_b.size(0)
                tr_sum += loss.item() * bs
                tr_n   += bs
                with torch.no_grad():
                    diff_sq = (mean - act_b) ** 2
                    for i in range(3):
                        tr_ch[i] += diff_sq[:, i].mean().item() * bs

            tr     = tr_sum / tr_n
            tr_ch  = [v / tr_n for v in tr_ch]

            # ── Val (skip if no val set) ──────────────────────────────────
            policy.eval()
            if len(val_loader.dataset) > 0:
                vl_sum, vl_n = 0.0, 0
                vl_ch = [0.0, 0.0, 0.0]

                with torch.no_grad():
                    for obs_b, act_b in val_loader:
                        obs_b = obs_b.to(device)
                        act_b = act_b.to(device)
                        _, _, mean = policy(obs_b)
                        loss = criterion(mean, act_b)
                        bs = obs_b.size(0)
                        vl_sum += loss.item() * bs
                        vl_n   += bs
                        diff_sq = (mean - act_b) ** 2
                        for i in range(3):
                            vl_ch[i] += diff_sq[:, i].mean().item() * bs

                vl    = vl_sum / vl_n
                vl_ch = [v / vl_n for v in vl_ch]
            else:
                vl, vl_ch = tr, tr_ch  # no val — use train loss for checkpointing

            is_best = vl < best_val
            if is_best:
                best_val = vl
                torch.save({"policy": policy.state_dict()}, best_ckpt)

            logger.info(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={tr:.6f} (s={tr_ch[0]:.4f} p={tr_ch[1]:.4f} b={tr_ch[2]:.4f})  "
                f"val={vl:.6f}   (s={vl_ch[0]:.4f} p={vl_ch[1]:.4f} b={vl_ch[2]:.4f})"
                + ("  *best*" if is_best else "")
            )

            writer.writerow({
                "epoch":       epoch,
                "train_loss":  round(tr,       6),
                "train_steer": round(tr_ch[0], 6),
                "train_pedal": round(tr_ch[1], 6),
                "train_brake": round(tr_ch[2], 6),
                "val_loss":    round(vl,       6),
                "val_steer":   round(vl_ch[0], 6),
                "val_pedal":   round(vl_ch[1], 6),
                "val_brake":   round(vl_ch[2], 6),
            })
            csv_file.flush()

    torch.save({"policy": policy.state_dict()}, final_ckpt)
    logger.info(f"best val loss : {best_val:.6f}")
    logger.info(f"best.pt       : {best_ckpt}")
    logger.info(f"final.pt      : {final_ckpt}")
    logger.info(f"log.csv       : {csv_path}")
    logger.info(
        "To load into SAC fine-tuning:\n"
        "    ckpt = torch.load('AICLONE/checkpoints/best.pt')\n"
        "    sac.policy.load_state_dict(ckpt['policy'])"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BC pre-train SAC GaussianPolicy actor")
    parser.add_argument("--data",       default="AICLONE/data/aiclone_dataset.npz")
    parser.add_argument("--out-dir",    default="AICLONE/checkpoints")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--val-frac",   type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--init-from-sac",
        default=None,
        metavar="SAC_CKPT",
        help="Warm-start actor from a SAC checkpoint (e.g. AICLONE/checkpoints/150_jitter_base.pt). "
             "Loads policy weights only — critics are ignored.",
    )
    args = parser.parse_args()

    device_str = (
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    npz_path = _REPO / args.data
    out_dir  = _REPO / args.out_dir

    # Print dataset stats
    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        logger.info(
            f"Dataset: {meta['n_steps']} steps  {meta['n_episodes']} episodes  "
            f"tracks={meta['tracks']}"
        )

    train_ds, val_ds = load_and_split(str(npz_path), args.val_frac, args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device_str == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device_str == "cuda"),
    )

    policy = GaussianPolicy(
        state_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_units=HIDDEN,
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"GaussianPolicy: obs={OBS_DIM}  action={ACTION_DIM}  params={n_params:,}")

    # ── Warm-start from SAC checkpoint (actor only) ───────────────────────────
    if args.init_from_sac:
        sac_ckpt_path = str((_REPO / args.init_from_sac).resolve())
        ckpt = torch.load(sac_ckpt_path, map_location=device)
        # SAC checkpoints store actor under "policy" key
        if "policy" in ckpt:
            policy.load_state_dict(ckpt["policy"])
            logger.info(f"Warm-started actor from SAC checkpoint (policy key): {sac_ckpt_path}")
        else:
            raise KeyError(
                f"No 'policy' key in {sac_ckpt_path}. "
                f"Available keys: {list(ckpt.keys())}"
            )

    train(policy, train_loader, val_loader,
          args.epochs, args.lr, out_dir, device)


if __name__ == "__main__":
    main()
