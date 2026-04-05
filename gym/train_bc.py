"""
train_bc.py — Behaviour Cloning trainer.

Reads a preprocessed .npz file (produced by preprocess_bc.py) and trains
an MLPActor with supervised MSE loss against human demonstration actions.

Prerequisites
-------------
    Run preprocess_bc.py first to build the .npz from raw pkl files.

Usage
-----
    .\\AssetoCorsa\\Scripts\\python.exe gym/train_bc.py \\
        --data data/processed/monza_ks_mazda_miata_mlp.npz \\
        --output-dir checkpoints/bc_monza

    Options:
        --epochs 30 --batch-size 256 --lr 3e-4
        --val-fraction 0.2  (episode-level split)
        --device auto|cuda|cpu
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
GYM_DIR   = REPO_ROOT / "gym"
DISCOR    = REPO_ROOT / "assetto_corsa_gym" / "algorithm" / "discor"

for p in [str(GYM_DIR), str(DISCOR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from policies.models import MLPActor   # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset — episode-level train/val split from .npz
# ---------------------------------------------------------------------------

def load_and_split(npz_path: str, val_fraction: float, seed: int):
    """
    Load obs/actions from .npz and do an episode-level train/val split.

    Episode boundaries come from `episode_ends` — the index of the last step
    in each episode. This prevents steps from the same episode appearing in
    both train and val (which would leak autocorrelated data).

    Returns
    -------
    train_ds, val_ds : TensorDataset
    obs_dim : int
    """
    data = np.load(npz_path)
    obs     = data["obs"]       # (N, obs_dim)
    actions = data["actions"]   # (N, 3)
    episode_ends = data["episode_ends"]   # (E,)

    E = len(episode_ends)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)
    n_val = max(1, int(E * val_fraction))
    val_ep_idx  = set(perm[:n_val].tolist())
    train_ep_idx = set(perm[n_val:].tolist())

    # Build step-level masks from episode boundaries
    N = len(obs)
    train_mask = np.zeros(N, dtype=bool)
    val_mask   = np.zeros(N, dtype=bool)

    prev_end = -1
    for i, end in enumerate(episode_ends):
        start = prev_end + 1
        if i in train_ep_idx:
            train_mask[start:end + 1] = True
        else:
            val_mask[start:end + 1] = True
        prev_end = end

    logger.info(
        f"Split: {train_mask.sum()} train steps ({sum(i in train_ep_idx for i in range(E))} eps) / "
        f"{val_mask.sum()} val steps ({n_val} eps)"
    )

    def make_ds(mask):
        obs_t = torch.from_numpy(obs[mask])
        act_t = torch.from_numpy(actions[mask])
        return TensorDataset(obs_t, act_t)

    return make_ds(train_mask), make_ds(val_mask), int(obs.shape[1])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _channel_losses(means: torch.Tensor, targets: torch.Tensor) -> dict:
    """Per-channel MSE — returns dict with steer/pedal/brake losses."""
    with torch.no_grad():
        diff_sq = (means - targets) ** 2
        return {
            "steer": diff_sq[:, 0].mean().item(),
            "pedal": diff_sq[:, 1].mean().item(),
            "brake": diff_sq[:, 2].mean().item(),
        }


def train(model, train_loader, val_loader, epochs, lr, output_dir, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs(output_dir, exist_ok=True)
    best_ckpt  = os.path.join(output_dir, "best.pt")
    final_ckpt = os.path.join(output_dir, "final.pt")
    best_val   = float("inf")

    # --- CSV log ---
    csv_path = os.path.join(output_dir, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_cols  = [
        "epoch",
        "train_loss", "train_steer", "train_pedal", "train_brake",
        "val_loss",   "val_steer",   "val_pedal",   "val_brake",
        "best_val",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=csv_cols)
    writer.writeheader()

    # --- TensorBoard ---
    tb_dir = os.path.join(output_dir, "tensorboard")
    tb = SummaryWriter(log_dir=tb_dir)
    logger.info(f"TensorBoard logs : {tb_dir}")
    logger.info(f"CSV log          : {csv_path}")
    logger.info(f"  run:  tensorboard --logdir {tb_dir}")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        tr_loss, n_tr = 0.0, 0
        tr_ch = {"steer": 0.0, "pedal": 0.0, "brake": 0.0}
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            optimizer.zero_grad()
            _, _, means = model(obs_b)
            loss = criterion(means, act_b)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * obs_b.size(0)
            n_tr    += obs_b.size(0)
            for k, v in _channel_losses(means, act_b).items():
                tr_ch[k] += v * obs_b.size(0)

        tr = tr_loss / n_tr
        tr_ch = {k: v / n_tr for k, v in tr_ch.items()}

        # --- Val ---
        model.eval()
        vl_loss, n_vl = 0.0, 0
        vl_ch = {"steer": 0.0, "pedal": 0.0, "brake": 0.0}
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                obs_b, act_b = obs_b.to(device), act_b.to(device)
                _, _, means = model(obs_b)
                loss = criterion(means, act_b)
                vl_loss += loss.item() * obs_b.size(0)
                n_vl    += obs_b.size(0)
                for k, v in _channel_losses(means, act_b).items():
                    vl_ch[k] += v * obs_b.size(0)

        vl = vl_loss / n_vl
        vl_ch = {k: v / n_vl for k, v in vl_ch.items()}

        is_best = vl < best_val
        if is_best:
            best_val = vl
            model.save(best_ckpt)

        # --- Console ---
        logger.info(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={tr:.6f} (s={tr_ch['steer']:.4f} p={tr_ch['pedal']:.4f} b={tr_ch['brake']:.4f})  "
            f"val={vl:.6f} (s={vl_ch['steer']:.4f} p={vl_ch['pedal']:.4f} b={vl_ch['brake']:.4f})"
            + ("  *best*" if is_best else "")
        )

        # --- TensorBoard ---
        tb.add_scalars("loss", {"train": tr, "val": vl}, epoch)
        tb.add_scalars("train/channel", tr_ch, epoch)
        tb.add_scalars("val/channel",   vl_ch, epoch)

        # --- CSV ---
        writer.writerow({
            "epoch":       epoch,
            "train_loss":  round(tr, 6),
            "train_steer": round(tr_ch["steer"], 6),
            "train_pedal": round(tr_ch["pedal"], 6),
            "train_brake": round(tr_ch["brake"], 6),
            "val_loss":    round(vl, 6),
            "val_steer":   round(vl_ch["steer"], 6),
            "val_pedal":   round(vl_ch["pedal"], 6),
            "val_brake":   round(vl_ch["brake"], 6),
            "best_val":    round(best_val, 6),
        })
        csv_file.flush()

    tb.close()
    csv_file.close()

    model.save(final_ckpt)
    logger.info(f"Done. best val={best_val:.6f}")
    logger.info(f"Best checkpoint : {best_ckpt}")
    logger.info(f"Final checkpoint: {final_ckpt}")
    logger.info(f"CSV log         : {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BC trainer — reads preprocessed .npz")
    parser.add_argument("--data",         required=True,
                        help="Path to preprocessed .npz file from preprocess_bc.py")
    parser.add_argument("--output-dir",   default="checkpoints/bc")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    # Companion JSON (optional — for obs_dim verification)
    npz_path  = str(Path(args.data).resolve())
    json_path = str(Path(args.data).with_suffix(".json"))
    if os.path.isfile(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        logger.info(
            f"Data: {meta.get('n_steps')} steps, "
            f"{meta.get('n_episodes')} episodes, "
            f"track={meta.get('track')}  car={meta.get('car')}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load + split
    train_ds, val_ds, obs_dim = load_and_split(
        npz_path, args.val_fraction, args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device_str == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device_str == "cuda"),
    )

    # Model
    model = MLPActor(obs_dim=obs_dim, action_dim=3, device=device_str)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MLPActor: obs_dim={obs_dim}  params={n_params:,}")

    train(model, train_loader, val_loader,
          args.epochs, args.lr, args.output_dir, device)


if __name__ == "__main__":
    main()
