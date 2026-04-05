"""
finetune_on_demo.py — Offline SAC fine-tuning on AICLONE demonstration data.

Takes a copy of an existing SAC checkpoint (e.g. 150_jitter) and runs
offline gradient steps using the AICLONE demo transitions (reward=1.0).
No AC connection required — purely offline.

How it works
------------
1. Copy source checkpoint → output dir (never modifies the original).
2. Load SAC (actor + critics + optimizers) from the copy.
3. Convert aiclone_dataset.npz into transitions:
       (obs[t], action[t], reward=1.0, next_obs=obs[t+1], done=0/1)
   done=1 only at episode boundaries (last step of each episode).
4. Run `grad_steps` SAC update_from_batch() calls, sampling random
   mini-batches from the demo transitions each time.
5. Save updated checkpoint to output dir.

Usage
-----
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/finetune_on_demo.py

    # Custom source model and grad steps
    ..\\AssetoCorsa\\Scripts\\python.exe AICLONE/finetune_on_demo.py \\
        --source  trained_models/SAC/150_jitter/model.pt \\
        --out-dir AICLONE/checkpoints/150_demo \\
        --grad-steps 10000 --batch-size 256

Output
------
    AICLONE/checkpoints/150_demo/finetuned.pt   — updated SAC checkpoint
    AICLONE/checkpoints/150_demo/log.csv        — per-step losses
"""

import argparse
import csv
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_REPO    = Path(__file__).resolve().parent.parent
_GYM_DIR = _REPO / "gym"
for _p in [str(_GYM_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sac.sac import SAC   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOG_EVERY = 500   # print a log line every N grad steps


# ── Demo buffer ───────────────────────────────────────────────────────────────

class DemoBuffer:
    """
    Holds all demonstration transitions in memory and samples random batches.

    Transitions are built from the npz as:
        obs[t], action[t], reward=1.0, next_obs=obs[t+1], done=0
    At episode boundaries (last step of each episode):
        done=1, next_obs=obs[t]  (terminal — no bootstrap)
    """

    def __init__(self, npz_path: str, device: torch.device):
        data         = np.load(npz_path)
        obs          = data["obs"]           # (N, 125)
        actions      = data["actions"]       # (N, 3)
        episode_ends = data["episode_ends"]  # (E,)  — index of last step per episode

        N = len(obs)
        terminal = np.zeros(N, dtype=np.float32)
        terminal[episode_ends] = 1.0

        # next_obs: shift by 1; at terminal steps use current obs (no bootstrap)
        next_obs = np.empty_like(obs)
        next_obs[:-1] = obs[1:]
        next_obs[-1]  = obs[-1]
        # Override episode boundaries: next_obs at terminal = current obs
        next_obs[episode_ends] = obs[episode_ends]

        rewards = np.ones(N, dtype=np.float32)

        self.obs      = torch.FloatTensor(obs).to(device)
        self.actions  = torch.FloatTensor(actions).to(device)
        self.rewards  = torch.FloatTensor(rewards).unsqueeze(1).to(device)   # (N,1)
        self.next_obs = torch.FloatTensor(next_obs).to(device)
        self.done     = torch.FloatTensor(terminal).unsqueeze(1).to(device)  # (N,1)
        self.N        = N

        logger.info(
            f"DemoBuffer: {N} transitions  "
            f"({int(terminal.sum())} terminal)  "
            f"device={device}"
        )

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.N, (batch_size,))
        return {
            "obs":      self.obs[idx],
            "action":   self.actions[idx],
            "reward":   self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "done":     self.done[idx],
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline SAC fine-tuning on AICLONE demo data"
    )
    parser.add_argument(
        "--source",
        default=str(_REPO / "trained_models" / "SAC" / "150_jitter" / "model.pt"),
        help="Source SAC checkpoint to fine-tune (default: 150_jitter)",
    )
    parser.add_argument(
        "--data",
        default=str(_REPO / "AICLONE" / "data" / "aiclone_dataset.npz"),
        help="AICLONE demo dataset (default: AICLONE/data/aiclone_dataset.npz)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO / "AICLONE" / "checkpoints" / "150_demo"),
    )
    parser.add_argument(
        "--grad-steps", type=int, default=10_000,
        help="Number of SAC gradient steps (default: 10000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    source_path = str(Path(args.source).resolve())
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Copy source checkpoint — never touch the original ─────────────────
    copy_path = str(out_dir / "source_copy.pt")
    shutil.copy2(source_path, copy_path)
    logger.info(f"Copied {source_path} → {copy_path}")

    # ── 2. Load SAC from copy ─────────────────────────────────────────────────
    sac = SAC.from_checkpoint(copy_path, device=device_str)
    logger.info(
        f"Loaded SAC: obs_dim={sac.obs_dim}  action_dim={sac.action_dim}  "
        f"alpha={sac.alpha.item():.4f}"
    )

    # ── 3. Load demo buffer ───────────────────────────────────────────────────
    npz_path = str(Path(args.data).resolve())
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Demo dataset not found: {npz_path}\n"
            f"Run preprocess_parquet.py first."
        )
    buffer = DemoBuffer(npz_path, device)

    # ── 4. Offline gradient steps ─────────────────────────────────────────────
    csv_path = out_dir / "log.csv"
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "step", "q_loss", "policy_loss", "alpha_loss", "alpha", "entropy"
    ])
    writer.writeheader()

    logger.info(f"Starting {args.grad_steps} offline gradient steps ...")

    running = {k: 0.0 for k in ["q_loss", "policy_loss", "alpha_loss", "alpha", "entropy"]}

    for step in range(1, args.grad_steps + 1):
        batch  = buffer.sample(args.batch_size)
        losses = sac.update_from_batch(batch)

        for k in running:
            running[k] += losses[k]

        writer.writerow({"step": step, **{k: round(losses[k], 6) for k in losses}})

        if step % LOG_EVERY == 0:
            avg = {k: running[k] / LOG_EVERY for k in running}
            logger.info(
                f"step {step:6d}/{args.grad_steps}  "
                f"q_loss={avg['q_loss']:.4f}  "
                f"policy_loss={avg['policy_loss']:.4f}  "
                f"alpha={avg['alpha']:.4f}  "
                f"entropy={avg['entropy']:.4f}"
            )
            running = {k: 0.0 for k in running}
            csv_file.flush()

    csv_file.close()

    # ── 5. Save fine-tuned checkpoint ─────────────────────────────────────────
    out_path = str(out_dir / "finetuned.pt")
    sac.save(out_path)
    logger.info(f"Fine-tuned checkpoint saved: {out_path}")
    logger.info(f"Loss log: {csv_path}")
    logger.info(
        "\nTo run the fine-tuned model:\n"
        f"    .\\AssetoCorsa\\Scripts\\python.exe gym/model_runners/run_model.py "
        f"--model 150_demo\n"
        f"  (copy {out_path} to trained_models/SAC/150_demo/model.pt first)"
    )


if __name__ == "__main__":
    main()
