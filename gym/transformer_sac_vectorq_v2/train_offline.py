"""
train_offline.py — Run training epochs on existing replay buffer (no AC needed).

Usage:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2/train_offline.py --epochs 10
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2/train_offline.py --epochs 5 --batch-size 512
"""

import sys
import os
import argparse
import time
import logging

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

# ── Logging ──────────────────────────────────────────────────────────────────
class _CleanFormatter(logging.Formatter):
    _RESET  = "\033[0m"
    _COLORS = {
        logging.DEBUG:    "\033[90m",
        logging.INFO:     "\033[97m",
        logging.WARNING:  "\033[93m",
        logging.ERROR:    "\033[91m",
    }
    _METRIC_COLOR = "\033[96m"

    def format(self, record):
        msg = record.getMessage()
        color = self._COLORS.get(record.levelno, self._RESET)
        if any(k in msg for k in ("q_loss=", "pi_loss=", "alpha=", "step=")):
            color = self._METRIC_COLOR
        return f"{color}{msg}{self._RESET}"

_handler = logging.StreamHandler()
_handler.setFormatter(_CleanFormatter())
logging.root.setLevel(logging.WARNING)
logging.root.handlers = [_handler]
logger = logging.getLogger("train_offline")
logger.setLevel(logging.INFO)
logger.propagate = True

# ── Constants (must match train.py) ──────────────────────────────────────────
TOKEN_DIM   = 50
WINDOW_SIZE = 75

TRANSFORMER_CONFIG = {
    "token_dim":      TOKEN_DIM,
    "action_dim":     3,
    "window_size":    WINDOW_SIZE,
    "d_model":        256,
    "n_heads":        4,
    "n_layers":       4,
    "ffn_dim":        1024,
    "policy_hidden":  [256],
    "q_hidden":       [256],
}

SAC_HYPERPARAMS = {
    "lr":             3e-4,
    "gamma":          0.992,
    "tau":            0.005,
    "target_entropy": -1.5,
}

CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
BUFFER_DIR     = os.path.join(CHECKPOINT_DIR, "buffers")


def main():
    parser = argparse.ArgumentParser(description="Offline training on existing buffer")
    parser.add_argument("--epochs",          type=int, default=5,      help="Number of training epochs")
    parser.add_argument("--batch-size",      type=int, default=256)
    parser.add_argument("--log-interval",    type=int, default=100)
    parser.add_argument("--checkpoint-freq", type=int, default=1,      help="Save checkpoint every N epochs")
    parser.add_argument("--device",          type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--chunk-size",      type=int, default=25_000,
                        help="Windows per chunk loaded into RAM (default 25K ≈ 750 MB)")
    parser.add_argument("--preload",         action="store_true",
                        help="Preload all buffers into RAM (~12 GB) for fast sampling")
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    # ── Replay buffer ────────────────────────────────────────────────────────
    from transformer_sac_vectorq_v2.replay_buffer import SixChannelMemmapBuffer

    buffer_dir = os.path.abspath(BUFFER_DIR)
    replay_buffer = SixChannelMemmapBuffer(
        base_dir=buffer_dir,
        capacity_per_buffer=args.buffer_capacity,
        token_dim=TOKEN_DIM,
        action_dim=TRANSFORMER_CONFIG["action_dim"],
        window_size=WINDOW_SIZE,
        reward_dim=3,
    )
    sizes = replay_buffer.sizes()
    total = sum(sizes.values())
    logger.info(f"Buffer loaded — sizes: {sizes}  total: {total}")

    if args.preload:
        logger.info("Preloading buffers into RAM ...")
        replay_buffer.preload()

    if total < args.batch_size:
        logger.error(f"Buffer has {total} entries, need at least {args.batch_size}. Exiting.")
        return

    # ── SAC ───────────────────────────────────────────────────────────────────
    from transformer_sac_vectorq_v2.sac import TransformerSAC

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    sac = TransformerSAC(
        **TRANSFORMER_CONFIG,
        **SAC_HYPERPARAMS,
        device=device,
    )

    if os.path.isfile(latest_ckpt):
        logger.info(f"Loading checkpoint: {latest_ckpt}")
        sac.load(latest_ckpt)
    else:
        logger.info("No checkpoint found — starting from scratch.")

    # ── Prefetcher ────────────────────────────────────────────────────────────
    from transformer_sac_vectorq_v2.replay_buffer import ChunkedPrefetcher

    prefetcher = ChunkedPrefetcher(
        buffer=replay_buffer,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        device=device,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    steps_per_epoch = max(total // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    global_step = 0

    logger.info(
        f"Starting offline training — "
        f"{args.epochs} epochs x {steps_per_epoch} steps = {total_steps} total  "
        f"batch_size={args.batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'=' * 60}\n  EPOCH {epoch}/{args.epochs}  ({steps_per_epoch} steps)\n{'=' * 60}")

        q_losses, policy_losses, alphas, entropies = [], [], [], []
        q_loss_per_ch_buf      = []
        entropy_per_action_buf = []
        alpha_per_action_buf   = []
        wait_times             = []
        gpu_times              = []
        t_start = time.perf_counter()

        prefetcher.start()

        for step in range(1, steps_per_epoch + 1):
            batch, wait_ms = prefetcher.get()
            wait_times.append(wait_ms)

            t_gpu_start = time.perf_counter()
            losses = sac.update_from_batch(batch)
            gpu_ms = (time.perf_counter() - t_gpu_start) * 1000.0
            gpu_times.append(gpu_ms)

            global_step += 1

            q_losses.append(losses["q_loss"])
            policy_losses.append(losses["policy_loss"])
            alphas.append(losses["alpha"])
            entropies.append(losses["entropy"])
            q_loss_per_ch_buf.append(losses.get("q_loss_per_channel", [0, 0, 0]))
            entropy_per_action_buf.append(losses["entropy_per_action"])
            alpha_per_action_buf.append(losses["alpha_per_action"])

            if step % args.log_interval == 0:
                recent = slice(-args.log_interval, None)
                elapsed = time.perf_counter() - t_start
                epa = np.mean(entropy_per_action_buf[recent], axis=0)
                apa = np.mean(alpha_per_action_buf[recent],   axis=0)
                qpc = np.mean(q_loss_per_ch_buf[recent],      axis=0)
                avg_wait = float(np.mean(wait_times[recent]))
                max_wait = float(max(wait_times[recent]))
                avg_gpu  = float(np.mean(gpu_times[recent]))
                logger.info(
                    f"  step={step}/{steps_per_epoch}  "
                    f"q_loss={float(np.mean(q_losses[recent])):.4f}  "
                    f"Qs=[{qpc[0]:.4f},{qpc[1]:.4f},{qpc[2]:.4f}]  "
                    f"pi_loss={float(np.mean(policy_losses[recent])):.4f}  "
                    f"alpha={float(np.mean(alphas[recent])):.4f}  "
                    f"H=[{epa[0]:.3f},{epa[1]:.3f},{epa[2]:.3f}]  "
                    f"A=[{apa[0]:.4f},{apa[1]:.4f},{apa[2]:.4f}]  "
                    f"swap={avg_wait:.1f}ms(max={max_wait:.0f})  gpu={avg_gpu:.1f}ms  "
                    f"elapsed={elapsed:.1f}s  "
                    f"global_step={global_step}"
                )

        prefetcher.stop()

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"  Epoch {epoch} done — "
            f"mean_q_loss={float(np.mean(q_losses)):.4f}  "
            f"mean_pi_loss={float(np.mean(policy_losses)):.4f}  "
            f"mean_alpha={float(np.mean(alphas)):.4f}  "
            f"mean_entropy={float(np.mean(entropies)):.4f}  "
            f"avg_wait={float(np.mean(wait_times)):.1f}ms  "
            f"avg_gpu={float(np.mean(gpu_times)):.1f}ms  "
            f"elapsed={elapsed:.1f}s"
        )

        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
            sac.save(latest_ckpt)
            logger.info(f"  Checkpoint saved: {latest_ckpt}")

        # Free CUDA cache between epochs.
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    logger.info(f"\nOffline training complete — {global_step} total gradient steps over {args.epochs} epochs.")


if __name__ == "__main__":
    main()
