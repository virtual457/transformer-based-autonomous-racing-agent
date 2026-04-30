"""
train.py — Vector Q v2 FINAL: offline training only.

Loads the local checkpoint + 6-channel memmap buffer, runs N epochs of
gradient updates, saves the updated checkpoint, and writes diagnostic
plots (loss curves, per-channel Q-loss, entropy, alpha) to ./outputs.

No AC interaction. No data collection. Pure offline fine-tuning on top
of the v2 buffer that was copied into this folder.

Run command:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2_final/train.py --epochs 3
"""

import sys
import os
import argparse
import logging
import time

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vectorq_v2_final.train")

# ── Token / window / model constants (MUST match v2 checkpoint) ──────────────
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

BUFFER_CAPACITY_PER_CHANNEL = 100_000

CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
BUFFER_DIR     = os.path.join(CHECKPOINT_DIR, "buffers")
OUTPUTS_DIR    = os.path.join(_THIS_DIR, "outputs")


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def _plot_training_curves(history: dict, out_dir: str) -> None:
    """Write loss/entropy/alpha plots to out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    steps = np.arange(1, len(history["q_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Q-loss (total)
    axes[0, 0].plot(steps, history["q_loss"], color="tab:blue")
    axes[0, 0].set_title("Q loss (total)")
    axes[0, 0].set_xlabel("gradient step")
    axes[0, 0].grid(alpha=0.3)

    # Per-channel Q-loss
    qpc = np.array(history["q_loss_per_channel"])
    axes[0, 1].plot(steps, qpc[:, 0], label="steer",    color="tab:orange")
    axes[0, 1].plot(steps, qpc[:, 1], label="throttle", color="tab:green")
    axes[0, 1].plot(steps, qpc[:, 2], label="brake",    color="tab:red")
    axes[0, 1].set_title("Q loss per channel")
    axes[0, 1].set_xlabel("gradient step")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Policy loss
    axes[0, 2].plot(steps, history["pi_loss"], color="tab:purple")
    axes[0, 2].set_title("Policy loss")
    axes[0, 2].set_xlabel("gradient step")
    axes[0, 2].grid(alpha=0.3)

    # Entropy per action
    epa = np.array(history["entropy_per_action"])
    axes[1, 0].plot(steps, epa[:, 0], label="steer",    color="tab:orange")
    axes[1, 0].plot(steps, epa[:, 1], label="throttle", color="tab:green")
    axes[1, 0].plot(steps, epa[:, 2], label="brake",    color="tab:red")
    axes[1, 0].set_title("Entropy per action")
    axes[1, 0].set_xlabel("gradient step")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Alpha per action
    apa = np.array(history["alpha_per_action"])
    axes[1, 1].plot(steps, apa[:, 0], label="steer",    color="tab:orange")
    axes[1, 1].plot(steps, apa[:, 1], label="throttle", color="tab:green")
    axes[1, 1].plot(steps, apa[:, 2], label="brake",    color="tab:red")
    axes[1, 1].set_title("Alpha per action")
    axes[1, 1].set_xlabel("gradient step")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Mean entropy
    axes[1, 2].plot(steps, history["entropy"], color="tab:cyan")
    axes[1, 2].set_title("Mean entropy")
    axes[1, 2].set_xlabel("gradient step")
    axes[1, 2].grid(alpha=0.3)

    fig.suptitle("Vector-Q v2 Final — Offline Training", fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"train_curves_{int(time.time())}.png")
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    logger.info(f"Training curves saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Vector Q v2 Final — offline training on local memmap buffer"
    )
    parser.add_argument("--epochs",        type=int, default=1,
                        help="Number of passes over the buffer.")
    parser.add_argument("--batch-size",    type=int, default=256)
    parser.add_argument("--chunk-size",    type=int, default=25_000)
    parser.add_argument("--log-interval",  type=int, default=100)
    parser.add_argument("--device",        type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-save",       action="store_true",
                        help="Do not overwrite checkpoints/latest.pt")
    parser.add_argument("--skip-q-plots",  action="store_true",
                        help="Skip Q-function diagnostic plots at the end.")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── Replay buffer ────────────────────────────────────────────────────────
    from transformer_sac_vectorq_v2_final.replay_buffer import (
        SixChannelMemmapBuffer, ChunkedPrefetcher,
    )

    buffer_dir = os.path.abspath(BUFFER_DIR)
    logger.info(f"Opening 6-channel memmap buffer at: {buffer_dir}")
    replay_buffer = SixChannelMemmapBuffer(
        base_dir=buffer_dir,
        capacity_per_buffer=BUFFER_CAPACITY_PER_CHANNEL,
        token_dim=TOKEN_DIM,
        action_dim=TRANSFORMER_CONFIG["action_dim"],
        window_size=WINDOW_SIZE,
        reward_dim=3,
    )
    sizes = replay_buffer.sizes()
    total = sum(sizes.values())
    logger.info(
        f"Buffer sizes — "
        f"s+={sizes['steer_pos']} s-={sizes['steer_neg']}  "
        f"t+={sizes['throttle_pos']} t-={sizes['throttle_neg']}  "
        f"b+={sizes['brake_pos']} b-={sizes['brake_neg']}  "
        f"total={total}"
    )

    if total < args.batch_size:
        logger.error(
            f"Buffer only has {total} entries, need at least {args.batch_size}. Abort."
        )
        return

    # ── SAC ──────────────────────────────────────────────────────────────────
    from transformer_sac_vectorq_v2_final.sac import TransformerSAC

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    sac = TransformerSAC(
        token_dim=TRANSFORMER_CONFIG["token_dim"],
        action_dim=TRANSFORMER_CONFIG["action_dim"],
        window_size=TRANSFORMER_CONFIG["window_size"],
        d_model=TRANSFORMER_CONFIG["d_model"],
        n_heads=TRANSFORMER_CONFIG["n_heads"],
        n_layers=TRANSFORMER_CONFIG["n_layers"],
        ffn_dim=TRANSFORMER_CONFIG["ffn_dim"],
        policy_hidden=TRANSFORMER_CONFIG["policy_hidden"],
        q_hidden=TRANSFORMER_CONFIG["q_hidden"],
        lr=SAC_HYPERPARAMS["lr"],
        gamma=SAC_HYPERPARAMS["gamma"],
        tau=SAC_HYPERPARAMS["tau"],
        target_entropy=SAC_HYPERPARAMS["target_entropy"],
        device=device,
    )

    if os.path.isfile(latest_ckpt):
        logger.info(f"Loading checkpoint: {latest_ckpt}")
        sac.load(latest_ckpt)
    else:
        logger.warning(f"No checkpoint at {latest_ckpt} — training from scratch.")

    # ── Train ────────────────────────────────────────────────────────────────
    steps_per_epoch = max(total // args.batch_size, 1)
    n_steps = steps_per_epoch * args.epochs
    logger.info(
        f"Offline training — epochs={args.epochs}  "
        f"steps_per_epoch={steps_per_epoch}  total_steps={n_steps}  "
        f"batch_size={args.batch_size}  chunk_size={args.chunk_size}"
    )

    prefetcher = ChunkedPrefetcher(
        buffer=replay_buffer,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        device=str(sac.device),
    )
    prefetcher.start()

    history = {
        "q_loss":              [],
        "q_loss_per_channel":  [],
        "pi_loss":             [],
        "alpha":               [],
        "alpha_per_action":    [],
        "entropy":             [],
        "entropy_per_action":  [],
    }

    t_start = time.perf_counter()
    wait_times, gpu_times = [], []

    try:
        for step in range(1, n_steps + 1):
            batch, wait_ms = prefetcher.get()
            wait_times.append(wait_ms)

            t_gpu = time.perf_counter()
            losses = sac.update_from_batch(batch)
            gpu_times.append((time.perf_counter() - t_gpu) * 1000.0)

            history["q_loss"].append(losses["q_loss"])
            history["q_loss_per_channel"].append(
                losses.get("q_loss_per_channel", [0, 0, 0])
            )
            history["pi_loss"].append(losses["policy_loss"])
            history["alpha"].append(losses["alpha"])
            history["alpha_per_action"].append(losses["alpha_per_action"])
            history["entropy"].append(losses["entropy"])
            history["entropy_per_action"].append(losses["entropy_per_action"])

            if step % args.log_interval == 0:
                recent = slice(-args.log_interval, None)
                qpc = np.mean(history["q_loss_per_channel"][recent], axis=0)
                epa = np.mean(history["entropy_per_action"][recent], axis=0)
                apa = np.mean(history["alpha_per_action"][recent], axis=0)
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f"step={step}/{n_steps}  "
                    f"q_loss={float(np.mean(history['q_loss'][recent])):.4f}  "
                    f"Qs=[{qpc[0]:.4f},{qpc[1]:.4f},{qpc[2]:.4f}]  "
                    f"pi={float(np.mean(history['pi_loss'][recent])):.4f}  "
                    f"H=[{epa[0]:.3f},{epa[1]:.3f},{epa[2]:.3f}]  "
                    f"A=[{apa[0]:.4f},{apa[1]:.4f},{apa[2]:.4f}]  "
                    f"swap={float(np.mean(wait_times[recent])):.1f}ms  "
                    f"gpu={float(np.mean(gpu_times[recent])):.1f}ms  "
                    f"elapsed={elapsed:.1f}s"
                )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        prefetcher.stop()

    # ── Save ─────────────────────────────────────────────────────────────────
    if not args.no_save:
        sac.save(latest_ckpt)
        logger.info(f"Checkpoint saved: {latest_ckpt}")

    # ── Plot training curves ─────────────────────────────────────────────────
    if len(history["q_loss"]) > 0:
        _plot_training_curves(history, OUTPUTS_DIR)

    # ── Q-function diagnostics (post-training) ───────────────────────────────
    if not args.skip_q_plots:
        try:
            logger.info("Generating Q-function diagnostics (plot_q_function) ...")
            import plot_q_function as _pqf
            _pqf.main()
        except Exception as exc:
            logger.warning(f"plot_q_function failed: {exc}")

    logger.info(
        f"Done — trained {len(history['q_loss'])} steps in "
        f"{time.perf_counter() - t_start:.1f}s"
    )


if __name__ == "__main__":
    main()
