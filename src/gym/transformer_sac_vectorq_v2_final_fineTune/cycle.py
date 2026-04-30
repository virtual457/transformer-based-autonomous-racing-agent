"""
cycle.py — Alternate collect ↔ train loop with periodic checkpoint backups.

One cycle =
    1. collect.py  --manage-ac  (AI-drive on, N episodes)
    2. train.py    --epochs E

Every `--backup-every` cycles, copies `checkpoints/latest.pt` into
`trained_models/SAC_VectorQ_V2/cycle_backups/<timestamp>_cycle<N>/latest.pt`.
Buffer is NEVER copied or deleted — hard rule.

Ctrl+C between cycles exits cleanly; mid-cycle Ctrl+C interrupts the
currently-running child script, then stops.

Run:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2_final_fineTune/cycle.py
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2_final_fineTune/cycle.py --cycles 20 --collect-episodes 8 --train-epochs 3 --backup-every 5
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_PYTHON    = os.path.join(_REPO_ROOT, "AssetoCorsa", "Scripts", "python.exe")

# Make gym/ac_lifecycle importable.
sys.path.insert(0, os.path.join(_REPO_ROOT, "gym"))

_CHECKPOINT     = os.path.join(_THIS_DIR, "checkpoints", "latest.pt")
_BACKUP_ROOT    = os.path.join(_REPO_ROOT, "trained_models", "SAC_VectorQ_V2",
                               "cycle_backups")
_COLLECT_SCRIPT = os.path.join(_THIS_DIR, "collect.py")
_TRAIN_SCRIPT   = os.path.join(_THIS_DIR, "train.py")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  cycle  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cycle")


def _run(cmd: list[str], label: str) -> int:
    """Run a subprocess, streaming stdout/stderr directly to console."""
    logger.info(f">> {label}: {' '.join(cmd)}")
    t0 = time.time()
    try:
        rc = subprocess.call(cmd, cwd=_REPO_ROOT)
    except KeyboardInterrupt:
        logger.warning(f"{label} interrupted by user.")
        raise
    dt = time.time() - t0
    logger.info(f"<< {label} exit={rc}  elapsed={dt/60.0:.1f} min")
    return rc


def _kill_ac() -> None:
    """Terminate AC processes before training so GPU/CPU are free."""
    try:
        from ac_lifecycle import kill_ac
        logger.info(">> kill_ac() — shutting AC down before train phase")
        kill_ac()
        # Short settle pause so the GPU driver releases resources.
        time.sleep(3.0)
        logger.info("<< kill_ac() done")
    except Exception as exc:
        logger.warning(f"kill_ac failed (continuing anyway): {exc}")


def _backup_checkpoint(cycle_num: int) -> str | None:
    if not os.path.isfile(_CHECKPOINT):
        logger.warning(f"No checkpoint at {_CHECKPOINT} — skipping backup.")
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join(_BACKUP_ROOT, f"{ts}_cycle{cycle_num:03d}")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "latest.pt")
    shutil.copy2(_CHECKPOINT, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    logger.info(f"OK checkpoint backed up -> {dest}  ({size_mb:.1f} MB)")
    return dest


def main():
    parser = argparse.ArgumentParser(
        description="Alternate collect/train loop with periodic backups"
    )
    parser.add_argument("--cycles",            type=int, default=20,
                        help="Total collect/train cycles to run.")
    parser.add_argument("--collect-episodes",  type=int, default=8,
                        help="Episodes per collect phase (passed to collect.py).")
    parser.add_argument("--collect-phases",    type=int, default=1,
                        help="Phases per collect call.")
    parser.add_argument("--train-epochs",      type=int, default=3,
                        help="Epochs per train call.")
    parser.add_argument("--train-batch-size",  type=int, default=256)
    parser.add_argument("--backup-every",      type=int, default=5,
                        help="Back up checkpoint every N cycles (0 = never).")
    parser.add_argument("--device",            type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-ai-drive",       action="store_true",
                        help="Disable AI autopilot warm-up during collect.")
    parser.add_argument("--no-manage-ac",      action="store_true",
                        help="Don't auto-launch AC (assume it's already running).")
    parser.add_argument("--skip-q-plots",      action="store_true",
                        help="Pass --skip-q-plots to train.py (faster).")
    parser.add_argument("--keep-ac-alive",     action="store_true",
                        help="Skip killing AC between collect and train phases.")
    parser.add_argument("--start-cycle",       type=int, default=1,
                        help="Starting cycle number (for resuming).")
    args = parser.parse_args()

    if not os.path.isfile(_PYTHON):
        logger.error(f"Python interpreter not found at {_PYTHON}")
        sys.exit(1)

    logger.info(
        f"Cycle loop — cycles={args.cycles}  "
        f"collect_eps={args.collect_episodes}  train_epochs={args.train_epochs}  "
        f"backup_every={args.backup_every}  manage_ac={not args.no_manage_ac}  "
        f"ai_drive={not args.no_ai_drive}"
    )

    end_cycle = args.start_cycle + args.cycles - 1
    try:
        for cycle_num in range(args.start_cycle, end_cycle + 1):
            logger.info("=" * 70)
            logger.info(f"CYCLE {cycle_num} / {end_cycle}")
            logger.info("=" * 70)

            # ── 1) Collect ────────────────────────────────────────────────────
            collect_cmd = [
                _PYTHON, _COLLECT_SCRIPT,
                "--phases",   str(args.collect_phases),
                "--episodes", str(args.collect_episodes),
                "--device",   args.device,
            ]
            if not args.no_manage_ac:
                collect_cmd.append("--manage-ac")
            if args.no_ai_drive:
                collect_cmd.append("--no-ai-drive")

            rc = _run(collect_cmd, label=f"cycle {cycle_num} COLLECT")
            if rc != 0:
                logger.error(f"collect.py failed (rc={rc}) — stopping cycle loop.")
                break

            # ── 1b) Kill AC so training has full GPU ─────────────────────────
            if not args.keep_ac_alive:
                _kill_ac()

            # ── 2) Train ──────────────────────────────────────────────────────
            train_cmd = [
                _PYTHON, _TRAIN_SCRIPT,
                "--epochs",     str(args.train_epochs),
                "--batch-size", str(args.train_batch_size),
                "--device",     args.device,
            ]
            if args.skip_q_plots:
                train_cmd.append("--skip-q-plots")

            rc = _run(train_cmd, label=f"cycle {cycle_num} TRAIN")
            if rc != 0:
                logger.error(f"train.py failed (rc={rc}) — stopping cycle loop.")
                break

            # ── 3) Periodic checkpoint backup ────────────────────────────────
            if args.backup_every > 0 and cycle_num % args.backup_every == 0:
                _backup_checkpoint(cycle_num)

    except KeyboardInterrupt:
        logger.warning("Cycle loop interrupted by user.")

    logger.info("Cycle loop complete.")


if __name__ == "__main__":
    main()
