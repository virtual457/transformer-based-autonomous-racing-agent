"""
train.py — Fine-tuning entry point for Transformer SAC on Assetto Corsa.

Differences from transformer_sac/train.py:
  - Loads checkpoint and buffer from transformer_sac_finetune/checkpoints/
  - Saves checkpoints to transformer_sac_finetune/checkpoints/
  - OOT tolerance: 1 frame — episode terminates immediately on first OOT frame
  - OOT reward: -1.0 per frame while off-track
  - AC-friendly: AC stays running during GPU training phases
  - All imports come from transformer_sac_finetune/ — nothing shared with
    transformer_sac/ so changes here never affect the main trainer

Run command:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_finetune/train.py --manage-ac
"""

import sys
import os
import argparse
import logging

# ── Stationary-car crash detection constants ──────────────────────────────────
_STATIONARY_FRAMES      = 30
_STATIONARY_THRESHOLD_M = 0.5

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

# ── Logging setup ─────────────────────────────────────────────────────────────
class _CleanFormatter(logging.Formatter):
    """Message-only formatter with ANSI colors. No timestamps, levels, or module names."""
    _RESET  = "\033[0m"
    _COLORS = {
        logging.DEBUG:    "\033[90m",   # dark grey
        logging.INFO:     "\033[97m",   # white
        logging.WARNING:  "\033[93m",   # yellow
        logging.ERROR:    "\033[91m",   # red
        logging.CRITICAL: "\033[95m",   # magenta
    }
    # Step/phase metrics get a different color to stand out
    _METRIC_COLOR = "\033[96m"          # cyan

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        color = self._COLORS.get(record.levelno, self._RESET)
        # Metrics lines (step=, q_loss=, etc.) in cyan
        if any(k in msg for k in ("q_loss=", "pi_loss=", "alpha=", "entropy=", "step=")):
            color = self._METRIC_COLOR
        return f"{color}{msg}{self._RESET}"

_handler = logging.StreamHandler()
_handler.setFormatter(_CleanFormatter())

# Root logger: WARNING+ only (suppresses noisy third-party libs)
logging.root.setLevel(logging.WARNING)
logging.root.handlers = [_handler]

# Our loggers: full INFO
for _name in ("finetune_transformer_sac", "transformer_sac_finetune"):
    _log = logging.getLogger(_name)
    _log.setLevel(logging.INFO)
    _log.propagate = True

logger = logging.getLogger("finetune_transformer_sac")

# ── Token / window constants ──────────────────────────────────────────────────
TOKEN_DIM   = 50
WINDOW_SIZE = 75

# ── Transformer hyperparameters (identical to main trainer — do NOT change) ───
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

# ── SAC hyperparameters ───────────────────────────────────────────────────────
SAC_HYPERPARAMS = {
    "lr":             3e-4,
    "gamma":          0.992,
    "tau":            0.005,
    "target_entropy": -1.5,  # tight finetuning: 68% actions in ±0.1, 95% in ±0.3
}

# ── Replay buffer config ──────────────────────────────────────────────────────
REPLAY_BUFFER_CAPACITY_WINDOWS = 50_000

# ── Checkpoint / buffer directory ─────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")

# ── Agent training config ─────────────────────────────────────────────────────
AGENT_CONFIG = {
    "episodes_per_phase":      10,
    "train_steps_per_phase":   None,   # None = min(steps_collected, 2_000)
    "batch_size":              256,
    "checkpoint_freq":         5,
    "log_interval":            100,
    "warmup_steps":            200,
}


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def build_env(config_path: str, work_dir: str):
    from omegaconf import OmegaConf
    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    from our_env import OurEnv

    cfg = OmegaConf.load(config_path)
    assert hasattr(cfg, "AssettoCorsa"), "config.yml missing 'AssettoCorsa' block"
    assert hasattr(cfg, "OurEnv"),       "config.yml missing 'OurEnv' block"

    cfg.AssettoCorsa.enable_out_of_track_termination = False
    cfg.AssettoCorsa.add_previous_obs_to_state       = False
    logger.info("Overrode enable_out_of_track_termination=False, add_previous_obs_to_state=False")

    os.makedirs(work_dir, exist_ok=True)
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    logger.info(f"AssettoCorsaEnv built — obs_dim={ac_env.state_dim}  action_dim={ac_env.action_dim}")

    our_env_cfg = OmegaConf.create({"our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)})
    env = OurEnv(ac_env, our_env_cfg)
    logger.info("OurEnv ready.")
    return env


def main():
    parser = argparse.ArgumentParser(description="Transformer SAC fine-tuning for Assetto Corsa")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'config.yml'),
        help="Path to config.yml",
    )
    parser.add_argument("--phases",         type=int,  default=None,   help="Number of phases (default: forever)")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--device",         type=str,  default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--verbose", "-v",  action="store_true")
    parser.add_argument(
        "--manage-ac", action="store_true",
        help="AC-friendly mode: launch AC once, keep alive during training phases.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 1. Preflight ──────────────────────────────────────────────────────────
    if args.manage_ac:
        logger.info("--manage-ac active — preflight skipped.")
    elif not args.skip_preflight:
        try:
            from preflight import run_preflight
            run_preflight(warn_only=False)
        except ImportError:
            logger.warning("preflight module not found — skipping.")
    else:
        logger.warning("Preflight skipped — ensure AC is running.")

    # ── 2. Device ─────────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── 3. Environment ────────────────────────────────────────────────────────
    config_path = os.path.abspath(args.config)
    work_dir    = os.path.join(_THIS_DIR, '..', '..', 'outputs', 'transformer_sac_finetune')
    env = build_env(config_path, work_dir)

    # ── 4. Replay buffer ──────────────────────────────────────────────────────
    from transformer_sac_finetune.replay_buffer import DualWindowReplayBuffer

    replay_buffer = DualWindowReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY_WINDOWS,
        token_dim=TOKEN_DIM,
        action_dim=TRANSFORMER_CONFIG["action_dim"],
        window_size=WINDOW_SIZE,
    )

    # ── 5. TransformerSAC ─────────────────────────────────────────────────────
    from transformer_sac_finetune.sac import TransformerSAC

    ckpt_dir    = CHECKPOINT_DIR
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")

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
        logger.info(f"Checkpoint found — resuming from {latest_ckpt}")
        sac.load(latest_ckpt)
    else:
        logger.info(f"No checkpoint at {latest_ckpt} — starting from scratch.")

    # ── 6. Agent config ───────────────────────────────────────────────────────
    agent_cfg = dict(AGENT_CONFIG)
    agent_cfg["checkpoint_dir"]       = ckpt_dir
    agent_cfg["stationary_frames"]    = _STATIONARY_FRAMES
    agent_cfg["stationary_threshold"] = _STATIONARY_THRESHOLD_M

    # ── 7. Agent ──────────────────────────────────────────────────────────────
    from transformer_sac_finetune.agent import FinetuneAgent

    agent = FinetuneAgent(
        env=env,
        sac=sac,
        replay_buffer=replay_buffer,
        config=agent_cfg,
        token_dim=TOKEN_DIM,
        window_size=WINDOW_SIZE,
        manage_ac=args.manage_ac,
    )
    # Drop the local reference so the agent holds the only one.
    # This allows _unload_buffer() to fully free the numpy arrays from RAM.
    del replay_buffer

    # ── 8. Run ────────────────────────────────────────────────────────────────
    logger.info(
        f"Starting Transformer SAC fine-tuning — "
        f"phases={'inf' if args.phases is None else args.phases}  "
        f"device={device}  "
        f"oot_limit=75 frames  oot_penalty=-1.0  "
        f"target_entropy={SAC_HYPERPARAMS['target_entropy']}"
    )
    agent.run(num_phases=args.phases)

    # ── 9. Cleanup ────────────────────────────────────────────────────────────
    logger.info("Fine-tuning complete — closing environment.")
    env.close()


if __name__ == "__main__":
    main()
