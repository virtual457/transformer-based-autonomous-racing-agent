"""
train.py — Vector Q v2: 6-channel memmap-backed replay buffer.

Key difference from transformer_sac_vectorq/train.py:
  - Uses SixChannelMemmapBuffer instead of DualWindowReplayBuffer.
  - Buffer lives on disk (memmap) — large capacity, low RAM.
  - No buffer save/load cycle — memmap persists automatically.

Run command:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2/train.py --manage-ac
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
    _RESET  = "\033[0m"
    _COLORS = {
        logging.DEBUG:    "\033[90m",
        logging.INFO:     "\033[97m",
        logging.WARNING:  "\033[93m",
        logging.ERROR:    "\033[91m",
        logging.CRITICAL: "\033[95m",
    }
    _METRIC_COLOR = "\033[96m"

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        color = self._COLORS.get(record.levelno, self._RESET)
        if any(k in msg for k in ("q_loss=", "pi_loss=", "alpha=", "entropy=", "step=")):
            color = self._METRIC_COLOR
        return f"{color}{msg}{self._RESET}"

_handler = logging.StreamHandler()
_handler.setFormatter(_CleanFormatter())

logging.root.setLevel(logging.WARNING)
logging.root.handlers = [_handler]

for _name in ("vectorq_v2_transformer_sac", "transformer_sac_vectorq_v2"):
    _log = logging.getLogger(_name)
    _log.setLevel(logging.INFO)
    _log.propagate = True

logger = logging.getLogger("vectorq_v2_transformer_sac")

# ── Token / window constants ──────────────────────────────────────────────────
TOKEN_DIM   = 50
WINDOW_SIZE = 75

# ── Transformer hyperparameters ───────────────────────────────────────────────
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
    "target_entropy": -1.5,
}

# ── Replay buffer config ──────────────────────────────────────────────────────
# 100K windows per channel buffer.  6 buffers x ~2.8 GB each = ~16.8 GB on disk.
BUFFER_CAPACITY_PER_CHANNEL = 100_000

# ── Checkpoint / buffer directory ─────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
BUFFER_DIR     = os.path.join(CHECKPOINT_DIR, "buffers")

# ── Agent training config ─────────────────────────────────────────────────────
AGENT_CONFIG = {
    "episodes_per_phase":      10,
    "train_steps_per_phase":   None,
    "train_epochs_per_phase":  2,
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
    import time as _time

    logger.info("build_env: loading config ...")
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    assert hasattr(cfg, "AssettoCorsa"), "config.yml missing 'AssettoCorsa' block"
    assert hasattr(cfg, "OurEnv"),       "config.yml missing 'OurEnv' block"

    cfg.AssettoCorsa.enable_out_of_track_termination = False
    cfg.AssettoCorsa.add_previous_obs_to_state       = False
    logger.info("build_env: overrides applied (OOT=False, prev_obs=False)")

    logger.info("build_env: importing make_ac_env ...")
    from AssettoCorsaEnv.assettoCorsa import make_ac_env

    os.makedirs(work_dir, exist_ok=True)
    logger.info("build_env: calling make_ac_env() ...")
    _t0 = _time.perf_counter()
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    _t1 = _time.perf_counter()
    logger.info(f"build_env: make_ac_env() done in {_t1 - _t0:.1f}s — obs_dim={ac_env.state_dim}  action_dim={ac_env.action_dim}")

    logger.info("build_env: importing OurEnv ...")
    from our_env import OurEnv
    our_env_cfg = OmegaConf.create({"our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)})
    env = OurEnv(ac_env, our_env_cfg)
    logger.info("build_env: OurEnv ready.")
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Transformer SAC v2 — 6-channel memmap buffer"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'config.yml'),
        help="Path to config.yml",
    )
    parser.add_argument("--phases",         type=int,  default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--device",         type=str,  default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--verbose", "-v",  action="store_true")
    parser.add_argument("--manage-ac",      action="store_true")
    parser.add_argument("--no-ai-drive",    action="store_true")
    parser.add_argument(
        "--buffer-capacity", type=int, default=BUFFER_CAPACITY_PER_CHANNEL,
        help=f"Windows per channel buffer (default: {BUFFER_CAPACITY_PER_CHANNEL}). "
             f"6 buffers total. ~2.8 GB disk per 100K windows.",
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
    logger.info("Importing torch (may take a few seconds) ...")
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── 3. Environment ────────────────────────────────────────────────────────
    # Launch AC before building env — make_ac_env connects to AC on init.
    if args.manage_ac:
        from ac_lifecycle import full_cycle
        logger.info("Launching AC via full_cycle() before env init ...")
        full_cycle(max_retries=3)

    config_path = os.path.abspath(args.config)
    work_dir    = os.path.join(_THIS_DIR, '..', '..', 'outputs', 'transformer_sac_vectorq_v2')
    env = build_env(config_path, work_dir)

    # ── 4. Replay buffer (6-channel memmap) ───────────────────────────────────
    from transformer_sac_vectorq_v2.replay_buffer import SixChannelMemmapBuffer

    buffer_dir = os.path.abspath(BUFFER_DIR)
    logger.info(
        f"Creating 6-channel memmap buffer at: {buffer_dir}  "
        f"capacity_per_channel={args.buffer_capacity}"
    )
    replay_buffer = SixChannelMemmapBuffer(
        base_dir=buffer_dir,
        capacity_per_buffer=args.buffer_capacity,
        token_dim=TOKEN_DIM,
        action_dim=TRANSFORMER_CONFIG["action_dim"],
        window_size=WINDOW_SIZE,
        reward_dim=3,
    )
    _sizes = replay_buffer.sizes()
    logger.info(f"Buffer initialised — sizes: {_sizes}  total: {sum(_sizes.values())}")

    # ── 5. TransformerSAC ─────────────────────────────────────────────────────
    from transformer_sac_vectorq_v2.sac import TransformerSAC

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
    agent_cfg["use_ai_drive"]         = not args.no_ai_drive

    # ── 7. Agent ──────────────────────────────────────────────────────────────
    from transformer_sac_vectorq_v2.agent import FinetuneAgent

    agent = FinetuneAgent(
        env=env,
        sac=sac,
        replay_buffer=replay_buffer,
        config=agent_cfg,
        token_dim=TOKEN_DIM,
        window_size=WINDOW_SIZE,
        manage_ac=args.manage_ac,
    )

    # ── 8. Run ────────────────────────────────────────────────────────────────
    logger.info(
        f"Starting Transformer SAC v2 — "
        f"phases={'inf' if args.phases is None else args.phases}  "
        f"device={device}  "
        f"buffer_capacity_per_channel={args.buffer_capacity}  "
        f"target_entropy={SAC_HYPERPARAMS['target_entropy']}"
    )
    agent.run(num_phases=args.phases)

    # ── 9. Cleanup ────────────────────────────────────────────────────────────
    logger.info("Training complete — closing environment.")
    env.close()


if __name__ == "__main__":
    main()
