"""
train_sac.py — Entry point for SAC training on Assetto Corsa.

Run commands:
    .\\AssetoCorsa\\Scripts\\python.exe gym/sac/train_sac.py
    # On startup: if checkpoints/sac_monza_v1/latest.pt exists, resumes from it.
    # Otherwise initialises fresh weights and starts from scratch.
    # After each training phase the model is saved to
    # checkpoints/sac_monza_v1/latest.pt (overwritten each phase).
    # To back up manually: copy latest.pt to latest_backup.pt before running.

Training loop:
    Each phase: collect episodes with SAC policy, then train for
    train_steps_per_phase gradient updates on the frames collected
    during that phase only; buffer is cleared after training.

Hyperparameters (held constant across all variants — do NOT tune mid-project):
    gamma:          0.992
    batch_size:     256
    lr:             3e-4
    hidden_units:   [256, 256, 256]
    tau:            0.005
    target_entropy: -3  (= -action_dim)

Replay buffer:
    DualReplayBuffer — positive_buffer (reward > 0) + negative_buffer (reward <= 0)
    Capacity: 25,000 transitions per sub-buffer (FIFO, oldest overwritten first)
    Buffers persist across phases (never cleared between phases)
    Sampling: 50/50 mix from each buffer, with graceful fallback if one is sparse

The environment is built from assetto_corsa_gym/config.yml via
make_ac_env(), then wrapped in OurEnv for reward / control modules.
"""

import sys
import os
import argparse
import logging

# ── Stationary-car crash detection constants ──────────────────────────────────
# If total displacement (oldest→newest position in the window) is below this
# threshold over _STATIONARY_FRAMES consecutive frames, the car is considered
# crashed / stuck and the episode is terminated.
_STATIONARY_FRAMES = 10
_STATIONARY_THRESHOLD_M = 0.5

# ── Path setup — identical to infer.py ────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("train_sac")


# ── SAC hyperparameters (fixed across all variants) ───────────────────────────
SAC_HYPERPARAMS = {
    "obs_dim":        125,
    "action_dim":     3,
    "hidden_units":   [256, 256, 256],
    "lr":             3e-4,
    "gamma":          0.992,
    "tau":            0.005,
    "target_entropy": -3.0,  # = -action_dim
}

# ── Replay buffer config ──────────────────────────────────────────────────────
# Each sub-buffer (positive / negative) holds up to this many transitions.
REPLAY_BUFFER_CAPACITY = 50_000

# ── Agent training config ─────────────────────────────────────────────────────
AGENT_CONFIG = {
    "episodes_per_phase":      10,
    "train_steps_per_phase":   None,    # None = half of current buffer size at train time
    "batch_size":              256,
    "checkpoint_freq":         5,
    "log_interval":            100,
    "warmup_steps":            37,
}


def _resolve_device(device_arg: str) -> str:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def build_env(config_path: str, work_dir: str):
    """
    Build the AssettoCorsaEnv and wrap it in OurEnv.

    Returns
    -------
    OurEnv instance
    """
    from omegaconf import OmegaConf
    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    from our_env import OurEnv

    cfg = OmegaConf.load(config_path)
    assert hasattr(cfg, "AssettoCorsa"), (
        f"config.yml at {config_path} is missing 'AssettoCorsa' block"
    )
    assert hasattr(cfg, "OurEnv"), (
        f"config.yml at {config_path} is missing 'OurEnv' block"
    )

    # In-memory overrides — config.yml on disk stays at its Vector-Q-pipeline
    # values.  These reconstruct the 125-dim obs shape the baseline SAC
    # checkpoints were trained against.  Action semantics are left as the
    # config.yml default (absolute, use_relative_actions=False).
    cfg.AssettoCorsa.add_previous_obs_to_state = True
    cfg.AssettoCorsa.use_target_speed          = False
    logger.info(
        "Env config overrides (in-memory only — config.yml on disk untouched): "
        "add_previous_obs_to_state=True, use_target_speed=False"
    )

    os.makedirs(work_dir, exist_ok=True)
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    logger.info(
        f"AssettoCorsaEnv built — "
        f"obs_dim={ac_env.state_dim}  action_dim={ac_env.action_dim}"
    )

    our_env_cfg = OmegaConf.create({
        "our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)
    })
    env = OurEnv(ac_env, our_env_cfg)
    logger.info("OurEnv ready.")
    return env


def main():
    parser = argparse.ArgumentParser(
        description="SAC training for Assetto Corsa (Variant A — MLP baseline)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            _THIS_DIR, '..', '..', 'assetto_corsa_gym', 'config.yml'
        ),
        help="Path to config.yml (default: assetto_corsa_gym/config.yml)",
    )
    parser.add_argument(
        "--phases",
        type=int,
        default=None,
        help="Number of phases to run (default: None = run forever)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip AC connection preflight checks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device (default: auto — uses CUDA if available)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (default: checkpoints/sac_monza_v1)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--manage-ac",
        action="store_true",
        help=(
            "Launch AC via Content Manager before training. "
            "Writes race.ini/assists.ini, kills existing AC, launches CM URI, "
            "gets car on track. Retries 3x then aborts. Requires Steam running."
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 1. Preflight ──────────────────────────────────────────────────────────
    # Skipped automatically when --manage-ac is set: full_cycle() already
    # confirms AC is live, so the preflight check would be redundant.
    if args.manage_ac:
        logger.info("--manage-ac active — preflight skipped (full_cycle will confirm AC is live).")
    elif not args.skip_preflight:
        try:
            from preflight import run_preflight
            run_preflight(warn_only=False)
        except ImportError:
            logger.warning(
                "preflight module not found — skipping preflight.  "
                "Use --skip-preflight to suppress this warning."
            )
    else:
        logger.warning("Preflight skipped — ensure AC is running in a session.")

    # ── 2. Device ─────────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── 3. Environment ────────────────────────────────────────────────────────
    config_path = os.path.abspath(args.config)
    logger.info(f"Loading config: {config_path}")

    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'outputs', 'sac',
    )
    env = build_env(config_path, work_dir)

    # ── 4. Replay buffer ──────────────────────────────────────────────────────
    # sys.path includes gym/ (via the '..') so 'sac' resolves to gym/sac/
    from sac.replay_buffer import DualReplayBuffer

    replay_buffer = DualReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY,
        obs_dim=SAC_HYPERPARAMS["obs_dim"],
        action_dim=SAC_HYPERPARAMS["action_dim"],
    )
    logger.info(
        f"DualReplayBuffer: capacity_per_buffer={REPLAY_BUFFER_CAPACITY:,}  "
        f"obs_dim={SAC_HYPERPARAMS['obs_dim']}  "
        f"action_dim={SAC_HYPERPARAMS['action_dim']}  "
        f"(positive_buffer + negative_buffer, 25k each, persist across phases)"
    )

    # ── 5. SAC algorithm ──────────────────────────────────────────────────────
    from sac.sac import SAC

    ckpt_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'checkpoints', 'sac_monza_v1',
    )

    sac = SAC(
        obs_dim=SAC_HYPERPARAMS["obs_dim"],
        action_dim=SAC_HYPERPARAMS["action_dim"],
        hidden_units=SAC_HYPERPARAMS["hidden_units"],
        lr=SAC_HYPERPARAMS["lr"],
        gamma=SAC_HYPERPARAMS["gamma"],
        tau=SAC_HYPERPARAMS["tau"],
        target_entropy=SAC_HYPERPARAMS["target_entropy"],
        device=device,
    )

    latest_ckpt = os.path.join(os.path.abspath(ckpt_dir), "latest.pt")
    if os.path.isfile(latest_ckpt):
        logger.info(f"Checkpoint found — resuming from {latest_ckpt}")
        sac.load(latest_ckpt)
    else:
        logger.info(f"No checkpoint found at {latest_ckpt} — starting from scratch.")

    # ── 6. Agent config ───────────────────────────────────────────────────────
    agent_cfg = dict(AGENT_CONFIG)
    agent_cfg["checkpoint_dir"] = ckpt_dir  # already resolved above
    agent_cfg["stationary_frames"]    = _STATIONARY_FRAMES
    agent_cfg["stationary_threshold"] = _STATIONARY_THRESHOLD_M

    # ── 7. Agent ──────────────────────────────────────────────────────────────
    from sac.agent import SACAgent

    agent = SACAgent(
        env=env,
        sac=sac,
        replay_buffer=replay_buffer,
        config=agent_cfg,
        manage_ac=args.manage_ac,
    )

    # ── 8. Run ────────────────────────────────────────────────────────────────
    logger.info(
        f"Starting SAC training — "
        f"phases={'inf' if args.phases is None else args.phases}  "
        f"device={device}  "
        f"gamma={SAC_HYPERPARAMS['gamma']}  "
        f"hidden={SAC_HYPERPARAMS['hidden_units']}  "
        f"target_entropy={SAC_HYPERPARAMS['target_entropy']}"
    )
    agent.run(num_phases=args.phases)

    # ── 9. Cleanup ────────────────────────────────────────────────────────────
    logger.info("Training complete — closing environment.")
    env.close()


if __name__ == "__main__":
    main()
