"""
train.py — Entry point for Transformer SAC training on Assetto Corsa (Variant D).

Run command:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac/train.py

    On startup: if checkpoints/sac_transformer_v1/latest.pt exists, resumes
    from it. Otherwise initialises fresh weights and starts from scratch.
    After each training phase the model is saved to
    checkpoints/sac_transformer_v1/latest.pt (overwritten each phase).
    To back up manually: copy latest.pt to latest_backup.pt before running.

Architecture:
    - Token: obs[:50] extracted from 125-dim env obs (obs[50:] dropped here)
    - Window: 75 frames (3 seconds at 25 Hz)
    - Transformer: d_model=256, n_heads=4, n_layers=4, ffn_dim=1024, Pre-LN
    - Policy head: MLP(256 -> 256 -> 6) -> mean+log_std -> tanh-squashed
    - Q heads: MLP(256+3 -> 256 -> 1) x2 (action concat after encoder)
    - Encoder trained by critic only; policy uses detached embedding

SAC hyperparameters (held constant across all variants — do NOT tune):
    gamma:          0.992
    batch_size:     256
    lr:             3e-4
    tau:            0.005
    target_entropy: -3.0  (= -action_dim)

Config override:
    cfg.AssettoCorsa.add_previous_obs_to_state = False is set programmatically
    after loading config.yml — do NOT edit config.yml.
    The env still returns 125-dim obs; obs[:TOKEN_DIM] is extracted in the agent.

Replay buffer:
    DualWindowReplayBuffer — window-based, pos/neg split by mean reward per window.
    Capacity: 50,000 windows per sub-buffer (pre-allocated circular arrays).
    Buffers persist across phases (never cleared between phases).
"""

import sys
import os
import argparse
import logging

# ── Stationary-car crash detection constants ──────────────────────────────────
# Transformer variant uses 30 frames (not 10 as in the MLP baseline) because
# the rolling 75-frame window means the car may legitimately move slowly during
# early-episode warmup without being stuck.  The MLP baseline uses 10 because
# its episode length is shorter and 10 frames is sufficient to detect a crash.
_STATIONARY_FRAMES = 30
_STATIONARY_THRESHOLD_M = 0.5

# ── Path setup — identical to gym/sac/train_sac.py ────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("train_transformer_sac")

# ── Token / window constants ───────────────────────────────────────────────────
TOKEN_DIM   = 50   # obs[:50] — basic obs + out_of_track + curvature + past actions + current action
WINDOW_SIZE = 75   # 3 seconds at 25 Hz

# ── Transformer hyperparameters (locked) ──────────────────────────────────────
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

# ── SAC hyperparameters (held constant — do NOT tune mid-project) ──────────────
SAC_HYPERPARAMS = {
    "lr":             3e-4,
    "gamma":          0.992,
    "tau":            0.005,
    "target_entropy": 0.5,   # floor: SAC increases alpha when entropy drops below this
}

# ── Replay buffer config ───────────────────────────────────────────────────────
# Each sub-buffer (positive / negative) holds up to this many pre-computed windows.
REPLAY_BUFFER_CAPACITY_WINDOWS = 50_000

# ── Agent training config ─────────────────────────────────────────────────────
AGENT_CONFIG = {
    "episodes_per_phase":      10,
    "train_steps_per_phase":   None,    # None = min(steps_collected, 2_000)
    "batch_size":              256,
    "checkpoint_freq":         5,
    "log_interval":            100,
    "warmup_steps":            200,     # nominal; actual is random 150-250 per episode
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

    Overrides cfg.AssettoCorsa.add_previous_obs_to_state = False after
    loading config so the env does not include the redundant prev-obs suffix
    in the raw obs.  The env will still return a 125-dim obs because the
    previous obs dims in the obs vector are zeroed/omitted, but obs[:50]
    (our token) is unaffected either way.

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

    # Override: disable the env's built-in OOT termination — our custom
    # 75-frame consecutive counter in agent.py handles termination instead.
    # Do NOT edit config.yml.
    cfg.AssettoCorsa.enable_out_of_track_termination = False
    logger.info(
        "Overrode cfg.AssettoCorsa.enable_out_of_track_termination = False "
        "(custom 75-frame OOT counter in agent.py handles termination)."
    )

    # Override: do not append previous obs to state — Transformer handles
    # temporal context via the rolling window.  Do NOT edit config.yml.
    cfg.AssettoCorsa.add_previous_obs_to_state = False
    logger.info(
        "Overrode cfg.AssettoCorsa.add_previous_obs_to_state = False "
        "(Transformer uses rolling window for temporal context)."
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
        description=(
            "Transformer SAC training for Assetto Corsa (Variant D — Transformer encoder)"
        )
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
        help=(
            "Override checkpoint directory "
            "(default: checkpoints/sac_transformer_v1)"
        ),
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
    if args.manage_ac:
        logger.info(
            "--manage-ac active — preflight skipped (full_cycle will confirm AC is live)."
        )
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
        '..', '..', 'outputs', 'transformer_sac',
    )
    env = build_env(config_path, work_dir)

    # ── 4. Replay buffer ──────────────────────────────────────────────────────
    from transformer_sac.replay_buffer import DualWindowReplayBuffer

    replay_buffer = DualWindowReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY_WINDOWS,
        token_dim=TOKEN_DIM,
        action_dim=TRANSFORMER_CONFIG["action_dim"],
        window_size=WINDOW_SIZE,
    )
    logger.info(
        f"DualWindowReplayBuffer: capacity_per_buffer={REPLAY_BUFFER_CAPACITY_WINDOWS} windows  "
        f"token_dim={TOKEN_DIM}  action_dim={TRANSFORMER_CONFIG['action_dim']}  "
        f"window_size={WINDOW_SIZE}  "
        f"(positive_buffer + negative_buffer, 50k windows each, persist across phases)"
    )

    # ── 5. TransformerSAC algorithm ───────────────────────────────────────────
    from transformer_sac.sac import TransformerSAC

    ckpt_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'checkpoints', 'sac_transformer_v1',
    )

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

    latest_ckpt = os.path.join(os.path.abspath(ckpt_dir), "latest.pt")
    if os.path.isfile(latest_ckpt):
        logger.info(f"Checkpoint found — resuming from {latest_ckpt}")
        sac.load(latest_ckpt)
    else:
        logger.info(f"No checkpoint found at {latest_ckpt} — starting from scratch.")

    # ── 6. Agent config ───────────────────────────────────────────────────────
    agent_cfg = dict(AGENT_CONFIG)
    agent_cfg["checkpoint_dir"]     = ckpt_dir
    agent_cfg["stationary_frames"]  = _STATIONARY_FRAMES
    agent_cfg["stationary_threshold"] = _STATIONARY_THRESHOLD_M

    # ── 7. Agent ──────────────────────────────────────────────────────────────
    from transformer_sac.agent import TransformerSACAgent

    agent = TransformerSACAgent(
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
        f"Starting Transformer SAC training (Variant D) — "
        f"phases={'inf' if args.phases is None else args.phases}  "
        f"device={device}  "
        f"d_model={TRANSFORMER_CONFIG['d_model']}  "
        f"n_layers={TRANSFORMER_CONFIG['n_layers']}  "
        f"window_size={WINDOW_SIZE}  "
        f"token_dim={TOKEN_DIM}  "
        f"gamma={SAC_HYPERPARAMS['gamma']}  "
        f"target_entropy={SAC_HYPERPARAMS['target_entropy']}"
    )
    agent.run(num_phases=args.phases)

    # ── 9. Cleanup ────────────────────────────────────────────────────────────
    logger.info("Training complete — closing environment.")
    env.close()


if __name__ == "__main__":
    main()
