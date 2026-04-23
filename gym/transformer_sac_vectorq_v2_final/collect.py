"""
collect.py — Vector Q v2 FINAL: online collection only (no gradient updates).

Drives the current policy in Assetto Corsa, pushes complete episodes into
the 6-channel memmap buffer, and flushes to disk. No SAC updates happen.

Use this to grow the buffer before running `train.py`, or to gather
evaluation episodes.

Run command:
    .\\AssetoCorsa\\Scripts\\python.exe gym/transformer_sac_vectorq_v2_final/collect.py --manage-ac --phases 1
"""

import sys
import os
import argparse
import logging

_STATIONARY_FRAMES      = 30
_STATIONARY_THRESHOLD_M = 0.5

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vectorq_v2_final.collect")

TOKEN_DIM   = 50
WINDOW_SIZE = 75

TRANSFORMER_CONFIG = {
    "token_dim": TOKEN_DIM, "action_dim": 3, "window_size": WINDOW_SIZE,
    "d_model": 256, "n_heads": 4, "n_layers": 4, "ffn_dim": 1024,
    "policy_hidden": [256], "q_hidden": [256],
}
SAC_HYPERPARAMS = {
    "lr": 3e-4, "gamma": 0.992, "tau": 0.005, "target_entropy": -1.5,
}
BUFFER_CAPACITY_PER_CHANNEL = 100_000

CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints")
BUFFER_DIR     = os.path.join(CHECKPOINT_DIR, "buffers")


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
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    cfg.AssettoCorsa.enable_out_of_track_termination = False
    cfg.AssettoCorsa.add_previous_obs_to_state       = False
    from AssettoCorsaEnv.assettoCorsa import make_ac_env
    os.makedirs(work_dir, exist_ok=True)
    _t0 = _time.perf_counter()
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    logger.info(f"make_ac_env done in {_time.perf_counter() - _t0:.1f}s")
    from our_env import OurEnv
    our_env_cfg = OmegaConf.create({"our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)})
    return OurEnv(ac_env, our_env_cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Vector Q v2 Final — online collection (no gradient)"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(_THIS_DIR, '..', '..', 'assetto_corsa_gym', 'config.yml'),
    )
    parser.add_argument("--phases",         type=int, default=1)
    parser.add_argument("--episodes",       type=int, default=10,
                        help="Episodes per phase.")
    parser.add_argument("--device",         type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--manage-ac",      action="store_true")
    parser.add_argument("--no-ai-drive",    action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--buffer-capacity", type=int, default=BUFFER_CAPACITY_PER_CHANNEL,
    )
    args = parser.parse_args()

    # ── Preflight + AC launch ────────────────────────────────────────────────
    if args.manage_ac:
        from ac_lifecycle import full_cycle
        logger.info("Launching AC via full_cycle() ...")
        full_cycle(max_retries=3)
    elif not args.skip_preflight:
        try:
            from preflight import run_preflight
            run_preflight(warn_only=False)
        except ImportError:
            logger.warning("preflight module not found — skipping.")

    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    config_path = os.path.abspath(args.config)
    work_dir    = os.path.join(_THIS_DIR, '..', '..', 'outputs', 'transformer_sac_vectorq_v2_final')
    env = build_env(config_path, work_dir)

    # ── Replay buffer ────────────────────────────────────────────────────────
    from transformer_sac_vectorq_v2_final.replay_buffer import SixChannelMemmapBuffer

    buffer_dir = os.path.abspath(BUFFER_DIR)
    replay_buffer = SixChannelMemmapBuffer(
        base_dir=buffer_dir,
        capacity_per_buffer=args.buffer_capacity,
        token_dim=TOKEN_DIM,
        action_dim=TRANSFORMER_CONFIG["action_dim"],
        window_size=WINDOW_SIZE,
        reward_dim=3,
    )
    logger.info(f"Buffer total before collection: {sum(replay_buffer.sizes().values())}")

    # ── SAC (inference only) ─────────────────────────────────────────────────
    from transformer_sac_vectorq_v2_final.sac import TransformerSAC

    sac = TransformerSAC(
        **TRANSFORMER_CONFIG,
        **SAC_HYPERPARAMS,
        device=device,
    )
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if os.path.isfile(latest_ckpt):
        logger.info(f"Loading checkpoint: {latest_ckpt}")
        sac.load(latest_ckpt)
    else:
        logger.warning(f"No checkpoint at {latest_ckpt} — using random policy!")

    # ── Agent (use FinetuneAgent, but only call collect_phase) ───────────────
    from transformer_sac_vectorq_v2_final.agent import FinetuneAgent

    agent_cfg = {
        "episodes_per_phase":   args.episodes,
        "checkpoint_dir":       CHECKPOINT_DIR,
        "stationary_frames":    _STATIONARY_FRAMES,
        "stationary_threshold": _STATIONARY_THRESHOLD_M,
        "use_ai_drive":         not args.no_ai_drive,
        "manage_ac":            args.manage_ac,
    }
    agent = FinetuneAgent(
        env=env, sac=sac, replay_buffer=replay_buffer, config=agent_cfg,
        token_dim=TOKEN_DIM, window_size=WINDOW_SIZE, manage_ac=args.manage_ac,
    )

    # ── Collection loop (no training) ────────────────────────────────────────
    try:
        for phase in range(args.phases):
            logger.info(f"\n{'=' * 60}\n  COLLECT PHASE {phase}\n{'=' * 60}")

            if args.manage_ac:
                from ac_lifecycle import full_cycle
                full_cycle(max_retries=3)
                agent._check_ac_alive(phase_num=phase, ep=0)

            stats = agent.collect_phase(phase)
            replay_buffer.flush()

            sizes = replay_buffer.sizes()
            logger.info(
                f"[Phase {phase}] COLLECT done — "
                f"episodes={stats['episodes']}  steps={stats['total_steps']}  "
                f"windows_added={stats['windows_added']}  "
                f"mean_reward={stats['mean_reward']:.3f}  "
                f"buf_total={sum(sizes.values())}"
            )
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user.")
    finally:
        replay_buffer.flush()
        env.close()
        logger.info("Collection complete — buffer flushed, env closed.")


if __name__ == "__main__":
    main()
