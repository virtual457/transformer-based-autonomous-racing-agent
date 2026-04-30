"""
collect.py — Production data collection entry point.

Runs the preflight checks, builds the environment, then collects
episodes with the chosen policy and saves them as parquet files.

Prerequisites
-------------
- Assetto Corsa must be running with:
    - Track: monza  (or whatever is in config.yml)
    - Car:   ks_mazda_miata
    - Session type: Practice or Hotlap (NOT race start screen)
- sensors_par plugin must be loaded (auto with AC if installed)
- vJoy Device 1 driver must be running
- Vjoy.ini active profile must have INPUT_METHOD=WHEEL

Usage
-----
    # From the project root:
    .\\AssetoCorsa\\Scripts\\python.exe gym/collect.py

    # Options:
    .\\AssetoCorsa\\Scripts\\python.exe gym/collect.py --episodes 20
    .\\AssetoCorsa\\Scripts\\python.exe gym/collect.py --policy random
    .\\AssetoCorsa\\Scripts\\python.exe gym/collect.py --policy neural --checkpoint path/to/model.pt
    .\\AssetoCorsa\\Scripts\\python.exe gym/collect.py --skip-preflight
"""

import sys
import os
import logging
import argparse

# Path setup — same as test_our_env.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("collect")


def build_policy(args, env=None):
    from policies.simple import ZeroPolicy, FullThrottlePolicy, RandomPolicy
    from policies.neural import NeuralPolicy
    from policies.math_policy import MathPolicy

    name = args.policy.lower()
    if name == "zero":
        return ZeroPolicy()
    elif name == "full_throttle":
        return FullThrottlePolicy()
    elif name == "random":
        return RandomPolicy(seed=args.seed)
    elif name == "neural":
        if not args.checkpoint:
            raise ValueError("--checkpoint required when --policy neural")
        return NeuralPolicy.from_checkpoint(
            args.checkpoint,
            obs_log_path=getattr(args, "obs_log", None),
            steer_deadzone=getattr(args, "steer_deadzone", 0.0),
        )
    elif name == "math":
        if env is None or env.track_info is None:
            raise ValueError(
                "MathPolicy needs track_info — is AC running with a session active?"
            )
        return MathPolicy(
            racing_line=env.track_info["fast_lane"],
            speed_arr=[max(15.0, float(s)) for s in env.track_info["speed_arr"]],
        )
    else:
        raise ValueError(f"Unknown policy: {args.policy}")


def main():
    parser = argparse.ArgumentParser(description="Collect AC episodes for offline training")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episodes_per_run from config")
    parser.add_argument("--policy", type=str, default="full_throttle",
                        choices=["zero", "full_throttle", "random", "neural", "math"],
                        help="Policy to use for data collection")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for --policy neural)")
    parser.add_argument("--obs-log", type=str, default=None,
                        help="If set, log obs+decision to this CSV file (--policy neural only)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for --policy random)")
    parser.add_argument("--episode-start", type=int, default=0,
                        help="Starting episode number (for resuming runs)")
    parser.add_argument("--warmup", type=float, default=0.0, metavar="SECS",
                        help="Full-throttle warm-up seconds before inference starts (default: 0)")
    parser.add_argument("--steer-deadzone", type=float, default=0.0, metavar="DZ",
                        help="Steer dead zone half-width in [-1,1] space (e.g. 0.1 → [-0.1,0.1] snaps to 0)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight checks (not recommended)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging for policy (shows per-step telemetry + decision)")
    parser.add_argument("--config", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            '..', 'assetto_corsa_gym', 'config.yml'),
                        help="Path to config.yml")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("policies.math_policy").setLevel(logging.DEBUG)
        logging.getLogger("our_env").setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # 1. Preflight
    # ------------------------------------------------------------------
    if not args.skip_preflight:
        from preflight import run_preflight
        run_preflight(warn_only=False)   # exits if any check fails
    else:
        logger.warning("Preflight skipped — make sure AC is running")

    # ------------------------------------------------------------------
    # 2. Config
    # ------------------------------------------------------------------
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    assert hasattr(cfg, "OurEnv"), "config.yml missing OurEnv block"

    our_env_cfg = OmegaConf.create({
        "our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)
    })

    # ------------------------------------------------------------------
    # 3. Build ACEnv
    # ------------------------------------------------------------------
    from AssettoCorsaEnv.assettoCorsa import make_ac_env

    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'outputs', 'collect',
    )
    os.makedirs(work_dir, exist_ok=True)

    logger.info("Building AssettoCorsaEnv ...")
    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    logger.info(f"ACEnv built. obs_dim={ac_env.state_dim}")

    # ------------------------------------------------------------------
    # 4. Build OurEnv
    # ------------------------------------------------------------------
    from our_env import OurEnv, collect_and_save

    if args.episodes is not None:
        # Override episodes_per_run in config
        our_env_cfg.our_env.episodes_per_run = args.episodes

    env = OurEnv(ac_env, our_env_cfg)
    logger.info(f"OurEnv ready. episodes_per_run={env.episodes_per_run}")

    # ------------------------------------------------------------------
    # 5. Build policy
    # ------------------------------------------------------------------
    policy = build_policy(args, env=env)
    logger.info(f"Policy: {policy}")

    # ------------------------------------------------------------------
    # 6. Collect
    # ------------------------------------------------------------------
    from step_logger import StepLogger

    step_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "collected_data", "step_decisions.csv",
    )
    step_logger = StepLogger(step_log_path)
    step_logger.open()
    logger.info(f"Step log: {step_log_path} (cleared for this run)")

    logger.info(
        f"Starting collection: {env.episodes_per_run} episodes "
        f"starting from ep {args.episode_start}"
    )
    paths = collect_and_save(
        env=env,
        policy=policy,
        episode_start=args.episode_start,
        step_logger=step_logger,
        throttle_warmup_s=args.warmup,
    )
    step_logger.close()

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    logger.info(f"\nCollection complete. {len(paths)} episodes saved:")
    for p in paths:
        logger.info(f"  {p}")

    env.close()


if __name__ == "__main__":
    main()
