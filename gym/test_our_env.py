"""
test_our_env.py — Smoke test for OurEnv.

REQUIREMENTS TO RUN:
  - Assetto Corsa must be running with the AC plugin loaded (sensors_par).
  - Run from the project root (D:/Git/virtual457-projects/AssetoCorsa/).

  Example:
      cd D:/Git/virtual457-projects/AssetoCorsa
      .\\AssetoCorsa\\Scripts\\python.exe gym/test_our_env.py

  If AC is NOT running, the test will block waiting for the UDP connection.
"""

import sys
import os
import logging
import numpy as np

# Add assetto_corsa_gym to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("test_our_env")


def main():
    # 1. Load config
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym', 'config.yml')
    cfg = OmegaConf.load(cfg_path)
    logger.info("Config loaded.")

    assert hasattr(cfg, "OurEnv"), "config.yml is missing the OurEnv block"
    our_env_cfg = OmegaConf.create({
        "our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)
    })

    # 2. Build ACEnv
    logger.info("Building AssettoCorsaEnv ...")
    from AssettoCorsaEnv.assettoCorsa import make_ac_env

    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'test_our_env')
    os.makedirs(work_dir, exist_ok=True)

    ac_env = make_ac_env(cfg=cfg, work_dir=work_dir)
    logger.info(f"ACEnv built. obs_dim={ac_env.state_dim}")

    # 3. Build OurEnv
    from our_env import OurEnv, save_episode
    from policies import FullThrottlePolicy, RandomPolicy, ZeroPolicy

    env = OurEnv(ac_env, our_env_cfg)
    logger.info("OurEnv constructed.")

    # 4. Collect one episode — swap policy here to test different behaviours:
    #   ZeroPolicy()          — car sits still (pipeline smoke test)
    #   FullThrottlePolicy()  — car drives straight until it crashes/goes off-track
    #   RandomPolicy(seed=42) — random exploration
    policy = FullThrottlePolicy()
    logger.info(f"Using policy: {policy}")
    trajectory = env.collect_episode(policy=policy)

    # 5. Validate
    T = trajectory["metadata"]["episode_steps"]
    logger.info(f"Episode completed. T={T} steps")
    assert T > 0, "Episode returned zero steps"

    obs      = trajectory["observations"]
    actions  = trajectory["actions"]
    rewards  = trajectory["rewards"]
    components = trajectory["reward_components"]

    assert obs.shape     == (T, ac_env.state_dim), f"obs shape mismatch: {obs.shape}"
    assert actions.shape == (T, 3),                f"actions shape mismatch: {actions.shape}"
    assert rewards.shape == (T,),                  f"rewards shape mismatch: {rewards.shape}"

    logger.info(f"  total_reward    : {trajectory['metadata']['total_reward']:.4f}")
    logger.info(f"  off_track_count : {trajectory['metadata']['off_track_count']}")
    logger.info(f"  max_speed_ms    : {trajectory['metadata']['max_speed_ms']:.2f}")
    for k, v in components.items():
        logger.info(f"  {k:15s} sum={v.sum():.4f}  mean={v.mean():.4f}  std={v.std():.4f}")

    # 6. Save and verify
    output_dir = our_env_cfg.our_env.data_output_path
    parquet_path = save_episode(trajectory, output_dir=output_dir, episode_number=0)

    assert os.path.isfile(parquet_path), f"Parquet not found: {parquet_path}"
    logger.info(f"Parquet written: {parquet_path}")

    import pandas as pd
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    assert len(df) == T, f"Parquet row count {len(df)} != episode steps {T}"
    logger.info(f"Parquet shape: {df.shape}")

    metadata_path = os.path.join(output_dir, "episodes_metadata.jsonl")
    assert os.path.isfile(metadata_path), f"Metadata JSONL not found"
    logger.info(f"Metadata JSONL written: {metadata_path}")

    logger.info("All assertions passed. OurEnv smoke test COMPLETE.")
    env.close()


if __name__ == "__main__":
    main()
