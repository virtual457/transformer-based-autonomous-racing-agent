"""
Offline environment setup test -- run WITHOUT Assetto Corsa open.

Checks every pre-training prerequisite:
  1. Core imports (torch, gym, omegaconf, wandb, numba, etc.)
  2. CUDA availability
  3. Config loads and target track/car are valid
  4. Track data files exist (CSVs + occupancy PKL)
  5. Car config files exist
  6. ReferenceLap loads and curvature look-ahead works
  7. Reward formula math (unit tests, no AC needed)
  8. SAC network instantiates at correct input dim

Usage:
  cd assetto_corsa_gym
  python test_setup.py
"""

import sys
import os

sys.path.extend([os.path.abspath('./assetto_corsa_gym')])

PASS = "  [PASS]"
FAIL = "  [FAIL]"

results = []

def check(name, fn):
    try:
        msg = fn()
        print(f"{PASS}  {name}" + (f" -- {msg}" if msg else ""))
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL}  {name} -- {e}")
        results.append((name, False, str(e)))


# --------------------------------------------------------------------------
# 1. Core imports
# --------------------------------------------------------------------------

def _import_torch():
    import torch
    return f"torch {torch.__version__}"

def _import_gym():
    import gym
    return f"gym {gym.__version__}"

def _import_omegaconf():
    from omegaconf import OmegaConf
    return "omegaconf OK"

def _import_wandb():
    import wandb
    return f"wandb {wandb.__version__}"

def _import_pandas():
    import pandas as pd
    return f"pandas {pd.__version__}"

def _import_pyarrow():
    import pyarrow
    return f"pyarrow {pyarrow.__version__}"

def _import_numba():
    import numba
    return f"numba {numba.__version__}"

def _import_scipy():
    import scipy
    return f"scipy {scipy.__version__}"

def _import_sklearn():
    import sklearn
    return f"scikit-learn {sklearn.__version__}"

print("\n-- 1. Core imports --")
check("torch",        _import_torch)
check("gym",          _import_gym)
check("omegaconf",    _import_omegaconf)
check("wandb",        _import_wandb)
check("pandas",       _import_pandas)
check("pyarrow",      _import_pyarrow)
check("numba",        _import_numba)
check("scipy",        _import_scipy)
check("scikit-learn", _import_sklearn)


# --------------------------------------------------------------------------
# 2. CUDA
# --------------------------------------------------------------------------

def _cuda():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name}, {mem:.1f} GB VRAM"

print("\n-- 2. CUDA --")
check("CUDA available", _cuda)


# --------------------------------------------------------------------------
# 3. Config
# --------------------------------------------------------------------------

CONFIGS_PATH = os.path.abspath("./assetto_corsa_gym/AssettoCorsaConfigs")
TRACKS_PATH  = os.path.join(CONFIGS_PATH, "tracks")
CARS_PATH    = os.path.join(CONFIGS_PATH, "cars")

cfg = None

def _load_config():
    global cfg
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config.yml")
    track = cfg.AssettoCorsa.track
    car   = cfg.AssettoCorsa.car
    return f"track={track}  car={car}"

def _track_in_config():
    import yaml
    track_config_path = os.path.join(TRACKS_PATH, "config.yaml")
    with open(track_config_path) as f:
        track_cfg = yaml.safe_load(f)
    track = cfg.AssettoCorsa.track
    assert track in track_cfg, f"Track '{track}' not in tracks/config.yaml"
    return f"{track} found in config.yaml"

print("\n-- 3. Config --")
check("config.yml loads",     _load_config)
check("track in config.yaml", _track_in_config)


# --------------------------------------------------------------------------
# 4. Track data files
# --------------------------------------------------------------------------

def _track_border_csv():
    track = cfg.AssettoCorsa.track
    p = os.path.join(TRACKS_PATH, f"{track}.csv")
    assert os.path.exists(p), f"Missing: {p}"
    import pandas as pd
    df = pd.read_csv(p)
    return f"{len(df)} rows"

def _track_racing_line_csv():
    track = cfg.AssettoCorsa.track
    p = os.path.join(TRACKS_PATH, f"{track}-racing_line.csv")
    assert os.path.exists(p), f"Missing: {p}"
    import pandas as pd
    df = pd.read_csv(p)
    return f"{len(df)} rows"

def _track_occupancy_pkl():
    track = cfg.AssettoCorsa.track
    p = os.path.join(TRACKS_PATH, f"{track}_0.1m.pkl")
    assert os.path.exists(p), (
        f"Missing occupancy grid: {p} -- "
        "run generate_track.ipynb to create it before training."
    )
    return f"found ({os.path.getsize(p) / 1024**2:.1f} MB)"

print("\n-- 4. Track data files --")
check("track border CSV",       _track_border_csv)
check("track racing line CSV",  _track_racing_line_csv)
check("track occupancy PKL",    _track_occupancy_pkl)


# --------------------------------------------------------------------------
# 5. Car config files
# --------------------------------------------------------------------------

def _car_steer_map():
    car = cfg.AssettoCorsa.car
    p = os.path.join(CARS_PATH, car, "steer_map.csv")
    assert os.path.exists(p), f"Missing: {p}"
    return car

def _car_brake_map():
    car = cfg.AssettoCorsa.car
    p = os.path.join(CARS_PATH, car, "brake_map.csv")
    assert os.path.exists(p), f"Missing: {p}"
    return car

print("\n-- 5. Car config files --")
check("steer_map.csv", _car_steer_map)
check("brake_map.csv", _car_brake_map)


# --------------------------------------------------------------------------
# 6. ReferenceLap
# --------------------------------------------------------------------------

def _reference_lap_loads():
    from AssettoCorsaEnv.reference_lap import ReferenceLap
    track = cfg.AssettoCorsa.track
    use_target_speed = cfg.AssettoCorsa.get("use_target_speed", False)
    rl_path = os.path.join(TRACKS_PATH, f"{track}-racing_line.csv")
    rl = ReferenceLap(rl_path, use_target_speed=use_target_speed)
    # td is the distance-interpolated array; rows = positions along track
    assert rl.td.shape[0] > 100, "Racing line too short"
    return f"{rl.td.shape[0]} distance points"

def _curvature_lookahead():
    from AssettoCorsaEnv.reference_lap import ReferenceLap
    track = cfg.AssettoCorsa.track
    use_target_speed = cfg.AssettoCorsa.get("use_target_speed", False)
    rl_path = os.path.join(TRACKS_PATH, f"{track}-racing_line.csv")
    rl = ReferenceLap(rl_path, use_target_speed=use_target_speed)
    # signature: get_curvature_segment(dist, LA_dist, vector_size)
    vec = rl.get_curvature_segment(dist=100.0, LA_dist=300.0, vector_size=12)
    assert len(vec) == 12, f"Expected 12-dim curvature vector, got {len(vec)}"
    return "12-dim curvature vector OK"

print("\n-- 6. ReferenceLap --")
check("ReferenceLap loads",        _reference_lap_loads)
check("curvature look-ahead 12d",  _curvature_lookahead)


# --------------------------------------------------------------------------
# 7. Reward formula
# --------------------------------------------------------------------------

def _reward_on_racing_line():
    """On racing line (gap=0), reward = speed/300."""
    speed_kmh = 150.0
    gap = 0.0
    reward = speed_kmh * (1.0 - abs(gap) / 12.0) / 300.0
    expected = 150.0 / 300.0
    assert abs(reward - expected) < 1e-9
    return f"reward={reward:.4f} (expected {expected:.4f})"

def _reward_at_gap_limit():
    """At gap=12m (outer boundary), reward should be 0."""
    speed_kmh = 200.0
    gap = 12.0
    reward = speed_kmh * (1.0 - abs(gap) / 12.0) / 300.0
    assert abs(reward) < 1e-9, f"Expected 0, got {reward}"
    return "reward=0.0 at gap=12m boundary"

def _reward_is_non_negative():
    """Reward must be >= 0 for any |gap| <= 12."""
    speeds = [50, 100, 200, 300]
    gaps   = [0, 3, 6, 11.9]
    for s in speeds:
        for g in gaps:
            r = s * (1.0 - abs(g) / 12.0) / 300.0
            assert r >= 0, f"Negative reward: speed={s}, gap={g}, r={r}"
    return f"checked {len(speeds)*len(gaps)} (speed, gap) combinations"

print("\n-- 7. Reward formula --")
check("reward on racing line", _reward_on_racing_line)
check("reward at gap=12m",     _reward_at_gap_limit)
check("reward non-negative",   _reward_is_non_negative)


# --------------------------------------------------------------------------
# 8. SAC network instantiation
# --------------------------------------------------------------------------

def _sac_policy_builds():
    import torch
    sys.path.insert(0, os.path.abspath("./algorithm/discor"))
    from discor.network import GaussianPolicy

    # obs dim approximation (see ac_env.py for full breakdown):
    # 11 telemetry + 11 raycasts + 1 OOT + 12 curvature + 9 past_actions + 3 current_actions = 47
    # add_previous_obs_to_state=True adds 3*47 = 141 -> ~188 total
    obs_dim    = 188
    action_dim = 3
    hidden     = [256, 256, 256]

    policy = GaussianPolicy(obs_dim, action_dim, hidden)
    policy.eval()
    dummy = torch.zeros(1, obs_dim)
    with torch.no_grad():
        action, entropies, means = policy(dummy)
    assert action.shape == (1, action_dim), f"Bad action shape: {action.shape}"
    return f"GaussianPolicy({obs_dim} -> {action_dim}) forward pass OK"

def _sac_q_builds():
    import torch
    sys.path.insert(0, os.path.abspath("./algorithm/discor"))
    from discor.network import TwinnedStateActionFunction

    obs_dim    = 188
    action_dim = 3
    hidden     = [256, 256, 256]

    q = TwinnedStateActionFunction(obs_dim, action_dim, hidden)
    q.eval()
    s = torch.zeros(1, obs_dim)
    a = torch.zeros(1, action_dim)
    with torch.no_grad():
        q1, q2 = q(s, a)
    assert q1.shape == (1, 1), f"Bad Q shape: {q1.shape}"
    return f"TwinnedStateActionFunction({obs_dim}+{action_dim} -> 1) forward pass OK"

print("\n-- 8. SAC network --")
check("GaussianPolicy builds",             _sac_policy_builds)
check("TwinnedStateActionFunction builds", _sac_q_builds)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\n{'='*60}")
print(f"  {passed}/{total} checks passed", end="")
if failed:
    print(f"  ({failed} FAILED)\n")
    print("  Failed checks:")
    for name, ok, err in results:
        if not ok:
            print(f"    x {name}: {err}")
else:
    print("  -- all clear, ready for Phase 2")
print("="*60)

sys.exit(0 if failed == 0 else 1)
