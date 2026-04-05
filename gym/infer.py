"""
infer.py — Live inference with a trained neural-network policy.

Connects to Assetto Corsa via UDP, builds the 125-dim observation vector in
real-time (same pipeline as collect.py — UDP packets → expand_state → get_obs),
and feeds it to a trained model.  Actions are sent back to AC via vJoy.

Frame-history warm-up
---------------------
The observation vector includes 3 frames of history (slots h-3, h-2, h-1).
After reset(), ac_env runs 2 flush steps internally, clears self.states, then
runs one more step — so history already has 1 real frame when we receive obs
from reset().  We run 3 additional zero-control steps before inference starts:

    After reset():   len(states)=1  (1 real frame; h-3,h-2 filled by warm-start)
    After warmup 1:  len(states)=2  (2 real frames; h-3 still warm-start copy)
    After warmup 2:  len(states)=3  (all 3 history slots are real frames!)
    After warmup 3:  len(states)=4  (sliding window: last-3 all real)  ← inference starts

Sliding window after that:
    Each step appends the current state to the deque.
    get_obs() always reads history[-3], history[-2], history[-1] → oldest→newest.
    Frame order is preserved: the oldest of the three is h-3, newest is h-1.

Speed warm-up (optional, --speed-warmup)
-----------------------------------------
After the history warm-up, the car is at rest (~0 m/s).  The BC model was
trained on data filtered to ≥5 m/s (mean ~32 m/s), so starting inference
at 0 m/s is out-of-distribution.

With --speed-warmup enabled, the agent applies full throttle (steer=0.5,
throttle=1.0, brake=0.0 in policy space) until the car reaches
--speed-warmup-target m/s OR --speed-warmup-max-steps steps elapse,
whichever comes first.  These steps are logged with phase="speed_warmup".
Inference starts only after the speed threshold is reached.

    After history warmup : speed ~ 0 m/s
    Speed warmup steps   : full throttle, steer centred
    Inference starts when : speed >= target  (or max-steps exhausted)

Obs layout (125 dims, matches training)
---------------------------------------
  [0–24]   Current basic obs: 14 car channels + 11 ray-cast wall sensors
  [25]     out_of_track flag
  [26–37]  Curvature look-ahead (12 points × next 300 m from racing line)
  [38–46]  Past 3 absolute steer / throttle / brake  (PAST_ACTIONS_WINDOW=3)
  [47–49]  Current applied steer / throttle / brake
  [50–74]  History frame h-3  (basic obs, 25 dims)
  [75–99]  History frame h-2  (basic obs, 25 dims)
  [100–124] History frame h-1 (basic obs, 25 dims)

Debug logging
-------------
Every episode clears debug_latest_episode.csv (next to this script) and
writes one row per frame — warmup AND inference — with:
  • All 125 obs dimensions (named columns matching neural.py layout)
  • Raw model output BEFORE the [-1,1]→[0,1] remap: steer_raw / throttle_raw / brake_raw
  • Final policy-space action [0,1]: steer / throttle / brake
  • AC-space action [-1,1]: steer_ac / throttle_ac / brake_ac
  • Key telemetry from the raw state dict for cross-checking

Usage
-----
    .\\AssetoCorsa\\Scripts\\python.exe gym/infer.py --checkpoint path/to/model.pt
    .\\AssetoCorsa\\Scripts\\python.exe gym/infer.py --checkpoint path/to/model.pt --skip-preflight
"""

import sys
import os
import csv
import time
import logging
import argparse
from pathlib import Path

# ── Path setup — identical to collect.py ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym', 'assetto_corsa_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assetto_corsa_gym', 'algorithm', 'discor'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("infer")

# ── Constants ─────────────────────────────────────────────────────────────────
WARMUP_FRAMES   = 3     # zero-control steps before first inference (see docstring)
LOG_EVERY       = 25    # terminal log: one line per second at 25 Hz
DEBUG_LOG_PATH  = Path(__file__).parent / "debug_latest_episode.csv"

# Speed warm-up defaults (overridden by CLI args)
_DEFAULT_SPEED_WARMUP_TARGET    = 30.0   # m/s — close to training mean (32 m/s); must clear pit lane limiter (~17 m/s)
_DEFAULT_SPEED_WARMUP_MAX_STEPS = 600    # ~24 s at 25 Hz — enough to exit pit lane and accelerate


# ── Debug logger ──────────────────────────────────────────────────────────────

class DebugLogger:
    """
    Writes one CSV row per frame to debug_latest_episode.csv.

    Columns
    -------
    episode, step, phase (warmup/infer)
    obs_0 … obs_124        — the exact 125-dim vector sent to the model
                             (named using NeuralPolicy._obs_column_names)
    steer_raw              — model output BEFORE [-1,1] → [0,1] remap
    throttle_raw           —   (multiply final action by 2 and subtract 1)
    brake_raw              —
    steer / throttle / brake          — final policy-space action [0,1]
    steer_ac / throttle_ac / brake_ac — AC-space action [-1,1] sent to vJoy
    speed_ms               — car speed from raw state
    nsp                    — NormalizedSplinePosition (0→1 track progress)
    gear                   — actualGear
    steer_angle_deg        — steerAngle from raw state (degrees)
    acc_status             — accStatus from raw state [0,1]
    brake_status           — brakeStatus from raw state [0,1]
    gap_m                  — lateral gap from racing line (metres)
    out_of_track           — 0/1 flag
    history_len            — len(env.env.states) at time of decision
    """

    # Key raw-state channels to log alongside the obs for cross-checking
    RAW_TELEM_COLS = [
        "speed",                     # m/s  (normalised by /80 → obs[8])
        "NormalizedSplinePosition",  # 0→1
        "actualGear",
        "steerAngle",                # degrees
        "accStatus",
        "brakeStatus",
        "gap",                       # lateral gap metres
        "out_of_track",
        "angular_velocity_y",
        "local_velocity_x",
        "local_velocity_y",
        "accelX",
        "accelY",
        "RPM",
    ]

    def __init__(self, path: Path, obs_dim: int):
        self.path    = path
        self.obs_dim = obs_dim

        # Named obs columns from NeuralPolicy (imported lazily to avoid
        # circular imports — NeuralPolicy is available after path setup)
        from policies.neural import NeuralPolicy
        self._obs_col_names = NeuralPolicy._obs_column_names(obs_dim)

        # Truncate file and write header once per infer run
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        header = (
            ["episode", "step", "phase"]
            + self._obs_col_names                              # 125 obs dims
            + ["steer_raw", "throttle_raw", "brake_raw"]       # model output in [-1,1]
            + ["steer", "throttle", "brake"]                   # policy space [0,1]
            + ["steer_ac", "throttle_ac", "brake_ac"]          # AC space [-1,1]
            + self.RAW_TELEM_COLS                              # raw state channels
            + ["history_len"]
        )
        self._writer.writerow(header)
        self._file.flush()
        logger.info(f"Debug log cleared and opened: {self.path}")

    def open_for_episode(self, episode: int):
        """No-op: file stays open across episodes; rows are appended continuously."""
        logger.info(f"Debug log: episode {episode} rows will be appended to {self.path}")

    def log(
        self,
        episode:    int,
        step:       int,
        phase:      str,          # "warmup" or "infer"
        obs:        np.ndarray,   # (125,)  — input to model
        action:     np.ndarray,   # (3,)    — policy-space [0,1]
        ac_action:  np.ndarray,   # (3,)    — AC-space [-1,1]
        raw_state:  dict,         # env.env.state after expand_state()
        history_len: int,
    ):
        if self._writer is None:
            return

        # Model raw output: remap [0,1] → [-1,1]  (inverse of NeuralPolicy remap)
        raw_action = action * 2.0 - 1.0   # steer_raw, throttle_raw, brake_raw

        telem_vals = [
            raw_state.get(col, float("nan")) for col in self.RAW_TELEM_COLS
        ]

        row = (
            [episode, step, phase]
            + [round(float(v), 6) for v in obs]
            + [round(float(v), 6) for v in raw_action]
            + [round(float(v), 6) for v in action]
            + [round(float(v), 6) for v in ac_action]
            + [round(float(v), 6) if isinstance(v, float) else v for v in telem_vals]
            + [history_len]
        )
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file   = None
            self._writer = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coast_ac_action(env) -> np.ndarray:
    """
    Neutral / coast in AC space: steer=0, throttle=off, brake=off.
    Policy space neutral = [0.5, 0.0, 0.0] → VJoyControl.map_action → [-1,-1,-1].
    """
    coast_policy = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    return env.control.map_action(coast_policy)


def _build_info(env) -> dict:
    """Build the info dict that NeuralPolicy.select_action() receives."""
    telem = env.telemetry.parse(env.env.state)
    info  = dict(env.env.state)
    info["telem"] = telem
    return info


# ── Warmup ────────────────────────────────────────────────────────────────────

def run_warmup(env, policy, episode: int, debug: DebugLogger) -> tuple:
    """
    Send WARMUP_FRAMES zero-control steps to populate the obs history.

    Returns (obs, info, done_during_warmup).
    """
    ac_coast       = _coast_ac_action(env)
    coast_policy   = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    obs, info, done = None, None, False

    logger.info(f"Warm-up: {WARMUP_FRAMES} zero-control frames ...")

    for i in range(WARMUP_FRAMES):
        env.env.set_actions(ac_coast)
        obs, _, done, buf_infos = env.env.step(action=None)
        env._read_latest_state()

        info        = _build_info(env)
        hist_len    = len(env.env.states)

        # Log warmup frame — action is coast (no model involved)
        debug.log(
            episode    = episode,
            step       = -(WARMUP_FRAMES - i),   # -3, -2, -1  so infer starts at 1
            phase      = "warmup",
            obs        = obs,
            action     = coast_policy,
            ac_action  = ac_coast,
            raw_state  = env.env.state,
            history_len = hist_len,
        )

        logger.info(
            f"  warmup {i + 1}/{WARMUP_FRAMES} — "
            f"history_len={hist_len}  "
            f"speed={info['telem'].speed_ms:.1f} m/s  "
            f"done={done}"
        )

        if done:
            logger.warning("Episode ended during warmup — will restart")
            break

    return obs, info, done


# ── Speed warm-up ─────────────────────────────────────────────────────────────

def run_speed_warmup(
    env,
    episode: int,
    debug: "DebugLogger",
    target_speed_ms: float,
    max_steps: int,
) -> tuple:
    """
    Apply full throttle (steer centred) until car reaches target_speed_ms or
    max_steps is exhausted.  This brings the car into the training distribution
    (BC data filtered to >= 5 m/s, mean ~32 m/s) before inference starts.

    Returns (obs, info, done_during_warmup, steps_taken).
    """
    # Full-throttle in policy space: steer=0.5 (centre), throttle=1.0, brake=0.0
    throttle_policy = np.array([0.5, 1.0, 0.0], dtype=np.float32)
    throttle_ac     = env.control.map_action(throttle_policy)

    obs, info, done = None, None, False
    steps_taken = 0

    logger.info(
        f"Speed warm-up: full throttle until {target_speed_ms:.1f} m/s "
        f"(max {max_steps} steps) ..."
    )

    for i in range(max_steps):
        env.env.set_actions(throttle_ac)
        obs, _, done, _ = env.env.step(action=None)
        env._read_latest_state()
        info = _build_info(env)
        steps_taken += 1

        speed_ms = float(info["telem"].speed_ms)

        debug.log(
            episode     = episode,
            step        = -(WARMUP_FRAMES) - i,   # continues negative count after history warmup
            phase       = "speed_warmup",
            obs         = obs,
            action      = throttle_policy,
            ac_action   = throttle_ac,
            raw_state   = env.env.state,
            history_len = len(env.env.states),
        )

        if i % 25 == 0 or speed_ms >= target_speed_ms:
            logger.info(
                f"  speed_warmup step {i + 1}/{max_steps} — "
                f"speed={speed_ms:.1f} m/s  target={target_speed_ms:.1f} m/s"
            )

        if done:
            logger.warning("Episode ended during speed warm-up — will restart")
            break

        if speed_ms >= target_speed_ms:
            logger.info(
                f"Speed warm-up complete after {steps_taken} steps — "
                f"speed={speed_ms:.1f} m/s"
            )
            break
    else:
        logger.warning(
            f"Speed warm-up: max_steps={max_steps} reached.  "
            f"speed={float(info['telem'].speed_ms):.1f} m/s (target={target_speed_ms:.1f} m/s).  "
            "Proceeding with inference anyway."
        )

    return obs, info, done, steps_taken


# ── Single episode ────────────────────────────────────────────────────────────

def run_episode(
    env,
    policy,
    episode: int,
    debug: "DebugLogger",
    speed_warmup: bool = False,
    speed_warmup_target: float = _DEFAULT_SPEED_WARMUP_TARGET,
    speed_warmup_max_steps: int = _DEFAULT_SPEED_WARMUP_MAX_STEPS,
) -> dict:
    """
    Run one inference episode with full per-frame debug logging.

    Flow per step:
      DECIDE  — policy.select_action(obs, info)    model forward pass
      LOG     — debug.log(obs, action, ac_action, raw_state)
      APPLY   — env.env.set_actions(ac_action)     non-blocking vJoy write
      READ    — env.env.step(action=None)           block ~40ms for next UDP packet
      DRAIN   — env._read_latest_state()            discard stale packets if late
    """
    logger.info(f"=== Episode {episode}: resetting car ===")
    obs, info = env.reset()

    # Clear debug CSV and write header for this episode
    debug.open_for_episode(episode)

    # ── History warm-up ────────────────────────────────────────────────────
    obs, info, warmup_done = run_warmup(env, policy, episode, debug)
    if warmup_done:
        logger.warning("Episode ended during warmup — skipping to next reset")
        return {"steps": 0, "done_reason": "warmup_terminated"}

    logger.info("History warm-up complete.")

    # ── Speed warm-up (optional) ────────────────────────────────────────────
    if speed_warmup:
        obs, info, speed_warmup_done, speed_warmup_steps = run_speed_warmup(
            env,
            episode,
            debug,
            target_speed_ms = speed_warmup_target,
            max_steps       = speed_warmup_max_steps,
        )
        if speed_warmup_done:
            logger.warning("Episode ended during speed warm-up — skipping to next reset")
            return {"steps": 0, "done_reason": "speed_warmup_terminated"}
        logger.info(
            f"Speed warm-up done ({speed_warmup_steps} steps) — inference started"
        )
    else:
        logger.info("Inference started (no speed warm-up)")

    # ── Inference loop ─────────────────────────────────────────────────────
    step = 0
    done = False

    while not done:
        t0 = time.perf_counter()

        # ── 1. DECIDE ──────────────────────────────────────────────────────
        #
        # obs : np.ndarray (125,)  — the exact vector built by ac_env.get_obs():
        #   UDP packet   → client.state (raw telemetry from plugin)
        #   expand_state → gap, out_of_track, sensors (ray-cast), LapDist
        #   get_obs      → normalise + curvature look-ahead + 3-frame history
        #
        # NeuralPolicy.select_action():
        #   obs → GaussianPolicy → tanh(mean) in [-1,1] → remap → [0,1]
        #
        action = np.array(policy.select_action(obs, info), dtype=np.float32)
        t1 = time.perf_counter()

        # ── 2. MAP action to AC space and log BEFORE sending ───────────────
        #
        # ac_action = policy_action * 2 - 1  (VJoyControl.map_action)
        #   policy=0.5 → ac=0.0  (steer centre / coast)
        #   policy=0.0 → ac=-1.0 (full left / no throttle)
        #   policy=1.0 → ac=+1.0 (full right / full throttle)
        #
        ac_action = env.control.map_action(action)

        # Log obs + action BEFORE applying (so we see what the model decided
        # given the obs it received this frame).
        # action[0] here is AFTER invert_steer (if enabled).
        # steer_raw = action*2-1 in [-1,1]; negative = left, positive = right.
        # When gap_m > 0 (car RIGHT of line): steer_raw should be NEGATIVE (left).
        debug.log(
            episode     = episode,
            step        = step + 1,
            phase       = "infer",
            obs         = obs,
            action      = action,
            ac_action   = ac_action,
            raw_state   = env.env.state,
            history_len = len(env.env.states),
        )

        # ── 3. APPLY — send to vJoy immediately (non-blocking) ─────────────
        env.env.set_actions(ac_action)
        t2 = time.perf_counter()

        # ── 4. READ — block for the next UDP physics packet (~40 ms) ────────
        obs, _, done, buf_infos = env.env.step(action=None)

        # Drain any extra packets that queued while the model was running.
        env._read_latest_state()
        t3 = time.perf_counter()

        # ── 5. Build info for next iteration ───────────────────────────────
        info = _build_info(env)

        step += 1

        # Periodic terminal log — 1 line/second
        if step % LOG_EVERY == 1:
            decide_ms = (t1 - t0) * 1000
            apply_ms  = (t2 - t1) * 1000
            wait_ms   = (t3 - t2) * 1000
            nsp       = float(env.env.state.get("NormalizedSplinePosition", 0.0))
            logger.info(
                f"ep={episode:3d} step={step:4d} | "
                f"decide={decide_ms:.1f}ms apply={apply_ms:.1f}ms wait={wait_ms:.1f}ms | "
                f"speed={info['telem'].speed_ms:.1f} m/s  nsp={nsp:.3f}  "
                f"history={len(env.env.states)} | "
                f"steer={action[0]:.3f}  throttle={action[1]:.3f}  brake={action[2]:.3f}"
            )

    logger.info(f"Episode {episode} done — {step} steps.  Debug log: {DEBUG_LOG_PATH}")
    return {"steps": step, "done_reason": "terminated"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live inference with a trained Assetto Corsa policy"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help=(
            "Path to model checkpoint:\n"
            "  New format (train_bc.py): .pt with 'state_dict' + 'config' keys\n"
            "  Legacy SAC: policy_net.pth (raw state_dict, obs_dim auto-inferred)"
        ),
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip preflight checks",
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'assetto_corsa_gym', 'config.yml',
        ),
        help="Path to config.yml (default: assetto_corsa_gym/config.yml)",
    )
    parser.add_argument(
        "--steer-deadzone", type=float, default=0.0, metavar="DZ",
        help="Steer dead-zone half-width in [-1,1] SAC space",
    )
    parser.add_argument(
        "--invert-steer", action="store_true",
        help=(
            "Enable steer inversion (OFF by default). "
            "The BC model (bc_monza_human_v1) learned the correct sign directly from "
            "human demonstrations: gap<0 (car LEFT) -> positive SAC steer (RIGHT). "
            "Inversion is wrong for this model and causes the car to drift further off-line. "
            "Only enable if your checkpoint was trained with an inverted steer convention."
        ),
    )
    parser.add_argument(
        "--speed-warmup", action="store_true",
        help=(
            "After history warm-up, apply full throttle until the car reaches "
            "--speed-warmup-target m/s (or --speed-warmup-max-steps).  "
            "Fixes the training distribution mismatch: BC data was recorded at "
            ">=5 m/s (mean 32 m/s) but reset() starts from rest (~0 m/s)."
        ),
    )
    parser.add_argument(
        "--speed-warmup-target", type=float, default=_DEFAULT_SPEED_WARMUP_TARGET,
        metavar="M/S",
        help=(
            f"Target speed in m/s before inference starts (default: {_DEFAULT_SPEED_WARMUP_TARGET}).  "
            "Ignored unless --speed-warmup is set."
        ),
    )
    parser.add_argument(
        "--speed-warmup-max-steps", type=int, default=_DEFAULT_SPEED_WARMUP_MAX_STEPS,
        metavar="N",
        help=(
            f"Safety cap on speed warm-up steps (default: {_DEFAULT_SPEED_WARMUP_MAX_STEPS}).  "
            "If speed threshold is not reached, inference proceeds anyway.  "
            "Ignored unless --speed-warmup is set."
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Stop after N episodes (default: run forever until Ctrl+C)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 1. Preflight ──────────────────────────────────────────────────────────
    if not args.skip_preflight:
        from preflight import run_preflight
        run_preflight(warn_only=False)
    else:
        logger.warning("Preflight skipped — make sure AC is running in a session")

    # ── 2. Config ─────────────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    assert hasattr(cfg, "OurEnv"), "config.yml missing OurEnv block"

    our_env_cfg = OmegaConf.create({
        "our_env": OmegaConf.to_container(cfg.OurEnv, resolve=True)
    })

    # ── 3. Build ACEnv ────────────────────────────────────────────────────────
    from AssettoCorsaEnv.assettoCorsa import make_ac_env

    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'outputs', 'infer',
    )
    os.makedirs(work_dir, exist_ok=True)

    logger.info("Building AssettoCorsaEnv ...")
    ac_env_instance = make_ac_env(cfg=cfg, work_dir=work_dir)
    logger.info(f"ACEnv built. obs_dim={ac_env_instance.state_dim}  action_dim={ac_env_instance.action_dim}")

    # ── 4. Build OurEnv ───────────────────────────────────────────────────────
    from our_env import OurEnv
    env = OurEnv(ac_env_instance, our_env_cfg)
    logger.info("OurEnv ready.")

    # ── 5. Load policy ────────────────────────────────────────────────────────
    from policies.neural import NeuralPolicy

    # Steer inversion is OFF by default.
    # The BC model learned the correct sign directly from human demonstrations:
    #   gap < 0 (car LEFT)  -> positive SAC steer (RIGHT)  [Correlation = -0.24]
    # Inverting flips the correct correction to the wrong direction, causing drift.
    # Use --invert-steer only for checkpoints trained with an inverted convention.
    invert_steer = args.invert_steer
    logger.info(f"Steer inversion: {'ON' if invert_steer else 'OFF'}")

    policy = NeuralPolicy.from_checkpoint(
        model_path      = args.checkpoint,
        deterministic   = True,
        steer_deadzone  = args.steer_deadzone,
        invert_steer    = invert_steer,
    )
    logger.info(f"Policy loaded: {args.checkpoint}")

    # ── 6. Build debug logger ─────────────────────────────────────────────────
    debug = DebugLogger(path=DEBUG_LOG_PATH, obs_dim=ac_env_instance.state_dim)
    logger.info(f"Debug log path: {DEBUG_LOG_PATH}  (cleared at each episode start)")

    # ── 7. Episode loop ───────────────────────────────────────────────────────
    if args.speed_warmup:
        logger.info(
            f"Speed warm-up: ON — target={args.speed_warmup_target:.1f} m/s  "
            f"max_steps={args.speed_warmup_max_steps}"
        )
    else:
        logger.info("Speed warm-up: OFF (use --speed-warmup to enable)")

    logger.info(
        "Starting inference. "
        f"episodes={'inf' if args.episodes is None else args.episodes}  "
        "Press Ctrl+C to stop."
    )

    episode     = 0
    total_steps = 0

    try:
        while args.episodes is None or episode < args.episodes:
            stats        = run_episode(
                env,
                policy,
                episode,
                debug,
                speed_warmup            = args.speed_warmup,
                speed_warmup_target     = args.speed_warmup_target,
                speed_warmup_max_steps  = args.speed_warmup_max_steps,
            )
            total_steps += stats["steps"]
            logger.info(
                f"Episode {episode} complete — "
                f"steps={stats['steps']}  total_steps={total_steps}"
            )
            episode += 1

    except KeyboardInterrupt:
        logger.info(
            f"\nStopped by user.  Episodes: {episode}  Total steps: {total_steps}"
        )
    finally:
        debug.close()
        policy.close()
        env.close()
        logger.info("vJoy released. Done.")


if __name__ == "__main__":
    main()
