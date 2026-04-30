"""
OurEnv — Clean wrapper around AssettoCorsaEnv.

Architecture:
    assetto_corsa_gym (UNTOUCHED — communication layer)
            ↓
    OurEnv  (this file — wires telemetry, reward, control modules)
            ↓
    telemetry/   rewards/   controls/   policies/

Do NOT modify any file inside assetto_corsa_gym/.

Usage:
    from our_env import OurEnv, save_episode, collect_and_save

    env = OurEnv(ac_env, cfg)
    trajectory = env.collect_episode(policy=my_policy)
    save_episode(trajectory, output_dir="collected_data/", episode_number=0)
"""

import os
import time
import json
import atexit
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

from telemetry.ac import ACTelemetry
from rewards.composite import CompositeReward
from rewards.reward_logger import RewardLogger
from controls.vjoy import VJoyControl

logger = logging.getLogger(__name__)

CRASH_BRAKE_DURATION_S = 5.0
CRASH_CONTROL_HZ = 25

# Scale factor applied to ref_lap target speeds at runtime.
# 0.9 = agent targets 90% of the racing line speed (conservative margin).
# Original ref_lap data is never modified.
TARGET_SPEED_SCALE = 0.9


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


# ---------------------------------------------------------------------------
# OurEnv
# ---------------------------------------------------------------------------

class OurEnv:
    """
    Wraps AssettoCorsaEnv. Provides:
      - Modular telemetry (ACTelemetry)
      - Modular, composable reward (CompositeReward)
      - Modular control mapping (VJoyControl)
      - collect_episode() for offline data collection
      - Standard gym interface: reset(), step(), close()

    Parameters
    ----------
    ac_env : AssettoCorsaEnv
        Already-constructed ACEnv instance.
    cfg : OmegaConf DictConfig
        Must contain an `our_env` sub-config. Supports two reward formats:

        Legacy (reward_weights block):
            our_env:
              reward_weights:
                w1_progress: 1.0
                w2_speed:    0.5
                w3_gap:      1.0
                w4_smoothness: 0.1
                w5_crash:    10.0

        Modular (reward list):
            our_env:
              reward:
                - type: ProgressReward
                  weight: 1.0
                - type: GapReward
                  weight: 1.0
                  negate: true
    """

    def __init__(self, ac_env, cfg):
        self.env = ac_env
        self.cfg = cfg
        our = cfg.our_env

        # -- Telemetry --
        self.telemetry = ACTelemetry()

        # -- Reward (support both config formats) --
        if hasattr(our, "reward_weights"):
            self.reward = CompositeReward.from_weights(our.reward_weights)
        elif hasattr(our, "reward"):
            self.reward = CompositeReward.from_config(our.reward)
        else:
            raise ValueError("cfg.our_env must have 'reward_weights' or 'reward'")

        # -- Reward truth logger --
        self.reward_logger = RewardLogger()

        # -- Control --
        self.control = VJoyControl(ac_client=self.env.client)

        # -- Collection config --
        self.data_output_path = str(our.data_output_path)
        self.episodes_per_run = int(our.episodes_per_run)

        # -- Track info (fetched once from plugin before UDP session starts) --
        # This is available to policies that need the racing line / speed profile
        # (e.g. MathPolicy). Fetched here so the TCP call doesn't interfere with
        # the UDP step loop that starts on first reset().
        try:
            self.track_info = self.env.client.simulation_management.get_track_info()
            logger.info(
                f"Track info loaded: {len(self.track_info.get('fast_lane', []))} racing-line points"
            )
        except Exception as e:
            self.track_info = None
            logger.warning(f"Could not fetch track info from plugin: {e}")

        # -- Gym pass-throughs --
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # -- Per-episode state --
        self._prev_lap_dist: float = 0.0
        self._prev_speed_ms: float = 0.0
        self._prev_action: np.ndarray = np.zeros(3, dtype=np.float32)
        self._pending_action: np.ndarray = np.zeros(3, dtype=np.float32)
        self._off_track_count: int = 0
        self._track_length: Optional[float] = None

        # -- Logging counters (used by step() to populate log_row context) --
        self._episode_number: int = 0
        self._step_in_episode: int = 0
        self._total_env_steps: int = 0

        # Register vJoy neutralization on any exit (Ctrl+C, crash, normal)
        atexit.register(self._neutralize_vjoy)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)

        if self._track_length is None and hasattr(self.env, "track_length"):
            self._track_length = float(self.env.track_length)

        self._prev_lap_dist = self._get_lap_dist()
        self._prev_speed_ms = 0.0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._off_track_count = 0
        self._step_in_episode = 0
        self._episode_number += 1
        self.reward.reset()

        # Build initial info from the state already read by ac_env.reset()
        # so the first policy.select_action() sees real telemetry, not zero defaults.
        info = dict(self.env.state)
        telem0 = self.telemetry.parse(self.env.state)
        info["telem"] = telem0

        return obs, info

    def set_actions(self, action: np.ndarray) -> None:
        """
        Send action to AC immediately (non-blocking).

        Expects action in AC space [-1, 1]^3 (SAC tanh output — no remapping).
        Stashes the action so the subsequent step(action=None) call can
        correctly compute the reward.

        This is the first half of the latency-overlap split:
            env.set_actions(action)   # fires immediately — ~0.1 ms
            # ... CPU work overlaps with AC's 40 ms physics tick ...
            env.step(action=None)     # waits for next UDP frame
        """
        self.env.set_actions(action)
        # Stash for the reward computation in the matching step(action=None) call.
        self._pending_action = np.array(action, dtype=np.float32)

    def step(self, action: Optional[np.ndarray]):
        """
        action : np.ndarray (3,) in AC space [-1,1] (SAC tanh output), or None
            [steering, throttle, brake]

        When action is None the caller must have already called set_actions()
        for this step.  step() will then only block waiting for the next UDP
        frame — it will NOT send any action to AC.  This is the second half of
        the latency-overlap split (see set_actions() docstring).

        When action is a numpy array the call is equivalent to the original
        non-overlap behaviour: map → send → wait → reward.

        Returns (obs, total_reward, done, info)
        info["reward_components"] → dict of named component values
        """
        if action is None:
            # Overlap path: action was already sent via set_actions().
            # Retrieve the stashed policy-space action for reward computation.
            effective_action = self._pending_action
            obs, _their_reward, done, info = self.env.step(action=None)
        else:
            # Normal (non-overlap) path: send and wait in one call.
            effective_action = np.array(action, dtype=np.float32)
            obs, _their_reward, done, info = self.env.step(effective_action)

        telem = self.telemetry.parse(self.env.state)
        if (self.env.ref_lap is not None
                and getattr(self.env.ref_lap, 'use_target_speed', False)):
            telem.target_speed_ms = float(
                self.env.ref_lap.get_target_speed_value(telem.lap_dist)
            ) * TARGET_SPEED_SCALE
        if self.env.ref_lap is not None:
            try:
                import numpy as _np, math as _math
                _yaw_idx  = self.env.ref_lap.channels_dist.index("yaw")
                _line_yaw = float(_np.interp(
                    telem.lap_dist,
                    self.env.ref_lap.distance_ch_dist,
                    self.env.ref_lap.td[:, _yaw_idx],
                ))
                _car_yaw  = float(self.env.state.get("yaw", 0.0))
                telem.yaw_error_rad = (_car_yaw - _line_yaw + _math.pi) % (2 * _math.pi) - _math.pi
            except Exception:
                pass
        result = self.reward.compute(
            telem, effective_action, self._prev_action,
            self._prev_lap_dist,
            self._track_length or 5793.0,
            prev_speed_ms=self._prev_speed_ms,
        )

        if telem.out_of_track:
            self._off_track_count += 1

        self._step_in_episode += 1
        self._total_env_steps += 1

        # Capture pre-update values for log before state is advanced
        _log_prev_lap_dist = self._prev_lap_dist
        _log_prev_action   = self._prev_action

        self._prev_action = effective_action
        self._prev_lap_dist = telem.lap_dist
        self._prev_speed_ms = telem.speed_ms

        # -- Reward truth log --
        gap_m_abs = abs(telem.gap_m)
        action_delta_norm = float(np.linalg.norm(
            effective_action - _log_prev_action, ord=2,
        ))
        log_row = result["log_row"]
        log_row.update({
            "run_id":            0,
            "phase":             0,
            "episode":           self._episode_number,
            "step":              self._step_in_episode,
            "total_env_steps":   self._total_env_steps,
            "speed_ms":          float(telem.speed_ms),
            "lap_dist":          float(telem.lap_dist),
            "gap_m":             float(gap_m_abs),
            "progress_delta_m":  float(telem.lap_dist - _log_prev_lap_dist),
            "out_of_track":      int(telem.out_of_track),
            "action_steer":      float(effective_action[0]),
            "action_throttle":   float(effective_action[1]),
            "action_brake":      float(effective_action[2]),
            "action_delta_norm": action_delta_norm,
        })
        self.reward_logger.push(log_row)
        if done:
            self.reward_logger.flush()

        info["reward_components"] = result["components"]
        info["reward_metrics"]    = result["metrics"]
        info["telem"] = telem
        # Expose raw state so policies (e.g. MathPolicy) can read telemetry fields
        # like NormalizedSplinePosition, world_position_x/y, yaw, speed.
        # buf_infos from ac_env.step() only has 'terminated'/'TimeLimit.truncated'.
        info.update(self.env.state)

        return obs, float(result["total"]), done, info

    def close(self):
        self._neutralize_vjoy()
        return self.env.close()

    # ------------------------------------------------------------------
    # Crash handler
    # ------------------------------------------------------------------

    def handle_crash(self) -> str:
        """
        Called after out-of-track termination.

        1. Full brakes + zero steer for CRASH_BRAKE_DURATION_S seconds
        2. Release to neutral
        3. Return 'crash_handled'
        """
        n_ticks = int(CRASH_BRAKE_DURATION_S * CRASH_CONTROL_HZ)
        tick_s = 1.0 / CRASH_CONTROL_HZ
        controls = self.env.client.controls

        logger.info(
            f"Crash handler: braking for {CRASH_BRAKE_DURATION_S}s ({n_ticks} ticks)"
        )
        logger.info(
            f"  vJoy BEFORE: steer={controls.get('steer', '?'):.3f}"
            f"  acc={controls.get('acc', '?'):.3f}"
            f"  brake={controls.get('brake', '?'):.3f}"
        )

        # Phase 1 — full brake
        for _ in range(n_ticks):
            self.control.execute_direct(np.array([0.0, -1.0, 1.0], dtype=np.float32))
            self.env.client.respond_to_server()
            time.sleep(tick_s)

        logger.info(
            f"  vJoy BRAKING: steer={controls.get('steer', '?'):.3f}"
            f"  acc={controls.get('acc', '?'):.3f}"
            f"  brake={controls.get('brake', '?'):.3f}"
        )

        # Phase 2 — neutral
        self.control.neutralize()
        self.env.client.respond_to_server()

        logger.info(
            f"  vJoy NEUTRAL: steer={controls.get('steer', '?'):.3f}"
            f"  acc={controls.get('acc', '?'):.3f}"
            f"  brake={controls.get('brake', '?'):.3f}"
        )
        logger.info("Crash handler: complete. crash_handled")
        return "crash_handled"

    # ------------------------------------------------------------------
    # vJoy neutralization on exit
    # ------------------------------------------------------------------

    def _neutralize_vjoy(self):
        """
        Set vJoy axes to true neutral then relinquish the device.

        Axis math (car_control.py, SCALE=16384):
            steer neutral (AC steer=0):  wAxisX = int(1.0 * 16384) = 16384
            throttle neutral (AC acc=-1): wAxisY = int(0.0 * 2 * 16384) = 0
            brake neutral (AC brake=-1):  wAxisZ = int(0.0 * 2 * 16384) = 0

        Vjoy.ini: THROTTLE/BRAKES have MIN=0, MAX=1 (not center-based).
        Axis=0 = pedal released = no input. There is no inversion.
        Setting wAxisY=0 and wAxisZ=0 before relinquish means the snap-to-0
        on device release causes no change to throttle or brake.
        """
        try:
            controls = self.env.client.controls

            if not hasattr(controls, "local_controls"):
                logger.info("vJoy executed by server — no local relinquish needed")
                return

            # Step 1 — send AC-space neutral through normal path
            controls.set_controls(steer=0.0, acc=-1.0, brake=-1.0)
            controls.apply_local_controls()
            time.sleep(0.1)

            # Step 2 — write neutral directly to vJoy DLL
            #   wAxisX=16384 (steer center)
            #   wAxisY=0     (throttle off — 0 = no gas in WHEEL/JOYSTICK mode)
            #   wAxisZ=0     (brake off)
            vj = controls.local_controls.vj
            STEER_CENTER = 16384
            PEDAL_OFF = 0
            joy_pos = vj.generateJoystickPosition(
                wAxisX=STEER_CENTER,
                wAxisY=PEDAL_OFF,
                wAxisZ=PEDAL_OFF,
            )
            vj.update(joy_pos)
            logger.info(
                f"vJoy axes set to neutral: "
                f"wAxisX={STEER_CENTER} wAxisY={PEDAL_OFF} wAxisZ={PEDAL_OFF}"
            )
            time.sleep(0.2)

            # Step 3 — relinquish (axes snap to 0, but throttle/brake already 0)
            controls.local_controls.close()
            logger.info("vJoy relinquished cleanly — no runaway throttle")

        except Exception as e:
            logger.warning(f"vJoy neutralize failed: {e}")

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _throttle_warmup(self, duration_s: float) -> tuple:
        """
        Apply full throttle for duration_s seconds before inference starts.

        Sends policy-space [steer=0.5, throttle=1.0, brake=0.0] at the normal
        40ms tick rate so the car builds speed into a range the model was
        trained on (min_speed_ms=5 m/s filter in preprocess_bc.py).

        Returns (obs, info) from the freshest packet after the warmup ends —
        ready to feed straight into policy.select_action().
        """
        warmup_action = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        ac_action = self.control.map_action(warmup_action)
        n_ticks = max(1, int(duration_s * 25))  # 25 Hz ≈ 40ms per step

        logger.info(f"Throttle warm-up: {duration_s:.1f}s ({n_ticks} ticks) ...")
        obs, info = None, None
        for _ in range(n_ticks):
            self.env.set_actions(ac_action)
            next_obs, _r, _done, buf_infos = self.env.step(action=None)
            self._read_latest_state()

            telem = self.telemetry.parse(self.env.state)
            self._prev_lap_dist = telem.lap_dist
            self._prev_speed_ms = telem.speed_ms

            info = dict(buf_infos)
            info["telem"] = telem
            info.update(self.env.state)
            obs = next_obs

        speed = self.telemetry.parse(self.env.state).speed_ms
        logger.info(f"Warm-up done — speed={speed:.1f} m/s")
        return obs, info

    def collect_episode(
        self,
        policy: Optional[Callable] = None,
        step_logger=None,
        episode_number: int = 0,
        throttle_warmup_s: float = 0.0,
    ) -> dict:
        """
        Run one full episode and return a trajectory dict.

        Parameters
        ----------
        policy : BasePolicy or callable(obs, info → action[0,1]³), optional
            Defaults to ZeroPolicy (car sits still) for smoke testing.

        Returns
        -------
        dict with keys:
            observations      np.ndarray (T, obs_dim)
            actions           np.ndarray (T, 3)
            rewards           np.ndarray (T,)
            reward_components dict of np.ndarray each (T,)
            metadata          dict
        """
        from policies.simple import ZeroPolicy
        if policy is None:
            policy = ZeroPolicy()

        obs, info = self.reset()  # also calls self.reward.reset() internally
        if hasattr(policy, "reset"):
            policy.reset()

        if throttle_warmup_s > 0.0:
            obs, info = self._throttle_warmup(throttle_warmup_s)

        self.reward.reset()  # reset speed accumulator at true episode start (after warmup)
        obs_list, action_list, reward_list = [], [], []
        comp_names = self.reward.component_names()
        comp_lists = {k: [] for k in comp_names}
        max_speed = 0.0
        last_components = {k: 0.0 for k in comp_names}

        step_times_ms = []   # total ms per step
        LOG_EVERY = 25       # log one line per second at 25Hz

        done = False
        while not done:
            # ── 1. DECIDE ─────────────────────────────────────────────────
            # obs and info are already the current packet (from reset or
            # the previous iteration's READ phase).
            t0 = time.perf_counter()
            action = np.array(policy.select_action(obs, info), dtype=np.float32)
            t1 = time.perf_counter()

            # ── 2. APPLY ──────────────────────────────────────────────────
            # Write to vJoy immediately — this is ~0.1ms, not 40ms.
            ac_action = self.control.map_action(action)
            self.env.set_actions(ac_action)
            t2 = time.perf_counter()

            # ── 3. READ next packet ───────────────────────────────────────
            # Block ~40ms waiting for AC's next UDP physics frame.
            # action=None skips set_actions() — we already applied above.
            next_obs, _their_reward, done, buf_infos = self.env.step(action=None)

            # Drain any packets that queued while policy was deciding.
            # If inference was slow (>40ms), env.state is now the freshest
            # available packet instead of the first stale one in the buffer.
            self._read_latest_state()
            t3 = time.perf_counter()

            # ── 4. Reward + telemetry on the new state ────────────────────
            telem = self.telemetry.parse(self.env.state)
            if (self.env.ref_lap is not None
                    and getattr(self.env.ref_lap, 'use_target_speed', False)):
                telem.target_speed_ms = float(
                    self.env.ref_lap.get_target_speed_value(telem.lap_dist)
                ) * TARGET_SPEED_SCALE
            if self.env.ref_lap is not None:
                try:
                    import numpy as _np, math as _math
                    _yaw_idx  = self.env.ref_lap.channels_dist.index("yaw")
                    _line_yaw = float(_np.interp(
                        telem.lap_dist,
                        self.env.ref_lap.distance_ch_dist,
                        self.env.ref_lap.td[:, _yaw_idx],
                    ))
                    _car_yaw  = float(self.env.state.get("yaw", 0.0))
                    telem.yaw_error_rad = (_car_yaw - _line_yaw + _math.pi) % (2 * _math.pi) - _math.pi
                except Exception:
                    pass
            result = self.reward.compute(
                telem, action, self._prev_action,
                self._prev_lap_dist,
                self._track_length or 5793.0,
                prev_speed_ms=self._prev_speed_ms,
            )
            if telem.out_of_track:
                self._off_track_count += 1

            # Capture pre-update values for log before state is advanced
            _cep_prev_lap_dist = self._prev_lap_dist
            _cep_prev_action   = self._prev_action

            self._prev_action = np.array(action, dtype=np.float32)
            self._prev_lap_dist = telem.lap_dist
            self._prev_speed_ms = telem.speed_ms

            # -- Reward truth log (collect_episode path) --
            _cep_step = len(obs_list) + 1  # 1-based; obs_list not yet appended
            self._total_env_steps += 1
            _cep_gap_m_abs = abs(telem.gap_m)
            _cep_action_delta = float(np.linalg.norm(
                np.array(action, dtype=np.float32) - _cep_prev_action, ord=2,
            ))
            cep_log_row = result["log_row"]
            cep_log_row.update({
                "run_id":            0,
                "phase":             0,
                "episode":           episode_number,
                "step":              _cep_step,
                "total_env_steps":   self._total_env_steps,
                "speed_ms":          float(telem.speed_ms),
                "lap_dist":          float(telem.lap_dist),
                "gap_m":             float(_cep_gap_m_abs),
                "progress_delta_m":  float(telem.lap_dist - _cep_prev_lap_dist),
                "out_of_track":      int(telem.out_of_track),
                "action_steer":      float(action[0]),
                "action_throttle":   float(action[1]),
                "action_brake":      float(action[2]),
                "action_delta_norm": _cep_action_delta,
            })
            self.reward_logger.push(cep_log_row)
            if done:
                self.reward_logger.flush()

            info = dict(buf_infos)
            info["reward_components"] = result["components"]
            info["reward_metrics"]    = result["metrics"]
            info["telem"] = telem
            info.update(self.env.state)
            reward = float(result["total"])

            decide_ms = (t1 - t0) * 1000
            apply_ms  = (t2 - t1) * 1000
            wait_ms   = (t3 - t2) * 1000
            total_ms  = (t3 - t0) * 1000
            step_times_ms.append(total_ms)

            obs_list.append(obs.copy())
            action_list.append(action.copy())
            reward_list.append(reward)

            components = info.get("reward_components", {})
            last_components = components
            for k in comp_names:
                comp_lists[k].append(components.get(k, 0.0))

            if telem.speed_ms > max_speed:
                max_speed = telem.speed_ms

            # Per-step decision log
            if step_logger is not None:
                step_logger.log(
                    episode=episode_number,
                    step=len(obs_list),
                    info=info,
                    action=action,
                    decision_ms=decide_ms,
                    apply_ms=apply_ms,
                    reward=reward,
                    policy=policy,
                )

            # Periodic step log — one line per second
            step_n = len(obs_list)
            if step_n % LOG_EVERY == 1:
                nsp = float(info.get("NormalizedSplinePosition", 0.0))
                gap = telem.gap_m
                logger.info(
                    f"step={step_n:4d} | "
                    f"decide={decide_ms:5.2f}ms  apply={apply_ms:4.2f}ms  wait={wait_ms:5.1f}ms | "
                    f"speed={telem.speed_ms:5.1f}m/s  nsp={nsp:.3f}  gap={gap:+.2f}m | "
                    f"steer={action[0]:.3f}  throttle={action[1]:.3f}  brake={action[2]:.3f}"
                )

            obs = next_obs

        # Handle crash — brake before next reset
        if last_components.get("r_crash", 0.0) > 0.0:
            self.handle_crash()

        T = len(obs_list)

        # Lap time from ACEnv stats
        lap_time_ms = None
        if hasattr(self.env, "episodes_stats") and self.env.episodes_stats:
            last = self.env.episodes_stats[-1]
            best_s = last.get("ep_bestLapTime", 0)
            if best_s and best_s > 0:
                lap_time_ms = best_s * 1000.0

        times = np.array(step_times_ms, dtype=np.float32)
        logger.info(
            f"Episode done: {T} steps | "
            f"step_time avg={times.mean():.1f}ms  max={times.max():.1f}ms  "
            f"min={times.min():.1f}ms  (budget=40ms)"
        )

        return {
            "observations":     np.array(obs_list,    dtype=np.float32),
            "actions":          np.array(action_list, dtype=np.float32),
            "rewards":          np.array(reward_list, dtype=np.float32),
            "reward_components": {
                k: np.array(v, dtype=np.float32) for k, v in comp_lists.items()
            },
            "step_times_ms":    times,
            "metadata": {
                "episode_steps":    T,
                "total_reward":     float(np.sum(reward_list)),
                "lap_time_ms":      lap_time_ms,
                "off_track_count":  self._off_track_count,
                "max_speed_ms":     max_speed,
                "step_time_avg_ms": float(times.mean()),
                "step_time_max_ms": float(times.max()),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_latest_state(self) -> int:
        """
        After a blocking step_sim() returns, drain any extra packets that
        queued up while the policy was running inference.

        If policy inference took > 40ms, one or more packets are already
        waiting in the OS UDP buffer. Reading only the first one means the
        next decision is based on stale state. This method discards all but
        the newest packet so every decision always sees the freshest state.

        Returns the number of stale packets discarded (0 = we were on time).
        """
        sock = self.env.client.socket
        if sock is None:
            return 0

        from AssettoCorsaEnv.ac_client import MAX_MSG_SIZE
        original_timeout = sock.gettimeout()
        sock.settimeout(0)           # non-blocking — only read what's already there

        discarded = 0
        latest_raw = None
        try:
            while True:
                data, _ = sock.recvfrom(MAX_MSG_SIZE)
                latest_raw = data    # keep overwriting — last one is freshest
                discarded += 1
        except Exception:
            pass                     # buffer empty

        sock.settimeout(original_timeout)

        if latest_raw is not None:
            # Re-parse and re-expand the freshest packet into env.state
            self.env.client.state.parse_server_str(latest_raw.decode())
            fresh_raw = self.env.client.state.copy()
            self.env.state, _ = self.env.expand_state(fresh_raw)
            if discarded > 1:
                logger.debug(f"Drained {discarded} stale packets — policy was late")

        return discarded

    def _get_lap_dist(self) -> float:
        if hasattr(self.env, "state") and self.env.state:
            return float(self.env.state.get("LapDist", 0.0))
        return 0.0


# ---------------------------------------------------------------------------
# Storage helpers (module-level — imported by test_our_env.py and collect.py)
# ---------------------------------------------------------------------------

def save_episode(
    trajectory: dict,
    output_dir: str,
    episode_number: int = 0,
) -> str:
    """
    Save one trajectory to:
      <output_dir>/episode_NNNNN_<timestamp>.parquet  — one row per step
      <output_dir>/episodes_metadata.jsonl            — one JSON line appended
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    parquet_name = f"episode_{episode_number:05d}_{ts}.parquet"
    parquet_path = output_dir / parquet_name

    observations = trajectory["observations"]
    actions = trajectory["actions"]
    rewards = trajectory["rewards"]
    components = trajectory["reward_components"]
    T = len(rewards)

    data = {"step": np.arange(T, dtype=np.int32)}

    for i in range(observations.shape[1] if observations.ndim == 2 else 0):
        data[f"obs_{i}"] = observations[:, i]

    for i in range(actions.shape[1] if actions.ndim == 2 else 0):
        data[f"action_{i}"] = actions[:, i]

    data["reward"] = rewards
    for k, v in components.items():
        data[k] = v

    pd.DataFrame(data).to_parquet(str(parquet_path), engine="pyarrow", index=False)
    logger.info(f"Saved episode parquet: {parquet_path}")

    metadata_path = output_dir / "episodes_metadata.jsonl"
    meta = dict(trajectory["metadata"])
    meta["episode_number"] = episode_number
    meta["parquet_file"] = parquet_name
    meta["timestamp"] = ts
    # JSON-serialise numpy scalars
    meta = {
        k: (int(v) if isinstance(v, np.integer) else
            float(v) if isinstance(v, np.floating) else v)
        for k, v in meta.items()
    }

    with open(str(metadata_path), "a") as f:
        f.write(json.dumps(meta) + "\n")
    logger.info(f"Appended metadata: {metadata_path}")

    return str(parquet_path)


def collect_and_save(
    env: OurEnv,
    policy=None,
    episode_start: int = 0,
    step_logger=None,
    throttle_warmup_s: float = 0.0,
) -> list:
    """Run env.episodes_per_run episodes, save each, return list of parquet paths."""
    paths = []
    for i in range(env.episodes_per_run):
        ep_num = episode_start + i
        logger.info(f"Collecting episode {ep_num} ...")
        trajectory = env.collect_episode(
            policy=policy,
            step_logger=step_logger,
            episode_number=ep_num,
            throttle_warmup_s=throttle_warmup_s,
        )
        path = save_episode(trajectory, env.data_output_path, episode_number=ep_num)
        paths.append(path)
        if step_logger is not None:
            step_logger.end_episode()
        meta = trajectory["metadata"]
        logger.info(
            f"Episode {ep_num}: steps={meta['episode_steps']}"
            f"  reward={meta['total_reward']:.3f}"
            f"  off_track={meta['off_track_count']}"
            f"  max_speed={meta['max_speed_ms']:.1f} m/s"
        )
    return paths
