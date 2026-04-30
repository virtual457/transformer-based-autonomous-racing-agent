"""
agent.py — SACAgent: phase-based training loop for Assetto Corsa.

Phase structure:
    Each phase: collect with SAC policy, then train for train_steps_per_phase steps.

Checkpoints are saved every checkpoint_freq phases to checkpoint_dir.

The env interface expected here:
    env.reset()                -> (obs, info)
    env.set_actions(action)    -> None   (non-blocking; maps policy→AC space and fires)
    env.step(action)           -> (obs, reward, done, info)   normal path
    env.step(action=None)      -> (obs, reward, done, info)   overlap path (action pre-sent)
    action space:                [-1, 1]^3 (policy space, mapped inside OurEnv)

Latency-overlap trick (collect_phase):
    obs = next_obs               # frame N has arrived
    action = select_action(obs)  # infer on frame N first — produces the correct action
    env.set_actions(action)      # send to AC immediately (~0.1 ms)
    replay_buffer.push(...)      # CPU work overlaps with AC's 40 ms physics tick
    env.step(action=None)        # block only for the remaining tick budget

Inference MUST happen before set_actions so that AC receives the action computed
from the frame that just arrived (frame N), not the stale action computed from
frame N-1.

Step 0 of each episode uses the normal env.step(action) because there is no
previously-computed action to pre-send.  The overlap pattern begins from step 1.
"""

import os
import time
import random
import logging
import collections
import math
from typing import Optional

import numpy as np

from .sac import SAC
from .replay_buffer import ReplayBuffer, DualReplayBuffer

logger = logging.getLogger(__name__)


_EXACT_REWARD_TOL = 1e-4  # tolerance for exact-value buckets (+1.0 / -1.0)

STEP_INTERVAL_S = 0.009  # 9 ms — target minimum action interval (matches finetune_sac.py)


def _reward_histogram(rewards: list) -> tuple:
    """Compute a 10-bin reward histogram over [-1.0, 1.0].

    Each bin spans 0.2.  Values outside [-1, 1] are clamped into the nearest
    boundary bin so that all frames are accounted for.

    Additionally, frames whose reward is within ``_EXACT_REWARD_TOL`` of
    exactly +1.0 or -1.0 are counted separately in two named buckets.  These
    exact-value counts are tallied *in addition to* the regular bins — i.e. a
    reward of 1.0 is still counted inside the ``[+0.8,+1.0)`` bin AND in the
    ``exact_pos`` bucket.  This makes the bin totals stay equal to len(rewards)
    while the named buckets give an easy read on boundary events.

    Returns
    -------
    bins : list of (bin_label_str, count)
        10-element list, one entry per bin.
    exact_counts : dict
        Keys ``"exact_pos"`` (reward == +1.0) and ``"exact_neg"``
        (reward == -1.0), each an int.
    """
    edges = [-1.0 + i * 0.2 for i in range(11)]  # 11 edges → 10 bins
    counts = [0] * 10
    exact_pos = 0  # frames with reward within _EXACT_REWARD_TOL of +1.0
    exact_neg = 0  # frames with reward within _EXACT_REWARD_TOL of -1.0
    for r in rewards:
        # Named exact-value buckets — checked BEFORE clamping.
        if abs(r - 1.0) <= _EXACT_REWARD_TOL:
            exact_pos += 1
        if abs(r - (-1.0)) <= _EXACT_REWARD_TOL:
            exact_neg += 1
        # Standard histogram — clamp to [edges[0], edges[-1]) so extreme
        # values land in the boundary bins rather than being dropped.
        r_clamped = max(edges[0], min(r, edges[-1] - 1e-9))
        idx = int((r_clamped - edges[0]) / 0.2)
        idx = max(0, min(idx, 9))
        counts[idx] += 1
    labels = [f"[{edges[i]:+.1f},{edges[i+1]:+.1f})" for i in range(10)]
    bins = list(zip(labels, counts))
    exact_counts = {"exact_pos": exact_pos, "exact_neg": exact_neg}
    return bins, exact_counts


def _print_episode_table(
    phase: int,
    ep: int,
    n_episodes: int,
    ep_steps: int,
    ep_reward: float,
    ep_positive_steps: int,
    total_env_steps: int,
    buffer_sizes: dict,
    info: dict,
    mean_inference_ms: float = 0.0,
    mean_interframe_ms: float = 0.0,
    mean_response_latency_ms: float = 0.0,
    last_frame_game_time: float = 0.0,
    last_response_latency_ms: float = 0.0,
    reward_distribution: Optional[list] = None,
) -> None:
    """Print a compact fixed-width table for one completed episode.

    Parameters
    ----------
    buffer_sizes : dict
        Must contain keys 'pos', 'neg', 'total'.
    reward_distribution : list or None
        Per-step reward values collected during the episode.  When provided,
        a 10-bin histogram over [-1.0, 1.0] (bin width 0.2) is appended to
        the table, followed by two named exact-value rows for reward == +1.0
        and reward == -1.0 (within ``_EXACT_REWARD_TOL``).
    """
    SEP = "+" + "-" * 22 + "+" + "-" * 16 + "+"
    def row(label: str, value: str) -> str:
        return f"| {label:<20} | {value:>14} |"

    # Resolve mean speed: prefer a direct key, else derive from speed_multiplier.
    _speed_val: Optional[float] = None
    for _k in ("avg_speed", "mean_speed"):
        if _k in info:
            _speed_val = float(info[_k])
            break
    if _speed_val is None and "speed_multiplier" in info:
        _speed_val = float(info["speed_multiplier"]) - 1.0 + 40.0

    lines = [
        SEP,
        row("Phase / Episode", f"{phase} / {ep} of {n_episodes}"),
        row("Steps (episode)", f"{ep_steps:,}"),
        row("Total env steps", f"{total_env_steps:,}"),
        row("Reward (episode)", f"{ep_reward:.4f}"),
        row("Avg reward/frame", f"{ep_reward / ep_steps:.6f}" if ep_steps > 0 else "N/A"),
        row(
            "Positive reward frames",
            f"{ep_positive_steps} / {ep_steps} ({100 * ep_positive_steps / ep_steps:.1f}%)"
            if ep_steps > 0 else "N/A",
        ),
    ]
    if _speed_val is not None:
        lines.append(row("Mean speed", f"{_speed_val:.2f}"))
    if "speed_multiplier" in info:
        lines.append(row("Speed multiplier", f"{float(info['speed_multiplier']):.4f}"))
    for _lap_k in ("lap_time", "lap_time_s"):
        if _lap_k in info:
            lines.append(row("Lap time", f"{float(info[_lap_k]):.3f}"))
            break
    for _ot_k in ("off_track_count", "off_track"):
        if _ot_k in info:
            lines.append(row("Off-track count", f"{int(info[_ot_k]):,}"))
            break
    # Timing rows — always shown so the user can spot latency regressions.
    lines.append(row("Avg inference time", f"{mean_inference_ms:.1f} ms"))
    _interframe_str = f"{mean_interframe_ms:.1f} ms"
    if mean_interframe_ms > 40.0:
        _interframe_str += " !!"
    lines.append(row("Avg inter-frame time", _interframe_str))
    lines.append(row("Avg response latency", f"{mean_response_latency_ms:.1f} ms"))
    lines.append(
        row(
            "Buffer (pos/neg/tot)",
            f"{buffer_sizes['pos']}/{buffer_sizes['neg']}/{buffer_sizes['total']}",
        )
    )

    # Reward distribution histogram — appended before the closing separator.
    if reward_distribution is not None and len(reward_distribution) > 0:
        bins, exact_counts = _reward_histogram(reward_distribution)
        lines.append(SEP)
        lines.append(row("Reward distribution", "frames"))
        lines.append(SEP)
        for bin_label, count in bins:
            lines.append(row(bin_label, f"{count:,}"))
        # Named exact-value buckets: reward == +1.0 and reward == -1.0.
        # These are displayed as a sub-section directly below the standard bins
        # so they stand out as distinct events rather than being averaged into
        # the boundary bins.
        lines.append(SEP)
        lines.append(row("Exact-value buckets", "frames"))
        lines.append(SEP)
        _pct = lambda n: f" ({100.0 * n / len(reward_distribution):.1f}%)" if reward_distribution else ""
        lines.append(
            row(
                "reward == +1.0",
                f"{exact_counts['exact_pos']:,}{_pct(exact_counts['exact_pos'])}",
            )
        )
        lines.append(
            row(
                "reward == -1.0",
                f"{exact_counts['exact_neg']:,}{_pct(exact_counts['exact_neg'])}",
            )
        )

    lines.append(SEP)
    print("\n".join(lines), flush=True)
    # Concrete per-episode timing example using AC's shared session clock.
    print(
        f"  [Timing] Last frame captured at {last_frame_game_time:.0f}ms (AC time), "
        f"action sent ~{last_frame_game_time + last_response_latency_ms:.0f}ms",
        flush=True,
    )


_DEFAULT_CONFIG = {
    "episodes_per_phase":      10,
    "train_steps_per_phase":   None,   # None = half of current buffer size at train time
    "batch_size":              128,
    "checkpoint_dir":          "checkpoints/sac_monza_v1",
    "checkpoint_freq":         5,
    "log_interval":            100,
    # Stationary-car crash detection: episode ends when total displacement
    # (oldest→newest position over the rolling window) is below the threshold.
    "stationary_frames":       10,
    "stationary_threshold":    0.5,    # metres
    # Per-dimension Gaussian noise added to actions during collection.
    # Order: [steering, throttle, brake]. None = disabled.
    "exploration_noise":       None,
}


class SACAgent:
    """
    Phase-based SAC training loop.

    Parameters
    ----------
    env : OurEnv
        Wrapped Assetto Corsa environment.
        reset() -> (obs, info)
        step(action) -> (obs, reward, done, info)
        action is in AC-space [-1, 1]^3.
    sac : SAC
        The SAC algorithm instance.
    replay_buffer : ReplayBuffer
        Pre-allocated replay buffer.
    config : dict
        Training configuration.  Keys and defaults:

        episodes_per_phase:    10     episodes collected each phase
        train_steps_per_phase: 20000  gradient steps each phase
        batch_size:            128    mini-batch size for update()
        checkpoint_dir:        'checkpoints/sac_monza_v1'
        checkpoint_freq:       5      save checkpoint every N phases
        log_interval:          100    log training stats every N gradient steps
    """

    def __init__(
        self,
        env,
        sac: SAC,
        replay_buffer: ReplayBuffer,
        config: dict = None,
        manage_ac: bool = False,
    ):
        self.env = env
        self.sac = sac
        self.replay_buffer = replay_buffer

        # Merge provided config with defaults
        cfg = dict(_DEFAULT_CONFIG)
        if config is not None:
            cfg.update(config)
        self.config = cfg

        # AC lifecycle management flag — when True, crash detection runs after
        # each episode and calls full_cycle() to relaunch if AC has crashed.
        # manage_ac can be set via the config dict OR the direct kwarg.
        # The direct kwarg takes precedence if both are supplied.
        self._manage_ac = config.get("manage_ac", False) if config else False
        if manage_ac:
            self._manage_ac = manage_ac

        # Counters that persist across phases
        self._total_env_steps = 0
        self._total_grad_steps = 0

        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)

        # Auto-load replay buffer if a buffer snapshot exists alongside the
        # SAC checkpoint.  This is a best-effort restore; a mismatch in
        # obs_dim/action_dim raises ValueError early so the user is not
        # silently training on wrong data.
        #
        # DualReplayBuffer saves two files: buffer_pos.npz + buffer_neg.npz.
        # Legacy single-buffer checkpoints (buffer.npz) are also attempted so
        # that old checkpoints keep working.
        _buf_base = os.path.join(self.config["checkpoint_dir"], "buffer")
        _dual_pos = _buf_base + "_pos.npz"
        _dual_neg = _buf_base + "_neg.npz"
        _legacy   = _buf_base + ".npz"

        if isinstance(self.replay_buffer, DualReplayBuffer):
            _has_dual = os.path.isfile(_dual_pos) or os.path.isfile(_dual_neg)
            if _has_dual:
                try:
                    self.replay_buffer.load(_buf_base + ".npz")
                    _sizes = self._buffer_sizes()
                    logger.info(
                        f"Dual replay buffer restored from {_buf_base}_{{pos,neg}}.npz  "
                        f"pos={_sizes['pos']} neg={_sizes['neg']} total={_sizes['total']}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"Could not load dual replay buffer from {_buf_base}: {exc}  "
                        f"Starting with empty buffers."
                    )
        else:
            if os.path.isfile(_legacy):
                try:
                    self.replay_buffer.load(_legacy)
                    logger.info(
                        f"Replay buffer restored from {_legacy}  "
                        f"(size={len(self.replay_buffer)})"
                    )
                except Exception as exc:
                    logger.warning(
                        f"Could not load replay buffer from {_legacy}: {exc}  "
                        f"Starting with empty buffer."
                    )

        logger.info(
            f"SACAgent ready — checkpoint_dir={self.config['checkpoint_dir']}  "
            f"manage_ac={self._manage_ac}"
        )

    # ------------------------------------------------------------------
    # Exploration noise helper
    # ------------------------------------------------------------------

    def _apply_exploration_noise(self, action: np.ndarray) -> np.ndarray:
        """Add per-dimension Gaussian noise to action and clip to [-1, 1]."""
        noise_cfg = self.config.get("exploration_noise")
        if not noise_cfg:
            return action
        noise = np.array([
            np.random.normal(0.0, noise_cfg[0]),  # steering
            np.random.normal(0.0, noise_cfg[1]),  # throttle
            np.random.normal(0.0, noise_cfg[2]),  # brake
        ], dtype=np.float32)
        return np.clip(action + noise, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Collection phase
    # ------------------------------------------------------------------

    def collect_phase(
        self,
        phase_num: int,
    ) -> dict:
        """
        Run episodes_per_phase episodes and store transitions in replay_buffer.

        Parameters
        ----------
        phase_num : int
            Current phase number (used for logging).

        Returns
        -------
        dict:
            episodes       number of episodes completed
            total_steps    steps collected this phase
            mean_reward    mean episode reward
            mean_ep_length mean episode length in steps
        """
        n_episodes = self.config["episodes_per_phase"]
        _buf_sizes = self._buffer_sizes()
        logger.info(
            f"[Phase {phase_num}] COLLECT — {n_episodes} episodes  "
            f"policy=SAC  "
            f"pos_buffer={_buf_sizes['pos']} neg_buffer={_buf_sizes['neg']} total={_buf_sizes['total']}"
        )

        episode_rewards = []
        episode_lengths = []
        phase_steps = 0

        _stat_frames    = int(self.config.get("stationary_frames",    10))
        _stat_threshold = float(self.config.get("stationary_threshold", 0.5))

        for ep in range(n_episodes):
            obs, _info = self.env.reset()
            ep_reward = 0.0
            ep_steps = 0
            ep_positive_steps = 0
            done = False
            ep_final_info = {}
            step_rewards: list = []
            inference_times: list = []
            interframe_times: list = []
            response_latency_times: list = []
            last_frame_game_time: float = 0.0
            last_response_latency_ms: float = 0.0
            # Rolling window of (x, y) positions for stationary-car detection.
            # Reset every episode so a prior episode's final position cannot
            # carry over and trigger a false positive on the next episode.
            _pos_window: collections.deque = collections.deque(maxlen=_stat_frames)

            # Warmup: straight, full throttle, no brake for ~6-10 s.
            # AC control space is [-1, +1] per VJoyControl.neutralize:
            #   steer=0 straight, throttle=+1 full, brake=-1 off.
            # Historical value [0.5, 1.0, 0.0] was wrong on both steer
            # (half-right) and brake (half-pressed).
            # Frames not added to buffer — transitions are discarded.
            _warmup_steps = random.randint(150, 250)
            for _ in range(_warmup_steps):
                obs, _, done, _ = self.env.step(np.array([0.0, 1.0, -1.0], dtype=np.float32))
                self.env._read_latest_state()

            # ── Latency-overlap episode loop ───────────────────────────────────
            # Step 0: no previously-computed action to pre-send, so use the normal
            # blocking path.  From step 1 onward the overlap trick is active:
            #   set_actions(action) fires immediately, then buffer push + next
            #   select_action() run in the ~40 ms window while AC processes the
            #   physics tick, and step(action=None) just waits for the result.
            #
            # Transition layout (what gets pushed to the replay buffer):
            #   obs       — state BEFORE the action was sent
            #   action    — the action sent in this step
            #   reward    — reward returned by step() for that action
            #   next_obs  — state AFTER the action resolved
            #   done      — episode-end flag from step()
            # ──────────────────────────────────────────────────────────────────

            # ── Step 0: blocking step to seed the overlap loop ────────────────
            # Uses the normal env.step(action) path because there is no
            # previously-computed action to pre-send.  The result is held in
            # (prev_obs, prev_action, reward, next_obs, done) — NOT yet pushed to
            # the buffer.  The push happens inside the first overlap iteration so
            # the bookkeeping is consistent with every subsequent iteration.
            _step0_ran = False
            if not done:
                _t_inf0_start = time.perf_counter()
                action = self.sac.select_action(obs, deterministic=False)
                action = self._apply_exploration_noise(action)
                _t_inf0_end = time.perf_counter()
                inference_times.append(_t_inf0_end - _t_inf0_start)
                prev_obs = obs
                prev_action = action
                next_obs, reward, done, step_info = self.env.step(action)
                self.env._read_latest_state()
                _step0_ran = True
                # Frame from step 0 just arrived — start inter-frame timer.
                _t_frame = time.perf_counter()
                if isinstance(step_info, dict):
                    # Stationary-car check: append position and test window.
                    _sx = step_info.get("world_position_x")
                    _sy = step_info.get("world_position_y")
                    if _sx is not None and _sy is not None:
                        _pos_window.append((float(_sx), float(_sy)))
                        if len(_pos_window) == _stat_frames:
                            _dx = _pos_window[-1][0] - _pos_window[0][0]
                            _dy = _pos_window[-1][1] - _pos_window[0][1]
                            if math.sqrt(_dx * _dx + _dy * _dy) < _stat_threshold:
                                logger.info(
                                    f"[Phase {phase_num}] ep={ep + 1} step=0 "
                                    f"stationary crash detected — "
                                    f"displacement < {_stat_threshold}m over "
                                    f"{_stat_frames} frames. Ending episode."
                                )
                                done = True
                    if done:
                        ep_final_info = step_info
                else:
                    if done:
                        ep_final_info = {}

            # ── Steps 1+: overlap trick ────────────────────────────────────────
            # Correct order per iteration:
            #   1. Advance obs to the frame that just arrived.
            #   2. Infer on frame N — produces the action AC needs RIGHT NOW.
            #   3. set_actions(action) — fire to AC immediately (non-blocking).
            #   4. Push the previous transition (CPU work, overlaps with AC tick).
            #   5. env.step(action=None) — wait for next frame.
            #
            # Inference MUST precede set_actions so AC receives an action computed
            # from the current frame, not the stale action from the previous frame.
            #
            # prev_action tracks the action that was actually sent to produce
            # next_obs, which is what belongs in the replay buffer transition.
            while not done:
                # 1. Advance obs to the arrived frame.
                obs = next_obs

                # 2. Infer on the current frame — this is the action AC needs next.
                _t_inf_start = time.perf_counter()
                action = self.sac.select_action(obs, deterministic=False)
                action = self._apply_exploration_noise(action)
                _t_inf_end = time.perf_counter()
                inference_times.append(_t_inf_end - _t_inf_start)

                # 3. Fire the freshly-computed action to AC immediately (non-blocking).
                self.env.set_actions(action)
                t_action_sent = time.perf_counter()
                _resp_ms = (t_action_sent - _t_frame) * 1000.0
                response_latency_times.append(_resp_ms)
                last_response_latency_ms = _resp_ms

                # 4. CPU work — overlaps with the AC physics tick.
                #    Push the transition that just completed: prev_obs →(prev_action)→ obs.
                #    prev_action is the action that was sent LAST iteration (or in step 0)
                #    and whose effect produced the obs we are now standing on.
                self.env._read_latest_state()
                self.replay_buffer.push(prev_obs, prev_action, reward, obs, float(done))

                # Accumulate episode statistics.
                ep_reward += reward
                ep_steps += 1
                step_rewards.append(reward)
                if reward > 0:
                    ep_positive_steps += 1
                phase_steps += 1
                self._total_env_steps += 1

                # 5. Wait for AC's physics frame — the only blocking call here.
                #    action=None because we already sent it in step 3.
                #    Record inter-frame time (inference + set_actions + push) just
                #    before blocking.
                prev_obs = obs
                prev_action = action
                _t_next = time.perf_counter()
                interframe_times.append(_t_next - _t_frame)
                if STEP_INTERVAL_S > 0.0:
                    _step_deadline = t_action_sent + STEP_INTERVAL_S
                    _remaining = _step_deadline - time.perf_counter()
                    if _remaining > 0.0:
                        time.sleep(_remaining)
                next_obs, reward, done, step_info = self.env.step(action=None)
                # Frame just arrived — reset inter-frame start for next iteration.
                _t_frame = time.perf_counter()
                if isinstance(step_info, dict):
                    last_frame_game_time = float(step_info.get("currentTime", 0.0))
                    # Stationary-car check: append position and test window.
                    _sx = step_info.get("world_position_x")
                    _sy = step_info.get("world_position_y")
                    if _sx is not None and _sy is not None:
                        _pos_window.append((float(_sx), float(_sy)))
                        if len(_pos_window) == _stat_frames:
                            _dx = _pos_window[-1][0] - _pos_window[0][0]
                            _dy = _pos_window[-1][1] - _pos_window[0][1]
                            if math.sqrt(_dx * _dx + _dy * _dy) < _stat_threshold:
                                logger.info(
                                    f"[Phase {phase_num}] ep={ep + 1} "
                                    f"step={ep_steps + 1} "
                                    f"stationary crash detected — "
                                    f"displacement < {_stat_threshold}m over "
                                    f"{_stat_frames} frames. Ending episode."
                                )
                                done = True
                    if done:
                        ep_final_info = step_info
                else:
                    if done:
                        ep_final_info = {}

            # Push the terminal transition that caused the loop to exit (or the
            # single step-0 transition if done=True immediately after step 0).
            # Two cases:
            #   a) done=True after step(action=None): loop exited before the push
            #      at step 4 ran for this last transition.  prev_obs/prev_action
            #      were set just before step(action=None); next_obs/reward/done
            #      hold what it returned.
            #   b) done=True immediately after step 0 (loop never ran): prev_obs
            #      and prev_action are the step-0 obs/action; next_obs/reward/done
            #      are from step 0's env.step(action) call.
            # In both cases prev_action is the action that produced next_obs.
            # Guard: skip if step 0 never ran (done was True from warmup, so
            # prev_obs/prev_action/reward/next_obs are undefined).
            if _step0_ran:
                self.replay_buffer.push(prev_obs, prev_action, reward, next_obs, float(done))
                ep_reward += reward
                ep_steps += 1
                step_rewards.append(reward)
                if reward > 0:
                    ep_positive_steps += 1
                phase_steps += 1
                self._total_env_steps += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_steps)
            _mean_inf_ms = (
                float(np.mean(inference_times)) * 1000.0 if inference_times else 0.0
            )
            _mean_ift_ms = (
                float(np.mean(interframe_times)) * 1000.0 if interframe_times else 0.0
            )
            _mean_resp_ms = (
                float(np.mean(response_latency_times)) if response_latency_times else 0.0
            )
            _print_episode_table(
                phase=phase_num,
                ep=ep + 1,
                n_episodes=n_episodes,
                ep_steps=ep_steps,
                ep_reward=ep_reward,
                ep_positive_steps=ep_positive_steps,
                total_env_steps=self._total_env_steps,
                buffer_sizes=self._buffer_sizes(),
                info=ep_final_info,
                mean_inference_ms=_mean_inf_ms,
                mean_interframe_ms=_mean_ift_ms,
                mean_response_latency_ms=_mean_resp_ms,
                last_frame_game_time=last_frame_game_time,
                last_response_latency_ms=last_response_latency_ms,
                reward_distribution=step_rewards,
            )
            _ep_buf = self._buffer_sizes()
            logger.info(
                f"  ep={ep + 1}/{n_episodes}  steps={ep_steps}  "
                f"reward={ep_reward:.3f}  "
                f"pos_buffer={_ep_buf['pos']} neg_buffer={_ep_buf['neg']} total={_ep_buf['total']}"
            )


        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        mean_ep_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

        stats = {
            "episodes":      n_episodes,
            "total_steps":   phase_steps,
            "mean_reward":   mean_reward,
            "mean_ep_length": mean_ep_length,
        }
        logger.info(
            f"[Phase {phase_num}] COLLECT done — "
            f"steps={phase_steps}  mean_reward={mean_reward:.3f}  "
            f"mean_ep_length={mean_ep_length:.0f}  "
            f"total_env_steps={self._total_env_steps}"
        )
        return stats

    # ------------------------------------------------------------------
    # Training phase
    # ------------------------------------------------------------------

    def train_phase(self, phase_num: int, steps_collected: int = 0) -> dict:
        """
        Run gradient updates.

        The number of gradient steps equals ``steps_collected`` (1:1 with env
        steps collected this phase), capped at 20,000.  If
        ``train_steps_per_phase`` is set explicitly in config it overrides
        this logic.

        Parameters
        ----------
        phase_num : int
        steps_collected : int
            Number of environment steps gathered during the preceding
            ``collect_phase``.  Used to compute the 1:1 gradient-step ratio.

        Returns
        -------
        dict:
            mean_q_loss, mean_policy_loss, mean_alpha, mean_entropy
        """
        configured = self.config["train_steps_per_phase"]
        if configured is not None:
            n_steps = configured
            _ratio_note = "fixed"
        else:
            n_steps = min(steps_collected, 20_000)
            _ratio_note = "1:1 with env steps collected"
        batch_size = self.config["batch_size"]
        log_interval = self.config["log_interval"]

        _buf = self._buffer_sizes()
        logger.info(
            f"[Phase {phase_num}] TRAIN — {n_steps} gradient steps ({_ratio_note})  "
            f"batch_size={batch_size}  "
            f"pos_buffer={_buf['pos']} neg_buffer={_buf['neg']} total={_buf['total']}"
        )

        q_losses, policy_losses, alphas, entropies = [], [], [], []
        t_start = time.perf_counter()

        for step in range(1, n_steps + 1):
            losses = self.sac.update(self.replay_buffer, batch_size)

            q_losses.append(losses["q_loss"])
            policy_losses.append(losses["policy_loss"])
            alphas.append(losses["alpha"])
            entropies.append(losses["entropy"])
            self._total_grad_steps += 1

            if step % log_interval == 0:
                recent = slice(-log_interval, None)
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f"  [Phase {phase_num}] step={step}/{n_steps}  "
                    f"q_loss={float(np.mean(q_losses[recent])):.4f}  "
                    f"pi_loss={float(np.mean(policy_losses[recent])):.4f}  "
                    f"alpha={float(np.mean(alphas[recent])):.4f}  "
                    f"entropy={float(np.mean(entropies[recent])):.4f}  "
                    f"elapsed={elapsed:.1f}s  "
                    f"total_grad_steps={self._total_grad_steps}"
                )

        stats = {
            "mean_q_loss":      float(np.mean(q_losses)),
            "mean_policy_loss": float(np.mean(policy_losses)),
            "mean_alpha":       float(np.mean(alphas)),
            "mean_entropy":     float(np.mean(entropies)),
        }
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"[Phase {phase_num}] TRAIN done — "
            f"mean_q_loss={stats['mean_q_loss']:.4f}  "
            f"mean_pi_loss={stats['mean_policy_loss']:.4f}  "
            f"mean_alpha={stats['mean_alpha']:.4f}  "
            f"mean_entropy={stats['mean_entropy']:.4f}  "
            f"elapsed={elapsed:.1f}s"
        )
        return stats

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _latest_path(self) -> str:
        return os.path.join(self.config["checkpoint_dir"], "latest.pt")

    def _buffer_base_path(self) -> str:
        """Base path for buffer files (without extension).

        DualReplayBuffer appends _pos.npz / _neg.npz; a plain ReplayBuffer
        adds .npz directly.
        """
        return os.path.join(self.config["checkpoint_dir"], "buffer")

    def _buffer_sizes(self) -> dict:
        """Return dict with keys 'pos', 'neg', 'total' for logging."""
        buf = self.replay_buffer
        if isinstance(buf, DualReplayBuffer):
            pos = len(buf.positive_buffer)
            neg = len(buf.negative_buffer)
        else:
            pos = len(buf)
            neg = 0
        return {"pos": pos, "neg": neg, "total": pos + neg}

    def save_checkpoint(self, phase_num: int) -> str:
        path = self._latest_path()
        self.sac.save(path)
        logger.info(f"Checkpoint saved (overwrite): {path}  (phase={phase_num})")
        buf_base = self._buffer_base_path()
        self.replay_buffer.save(buf_base + ".npz")
        _sizes = self._buffer_sizes()
        logger.info(
            f"Replay buffer saved: {buf_base}_{{pos,neg}}.npz  "
            f"pos={_sizes['pos']} neg={_sizes['neg']} total={_sizes['total']}"
        )
        return path

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_phases: Optional[int] = None) -> None:
        """
        Main training loop.

        Each phase: collect with SAC policy, then train for train_steps_per_phase
        gradient steps on the frames collected during that phase only.
        Checkpoints saved every checkpoint_freq phases.

        Parameters
        ----------
        num_phases : int or None
            Stop after this many phases.  None = run forever (until Ctrl+C).
        """
        ckpt_freq = self.config["checkpoint_freq"]

        logger.info(
            f"SACAgent.run() — num_phases={'inf' if num_phases is None else num_phases}  "
            f"checkpoint_freq={ckpt_freq}"
        )

        phase = 0

        try:
            while num_phases is None or phase < num_phases:
                _ph_buf = self._buffer_sizes()
                logger.info(
                    f"\n{'=' * 60}\n"
                    f"  PHASE {phase}  "
                    f"(env_steps={self._total_env_steps}  "
                    f"grad_steps={self._total_grad_steps}  "
                    f"pos_buffer={_ph_buf['pos']} neg_buffer={_ph_buf['neg']} total={_ph_buf['total']})\n"
                    f"{'=' * 60}"
                )

                # --- Launch AC before collection ---
                if self._manage_ac:
                    try:
                        from ac_lifecycle import full_cycle
                    except ImportError as exc:
                        raise RuntimeError(
                            "manage_ac=True but ac_lifecycle.py could not be imported. "
                            f"Original error: {exc}"
                        ) from exc
                    logger.info(f"[Phase {phase}] Launching AC via full_cycle() ...")
                    try:
                        full_cycle(max_retries=3)
                    except Exception as exc:
                        logger.error(
                            f"[Phase {phase}] full_cycle() raised an exception: {exc}"
                        )
                        raise
                    logger.info(f"[Phase {phase}] AC is live — starting collection.")

                # --- Collection ---
                collect_stats = self.collect_phase(phase)

                # --- Kill AC before GPU training to free resources ---
                if self._manage_ac:
                    from ac_lifecycle import kill_ac
                    logger.info(f"[Phase {phase}] Killing AC before training phase ...")
                    kill_ac()
                    logger.info(f"[Phase {phase}] AC killed — starting GPU training.")

                # --- Training ---
                train_stats = self.train_phase(phase, steps_collected=collect_stats["total_steps"])

                # --- Phase summary ---
                self._log_phase_summary(phase, collect_stats, train_stats)

                # --- Save checkpoint ---
                self.save_checkpoint(phase)

                phase += 1

        except KeyboardInterrupt:
            logger.info(
                f"\nTraining stopped by user at phase {phase}  "
                f"(env_steps={self._total_env_steps}  "
                f"grad_steps={self._total_grad_steps})"
            )
            # Save latest checkpoint on interrupt
            self.save_checkpoint(phase)
            logger.info(f"Latest checkpoint saved on interrupt.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_phase_summary(
        self,
        phase: int,
        collect_stats: dict,
        train_stats: Optional[dict],
    ) -> None:
        """Print a one-block summary for the completed phase."""
        lines = [
            f"\n--- Phase {phase} summary ---",
            f"  Collection:  episodes={collect_stats['episodes']}  "
            f"steps={collect_stats['total_steps']}  "
            f"mean_reward={collect_stats['mean_reward']:.3f}  "
            f"mean_ep_len={collect_stats['mean_ep_length']:.0f}",
            f"  Buffer:      pos_buffer={self._buffer_sizes()['pos']} "
            f"neg_buffer={self._buffer_sizes()['neg']} total={self._buffer_sizes()['total']} transitions",
            f"  Total env:   {self._total_env_steps} steps",
        ]
        if train_stats is not None:
            lines += [
                f"  Training:    mean_q_loss={train_stats['mean_q_loss']:.4f}  "
                f"mean_pi_loss={train_stats['mean_policy_loss']:.4f}  "
                f"mean_alpha={train_stats['mean_alpha']:.4f}  "
                f"mean_entropy={train_stats['mean_entropy']:.4f}",
                f"  Total grad:  {self._total_grad_steps} steps",
            ]
        else:
            lines.append("  Training:    skipped (interrupted before training)")
        lines.append("---")
        logger.info("\n".join(lines))
