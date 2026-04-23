"""
agent.py — TransformerSACAgent: phase-based training loop for Assetto Corsa.

This file follows gym/sac/agent.py EXACTLY in structure:
    - Same phase structure (collect_phase -> train_phase)
    - Same latency-overlap trick (set_actions -> CPU work -> step(None))
    - Same stationary-car detection
    - Same warmup steps
    - Same _print_episode_table logging style
    - Same checkpoint saving pattern

Key differences from gym/sac/agent.py:
    1. A rolling deque of window_size obs tokens is maintained per episode.
       At each step: token = obs[:TOKEN_DIM], pushed to deque, the full
       deque (as a numpy array) is passed to select_action as obs_window.
    2. Per-episode lists of (token, action, reward, done) are accumulated
       during collection. At episode end, the full episode is pushed to
       the replay buffer as a sequence via push_episode().
    3. The replay buffer is a DualWindowReplayBuffer (window-based,
       pre-computed windows in pre-allocated circular arrays).

The env still returns 125-dim obs. The token extraction (obs[:TOKEN_DIM])
happens here in the agent loop, not in the environment.

Latency-overlap trick (collect_phase):
    obs = next_obs               # frame N has arrived
    token = obs[:TOKEN_DIM]      # extract 50-dim token
    obs_deque.append(token)      # update rolling window
    obs_window = deque_to_array  # (window_size, token_dim)
    action = select_action(obs_window)  # infer on frame N
    env.set_actions(action)             # send to AC immediately
    # CPU work (episode list append) overlaps with AC physics tick
    env.step(action=None)               # block only for remaining tick

Inference MUST happen before set_actions (identical constraint to flat SAC).
"""

import os
import time
import random
import logging
import collections
import math
from typing import Optional

import numpy as np

from .sac import TransformerSAC
from .replay_buffer import DualWindowReplayBuffer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


_EXACT_REWARD_TOL = 1e-4  # tolerance for exact-value buckets (+1.0 / -1.0)
STEP_INTERVAL_S = 0.009   # 9 ms — target minimum action interval

_OOT_CONSECUTIVE_LIMIT = 1   # terminate immediately on first OOT frame
_OOT_PENALTY_REWARD    = -1.0  # reward override applied every frame the car is off-track


def _reward_histogram(rewards: list) -> tuple:
    """Compute a 10-bin reward histogram over [-1.0, 1.0].

    Identical implementation to gym/sac/agent.py _reward_histogram.
    """
    edges = [-1.0 + i * 0.2 for i in range(11)]
    counts = [0] * 10
    exact_pos = 0
    exact_neg = 0
    for r in rewards:
        if abs(r - 1.0) <= _EXACT_REWARD_TOL:
            exact_pos += 1
        if abs(r - (-1.0)) <= _EXACT_REWARD_TOL:
            exact_neg += 1
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

    Identical in format to gym/sac/agent.py _print_episode_table.

    Parameters
    ----------
    buffer_sizes : dict
        Must contain keys 'pos', 'neg', 'total'.
    """
    SEP = "+" + "-" * 22 + "+" + "-" * 16 + "+"

    def row(label: str, value: str) -> str:
        return f"| {label:<20} | {value:>14} |"

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
        row(
            "Avg reward/frame",
            f"{ep_reward / ep_steps:.6f}" if ep_steps > 0 else "N/A",
        ),
        row(
            "Positive reward frames",
            (
                f"{ep_positive_steps} / {ep_steps} "
                f"({100 * ep_positive_steps / ep_steps:.1f}%)"
                if ep_steps > 0 else "N/A"
            ),
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

    if reward_distribution is not None and len(reward_distribution) > 0:
        bins, exact_counts = _reward_histogram(reward_distribution)
        lines.append(SEP)
        lines.append(row("Reward distribution", "frames"))
        lines.append(SEP)
        for bin_label, count in bins:
            lines.append(row(bin_label, f"{count:,}"))
        lines.append(SEP)
        lines.append(row("Exact-value buckets", "frames"))
        lines.append(SEP)
        _pct = (
            lambda n: f" ({100.0 * n / len(reward_distribution):.1f}%)"
            if reward_distribution else ""
        )
        lines.append(
            row("reward == +1.0", f"{exact_counts['exact_pos']:,}{_pct(exact_counts['exact_pos'])}")
        )
        lines.append(
            row("reward == -1.0", f"{exact_counts['exact_neg']:,}{_pct(exact_counts['exact_neg'])}")
        )

    lines.append(SEP)
    print("\n".join(lines), flush=True)
    print(
        f"  [Timing] Last frame captured at {last_frame_game_time:.0f}ms (AC time), "
        f"action sent ~{last_frame_game_time + last_response_latency_ms:.0f}ms",
        flush=True,
    )


_DEFAULT_CONFIG = {
    "episodes_per_phase":      10,
    "train_steps_per_phase":   None,   # None = min(steps_collected, 2_000)
    "batch_size":              256,
    "checkpoint_dir":          "checkpoints/sac_transformer_v1",
    "checkpoint_freq":         5,
    "log_interval":            100,
    "exploration_noise":       None,
    "stationary_frames":       30,
    "stationary_threshold":    0.5,    # metres
}


class FinetuneAgent:
    """
    Phase-based Transformer-SAC training loop.

    Parameters
    ----------
    env : OurEnv
        Wrapped Assetto Corsa environment.
        reset() -> (obs, info)
        step(action) -> (obs, reward, done, info)
        set_actions(action) -> None  (non-blocking)
        step(action=None) -> (obs, reward, done, info)  (overlap path)
        action space: [-1, 1]^3
    sac : TransformerSAC
        The TransformerSAC algorithm instance.
    replay_buffer : DualWindowReplayBuffer
        Window-based replay buffer (pre-computed windows, circular).
    config : dict
        Training configuration.
    token_dim : int
        How many dims to extract from obs as the token (50 = obs[:50]).
    window_size : int
        Rolling window length for the Transformer (75).
    manage_ac : bool
        If True, launches and kills AC around each phase via ac_lifecycle.
    """

    def __init__(
        self,
        env,
        sac: TransformerSAC,
        replay_buffer,
        config: dict = None,
        token_dim: int = 50,
        window_size: int = 75,
        manage_ac: bool = False,
    ):
        self.env = env
        self.sac = sac
        self.replay_buffer = replay_buffer
        self.token_dim = token_dim
        self.window_size = window_size

        cfg = dict(_DEFAULT_CONFIG)
        if config is not None:
            cfg.update(config)
        self.config = cfg

        self._manage_ac = config.get("manage_ac", False) if config else False
        if manage_ac:
            self._manage_ac = manage_ac

        self._total_env_steps = 0
        self._total_grad_steps = 0

        # Store buffer constructor params so we can recreate it after unloading.
        if isinstance(replay_buffer, DualWindowReplayBuffer):
            self._buf_capacity   = replay_buffer.capacity
            self._buf_action_dim = replay_buffer.action_dim
        else:
            self._buf_capacity   = 50_000
            self._buf_action_dim = 3

        # Episodes collected while the buffer is unloaded (manage_ac mode).
        self._pending_episodes: list = []
        # Cached sizes so we can log even when buffer is None.
        self._buffer_size_cache: dict = {"pos": 0, "neg": 0, "total": 0}

        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)

        # Auto-load replay buffer if a snapshot exists alongside the checkpoint.
        _buf_base = os.path.join(self.config["checkpoint_dir"], "buffer")
        _dual_pos = _buf_base + "_pos.npz"
        _dual_neg = _buf_base + "_neg.npz"

        if isinstance(self.replay_buffer, DualWindowReplayBuffer):
            _has_dual = os.path.isfile(_dual_pos) or os.path.isfile(_dual_neg)
            if _has_dual:
                try:
                    self.replay_buffer.load(_buf_base + ".npz")
                    _sizes = self._buffer_sizes()
                    self._buffer_size_cache = _sizes
                    logger.info(
                        f"Replay buffer restored — "
                        f"pos={_sizes['pos']} neg={_sizes['neg']} total={_sizes['total']}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"Could not load buffer from {_buf_base}: {exc}  "
                        f"Starting with empty buffers."
                    )

        logger.info(
            f"TransformerSACAgent ready (window-based buffer) — "
            f"token_dim={token_dim}  window_size={window_size}  "
            f"checkpoint_dir={self.config['checkpoint_dir']}  "
            f"manage_ac={self._manage_ac}"
        )

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _obs_to_token(self, obs: np.ndarray) -> np.ndarray:
        """Extract the 50-dim token from the 125-dim env obs."""
        return obs[:self.token_dim].astype(np.float32)

    def _make_obs_window(self, obs_deque: collections.deque) -> np.ndarray:
        """
        Convert the rolling obs deque to a (window_size, token_dim) array.

        When the deque is shorter than window_size (start of episode), the
        missing positions are filled by repeating the first (oldest) token.
        This matches the left-padding convention in DualWindowReplayBuffer.push_episode.
        """
        W = self.window_size
        available = list(obs_deque)   # oldest first
        n_avail = len(available)
        if n_avail == W:
            return np.stack(available, axis=0)
        # Pad left with first token.
        first = available[0]
        pad = [first] * (W - n_avail)
        return np.stack(pad + available, axis=0)

    # ------------------------------------------------------------------
    # Exploration noise helper
    # ------------------------------------------------------------------

    def _apply_exploration_noise(self, action: np.ndarray) -> np.ndarray:
        """Add per-dimension Gaussian noise to action and clip to [-1, 1]."""
        noise_cfg = self.config.get("exploration_noise")
        if not noise_cfg:
            return action
        noise = np.array([
            np.random.normal(0.0, noise_cfg[0]),
            np.random.normal(0.0, noise_cfg[1]),
            np.random.normal(0.0, noise_cfg[2]),
        ], dtype=np.float32)
        return np.clip(action + noise, -1.0, 1.0)

    # ------------------------------------------------------------------
    # AC alive check
    # ------------------------------------------------------------------

    def _check_ac_alive(
        self,
        phase_num: int,
        ep: int,
        probe_frames: int = 200,
        progress_threshold_m: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        """
        Detect a silent AC freeze before each episode.

        Applies full throttle for `probe_frames` steps and measures how far
        the car travels along the track (LapDist delta, metres).  If the car
        does not reach `progress_threshold_m` the AC session is restarted via
        full_cycle().  After the probe the caller is responsible for calling
        env.reset() to put the car back at the start line.

        Parameters
        ----------
        phase_num : int
            Current training phase (for log messages only).
        ep : int
            Current episode index within the phase (for log messages only).
        probe_frames : int
            Number of env.step() calls to send at full throttle.  At ~25 Hz
            that is 2 seconds of driving; a live car will cover at least
            progress_threshold_m metres in that time.
        progress_threshold_m : float
            Minimum LapDist delta (metres) required to consider AC responsive.
            Default 5.0 m is safely below any real-car speed at full throttle
            for 2 s (even 3 m/s gives 6 m) but well above a frozen car (0 m).
        max_retries : int
            How many full_cycle() restarts to attempt before raising.

        Raises
        ------
        RuntimeError
            If AC fails the liveness probe after `max_retries` restarts.
        """
        _probe_action = np.array([0.0, 1.0, -1.0], dtype=np.float32)  # center steer, full throttle delta, release brake

        def _run_probe() -> float:
            """
            Run full-throttle probe and return total LapDist progress (metres).

            our_env.reset() returns (obs, info) where info includes the raw
            env state via `info = dict(self.env.state)`.  our_env.step()
            returns (obs, reward, done, info) where info also contains the full
            raw state via `info.update(self.env.state)`.  Both provide the
            "LapDist" key (metres along current lap) directly from the AC plugin.
            Falls back to 0.0 if the key is absent so the retry path fires.
            """
            # One blocking reset to put the car on track before probing.
            _obs, _info = self.env.reset()
            # Wait for AC to finish placing the car on track before measuring.
            # Without this, LapDist reads 0.0 for the first ~2s after reset
            # causing the probe to fail even when AC is healthy.
            import time as _time
            _time.sleep(3.0)
            # Re-read LapDist after settle — first reading after reset is unreliable.
            _, _, _, _settle_info = self.env.step(_probe_action)
            lap_dist_start = float(_settle_info.get("LapDist", 0.0))

            _last_info = _info
            for _ in range(probe_frames):
                _obs, _r, _done, _last_info = self.env.step(_probe_action)
                if _done:
                    break

            lap_dist_end = float(_last_info.get("LapDist", 0.0))
            return lap_dist_end - lap_dist_start

        for attempt in range(1, max_retries + 1):
            progress_m = _run_probe()
            if progress_m >= progress_threshold_m:
                logger.info(
                    "[Phase %d] ep=%d  _check_ac_alive: probe passed — "
                    "LapDist delta=%.2f m (threshold=%.1f m).",
                    phase_num, ep + 1, progress_m, progress_threshold_m,
                )
                return  # AC is alive; caller will call reset() to restore start position

            logger.warning(
                "[Phase %d] ep=%d  _check_ac_alive: probe FAILED — "
                "LapDist delta=%.2f m < threshold=%.1f m (attempt %d/%d).  "
                "Restarting AC via full_cycle() ...",
                phase_num, ep + 1, progress_m, progress_threshold_m,
                attempt, max_retries,
            )

            # Restart AC.
            try:
                from ac_lifecycle import full_cycle
            except ImportError as exc:
                raise RuntimeError(
                    "_check_ac_alive: could not import ac_lifecycle.full_cycle.  "
                    f"Original error: {exc}"
                ) from exc

            try:
                full_cycle(max_retries=3)
            except Exception as exc:
                logger.error(
                    "[Phase %d] ep=%d  _check_ac_alive: full_cycle() failed on "
                    "restart attempt %d/%d: %s",
                    phase_num, ep + 1, attempt, max_retries, exc,
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"_check_ac_alive: AC did not respond after {max_retries} "
                        f"restart attempts.  Last probe progress: {progress_m:.2f} m "
                        f"(threshold: {progress_threshold_m:.1f} m).  "
                        "Training cannot continue — stopping cleanly."
                    ) from exc
                continue  # try the probe again after full_cycle raised

        # Reached if all retries passed full_cycle but probe still failed.
        raise RuntimeError(
            f"_check_ac_alive: AC did not respond after {max_retries} restart "
            f"attempts.  Last probe progress was below {progress_threshold_m:.1f} m.  "
            "Training cannot continue — stopping cleanly."
        )

    # ------------------------------------------------------------------
    # Collection phase
    # ------------------------------------------------------------------

    def collect_phase(self, phase_num: int) -> dict:
        """
        Run episodes_per_phase episodes and push complete episodes to
        replay_buffer.

        The rolling obs deque (window_size tokens, maxlen=window_size) is maintained per episode.
        Each step appends obs[:token_dim] to the deque and passes the full
        deque window to select_action.

        Per-episode lists accumulate (token, action, reward, done) and the
        complete episode is pushed via push_episode() at episode end.

        Latency-overlap trick is preserved exactly as in gym/sac/agent.py.

        Returns
        -------
        dict: episodes, total_steps, mean_reward, mean_ep_length
        """
        n_episodes = self.config["episodes_per_phase"]
        _buf_sizes = self._buffer_sizes()
        logger.info(
            f"[Phase {phase_num}] COLLECT — {n_episodes} episodes  "
            f"policy=TransformerSAC  "
            f"pos_buffer={_buf_sizes['pos']} neg_buffer={_buf_sizes['neg']} "
            f"total={_buf_sizes['total']}"
        )

        episode_rewards = []
        episode_lengths = []
        phase_steps = 0
        phase_windows_added = 0

        _stat_frames    = int(self.config.get("stationary_frames", 30))
        _stat_threshold = float(self.config.get("stationary_threshold", 0.5))

        for ep in range(n_episodes):

            obs, _info = self.env.reset()

            # Per-episode rolling context window (deque auto-truncates at maxlen=W).
            obs_deque: collections.deque = collections.deque(maxlen=self.window_size)
            # Per-episode transition lists for push_episode() at end.
            ep_tokens:  list = []
            ep_actions: list = []
            ep_rewards: list = []
            ep_dones:   list = []
            # Per-episode logging lists (rich data for episode_logger).
            ep_means:      list = []
            ep_stds:       list = []
            ep_components: list = []  # reward_components dict per frame
            ep_metrics:    list = []  # reward_metrics dict per frame (speed, gap, yaw)

            ep_reward          = 0.0
            ep_steps           = 0
            ep_positive_steps  = 0
            done               = False
            ep_final_info      = {}
            step_rewards:      list = []
            inference_times:   list = []
            interframe_times:  list = []
            response_latency_times: list = []
            last_frame_game_time: float = 0.0
            last_response_latency_ms: float = 0.0

            _pos_window: collections.deque = collections.deque(maxlen=_stat_frames)
            _oot_consecutive: int = 0  # consecutive frames off track this episode
            _crash_termination: bool = False  # True if episode ended by stationary/OOT

            # Warmup: full throttle for 6-10 seconds (150-250 steps at 25 Hz).
            # Warmup transitions ARE stored in the episode buffer so that even
            # very short policy episodes have 75+ real frames of context.
            # They are NOT counted in ep_reward / ep_steps (logging stays clean).
            _warmup_action = np.array([0.0, 1.0, -1.0], dtype=np.float32)  # center steer, full throttle, no brake
            _warmup_steps = self.window_size  # 75 steps — fills obs_deque
            _wu_prev_obs = obs  # obs from env.reset()
            for _ in range(_warmup_steps):
                _wu_prev_token = self._obs_to_token(_wu_prev_obs)
                obs, _wu_reward, done, _ = self.env.step(_warmup_action)
                self.env._read_latest_state()
                obs_deque.append(self._obs_to_token(obs))
                ep_tokens.append(_wu_prev_token)
                ep_actions.append(_warmup_action.copy())
                ep_rewards.append(float(_wu_reward))
                ep_dones.append(0.0)
                _wu_prev_obs = obs
                if done:
                    break

            # After warmup the deque holds up to window_size real frames —
            # no padding seed needed.
            _warmup_token = self._obs_to_token(obs)

            # ── Step 0: blocking step to seed the overlap loop ────────────────
            _step0_ran = False
            if not done:
                # Build window from current deque (has warmup token).
                _obs_window0 = self._make_obs_window(obs_deque)

                _t_inf0_start = time.perf_counter()
                action, _mean0, _std0 = self.sac.select_action(_obs_window0, deterministic=False)
                action = self._apply_exploration_noise(action)
                _t_inf0_end = time.perf_counter()
                inference_times.append(_t_inf0_end - _t_inf0_start)

                # Store step-0 token (BEFORE the action is applied — this is obs).
                _token0 = _warmup_token  # token extracted from post-warmup obs

                prev_token  = _token0
                prev_action = action
                prev_mean   = _mean0
                prev_std    = _std0
                next_obs, reward, done, step_info = self.env.step(action)
                prev_components = step_info.get("reward_components", {}) if isinstance(step_info, dict) else {}
                prev_metrics    = step_info.get("reward_metrics",    {}) if isinstance(step_info, dict) else {}
                self.env._read_latest_state()
                _step0_ran = True

                _t_frame = time.perf_counter()

                if isinstance(step_info, dict):
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
                                    f"stationary detected — displacement < "
                                    f"{_stat_threshold}m over {_stat_frames} frames."
                                )
                                done = True
                                _crash_termination = True
                    # Custom OOT consecutive counter (step 0).
                    if step_info.get("out_of_track"):
                        _oot_consecutive += 1
                        reward = _OOT_PENALTY_REWARD  # override reward every OOT frame
                        if _oot_consecutive >= _OOT_CONSECUTIVE_LIMIT:
                            logger.info(
                                f"[Phase {phase_num}] ep={ep + 1} step=0 "
                                f"OOT for {_oot_consecutive} consecutive frames "
                                f"(>= {_OOT_CONSECUTIVE_LIMIT}) — ending episode."
                            )
                            done = True
                            _crash_termination = True
                    else:
                        _oot_consecutive = 0
                    if done:
                        ep_final_info = step_info
                else:
                    if done:
                        ep_final_info = {}

            # ── Steps 1+: overlap trick ────────────────────────────────────────
            #
            # Correct order per iteration (same as gym/sac/agent.py):
            #   1. Advance obs to the arrived frame; extract token; update deque.
            #   2. Build obs_window from deque.
            #   3. Infer on window N — produces the action AC needs RIGHT NOW.
            #   4. set_actions(action) — fire to AC immediately (non-blocking).
            #   5. CPU work (episode list append) overlaps with AC tick.
            #   6. env.step(action=None) — wait for next frame.
            #
            # prev_token/prev_action track what produced next_obs, so they are
            # appended to the episode lists (not the current step's token/action).
            while not done:
                # 1. Advance obs to the arrived frame.
                obs = next_obs
                _cur_token = self._obs_to_token(obs)
                obs_deque.append(_cur_token)

                # 2. Build current window.
                _obs_window = self._make_obs_window(obs_deque)

                # 3. Infer on current window.
                _t_inf_start = time.perf_counter()
                action, _mean, _std = self.sac.select_action(_obs_window, deterministic=False)
                action = self._apply_exploration_noise(action)
                _t_inf_end = time.perf_counter()
                inference_times.append(_t_inf_end - _t_inf_start)

                # 4. Fire the freshly-computed action to AC immediately.
                self.env.set_actions(action)
                t_action_sent = time.perf_counter()
                _resp_ms = (t_action_sent - _t_frame) * 1000.0
                response_latency_times.append(_resp_ms)
                last_response_latency_ms = _resp_ms

                # 5. CPU work — append prev transition to episode lists.
                self.env._read_latest_state()
                ep_tokens.append(prev_token)
                ep_actions.append(prev_action)
                ep_rewards.append(float(reward))
                ep_dones.append(0.0)  # done flag belongs to NEXT transition
                ep_means.append(prev_mean)
                ep_stds.append(prev_std)
                ep_components.append(prev_components)
                ep_metrics.append(prev_metrics)

                ep_reward += reward
                ep_steps += 1
                step_rewards.append(reward)
                if reward > 0:
                    ep_positive_steps += 1
                phase_steps += 1
                self._total_env_steps += 1

                # 6. Wait for AC's physics frame.
                prev_token  = _cur_token
                prev_action = action
                prev_mean   = _mean
                prev_std    = _std
                _t_next = time.perf_counter()
                interframe_times.append(_t_next - _t_frame)
                if STEP_INTERVAL_S > 0.0:
                    _step_deadline = t_action_sent + STEP_INTERVAL_S
                    _remaining = _step_deadline - time.perf_counter()
                    if _remaining > 0.0:
                        time.sleep(_remaining)
                next_obs, reward, done, step_info = self.env.step(action=None)
                _t_frame = time.perf_counter()
                prev_components = step_info.get("reward_components", {}) if isinstance(step_info, dict) else {}
                prev_metrics    = step_info.get("reward_metrics",    {}) if isinstance(step_info, dict) else {}

                if isinstance(step_info, dict):
                    last_frame_game_time = float(step_info.get("currentTime", 0.0))
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
                                    f"step={ep_steps + 1} stationary detected — "
                                    f"displacement < {_stat_threshold}m over "
                                    f"{_stat_frames} frames."
                                )
                                done = True
                                _crash_termination = True
                    # Custom OOT consecutive counter (overlap loop).
                    if step_info.get("out_of_track"):
                        _oot_consecutive += 1
                        reward = _OOT_PENALTY_REWARD  # override reward every OOT frame
                        if _oot_consecutive >= _OOT_CONSECUTIVE_LIMIT:
                            logger.info(
                                f"[Phase {phase_num}] ep={ep + 1} "
                                f"step={ep_steps + 1} "
                                f"OOT for {_oot_consecutive} consecutive frames "
                                f"(>= {_OOT_CONSECUTIVE_LIMIT}) — ending episode."
                            )
                            done = True
                            _crash_termination = True
                    else:
                        _oot_consecutive = 0
                    if done:
                        ep_final_info = step_info
                else:
                    if done:
                        ep_final_info = {}

            # Push terminal transition (same guard as gym/sac/agent.py).
            if _step0_ran:
                ep_tokens.append(prev_token)
                ep_actions.append(prev_action)
                ep_means.append(prev_mean)
                ep_stds.append(prev_std)
                ep_components.append(prev_components)
                ep_metrics.append(prev_metrics)
                _terminal_reward = -1.0 if _crash_termination else float(reward)
                ep_rewards.append(_terminal_reward)
                ep_dones.append(1.0)   # terminal frame
                ep_reward += _terminal_reward
                ep_steps += 1
                step_rewards.append(_terminal_reward)
                if _terminal_reward > 0:
                    ep_positive_steps += 1
                phase_steps += 1
                self._total_env_steps += 1

            # Push the complete episode to the replay buffer.
            # DualWindowReplayBuffer requires >= window_size + 1 frames (76).
            if len(ep_tokens) >= self.window_size + 1:
                if self.replay_buffer is not None:
                    _ep_windows = self.replay_buffer.push_episode(
                        obs_tokens=np.stack(ep_tokens,  axis=0),
                        actions=np.stack(ep_actions,    axis=0),
                        rewards=np.array(ep_rewards,    dtype=np.float32),
                        dones=np.array(ep_dones,        dtype=np.float32),
                    )
                    phase_windows_added += _ep_windows
                else:
                    # Buffer unloaded (manage_ac RAM optimisation) — stash for deferred push.
                    self._pending_episodes.append({
                        "obs_tokens": np.stack(ep_tokens,  axis=0),
                        "actions":    np.stack(ep_actions, axis=0),
                        "rewards":    np.array(ep_rewards, dtype=np.float32),
                        "dones":      np.array(ep_dones,   dtype=np.float32),
                    })
                    phase_windows_added += max(0, len(ep_tokens) - self.window_size)

            # Episode logging and plotting (only for real policy steps, not pure warmup).
            if ep_steps > 0:
                try:
                    from transformer_sac_finetune.episode_logger import plot_episode
                    _n = len(ep_components)  # policy steps only (excludes warmup)
                    plot_episode(
                        episode_num=ep + 1,
                        actions=ep_actions[-_n:],
                        means=ep_means,
                        stds=ep_stds,
                        rewards=ep_rewards[-_n:],
                        components=ep_components,
                        metrics=ep_metrics,
                        output_dir=os.path.join(_THIS_DIR, "outputs"),
                    )
                except Exception as _e:
                    import traceback
                    logger.warning(f"episode_logger failed: {_e}\n{traceback.format_exc()}")
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
                f"pos_buffer={_ep_buf['pos']} neg_buffer={_ep_buf['neg']} "
                f"total={_ep_buf['total']}"
            )

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        mean_ep_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

        stats = {
            "episodes":        n_episodes,
            "total_steps":     phase_steps,
            "windows_added":   phase_windows_added,
            "mean_reward":     mean_reward,
            "mean_ep_length":  mean_ep_length,
        }
        logger.info(
            f"[Phase {phase_num}] COLLECT done — "
            f"steps={phase_steps}  windows_added={phase_windows_added}  "
            f"mean_reward={mean_reward:.3f}  "
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

        Number of gradient steps = 3 × total_buffer / batch_size (3 full passes),
        or the fixed value in train_steps_per_phase if set.

        The buffer must have enough sampleable steps to fill a batch.
        If not, training is skipped with a warning.

        Returns
        -------
        dict: mean_q_loss, mean_policy_loss, mean_alpha, mean_entropy
        """
        configured = self.config["train_steps_per_phase"]
        if configured is not None:
            n_steps = configured
            _ratio_note = "fixed"
        else:
            # 3 full passes over the complete buffer.
            total_buf = self._buffer_sizes()["total"]
            batch_size_cfg = self.config["batch_size"]
            n_steps = max((3 * total_buf) // batch_size_cfg, 1)
            _ratio_note = f"3x buffer passes ({total_buf} windows)"

        batch_size   = self.config["batch_size"]
        log_interval = self.config["log_interval"]

        _buf = self._buffer_sizes()
        logger.info(
            f"[Phase {phase_num}] TRAIN — {n_steps} gradient steps ({_ratio_note})  "
            f"batch_size={batch_size}  "
            f"pos_buffer={_buf['pos']} neg_buffer={_buf['neg']} total={_buf['total']}"
        )

        if not self.replay_buffer.is_ready(batch_size):
            logger.warning(
                f"[Phase {phase_num}] Buffer has {len(self.replay_buffer)} sampleable steps "
                f"(need {batch_size}) — skipping training this phase."
            )
            return {
                "mean_q_loss": 0.0,
                "mean_policy_loss": 0.0,
                "mean_alpha": self.sac.alpha.item(),
                "mean_entropy": 0.0,
            }

        q_losses, policy_losses, alphas, entropies = [], [], [], []
        entropy_per_action_buf = []   # list of [H_steer, H_throttle, H_brake]
        alpha_per_action_buf   = []   # list of [a_steer,  a_throttle, a_brake]
        t_start = time.perf_counter()

        for step in range(1, n_steps + 1):
            losses = self.sac.update(self.replay_buffer, batch_size)
            q_losses.append(losses["q_loss"])
            policy_losses.append(losses["policy_loss"])
            alphas.append(losses["alpha"])
            entropies.append(losses["entropy"])
            entropy_per_action_buf.append(losses["entropy_per_action"])
            alpha_per_action_buf.append(losses["alpha_per_action"])
            self._total_grad_steps += 1

            if step % log_interval == 0:
                recent = slice(-log_interval, None)
                elapsed = time.perf_counter() - t_start
                epa = np.mean(entropy_per_action_buf[recent], axis=0)   # (action_dim,)
                apa = np.mean(alpha_per_action_buf[recent],   axis=0)   # (action_dim,)
                logger.info(
                    f"  step={step}/{n_steps}  "
                    f"q_loss={float(np.mean(q_losses[recent])):.4f}  "
                    f"pi_loss={float(np.mean(policy_losses[recent])):.4f}  "
                    f"alpha={float(np.mean(alphas[recent])):.4f}  "
                    f"entropy={float(np.mean(entropies[recent])):.4f}  "
                    f"H=[{epa[0]:.3f},{epa[1]:.3f},{epa[2]:.3f}]  "
                    f"A=[{apa[0]:.4f},{apa[1]:.4f},{apa[2]:.4f}]  "
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
        return os.path.join(self.config["checkpoint_dir"], "buffer")

    def _buffer_sizes(self) -> dict:
        """Return dict with keys 'pos', 'neg', 'total' for logging.
        Returns cached sizes when the buffer is unloaded (None)."""
        buf = self.replay_buffer
        if buf is None:
            return self._buffer_size_cache
        if isinstance(buf, DualWindowReplayBuffer):
            pos = len(buf.positive_buffer)
            neg = len(buf.negative_buffer)
        else:
            pos = len(buf)
            neg = 0
        return {"pos": pos, "neg": neg, "total": pos + neg}

    def _unload_buffer(self) -> None:
        """Save the replay buffer to disk and release it from RAM."""
        import gc
        if self.replay_buffer is None:
            return
        buf_base = self._buffer_base_path()
        self.replay_buffer.save(buf_base + ".npz")
        self._buffer_size_cache = self._buffer_sizes()
        del self.replay_buffer
        self.replay_buffer = None
        gc.collect()
        logger.info(
            f"Buffer unloaded from RAM — "
            f"pos={self._buffer_size_cache['pos']} neg={self._buffer_size_cache['neg']} "
            f"(saved to disk)"
        )

    def _load_buffer(self) -> int:
        """Recreate buffer from disk, push any pending episodes, return windows pushed."""
        import gc
        self.replay_buffer = DualWindowReplayBuffer(
            capacity=self._buf_capacity,
            token_dim=self.token_dim,
            action_dim=self._buf_action_dim,
            window_size=self.window_size,
        )
        buf_base = self._buffer_base_path()
        if os.path.isfile(buf_base + "_pos.npz") or os.path.isfile(buf_base + "_neg.npz"):
            try:
                self.replay_buffer.load(buf_base + ".npz")
            except Exception as exc:
                logger.warning(f"Could not reload buffer from disk: {exc} — starting fresh.")

        windows_pushed = 0
        for ep in self._pending_episodes:
            if len(ep["obs_tokens"]) >= self.window_size + 1:
                windows_pushed += self.replay_buffer.push_episode(
                    obs_tokens=ep["obs_tokens"],
                    actions=ep["actions"],
                    rewards=ep["rewards"],
                    dones=ep["dones"],
                )
        self._pending_episodes.clear()
        gc.collect()
        self._buffer_size_cache = self._buffer_sizes()
        logger.info(
            f"Buffer loaded — pushed {windows_pushed} new windows — "
            f"pos={self._buffer_size_cache['pos']} neg={self._buffer_size_cache['neg']}"
        )
        return windows_pushed

    def save_checkpoint(self, phase_num: int) -> str:
        path = self._latest_path()
        self.sac.save(path)
        logger.info(f"Checkpoint saved (overwrite): {path}  (phase={phase_num})")
        if self.replay_buffer is not None:
            buf_base = self._buffer_base_path()
            self.replay_buffer.save(buf_base + ".npz")
            _sizes = self._buffer_sizes()
            self._buffer_size_cache = _sizes
            logger.info(
                f"Replay buffer saved: {buf_base}_{{pos,neg}}.npz  "
                f"pos={_sizes['pos']} neg={_sizes['neg']} total={_sizes['total']}"
            )
        else:
            logger.info("Buffer unloaded — skipping buffer save (already on disk).")
        return path

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_phases: Optional[int] = None) -> None:
        """
        Main training loop.

        Each phase: collect episodes (push_episode), then train on the
        episode-level buffer.  Checkpoint saved every checkpoint_freq phases.

        Parameters
        ----------
        num_phases : int or None
            Stop after this many phases.  None = run forever (until Ctrl+C).
        """
        ckpt_freq = self.config["checkpoint_freq"]

        logger.info(
            f"TransformerSACAgent.run() — "
            f"num_phases={'inf' if num_phases is None else num_phases}  "
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
                    f"pos_buffer={_ph_buf['pos']} neg_buffer={_ph_buf['neg']} "
                    f"total={_ph_buf['total']})\n"
                    f"{'=' * 60}"
                )

                # Unload buffer from RAM before launching AC to free memory.
                if self._manage_ac:
                    self._unload_buffer()

                # Launch AC before collection (only if not already live).
                if self._manage_ac:
                    try:
                        from ac_lifecycle import full_cycle, is_ac_live, _is_plugin_ready, PLUGIN_EGO_PORT
                    except ImportError as exc:
                        raise RuntimeError(
                            "manage_ac=True but ac_lifecycle.py could not be imported. "
                            f"Original error: {exc}"
                        ) from exc
                    if is_ac_live() and _is_plugin_ready(PLUGIN_EGO_PORT):
                        logger.info(f"[Phase {phase}] AC already live and plugin responding — skipping full_cycle().")
                    elif is_ac_live():
                        # AC is running but plugin not ready yet — wait up to 60s before giving up
                        logger.info(f"[Phase {phase}] AC live but plugin not ready — waiting up to 60s ...")
                        from ac_lifecycle import PLUGIN_EGO_PORT as _PORT
                        import time as _time
                        deadline = _time.time() + 60
                        while _time.time() < deadline:
                            if _is_plugin_ready(_PORT):
                                logger.info(f"[Phase {phase}] Plugin ready.")
                                break
                            _time.sleep(2)
                        else:
                            logger.warning(f"[Phase {phase}] Plugin did not respond in 60s — falling back to full_cycle().")
                            try:
                                full_cycle(max_retries=3)
                            except Exception as exc:
                                logger.error(f"[Phase {phase}] full_cycle() raised: {exc}")
                                raise
                    else:
                        logger.info(f"[Phase {phase}] AC not live — launching via full_cycle() ...")
                        try:
                            full_cycle(max_retries=3)
                        except Exception as exc:
                            logger.error(
                                f"[Phase {phase}] full_cycle() raised an exception: {exc}"
                            )
                            raise
                    logger.info(f"[Phase {phase}] Verifying car can move ...")
                    self._check_ac_alive(phase_num=phase, ep=0)
                    logger.info(f"[Phase {phase}] AC is live — starting collection.")

                collect_stats = self.collect_phase(phase)

                # Kill AC before GPU training to free VRAM for PyTorch.
                # full_cycle() will relaunch it at the start of the next collection phase.
                if self._manage_ac:
                    try:
                        from ac_lifecycle import kill_ac as _kill_ac
                        _kill_ac()
                        logger.info(f"[Phase {phase}] AC killed — starting GPU training.")
                    except Exception as _e:
                        logger.warning(f"[Phase {phase}] kill_ac() failed: {_e} — continuing anyway.")

                # Reload buffer and push episodes collected while it was unloaded.
                if self._manage_ac:
                    actual_windows = self._load_buffer()
                    collect_stats["windows_added"] = actual_windows

                train_stats = self.train_phase(
                    phase, steps_collected=collect_stats["windows_added"]
                )

                # Clear PyTorch VRAM before AC relaunches on the next phase.
                try:
                    import torch as _torch
                    _torch.cuda.empty_cache()
                except Exception:
                    pass

                self._log_phase_summary(phase, collect_stats, train_stats)
                self.save_checkpoint(phase)

                phase += 1

        except KeyboardInterrupt:
            logger.info(
                f"\nTraining stopped by user at phase {phase}  "
                f"(env_steps={self._total_env_steps}  "
                f"grad_steps={self._total_grad_steps})"
            )
            self.save_checkpoint(phase)
            logger.info("Latest checkpoint saved on interrupt.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_phase_summary(
        self,
        phase: int,
        collect_stats: dict,
        train_stats: Optional[dict],
    ) -> None:
        lines = [
            f"\n--- Phase {phase} summary ---",
            f"  Collection:  episodes={collect_stats['episodes']}  "
            f"steps={collect_stats['total_steps']}  "
            f"mean_reward={collect_stats['mean_reward']:.3f}  "
            f"mean_ep_len={collect_stats['mean_ep_length']:.0f}",
            f"  Buffer:      pos_buffer={self._buffer_sizes()['pos']} "
            f"neg_buffer={self._buffer_sizes()['neg']} "
            f"total={self._buffer_sizes()['total']} sampleable steps",
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
        logger.info("\n".join(lines))
