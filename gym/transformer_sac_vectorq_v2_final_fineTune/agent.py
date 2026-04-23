"""
agent.py — TransformerSACAgent: phase-based training loop for Assetto Corsa.

V2 variant: 6-channel memmap-backed replay buffer.

Key differences from transformer_sac_vectorq/agent.py:
    1. Uses SixChannelMemmapBuffer instead of DualWindowReplayBuffer.
       Buffer is always live on disk — no unload/load cycle needed.
    2. Simplified checkpoint saving: flush() instead of save().
    3. No _pending_episodes or _buffer_size_cache — buffer is always available.
    4. Episode table shows per-channel (s/t/b +/-) buffer sizes.

Everything else (collect_phase overlap trick, warmup, stationary detection,
OOT handling, vector reward computation) is identical to the v1 agent.
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
from .replay_buffer import SixChannelMemmapBuffer
from .vector_reward import compute_vector_reward, compute_vector_reward_detailed

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


_EXACT_REWARD_TOL = 1e-4
STEP_INTERVAL_S = 0.009

_OOT_CONSECUTIVE_LIMIT = 1
_OOT_PENALTY_REWARD    = -1.0


def _reward_histogram(rewards: list) -> tuple:
    """Compute a 10-bin reward histogram over [-1.0, 1.0]."""
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
    """Print a compact fixed-width table for one completed episode."""
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

    # 6-channel buffer sizes.
    _s = buffer_sizes
    _total = _s.get("total", sum(_s.values()))
    lines.append(row(
        "Buf s+/s-",
        f"{_s.get('steer_pos',0)}/{_s.get('steer_neg',0)}",
    ))
    lines.append(row(
        "Buf t+/t-",
        f"{_s.get('throttle_pos',0)}/{_s.get('throttle_neg',0)}",
    ))
    lines.append(row(
        "Buf b+/b-",
        f"{_s.get('brake_pos',0)}/{_s.get('brake_neg',0)}",
    ))
    lines.append(row("Buf total (6ch)", f"{_total:,}"))

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
    "train_steps_per_phase":   None,
    "train_epochs_per_phase":  1,
    "batch_size":              256,
    "chunk_size":              25_000,
    "checkpoint_dir":          "checkpoints/sac_transformer_v2",
    "checkpoint_freq":         5,
    "log_interval":            100,
    "exploration_noise":       None,
    "stationary_frames":       30,
    "stationary_threshold":    0.5,
}


class FinetuneAgent:
    """
    Phase-based Transformer-SAC training loop with 6-channel memmap buffer.

    The buffer is always on disk via memmap — no unload/load cycle needed.
    RAM usage stays low regardless of buffer capacity.
    """

    def __init__(
        self,
        env,
        sac: TransformerSAC,
        replay_buffer: SixChannelMemmapBuffer,
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
        self._use_ai_drive = config.get("use_ai_drive", True) if config else True

        self._total_env_steps = 0
        self._total_grad_steps = 0

        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)

        logger.info(
            f"TransformerSACAgent v2 ready (6-channel memmap buffer) — "
            f"token_dim={token_dim}  window_size={window_size}  "
            f"checkpoint_dir={self.config['checkpoint_dir']}  "
            f"manage_ac={self._manage_ac}  "
            f"buffer_dir={replay_buffer.base_dir}"
        )

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _obs_to_token(self, obs: np.ndarray) -> np.ndarray:
        return obs[:self.token_dim].astype(np.float32)

    def _make_obs_window(self, obs_deque: collections.deque) -> np.ndarray:
        W = self.window_size
        available = list(obs_deque)
        n_avail = len(available)
        if n_avail == W:
            return np.stack(available, axis=0)
        first = available[0]
        pad = [first] * (W - n_avail)
        return np.stack(pad + available, axis=0)

    # ------------------------------------------------------------------
    # Exploration noise helper
    # ------------------------------------------------------------------

    def _apply_exploration_noise(self, action: np.ndarray) -> np.ndarray:
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
        """Detect a silent AC freeze before each episode."""
        _probe_action = np.array([0.0, 1.0, -1.0], dtype=np.float32)

        def _run_probe() -> float:
            _obs, _info = self.env.reset()
            import time as _time
            _time.sleep(3.0)
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
                return

            logger.warning(
                "[Phase %d] ep=%d  _check_ac_alive: probe FAILED — "
                "LapDist delta=%.2f m < threshold=%.1f m (attempt %d/%d).  "
                "Restarting AC via full_cycle() ...",
                phase_num, ep + 1, progress_m, progress_threshold_m,
                attempt, max_retries,
            )

            try:
                from ac_lifecycle import full_cycle, kill_ac
            except ImportError as exc:
                raise RuntimeError(
                    "_check_ac_alive: could not import ac_lifecycle.full_cycle.  "
                    f"Original error: {exc}"
                ) from exc

            try:
                logger.info(
                    "[Phase %d] ep=%d  _check_ac_alive: killing AC before full_cycle ...",
                    phase_num, ep + 1,
                )
                kill_ac()
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
                        f"restart attempts."
                    ) from exc
                continue

        raise RuntimeError(
            f"_check_ac_alive: AC did not respond after {max_retries} restart "
            f"attempts."
        )

    # ------------------------------------------------------------------
    # Collection phase
    # ------------------------------------------------------------------

    def collect_phase(self, phase_num: int) -> dict:
        """
        Run episodes_per_phase episodes and push complete episodes to
        the 6-channel memmap buffer.
        """
        n_episodes = self.config["episodes_per_phase"]
        _buf_sizes = self._buffer_sizes()
        logger.info(
            f"[Phase {phase_num}] COLLECT — {n_episodes} episodes  "
            f"policy=TransformerSAC  "
            f"buffer_total={_buf_sizes['total']}"
        )

        episode_rewards = []
        episode_lengths = []
        phase_steps = 0
        phase_windows_added = 0

        _stat_frames    = int(self.config.get("stationary_frames", 30))
        _stat_threshold = float(self.config.get("stationary_threshold", 0.5))

        for ep in range(n_episodes):

            if self._manage_ac and self._use_ai_drive:
                try:
                    from ac_lifecycle import randomize_start_position
                    randomize_start_position(wait_s=25.0)
                except Exception as _rnd_exc:
                    logger.warning(
                        f"[Phase {phase_num}] ep={ep + 1} "
                        f"randomize_start_position failed: {_rnd_exc}"
                    )

            obs, _info = self.env.reset()

            obs_deque: collections.deque = collections.deque(maxlen=self.window_size)
            ep_tokens:  list = []
            ep_actions: list = []
            ep_rewards: list = []
            ep_dones:   list = []
            ep_means:      list = []
            ep_stds:       list = []
            ep_components: list = []
            ep_metrics:    list = []
            ep_vec_breakdowns: list = []

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
            _oot_consecutive: int = 0
            _crash_termination: bool = False

            # Warmup: full throttle to fill obs_deque.
            _warmup_action = np.array([0.0, 1.0, -1.0], dtype=np.float32)
            _warmup_steps = self.window_size
            _wu_prev_obs = obs
            _wu_prev_action = _warmup_action.copy()
            for _ in range(_warmup_steps):
                _wu_prev_token = self._obs_to_token(_wu_prev_obs)
                obs, _wu_reward, done, _wu_info = self.env.step(_warmup_action)
                self.env._read_latest_state()
                obs_deque.append(self._obs_to_token(obs))
                _wu_comp = _wu_info.get("reward_components", {}) if isinstance(_wu_info, dict) else {}
                _wu_met  = _wu_info.get("reward_metrics",    {}) if isinstance(_wu_info, dict) else {}
                _wu_oot  = _wu_comp.get("r_crash", 0.0) >= 1.0
                _wu_vec_reward = compute_vector_reward(
                    _wu_comp, _wu_met, _warmup_action, _wu_prev_action, _wu_oot,
                )
                ep_tokens.append(_wu_prev_token)
                ep_actions.append(_warmup_action.copy())
                ep_rewards.append(_wu_vec_reward)
                ep_dones.append(0.0)
                _wu_prev_obs = obs
                _wu_prev_action = _warmup_action.copy()
                if done:
                    break

            _warmup_token = self._obs_to_token(obs)

            # ── Step 0: blocking step to seed the overlap loop ────────────
            _step0_ran = False
            if not done:
                _obs_window0 = self._make_obs_window(obs_deque)

                _t_inf0_start = time.perf_counter()
                action, _mean0, _std0 = self.sac.select_action(_obs_window0, deterministic=False)
                action = self._apply_exploration_noise(action)
                _t_inf0_end = time.perf_counter()
                inference_times.append(_t_inf0_end - _t_inf0_start)

                _token0 = _warmup_token

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
                    if step_info.get("out_of_track"):
                        _oot_consecutive += 1
                        reward = _OOT_PENALTY_REWARD
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

            # ── Steps 1+: overlap trick ───────────────────────────────────
            while not done:
                obs = next_obs
                _cur_token = self._obs_to_token(obs)
                obs_deque.append(_cur_token)

                _obs_window = self._make_obs_window(obs_deque)

                _t_inf_start = time.perf_counter()
                action, _mean, _std = self.sac.select_action(_obs_window, deterministic=False)
                action = self._apply_exploration_noise(action)
                _t_inf_end = time.perf_counter()
                inference_times.append(_t_inf_end - _t_inf_start)

                self.env.set_actions(action)
                t_action_sent = time.perf_counter()
                _resp_ms = (t_action_sent - _t_frame) * 1000.0
                response_latency_times.append(_resp_ms)
                last_response_latency_ms = _resp_ms

                self.env._read_latest_state()
                _oot = prev_components.get("r_crash", 0.0) >= 1.0
                _vr_detail = compute_vector_reward_detailed(
                    prev_components, prev_metrics, prev_action,
                    ep_actions[-1] if ep_actions else prev_action,
                    _oot,
                )
                _vec_reward = _vr_detail["reward_vec"]
                ep_vec_breakdowns.append(_vr_detail)
                ep_tokens.append(prev_token)
                ep_actions.append(prev_action)
                ep_rewards.append(_vec_reward)
                ep_dones.append(0.0)
                ep_means.append(prev_mean)
                ep_stds.append(prev_std)
                ep_components.append(prev_components)
                ep_metrics.append(prev_metrics)

                _scalar_reward = float(np.sum(_vec_reward))
                ep_reward += _scalar_reward
                ep_steps += 1
                step_rewards.append(_scalar_reward)
                if _scalar_reward > 0:
                    ep_positive_steps += 1
                phase_steps += 1
                self._total_env_steps += 1

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
                    if step_info.get("out_of_track"):
                        _oot_consecutive += 1
                        reward = _OOT_PENALTY_REWARD
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

            # Push terminal transition.
            if _step0_ran:
                ep_tokens.append(prev_token)
                ep_actions.append(prev_action)
                ep_means.append(prev_mean)
                ep_stds.append(prev_std)
                ep_components.append(prev_components)
                ep_metrics.append(prev_metrics)
                if _crash_termination:
                    _terminal_vec = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
                    ep_vec_breakdowns.append({"reward_vec": _terminal_vec,
                        "steer": {"crash": -1.0}, "throttle": {"crash": -1.0}, "brake": {"crash": -1.0}})
                else:
                    _term_oot = prev_components.get("r_crash", 0.0) >= 1.0
                    _term_detail = compute_vector_reward_detailed(
                        prev_components, prev_metrics, prev_action,
                        ep_actions[-1] if len(ep_actions) >= 2 else prev_action,
                        _term_oot,
                    )
                    _terminal_vec = _term_detail["reward_vec"]
                    ep_vec_breakdowns.append(_term_detail)
                ep_rewards.append(_terminal_vec)
                ep_dones.append(1.0)
                _terminal_scalar = float(np.sum(_terminal_vec))
                ep_reward += _terminal_scalar
                ep_steps += 1
                step_rewards.append(_terminal_scalar)
                if _terminal_scalar > 0:
                    ep_positive_steps += 1
                phase_steps += 1
                self._total_env_steps += 1

            # Push the complete episode to the 6-channel memmap buffer.
            if len(ep_tokens) >= self.window_size + 1:
                _ep_windows = self.replay_buffer.push_episode(
                    obs_tokens=np.stack(ep_tokens,  axis=0),
                    actions=np.stack(ep_actions,    axis=0),
                    rewards=np.array(ep_rewards,    dtype=np.float32),
                    dones=np.array(ep_dones,        dtype=np.float32),
                )
                phase_windows_added += _ep_windows

            # Episode logging and plotting.
            if ep_steps > 0:
                try:
                    from transformer_sac_vectorq_v2_final_fineTune.episode_logger import plot_episode
                    _n = len(ep_components)
                    plot_episode(
                        episode_num=ep + 1,
                        actions=ep_actions[-_n:],
                        means=ep_means,
                        stds=ep_stds,
                        reward_vecs=ep_rewards[-_n:],
                        vec_breakdowns=ep_vec_breakdowns,
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
                f"total_buf={_ep_buf['total']}"
            )

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        mean_ep_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

        # Flush after all episodes in the phase are collected.
        self.replay_buffer.flush()

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
        """Run gradient updates."""
        configured = self.config["train_steps_per_phase"]
        if configured is not None:
            n_steps = configured
            _ratio_note = "fixed"
        else:
            total_buf = self._buffer_sizes()["total"]
            batch_size_cfg = self.config["batch_size"]
            epochs = max(1, int(self.config.get("train_epochs_per_phase", 1)))
            n_steps = max(epochs * total_buf // batch_size_cfg, 1)
            _ratio_note = f"{epochs}x pass ({total_buf} total entries)"

        batch_size   = self.config["batch_size"]
        log_interval = self.config["log_interval"]

        _buf = self._buffer_sizes()
        logger.info(
            f"[Phase {phase_num}] TRAIN — {n_steps} gradient steps ({_ratio_note})  "
            f"batch_size={batch_size}  total_buf={_buf['total']}"
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

        from .replay_buffer import ChunkedPrefetcher

        chunk_size = self.config.get("chunk_size", 25_000)
        prefetcher = ChunkedPrefetcher(
            buffer=self.replay_buffer,
            chunk_size=chunk_size,
            batch_size=batch_size,
            device=str(self.sac.device),
        )
        prefetcher.start()

        q_losses, policy_losses, alphas, entropies = [], [], [], []
        q_loss_per_ch_buf      = []
        entropy_per_action_buf = []
        alpha_per_action_buf   = []
        wait_times             = []
        gpu_times              = []
        t_start = time.perf_counter()

        for step in range(1, n_steps + 1):
            batch, wait_ms = prefetcher.get()
            wait_times.append(wait_ms)

            t_gpu_start = time.perf_counter()
            losses = self.sac.update_from_batch(batch)
            gpu_ms = (time.perf_counter() - t_gpu_start) * 1000.0
            gpu_times.append(gpu_ms)

            q_losses.append(losses["q_loss"])
            policy_losses.append(losses["policy_loss"])
            alphas.append(losses["alpha"])
            entropies.append(losses["entropy"])
            q_loss_per_ch_buf.append(losses.get("q_loss_per_channel", [0, 0, 0]))
            entropy_per_action_buf.append(losses["entropy_per_action"])
            alpha_per_action_buf.append(losses["alpha_per_action"])
            self._total_grad_steps += 1

            if step % log_interval == 0:
                recent = slice(-log_interval, None)
                elapsed = time.perf_counter() - t_start
                epa = np.mean(entropy_per_action_buf[recent], axis=0)
                apa = np.mean(alpha_per_action_buf[recent],   axis=0)
                qpc = np.mean(q_loss_per_ch_buf[recent],      axis=0)
                avg_wait = float(np.mean(wait_times[recent]))
                max_wait = float(max(wait_times[recent]))
                avg_gpu  = float(np.mean(gpu_times[recent]))
                logger.info(
                    f"  step={step}/{n_steps}  "
                    f"q_loss={float(np.mean(q_losses[recent])):.4f}  "
                    f"Qs=[{qpc[0]:.4f},{qpc[1]:.4f},{qpc[2]:.4f}]  "
                    f"pi_loss={float(np.mean(policy_losses[recent])):.4f}  "
                    f"alpha={float(np.mean(alphas[recent])):.4f}  "
                    f"H=[{epa[0]:.3f},{epa[1]:.3f},{epa[2]:.3f}]  "
                    f"A=[{apa[0]:.4f},{apa[1]:.4f},{apa[2]:.4f}]  "
                    f"swap={avg_wait:.1f}ms(max={max_wait:.0f})  gpu={avg_gpu:.1f}ms  "
                    f"elapsed={elapsed:.1f}s  "
                    f"total_grad_steps={self._total_grad_steps}"
                )

        prefetcher.stop()

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
            f"avg_swap={float(np.mean(wait_times)):.1f}ms  "
            f"avg_gpu={float(np.mean(gpu_times)):.1f}ms  "
            f"elapsed={elapsed:.1f}s"
        )
        return stats

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _latest_path(self) -> str:
        return os.path.join(self.config["checkpoint_dir"], "latest.pt")

    def _buffer_sizes(self) -> dict:
        """Return dict with per-channel sizes + total."""
        sizes = self.replay_buffer.sizes()
        sizes["total"] = sum(sizes.values())
        return sizes

    def save_checkpoint(self, phase_num: int) -> str:
        path = self._latest_path()
        self.sac.save(path)
        logger.info(f"Checkpoint saved (overwrite): {path}  (phase={phase_num})")
        # Flush memmap buffers to ensure all data is persisted.
        self.replay_buffer.flush()
        _sizes = self._buffer_sizes()
        logger.info(
            f"Buffer flushed — "
            f"s+={_sizes['steer_pos']} s-={_sizes['steer_neg']}  "
            f"t+={_sizes['throttle_pos']} t-={_sizes['throttle_neg']}  "
            f"b+={_sizes['brake_pos']} b-={_sizes['brake_neg']}  "
            f"total={_sizes['total']}"
        )
        return path

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_phases: Optional[int] = None) -> None:
        """
        Main training loop.

        Each phase: collect episodes -> train on buffer.
        Buffer is always on disk (memmap) — no unload/load needed.
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
                    f"buf_total={_ph_buf['total']})\n"
                    f"{'=' * 60}"
                )

                # Ensure AC is running and responsive before collection.
                if self._manage_ac:
                    try:
                        from ac_lifecycle import full_cycle, is_ac_live, _is_plugin_ready, PLUGIN_EGO_PORT
                    except ImportError as exc:
                        raise RuntimeError(
                            "manage_ac=True but ac_lifecycle.py could not be imported. "
                            f"Original error: {exc}"
                        ) from exc
                    logger.info(f"[Phase {phase}] Launching AC via full_cycle() ...")
                    try:
                        full_cycle(max_retries=3)
                    except Exception as exc:
                        logger.error(f"[Phase {phase}] full_cycle() raised: {exc}")
                        raise
                    logger.info(f"[Phase {phase}] Verifying car can move ...")
                    self._check_ac_alive(phase_num=phase, ep=0)
                    logger.info(f"[Phase {phase}] AC is live — starting collection.")

                collect_stats = self.collect_phase(phase)

                # Send full brakes to stop the car before training.
                _brake_action = np.array([0.0, -1.0, 1.0], dtype=np.float32)
                _brake_frames = 30
                logger.info(f"[Phase {phase}] Sending full brakes for {_brake_frames} frames ...")
                for _ in range(_brake_frames):
                    self.env.step(_brake_action)

                # Free collection-phase RAM before training.
                import gc
                gc.collect()

                train_stats = self.train_phase(
                    phase, steps_collected=collect_stats["windows_added"]
                )

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
        _sizes = self._buffer_sizes()
        lines = [
            f"\n--- Phase {phase} summary ---",
            f"  Collection:  episodes={collect_stats['episodes']}  "
            f"steps={collect_stats['total_steps']}  "
            f"mean_reward={collect_stats['mean_reward']:.3f}  "
            f"mean_ep_len={collect_stats['mean_ep_length']:.0f}",
            f"  Buffer:      s+={_sizes['steer_pos']} s-={_sizes['steer_neg']}  "
            f"t+={_sizes['throttle_pos']} t-={_sizes['throttle_neg']}  "
            f"b+={_sizes['brake_pos']} b-={_sizes['brake_neg']}  "
            f"total={_sizes['total']}",
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
