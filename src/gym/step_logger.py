"""
step_logger.py — Per-step decision log.

Writes one CSV row per step across all episodes in a run.
The file is cleared (truncated) at open() so each run starts fresh.

Columns
-------
episode, step,
nsp, speed_ms, pos_x, pos_y, yaw_deg,          ← telemetry from info
target_speed_ms, heading_err_deg,               ← policy internals (MathPolicy only; blank otherwise)
steer, throttle, brake,                         ← decision output [0,1]
decision_ms, apply_ms,                          ← timing
reward                                          ← step reward
"""

import csv
import math
from pathlib import Path


_COLUMNS = [
    "episode", "step",
    "nsp", "speed_ms", "pos_x", "pos_y", "yaw_deg",
    "target_speed_ms", "heading_err_deg", "lookahead_pts", "lookahead_x", "lookahead_y",
    "steer", "throttle", "brake",
    "decision_ms", "apply_ms",
    "reward",
]


class StepLogger:
    """
    Open once per run, call log() every step, close() at the end.

    Usage
    -----
        logger = StepLogger("collected_data/step_decisions.csv")
        logger.open()
        ...
        logger.log(episode=0, step=1, info=info, action=action,
                   decision_ms=0.02, apply_ms=40.1, reward=-0.5, policy=policy)
        ...
        logger.close()
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._file = None
        self._writer = None

    def open(self):
        """Create (or truncate) the CSV file and write the header."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_COLUMNS)
        self._file.flush()

    def log(
        self,
        episode: int,
        step: int,
        info: dict,
        action,           # np.ndarray (3,) — [steer, throttle, brake]
        decision_ms: float,
        apply_ms: float,
        reward: float,
        policy=None,      # optional — pulled for last_step_info if available
    ):
        if self._writer is None:
            return

        # Telemetry from info (populated by OurEnv.step → info.update(self.env.state))
        nsp     = info.get("NormalizedSplinePosition", "")
        speed   = info.get("speed", "")
        pos_x   = info.get("world_position_x", "")
        pos_y   = info.get("world_position_y", "")
        yaw_raw = info.get("yaw", None)
        yaw_deg = round(math.degrees(yaw_raw), 2) if yaw_raw is not None else ""

        # Policy internals — MathPolicy stores these after each select_action()
        extra = getattr(policy, "last_step_info", {}) or {}
        target_speed   = extra.get("target_speed_ms", "")
        heading_err    = extra.get("heading_err_deg", "")
        lookahead_pts  = extra.get("lookahead_pts", "")
        lookahead_x    = extra.get("lookahead_x", "")
        lookahead_y    = extra.get("lookahead_y", "")

        self._writer.writerow([
            episode, step,
            _fmt(nsp), _fmt(speed), _fmt(pos_x), _fmt(pos_y), yaw_deg,
            _fmt(target_speed), _fmt(heading_err), lookahead_pts, _fmt(lookahead_x), _fmt(lookahead_y),
            _fmt(action[0]), _fmt(action[1]), _fmt(action[2]),
            round(decision_ms, 3), round(apply_ms, 3),
            _fmt(reward),
        ])
        self._file.flush()

    def end_episode(self):
        """Write a blank separator row after each episode."""
        if self._writer is None:
            return
        self._writer.writerow([""] * len(_COLUMNS))
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None


def _fmt(v):
    """Round floats to 4 dp; pass through strings and empty strings."""
    if isinstance(v, float):
        return round(v, 4)
    return v
