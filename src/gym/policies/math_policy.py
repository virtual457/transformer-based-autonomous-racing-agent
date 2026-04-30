"""
Math policy — rule-based autopilot using AC's fast_lane.ai optimal-speed profile.

Algorithms:
  Steering  — Pure Pursuit: aim at a lookahead point on the racing line.
  Speed     — P-controller: match current speed to fast_lane optimal speed.

Uses info dict fields: world_position_x/y, speed, yaw, NormalizedSplinePosition.
Does NOT use the obs vector — driven entirely by raw AC telemetry from info.

Two construction paths:
    MathPolicy.from_client(ac_client)   ← preferred: fetches from the live plugin
    MathPolicy.from_file(track_name)    ← fallback: reads fast_lane.ai from disk
"""

import os
import struct
import math
import logging
import numpy as np

from policies.base import BasePolicy

logger = logging.getLogger(__name__)

AC_INSTALL = r"D:\SteamLibrary\steamapps\common\assettocorsa"


class MathPolicy(BasePolicy):
    """
    Rule-based autopilot driven by AC's fast_lane.ai optimal-speed profile.

    Steering  : Pure Pursuit — aims at a lookahead point on the racing line.
    Speed     : P-controller — matches current speed to fast_lane target speed.

    Do not construct directly — use from_client() or from_file().

    Parameters
    ----------
    racing_line  : list of (float, float) — (client_x, client_y) per point
    speed_arr    : list of float — optimal speed m/s per point
    lookahead    : racing-line points ahead to aim at (tune per track/speed)
    kp_throttle  : throttle gain per m/s of speed deficit
    kp_brake     : brake gain per m/s of speed excess
    """

    def __init__(
        self,
        racing_line:      list,
        speed_arr:        list,
        lookahead_time_s: float = 1.0,
        kp_throttle:      float = 0.05,
        kp_brake:         float = 0.08,
        brake_scale:      float = 5.0,
        steer_scale:      float = 0.2,
    ):
        self.racing_line         = racing_line
        self.speed_arr           = speed_arr
        self.n                   = len(racing_line)
        self.lookahead_time_s    = lookahead_time_s
        self.kp_throttle         = kp_throttle
        self.kp_brake            = kp_brake
        self.brake_scale         = brake_scale
        self.steer_scale         = steer_scale
        self._last_heading_error = 0.0
        self.last_step_info      = {}   # populated each select_action(); read by StepLogger

        # Average metres between consecutive racing-line points
        pts = np.array(racing_line, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        wrap  = pts[0] - pts[-1]
        total_len = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum()
                          + np.hypot(wrap[0], wrap[1]))
        self.meters_per_point = total_len / self.n

        logger.info(
            f"MathPolicy ready: {self.n} racing-line points, "
            f"{self.meters_per_point:.2f} m/point, "
            f"lookahead_time={lookahead_time_s}s, "
            f"speed range [{min(speed_arr):.1f}, {max(speed_arr):.1f}] m/s"
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_client(cls, ac_client, lookahead_time_s: float = 1.0,
                    kp_throttle: float = 0.05, kp_brake: float = 0.08) -> "MathPolicy":
        """
        Fetch racing line and speed profile from the live AC plugin.

        The plugin's simulation management server (port 2347) already has
        fast_lane.ai parsed — no need to re-read the binary from disk.

        Parameters
        ----------
        ac_client : AssettoCorsaEnv.ac_client.Client
            The AC client instance (available as env.client or ac_env.client).
        """
        logger.info("MathPolicy: fetching track info from plugin ...")
        track_info = ac_client.simulation_management.get_track_info()

        # plugin Track class stores:
        #   fast_lane  = [(AC_z, AC_x), ...]  already in client coordinate space
        #   speed_arr  = [max(15, speed), ...]  optimal m/s per point
        racing_line = track_info["fast_lane"]
        speed_arr   = [max(15.0, float(s)) for s in track_info["speed_arr"]]

        logger.info(f"MathPolicy: got {len(racing_line)} points from plugin")
        return cls(racing_line, speed_arr, lookahead_time_s, kp_throttle, kp_brake)

    @classmethod
    def from_file(cls, track_name: str = "monza", track_config: str = "",
                  ac_install: str = AC_INSTALL, lookahead: int = 30,
                  kp_throttle: float = 0.05, kp_brake: float = 0.08) -> "MathPolicy":
        """
        Read fast_lane.ai binary directly from the AC install directory.

        Use this only when the AC plugin is not running (e.g. offline testing).

        Binary layout (structures.py):
            Header : 4 × int32
            Ideal  : count × (4 × float32 + int32)  → (x, y, z, dist, id)
            Detail : count × 18 × float32  — [1] = optimal speed (m/s)

        Coordinate mapping:
            client_x = AC_z  (ideal[i][2])
            client_y = AC_x  (ideal[i][0])
        """
        base = os.path.join(ac_install, "content", "tracks", track_name)
        if track_config and os.path.isdir(os.path.join(base, track_config)):
            base = os.path.join(base, track_config)

        path = os.path.join(base, "ai", "fast_lane.ai")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"fast_lane.ai not found: {path}\n"
                f"Check AC_INSTALL and track name match AC content/tracks/<name>."
            )

        with open(path, "rb") as f:
            _hdr, count, _u1, _u2 = struct.unpack("4i", f.read(16))
            ideal  = [struct.unpack("4f i", f.read(20)) for _ in range(count)]
            detail = [struct.unpack("18f",  f.read(72)) for _ in range(count)]

        racing_line = [(p[2], p[0]) for p in ideal]
        speed_arr   = [max(15.0, d[1]) for d in detail]

        logger.info(f"MathPolicy: loaded {len(racing_line)} points from {path}")
        return cls(racing_line, speed_arr, lookahead, kp_throttle, kp_brake)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Returns [steering, throttle, brake] all in [0, 1].
            steering: 0.0 = full left, 0.5 = straight, 1.0 = full right
            throttle: 0.0 = none,      1.0 = full
            brake:    0.0 = none,       1.0 = full
        """
        # -- Read telemetry --
        car_x = float(info.get("world_position_x", 0.0))
        car_y = float(info.get("world_position_y", 0.0))
        speed = float(info.get("speed", 0.0))
        yaw   = float(info.get("yaw",   0.0))
        nsp   = float(info.get("NormalizedSplinePosition", 0.0))

        # -- Compute decision --
        idx          = int(nsp * self.n) % self.n
        target_speed = self.speed_arr[idx]
        lookahead    = max(5, min(self.n // 4,
                          int(self.lookahead_time_s * speed / self.meters_per_point)))
        steer        = self._pure_pursuit(car_x, car_y, yaw, idx, lookahead)
        throttle, brake = self._speed_control(speed, target_speed)
        steer_out    = (-steer + 1.0) / 2.0  # negate + convert [-1,+1] → [0,1]

        # -- Expose internals for StepLogger --
        tx, ty = self.racing_line[(idx + lookahead) % self.n]
        self.last_step_info = {
            "target_speed_ms":  round(target_speed, 4),
            "heading_err_deg":  round(math.degrees(self._last_heading_error), 2),
            "lookahead_pts":    lookahead,
            "steer":            round(steer_out, 4),
            "throttle":         round(throttle, 4),
            "brake":            round(brake, 4),
            "lookahead_x":      round(tx, 4),
            "lookahead_y":      round(ty, 4),
        }

        # -- Log telemetry → decision (DEBUG level, use -v to see) --
        logger.debug(
            f"telemetry: pos=({car_x:.1f},{car_y:.1f})  speed={speed:.1f}m/s  "
            f"yaw={math.degrees(yaw):.1f}°  nsp={nsp:.4f}  idx={idx}/{self.n} | "
            f"target_speed={target_speed:.1f}m/s  "
            f"heading_err={math.degrees(self._last_heading_error):.1f}° | "
            f"decision: steer={steer_out:.3f}  throttle={throttle:.3f}  brake={brake:.3f}"
        )

        return np.array([steer_out, throttle, brake], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pure_pursuit(self, car_x: float, car_y: float, yaw: float, idx: int, lookahead: int) -> float:
        """Steer in [-1, +1] toward a lookahead point on the racing line."""
        tx, ty = self.racing_line[(idx + lookahead) % self.n]
        heading_error = math.atan2(ty - car_y, tx - car_x) - yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        self._last_heading_error = heading_error
        return max(-1.0, min(1.0, heading_error / (math.pi / 4) * self.steer_scale))

    def _speed_control(self, speed: float, target: float):
        """P-controller. Returns (throttle, brake) in [0, 1]."""
        error = target - speed
        if error > 0:
            return min(1.0, error * self.kp_throttle), 0.0
        else:
            return 0.0, min(1.0, -error * self.kp_brake * self.brake_scale)
