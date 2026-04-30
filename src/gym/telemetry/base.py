"""
BaseTelemetry — abstract interface for reading AC telemetry.

All reward components and crash detection read from TelemetryFrame,
not directly from ac_env.state. This decouples reward logic from the
specific dict keys that AC's plugin happens to produce.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TelemetryFrame:
    """
    One snapshot of car telemetry, parsed from a raw AC state dict.

    All consumers (rewards, crash detection, logging) read from here.
    Units: SI — metres, metres/second.
    """
    speed_ms: float        # current speed (m/s)
    lap_dist: float        # distance along lap (m)
    gap_m: float           # signed lateral gap from racing line (m)
    n_tyres_out: int       # number of tyres outside track limits (0–4)
    out_of_track: bool     # True when n_tyres_out > OOT_THRESHOLD

    # Optional fields — not always populated
    lap_time_s: float = 0.0
    best_lap_s: float = 0.0
    target_speed_ms: float = 0.0   # populated by our_env.py when use_target_speed=True
    yaw_error_rad: float = 0.0     # car heading minus racing-line heading, normalised to [-π, π]
    raw: dict = field(default_factory=dict, repr=False)


class BaseTelemetry(ABC):
    """
    Parses a raw state dict into a TelemetryFrame.

    Implement parse() for each data source (live AC, replay, mock).
    """

    @abstractmethod
    def parse(self, raw_state: dict) -> TelemetryFrame:
        """
        Parameters
        ----------
        raw_state : dict
            Raw state dict from the last ac_env.step() or ac_env.state.

        Returns
        -------
        TelemetryFrame
        """
