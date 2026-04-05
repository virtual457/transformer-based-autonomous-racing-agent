"""
ACTelemetry — reads from ac_env.state dict produced by assetto_corsa_gym.

Key state dict fields (set by the AC plugin sensors_par):
    speed           float   current speed (m/s)
    LapDist         float   distance along current lap (m)
    gap             float   signed lateral gap from racing line (m)
    numberOfTyresOut int    tyres outside track limits (0-4)
    out_of_track    bool    set by ac_env when numberOfTyresOut > 2
                            (requires use_ac_out_of_track=True in config)
    currentTime     float   current lap time (s)
    lastLap         float   last completed lap time (s)
"""

from telemetry.base import BaseTelemetry, TelemetryFrame


OOT_TYRES_THRESHOLD = 2   # >2 tyres out → out of track


class ACTelemetry(BaseTelemetry):
    """
    Reads ac_env.state and returns a structured TelemetryFrame.

    Falls back gracefully when fields are missing (e.g. first reset step).
    """

    def parse(self, raw_state: dict) -> TelemetryFrame:
        n_tyres_out = int(raw_state.get("numberOfTyresOut", 0))

        # ac_env sets out_of_track when use_ac_out_of_track=True.
        # Fall back to tyre count if not present.
        out_of_track = raw_state.get("out_of_track", n_tyres_out > OOT_TYRES_THRESHOLD)

        return TelemetryFrame(
            speed_ms=float(raw_state.get("speed", 0.0)),
            lap_dist=float(raw_state.get("LapDist", 0.0)),
            gap_m=float(raw_state.get("gap", 0.0)),
            n_tyres_out=n_tyres_out,
            out_of_track=bool(out_of_track),
            lap_time_s=float(raw_state.get("currentTime", 0.0)),
            best_lap_s=float(raw_state.get("lastLap", 0.0)),
            target_speed_ms=0.0,   # populated by our_env.py when use_target_speed=True
            raw=raw_state,
        )
