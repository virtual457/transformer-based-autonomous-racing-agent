"""
NeuralPolicy — thin shell that loads any registered model and exposes the
BasePolicy interface to OurEnv.

Old call sites that passed obs_dim / hidden_units / action_dim directly
still work because those kwargs are forwarded to the model's from_checkpoint().
"""

import csv
from pathlib import Path

import numpy as np
import torch
from policies.base import BasePolicy
from policies.models import load_model


class NeuralPolicy(BasePolicy):
    """Wraps any registered BaseModel for inference inside OurEnv.

    Parameters
    ----------
    model_path : str
        Path to a saved checkpoint (.pt).
    model_type : str
        Key in MODEL_REGISTRY. Currently "mlp" (default) is the only option.
    device : str
        "auto" (default) -> CUDA if available, else CPU.
        "cuda" / "cpu" -> explicit override.
    deterministic : bool
        True (default) -> tanh(mean), no sampling. Use for evaluation.
        False -> sample from policy distribution. Use for exploration.
    obs_log_path : str or None
        If set, every select_action() call appends one row to this CSV:
        step, obs_0 … obs_124, steer, throttle, brake.
        File is created (or truncated) on first call.
    **model_kwargs
        Forwarded verbatim to the model's from_checkpoint().
        Legacy callers may pass obs_dim=, action_dim=, hidden_units= here.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "mlp",
        device: str = "auto",
        deterministic: bool = True,
        obs_log_path: str = None,
        invert_steer: bool = False,
        steer_deadzone: float = 0.0,
        **model_kwargs,
    ):
        # Resolve device here so it is consistent with what the model uses
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self.deterministic = deterministic

        self.model = load_model(
            model_type=model_type,
            checkpoint_path=model_path,
            device=device,
            **model_kwargs,
        )
        self.model.eval()
        self.invert_steer = invert_steer
        self.steer_deadzone = steer_deadzone  # half-width around 0.5; 0.1 → [0.4,0.6] snaps to 0.5

        # Obs logging
        self._obs_log_path = obs_log_path
        self._obs_log_file = None
        self._obs_log_writer = None
        self._step_count = 0
        self.last_obs: np.ndarray = None   # last raw obs passed to select_action

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        model_type: str = "mlp",
        device: str = "auto",
        deterministic: bool = True,
        obs_log_path: str = None,
        invert_steer: bool = False,
        steer_deadzone: float = 0.0,
        **model_kwargs,
    ) -> "NeuralPolicy":
        """Convenience constructor — identical to __init__, provided for symmetry."""
        return cls(
            model_path=model_path,
            model_type=model_type,
            device=device,
            deterministic=deterministic,
            obs_log_path=obs_log_path,
            invert_steer=invert_steer,
            steer_deadzone=steer_deadzone,
            **model_kwargs,
        )

    def select_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Map observation to action in policy space [0, 1].

        The model outputs tanh-squashed values in [-1, 1] (SAC internal space).
        BasePolicy contract requires [0, 1], so we remap here.

        Parameters
        ----------
        obs  : np.ndarray, shape (obs_dim,)
        info : dict — raw AC state from the last step (unused here)

        Returns
        -------
        np.ndarray, shape (3,), dtype float32, values in [0, 1]
            [steer, throttle, brake]
        """
        self.last_obs = obs

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
            action_t = self.model.get_action(obs_t, deterministic=self.deterministic)
            raw = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        # Apply steer deadzone in [-1, 1] SAC space before remapping
        if self.steer_deadzone > 0.0 and abs(raw[0]) <= self.steer_deadzone:
            raw[0] = 0.0

        # Remap [-1, 1] -> [0, 1] to match BasePolicy contract
        # steerAngle>0=RIGHT in AC, so positive SAC=RIGHT, policy>0.5=RIGHT — matches VJoyControl.
        action = (raw + 1.0) / 2.0

        if self.invert_steer:
            # Flip steering: 0.0↔1.0, 0.5 stays 0.5 (straight unaffected)
            action[0] = 1.0 - action[0]

        self._write_obs_log(obs, action)
        return action

    @staticmethod
    def _obs_column_names(obs_dim: int) -> list:
        """
        Return a human-readable name for every dimension of the observation vector.

        Layout (125 dims, config: enable_sensors=True, add_previous_obs_to_state=True,
                use_target_speed=False, enable_task_id_in_obs=False, N_RAYS=11):

          [0–13]   Base sensor channels (normalised)
          [14–24]  Ray-cast wall distances  R90→FWD→L90  (11 rays)
          [25]     out_of_track flag
          [26–37]  Curvature look-ahead (12 points, next 300 m)
          [38–46]  Past 3 steering / throttle / brake values
          [47–49]  Current applied steer / throttle / brake
          [50–74]  History t-3: base channels + rays  (25 dims)
          [75–99]  History t-2: base channels + rays  (25 dims)
          [100–124] History t-1: base channels + rays (25 dims)

        Falls back to "obs_{i}" for any unexpected obs_dim.
        """
        # ── Block 1: base sensor channels (14) ───────────────────────────
        base_channels = [
            "speed_norm",           # 0  speed / 80 m/s
            "gap_norm",             # 1  lateral gap from racing line / 10 m
            "last_ff_norm",         # 2  force feedback / 1
            "rpm_norm",             # 3  RPM / 10000
            "accel_x_norm",         # 4  longitudinal G / 5
            "accel_y_norm",         # 5  lateral G / 5
            "gear_norm",            # 6  gear / 8
            "yaw_rate_norm",        # 7  angular_velocity_y / π
            "vel_x_norm",           # 8  local_velocity_x / 80 m/s
            "vel_y_norm",           # 9  local_velocity_y / 20 m/s  (sideslip)
            "slip_fl_norm",         # 10 SlipAngle_fl / 25°
            "slip_fr_norm",         # 11 SlipAngle_fr / 25°
            "slip_rl_norm",         # 12 SlipAngle_rl / 25°
            "slip_rr_norm",         # 13 SlipAngle_rr / 25°
        ]

        # ── Block 2: ray sensors (11) — index 0=R90 … 5=FWD … 10=L90 ───
        ray_labels = ["R90", "R67", "R45", "R22", "R11", "FWD", "L11", "L22", "L45", "L67", "L90"]
        ray_channels = [f"ray_{lbl}_norm" for lbl in ray_labels]

        # basic obs = base_channels + rays  (25 dims total)
        basic = base_channels + ray_channels

        # ── Full 125-dim layout ──────────────────────────────────────────
        names = (
            basic                                                       # [0–24]  current basic obs
            + ["out_of_track"]                                          # [25]
            + [f"curv_ahead_{i}" for i in range(12)]                   # [26–37] curvature look-ahead
            + [f"steer_t{t}" for t in [-3, -2, -1]]                    # [38–40] past steering
            + [f"throttle_t{t}" for t in [-3, -2, -1]]                 # [41–43] past throttle
            + [f"brake_t{t}" for t in [-3, -2, -1]]                    # [44–46] past brake
            + ["steer_applied", "throttle_applied", "brake_applied"]    # [47–49] current action
            + [f"{ch}_h3" for ch in basic]                             # [50–74]  history t-3
            + [f"{ch}_h2" for ch in basic]                             # [75–99]  history t-2
            + [f"{ch}_h1" for ch in basic]                             # [100–124] history t-1
        )

        if len(names) == obs_dim:
            return names

        # Unexpected obs_dim — fall back to generic names
        return [f"obs_{i}" for i in range(obs_dim)]

    def _write_obs_log(self, obs: np.ndarray, action: np.ndarray):
        """Append one row to the obs log CSV (no-op if obs_log_path not set)."""
        if self._obs_log_path is None:
            return
        # Lazy open — truncate on first call
        if self._obs_log_writer is None:
            path = Path(self._obs_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._obs_log_file = open(path, "w", newline="", encoding="utf-8")
            obs_cols = self._obs_column_names(len(obs))
            header = ["step"] + obs_cols + ["steer", "throttle", "brake"]
            self._obs_log_writer = csv.writer(self._obs_log_file)
            self._obs_log_writer.writerow(header)

        row = [self._step_count] + [round(float(v), 6) for v in obs] + [round(float(v), 6) for v in action]
        self._obs_log_writer.writerow(row)
        self._obs_log_file.flush()
        self._step_count += 1

    def close(self):
        """Close the obs log file if open."""
        if self._obs_log_file:
            self._obs_log_file.close()
            self._obs_log_file = None
            self._obs_log_writer = None
