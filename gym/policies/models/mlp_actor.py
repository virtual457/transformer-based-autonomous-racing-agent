"""
MLPActor — behaviour-cloning / SAC-checkpoint MLP policy model.

Wraps GaussianPolicy from assetto_corsa_gym for inference only.
Supports two checkpoint formats:
  1. New format (written by BaseModel.save):
       {"state_dict": ..., "config": {"obs_dim": ..., "action_dim": ..., "hidden_units": [...]}}
  2. Legacy SAC format (raw state_dict only):
       torch.save(actor.state_dict(), path)
       obs_dim is inferred from the first Linear layer weight shape.

Training (BC) writes format 1 via save(). Existing SAC checkpoints
from assetto_corsa_gym use format 2 and are handled transparently.
"""

import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from policies.models.base_model import BaseModel

# ---------------------------------------------------------------------------
# Resolve path to GaussianPolicy inside assetto_corsa_gym.
# This is the canonical actor used by the SAC implementation — reuse it
# for inference so checkpoints are byte-for-byte compatible.
# ---------------------------------------------------------------------------
_DISCOR_PATH = str(
    Path(__file__).resolve().parents[4]
    / "assetto_corsa_gym"
    / "algorithm"
    / "discor"
)
if _DISCOR_PATH not in sys.path:
    sys.path.insert(0, _DISCOR_PATH)

from discor.network import GaussianPolicy  # noqa: E402  (path manipulation above)


class MLPActor(BaseModel):
    """Gaussian MLP policy for Assetto Corsa continuous control.

    Architecture: GaussianPolicy(obs_dim -> hidden_units -> 2*action_dim)
    Output: tanh(mean) in [-1, 1] for steer / throttle-pedal / brake.

    Parameters
    ----------
    obs_dim : int
        Observation vector size (125 with add_previous_obs_to_state=True).
    action_dim : int
        Action dimension — 3 for [steer, pedal, brake].
    hidden_units : list of int
        MLP hidden layer sizes. Default [256, 256, 256] matches SAC config.
    device : str or torch.device
        Device to place weights on. "auto" -> CUDA if available, else CPU.
    """

    DEFAULT_HIDDEN = [256, 256, 256]

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 3,
        hidden_units: List[int] = None,
        device: str = "auto",
    ):
        super().__init__()
        if hidden_units is None:
            hidden_units = self.DEFAULT_HIDDEN

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_units = list(hidden_units)

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self.net = GaussianPolicy(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
        ).to(self._device)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return action tensor of shape (1, action_dim) in [-1, 1].

        Parameters
        ----------
        obs : torch.Tensor, shape (1, obs_dim), on self._device
        deterministic : bool
            True  -> tanh(mean), no randomness — use during evaluation.
            False -> sample from Normal(mean, std) then apply tanh.
        """
        with torch.no_grad():
            # GaussianPolicy.forward() returns (sampled_actions, entropies, tanh_means)
            sampled, _entropies, tanh_means = self.net(obs)
            return tanh_means if deterministic else sampled

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "MLPActor":
        """Load MLPActor from checkpoint.

        Handles both checkpoint formats:
        - New: dict with "config" and "state_dict" keys
        - Legacy SAC: raw state_dict (obs_dim inferred from weight shape)

        kwargs accepted:
            device    : str   — override device (default "auto")
            obs_dim   : int   — override obs_dim (for legacy checkpoints)
            action_dim: int   — override action_dim (legacy checkpoints)
            hidden_units: list — override hidden_units (legacy checkpoints)
        """
        device = kwargs.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(path, map_location=device)

        if isinstance(ckpt, dict) and "config" in ckpt:
            # --- New format: config embedded ---
            cfg = ckpt["config"]
            obs_dim = kwargs.get("obs_dim", cfg["obs_dim"])
            action_dim = kwargs.get("action_dim", cfg.get("action_dim", 3))
            hidden_units = kwargs.get("hidden_units", cfg.get("hidden_units", cls.DEFAULT_HIDDEN))
            state_dict = ckpt["state_dict"]
        else:
            # --- Legacy SAC format: raw state_dict ---
            # Keys look like: "net.0.weight", "net.0.bias", "net.2.weight", ...
            # Infer obs_dim from the first Linear layer weight: shape (hidden_0, obs_dim)
            state_dict = ckpt
            first_weight_key = next(k for k in state_dict if "weight" in k)
            obs_dim = kwargs.get("obs_dim", state_dict[first_weight_key].shape[1])
            action_dim = kwargs.get("action_dim", 3)
            hidden_units = kwargs.get("hidden_units", cls.DEFAULT_HIDDEN)

        model = cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
            device=device,
        )
        model.net.load_state_dict(state_dict)
        model.net.eval()
        return model

    def get_config(self) -> dict:
        return {
            "model_type": "mlp",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_units": self.hidden_units,
        }

    def save(self, path: str) -> None:
        """Save model to path as {"state_dict": ..., "config": ...}.

        Overrides BaseModel.save() to store the inner GaussianPolicy
        state_dict (without the "net." prefix), ensuring compatibility with
        legacy SAC checkpoints and with from_checkpoint().
        """
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {"state_dict": self.net.state_dict(), "config": self.get_config()},
            path,
        )

    # ------------------------------------------------------------------
    # nn.Module forward — delegates to GaussianPolicy
    # ------------------------------------------------------------------

    def forward(self, states: torch.Tensor):
        """Full forward pass returning (sampled_actions, entropies, tanh_means).

        Used by train_bc.py which needs the mean for MSE loss computation.
        """
        return self.net(states)
