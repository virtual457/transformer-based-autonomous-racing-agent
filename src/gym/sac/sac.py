"""
sac.py — Soft Actor-Critic algorithm.

Hyperparameters (held constant across all variants for valid ablation):
    gamma:          0.992
    lr:             3e-4
    hidden_units:   [256, 256, 256]
    tau:            0.005
    target_entropy: -action_dim  (auto-set if None)

Update order each call to update():
    1. Sample batch from replay buffer
    2. Compute Q-network targets (with entropy regularisation)
    3. Update Q-networks (twin)
    4. Update policy (maximise Q - alpha * log_pi)
    5. Update alpha (auto-temperature)
    6. Soft-update target Q-networks

All networks are independent of the discor/ codebase.
"""

import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .network import GaussianPolicy, TwinQNetwork
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class SAC:
    """
    Soft Actor-Critic.

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation vector.
    action_dim : int
        Number of action dimensions.
    hidden_units : list[int]
        Hidden layer widths for all networks. Default: [256, 256, 256].
    lr : float
        Learning rate for all optimizers. Default: 3e-4.
    gamma : float
        Discount factor. Default: 0.992.
    tau : float
        Soft-update coefficient for target networks. Default: 0.005.
    target_entropy : float or None
        Desired policy entropy. Defaults to -action_dim.
    device : str
        PyTorch device string. Default: 'cuda'.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_units: list = None,
        lr: float = 3e-4,
        gamma: float = 0.992,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        device: str = "cuda",
    ):
        if hidden_units is None:
            hidden_units = [256, 256, 256]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_units = list(hidden_units)
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(action_dim)
        self.device = torch.device(device)

        # Networks
        self.policy = GaussianPolicy(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
        ).to(self.device)

        self.twin_q = TwinQNetwork(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
        ).to(self.device)

        self.twin_q_target = TwinQNetwork(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_units=hidden_units,
        ).to(self.device)

        # Initialise target Q to match online Q (hard copy)
        self.twin_q_target.load_state_dict(self.twin_q.state_dict())
        for p in self.twin_q_target.parameters():
            p.requires_grad = False

        # Auto-temperature: log_alpha is the learnable parameter
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.twin_q.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        logger.info(
            f"SAC init — obs_dim={obs_dim}  action_dim={action_dim}  "
            f"hidden={hidden_units}  gamma={gamma}  tau={tau}  "
            f"target_entropy={self.target_entropy:.2f}  device={device}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature (always positive)."""
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select an action for one observation.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)
        deterministic : bool
            If True, return tanh(mean) without sampling.

        Returns
        -------
        action : np.ndarray, shape (action_dim,)  in [-1, 1]
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device).clamp(-3.0, 3.0)

        with torch.no_grad():
            if deterministic:
                _, _, mean_action = self.policy(obs_t)
                action = mean_action
            else:
                action, _, _ = self.policy(obs_t)

        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 128,
    ) -> dict:
        """
        One gradient step across all networks.

        Steps:
            1. Sample batch
            2. Compute Q targets (Bellman with entropy)
            3. Update twin Q-networks
            4. Update policy (maximise Q - alpha * log_pi)
            5. Update alpha (auto-temperature)
            6. Soft-update target Q-networks

        Returns
        -------
        dict:
            q_loss, policy_loss, alpha_loss, alpha, entropy
        """
        batch = replay_buffer.sample(batch_size, device=str(self.device))
        return self.update_from_batch(batch)

    def update_from_batch(self, batch: dict) -> dict:
        """
        One gradient step given a pre-sampled batch dict.

        Identical update logic to update(), but accepts a batch dict directly
        instead of sampling from a replay buffer.  Used by finetune_sac.py
        where the bucketed buffer performs its own stratified sampling before
        calling this method.

        Parameters
        ----------
        batch : dict
            Keys: obs, action, reward, next_obs, done.
            Each value is a torch.Tensor already on self.device.

        Returns
        -------
        dict:
            q_loss, policy_loss, alpha_loss, alpha, entropy
        """
        obs      = batch["obs"].clamp(-3.0, 3.0)        # (B, obs_dim)
        action   = batch["action"]                     # (B, action_dim)
        reward   = batch["reward"]                     # (B, 1)
        next_obs = batch["next_obs"].clamp(-3.0, 3.0)  # (B, obs_dim)
        done     = batch["done"]                       # (B, 1)

        alpha = self.alpha.detach()

        # ----------------------------------------------------------
        # 1. Q-network targets
        # ----------------------------------------------------------
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy(next_obs)
            q1_target, q2_target = self.twin_q_target(next_obs, next_action)
            min_q_target = torch.min(q1_target, q2_target)
            # Entropy-regularised Bellman target
            # done=1 means terminal — no bootstrap
            q_target = reward + self.gamma * (1.0 - done) * (
                min_q_target - alpha * next_log_prob
            )

        # ----------------------------------------------------------
        # 2. Q-network update
        # ----------------------------------------------------------
        q1, q2 = self.twin_q(obs, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.twin_q.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        # ----------------------------------------------------------
        # 3. Policy update
        # ----------------------------------------------------------
        new_action, log_prob, _ = self.policy(obs)
        q1_new, q2_new = self.twin_q(obs, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        # Maximise (Q - alpha * log_pi)  ↔  minimise -(Q - alpha * log_pi)
        policy_loss = (alpha * log_prob - min_q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # ----------------------------------------------------------
        # 4. Alpha (temperature) update
        # ----------------------------------------------------------
        # log_alpha * (-log_pi - target_entropy)  — gradient through log_alpha only
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Cap alpha at 1.0: clamp log_alpha <= 0  =>  alpha = exp(log_alpha) <= 1.0
        with torch.no_grad():
            self.log_alpha.clamp_(max=0.0)

        # ----------------------------------------------------------
        # 5. Soft-update target Q-networks
        # ----------------------------------------------------------
        self._soft_update_target()

        # ----------------------------------------------------------
        # Logging values
        # ----------------------------------------------------------
        entropy = -log_prob.detach().mean().item()

        return {
            "q_loss":      q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss":  alpha_loss.item(),
            "alpha":       self.alpha.item(),
            "entropy":     entropy,
        }

    # ------------------------------------------------------------------
    # Soft update
    # ------------------------------------------------------------------

    def _soft_update_target(self) -> None:
        """tau * online + (1 - tau) * target  for each parameter."""
        for online_p, target_p in zip(
            self.twin_q.parameters(), self.twin_q_target.parameters()
        ):
            target_p.data.mul_(1.0 - self.tau)
            target_p.data.add_(self.tau * online_p.data)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save all state dicts and hyperparameter config to path.

        The checkpoint is a single .pt file loadable with torch.load().
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "config": {
                "obs_dim":        self.obs_dim,
                "action_dim":     self.action_dim,
                "hidden_units":   self.hidden_units,
                "lr":             self.lr,
                "gamma":          self.gamma,
                "tau":            self.tau,
                "target_entropy": self.target_entropy,
            },
            "policy":             self.policy.state_dict(),
            "twin_q":             self.twin_q.state_dict(),
            "twin_q_target":      self.twin_q_target.state_dict(),
            "log_alpha":          self.log_alpha.detach().cpu(),
            "policy_optimizer":   self.policy_optimizer.state_dict(),
            "q_optimizer":        self.q_optimizer.state_dict(),
            "alpha_optimizer":    self.alpha_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"SAC checkpoint saved: {path}")

    def load(self, path: str) -> None:
        """
        Load all state dicts from a checkpoint produced by save().

        The checkpoint must have been saved with the same architecture.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.twin_q.load_state_dict(checkpoint["twin_q"])
        self.twin_q_target.load_state_dict(checkpoint["twin_q_target"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        logger.info(f"SAC checkpoint loaded: {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda") -> "SAC":
        """
        Construct a SAC instance from a checkpoint file.

        The saved config is used to reconstruct the exact architecture.
        """
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint["config"]
        agent = cls(
            obs_dim=cfg["obs_dim"],
            action_dim=cfg["action_dim"],
            hidden_units=cfg.get("hidden_units", [256, 256, 256]),
            lr=cfg.get("lr", 3e-4),
            gamma=cfg.get("gamma", 0.992),
            tau=cfg.get("tau", 0.005),
            target_entropy=cfg.get("target_entropy", None),
            device=device,
        )
        agent.load(path)
        return agent
