"""
sac.py — Soft Actor-Critic with Transformer encoder (Variant D).

Architecture:
    Two independent TransformerEncoders process an observation window of shape
    (window_size, token_dim) -> (B, d_model) embedding.

    Critic path:  critic_encoder (online) -> TwinQHead(embedding, action)
    Policy path:  policy_encoder (online) -> PolicyHead

    policy_encoder is trained by policy gradient directly — no detach.
    critic_encoder is trained by critic loss only.
    critic_encoder_target: soft-updated copy of critic_encoder used to compute
    stable Bellman targets.  Updated with the same tau as the Q-networks.

Update order (identical to gym/sac/sac.py):
    1. Encode next_obs_seq with critic_encoder_target (no grad)
    2. Compute entropy-regularised Bellman target
    3. Update Q-networks (encode obs_seq with critic_encoder, through critic)
    4. Update policy (encode obs_seq with policy_encoder, through policy head — no detach)
    5. Update alpha (auto-temperature)
    6. Soft-update critic_encoder_target + target Q-heads

SAC hyperparameters (held constant — do not tune mid-project):
    gamma:          0.992
    lr:             3e-4
    tau:            0.005
    target_entropy: -3.0  (= -action_dim)
    batch_size:     256
"""

import os
import copy
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .network import TransformerEncoder, PolicyHead, TwinQHead
from .replay_buffer import DualWindowReplayBuffer

logger = logging.getLogger(__name__)


class TransformerSAC:
    """
    SAC with separate policy and critic Transformer encoders.

    Parameters
    ----------
    token_dim : int
        Dimensionality of each input token (50).
    action_dim : int
        Number of action dimensions (3: steer, throttle, brake).
    window_size : int
        Number of frames in each observation window (75).
    d_model : int
        Transformer internal width (256).
    n_heads : int
        Number of attention heads (4).
    n_layers : int
        Number of encoder layers (4).
    ffn_dim : int
        Feed-forward hidden dimension in each layer (1024).
    policy_hidden : list[int]
        Hidden units for PolicyHead MLP (default: [256]).
    q_hidden : list[int]
        Hidden units for each Q-head MLP (default: [256]).
    lr : float
        Learning rate for all optimizers (3e-4).
    gamma : float
        Discount factor (0.992).
    tau : float
        Soft-update coefficient (0.005).
    target_entropy : float
        Desired policy entropy (-3.0).
    device : str
        PyTorch device string.
    """

    def __init__(
        self,
        token_dim: int = 50,
        action_dim: int = 3,
        window_size: int = 75,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        policy_hidden: list = None,
        q_hidden: list = None,
        lr: float = 3e-4,
        gamma: float = 0.992,
        tau: float = 0.005,
        target_entropy: float = -3.0,
        device: str = "cuda",
    ):
        if policy_hidden is None:
            policy_hidden = [256]
        if q_hidden is None:
            q_hidden = [256]

        self.token_dim = token_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.policy_hidden = list(policy_hidden)
        self.q_hidden = list(q_hidden)
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = float(target_entropy)
        self.device = torch.device(device)

        # ── Networks ──────────────────────────────────────────────────────────

        # Policy encoder: trained by policy gradient (no detach on its embeddings).
        self.policy_encoder = TransformerEncoder(
            token_dim=token_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            window_size=window_size,
        ).to(self.device)

        # Critic encoder: trained by critic loss only.
        self.critic_encoder = TransformerEncoder(
            token_dim=token_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            window_size=window_size,
        ).to(self.device)

        # Target critic encoder: soft-copied from critic_encoder, used for Bellman targets.
        self.critic_encoder_target = copy.deepcopy(self.critic_encoder).to(self.device)
        for p in self.critic_encoder_target.parameters():
            p.requires_grad = False

        # Policy head — gradients flow through policy_encoder.
        self.policy_head = PolicyHead(
            d_model=d_model,
            action_dim=action_dim,
            hidden_units=policy_hidden,
        ).to(self.device)

        # Online twin Q-head.
        self.twin_q = TwinQHead(
            d_model=d_model,
            action_dim=action_dim,
            hidden_units=q_hidden,
        ).to(self.device)

        # Target twin Q-head.
        self.twin_q_target = copy.deepcopy(self.twin_q).to(self.device)
        for p in self.twin_q_target.parameters():
            p.requires_grad = False

        # ── Auto-temperature ──────────────────────────────────────────────────
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        # ── Optimizers ────────────────────────────────────────────────────────
        # Critic optimizer: critic_encoder + twin_q parameters together so
        # the critic loss trains both.
        self.q_optimizer = optim.Adam(
            list(self.critic_encoder.parameters()) + list(self.twin_q.parameters()),
            lr=lr,
        )
        # Policy optimizer: policy_encoder + policy_head parameters so
        # the policy gradient flows through both.
        self.policy_optimizer = optim.Adam(
            list(self.policy_encoder.parameters()) + list(self.policy_head.parameters()),
            lr=lr,
        )
        self.alpha_optimizer  = optim.Adam([self.log_alpha], lr=lr)

        logger.info(
            f"TransformerSAC init — "
            f"token_dim={token_dim}  action_dim={action_dim}  window_size={window_size}  "
            f"d_model={d_model}  n_heads={n_heads}  n_layers={n_layers}  ffn_dim={ffn_dim}  "
            f"gamma={gamma}  tau={tau}  target_entropy={target_entropy}  device={device}"
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
        obs_window: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select an action for a single observation window.

        Parameters
        ----------
        obs_window : np.ndarray, shape (window_size, token_dim)
            Rolling window of the last window_size observation tokens.
        deterministic : bool
            If True, return tanh(mean) without sampling.

        Returns
        -------
        action : np.ndarray, shape (action_dim,)  in [-1, 1]
        """
        # Add batch dimension: (1, W, T)
        x = torch.as_tensor(obs_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Clamp tokens to [-3, 3] for stability (same as flat SAC).
        x = x.clamp(-3.0, 3.0)

        with torch.no_grad():
            embedding = self.policy_encoder(x)               # (1, d_model)
            if deterministic:
                _, _, mean_action = self.policy_head(embedding)
                action = mean_action
            else:
                action, _, _ = self.policy_head(embedding)

        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        replay_buffer,
        batch_size: int = 256,
    ) -> dict:
        """
        Sample from replay buffer and run one gradient step.

        Parameters
        ----------
        replay_buffer : DualWindowReplayBuffer
        batch_size : int

        Returns
        -------
        dict: q_loss, policy_loss, alpha_loss, alpha, entropy
        """
        batch = replay_buffer.sample(batch_size, device=str(self.device))
        return self.update_from_batch(batch)

    def update_from_batch(self, batch: dict) -> dict:
        """
        One gradient step given a pre-sampled batch dict.

        Parameters
        ----------
        batch : dict
            Keys: obs_seq (B, W, T), action (B, 3), reward (B, 1),
                  next_obs_seq (B, W, T), done (B, 1).
            All tensors already on self.device.

        Returns
        -------
        dict: q_loss, policy_loss, alpha_loss, alpha, entropy
        """
        obs_seq      = batch["obs_seq"].clamp(-3.0, 3.0)       # (B, W, token_dim)
        action       = batch["action"]                          # (B, action_dim)
        reward       = batch["reward"]                          # (B, 1)
        next_obs_seq = batch["next_obs_seq"].clamp(-3.0, 3.0)  # (B, W, token_dim)
        done         = batch["done"]                            # (B, 1)

        alpha = self.alpha.detach()

        # ------------------------------------------------------------------
        # 1. Q-network targets (target critic encoder, no grad)
        # ------------------------------------------------------------------
        with torch.no_grad():
            # policy_encoder produces the embedding policy_head was trained on.
            next_emb_policy = self.policy_encoder(next_obs_seq)           # (B, d_model)
            next_action, next_log_prob, _ = self.policy_head(next_emb_policy)
            # critic_encoder_target produces the embedding the Q-heads were trained on.
            next_emb_critic_t = self.critic_encoder_target(next_obs_seq)  # (B, d_model)
            q1_t, q2_t = self.twin_q_target(next_emb_critic_t, next_action)
            min_q_t = torch.min(q1_t, q2_t)
            # Entropy-regularised Bellman target.
            q_target = reward + self.gamma * (1.0 - done) * (
                min_q_t - alpha * next_log_prob
            )

        # ------------------------------------------------------------------
        # 2. Q-network update (critic_encoder + twin_q)
        # ------------------------------------------------------------------
        obs_emb = self.critic_encoder(obs_seq)     # (B, d_model) — gradient flows into critic_encoder
        q1, q2  = self.twin_q(obs_emb, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss  = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_encoder.parameters()) + list(self.twin_q.parameters()),
            max_norm=1.0,
        )
        self.q_optimizer.step()

        # ------------------------------------------------------------------
        # 3. Policy update (policy_encoder trained here — no detach needed)
        # ------------------------------------------------------------------
        # Encode with policy_encoder: gradient flows naturally into it.
        obs_emb_policy = self.policy_encoder(obs_seq)  # (B, d_model)

        new_action, log_prob, _ = self.policy_head(obs_emb_policy)
        # Q-value for the new action (through critic_encoder — detach so critic
        # is not inadvertently trained by the policy loss).
        with torch.no_grad():
            obs_emb_critic_det = self.critic_encoder(obs_seq)
        q1_new, q2_new = self.twin_q(obs_emb_critic_det, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        # Minimise -(Q - alpha * log_pi)
        policy_loss = (alpha * log_prob - min_q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_encoder.parameters()) + list(self.policy_head.parameters()),
            max_norm=1.0,
        )
        self.policy_optimizer.step()

        # ------------------------------------------------------------------
        # 4. Alpha (temperature) update
        # ------------------------------------------------------------------
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Cap alpha at 1.0: clamp log_alpha <= 0  =>  alpha = exp(log_alpha) <= 1.0
        with torch.no_grad():
            self.log_alpha.clamp_(max=0.0)

        # ------------------------------------------------------------------
        # 5. Soft-update target encoder and target Q-heads
        # ------------------------------------------------------------------
        self._soft_update_targets()

        # ------------------------------------------------------------------
        # Logging values
        # ------------------------------------------------------------------
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

    def _soft_update_targets(self) -> None:
        """tau * online + (1 - tau) * target for each parameter.
        Only critic_encoder_target is soft-updated; policy_encoder has no target."""
        for online_p, target_p in zip(
            self.critic_encoder.parameters(), self.critic_encoder_target.parameters()
        ):
            target_p.data.mul_(1.0 - self.tau)
            target_p.data.add_(self.tau * online_p.data)

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

        Format is identical in spirit to gym/sac/sac.py so load() is symmetric.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "config": {
                "token_dim":      self.token_dim,
                "action_dim":     self.action_dim,
                "window_size":    self.window_size,
                "d_model":        self.d_model,
                "n_heads":        self.n_heads,
                "n_layers":       self.n_layers,
                "ffn_dim":        self.ffn_dim,
                "policy_hidden":  self.policy_hidden,
                "q_hidden":       self.q_hidden,
                "lr":             self.lr,
                "gamma":          self.gamma,
                "tau":            self.tau,
                "target_entropy": self.target_entropy,
            },
            "policy_encoder":        self.policy_encoder.state_dict(),
            "critic_encoder":        self.critic_encoder.state_dict(),
            "critic_encoder_target": self.critic_encoder_target.state_dict(),
            "policy_head":           self.policy_head.state_dict(),
            "twin_q":                self.twin_q.state_dict(),
            "twin_q_target":         self.twin_q_target.state_dict(),
            "log_alpha":             self.log_alpha.detach().cpu(),
            "q_optimizer":           self.q_optimizer.state_dict(),
            "policy_optimizer":      self.policy_optimizer.state_dict(),
            "alpha_optimizer":       self.alpha_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"TransformerSAC checkpoint saved: {path}")

    def load(self, path: str) -> None:
        """
        Load all state dicts from a checkpoint produced by save().

        The checkpoint must have been saved with the same architecture.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_encoder.load_state_dict(checkpoint["policy_encoder"])
        self.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        self.critic_encoder_target.load_state_dict(checkpoint["critic_encoder_target"])
        self.policy_head.load_state_dict(checkpoint["policy_head"])
        self.twin_q.load_state_dict(checkpoint["twin_q"])
        self.twin_q_target.load_state_dict(checkpoint["twin_q_target"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        logger.info(f"TransformerSAC checkpoint loaded: {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda") -> "TransformerSAC":
        """
        Construct a TransformerSAC instance from a checkpoint file.

        The saved config is used to reconstruct the exact architecture.
        """
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint["config"]
        agent = cls(
            token_dim=cfg["token_dim"],
            action_dim=cfg["action_dim"],
            window_size=cfg.get("window_size", 75),
            d_model=cfg.get("d_model", 256),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 4),
            ffn_dim=cfg.get("ffn_dim", 1024),
            policy_hidden=cfg.get("policy_hidden", [256]),
            q_hidden=cfg.get("q_hidden", [256]),
            lr=cfg.get("lr", 3e-4),
            gamma=cfg.get("gamma", 0.992),
            tau=cfg.get("tau", 0.005),
            target_entropy=cfg.get("target_entropy", -3.0),
            device=device,
        )
        agent.load(path)
        return agent
