"""
sac.py — Soft Actor-Critic with Transformer encoder (Vector Q variant).

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
from .replay_buffer import SixChannelMemmapBuffer

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
        self.device = torch.device(device)
        # target_entropy may be a scalar (broadcast across all action channels) OR
        # a per-channel list/tuple/array of length action_dim.
        if isinstance(target_entropy, (list, tuple)) or (
            hasattr(target_entropy, "__len__") and not isinstance(target_entropy, str)
        ):
            _te = list(target_entropy)
            if len(_te) != action_dim:
                raise ValueError(
                    f"target_entropy list length {len(_te)} != action_dim {action_dim}"
                )
            self.target_entropy = torch.tensor(
                _te, dtype=torch.float32, device=self.device
            )
        else:
            self.target_entropy = float(target_entropy)

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

        # ── Auto-temperature (per-action) ─────────────────────────────────────
        # One log_alpha per action dimension — each dimension auto-tunes its own
        # entropy independently, all targeting the same target_entropy.
        self.log_alpha = torch.zeros(action_dim, requires_grad=True, device=self.device)

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
        """Per-action temperatures, shape (action_dim,), always positive."""
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs_window: np.ndarray,
        deterministic: bool = False,
    ) -> tuple:
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
        mean   : np.ndarray, shape (action_dim,)  tanh(mean) in [-1, 1]
        std    : np.ndarray, shape (action_dim,)  pre-tanh std > 0
        """
        from .network import LOG_STD_MIN, LOG_STD_MAX
        from torch.distributions import Normal

        # Add batch dimension: (1, W, T)
        x = torch.as_tensor(obs_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        x = x.clamp(-3.0, 3.0)

        self.policy_encoder.eval()
        self.policy_head.eval()
        with torch.no_grad():
            embedding = self.policy_encoder(x)                        # (1, d_model)
            out = self.policy_head.net(embedding)
            mean_raw, log_std = out.chunk(2, dim=-1)
            log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
            std = log_std.exp()
            if deterministic:
                action = torch.tanh(mean_raw)
            else:
                action = torch.tanh(Normal(mean_raw, std).rsample())
        self.policy_encoder.train()
        self.policy_head.train()

        return (
            action.squeeze(0).cpu().numpy(),
            torch.tanh(mean_raw).squeeze(0).cpu().numpy(),
            std.squeeze(0).cpu().numpy(),
        )

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

        Vector Q variant: reward is (B, 3), Q outputs are (B, 3).
        Each Q channel [steer, throttle, brake] has its own Bellman target.

        Parameters
        ----------
        batch : dict
            Keys: obs_seq (B, W, T), action (B, 3), reward (B, 3),
                  next_obs_seq (B, W, T), done (B, 1).
            All tensors already on self.device.

        Returns
        -------
        dict: q_loss, policy_loss, alpha_loss, alpha, entropy, per-channel losses
        """
        obs_seq      = batch["obs_seq"].clamp(-3.0, 3.0)       # (B, W, token_dim)
        action       = batch["action"]                          # (B, action_dim)
        reward       = batch["reward"]                          # (B, 3) — vector reward
        next_obs_seq = batch["next_obs_seq"].clamp(-3.0, 3.0)  # (B, W, token_dim)
        done         = batch["done"]                            # (B, 1)

        alpha = self.alpha.detach()   # (action_dim,)

        # ------------------------------------------------------------------
        # 1. Q-network targets — per-channel Bellman (target critic encoder)
        # ------------------------------------------------------------------
        with torch.no_grad():
            next_emb_policy = self.policy_encoder(next_obs_seq)                # (B, d_model)
            next_action, next_log_prob, _ = self.policy_head(next_emb_policy)  # (B, action_dim)
            next_emb_critic_t = self.critic_encoder_target(next_obs_seq)       # (B, d_model)
            q1_t, q2_t = self.twin_q_target(next_emb_critic_t, next_action)   # each (B, 3)
            min_q_t = torch.min(q1_t, q2_t)                                   # (B, 3)

            # Per-channel entropy term: alpha_d * log_prob_d  → (B, 3)
            next_entropy_term = alpha.unsqueeze(0) * next_log_prob             # (B, 3)

            # Per-channel Bellman target: (B, 3)
            # done is (B, 1) — broadcasts across 3 channels.
            q_target = reward + self.gamma * (1.0 - done) * (
                min_q_t - next_entropy_term
            )
            q_max = 1.0 / (1.0 - self.gamma)   # 125.0 for gamma=0.992
            q_target = q_target.clamp(-q_max, q_max)

        # ------------------------------------------------------------------
        # 2. Q-network update — per-channel MSE
        # ------------------------------------------------------------------
        obs_emb = self.critic_encoder(obs_seq)               # (B, d_model)
        obs_emb_critic_det = obs_emb.detach()
        q1, q2 = self.twin_q(obs_emb, action)               # each (B, 3)

        # Per-channel MSE (averaged over batch AND channels).
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
        # 3. Policy update — per-channel pi_loss
        # ------------------------------------------------------------------
        obs_emb_policy = self.policy_encoder(obs_seq)                         # (B, d_model)
        new_action, log_prob, _ = self.policy_head(obs_emb_policy)            # (B, action_dim)
        q1_new, q2_new = self.twin_q(obs_emb_critic_det, new_action)         # each (B, 3)
        min_q_new = torch.min(q1_new, q2_new)                                # (B, 3)

        # Per-channel policy loss: alpha_d * log_prob_d - Q_d  for each d.
        # Shape: (B, 3) → mean over batch and channels.
        per_channel_pi = alpha.unsqueeze(0) * log_prob - min_q_new            # (B, 3)
        policy_loss = per_channel_pi.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_encoder.parameters()) + list(self.policy_head.parameters()),
            max_norm=1.0,
        )
        self.policy_optimizer.step()

        # ------------------------------------------------------------------
        # 4. Alpha (temperature) update — unchanged, per-action
        # ------------------------------------------------------------------
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy).mean(dim=0)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha.clamp_(min=-5.0, max=0.0)

        # ------------------------------------------------------------------
        # 5. Soft-update target encoder and target Q-heads
        # ------------------------------------------------------------------
        self._soft_update_targets()

        # ------------------------------------------------------------------
        # Logging values
        # ------------------------------------------------------------------
        per_action_entropy = (-log_prob.detach().mean(dim=0).cpu()).tolist()
        entropy = float(sum(per_action_entropy) / len(per_action_entropy))

        # Per-channel Q loss for diagnostics.
        with torch.no_grad():
            q1_per_ch = ((q1 - q_target) ** 2).mean(dim=0).cpu().tolist()  # [steer, throttle, brake]

        return {
            "q_loss":             q_loss.item(),
            "q_loss_per_channel": q1_per_ch,     # [steer, throttle, brake]
            "policy_loss":        policy_loss.item(),
            "alpha_loss":         alpha_loss.item(),
            "alpha":              self.alpha.mean().item(),
            "alpha_per_action":   self.alpha.detach().cpu().tolist(),
            "entropy":            entropy,
            "entropy_per_action": per_action_entropy,
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
        saved_log_alpha = checkpoint["log_alpha"].to(self.device)
        if saved_log_alpha.shape != self.log_alpha.shape:
            # Old scalar checkpoint → broadcast value to all per-action dims.
            self.log_alpha.data.fill_(saved_log_alpha.flatten()[0].item())
            logger.info("log_alpha shape mismatch — broadcast scalar to per-action vector.")
        else:
            self.log_alpha.data.copy_(saved_log_alpha)
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        # alpha_optimizer is always reset fresh — its state depends on log_alpha shape
        # which may have changed (scalar → per-action vector).
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        logger.info("alpha_optimizer reset fresh (not restored from checkpoint).")
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
