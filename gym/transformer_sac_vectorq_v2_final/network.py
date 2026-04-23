"""
network.py — Neural network architectures for Transformer SAC (Vector Q variant).

Architecture overview:
    - TransformerEncoder: projects token_dim -> d_model, applies positional
      embedding, runs a Pre-LN TransformerEncoder, mean-pools all 75 tokens
      to produce a single (B, d_model) embedding.
    - PolicyHead: MLP(d_model -> hidden -> action_dim*2) -> mean + log_std
      -> tanh-squashed action with exact log_prob correction copied from
      gym/sac/network.py GaussianPolicy.
    - TwinQHead: two independent MLPs, each taking cat(embedding, action)
      -> scalar Q-value.

Design decisions (locked):
    d_model = 256, n_heads = 4, n_layers = 4, ffn_dim = 1024
    norm_first = True (Pre-LN for stability)
    window_size = 75, token_dim = 50
    Mean pool over all 75 tokens — not last-token — for better gradient flow.
    Policy uses detached embedding so critic updates train the encoder.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

# ── log_std bounds (identical to gym/sac/network.py) ──────────────────────────
LOG_STD_MIN = -20
LOG_STD_MAX = 2
# Small constant for numerical stability in log(1 - tanh^2(x))
TANH_LOG_EPSILON = 1e-6


# ── Shared MLP builder ─────────────────────────────────────────────────────────

def _make_mlp(input_dim: int, output_dim: int, hidden_units: list) -> nn.Sequential:
    """
    Build a fully-connected MLP with Xavier-uniform weight initialisation.
    No activation after the final linear layer (matches gym/sac/network.py).
    """
    layers = []
    in_dim = input_dim
    for h in hidden_units:
        linear = nn.Linear(in_dim, h)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.ReLU())
        in_dim = h
    final = nn.Linear(in_dim, output_dim)
    nn.init.xavier_uniform_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


# ── TransformerEncoder ─────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Encodes a sequence of observation tokens into a single embedding vector.

    Pipeline:
        1. Linear projection: token_dim -> d_model
        2. Add learned positional embedding (shape: window_size x d_model)
        3. Pre-LN TransformerEncoder (norm_first=True)
        4. Mean pool over all tokens -> (B, d_model)

    Parameters
    ----------
    token_dim : int
        Dimensionality of each input token (50 for this project: obs[:50]).
    d_model : int
        Internal transformer width (256).
    n_heads : int
        Number of attention heads (4).
    n_layers : int
        Number of encoder layers (4).
    ffn_dim : int
        Feed-forward hidden dimension (1024 = d_model * 4).
    window_size : int
        Number of tokens in each sequence (75 = 3 seconds at 25 Hz).
    dropout : float
        Dropout probability inside transformer layers (default 0.0).
    """

    def __init__(
        self,
        token_dim: int = 50,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        window_size: int = 75,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model
        self.window_size = window_size

        # Project raw token to model dimension.
        self.input_proj = nn.Linear(token_dim, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Learned positional embedding — one vector per position.
        self.pos_embedding = nn.Embedding(window_size, d_model)

        # Pre-LN transformer encoder (norm_first=True — more stable early training).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, window_size, token_dim)

        Returns
        -------
        embedding : torch.Tensor, shape (B, d_model)
        """
        B, W, _ = x.shape
        # Positional indices: (1, W) broadcast over batch.
        positions = torch.arange(W, device=x.device).unsqueeze(0)  # (1, W)
        # Project tokens and add positional embedding.
        x = self.input_proj(x) + self.pos_embedding(positions)     # (B, W, d_model)
        # Run transformer.
        x = self.transformer(x)                                     # (B, W, d_model)
        # Mean pool over the sequence dimension.
        embedding = x.mean(dim=1)                                   # (B, d_model)
        return embedding


# ── PolicyHead ─────────────────────────────────────────────────────────────────

class PolicyHead(nn.Module):
    """
    Tanh-squashed Gaussian policy head.

    Takes a (B, d_model) embedding (typically detached from the encoder's
    computation graph) and outputs an action in [-1, 1] with its log-prob.

    The log_prob correction for the tanh squash is copied exactly from
    gym/sac/network.py GaussianPolicy.

    Parameters
    ----------
    d_model : int
        Input embedding dimension (256).
    action_dim : int
        Number of action dimensions (3: steer, throttle, brake).
    hidden_units : list[int]
        Hidden layer widths (default: [256]).
    """

    def __init__(
        self,
        d_model: int = 256,
        action_dim: int = 3,
        hidden_units: list = None,
    ):
        super().__init__()
        if hidden_units is None:
            hidden_units = [256]
        self.action_dim = action_dim
        # Output: mean + log_std concatenated (2 * action_dim outputs).
        self.net = _make_mlp(
            input_dim=d_model,
            output_dim=action_dim * 2,
            hidden_units=hidden_units,
        )

    def forward(self, embedding: torch.Tensor):
        """
        Parameters
        ----------
        embedding : torch.Tensor, shape (B, d_model)
            May be live (policy update path, gradients flow into policy_encoder)
            or detached (critic update path).  The caller controls detachment.

        Returns
        -------
        action      : (B, action_dim)  in [-1, 1]
        log_prob    : (B, action_dim)  per-action log probability (NOT summed)
        mean_action : (B, action_dim)  tanh(mean) in [-1, 1]
        """
        out = self.net(embedding)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)
        # Reparametrised sample.
        x = dist.rsample()
        action = torch.tanh(x)

        # log-prob corrected for tanh squash, kept per-action (B, action_dim).
        # Callers sum with per-action alpha weights.
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1.0 - action.pow(2) + TANH_LOG_EPSILON)
        # shape: (B, action_dim) — NOT summed

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

    def sample(self, embedding: torch.Tensor):
        """Alias for forward() — kept for readability at call sites."""
        return self.forward(embedding)


# ── TwinQHead ──────────────────────────────────────────────────────────────────

class TwinQHead(nn.Module):
    """
    Two independent Q-networks (clipped double-Q trick) with vector output.

    Input: cat(embedding, action) where embedding is (B, d_model) and
    action is (B, action_dim).  Each network outputs a 3-dim vector:
    [Q_steer, Q_throttle, Q_brake] — one value per action channel.

    The last layer is Linear(hidden, reward_dim) where reward_dim = 3.
    Each output neuron receives gradient only from its own Bellman target.
    Shared hidden layers receive merged gradients from all channels.

    Parameters
    ----------
    d_model : int
        Embedding dimension (256).
    action_dim : int
        Number of action dimensions (3).
    reward_dim : int
        Number of reward channels (3: steer, throttle, brake).
    hidden_units : list[int]
        Hidden layer widths (default: [256]).
    """

    def __init__(
        self,
        d_model: int = 256,
        action_dim: int = 3,
        reward_dim: int = 3,
        hidden_units: list = None,
    ):
        super().__init__()
        if hidden_units is None:
            hidden_units = [256]
        in_dim = d_model + action_dim
        self.reward_dim = reward_dim
        self.q1_net = _make_mlp(in_dim, reward_dim, hidden_units)
        self.q2_net = _make_mlp(in_dim, reward_dim, hidden_units)

    def forward(
        self, embedding: torch.Tensor, action: torch.Tensor
    ):
        """
        Parameters
        ----------
        embedding : (B, d_model)
        action    : (B, action_dim)

        Returns
        -------
        q1, q2 : each (B, reward_dim)
        """
        x = torch.cat([embedding, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def q1(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 — used for policy gradient computation."""
        x = torch.cat([embedding, action], dim=-1)
        return self.q1_net(x)
