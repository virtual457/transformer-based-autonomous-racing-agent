"""
network.py — Neural network architectures for SAC.

GaussianPolicy:
    MLP that outputs a tanh-squashed Gaussian action.
    Returns (action, log_prob, mean) where all are in [-1, 1].
    log_prob is corrected for the tanh squash.

TwinQNetwork:
    Two independent Q-networks (clipped double-Q trick).
    Input: concatenated (state, action).
    Output: scalar Q-value each.

Both are independent of the discor/ codebase — do NOT import from there.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2
# Small constant for numerical stability in log(1 - tanh^2(x))
TANH_LOG_EPSILON = 1e-6


def create_linear_network(
    input_dim: int,
    output_dim: int,
    hidden_units: list,
    activation=nn.ReLU,
) -> nn.Sequential:
    """
    Build a fully-connected MLP with Xavier-uniform weight initialisation.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector.
    output_dim : int
        Size of the output vector.
    hidden_units : list[int]
        Width of each hidden layer (e.g. [256, 256, 256]).
    activation : nn.Module class
        Activation function inserted after each hidden layer (default: ReLU).

    Returns
    -------
    nn.Sequential
        The constructed MLP (activation applied after every hidden layer;
        NO activation after the final linear layer).
    """
    layers = []
    in_dim = input_dim

    for hidden_dim in hidden_units:
        linear = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(activation())
        in_dim = hidden_dim

    final = nn.Linear(in_dim, output_dim)
    nn.init.xavier_uniform_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)

    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """
    Tanh-squashed Gaussian policy.

    Architecture:
        state  →  MLP(hidden_units)  →  Linear(2 * action_dim)
                                        ├── mean
                                        └── log_std  (clamped to [LOG_STD_MIN, LOG_STD_MAX])

    The policy samples x ~ N(mean, std), then applies tanh to get an action in [-1, 1].
    The log-probability is corrected for the tanh squash.

    Parameters
    ----------
    state_dim : int
        Dimension of the input state vector.
    action_dim : int
        Number of action dimensions (3 for steer/throttle/brake).
    hidden_units : list[int]
        Hidden layer widths (default: [256, 256, 256]).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: list = None,
    ):
        super().__init__()
        if hidden_units is None:
            hidden_units = [256, 256, 256]

        self.action_dim = action_dim
        # Output: mean + log_std for each action dimension
        self.net = create_linear_network(
            input_dim=state_dim,
            output_dim=action_dim * 2,
            hidden_units=hidden_units,
        )

    def forward(self, state: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        state : torch.Tensor, shape (batch, state_dim)

        Returns
        -------
        action : torch.Tensor, shape (batch, action_dim)  — in [-1, 1]
        log_prob : torch.Tensor, shape (batch, 1)
        mean : torch.Tensor, shape (batch, action_dim)  — tanh(mean) in [-1, 1]
        """
        out = self.net(state)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)
        # Reparametrised sample
        x = dist.rsample()
        action = torch.tanh(x)

        # log-prob corrected for tanh squash:
        #   log pi(a|s) = log N(x|mu,sigma) - sum log(1 - tanh^2(x) + eps)
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1.0 - action.pow(2) + TANH_LOG_EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (batch, 1)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

    def sample(self, state: torch.Tensor):
        """Alias for forward() — kept for readability at call sites."""
        return self.forward(state)


class TwinQNetwork(nn.Module):
    """
    Two independent Q-networks (clipped double-Q trick).

    Input: concatenated (state, action) → each network outputs a scalar Q-value.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_units : list[int]
        Default: [256, 256, 256]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: list = None,
    ):
        super().__init__()
        if hidden_units is None:
            hidden_units = [256, 256, 256]

        in_dim = state_dim + action_dim
        self.q1_net = create_linear_network(
            input_dim=in_dim,
            output_dim=1,
            hidden_units=hidden_units,
        )
        self.q2_net = create_linear_network(
            input_dim=in_dim,
            output_dim=1,
            hidden_units=hidden_units,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Compute both Q-values.

        Parameters
        ----------
        state  : (batch, state_dim)
        action : (batch, action_dim)  — expected in [-1, 1]

        Returns
        -------
        q1, q2 : each (batch, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 — used for policy gradient computation."""
        x = torch.cat([state, action], dim=-1)
        return self.q1_net(x)
