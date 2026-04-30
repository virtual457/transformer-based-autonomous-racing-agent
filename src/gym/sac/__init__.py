"""
gym/sac — SAC training infrastructure for Assetto Corsa.

Variant A: SAC + MLP baseline (control experiment).

Modules:
    network         — GaussianPolicy, TwinQNetwork, create_linear_network
    replay_buffer   — ReplayBuffer (ring-buffer, numpy pre-allocation)
    sac             — SAC algorithm (losses, update, save/load)
    agent           — SACAgent (collect/train phase loop, checkpointing)
"""

from .network import GaussianPolicy, TwinQNetwork, create_linear_network
from .replay_buffer import ReplayBuffer
from .sac import SAC
from .agent import SACAgent

__all__ = [
    "GaussianPolicy",
    "TwinQNetwork",
    "create_linear_network",
    "ReplayBuffer",
    "SAC",
    "SACAgent",
]
