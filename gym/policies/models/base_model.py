"""
BaseModel — ABC that every policy model must implement.

Design principle: each model in this package is fully self-contained.
It owns its architecture, checkpoint format, loading logic, and action
inference. NeuralPolicy is a thin shell that delegates to the model;
swapping architectures requires changing one argument only.
"""

import os
from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all policy network models.

    Every subclass must implement:
        get_action(obs, deterministic) -> action tensor
        from_checkpoint(path, **kwargs) -> instance
        get_config() -> dict of architecture hyperparams

    The save() method is provided here and writes a unified wrapper:
        {"state_dict": ..., "config": ...}
    Each model's from_checkpoint() must be able to read this wrapper,
    but may also handle legacy raw state_dict checkpoints.
    """

    @abstractmethod
    def get_action(self, obs: "torch.Tensor", deterministic: bool = True) -> "torch.Tensor":
        """Return an action given an observation.

        Parameters
        ----------
        obs : torch.Tensor, shape (1, obs_dim)
            Observation tensor, already on the model's device.
        deterministic : bool
            True  -> return tanh(mean), no sampling
            False -> sample from the policy distribution

        Returns
        -------
        torch.Tensor, shape (1, action_dim), values in [-1, 1]
        """

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "BaseModel":
        """Load a model from a checkpoint file.

        Each model defines its own loading logic. May accept extra kwargs
        such as device, obs_dim, action_dim for overrides or legacy paths.
        """

    def save(self, path: str) -> None:
        """Save model to path as {"state_dict": ..., "config": ...}.

        Creates parent directories if they do not exist.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "config": self.get_config()},
            path,
        )

    @abstractmethod
    def get_config(self) -> dict:
        """Return the architecture hyperparams needed to reconstruct this model.

        The dict returned here is saved alongside the state_dict and passed
        back to from_checkpoint() during loading.
        """
