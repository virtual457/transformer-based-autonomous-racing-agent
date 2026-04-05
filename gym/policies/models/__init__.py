"""
Model registry for NeuralPolicy.

Adding a new architecture:
  1. Create gym/policies/models/my_model.py with a class that inherits BaseModel.
  2. Register it here: MODEL_REGISTRY["my_key"] = MyModel
  3. Pass model_type="my_key" to NeuralPolicy or load_model().

Nothing else changes — NeuralPolicy and train_bc.py are unaffected.
"""

from policies.models.base_model import BaseModel
from policies.models.mlp_actor import MLPActor

MODEL_REGISTRY: dict = {
    "mlp": MLPActor,
}


def load_model(model_type: str, checkpoint_path: str, **kwargs) -> BaseModel:
    """Load a model by type from a checkpoint file.

    Parameters
    ----------
    model_type : str
        Key in MODEL_REGISTRY (e.g. "mlp").
    checkpoint_path : str
        Path to the .pt checkpoint file.
    **kwargs
        Forwarded to the model's from_checkpoint() method.
        Common kwargs: device, obs_dim, action_dim, hidden_units.

    Returns
    -------
    BaseModel instance, weights loaded, eval mode, on the requested device.

    Raises
    ------
    ValueError if model_type is not in MODEL_REGISTRY.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type].from_checkpoint(checkpoint_path, **kwargs)


__all__ = [
    "BaseModel",
    "MLPActor",
    "MODEL_REGISTRY",
    "load_model",
]
