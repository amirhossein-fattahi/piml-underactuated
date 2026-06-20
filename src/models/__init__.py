# src/models/__init__.py
from .mlp import VanillaMLP
from .lnn import LagrangianNN

MODEL_REGISTRY = {
    "Vanilla_MLP": VanillaMLP,
    "Lagrangian_NN": LagrangianNN,
}


def build_model(cfg):
    """Builds a model from a Hydra config (expects cfg.model, cfg.env, cfg.dt)."""
    name = cfg.model.name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    model_cfg = {
        "state_dim": cfg.env.state_dim,
        "action_dim": cfg.env.action_dim,
        "dt": cfg.dt,
        # Known actuation matrix B (physics prior used by structured models).
        "actuation": list(cfg.env.get("actuation", [0.0, 1.0])),
    }
    return MODEL_REGISTRY[name](model_cfg)
