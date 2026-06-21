# src/models/__init__.py
from .mlp import VanillaMLP
from .lnn import LagrangianNN
from .hnn import HamiltonianNN
from .gp import VanillaGP
from .lgp import LagrangianGP

MODEL_REGISTRY = {
    "Vanilla_MLP": VanillaMLP,
    "Lagrangian_NN": LagrangianNN,
    "Hamiltonian_NN": HamiltonianNN,
    "Vanilla_GP": VanillaGP,
    "Lagrangian_GP": LagrangianGP,
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
        # Optimizer settings (used by models that train an internal NN, e.g. LGP).
        "learning_rate": cfg.get("learning_rate", 1e-3),
        "batch_size": cfg.get("batch_size", 256),
        "lnn_epochs": cfg.model.get("lnn_epochs", 100),
    }
    return MODEL_REGISTRY[name](model_cfg)
