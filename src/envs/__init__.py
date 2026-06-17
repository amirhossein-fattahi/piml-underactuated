# src/envs/__init__.py
from .acrobot import AcrobotEnv
from .pendubot import PendubotEnv

ENV_REGISTRY = {
    "Acrobot": AcrobotEnv,
    "Pendubot": PendubotEnv,
}


def build_env(cfg):
    """Builds an environment from a Hydra config (expects cfg.env and cfg.dt)."""
    name = cfg.env.name
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env '{name}'. Available: {list(ENV_REGISTRY)}")
    # Merge top-level dt into the env config; physical params fall back to
    # the sensible defaults defined in DoublePendulumBase.
    env_cfg = {
        "dt": cfg.dt,
        "gravity": cfg.env.get("gravity", 9.81),
    }
    return ENV_REGISTRY[name](env_cfg)
