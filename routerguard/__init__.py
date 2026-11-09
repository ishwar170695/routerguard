"""RouterGuard: post-hoc load balancing + redundancy-aware expert pruning for sparse MoE LLMs.

Import submodules explicitly (``from routerguard import env, models``); this package
__init__ deliberately imports no heavy dependencies so tests stay fast.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "env",
    "models",
    "data",
    "stats",
    "moe",
    "balance",
    "similarity",
    "prune",
    "eval",
    "run",
]
