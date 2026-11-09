"""Expert rankings and pruning. Step 3 ships the *baselines* only (random, frequency);
the redundancy-aware greedy search arrives in Step 4 and plugs into the same API.

Ranking convention: ``scores[layer][e]`` — **higher means more important** (kept first).
``ratio`` is always the fraction of *routed* experts removed in that layer.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from . import env as rgenv
from . import moe
from .stats import layer_index_from_name


def num_keep(num_experts: int, ratio: float) -> int:
    """How many experts survive a removal ratio. Never returns 0."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    return max(1, int(round(num_experts * (1.0 - ratio))))


def top_k_indices(scores: torch.Tensor, k: int) -> list[int]:
    """Indices of the ``k`` highest scores, returned in ascending expert order."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, scores.numel())
    order = torch.argsort(torch.as_tensor(scores).float(), descending=True, stable=True)
    return sorted(int(i) for i in order[:k])


# ------------------------------------------------------------------ score sources


def frequency_scores(per_layer_counts: dict[int, Sequence[int]]) -> dict[int, torch.Tensor]:
    """Utilization baseline: an expert's calibration load is its importance."""
    return {int(layer): torch.as_tensor(counts, dtype=torch.float32) for layer, counts in per_layer_counts.items()}


def random_scores(experts_per_layer: dict[int, int], seed: int = 0) -> dict[int, torch.Tensor]:
    """Random baseline: a fixed permutation per layer from an explicit seed."""
    generator = torch.Generator().manual_seed(int(seed))
    scores: dict[int, torch.Tensor] = {}
    for layer in sorted(experts_per_layer):
        n = int(experts_per_layer[layer])
        scores[layer] = torch.randperm(n, generator=generator).float()
    return scores


def experts_per_layer(model: torch.nn.Module) -> dict[int, int]:
    return {
        layer_index_from_name(name): int(block.num_experts)
        for name, block in rgenv.find_moe_blocks(model)
    }


# ------------------------------------------------------------------- keep sets


def keeps_from_scores(scores: dict[int, torch.Tensor], ratio: float) -> dict[int, list[int]]:
    keeps: dict[int, list[int]] = {}
    for layer, layer_scores in scores.items():
        n = int(layer_scores.numel())
        keeps[int(layer)] = top_k_indices(layer_scores, num_keep(n, ratio))
    return keeps


def frequency_keeps(per_layer_counts: dict[int, Sequence[int]], ratio: float) -> dict[int, list[int]]:
    return keeps_from_scores(frequency_scores(per_layer_counts), ratio)


def random_keeps(experts: dict[int, int], ratio: float, seed: int = 0) -> dict[int, list[int]]:
    return keeps_from_scores(random_scores(experts, seed), ratio)


def keeps_for(
    model: torch.nn.Module,
    kind: str,
    ratio: float,
    per_layer_counts: dict[int, Sequence[int]] | None = None,
    seed: int = 0,
    scores: dict[int, torch.Tensor] | None = None,
) -> dict[int, list[int]]:
    """Dispatch for the baselines the protocol registers."""
    if scores is not None:
        return keeps_from_scores(scores, ratio)
    if kind == "frequency":
        if per_layer_counts is None:
            raise ValueError("frequency pruning needs per_layer_counts from calibration")
        return frequency_keeps(per_layer_counts, ratio)
    if kind == "random":
        return random_keeps(experts_per_layer(model), ratio, seed=seed)
    if kind == "redundancy":
        raise NotImplementedError("redundancy ranking lands in Step 4 (similarity.py + greedy prune)")
    raise ValueError(f"unknown pruning kind {kind!r}")


# --------------------------------------------------------------------- apply


def prune_with_keeps(model: torch.nn.Module, keeps: dict[int, list[int]]) -> dict[str, Any]:
    """Real removal: drop experts and slice routers. Returns the removal report."""
    report = moe.apply_removal(model, keeps)
    report["kept_per_layer"] = {int(k): len(v) for k, v in keeps.items()}
    return report


def prune_by_kind(
    model: torch.nn.Module,
    kind: str,
    ratio: float,
    per_layer_counts: dict[int, Sequence[int]] | None = None,
    seed: int = 0,
    scores: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    keeps = keeps_for(model, kind, ratio, per_layer_counts=per_layer_counts, seed=seed, scores=scores)
    return keeps, prune_with_keeps(model, keeps)


def removal_counts(experts: dict[int, int], ratio: float) -> dict[int, tuple[int, int]]:
    """``{layer: (n_before, n_after)}`` for a ratio, without touching a model."""
    return {layer: (n, num_keep(n, ratio)) for layer, n in sorted(experts.items())}
