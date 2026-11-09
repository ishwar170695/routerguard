"""Inter-expert output similarity: how redundant are two experts in the same layer?

Metric, exactly as pre-registered (``protocol.yaml: similarity.metric``)
-----------------------------------------------------------------------
For one layer, every expert is run on **the same tokens** — the hidden states that
actually enter that MoE block — and the outputs are compared with a mean-centered
cosine similarity:

    z_{i,t} = f_i(x_t) - mean_j f_j(x_t)          # centre across experts, per token
    S_ij    = <z_i, z_j> / (||z_i|| ||z_j||)      # in [-1, 1], S_ii = 1

Centering removes the direction every expert shares (the residual stream component),
which is what makes the metric about *uniqueness* rather than about "experts are all
roughly aligned".

Cost and memory
---------------
Collecting this naively needs (n_experts x n_tokens x hidden) floats. Instead we
accumulate only the Gram matrix: one forward pass over the calibration set, with a
hook on each MoE block that runs all experts on the incoming tokens in chunks and
adds ``sum_t z z^T`` plus squared norms. Memory is O(n_experts^2) per layer, and each
layer obeys ``budget.max_tokens_per_layer``. Matrices are cached as ``.npy`` per
layer and existing files are skipped, so a killed session resumes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from . import env as rgenv
from .stats import layer_index_from_name

METRIC = "mean_centered_cosine"


@dataclass
class SimilarityReport:
    matrices: dict[int, np.ndarray]
    tokens_per_layer: dict[int, int]
    elapsed_s: float
    cached: list[int] = field(default_factory=list)
    computed: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": METRIC,
            "layers": sorted(int(k) for k in self.matrices),
            "tokens_per_layer": {int(k): int(v) for k, v in self.tokens_per_layer.items()},
            "num_experts_per_layer": {
                int(k): int(v.shape[0]) for k, v in self.matrices.items()
            },
            "symmetry_error": {
                int(k): float(np.abs(v - v.T).max()) for k, v in self.matrices.items()
            },
            "elapsed_s": round(self.elapsed_s, 2),
            "cached": [int(x) for x in self.cached],
            "computed": [int(x) for x in self.computed],
        }


def _accumulators(n_experts: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((n_experts, n_experts), dtype=torch.float64, device=device),
        torch.zeros(n_experts, dtype=torch.float64, device=device),
    )


@torch.no_grad()
def _accumulate(
    block,
    hidden_states: torch.Tensor,
    gram: torch.Tensor,
    norms: torch.Tensor,
    chunk_tokens: int,
    budget: int,
    used: int,
) -> int:
    """Add the contribution of one batch of tokens to the running Gram matrix."""
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    n_tokens = flat.shape[0]
    if used >= budget:
        return used
    flat = flat[: max(0, budget - used)]
    n_tokens = flat.shape[0]
    n_experts = len(block.experts)

    for start in range(0, n_tokens, chunk_tokens):
        chunk = flat[start : start + chunk_tokens]
        # Every expert on the SAME tokens -> (n_experts, chunk, hidden)
        outputs = torch.stack([block.experts[i](chunk) for i in range(n_experts)], dim=0)
        centered = outputs - outputs.mean(dim=0, keepdim=True)
        wide = centered.reshape(n_experts, -1).to(torch.float64)
        gram += wide @ wide.T
        norms += (wide * wide).sum(dim=1)
        del outputs, centered, wide
    return used + n_tokens


def gram_to_cosine(gram: torch.Tensor, norms: torch.Tensor) -> np.ndarray:
    """Cosine matrix from accumulated Gram + squared norms; diagonal forced to exactly 1."""
    n = gram.shape[0]
    denom = torch.sqrt(norms.clamp_min(1e-30))
    cosine = gram / denom[:, None] / denom[None, :]
    cosine = torch.nan_to_num(cosine, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    cosine.fill_diagonal_(1.0)
    if cosine.shape != (n, n):
        raise RuntimeError(f"cosine matrix is {tuple(cosine.shape)}, expected {(n, n)}")
    return cosine.cpu().numpy().astype(np.float32)


class SimilarityCollector:
    """One pass over ``input_ids``; accumulates a cosine matrix per MoE layer.

    Usage::

        with SimilarityCollector(model, max_tokens_per_layer=100_000) as sim:
            for batch in batches:
                model(input_ids=batch, use_cache=False)
        matrices = sim.matrices()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_tokens_per_layer: int = 100_000,
        chunk_tokens: int = 256,
        device: str | None = None,
    ):
        self.model = model
        self.max_tokens_per_layer = int(max_tokens_per_layer)
        self.chunk_tokens = int(chunk_tokens)
        self.device = device
        blocks = rgenv.find_moe_blocks(model)
        if not blocks:
            raise RuntimeError("no MoE blocks found")
        self.blocks: dict[int, Any] = {}
        for name, block in blocks:
            layer = layer_index_from_name(name)
            if layer in self.blocks:
                raise RuntimeError(f"duplicate MoE layer index {layer}")
            self.blocks[layer] = block
        self.layers = sorted(self.blocks)
        self._gram: dict[int, torch.Tensor] = {}
        self._norms: dict[int, torch.Tensor] = {}
        self._used: dict[int, int] = {layer: 0 for layer in self.layers}
        self._handles: list[Any] = []
        for layer in self.layers:
            n = len(self.blocks[layer].experts)
            gram, norms = _accumulators(n, "cpu")
            self._gram[layer] = gram
            self._norms[layer] = norms

    def _hook(self, layer: int):
        block = self.blocks[layer]

        def hook(module, inputs):
            if not inputs:
                return
            states = inputs[0]
            gram = self._gram[layer].to(states.device)
            norms = self._norms[layer].to(states.device)
            self._used[layer] = _accumulate(
                block, states, gram, norms, self.chunk_tokens,
                self.max_tokens_per_layer, self._used[layer],
            )
            self._gram[layer] = gram
            self._norms[layer] = norms

        return hook

    def __enter__(self) -> "SimilarityCollector":
        for layer in self.layers:
            self._handles.append(self.blocks[layer].register_forward_pre_hook(self._hook(layer)))
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @torch.no_grad()
    def collect(self, input_ids: torch.Tensor, batch_size: int = 1, progress: bool = False) -> None:
        ids = input_ids.to(self.device) if self.device is not None else input_ids
        n_batches = (ids.shape[0] + batch_size - 1) // batch_size
        for index, start in enumerate(range(0, ids.shape[0], batch_size)):
            self.model(input_ids=ids[start : start + batch_size], use_cache=False)
            if progress and index % 5 == 0:
                print(f"  [similarity] batch {index + 1}/{n_batches}", flush=True)

    def matrices(self) -> dict[int, np.ndarray]:
        return {layer: gram_to_cosine(self._gram[layer], self._norms[layer]) for layer in self.layers}

    def tokens_per_layer(self) -> dict[int, int]:
        return dict(self._used)


# ------------------------------------------------------------------------- caching


def _layer_path(out_dir: Path, layer: int) -> Path:
    return out_dir / f"layer_{layer:02d}.npy"


def build_similarity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    out_dir: str | Path,
    layers: Sequence[int] | None = None,
    max_tokens_per_layer: int = 100_000,
    chunk_tokens: int = 256,
    batch_size: int = 1,
    device: str | None = None,
    resume: bool = True,
    progress: bool = True,
) -> SimilarityReport:
    """Compute and cache one cosine matrix per layer. Existing files are skipped."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_layers = sorted(layer_index_from_name(n) for n, _ in rgenv.find_moe_blocks(model))
    wanted = list(all_layers if layers is None else layers)

    cached = [layer for layer in wanted if resume and _layer_path(out, layer).exists()]
    missing = [layer for layer in wanted if layer not in cached]
    started = time.time()

    matrices: dict[int, np.ndarray] = {}
    tokens: dict[int, int] = {}
    for layer in cached:
        matrices[layer] = load_similarity(out, layer)

    if missing:
        if progress:
            print(f"  [similarity] computing layers {missing} (cached: {cached})", flush=True)
        collector = SimilarityCollector(
            model, max_tokens_per_layer=max_tokens_per_layer,
            chunk_tokens=chunk_tokens, device=device,
        )
        with collector:
            collector.collect(input_ids, batch_size=batch_size, progress=progress)
        computed = collector.matrices()
        used = collector.tokens_per_layer()
        for layer in missing:
            matrix = computed[layer]
            np.save(_layer_path(out, layer), matrix)
            matrices[layer] = matrix
            tokens[layer] = used[layer]
    else:
        if progress:
            print("  [similarity] all requested layers cached; no forward pass needed", flush=True)

    for layer in cached:
        tokens.setdefault(layer, 0)
    return SimilarityReport(
        matrices={layer: matrices[layer] for layer in sorted(matrices)},
        tokens_per_layer=tokens,
        elapsed_s=time.time() - started,
        cached=cached,
        computed=missing,
    )


def load_similarity(out_dir: str | Path, layer: int) -> np.ndarray:
    path = _layer_path(Path(out_dir), layer)
    if not path.exists():
        raise FileNotFoundError(f"no cached similarity for layer {layer} at {path}")
    return np.load(path)


def load_all_similarity(out_dir: str | Path, layers: Iterable[int] | None = None) -> dict[int, np.ndarray]:
    out = Path(out_dir)
    found = sorted(int(p.stem.split("_")[-1]) for p in out.glob("layer_*.npy"))
    wanted = [layer for layer in found if layers is None or layer in set(layers)]
    return {layer: load_similarity(out, layer) for layer in wanted}


# ------------------------------------------------------------------------ scores


def pairwise_mean_similarity(sim: np.ndarray, keep: Sequence[int] | None = None) -> torch.Tensor:
    """For each expert, mean cosine to the OTHER kept experts (higher = more redundant)."""
    matrix = torch.as_tensor(np.asarray(sim, dtype=np.float32))
    if keep is not None:
        idx = torch.as_tensor(list(keep), dtype=torch.long)
        matrix = matrix.index_select(0, idx).index_select(1, idx)
    n = matrix.shape[0]
    if n <= 1:
        return torch.zeros(n)
    total = matrix.sum(dim=1) - matrix.diagonal()
    return total / (n - 1)


def uniqueness_scores(sim: np.ndarray, keep: Sequence[int] | None = None) -> torch.Tensor:
    """Higher = more unique. This is the *primary* importance term."""
    return -pairwise_mean_similarity(sim, keep)


def redundancy_pairs(sim: np.ndarray, top: int = 5) -> list[tuple[int, int, float]]:
    """Most similar expert pairs, for reporting/diagnostics (excludes the diagonal)."""
    matrix = np.asarray(sim, dtype=np.float32).copy()
    np.fill_diagonal(matrix, -np.inf)
    flat = [
        (i, j, float(matrix[i, j]))
        for i in range(matrix.shape[0])
        for j in range(i + 1, matrix.shape[1])
    ]
    flat.sort(key=lambda item: item[2], reverse=True)
    return flat[:top]
