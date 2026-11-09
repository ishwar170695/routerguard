"""Patched MoE forward: selection-only bias, mask simulation, real expert removal.

The one rule this module exists to enforce
------------------------------------------
Loss-Free Balancing selects on ``s + b`` but mixes with weights from the *unbiased*
``s``. A plain logit hook cannot express that (it would bias the mixing weights too),
so we replace ``forward`` on each MoE block with a faithful mirror of
transformers 4.48.3's own loop and change only the selection line.

Guarantees, stated precisely
----------------------------
* ``bias=None, keep=None``  -> **bitwise identical** to the unpatched model
  (``torch.equal``). This is the strong guarantee: the code path is upstream's.
* mask simulation vs real removal -> identical expert *selection* (exact, integer),
  outputs equal to float rounding. Bitwise equality is not claimed here because
  pruning changes the router GEMM shape (out_features shrinks), and a GEMM's
  reduction order is not invariant to N. Mask simulation therefore selects the kept
  router columns with ``index_select`` (not ``-inf`` masking) so softmax/top-k see
  exactly the same tensor as the pruned model; what remains is only the GEMM shape.

Removal semantics (see protocol.yaml: ``pruning.removal_semantics: reslice``)
----------------------------------------------------------------------------
The pruned model is the native one: gate rows are sliced, so softmax runs over the
kept experts only. Mask simulation must match that, which is exactly what column
selection does. Consequences: the removed experts' probability mass is redistributed
by renormalization, and ``top_k`` is clamped to the number of kept experts.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from . import env as rgenv
from .stats import aggregate, counts_from_selection, layer_index_from_name


# --------------------------------------------------------------------------- core


def moe_forward(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    keep_idx: torch.Tensor | None = None,
    recorder: "SelectionRecorder | None" = None,
    layer: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of Olmoe/Qwen2Moe ``SparseMoeBlock.forward`` with a pluggable router.

    bias:      float32 ``(n_experts,)`` added to router logits for SELECTION only.
    keep_idx:  long ``(n_kept,)`` original expert indices still present. Selection and
               softmax then run over these columns, matching a gate-sliced model.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = block.gate(hidden_states)

    if keep_idx is None:
        sel_logits = router_logits
        n_experts = int(block.num_experts)
    else:
        # Same tensor the pruned model's sliced gate would produce, so softmax/top-k
        # see identical values in identical order.
        sel_logits = router_logits.index_select(1, keep_idx)
        n_experts = int(keep_idx.numel())

    top_k = min(int(block.top_k), n_experts)

    if bias is None:
        routing_weights = F.softmax(sel_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    else:
        bias_k = bias if keep_idx is None else bias.index_select(0, keep_idx)
        # SELECTION on biased scores, MIXING from unbiased probabilities.
        selection_scores = sel_logits.to(torch.float32) + bias_k.to(torch.float32)
        _, selected_experts = torch.topk(selection_scores, top_k, dim=-1)
        routing_weights = F.softmax(sel_logits, dim=1, dtype=torch.float).gather(1, selected_experts)

    if block.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    if recorder is not None:
        original = selected_experts if keep_idx is None else keep_idx[selected_experts]
        recorder.record(
            layer,
            original.detach(),
            selection_scores.detach() if bias is not None else None,
            n_experts=int(block.num_experts) if keep_idx is None else int(block.num_experts),
        )

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=n_experts).permute(2, 1, 0)

    for pos in range(n_experts):
        expert_layer = block.experts[pos] if keep_idx is None else block.experts[int(keep_idx[pos])]
        idx, top_x = torch.where(expert_mask[pos])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    # Qwen1.5-MoE: one always-on shared expert, added after the routed sum (as upstream).
    shared_expert = getattr(block, "shared_expert", None)
    if shared_expert is not None:
        shared_expert_output = shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(block.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


# ------------------------------------------------------------------------ recorder


class SelectionRecorder:
    """Records the *actual* (post-bias, post-mask) selection per layer."""

    def __init__(self, keep_scores: bool = False):
        self.keep_scores = keep_scores
        self._counts: dict[int, torch.Tensor] = {}
        self._num_experts: dict[int, int] = {}
        self._scores: dict[int, list[torch.Tensor]] = {}
        self._tokens: dict[int, int] = {}

    def record(self, layer, selected, scores, n_experts: int) -> None:
        layer = int(layer if layer is not None else -1)
        counts = counts_from_selection(selected, n_experts).cpu()
        if layer in self._counts:
            self._counts[layer] = self._counts[layer] + counts
        else:
            self._counts[layer] = counts
        self._num_experts[layer] = n_experts
        self._tokens[layer] = self._tokens.get(layer, 0) + int(selected.shape[0])
        if self.keep_scores and scores is not None:
            self._scores.setdefault(layer, []).append(scores.cpu())

    def reset(self) -> None:
        self._counts.clear()
        self._scores.clear()
        self._tokens.clear()

    @property
    def tokens(self) -> int:
        """Tokens seen *per layer* (not summed over layers)."""
        return max(self._tokens.values()) if self._tokens else 0

    def tokens_per_layer(self) -> dict[int, int]:
        return dict(self._tokens)

    def per_layer_counts(self) -> dict[int, torch.Tensor]:
        return {layer: c.clone() for layer, c in sorted(self._counts.items())}

    def num_experts(self) -> dict[int, int]:
        return dict(self._num_experts)

    def report(self) -> dict[str, Any]:
        out = aggregate(self.per_layer_counts())
        out["tokens_seen"] = self.tokens
        out["tokens_per_layer"] = self.tokens_per_layer()
        out["per_layer_counts"] = {int(k): v.tolist() for k, v in self.per_layer_counts().items()}
        return out


# --------------------------------------------------------------------------- patch


def _resolve_per_layer(value: Any, layers: Sequence[int]) -> dict[int, Any]:
    """Accept None, a single tensor/dict, or a full {layer: value} mapping."""
    if value is None:
        return {layer: None for layer in layers}
    if isinstance(value, dict):
        return {layer: value.get(layer) for layer in layers}
    return {layer: value for layer in layers}


class MoePatch:
    """Context manager replacing every MoE block's forward with :func:`moe_forward`.

    Usage::

        with MoePatch(model, bias=bias_per_layer, record=True) as patch:
            model(input_ids=ids)
            counts = patch.recorder.per_layer_counts()
    """

    def __init__(
        self,
        model: nn.Module,
        bias: torch.Tensor | dict[int, torch.Tensor] | None = None,
        keep_idx: torch.Tensor | dict[int, torch.Tensor] | None = None,
        record: bool = False,
        keep_scores: bool = False,
    ):
        self.model = model
        self.blocks = rgenv.find_moe_blocks(model)
        if not self.blocks:
            raise RuntimeError("no MoE blocks found; check env.assert_moe_structure(model)")
        self.layers: dict[int, nn.Module] = {}
        for name, block in self.blocks:
            layer = layer_index_from_name(name)
            if layer in self.layers:
                raise RuntimeError(f"duplicate layer index {layer} from {name}")
            self.layers[layer] = block
        self._order = sorted(self.layers)
        self.bias = _resolve_per_layer(bias, self._order)
        self.keep_idx = _resolve_per_layer(keep_idx, self._order)
        self.recorder = SelectionRecorder(keep_scores=keep_scores) if record else None
        self._originals: dict[int, Any] = {}
        self._active = False

    # -- per-layer setters (used by the balance loop in Step 3) -----------------
    def set_bias(self, bias: torch.Tensor | dict[int, torch.Tensor] | None) -> None:
        self.bias = _resolve_per_layer(bias, self._order)

    def set_keep(self, keep_idx: torch.Tensor | dict[int, torch.Tensor] | None) -> None:
        self.keep_idx = _resolve_per_layer(keep_idx, self._order)

    def bias_for(self, layer: int) -> torch.Tensor | None:
        value = self.bias.get(layer)
        if value is None:
            return None
        block = self.layers[layer]
        n = int(block.num_experts)
        if value.numel() != n:
            raise ValueError(f"layer {layer}: bias has {value.numel()} entries, model has {n} experts")
        return value

    def keep_for(self, layer: int) -> torch.Tensor | None:
        return self.keep_idx.get(layer)

    # -- context management ----------------------------------------------------
    def __enter__(self) -> "MoePatch":
        if self._active:
            raise RuntimeError("MoePatch is already active")
        for layer, block in self.layers.items():
            self._originals[layer] = block.forward
            block.forward = types.MethodType(_bound_forward(self, layer), block)
        self._active = True
        return self

    def __exit__(self, *exc_info) -> None:
        for layer, block in self.layers.items():
            original = self._originals.pop(layer, None)
            if original is not None:
                block.forward = original
        self._active = False


def _bound_forward(patch: MoePatch, layer: int):
    def forward(self, hidden_states):
        return moe_forward(
            self,
            hidden_states,
            bias=patch.bias_for(layer),
            keep_idx=patch.keep_for(layer),
            recorder=patch.recorder,
            layer=layer,
        )

    return forward


class MaskSim(MoePatch):
    """Simulate pruning without touching weights: same forward, restricted experts."""

    def __init__(self, model: nn.Module, keep_per_layer: dict[int, Sequence[int]] | Sequence[int],
                 record: bool = False):
        blocks = rgenv.find_moe_blocks(model)
        layers = [layer_index_from_name(name) for name, _ in blocks]
        if isinstance(keep_per_layer, dict):
            per_layer = keep_per_layer
        else:
            per_layer = {layer: keep_per_layer for layer in layers}
        keep_idx = {}
        for layer, block in zip(layers, [b for _, b in blocks]):
            indices = per_layer[layer]
            keep_idx[layer] = torch.as_tensor(list(indices), dtype=torch.long, device=block.gate.weight.device)
        super().__init__(model, keep_idx=keep_idx, record=record)


# ------------------------------------------------------------------------ removal


def keep_mask_from_indices(keep: Sequence[int], num_experts: int, device=None) -> torch.Tensor:
    mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
    mask[torch.as_tensor(list(keep), dtype=torch.long, device=device)] = True
    return mask


def keep_indices_from_ratio(num_experts: int, ratio: float, generator: torch.Generator | None = None,
                            scores: torch.Tensor | None = None) -> list[int]:
    """Which experts survive. ``scores`` = importance (higher kept); None = seeded random."""
    n_keep = max(1, int(round(num_experts * (1.0 - ratio))))
    if scores is not None:
        order = torch.argsort(torch.as_tensor(scores).float(), descending=True)
        return sorted(int(i) for i in order[:n_keep])
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(num_experts, generator=generator)
    return sorted(int(i) for i in perm[:n_keep])


def prune_block_inplace(block: nn.Module, keep: Sequence[int]) -> dict[str, Any]:
    """Physically drop experts and slice the gate. Kept experts stay in index order."""
    keep = [int(i) for i in sorted(keep)]
    old_n = int(block.num_experts)
    if len(keep) != len(set(keep)) or any(i < 0 or i >= old_n for i in keep):
        raise ValueError(f"invalid keep indices {keep} for {old_n} experts")
    if len(keep) == 0:
        raise ValueError("cannot prune every expert")

    old_gate = block.gate
    kept_experts = [block.experts[i] for i in keep]
    new_gate = nn.Linear(old_gate.in_features, len(keep), bias=old_gate.bias is not None)
    with torch.no_grad():
        new_gate.weight.copy_(old_gate.weight.detach()[keep])
        if old_gate.bias is not None:
            new_gate.bias.copy_(old_gate.bias.detach()[keep])
    new_gate = new_gate.to(device=old_gate.weight.device, dtype=old_gate.weight.dtype)
    new_gate.weight.requires_grad_(old_gate.weight.requires_grad)

    block.experts = nn.ModuleList(kept_experts)
    block.gate = new_gate
    block.num_experts = len(keep)
    block.top_k = min(int(block.top_k), len(keep))

    return {
        "num_experts_before": old_n,
        "num_experts_after": len(keep),
        "removed": sorted(set(range(old_n)) - set(keep)),
        "top_k": int(block.top_k),
    }


def apply_removal(model: nn.Module, keep_per_layer: dict[int, Sequence[int]] | Sequence[int]) -> dict[str, Any]:
    """Prune every MoE layer in place (native pruned model) and update the config."""
    blocks = rgenv.find_moe_blocks(model)
    layers = [layer_index_from_name(name) for name, _ in blocks]
    per_layer = keep_per_layer if isinstance(keep_per_layer, dict) else {layer: keep_per_layer for layer in layers}
    report: dict[str, Any] = {"layers": {}, "keep": {}}
    for layer, (_, block) in zip(layers, blocks):
        info = prune_block_inplace(block, per_layer[layer])
        report["layers"][layer] = info
        report["keep"][layer] = [int(i) for i in sorted(per_layer[layer])]

    counts = {info["num_experts_after"] for info in report["layers"].values()}
    if len(counts) == 1:
        n = counts.pop()
        for attr in ("num_experts",):
            if hasattr(model.config, attr):
                setattr(model.config, attr, n)
    report["num_experts"] = sorted({info["num_experts_after"] for info in report["layers"].values()})
    report["removed_total"] = sum(len(info["removed"]) for info in report["layers"].values())
    return report


# ------------------------------------------------------------------------- helpers


@torch.no_grad()
def forward_logits(model: nn.Module, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False, **kwargs)
    return out.logits


def equivalence_check(
    a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-5
) -> dict[str, Any]:
    """Report both the strict (bitwise) and tolerant verdicts; tests assert on both."""
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    bitwise = bool(torch.equal(a, b))
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-12)
    return {
        "bitwise_equal": bitwise,
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float((diff / denom).max()),
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "n_elements": int(a.numel()),
    }
