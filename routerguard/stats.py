"""Expert load statistics: counts, Gini, max-violation, and router-distribution KL.

Everything here is *observation only*: it never changes the forward pass. The
collector hooks ``block.gate`` (a plain ``nn.Linear``) and reads its output, so it
is valid for unpatched models and for the Step-2 patched forward alike. Note the
distinction the method depends on: reading gate logits is fine, but implementing
the bias itself through a logit hook would silently change the mixing weights.
"""

from __future__ import annotations

from typing import Any

import torch

from . import env as rgenv


def counts_from_selection(selected: torch.Tensor, num_experts: int) -> torch.Tensor:
    """(tokens, top_k) selected indices -> (num_experts,) load counts."""
    return torch.bincount(selected.reshape(-1).to(torch.long), minlength=num_experts)[:num_experts]


def gini(counts: torch.Tensor | list[float]) -> float:
    """Gini coefficient of expert load. 0 = perfectly uniform, ->1 = all load on one."""
    x = torch.as_tensor(counts, dtype=torch.float64).flatten()
    n = x.numel()
    total = x.sum()
    if n == 0 or total <= 0:
        return 0.0
    x, _ = torch.sort(x)
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    return float((2.0 * (idx * x).sum() / (n * total)) - (n + 1.0) / n)


def max_violation(counts: torch.Tensor | list[float]) -> float:
    """max_i |c_i / mean(c) - 1|: how far the worst expert is from a uniform target."""
    x = torch.as_tensor(counts, dtype=torch.float64).flatten()
    mean = x.mean()
    if x.numel() == 0 or mean <= 0:
        return 0.0
    return float((x / mean - 1.0).abs().max())


def summary(counts: torch.Tensor | list[float]) -> dict[str, Any]:
    """Load-statistics block for one layer (or pooled over layers)."""
    x = torch.as_tensor(counts, dtype=torch.float64).flatten()
    n = x.numel()
    total = float(x.sum())
    mean = float(x.mean()) if n else 0.0
    std = float(x.std(unbiased=False)) if n else 0.0
    if total > 0:
        p = x / total
        entropy = float(-(p[p > 0] * p[p > 0].log()).sum())
    else:
        entropy = 0.0
    return {
        "n_experts": n,
        "total_assignments": total,
        "mean": mean,
        "min": float(x.min()) if n else 0.0,
        "max": float(x.max()) if n else 0.0,
        "cv": (std / mean) if mean > 0 else 0.0,
        "gini": gini(x),
        "max_violation": max_violation(x),
        "dead_experts": int((x == 0).sum()),
        "top1_share": float(x.max() / total) if total > 0 else 0.0,
        "entropy_nats": entropy,
        "entropy_normalized": (entropy / torch.log(torch.tensor(float(n))).item()) if n > 1 and total > 0 else 0.0,
    }


def kl(p: torch.Tensor | list[float], q: torch.Tensor | list[float], eps: float = 1e-12) -> float:
    """KL(p || q) in nats between two load distributions (normalized internally)."""
    p_t = torch.as_tensor(p, dtype=torch.float64).flatten()
    q_t = torch.as_tensor(q, dtype=torch.float64).flatten()
    if p_t.shape != q_t.shape:
        raise ValueError(f"shape mismatch: {tuple(p_t.shape)} vs {tuple(q_t.shape)}")
    p_sum, q_sum = p_t.sum(), q_t.sum()
    if p_sum <= 0 or q_sum <= 0:
        return 0.0
    p_n = p_t / p_sum
    q_n = (q_t / q_sum).clamp_min(eps)
    mask = p_n > 0
    return float((p_n[mask] * (p_n[mask].log() - q_n[mask].log())).sum())


def transfer_ratio(calib_gain: float, eval_gain: float) -> float:
    """Fraction of a calibration-time improvement that survives on held-out data."""
    if abs(calib_gain) < 1e-12:
        return 0.0
    return float(eval_gain / calib_gain)


def aggregate(per_layer: dict[int, torch.Tensor]) -> dict[str, Any]:
    """Summarize across layers: mean/worst-layer Gini plus pooled counts."""
    if not per_layer:
        return {}
    layers = sorted(per_layer)
    ginis = {layer: gini(per_layer[layer]) for layer in layers}
    violations = {layer: max_violation(per_layer[layer]) for layer in layers}
    pooled = torch.stack([torch.as_tensor(per_layer[layer], dtype=torch.float64) for layer in layers]).sum(0)
    worst = max(ginis, key=ginis.get)
    return {
        "n_layers": len(layers),
        "gini_mean": float(sum(ginis.values()) / len(ginis)),
        "gini_max": float(max(ginis.values())),
        "gini_min": float(min(ginis.values())),
        "gini_worst_layer": int(worst),
        "max_violation_mean": float(sum(violations.values()) / len(violations)),
        "max_violation_max": float(max(violations.values())),
        "pooled": summary(pooled),
        "per_layer_gini": {int(k): float(v) for k, v in ginis.items()},
        "per_layer_max_violation": {int(k): float(v) for k, v in violations.items()},
    }


def layer_index_from_name(name: str) -> int:
    """``model.layers.7.mlp`` -> 7 (falls back to sorted order upstream)."""
    for part in reversed(name.split(".")):
        if part.isdigit():
            return int(part)
    return -1


class UnbiasedLoadStats:
    """Context manager collecting per-layer *unbiased* top-k load counts.

    On the unpatched model this equals the model's real routing. After Step 2 the
    patched forward still calls ``gate``, so this remains a valid *counterfactual*
    (no-bias) reference, which is exactly what the balance step needs to compare
    against. Use ``moe.SelectionRecorder`` for the model's actual selection.

    Usage::

        with UnbiasedLoadStats(model) as ls:
            for batch in split.batches(8):
                model(batch)
        per_layer = ls.per_layer_counts()   # {layer_idx: LongTensor(num_experts)}
    """

    def __init__(self, model: Any, top_k: int | None = None):
        self.model = model
        self.top_k = int(top_k or getattr(model.config, "num_experts_per_tok", 0))
        if self.top_k <= 0:
            raise ValueError("could not determine top_k")
        self._blocks = rgenv.find_moe_blocks(model)
        if not self._blocks:
            raise RuntimeError("no MoE blocks found; call env.assert_moe_structure first")
        self._counts: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self.n_tokens = 0

    def _hook(self, layer: int, num_experts: int):
        def hook(module, inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(logits):
                return
            with torch.no_grad():
                flat = logits.detach().reshape(-1, logits.shape[-1]).float()
                selected = torch.topk(flat, min(self.top_k, flat.shape[-1]), dim=-1).indices
                counts = counts_from_selection(selected, num_experts).cpu()
                if layer in self._counts:
                    self._counts[layer] += counts
                else:
                    self._counts[layer] = counts
                self.n_tokens += int(flat.shape[0])
        return hook

    def __enter__(self) -> "UnbiasedLoadStats":
        for name, block in self._blocks:
            layer = layer_index_from_name(name)
            self._handles.append(
                block.gate.register_forward_hook(self._hook(layer, len(block.experts)))
            )
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        self._counts.clear()
        self.n_tokens = 0

    def per_layer_counts(self) -> dict[int, torch.Tensor]:
        return {layer: counts.clone() for layer, counts in sorted(self._counts.items())}

    def pooled_counts(self) -> torch.Tensor:
        if not self._counts:
            return torch.zeros(0, dtype=torch.long)
        return torch.stack([self._counts[k] for k in sorted(self._counts)]).sum(0)

    def report(self, include_counts: bool = True) -> dict[str, Any]:
        per_layer = self.per_layer_counts()
        out = aggregate(per_layer)
        out["tokens_seen"] = self.n_tokens
        out["top_k"] = self.top_k
        if include_counts:
            # 64 ints per layer: cheap, and lets KL/transfer be recomputed later.
            out["per_layer_counts"] = {int(k): v.tolist() for k, v in per_layer.items()}
        return out


@torch.no_grad()
def collect_load_stats(
    model: Any,
    split,
    batch_size: int = 8,
    top_k: int | None = None,
    device: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Run ``split`` through ``model`` once and return the load-statistics report."""
    input_ids = split.input_ids.to(device) if device is not None else split.input_ids
    tokens = 0
    with UnbiasedLoadStats(model, top_k=top_k) as stats:
        for start in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[start : start + batch_size]
            if max_tokens is not None and tokens >= max_tokens:
                break
            model(input_ids=batch, use_cache=False)
            tokens += batch.numel()
    report = stats.report()
    report["max_tokens_cap"] = max_tokens
    return report
