"""Calibration-time Loss-Free Balancing (LFB, arXiv:2408.15664) plus its two guards.

Method, stated exactly
----------------------
Per layer and per routed expert we keep a bias ``b_i`` (float32). Selection is
``topk(s + b)``; the mixing weights stay the *unbiased* ones (see ``moe.moe_forward``).
After each calibration batch, with ``c_i`` the cumulative load and ``c_bar`` its mean,

    b_i <- b_i + u * sign(c_bar - c_i)

where ``u`` comes from ``protocol.yaml`` (``balance.default_u``). Weights never change
and nothing is trained; the biases are fitted once on calibration data and frozen.

Guards (both numeric, both pre-registered)
------------------------------------------
``max_kl_nats``                  per-layer KL(before || after) load distribution on the
                                 held-out eval split must stay small.
``min_balance_transfer_ratio``   fraction of the calibration-time Gini gain that
                                 survives on eval, ``gain_eval / gain_calib``.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from . import env as rgenv
from . import moe, stats


# --------------------------------------------------------------------- primitives


def zero_biases(model: torch.nn.Module) -> dict[int, torch.Tensor]:
    """Fresh ``{layer: zeros(n_experts, float32)}`` on the gate's device."""
    out: dict[int, torch.Tensor] = {}
    for name, block in rgenv.find_moe_blocks(model):
        layer = stats.layer_index_from_name(name)
        out[layer] = torch.zeros(
            int(block.num_experts), dtype=torch.float32, device=block.gate.weight.device
        )
    return out


def biases_to_cpu(biases: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {int(k): v.detach().float().cpu().clone() for k, v in biases.items()}


def biases_to(biases: dict[int, torch.Tensor], device: str | torch.device) -> dict[int, torch.Tensor]:
    return {int(k): v.detach().float().to(device).clone() for k, v in biases.items()}


@torch.no_grad()
def measure_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    bias: dict[int, torch.Tensor] | None = None,
    batch_size: int = 4,
    device: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Load statistics of the model's *actual* (possibly biased) selection."""
    ids = input_ids.to(device) if device is not None else input_ids
    seen = 0
    with moe.MoePatch(model, bias=bias, record=True) as patch:
        for start in range(0, ids.shape[0], batch_size):
            if max_tokens is not None and seen >= max_tokens:
                break
            batch = ids[start : start + batch_size]
            model(input_ids=batch, use_cache=False)
            seen += int(batch.numel())
    report = patch.recorder.report()
    report["max_tokens_cap"] = max_tokens
    return report


@torch.no_grad()
def fit_biases(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    u: float = 0.001,
    iters: int = 1,
    batch_size: int = 4,
    device: str | None = None,
    max_tokens: int | None = None,
    progress: bool = True,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Fit per-expert router biases on calibration data. No weight updates.

    The update runs once per batch (online LFB): forward with the current biases,
    accumulate load, then nudge each bias by ``u * sign(mean(cum) - cum)``.
    """
    if u <= 0:
        raise ValueError(f"u must be > 0, got {u}")
    ids = input_ids.to(device) if device is not None else input_ids
    biases = zero_biases(model)
    cumulative = {layer: torch.zeros_like(b, dtype=torch.float64) for layer, b in biases.items()}
    n_batches = (ids.shape[0] + batch_size - 1) // batch_size
    seen = 0
    started = time.time()
    updates = 0

    for epoch in range(max(1, int(iters))):
        for index, start in enumerate(range(0, ids.shape[0], batch_size)):
            if max_tokens is not None and seen >= max_tokens:
                break
            batch = ids[start : start + batch_size]
            with moe.MoePatch(model, bias=biases, record=True) as patch:
                model(input_ids=batch, use_cache=False)
            per_layer = patch.recorder.per_layer_counts()
            for layer, counts in per_layer.items():
                cumulative[layer] += counts.to(torch.float64)
                target = cumulative[layer].mean()
                # sign(c_bar - c_i): push under-loaded experts up, over-loaded down.
                biases[layer] += u * torch.sign(target - cumulative[layer].to(torch.float32))
            updates += 1
            seen += int(batch.numel())
            if progress and (index % 10 == 0):
                print(
                    f"  [balance] epoch {epoch} batch {index + 1}/{n_batches} "
                    f"u={u} tokens={seen}",
                    flush=True,
                )

    info = {
        "u": float(u),
        "iters": int(iters),
        "updates": updates,
        "tokens_seen": seen,
        "elapsed_s": round(time.time() - started, 2),
        "final_bias_abs_mean": {
            int(layer): float(bias.abs().mean()) for layer, bias in biases.items()
        },
        "final_cumulative_load": {
            int(layer): cumulative[layer].tolist() for layer in sorted(cumulative)
        },
    }
    return biases, info


# --------------------------------------------------------------------- evaluation


@dataclass
class BalanceResult:
    """Everything needed to decide whether calibration-time balancing worked."""

    u: float
    iters: int
    biases: dict[int, torch.Tensor]
    calib_before: dict[str, Any]
    calib_after: dict[str, Any]
    eval_before: dict[str, Any]
    eval_after: dict[str, Any]
    fit_info: dict[str, Any] = field(default_factory=dict)

    # -- derived numbers -------------------------------------------------------
    @property
    def gini_gain_calib(self) -> float:
        return self.calib_before["gini_mean"] - self.calib_after["gini_mean"]

    @property
    def gini_gain_eval(self) -> float:
        return self.eval_before["gini_mean"] - self.eval_after["gini_mean"]

    @property
    def transfer_ratio(self) -> float:
        return stats.transfer_ratio(self.gini_gain_calib, self.gini_gain_eval)

    def kl_per_layer(self) -> dict[int, float]:
        before = self.eval_before["per_layer_counts"]
        after = self.eval_after["per_layer_counts"]
        return {int(layer): stats.kl(before[layer], after[layer]) for layer in sorted(before)}

    @property
    def kl_mean(self) -> float:
        per_layer = self.kl_per_layer()
        return float(sum(per_layer.values()) / len(per_layer)) if per_layer else 0.0

    @property
    def kl_max(self) -> float:
        per_layer = self.kl_per_layer()
        return float(max(per_layer.values())) if per_layer else 0.0

    @property
    def max_abs_bias(self) -> float:
        return max((float(b.abs().max()) for b in self.biases.values()), default=0.0)

    @property
    def improved_calib(self) -> bool:
        return self.gini_gain_calib > 0

    @property
    def improved_eval(self) -> bool:
        return self.gini_gain_eval > 0

    def warnings(self) -> list[str]:
        """Overshoot detection. Sign updates have a real instability boundary:
        once ``u * n_updates`` approaches the router logit scale, the bias
        over-corrects and Gini rises instead of falling.
        """
        out: list[str] = []
        if not self.improved_calib:
            out.append(
                f"balancing INCREASED calibration Gini "
                f"({self.calib_before['gini_mean']:.4f} -> {self.calib_after['gini_mean']:.4f}); "
                f"u={self.u} is probably too large (max|b|={self.max_abs_bias:.4f})"
            )
        elif not self.improved_eval:
            out.append(
                "calibration Gini improved but eval Gini did not; the bias did not transfer"
            )
        return out

    def gate_inputs(self) -> dict[str, float]:
        """Numbers keyed by the gate names in ``protocol.yaml``."""
        return {
            "max_kl_nats": self.kl_mean,
            "min_balance_transfer_ratio": self.transfer_ratio,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "u": self.u,
            "iters": self.iters,
            "gini_calib": {"before": self.calib_before["gini_mean"], "after": self.calib_after["gini_mean"]},
            "gini_eval": {"before": self.eval_before["gini_mean"], "after": self.eval_after["gini_mean"]},
            "gini_gain_calib": self.gini_gain_calib,
            "gini_gain_eval": self.gini_gain_eval,
            "transfer_ratio": self.transfer_ratio,
            "kl_mean_nats": self.kl_mean,
            "kl_max_nats": self.kl_max,
            "improved": {"calib": self.improved_calib, "eval": self.improved_eval},
            "max_abs_bias": self.max_abs_bias,
            "warnings": self.warnings(),
            "kl_per_layer": self.kl_per_layer(),
            "max_violation_calib": {
                "before": self.calib_before["max_violation_max"],
                "after": self.calib_after["max_violation_max"],
            },
            "max_violation_eval": {
                "before": self.eval_before["max_violation_max"],
                "after": self.eval_after["max_violation_max"],
            },
            "tokens": {
                "calib": self.calib_before["tokens_seen"],
                "eval": self.eval_before["tokens_seen"],
            },
            "fit": self.fit_info,
        }

    # -- persistence -----------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "u": self.u,
                "iters": self.iters,
                "biases": biases_to_cpu(self.biases),
                "summary": self.summary(),
            },
            target,
        )
        return target

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def evaluate_balance(
    model: torch.nn.Module,
    calib_ids: torch.Tensor,
    eval_ids: torch.Tensor,
    protocol: dict[str, Any],
    u: float | None = None,
    iters: int | None = None,
    batch_size: int = 4,
    max_tokens: int | None = None,
    device: str = "cuda",
    progress: bool = True,
) -> BalanceResult:
    """Baseline stats -> fit biases on calib -> balanced stats, on calib AND eval."""
    cfg = protocol.get("balance", {})
    u = float(u if u is not None else cfg.get("default_u", 0.001))
    iters = int(iters if iters is not None else cfg.get("iters", 1))

    calib_before = measure_model(model, calib_ids, None, batch_size, device, max_tokens)
    eval_before = measure_model(model, eval_ids, None, batch_size, device, max_tokens)
    biases, fit_info = fit_biases(
        model, calib_ids, u=u, iters=iters, batch_size=batch_size,
        device=device, max_tokens=max_tokens, progress=progress,
    )
    calib_after = measure_model(model, calib_ids, biases, batch_size, device, max_tokens)
    eval_after = measure_model(model, eval_ids, biases, batch_size, device, max_tokens)

    result = BalanceResult(
        u=u,
        iters=iters,
        biases=biases_to_cpu(biases),
        calib_before=calib_before,
        calib_after=calib_after,
        eval_before=eval_before,
        eval_after=eval_after,
        fit_info=fit_info,
    )
    if progress:
        for warning in result.warnings():
            print(f"  [balance] WARNING: {warning}", flush=True)
    return result


def apply_bias(model: torch.nn.Module, biases: dict[int, torch.Tensor], device: str = "cuda"):
    """Context manager that runs the model with frozen calibration biases."""
    return moe.MoePatch(model, bias=biases_to(biases, device))
