"""Evaluation: wikitext perplexity, MMLU 3-shot subset, memory delta, tokens/sec.

All three comparison rules the protocol depends on live here:
  * perplexity is token-weighted over the whole split, computed identically for
    pruned and unpruned models;
  * MMLU scores the four answer letters from the logits at the final position;
  * memory is reported both *measured* (allocator counters) and *estimated*
    (parameters x bytes-per-parameter at the run's precision).
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

MMLU_LETTERS = "ABCD"

# bytes per parameter at each storage precision, used only for the analytic estimate
BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "8bit": 1.0, "4bit": 0.5, "fp32": 4.0}


# ------------------------------------------------------------------- perplexity


def nll_sum(model, input_ids: torch.Tensor, position_chunk: int = 256) -> tuple[float, int]:
    """Summed negative log-likelihood and target-token count for one batch."""
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1]
    targets = input_ids[:, 1:]
    vocab = logits.shape[-1]
    total = 0.0
    for start in range(0, logits.shape[1], position_chunk):
        chunk = logits[:, start : start + position_chunk].float().reshape(-1, vocab)
        target = targets[:, start : start + position_chunk].reshape(-1)
        total += float(F.cross_entropy(chunk, target, reduction="sum"))
        del chunk
    return total, int(targets.numel())


@torch.no_grad()
def wikitext_perplexity(
    model,
    input_ids: torch.Tensor,
    batch_size: int = 1,
    device: str | None = "cuda",
    position_chunk: int = 256,
    progress: bool = False,
) -> dict[str, Any]:
    """Token-weighted perplexity over fixed-length windows."""
    ids = input_ids.to(device) if device is not None else input_ids
    total_nll, total_tokens = 0.0, 0
    n_batches = (ids.shape[0] + batch_size - 1) // batch_size
    for index, start in enumerate(range(0, ids.shape[0], batch_size)):
        batch = ids[start : start + batch_size]
        nll, tokens = nll_sum(model, batch, position_chunk=position_chunk)
        total_nll += nll
        total_tokens += tokens
        if progress and index % 10 == 0:
            print(f"  [ppl] batch {index + 1}/{n_batches} tokens={total_tokens}", flush=True)
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "perplexity": math.exp(mean_nll) if mean_nll < 700 else float("inf"),
        "nll_per_token": mean_nll,
        "tokens": total_tokens,
        "seqs": int(ids.shape[0]),
        "seq_len": int(ids.shape[1]),
    }


# ------------------------------------------------------------------------ MMLU


def resolve_letter_ids(tokenizer, letters: str = MMLU_LETTERS) -> list[int]:
    """Token ids for the answer letters, requiring a single token each."""
    ids: list[int] = []
    for letter in letters:
        for form in (f" {letter}", letter):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(int(encoded[0]))
                break
        else:
            raise ValueError(
                f"answer letter {letter!r} is not a single token for this tokenizer "
                f"(tried ' {letter}' and {letter!r}); MMLU scoring would be ambiguous"
            )
    if len(set(ids)) != len(ids):
        raise ValueError(f"answer letters collide on token ids: {ids}")
    return ids


def _encode_batch(tokenizer, prompts: Sequence[str], device: str | None):
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    width = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    for row, ids in enumerate(encoded):
        # left padding so the final position is the real last token
        input_ids[row, width - len(ids) :] = torch.as_tensor(ids, dtype=torch.long)
        attention[row, width - len(ids) :] = 1
    if device is not None:
        input_ids, attention = input_ids.to(device), attention.to(device)
    return input_ids, attention


@torch.no_grad()
def mmlu_accuracy(
    model,
    tokenizer,
    prompts: Sequence[dict[str, Any]],
    batch_size: int = 8,
    device: str | None = "cuda",
    progress: bool = False,
) -> dict[str, Any]:
    """3-shot-style accuracy: argmax over the four letter tokens at the last position."""
    if not prompts:
        raise ValueError("no MMLU prompts")
    letter_ids = torch.as_tensor(resolve_letter_ids(tokenizer), dtype=torch.long)
    if device is not None:
        letter_ids = letter_ids.to(device)

    correct = 0
    per_subject: dict[str, list[int]] = {}
    predictions: list[int] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for index, start in enumerate(range(0, len(prompts), batch_size)):
        chunk = prompts[start : start + batch_size]
        input_ids, attention = _encode_batch(tokenizer, [p["prompt"] for p in chunk], device)
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        final = logits[:, -1, :].float()
        scores = final.index_select(1, letter_ids)
        predicted = scores.argmax(dim=-1).tolist()
        for offset, item in enumerate(chunk):
            pred = int(predicted[offset])
            predictions.append(pred)
            target = int(item["answer_idx"])
            hit = int(pred == target)
            correct += hit
            per_subject.setdefault(item["subject"], []).append(hit)
        if progress and index % 10 == 0:
            print(f"  [mmlu] batch {index + 1}/{n_batches} acc={correct / len(predictions):.4f}", flush=True)

    return {
        "accuracy": correct / len(prompts),
        "n_questions": len(prompts),
        "n_correct": correct,
        "n_shot": int(prompts[0].get("n_shot", 0)),
        "per_subject": {
            subject: {"accuracy": sum(v) / len(v), "n": len(v)}
            for subject, v in sorted(per_subject.items())
        },
        "predictions": predictions,
    }


# ---------------------------------------------------------------------- memory


def memory_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    }


def param_report(model, precision: str = "bf16") -> dict[str, Any]:
    """Parameter accounting; the estimate is a labeling aid, not the headline."""
    total = sum(p.numel() for p in model.parameters())
    bytes_per = BYTES_PER_PARAM.get(precision)
    return {
        "params_total": int(total),
        "params_billions": round(total / 1e9, 3),
        "precision": precision,
        "bytes_per_param": bytes_per,
        "estimated_gb": round(total * bytes_per / 1024**3, 3) if bytes_per else None,
    }


# ------------------------------------------------------------------ throughput


@torch.no_grad()
def throughput(
    model,
    seq_len: int = 512,
    warmup: int = 5,
    runs: int = 10,
    batch_size: int = 1,
    device: str = "cuda",
    seed: int = 0,
) -> dict[str, Any]:
    """Tokens/sec with multi-run variance; secondary metric, so report the spread."""
    generator = torch.Generator().manual_seed(seed)
    vocab = int(getattr(model.config, "vocab_size", 32000))
    input_ids = torch.randint(0, min(vocab, 30000), (batch_size, seq_len), generator=generator, device=device)

    for _ in range(warmup):
        model(input_ids=input_ids, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        model(input_ids=input_ids, use_cache=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    tokens = seq_len * batch_size
    per_run = [tokens / t for t in times]
    mean_tps = statistics.mean(per_run)
    return {
        "tokens_per_sec_mean": mean_tps,
        "tokens_per_sec_std": statistics.pstdev(per_run) if len(per_run) > 1 else 0.0,
        "tokens_per_sec_min": min(per_run),
        "tokens_per_sec_max": max(per_run),
        "cv": (statistics.pstdev(per_run) / mean_tps) if len(per_run) > 1 and mean_tps > 0 else 0.0,
        "seconds_per_run": times,
        "runs": runs,
        "warmup": warmup,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }


# ------------------------------------------------------------------- everything


def evaluate_all(
    model,
    tokenizer=None,
    eval_ids: torch.Tensor | None = None,
    prompts: Sequence[dict[str, Any]] | None = None,
    protocol: dict[str, Any] | None = None,
    precision: str = "bf16",
    device: str = "cuda",
    batch_size: int = 1,
    mmlu_batch_size: int = 8,
    do_mmlu: bool = True,
    do_throughput: bool = True,
) -> dict[str, Any]:
    """One call for a full evaluation point (used by notebook 02's curve)."""
    metrics: dict[str, Any] = {"params": param_report(model, precision)}
    cfg = (protocol or {}).get("metrics", {})

    if eval_ids is not None:
        metrics["perplexity"] = wikitext_perplexity(model, eval_ids, batch_size=batch_size, device=device)
    if prompts is not None and do_mmlu:
        mmlu = mmlu_accuracy(model, tokenizer, prompts, batch_size=mmlu_batch_size, device=device)
        mmlu.pop("predictions", None)
        metrics["mmlu"] = mmlu
    if do_throughput:
        runner = cfg.get("throughput", {})
        metrics["throughput"] = throughput(
            model,
            seq_len=int(runner.get("seq_len", 512)),
            warmup=int(runner.get("warmup", 5)),
            runs=int(runner.get("runs", 10)),
            device=device,
        )
    metrics["memory"] = memory_snapshot()
    return metrics
