"""Deterministic calibration / held-out evaluation splits.

Rules that matter for the protocol:
  * calibration reads wikitext train, evaluation reads wikitext test -> disjoint by
    construction, so a "held-out" eval is actually held out.
  * every split is derived from an explicit seed and reports the token offset it
    used, so a split can be reproduced exactly from ``meta``.
  * tokenized ids are cached as tensors (small: 200x1024 int64 ~ 1.6 MB).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

MMLU_LETTERS = "ABCD"


@dataclass
class TokenSplit:
    name: str
    input_ids: torch.Tensor  # (n_seqs, seq_len) long
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def n_tokens(self) -> int:
        return int(self.input_ids.numel())

    def to(self, device: str | torch.device) -> "TokenSplit":
        return TokenSplit(self.name, self.input_ids.to(device), dict(self.meta))

    def batches(self, batch_size: int) -> Iterable[torch.Tensor]:
        for start in range(0, len(self), batch_size):
            yield self.input_ids[start : start + batch_size]

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self)} seqs x {self.seq_len} tok "
            f"({self.n_tokens} tokens) from {self.meta.get('dataset')}/"
            f"{self.meta.get('config')}/{self.meta.get('split')} @ offset {self.meta.get('token_offset')}"
        )


def _load_wikitext_rows(dataset: str, config: str, split: str, cache_dir: str | None = None):
    from datasets import load_dataset

    try:
        return load_dataset(dataset, config, split=split, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 - older/renamed repos
        return load_dataset("wikitext", config, split=split, cache_dir=cache_dir)


def _concat_token_ids(rows, tokenizer, need: int) -> list[int]:
    ids: list[int] = []
    eos = tokenizer.eos_token_id
    for row in rows:
        text = row.get("text") or ""
        if not text.strip():
            continue
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if eos is not None:
            ids.append(eos)
        if len(ids) >= need:
            break
    return ids


def build_wikitext_split(
    tokenizer,
    name: str,
    n_seqs: int,
    seq_len: int,
    dataset: str = "wikitext",
    config: str = "wikitext-103-raw-v1",
    split: str = "train",
    seed: int = 42,
    cache_path: str | Path | None = None,
    cache_dir: str | None = None,
) -> TokenSplit:
    """Contiguous, seeded block of ``n_seqs`` x ``seq_len`` tokens."""
    if cache_path is not None:
        path = Path(cache_path)
        if path.exists():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            ids = blob["input_ids"]
            if ids.shape == (n_seqs, seq_len):
                return TokenSplit(name, ids.long(), dict(blob.get("meta", {})))
            print(f"[data] ignoring stale cache {path} (shape {tuple(ids.shape)})")

    rng = np.random.default_rng(seed)
    need_tokens = n_seqs * seq_len
    # Grab extra so the seeded offset still leaves a full block.
    rows = _load_wikitext_rows(dataset, config, split, cache_dir)
    ids = _concat_token_ids(rows, tokenizer, need=need_tokens + seq_len)
    if len(ids) < need_tokens:
        raise RuntimeError(f"{dataset}/{config}/{split}: only {len(ids)} tokens available, need {need_tokens}")

    max_offset = len(ids) - need_tokens
    offset = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
    block = torch.tensor(ids[offset : offset + need_tokens], dtype=torch.long)
    input_ids = block.view(n_seqs, seq_len).contiguous()

    meta = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "seed": seed,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "token_offset": offset,
        "available_tokens": len(ids),
        "tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
    }
    out = TokenSplit(name, input_ids, meta)
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"input_ids": input_ids, "meta": meta}, path)
    return out


def build_splits(
    tokenizer,
    protocol: dict[str, Any],
    cache_dir: str | Path = "results/splits",
    cache_dir_hf: str | None = None,
) -> dict[str, TokenSplit]:
    """Calibration + held-out eval splits straight from ``protocol.yaml``."""
    calib_cfg = protocol["calibration"]
    eval_cfg = protocol["eval"]
    cache = Path(cache_dir)
    calib = build_wikitext_split(
        tokenizer,
        name="calib",
        n_seqs=int(calib_cfg["n_seqs"]),
        seq_len=int(calib_cfg["seq_len"]),
        dataset=str(calib_cfg.get("dataset", "wikitext")),
        config=str(calib_cfg.get("config", "wikitext-103-raw-v1")),
        split=str(calib_cfg.get("split", "train")),
        seed=int(calib_cfg.get("seed", 42)),
        cache_path=cache / "calib.pt",
        cache_dir=cache_dir_hf,
    )
    evalset = build_wikitext_split(
        tokenizer,
        name="eval",
        n_seqs=int(eval_cfg["n_seqs"]),
        seq_len=int(eval_cfg["seq_len"]),
        dataset=str(eval_cfg.get("dataset", "wikitext")),
        config=str(eval_cfg.get("config", "wikitext-103-raw-v1")),
        split=str(eval_cfg.get("split", "test")),
        seed=int(eval_cfg.get("seed", 42)),
        cache_path=cache / "eval.pt",
        cache_dir=cache_dir_hf,
    )
    return {"calib": calib, "eval": evalset}


# --------------------------------------------------------------------------- MMLU


def _format_question(question: str, choices: list[str]) -> str:
    body = "\n".join(f"{MMLU_LETTERS[i]}. {choice}" for i, choice in enumerate(choices))
    return f"{question}\n{body}\nAnswer:"


def _few_shot_prefix(shots: list[dict[str, Any]], subject: str) -> str:
    lines = [f"The following are multiple choice questions (with answers) about {subject}.", ""]
    for shot in shots:
        lines.append(_format_question(shot["question"], shot["choices"]))
        lines.append(f" {MMLU_LETTERS[int(shot['answer'])]}")
        lines.append("")
    return "\n".join(lines)


def build_mmlu_prompts(
    n_questions: int = 500,
    n_shot: int = 3,
    seed: int = 42,
    dataset: str = "cais/mmlu",
    config: str = "all",
    cache_dir: str | None = None,
    shot_split: str = "dev",
    question_split: str = "test",
) -> list[dict[str, Any]]:
    """Deterministic MMLU subset with per-subject few-shot exemplars."""
    from datasets import load_dataset

    dev = load_dataset(dataset, config, split=shot_split, cache_dir=cache_dir)
    test = load_dataset(dataset, config, split=question_split, cache_dir=cache_dir)

    rng = np.random.default_rng(seed)
    n = min(int(n_questions), len(test))
    picked = np.sort(rng.choice(len(test), size=n, replace=False))

    shots_by_subject: dict[str, list[dict[str, Any]]] = {}
    if n_shot > 0:
        by_subject: dict[str, list[int]] = {}
        for idx, subject in enumerate(dev["subject"]):
            by_subject.setdefault(subject, []).append(idx)
        for subject, indices in by_subject.items():
            k = min(n_shot, len(indices))
            # Stable per-subject seed: str.hash() is salted per process, so use crc32.
            subject_seed = zlib.crc32(subject.encode("utf-8")) % (2**31)
            chosen = np.random.default_rng([seed, subject_seed]).choice(
                indices, size=k, replace=False
            )
            shots_by_subject[subject] = [dev[int(i)] for i in sorted(chosen)]

    prompts: list[dict[str, Any]] = []
    for i in picked:
        row = test[int(i)]
        subject = row["subject"]
        choices = list(row["choices"])
        answer_idx = int(row["answer"])
        prefix = _few_shot_prefix(shots_by_subject.get(subject, []), subject) if n_shot else ""
        prompt = prefix + _format_question(row["question"], choices) if prefix else _format_question(row["question"], choices)
        prompts.append(
            {
                "index": int(i),
                "subject": subject,
                "prompt": prompt,
                "choices": choices,
                "answer_idx": answer_idx,
                "answer_letter": MMLU_LETTERS[answer_idx],
                "n_shot": n_shot,
            }
        )
    return prompts
