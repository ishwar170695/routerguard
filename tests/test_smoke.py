"""Tiny end-to-end pipeline: 2 MoE layers x 8 samples, must finish in well under 60 s.

Deliberately runs on CPU with tiny randomly-initialised models built from the real
transformers classes, so it is a *logic* smoke test (no downloads, no GPU, no
numerics claims). Real-checkpoint behaviour is covered by the opt-in tests in
``test_equivalence.py`` and by the notebooks.
"""

import time

import pytest
import torch

from routerguard import data, env, moe, stats
from tests.conftest import build_model, make_ids

TIME_BUDGET_S = 60.0


def _split(n_seqs=8, seq_len=16, vocab=128, seed=5) -> data.TokenSplit:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n_seqs, seq_len), generator=generator)
    return data.TokenSplit("synthetic", ids, {"dataset": "synthetic", "split": "n/a", "token_offset": 0})


@torch.no_grad()
def _perplexity(model, input_ids: torch.Tensor) -> float:
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1].float()
    targets = input_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    return float(torch.exp(loss))


def test_full_pipeline_smoke(flavor):
    started = time.time()
    model = build_model(flavor, seed=7)
    split = _split()

    # 1. structure
    structure = env.assert_moe_structure(model)
    assert structure["num_moe_layers"] == 2
    assert structure["num_experts_per_layer"] == 8
    assert structure["has_shared_expert"] == (flavor == "qwen")

    # 2. baseline load statistics
    report = stats.collect_load_stats(model, split, batch_size=4)
    assert report["tokens_seen"] == split.n_tokens
    assert 0.0 <= report["gini_mean"] <= 1.0
    assert len(report["per_layer_counts"]) == 2

    # 3. no-op patch must not change the model
    with torch.no_grad():
        reference = model(input_ids=split.input_ids, use_cache=False).logits
        with moe.MoePatch(model, record=True) as patch:
            patched = model(input_ids=split.input_ids, use_cache=False).logits
            rec = patch.recorder.report()
    assert torch.equal(patched, reference)
    assert rec["tokens_seen"] == split.n_tokens

    # 4. balance-shaped bias must be applied and must be reversible
    bias = torch.zeros(model.config.num_experts)
    bias[0] = 1.0
    with torch.no_grad(), moe.MoePatch(model, bias=bias, record=True) as patch:
        biased = model(input_ids=split.input_ids, use_cache=False).logits
        biased_counts = patch.recorder.per_layer_counts()
    assert not torch.equal(biased, reference), "a +1.0 bias on one expert changed nothing"
    for layer, counts in biased_counts.items():
        assert int(counts[0]) >= int(rec["per_layer_counts"][layer][0]), "bias did not shift load to expert 0"
    with torch.no_grad():
        after = model(input_ids=split.input_ids, use_cache=False).logits
    assert torch.equal(after, reference), "patch leaked state after exit"

    # 5. mask simulation vs real removal
    keep = [0, 1, 2, 3, 4]
    with torch.no_grad(), moe.MaskSim(model, keep):
        masked = model(input_ids=split.input_ids, use_cache=False).logits
    pruned = build_model(flavor, seed=7)
    moe.apply_removal(pruned, keep)
    with torch.no_grad():
        removed = pruned(input_ids=split.input_ids, use_cache=False).logits
    check = moe.equivalence_check(masked, removed, atol=1e-6, rtol=1e-5)
    assert check["allclose"], check

    # 6. perplexity + parameter accounting
    ppl_before = _perplexity(model, split.input_ids)
    ppl_after = _perplexity(pruned, split.input_ids)
    assert torch.isfinite(torch.tensor([ppl_before, ppl_after])).all()
    params_before = sum(p.numel() for p in model.parameters())
    params_after = sum(p.numel() for p in pruned.parameters())
    assert params_after < params_before, "pruning did not reduce parameters"

    elapsed = time.time() - started
    print(
        f"\n[{flavor}] ppl {ppl_before:.3f} -> {ppl_after:.3f} | "
        f"params {params_before:,} -> {params_after:,} ({100 * (1 - params_after / params_before):.1f}% smaller) | "
        f"{elapsed:.1f}s"
    )
    assert elapsed < TIME_BUDGET_S, f"smoke test took {elapsed:.1f}s > {TIME_BUDGET_S}s"


def test_keep_indices_from_ratio():
    keep = moe.keep_indices_from_ratio(60, 0.25)
    assert len(keep) == 45
    assert keep == sorted(keep)
    assert len(moe.keep_indices_from_ratio(64, 0.5)) == 32
    # deterministic for a given generator seed
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    assert moe.keep_indices_from_ratio(8, 0.5, generator=g1) == moe.keep_indices_from_ratio(8, 0.5, generator=g2)
    # importance-based selection keeps the top scores
    scores = torch.arange(8, dtype=torch.float32)
    assert moe.keep_indices_from_ratio(8, 0.5, scores=scores) == [4, 5, 6, 7]


def test_prune_block_rejects_bad_indices(flavor):
    model = build_model(flavor)
    block = env.find_moe_blocks(model)[0][1]
    with pytest.raises(ValueError):
        moe.prune_block_inplace(block, [0, 0, 1])
    with pytest.raises(ValueError):
        moe.prune_block_inplace(block, [99])
    with pytest.raises(ValueError):
        moe.prune_block_inplace(block, [])
