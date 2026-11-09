"""Step 4 tests: similarity collection, greedy redundancy removal, 2x2 support.

The similarity metric is checked against an independent from-scratch computation on the
same captured layer inputs, and the greedy search is checked against constructed
matrices where the answer is known by inspection.
"""

import numpy as np
import pytest
import torch

from routerguard import env, moe, prune, similarity
from routerguard.stats import layer_index_from_name
from tests.conftest import build_model, make_ids


# ------------------------------------------------------------------ similarity


def _reference_cosine(model, layer, input_ids):
    """From scratch: capture the layer input, run all experts, centre, cosine."""
    blocks = {layer_index_from_name(n): b for n, b in env.find_moe_blocks(model)}
    block = blocks[layer]
    captured = []

    def pre_hook(module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = block.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    handle.remove()

    x = captured[0].reshape(-1, captured[0].shape[-1])
    with torch.no_grad():
        outputs = torch.stack([e(x) for e in block.experts], dim=0)  # (n_experts, tokens, hidden)
        centered = outputs - outputs.mean(dim=0, keepdim=True)
        flat = centered.reshape(len(block.experts), -1).to(torch.float64)
        gram = flat @ flat.T
        norms = (flat * flat).sum(dim=1)
    return similarity.gram_to_cosine(gram, norms)


def test_similarity_matches_independent_reference(flavor):
    model = build_model(flavor)
    ids = make_ids(batch=2, seq=8, seed=11)
    with similarity.SimilarityCollector(model, max_tokens_per_layer=10_000) as collector:
        collector.collect(ids, batch_size=1)
    matrices = collector.matrices()

    assert sorted(matrices) == [0, 1]
    for layer in sorted(matrices):
        expected = _reference_cosine(model, layer, ids)
        got = matrices[layer]
        assert got.shape == expected.shape
        assert np.allclose(got, expected, atol=1e-5), (
            f"layer {layer}: max diff {np.abs(got - expected).max():.2e}"
        )


def test_similarity_matrix_properties(flavor):
    model = build_model(flavor)
    ids = make_ids(batch=2, seq=8, seed=12)
    collector = similarity.SimilarityCollector(model, max_tokens_per_layer=10_000)
    with collector:
        collector.collect(ids, batch_size=2)
    for layer, matrix in collector.matrices().items():
        assert np.allclose(matrix, matrix.T, atol=1e-6), "not symmetric"
        assert np.allclose(np.diag(matrix), 1.0, atol=1e-6), "diagonal must be 1"
        assert matrix.min() >= -1.0 - 1e-6 and matrix.max() <= 1.0 + 1e-6
        assert matrix.shape == (model.config.num_experts, model.config.num_experts)


def test_identical_experts_have_cosine_one(flavor):
    model = build_model(flavor)
    block = env.find_moe_blocks(model)[0][1]
    block.experts[3].load_state_dict(block.experts[1].state_dict())
    ids = make_ids(batch=2, seq=8, seed=13)
    collector = similarity.SimilarityCollector(model, max_tokens_per_layer=10_000)
    with collector:
        collector.collect(ids, batch_size=2)
    matrix = collector.matrices()[0]
    assert matrix[1, 3] == pytest.approx(1.0, abs=1e-6), matrix[1, 3]
    # and they are the most redundant pair in the layer
    top_pair = similarity.redundancy_pairs(matrix, top=1)[0]
    assert {top_pair[0], top_pair[1]} == {1, 3}, top_pair


def test_token_budget_is_respected(flavor):
    model = build_model(flavor)
    ids = make_ids(batch=4, seq=8, seed=14)  # 32 tokens
    collector = similarity.SimilarityCollector(model, max_tokens_per_layer=8)
    with collector:
        collector.collect(ids, batch_size=2)
    for layer, used in collector.tokens_per_layer().items():
        assert used == 8, f"layer {layer} used {used} tokens, budget was 8"


def test_caching_and_resume(flavor, tmp_path):
    model = build_model(flavor)
    ids = make_ids(batch=2, seq=8, seed=15)
    first = similarity.build_similarity(
        model, ids, tmp_path / "sim", max_tokens_per_layer=10_000, batch_size=1, progress=False
    )
    assert sorted(first.computed) == [0, 1]
    assert first.cached == []
    assert (tmp_path / "sim" / "layer_00.npy").exists()

    second = similarity.build_similarity(
        model, ids, tmp_path / "sim", max_tokens_per_layer=10_000, batch_size=1, progress=False
    )
    assert second.computed == [], "cached layers must not be recomputed"
    assert sorted(second.cached) == [0, 1]
    for layer in (0, 1):
        assert np.array_equal(first.matrices[layer], second.matrices[layer])

    resumed = similarity.load_all_similarity(tmp_path / "sim")
    assert sorted(resumed) == [0, 1]
    with pytest.raises(FileNotFoundError):
        similarity.load_similarity(tmp_path / "sim", 99)


def test_uniqueness_scores_are_negated_redundancy(flavor):
    model = build_model(flavor)
    ids = make_ids(batch=2, seq=8, seed=16)
    collector = similarity.SimilarityCollector(model, max_tokens_per_layer=10_000)
    with collector:
        collector.collect(ids, batch_size=2)
    matrix = collector.matrices()[0]
    redundancy = similarity.pairwise_mean_similarity(matrix)
    uniqueness = similarity.uniqueness_scores(matrix)
    assert torch.allclose(uniqueness, -redundancy)
    keep = [0, 2, 4]
    sub = similarity.pairwise_mean_similarity(matrix, keep)
    assert sub.numel() == 3


# ---------------------------------------------------------------------- greedy


def _handmade_sim(n=6, pairs=((0, 5),)):
    matrix = np.eye(n, dtype=np.float32) * 0.0 + 0.1
    np.fill_diagonal(matrix, 1.0)
    for i, j in pairs:
        matrix[i, j] = matrix[j, i] = 0.95
    return matrix


def test_greedy_drops_the_redundant_pair_first():
    matrix = _handmade_sim(6, pairs=((0, 5),))
    keeps, info = prune.greedy_redundant_keeps(matrix, ratio=1 / 6)  # drop exactly one
    assert info["steps"] == 1 and not info["truncated"]
    assert len(keeps) == 5
    assert set(keeps) in ({1, 2, 3, 4, 5}, {0, 1, 2, 3, 4}), keeps
    assert 0 not in keeps or 5 not in keeps, "one of the redundant pair must go"


def test_greedy_recomputes_between_steps():
    # 0,1,2 are mutually similar; dropping them all should follow in sequence.
    matrix = _handmade_sim(6, pairs=((0, 1), (1, 2), (0, 2)))
    keeps, info = prune.greedy_redundant_keeps(matrix, ratio=3 / 6)
    assert len(keeps) == 3
    assert not set(keeps) & {0, 1, 2}, f"redundant cluster survived: {keeps}"
    assert info["steps"] == 3


def test_greedy_respects_ratio_and_ordering():
    matrix = _handmade_sim(8)
    for ratio, expected in ((0.25, 6), (0.5, 4)):
        keeps, info = prune.greedy_redundant_keeps(matrix, ratio)
        assert len(keeps) == expected
        assert keeps == sorted(keeps)
        assert info["target_keep"] == expected


def test_greedy_is_deterministic():
    matrix = _handmade_sim(8)
    a, _ = prune.greedy_redundant_keeps(matrix, 0.5)
    b, _ = prune.greedy_redundant_keeps(matrix, 0.5)
    assert a == b


def test_greedy_truncates_at_step_cap():
    matrix = _handmade_sim(8)
    keeps, info = prune.greedy_redundant_keeps(matrix, 0.5, max_steps=1)
    assert info["truncated"] is True
    assert info["steps"] == 1
    assert len(keeps) == 7


def test_utilization_weight_protects_busy_experts():
    # All experts equally similar, so only utilization can decide.
    matrix = np.full((4, 4), 0.5, dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    utilization = [100, 10, 5, 1]  # expert 0 is heavily used
    keeps, _ = prune.greedy_redundant_keeps(matrix, 0.5, utilization, utilization_weight=1.0)
    assert 0 in keeps, f"heavily used expert was pruned: {keeps}"
    assert len(keeps) == 2


def test_redundancy_keeps_per_layer(flavor):
    model = build_model(flavor)
    sim = {layer: _handmade_sim(model.config.num_experts) for layer in (0, 1)}
    keeps, infos = prune.redundancy_keeps(sim, 0.5)
    assert sorted(keeps) == [0, 1]
    assert all(len(v) == model.config.num_experts // 2 for v in keeps.values())
    assert infos[0]["steps"] == model.config.num_experts // 2


def test_keeps_for_requires_similarity(flavor):
    model = build_model(flavor)
    with pytest.raises(ValueError, match="needs similarity matrices"):
        prune.keeps_for(model, "redundancy", 0.5)


def test_keeps_for_redundancy_dispatch(flavor):
    model = build_model(flavor)
    sim = {layer: _handmade_sim(8) for layer in (0, 1)}
    keeps = prune.keeps_for(model, "redundancy", 0.25, sim=sim)
    assert all(len(v) == 6 for v in keeps.values())


# ------------------------------------------------------------------- 2x2 support


def test_masksim_accepts_bias(flavor):
    model = build_model(flavor)
    ids = make_ids()
    bias = torch.zeros(model.config.num_experts)
    bias[0] = 5.0
    keeps = [0, 1, 2, 3, 4]

    with torch.no_grad():
        with moe.MaskSim(model, keeps, record=True) as plain:
            plain_out = model(input_ids=ids, use_cache=False).logits
            plain_counts = plain.recorder.per_layer_counts()
        with moe.MaskSim(model, keeps, record=True, bias=bias) as balanced:
            biased_out = model(input_ids=ids, use_cache=False).logits
            biased_counts = balanced.recorder.per_layer_counts()

    assert not torch.equal(plain_out, biased_out), "bias had no effect under mask simulation"
    for layer in plain_counts:
        assert int(biased_counts[layer][0]) > int(plain_counts[layer][0]), (
            f"layer {layer}: bias did not shift load toward expert 0"
        )
        assert int(biased_counts[layer].sum()) == int(plain_counts[layer].sum())


def test_masksim_bias_device_and_dtype_are_coerced(flavor):
    """Biases live on CPU between phases; the forward must not see a CPU float64 tensor."""
    model = build_model(flavor)
    bias = torch.zeros(model.config.num_experts, dtype=torch.float64)
    bias[2] = 1.0
    with torch.no_grad(), moe.MoePatch(model, bias=bias):
        out = model(input_ids=make_ids(), use_cache=False).logits
    assert torch.isfinite(out).all()


def test_masksim_bias_indexing_is_in_unpruned_space(flavor):
    """Bias is length n_experts (unpruned); a wrong length must fail loudly."""
    model = build_model(flavor)
    with moe.MaskSim(model, [0, 1], bias=torch.zeros(3)):
        with pytest.raises(ValueError, match="bias has 3 entries"):
            with torch.no_grad():
                model(input_ids=make_ids(), use_cache=False)
