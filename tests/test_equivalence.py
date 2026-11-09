"""Equivalence tests. These gate everything downstream: do not trust any balancing or
pruning number until this file passes on BOTH OLMoE and Qwen.

What is asserted, and how strongly
----------------------------------
1. No-op patch (``bias=None``, no mask) is **bitwise identical** to the unpatched model.
   This is the load-bearing guarantee: our mirrored forward must reproduce upstream's
   ops exactly, so any later difference is attributable to the bias, not to our rewrite.
2. Zero bias is also bitwise identical (``s + 0`` must be exact).
3. Mask simulation == real removal: identical selection (exact, integer, in original
   expert space) and outputs equal within float rounding. Bitwise is reported but not
   required, because pruning changes the router GEMM shape.
4. ``norm_topk_prob`` is exercised in both states, because it decides whether masking
   and removal can diverge through renormalization.
5. The bias really is selection-only: the test also proves the *biased-weight* variant
   would give a different answer, so the check is not vacuous.
"""

import copy

import pytest
import torch

from routerguard import env, moe
from routerguard.stats import layer_index_from_name
from tests.conftest import build_model, make_ids


# ---------------------------------------------------------------- 1. no-op identity


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_noop_patch_is_bitwise_identical(flavor, dtype, norm_topk_prob):
    model = build_model(flavor, norm_topk_prob=norm_topk_prob, dtype=dtype)
    ids = make_ids()
    with torch.no_grad():
        reference = model(input_ids=ids, use_cache=False).logits
        with moe.MoePatch(model) as patch:
            assert patch.layers, "patch installed on no layers"
            patched = model(input_ids=ids, use_cache=False).logits
    result = moe.equivalence_check(patched, reference)
    assert result["bitwise_equal"], f"no-op patch is not bitwise identical: {result}"
    # Forward must be restored afterwards.
    assert not patch._active


def test_noop_patch_restores_forward(flavor):
    model = build_model(flavor)
    before = model.model.layers[0].mlp.forward
    with moe.MoePatch(model):
        assert model.model.layers[0].mlp.forward != before
    assert model.model.layers[0].mlp.forward == before


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_zero_bias_is_bitwise_identical(flavor, norm_topk_prob):
    model = build_model(flavor, norm_topk_prob=norm_topk_prob)
    ids = make_ids()
    n_experts = model.config.num_experts
    zeros = torch.zeros(n_experts, dtype=torch.float32)
    with torch.no_grad():
        reference = model(input_ids=ids, use_cache=False).logits
        with moe.MoePatch(model, bias=zeros):
            patched = model(input_ids=ids, use_cache=False).logits
    assert torch.equal(patched, reference), moe.equivalence_check(patched, reference)


# ------------------------------------------------- 3./4. mask simulation == removal


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_mask_sim_matches_real_removal(flavor, norm_topk_prob):
    keep = [0, 2, 3, 5, 7]  # 5 of 8 experts survive
    model = build_model(flavor, norm_topk_prob=norm_topk_prob)
    ids = make_ids(batch=2, seq=16)
    n_experts = model.config.num_experts
    layers = sorted(layer_index_from_name(name) for name, _ in env.find_moe_blocks(model))

    with torch.no_grad():
        # (a) mask simulation on the intact model
        with moe.MaskSim(model, {layer: keep for layer in layers}, record=True) as sim:
            sim_logits = model(input_ids=ids, use_cache=False).logits
            sim_counts = sim.recorder.per_layer_counts()
            sim_tokens = sim.recorder.tokens

        # (b) real removal on a copy
        pruned = copy.deepcopy(model)
        report = moe.apply_removal(pruned, {layer: keep for layer in layers})
        with moe.MoePatch(pruned, record=True) as real:
            real_logits = pruned(input_ids=ids, use_cache=False).logits
            real_counts = real.recorder.per_layer_counts()

    # Selection must be *exactly* the same, compared in the original expert space.
    keep_t = torch.as_tensor(keep, dtype=torch.long)
    for layer in layers:
        mapped = torch.zeros(n_experts, dtype=torch.long)
        mapped[keep_t] = real_counts[layer]
        assert torch.equal(sim_counts[layer], mapped), (
            f"layer {layer}: selection differs\n sim  {sim_counts[layer]}\n real {mapped}"
        )
        assert int(real_counts[layer].sum()) == sim_tokens * min(model.config.num_experts_per_tok, len(keep))

    result = moe.equivalence_check(sim_logits, real_logits, atol=1e-6, rtol=1e-5)
    assert result["allclose"], f"mask sim != real removal: {result}"
    assert report["removed_total"] == len(layers) * (n_experts - len(keep))


def test_removal_clamps_top_k(flavor):
    """top_k > kept experts must not crash: top_k clamps to the kept count."""
    model = build_model(flavor)
    keep = [1, 4]
    assert model.config.num_experts_per_tok == 3
    report = moe.apply_removal(model, keep)
    block = env.find_moe_blocks(model)[0][1]
    assert block.top_k == 2, report
    assert block.num_experts == 2
    assert len(block.experts) == 2
    with torch.no_grad():
        out = model(input_ids=make_ids(), use_cache=False).logits
    assert torch.isfinite(out).all()


def test_removal_updates_gate_and_config(flavor):
    model = build_model(flavor)
    keep = [0, 3, 6]
    old_gate = model.model.layers[0].mlp.gate
    old_rows = old_gate.weight.detach()[keep].clone()
    moe.apply_removal(model, keep)
    new_gate = model.model.layers[0].mlp.gate
    assert new_gate.out_features == len(keep)
    assert torch.equal(new_gate.weight.detach(), old_rows), "gate rows must be sliced, not reinitialized"
    assert model.config.num_experts == len(keep)
    assert model.model.layers[0].mlp.experts[0] is not None


def test_shared_expert_never_pruned(flavor):
    """Qwen's always-on shared expert must survive pruning untouched."""
    model = build_model(flavor)
    blocks = env.find_moe_blocks(model)
    has_shared = hasattr(blocks[0][1], "shared_expert")
    if not has_shared:
        pytest.skip("no shared expert in this flavor")
    shared_before = blocks[0][1].shared_expert
    gate_before = blocks[0][1].shared_expert_gate
    moe.apply_removal(model, [0, 1])
    assert blocks[0][1].shared_expert is shared_before
    assert blocks[0][1].shared_expert_gate is gate_before


# --------------------------------------------------- 5. selection-only bias is real


def test_bias_is_selection_only_not_weighting(flavor):
    """Selection on (s+b), mixing from unbiased softmax(s) -- and prove it differs
    from the naive 'bias everything' implementation."""
    model = build_model(flavor)
    block = env.find_moe_blocks(model)[0][1]
    hidden = model.config.hidden_size
    torch.manual_seed(3)
    x = torch.randn(1, 6, hidden)
    n_experts, top_k = block.num_experts, block.top_k
    bias = torch.zeros(n_experts)
    bias[0] = 5.0  # force expert 0 into the selection

    out, logits = moe.moe_forward(block, x, bias=bias)

    flat = x.reshape(-1, hidden)
    probs = torch.softmax(logits, dim=1, dtype=torch.float)
    sel = torch.topk(logits.float() + bias, top_k, dim=-1).indices
    assert (sel == 0).any(), "bias failed to change the selection"

    unbiased_w = probs.gather(1, sel)
    biased_w = torch.softmax(logits.float() + bias, dim=1).gather(1, sel)
    assert not torch.allclose(unbiased_w, biased_w), "test is vacuous: weights coincide"

    # Reference: unbiased weights, same experts.
    expected = torch.zeros(flat.shape[0], hidden)
    for slot in range(top_k):
        for token in range(flat.shape[0]):
            e = int(sel[token, slot])
            expected[token] += block.experts[e](flat[token : token + 1])[0] * unbiased_w[token, slot]
    if hasattr(block, "shared_expert"):
        shared = block.shared_expert(flat) * torch.sigmoid(block.shared_expert_gate(flat))
        expected = expected + shared
    assert torch.allclose(out.reshape(-1, hidden), expected, atol=1e-5, rtol=1e-4), "weights are not the unbiased ones"


def test_bias_length_validated(flavor):
    model = build_model(flavor)
    with moe.MoePatch(model, bias=torch.zeros(3)):
        with pytest.raises(ValueError, match="bias has 3 entries"):
            with torch.no_grad():
                model(input_ids=make_ids(), use_cache=False)


def test_recorder_counts_match_manual(flavor):
    model = build_model(flavor)
    ids = make_ids(batch=3, seq=8)
    with torch.no_grad():
        with moe.MoePatch(model, record=True) as patch:
            model(input_ids=ids, use_cache=False)
    counts = patch.recorder.per_layer_counts()
    top_k = model.config.num_experts_per_tok
    for layer, c in counts.items():
        assert int(c.sum()) == 3 * 8 * top_k
        assert c.numel() == model.config.num_experts
    assert patch.recorder.tokens == 3 * 8


# ------------------------------------------------------- real checkpoints (opt-in)


@pytest.mark.parametrize("which", ["olmoe", "qwen"])
def test_noop_patch_bitwise_real_checkpoint(which):
    from tests.conftest import REAL_MODELS

    if not REAL_MODELS or not torch.cuda.is_available():
        pytest.skip("set ROUTERGUARD_REAL_MODELS=1 on a GPU runtime")

    from routerguard import models

    if which == "olmoe":
        model, tokenizer, info = models.load_olmoe()
    else:
        model, tokenizer, info = models.load_qwen()
    ids = torch.randint(0, 1000, (1, 256), device="cuda")
    with torch.no_grad():
        reference = model(input_ids=ids, use_cache=False).logits
        with moe.MoePatch(model):
            patched = model(input_ids=ids, use_cache=False).logits
    result = moe.equivalence_check(patched, reference)
    assert result["bitwise_equal"], f"{which} {info.precision}: {result}"
