"""Step 3 tests: balancing loop, guards, evaluation primitives, pruning baselines.

Every test here runs on CPU with tiny models or stubs — no downloads, no GPU. The
point is that these functions are *executed* and checked against an independent
reference, not merely imported.
"""

from types import SimpleNamespace

import pytest
import torch

from routerguard import balance, eval as rgeval, moe, prune, stats
from tests.conftest import build_model, make_ids


def _skewed_model(flavor="olmoe", boost=3.0, seed=0):
    """A model whose router loves expert 0, so balancing has something to fix."""
    model = build_model(flavor, seed=seed)
    with torch.no_grad():
        for _, block in __import__("routerguard.env", fromlist=["env"]).find_moe_blocks(model):
            block.gate.weight[0] *= boost
    return model


# ------------------------------------------------------------------- balancing


def test_fit_biases_reduces_gini(flavor):
    model = _skewed_model(flavor, boost=3.0)
    ids = make_ids(batch=12, seq=16, seed=3)

    before = balance.measure_model(model, ids)
    # u must stay small: the sign update overshoots once u * n_updates approaches
    # the router logit scale (see test_large_u_overshoots).
    biases, info = balance.fit_biases(model, ids, u=0.005, iters=2, batch_size=4, progress=False)
    after = balance.measure_model(model, ids, biases)

    assert len(biases) == 2
    for layer, bias in biases.items():
        assert bias.dtype == torch.float32
        assert bias.numel() == model.config.num_experts
    assert any(float(b.abs().sum()) > 0 for b in biases.values()), "biases never moved"
    assert info["updates"] == 2 * 3, info
    assert info["tokens_seen"] == ids.numel() * 2, info  # one pass per epoch
    assert after["gini_mean"] < before["gini_mean"], (
        f"balancing did not reduce Gini: {before['gini_mean']:.4f} -> {after['gini_mean']:.4f}"
    )
    for layer, counts in after["per_layer_counts"].items():
        assert sum(counts) == sum(before["per_layer_counts"][layer])


def test_fit_biases_rejects_non_positive_u(flavor):
    model = build_model(flavor)
    with pytest.raises(ValueError, match="u must be > 0"):
        balance.fit_biases(model, make_ids(), u=0.0)


def test_zero_bias_measurement_matches_baseline(flavor):
    model = _skewed_model(flavor)
    ids = make_ids()
    plain = balance.measure_model(model, ids)
    zeroed = balance.measure_model(model, ids, balance.zero_biases(model))
    assert plain["per_layer_counts"] == zeroed["per_layer_counts"]


def test_large_u_overshoots(flavor):
    """Documents the sign-update instability boundary: too-large u makes Gini worse.

    This is why protocol.yaml fixes u at 0.001 and why evaluate_balance() warns
    instead of trusting the direction of the update.
    """
    model = _skewed_model(flavor, boost=3.0)
    ids = make_ids(batch=12, seq=16, seed=3)
    before = balance.measure_model(model, ids)["gini_mean"]
    biases, _ = balance.fit_biases(model, ids, u=0.05, iters=2, batch_size=4, progress=False)
    after = balance.measure_model(model, ids, biases)["gini_mean"]
    assert after > before, (
        f"expected overshoot at u=0.05, but Gini improved ({before:.4f} -> {after:.4f}); "
        "if the update was changed, re-check the u regime and update protocol.yaml"
    )


def test_evaluate_balance_warns_on_overshoot(flavor):
    model = _skewed_model(flavor, boost=3.0)
    calib = make_ids(batch=12, seq=16, seed=3)
    eval_ids = make_ids(batch=8, seq=16, seed=4)
    result = balance.evaluate_balance(
        model, calib, eval_ids, {"balance": {"default_u": 0.05, "iters": 2}},
        batch_size=4, device=None, progress=False,
    )
    assert not result.improved_calib
    assert any("too large" in w for w in result.warnings()), result.warnings()
    assert result.summary()["warnings"]


def test_evaluate_balance_reports_guards(flavor):
    model = _skewed_model(flavor, boost=3.0)
    calib = make_ids(batch=12, seq=16, seed=3)
    eval_ids = make_ids(batch=8, seq=16, seed=4)
    protocol = {
        "balance": {"default_u": 0.005, "iters": 2},
        "gates": {"max_kl_nats": 0.05, "min_balance_transfer_ratio": 0.5},
    }
    result = balance.evaluate_balance(
        model, calib, eval_ids, protocol, batch_size=4, device=None, progress=False
    )
    assert result.u == 0.005 and result.iters == 2
    assert result.calib_after["gini_mean"] < result.calib_before["gini_mean"]
    assert result.gini_gain_calib > 0
    assert result.max_abs_bias > 0
    assert set(result.kl_per_layer()) == set(result.eval_before["per_layer_counts"])
    assert result.kl_mean >= 0
    gates = balance.BalanceResult.gate_inputs(result)
    assert set(gates) == {"max_kl_nats", "min_balance_transfer_ratio"}
    assert isinstance(result.transfer_ratio, float)
    summary = result.summary()
    for key in ("gini_calib", "gini_eval", "transfer_ratio", "kl_per_layer", "fit"):
        assert key in summary
    # biases round-trip through disk
    path = result.save("/tmp/rg_test_biases.pt")
    loaded = balance.BalanceResult.load(path)
    assert set(loaded["biases"]) == set(result.biases)
    assert torch.equal(loaded["biases"][0], result.biases[0])


def test_apply_bias_context_changes_selection(flavor):
    model = _skewed_model(flavor)
    ids = make_ids()
    biases = {layer: torch.zeros(model.config.num_experts) for layer in range(2)}
    biases[0][5] = 5.0
    with torch.no_grad(), balance.apply_bias(model, biases, device="cpu"):
        out = model(input_ids=ids, use_cache=False).logits
    assert torch.isfinite(out).all()
    with torch.no_grad():
        after = model(input_ids=ids, use_cache=False).logits
    assert not torch.equal(out, after), "bias did not survive into the forward path"


# ------------------------------------------------------------------ perplexity


class _FlatModel(torch.nn.Module):
    """Logits all zero -> perplexity is exactly the vocabulary size."""

    def __init__(self, vocab=50, seq=8):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab)
        self.seq = seq

    def forward(self, input_ids, use_cache=False, **kwargs):
        batch, seq = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, seq, self.config.vocab_size))


def test_perplexity_of_uniform_model_equals_vocab_size():
    model = _FlatModel(vocab=50)
    ids = torch.randint(0, 50, (3, 8))
    result = rgeval.wikitext_perplexity(model, ids, batch_size=1, device=None)
    assert result["perplexity"] == pytest.approx(50.0, rel=1e-5)
    assert result["tokens"] == 3 * 7  # last token of each window has no next-token target


def test_perplexity_is_token_weighted_across_batches(flavor):
    model = build_model(flavor)
    ids = make_ids(batch=6, seq=12, seed=9)
    one = rgeval.wikitext_perplexity(model, ids, batch_size=1, device=None)
    all_at_once = rgeval.wikitext_perplexity(model, ids, batch_size=6, device=None)
    assert one["perplexity"] == pytest.approx(all_at_once["perplexity"], rel=1e-6)
    assert one["tokens"] == all_at_once["tokens"] == 6 * 11


# ----------------------------------------------------------------------- MMLU


class _CharTokenizer:
    """Every character is one token; enough to exercise the scoring path."""

    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


class _LetterStub(torch.nn.Module):
    """Predicts the letter encoded by the first character of the prompt."""

    def __init__(self, letter_ids, invert=False):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=256)
        self.letter_ids = list(letter_ids)
        self.invert = invert

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 256)
        for row in range(batch):
            if attention_mask is not None:
                first = int(input_ids[row][attention_mask[row].bool()][0])
            else:
                first = int(input_ids[row][0])
            slot = self.letter_ids.index(first) if first in self.letter_ids else 0
            if self.invert:
                slot = (slot + 1) % 4
            logits[row, -1, self.letter_ids[slot]] = 10.0
        return SimpleNamespace(logits=logits)


def _letter_prompts():
    # first character encodes the answer letter; lengths vary to exercise left padding
    letters = "ABCD"
    prompts = []
    for i, letter in enumerate(letters):
        prompts.append(
            {
                "prompt": f"{letter} what is {'x' * (i + 1)}",
                "answer_idx": i,
                "subject": "stub",
                "n_shot": 3,
            }
        )
    return prompts


def test_resolve_letter_ids():
    tokenizer = _CharTokenizer()
    ids = rgeval.resolve_letter_ids(tokenizer)
    assert ids == [ord(c) % 256 for c in "ABCD"]


def test_mmlu_scoring_perfect_and_worst():
    tokenizer = _CharTokenizer()
    letter_ids = rgeval.resolve_letter_ids(tokenizer)
    prompts = _letter_prompts()

    good = rgeval.mmlu_accuracy(_LetterStub(letter_ids), tokenizer, prompts, batch_size=4, device=None)
    assert good["accuracy"] == 1.0, good
    assert good["n_questions"] == 4
    assert good["per_subject"]["stub"]["accuracy"] == 1.0

    bad = rgeval.mmlu_accuracy(_LetterStub(letter_ids, invert=True), tokenizer, prompts, batch_size=4, device=None)
    assert bad["accuracy"] == 0.0, bad


def test_mmlu_left_padding_keeps_predictions_aligned():
    tokenizer = _CharTokenizer()
    letter_ids = rgeval.resolve_letter_ids(tokenizer)
    prompts = _letter_prompts()
    stub = _LetterStub(letter_ids)
    batched = rgeval.mmlu_accuracy(stub, tokenizer, prompts, batch_size=4, device=None)
    single = rgeval.mmlu_accuracy(stub, tokenizer, prompts, batch_size=1, device=None)
    assert batched["accuracy"] == single["accuracy"] == 1.0


def test_mmlu_rejects_empty_prompts():
    with pytest.raises(ValueError, match="no MMLU prompts"):
        rgeval.mmlu_accuracy(_LetterStub([1, 2, 3, 4]), _CharTokenizer(), [], device=None)


# ---------------------------------------------------------- memory / throughput


def test_param_report_and_memory_snapshot(flavor):
    model = build_model(flavor)
    report = rgeval.param_report(model, "8bit")
    assert report["params_total"] == sum(p.numel() for p in model.parameters())
    assert report["bytes_per_param"] == 1.0
    assert report["estimated_gb"] == pytest.approx(round(report["params_total"] / 1024**3, 3), abs=1e-9)
    snapshot = rgeval.memory_snapshot()
    assert "available" in snapshot


def test_throughput_reports_variance(flavor):
    model = build_model(flavor)
    result = rgeval.throughput(model, seq_len=8, warmup=1, runs=3, device="cpu")
    assert len(result["seconds_per_run"]) == 3
    assert result["tokens_per_sec_mean"] > 0
    assert result["tokens_per_sec_min"] <= result["tokens_per_sec_mean"] <= result["tokens_per_sec_max"]
    assert result["cv"] >= 0


def test_evaluate_all_bundle(flavor):
    model = build_model(flavor)
    tokenizer = _CharTokenizer()
    bundle = rgeval.evaluate_all(
        model,
        tokenizer,
        eval_ids=make_ids(batch=2, seq=8),
        prompts=_letter_prompts(),
        protocol={"metrics": {"throughput": {"seq_len": 8, "warmup": 1, "runs": 2}}},
        precision="8bit",
        device=None,
        do_throughput=True,
    )
    assert set(bundle) == {"params", "perplexity", "mmlu", "throughput", "memory"}
    assert bundle["perplexity"]["tokens"] == 2 * 7
    assert "predictions" not in bundle["mmlu"], "predictions must not bloat the metrics file"


# ------------------------------------------------------------------ baselines


def test_num_keep_matches_protocol_ratios():
    assert prune.num_keep(60, 0.25) == 45
    assert prune.num_keep(60, 0.50) == 30
    assert prune.num_keep(64, 0.25) == 48
    assert prune.num_keep(64, 0.50) == 32
    assert prune.num_keep(2, 0.9) == 1  # never prunes everything
    with pytest.raises(ValueError):
        prune.num_keep(8, 1.0)


def test_frequency_keeps_the_most_used_experts():
    counts = {0: [100, 1, 50, 2], 1: [0, 0, 0, 9]}
    keeps = prune.frequency_keeps(counts, ratio=0.5)
    assert keeps[0] == [0, 2]
    # ties break toward the lower expert index (stable argsort), deterministically
    assert keeps[1] == [0, 3]


def test_random_keeps_are_seeded(flavor):
    model = build_model(flavor)
    experts = prune.experts_per_layer(model)
    a = prune.random_keeps(experts, 0.5, seed=42)
    b = prune.random_keeps(experts, 0.5, seed=42)
    c = prune.random_keeps(experts, 0.5, seed=123)
    assert a == b, "same seed must give the same keep set"
    assert a != c, "different seeds should differ"
    for layer, keep in a.items():
        assert len(keep) == 4
        assert keep == sorted(keep)


def test_keeps_for_dispatch(flavor):
    model = build_model(flavor)
    counts = {0: list(range(8)), 1: list(range(8, 16))}
    assert prune.keeps_for(model, "frequency", 0.5, per_layer_counts=counts)
    assert prune.keeps_for(model, "random", 0.5, seed=1)
    with pytest.raises(ValueError, match="needs per_layer_counts"):
        prune.keeps_for(model, "frequency", 0.5)
    with pytest.raises(ValueError, match="needs similarity matrices"):
        prune.keeps_for(model, "redundancy", 0.5)
    with pytest.raises(ValueError, match="unknown pruning kind"):
        prune.keeps_for(model, "magic", 0.5)


def test_prune_by_kind_end_to_end(flavor):
    model = build_model(flavor)
    counts = {layer: [10] * 8 for layer in (0, 1)}
    keeps, report = prune.prune_by_kind(model, "frequency", 0.5, per_layer_counts=counts)
    assert report["num_experts"] == [4]
    assert model.config.num_experts == 4
    for layer, keep in keeps.items():
        assert len(keep) == 4
    with torch.no_grad():
        out = model(input_ids=make_ids(), use_cache=False).logits
    assert torch.isfinite(out).all()
    assert moe.equivalence_check(out, out)["bitwise_equal"]


def test_removal_counts_preview(flavor):
    model = build_model(flavor)
    counts = prune.removal_counts(prune.experts_per_layer(model), 0.25)
    assert counts == {0: (8, 6), 1: (8, 6)}
