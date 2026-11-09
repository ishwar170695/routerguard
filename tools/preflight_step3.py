"""Execute notebook 02's cell sequence against a tiny model on CPU.

This is the check that matters: it runs the same calls in the same order as
notebooks/02_olmoe_balance_and_baselines.ipynb and asserts the keys the notebook
reads actually exist and survive JSON serialization.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path.home() / "routerguard"))

from routerguard import balance, data, env, eval as rgeval, moe, models, prune, run as rgrun, stats  # noqa: E402
from tests.conftest import build_model, make_ids  # noqa: E402

FAILS = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"  FAIL  {name} — {type(exc).__name__}: {exc}")
        FAILS.append(name)


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    name_or_path = "char-stub"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


def letter_prompts(n=8):
    prompts = []
    for i in range(n):
        letter = "ABCD"[i % 4]
        prompts.append(
            {
                "index": i,
                "subject": f"sub{i % 3}",
                "prompt": f"{letter} question number {i} " + "x" * (i % 5),
                "choices": ["a", "b", "c", "d"],
                "answer_idx": i % 4,
                "answer_letter": letter,
                "n_shot": 3,
            }
        )
    return prompts


PROTOCOL = rgrun.load_protocol(Path.home() / "routerguard" / "configs" / "protocol.yaml")
RUN = rgrun.new_run("selfcheck_step3", protocol=PROTOCOL, results_root="/tmp/rg_sc3/results")
tokenizer = CharTokenizer()
prompts = letter_prompts()

print(f"pre-flight step 3 (run {RUN.run_id})")

calib_ids = make_ids(batch=12, seq=16, seed=3)
eval_ids = make_ids(batch=6, seq=16, seed=4)
state = {}


def cell_03_model():
    model = build_model("olmoe", seed=0)
    state["model"] = model
    structure = env.assert_moe_structure(model)
    assert structure["num_experts_per_layer"] == 8
    return f"{structure['num_moe_layers']} layers x {structure['num_experts_per_layer']} experts"


def cell_04_baseline():
    model = state["model"]
    baseline = rgeval.evaluate_all(
        model, tokenizer, eval_ids=eval_ids, prompts=prompts,
        protocol={"metrics": {"throughput": {"seq_len": 16, "warmup": 1, "runs": 2}}},
        precision="8bit", device=None, batch_size=1, mmlu_batch_size=4, do_throughput=True,
    )
    for key in ("params", "perplexity", "mmlu", "throughput", "memory"):
        assert key in baseline, key
    assert "predictions" not in baseline["mmlu"]
    RUN.save_json("metrics_baseline", baseline)
    state["baseline"] = baseline
    calib_stats = balance.measure_model(model, calib_ids, None, batch_size=4, device=None)
    assert "per_layer_counts" in calib_stats
    state["calib_stats"] = calib_stats
    return (
        f"ppl={baseline['perplexity']['perplexity']:.2f} "
        f"mmlu={baseline['mmlu']['accuracy']:.2f} gini={calib_stats['gini_mean']:.3f}"
    )


def cell_05_balance():
    model = state["model"]
    result = balance.evaluate_balance(
        model, calib_ids, eval_ids, PROTOCOL, u=0.01, iters=2, batch_size=4, device=None, progress=False
    )
    summary = result.summary()
    summary["gates"] = rgrun.summarize_gates(PROTOCOL, result.gate_inputs())
    for key in (
        "gini_calib", "gini_eval", "transfer_ratio", "kl_mean_nats", "kl_per_layer",
        "improved", "max_abs_bias", "warnings", "fit", "gates",
    ):
        assert key in summary, key
    RUN.save_json("balance", summary)
    result.save(RUN.dir / "balance_biases.pt")
    reloaded = balance.BalanceResult.load(RUN.dir / "balance_biases.pt")
    assert set(reloaded["biases"]) == set(result.biases)
    printed = rgrun.format_gates(summary["gates"])
    assert "gini_skip_balance" in printed and "not measured" in printed, printed
    state["balance"] = result
    return f"gini {result.calib_before['gini_mean']:.3f}->{result.calib_after['gini_mean']:.3f}, transfer {result.transfer_ratio:.3f}"


def cell_06_ppl_cost():
    model = state["model"]
    ppl_plain = rgeval.wikitext_perplexity(model, eval_ids, batch_size=1, device=None)
    with torch.no_grad(), balance.apply_bias(model, state["balance"].biases, device="cpu"):
        ppl_biased = rgeval.wikitext_perplexity(model, eval_ids, batch_size=1, device=None)
    payload = {
        "without_bias": ppl_plain,
        "with_bias": ppl_biased,
        "delta": ppl_biased["perplexity"] - ppl_plain["perplexity"],
    }
    RUN.save_json("balance_ppl_cost", payload)
    with torch.no_grad():
        after = rgeval.wikitext_perplexity(model, eval_ids, batch_size=1, device=None)
    assert after["perplexity"] == pytest.approx(ppl_plain["perplexity"], rel=1e-9), "bias leaked out of context"
    return f"delta={payload['delta']:+.4f}"


def cell_07_plots_data():
    result = state["balance"]
    for split_name in ("calib_before", "calib_after", "eval_before", "eval_after"):
        report = getattr(result, split_name)
        ginis = report["per_layer_gini"]
        assert isinstance(ginis, dict) and ginis
        keys = list(ginis)[:1]
        assert isinstance(keys[0], int), f"{split_name} per_layer_gini keys are {type(keys[0])}"
        # notebook 07 indexes with both int and str; verify the fallback resolves
        for layer in sorted(ginis):
            value = ginis[str(layer)] if str(layer) in ginis else ginis[layer]
            assert value >= 0
        counts = report["per_layer_counts"]
        assert sum(counts[sorted(counts)[0]]) > 0
    return "4 splits x per-layer gini/counts readable"


def cell_08_curve():
    model = state["model"]
    calib_counts = state["calib_stats"]["per_layer_counts"]
    curve = []
    for kind in PROTOCOL["design"]["baselines"]:
        for ratio in [float(r) for r in PROTOCOL["pruning"]["ratios"]]:
            keeps = prune.keeps_for(model, kind, ratio, per_layer_counts=calib_counts, seed=42)
            with torch.no_grad(), moe.MaskSim(model, keeps):
                ppl = rgeval.wikitext_perplexity(model, eval_ids, batch_size=1, device=None)
                point = {
                    "kind": kind, "ratio": ratio, "seed": 42,
                    "kept_per_layer": [len(v) for v in keeps.values()][:1],
                    "perplexity": ppl["perplexity"], "tokens": ppl["tokens"],
                }
            curve.append(point)
    assert len(curve) == len(PROTOCOL["design"]["baselines"]) * len(PROTOCOL["pruning"]["ratios"])
    assert all(p["kept_per_layer"][0] in (6, 4) for p in curve), curve
    RUN.save_json("curve_baselines", {"baseline_ppl": state["baseline"]["perplexity"]["perplexity"], "points": curve})
    with torch.no_grad():
        restored = rgeval.wikitext_perplexity(model, eval_ids, batch_size=1, device=None)
    assert restored["perplexity"] == pytest.approx(state["baseline"]["perplexity"]["perplexity"], rel=1e-9), "MaskSim leaked"
    return " ".join(f"{p['kind']}/{p['ratio']:.2f}:{p['perplexity']:.2f}" for p in curve)


def cell_09_memory():
    model = state["model"]
    calib_counts = state["calib_stats"]["per_layer_counts"]
    before = rgeval.param_report(model, "8bit")
    keeps = prune.keeps_for(model, "frequency", 0.50, per_layer_counts=calib_counts, seed=42)
    removal = prune.prune_with_keeps(model, keeps)
    after = rgeval.param_report(model, "8bit")
    payload = {
        "removal": {k: v for k, v in removal.items() if k != "layers"},
        "params_before": before, "params_after": after,
        "measured_before": rgeval.memory_snapshot(), "measured_after": rgeval.memory_snapshot(),
    }
    RUN.save_json("memory_saved_50pct", payload)
    assert after["params_total"] < before["params_total"]
    assert removal["num_experts"] == [4]
    assert payload["removal"]["removed_total"] == 8
    with torch.no_grad():
        ppl = rgeval.wikitext_perplexity(model, eval_ids, batch_size=1, device=None)
    assert torch.isfinite(torch.tensor(ppl["perplexity"]))
    return f"experts {before['params_total']}->{after['params_total']} params, ppl {ppl['perplexity']:.2f}"


check("cell 03 model load + structure", cell_03_model)
check("cell 04 baseline metrics", cell_04_baseline)
check("cell 05 balance + gates", cell_05_balance)
check("cell 06 ppl cost of bias", cell_06_ppl_cost)
check("cell 07 plot inputs", cell_07_plots_data)
check("cell 08 compression curve", cell_08_curve)
check("cell 09 real removal + memory", cell_09_memory)


def run_artifacts():
    expected = ["config.resolved.yaml", "git_sha.txt", "env.txt", "balance_biases.pt"]
    for name in expected:
        assert (RUN.dir / name).exists(), f"missing {name}"
    metrics = sorted(p.name for p in RUN.metrics_dir.glob("*.json"))
    assert "metrics_baseline.json" in metrics, metrics
    assert "balance.json" in metrics, metrics
    for path in RUN.metrics_dir.glob("*.json"):
        json.loads(path.read_text())
    return f"{len(metrics)} metrics json: {', '.join(metrics)}"


check("run artifacts", run_artifacts)

print()
if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("all step-3 pre-flight checks passed")
