"""Execute notebook 03's cell sequence against a tiny model on CPU.

Checks the calls, the returned keys, the MaskSim+bias combination, the cache behaviour and
JSON serializability — the things that otherwise fail 20 minutes into a Colab session.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path.home() / "routerguard"))

from routerguard import balance, env, eval as rgeval, moe, prune, run as rgrun, similarity  # noqa: E402
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


PROTOCOL = rgrun.load_protocol(Path.home() / "routerguard" / "configs" / "protocol.yaml")
RUN = rgrun.new_run("selfcheck_step4", protocol=PROTOCOL, results_root="/tmp/rg_sc4/results")
calib_ids = make_ids(batch=8, seq=16, seed=3)
eval_ids = make_ids(batch=4, seq=16, seed=4)
state = {}

print(f"pre-flight step 4 (run {RUN.run_id})")


def cell_03():
    model = build_model("olmoe", seed=0)
    state["model"] = model
    baseline = rgeval.wikitext_perplexity(model, eval_ids, batch_size=1, device=None)
    RUN.save_json("baseline_ppl", baseline)
    counts = balance.measure_model(model, calib_ids, None, batch_size=4, device=None)
    assert "per_layer_counts" in counts
    state["baseline_ppl"] = baseline
    state["unbalanced_counts"] = counts
    return f"baseline ppl={baseline['perplexity']:.2f} gini={counts['gini_mean']:.3f}"


def cell_04():
    result = balance.evaluate_balance(
        state["model"], calib_ids, eval_ids, PROTOCOL, u=0.01, iters=1,
        batch_size=4, device=None, progress=False,
    )
    summary = result.summary()
    RUN.save_json("balance", summary)
    result.save(RUN.dir / "balance_biases.pt")
    state["biases"] = result.biases
    state["balanced_counts"] = result.calib_after
    reloaded = balance.BalanceResult.load(RUN.dir / "balance_biases.pt")
    assert set(reloaded["biases"]) == set(result.biases)
    return f"gini {result.calib_before['gini_mean']:.3f}->{result.calib_after['gini_mean']:.3f}"


def cell_05_probe():
    import time

    t0 = time.time()
    with similarity.SimilarityCollector(model=state["model"], max_tokens_per_layer=100) as probe:
        with torch.no_grad():
            state["model"](input_ids=calib_ids[:2], use_cache=False)
        matrices = probe.matrices()
    elapsed = time.time() - t0
    assert sorted(matrices) == [0, 1]
    payload = {"probe_s": elapsed, "projected_s": elapsed * 4, "cap_s": PROTOCOL["budget"]["abort_if_projected_over_seconds"]}
    RUN.save_json("similarity_timing_probe", payload)
    assert payload["projected_s"] < payload["cap_s"], payload
    return f"probe {elapsed:.2f}s"


def cell_06_similarity():
    report = similarity.build_similarity(
        state["model"], calib_ids, RUN.dir / "sim_unbalanced",
        max_tokens_per_layer=int(PROTOCOL["budget"]["max_tokens_per_layer"]),
        batch_size=4, device=None,
    )
    assert sorted(report.matrices) == [0, 1]
    assert report.computed == [0, 1]
    RUN.save_json("similarity_unbalanced", report.to_dict())
    for layer in report.matrices:
        top = similarity.redundancy_pairs(report.matrices[layer], top=3)
        assert len(top) == 3
    state["sim_unbalanced"] = report.matrices
    return f"{report.elapsed_s:.2f}s, tokens {sorted(set(report.tokens_per_layer.values()))}"


def cell_07_similarity_under_bias():
    """Both branches: a strong bias must change the matrices, a small one must skip."""
    model = state["model"]
    n = model.config.num_experts

    # (a) plumbing check: a deliberately large bias must move the layer inputs
    strong = {layer: torch.zeros(n) for layer in range(2)}
    strong[0][0] = 6.0
    with torch.no_grad():
        without = similarity.build_similarity(
            model, calib_ids, RUN.dir / "sim_no_bias", max_tokens_per_layer=10_000,
            batch_size=4, device=None, progress=False,
        ).matrices
        with balance.apply_bias(model, strong, device="cpu"):
            with_bias = similarity.build_similarity(
                model, calib_ids, RUN.dir / "sim_with_bias", max_tokens_per_layer=10_000,
                batch_size=4, device=None, progress=False,
            ).matrices
    drift = {int(layer): float(abs(with_bias[layer] - without[layer]).mean()) for layer in without}
    # Layer 0's inputs are the embeddings, so no router bias can change them. Deeper
    # layers shift because their inputs depend on earlier routing decisions.
    assert drift[0] == 0.0, f"layer 0 similarity must be router-invariant, got {drift[0]}"
    assert max(drift[layer] for layer in drift if layer > 0) > 0, (
        "a saturated bias did not change any deeper layer's inputs; the collector ignores the router"
    )

    # (b) the notebook's decision rule, evaluated on the real fitted bias
    per_layer = {
        layer: stats_kl(state["unbalanced_counts"]["per_layer_counts"][layer],
                        state["balanced_counts"]["per_layer_counts"][layer])
        for layer in state["unbalanced_counts"]["per_layer_counts"]
    }
    mean_kl = sum(per_layer.values()) / len(per_layer)
    state["sim_balanced"] = state["sim_unbalanced"]
    if mean_kl >= 1e-2:
        state["sim_balanced"] = with_bias
    RUN.save_json("similarity_balanced", {"mean_kl": mean_kl, "recomputed": mean_kl >= 1e-2})
    return (
        f"drift layer0={drift[0]:.5f} (invariant), layer1={drift[1]:.5f}, "
        f"mean_kl(fitted)={mean_kl:.5f} -> recompute={mean_kl >= 1e-2}"
    )


def stats_kl(a, b):
    from routerguard import stats

    return stats.kl(a, b)


def cell_08_ablation():
    model = state["model"]
    rows, cells = {}, []
    weight = float(PROTOCOL["design"].get("utilization_weight", 0.25))
    for router_state in ("unbalanced", "balanced"):
        bias = None if router_state == "unbalanced" else state["biases"]
        counts = state["unbalanced_counts"] if router_state == "unbalanced" else state["balanced_counts"]
        sim = state["sim_unbalanced"] if router_state == "unbalanced" else state["sim_balanced"]
        for ranking in ("frequency", "redundancy"):
            for ratio in [float(r) for r in PROTOCOL["pruning"]["ratios"]]:
                keeps = prune.keeps_for(
                    model, ranking, ratio, per_layer_counts=counts["per_layer_counts"],
                    sim=sim, utilization_weight=weight,
                    max_steps=int(PROTOCOL["budget"]["max_greedy_steps"]),
                )
                with torch.no_grad(), moe.MaskSim(model, keeps, bias=bias):
                    ppl = rgeval.wikitext_perplexity(model, eval_ids, batch_size=1, device=None)
                point = {
                    "router": router_state, "ranking": ranking, "ratio": ratio,
                    "perplexity": ppl["perplexity"], "kept_per_layer": len(keeps[sorted(keeps)[0]]),
                    "delta_vs_baseline": ppl["perplexity"] - state["baseline_ppl"]["perplexity"],
                }
                cells.append(point)
                rows[(router_state, ranking, ratio)] = ppl["perplexity"]
    assert len(cells) == 8, len(cells)
    assert all(c["kept_per_layer"] == 6 for c in cells if c["ratio"] == 0.25), cells
    RUN.save_json("ablation_2x2", {"baseline_ppl": state["baseline_ppl"]["perplexity"], "cells": cells})
    state["rows"] = rows
    return "8 cells"


def cell_09_gates():
    rows = state["rows"]
    ratios = [float(r) for r in PROTOCOL["pruning"]["ratios"]]
    gains = {
        f"{router}@{ratio:.2f}": rows[(router, "frequency", ratio)] - rows[(router, "redundancy", ratio)]
        for router in ("unbalanced", "balanced")
        for ratio in ratios
    }
    key = f"balanced@{max(ratios):.2f}"
    gates = rgrun.summarize_gates(PROTOCOL, {"min_gain_over_frequency_pts": gains[key]})
    assert "min_gain_over_frequency_pts" in gates
    assert gates["min_gain_over_frequency_pts"]["status"] in ("pass", "fail")
    printed = rgrun.format_gates(gates)
    assert "min_gain_over_frequency_pts" in printed
    RUN.save_json("step4_gates", {"gains": gains, "gates": gates, "headline_cell": key})
    return f"gain(balanced@50%)={gains[key]:+.4f} -> {gates['min_gain_over_frequency_pts']['status']}"


check("cell 03 baseline + counts", cell_03)
check("cell 04 balance + biases", cell_04)
check("cell 05 timing probe", cell_05_probe)
check("cell 06 similarity (cached)", cell_06_similarity)
check("cell 07 similarity under bias", cell_07_similarity_under_bias)
check("cell 08 2x2 ablation", cell_08_ablation)
check("cell 09 gate", cell_09_gates)


def resume_check():
    report = similarity.build_similarity(
        state["model"], calib_ids, RUN.dir / "sim_unbalanced",
        batch_size=4, device=None, progress=False,
    )
    assert report.computed == [], report.computed
    assert sorted(report.cached) == [0, 1]
    return "second call reused cache, no forward pass"


def artifacts():
    for name in ("config.resolved.yaml", "env.txt", "balance_biases.pt"):
        assert (RUN.dir / name).exists(), name
    names = sorted(p.name for p in RUN.metrics_dir.glob("*.json"))
    for path in RUN.metrics_dir.glob("*.json"):
        json.loads(path.read_text())
    assert "ablation_2x2.json" in names and "step4_gates.json" in names, names
    return ", ".join(names)


check("similarity cache resume", resume_check)
check("run artifacts", artifacts)

print()
if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("all step-4 pre-flight checks passed")
