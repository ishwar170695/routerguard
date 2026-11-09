"""CPU self-check for Step 1 modules: real execution, fake model, no GPU, no HF download.

Exercises env.assert_moe_structure (pass + v5-fused failure), stats counting/Gini/KL,
data split determinism + cache round-trip, run bookkeeping, and gate directions.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path.home() / "routerguard"))

from routerguard import data, env, stats  # noqa: E402

FAILS = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name} — {type(exc).__name__}: {exc}")
        FAILS.append(name)


def expect_raises(name, exc_type, fn):
    try:
        fn()
    except exc_type as exc:
        print(f"  PASS  {name} — raised {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name} — wrong exception {type(exc).__name__}: {exc}")
        FAILS.append(name)
        return
    print(f"  FAIL  {name} — did not raise")
    FAILS.append(name)


# ------------------------------------------------------------------ fake MoE model
class Cfg:
    num_experts = 8
    num_experts_per_tok = 2
    num_hidden_layers = 2
    hidden_size = 16
    norm_topk_prob = False


class Expert(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc = nn.Linear(h, h, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc(self.act(x))


class Block(nn.Module):
    """Mirror of OlmoeSparseMoeBlock's math (norm_topk_prob=False)."""

    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.norm_topk_prob = cfg.norm_topk_prob
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg.hidden_size) for _ in range(cfg.num_experts)])

    def forward(self, x):
        b, s, h = x.shape
        x = x.view(-1, h)
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=1, dtype=torch.float)
        w, sel = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            w = w / w.sum(-1, keepdim=True)
        w = w.to(x.dtype)
        out = torch.zeros((b * s, h), dtype=x.dtype, device=x.device)
        mask = torch.nn.functional.one_hot(sel, num_classes=self.num_experts).permute(2, 1, 0)
        for e in range(self.num_experts):
            idx, top_x = torch.where(mask[e])
            cur = x[None, top_x].reshape(-1, h)
            out.index_add_(0, top_x, (self.experts[e](cur) * w[top_x, idx, None]).to(x.dtype))
        return out.view(b, s, h), logits


class Wrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(cfg.num_hidden_layers):
            layer = nn.Module()
            layer.mlp = Block(cfg)
            self.model.layers.append(layer)

    def forward(self, x):
        for layer in self.model.layers:
            out, _ = layer.mlp(x)
            x = x + out
        return x


class Fused(nn.Module):
    """transformers v5 style: experts fused into one 3D tensor."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 4, bias=False)
        self.experts = nn.Parameter(torch.zeros(4, 8, 16))


torch.manual_seed(0)
model = Wrapper(Cfg())

print("env / structure")
check("assert_moe_structure passes", lambda: env.assert_moe_structure(model, expect_layers=2, expect_experts=8)["num_moe_layers"])
expect_raises("fused v5 layout rejected", RuntimeError, lambda: env.assert_moe_structure(Fused()))

print("stats")
x = torch.randn(4, 5, 16)


def counting_matches():
    # Independent reference: record each gate's real input, then select with argsort
    # (not topk) so the collector's top-k + bincount path is checked against different code.
    recorded = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def keep(module, inputs, output, i=i):
            recorded.setdefault(i, []).append(inputs[0].detach().clone())
        handles.append(layer.mlp.gate.register_forward_hook(keep))

    with stats.UnbiasedLoadStats(model) as ls:
        model(x)
        got = ls.per_layer_counts()
    for h in handles:
        h.remove()

    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            logits = layer.mlp.gate(recorded[i][0].reshape(-1, 16))
            sel = logits.argsort(dim=-1, descending=True)[:, :2]
            manual = torch.bincount(sel.reshape(-1), minlength=8)
            assert torch.equal(got[i], manual), f"layer {i}: {got[i]} != {manual}"
    total = sum(int(c.sum()) for c in got.values())
    assert total == 4 * 5 * 2 * 2, total
    return f"{len(got)} layers, {total} assignments match argsort reference"


check("collector counts == manual topk", counting_matches)
check("gini uniform = 0", lambda: abs(stats.gini([5] * 8)) < 1e-12 or (_ for _ in ()).throw(AssertionError(stats.gini([5] * 8))))


def gini_onehot():
    g = stats.gini([100, 0, 0, 0, 0, 0, 0, 0])
    assert abs(g - 7 / 8) < 1e-9, g
    return f"{g:.4f} == (n-1)/n"


check("gini one-hot = (n-1)/n", gini_onehot)


def kl_props():
    p = torch.tensor([10.0, 20, 30, 40])
    assert abs(stats.kl(p, p)) < 1e-12
    big = stats.kl(torch.tensor([1.0, 0, 0, 0]), torch.tensor([1.0, 1, 1, 1]))
    assert big > 0.5, big
    return f"KL(p,p)=0, KL(onehot||uniform)={big:.3f}"


check("kl properties", kl_props)


def violation_uniform():
    assert stats.max_violation([3] * 8) < 1e-12
    v = stats.max_violation([16, 0, 0, 0])
    assert abs(v - 3.0) < 1e-9, v
    return f"uniform=0, [16,0,0,0]={v:.1f}"


check("max_violation", violation_uniform)


def aggregate_ok():
    per_layer = {0: torch.tensor([10, 10, 10, 10]), 1: torch.tensor([40, 0, 0, 0])}
    agg = stats.aggregate(per_layer)
    assert agg["n_layers"] == 2
    assert agg["gini_worst_layer"] == 1
    assert abs(agg["pooled"]["total_assignments"] - 80) < 1e-9
    return f"worst={agg['gini_worst_layer']}, gini_mean={agg['gini_mean']:.3f}"


check("aggregate", aggregate_ok)

print("data")


class FakeTok:
    eos_token_id = 1
    name_or_path = "fake-tokenizer"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + (ord(c) % 50) for c in text]}


fake_rows = [{"text": "abcdefghij " * 40} for _ in range(200)]
data._load_wikitext_rows = lambda *a, **k: fake_rows


def split_determinism():
    a = data.build_wikitext_split(FakeTok(), "calib", 4, 16, seed=42)
    b = data.build_wikitext_split(FakeTok(), "calib", 4, 16, seed=42)
    c = data.build_wikitext_split(FakeTok(), "calib", 4, 16, seed=7)
    assert torch.equal(a.input_ids, b.input_ids), "same seed must give same split"
    assert a.input_ids.shape == (4, 16)
    assert a.meta["token_offset"] != c.meta["token_offset"] or not torch.equal(a.input_ids, c.input_ids)
    return f"shape={tuple(a.input_ids.shape)}, offset={a.meta['token_offset']}"


check("split determinism", split_determinism)


def split_cache():
    path = Path("/tmp/rg_selfcheck/split.pt")
    first = data.build_wikitext_split(FakeTok(), "calib", 2, 8, seed=42, cache_path=path)
    second = data.build_wikitext_split(FakeTok(), "calib", 2, 8, seed=42, cache_path=path)
    assert path.exists()
    assert torch.equal(first.input_ids, second.input_ids)
    assert second.meta["token_offset"] == first.meta["token_offset"]
    return f"cached {path.stat().st_size} bytes"


check("split cache round-trip", split_cache)

print("run bookkeeping")


def run_bookkeeping():
    from routerguard import run as rgrun

    protocol = rgrun.load_protocol(Path.home() / "routerguard" / "configs" / "protocol.yaml")
    run = rgrun.new_run("selfcheck", protocol=protocol, results_root="/tmp/rg_selfcheck/results")
    for name in ("config.resolved.yaml", "git_sha.txt", "env.txt", "cmdline.txt"):
        assert (run.dir / name).exists(), name
    run.save_json("probe", {"gini": torch.tensor([1.0, 2.0]), "n": 3})
    assert (run.metrics_dir / "probe.json").exists()
    gates = rgrun.summarize_gates(
        protocol,
        {
            "gini_skip_balance": 0.10,
            "min_gain_over_frequency_pts": 0.5,
            "min_balance_transfer_ratio": 0.9,
        },
    )
    assert gates["gini_skip_balance"]["status"] == "pass"
    assert gates["min_gain_over_frequency_pts"]["status"] == "fail"
    assert gates["min_balance_transfer_ratio"]["status"] == "pass"
    assert gates["max_kl_nats"]["status"] == "not_measured"
    return f"{run.run_id}, 4 artifacts + gates ok"


check("new_run + gates", run_bookkeeping)

print()
if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("all Step-1 functional self-checks passed")
