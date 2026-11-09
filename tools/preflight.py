"""Local pre-flight for Step 1 artifacts (no torch/GPU needed).

Checks: every module compiles, the protocol parses and has the keys the code reads,
the notebook is valid JSON and every code cell compiles, and the file tree is complete.
"""

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "routerguard")
failures = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name} — {type(exc).__name__}: {exc}")
        failures.append(name)


def modules():
    files = sorted((ROOT / "routerguard").glob("*.py"))
    assert files, "no modules found"
    for path in files:
        py_compile.compile(str(path), doraise=True)
    return f"{len(files)} modules"


PROTOCOL_KEYS = {
    "models": ["dev", "target"],
    "calibration": ["n_seqs", "seq_len", "dataset", "config", "split", "seed"],
    "eval": ["n_seqs", "seq_len", "dataset", "config", "split", "seed"],
    "pruning": ["ratios", "seeds", "removal_semantics", "keep_shared_expert"],
    "metrics": ["primary", "perplexity", "mmlu", "throughput"],
    "gates": ["gini_skip_balance", "max_kl_nats", "min_gain_over_frequency_pts", "min_balance_transfer_ratio"],
    "balance": ["method", "default_u", "iters", "selection_only"],
    "similarity": ["metric", "dtype", "save_dir"],
    "budget": ["max_tokens_per_layer", "max_layers", "max_greedy_steps"],
    "design": ["ablation", "baselines", "fallback"],
}


def protocol():
    import yaml

    data = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    missing = {k: [s for s in subs if s not in data.get(k, {})] for k, subs in PROTOCOL_KEYS.items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"missing protocol keys: {missing}"
    assert data["pruning"]["removal_semantics"] == "reslice", "removal semantics changed"
    assert data["calibration"]["split"] != data["eval"]["split"], "calib/eval must not share a split"
    assert data["pruning"]["keep_shared_expert"] is True
    return f"{len(data)} top-level keys, gates={list(data['gates'])}"


def notebook():
    path = ROOT / "notebooks" / "01_setup_and_load_stats.ipynb"
    nb = json.loads(path.read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "no code cells"
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        # Strip IPython magics before compiling.
        src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("%", "!")))
        try:
            compile(src, f"cell{i}", "exec")
        except SyntaxError as exc:
            raise SyntaxError(f"code cell {i}: {exc}") from exc
    return f"{len(nb['cells'])} cells, {len(code_cells)} code cells compile"


INSTALL_CELL_GUARDS = {
    "notebooks/01_setup_and_load_stats.ipynb": ("huggingface-hub>=0.24,<1.0", "datasets>=2.19,<5"),
    "notebooks/02_olmoe_balance_and_baselines.ipynb": ("huggingface-hub>=0.24,<1.0", "datasets>=2.19,<5"),
}


def install_pins():
    """A bare `pip install datasets` silently breaks the transformers pin; forbid it."""
    for rel, required in INSTALL_CELL_GUARDS.items():
        path = ROOT / rel
        assert path.exists(), f"missing notebook {rel}"
        text = path.read_text()
        for needle in required:
            assert needle in text, f"{rel} install cell is missing the bound {needle!r}"
        assert '"datasets"' not in text, (
            f"{rel} installs unbounded 'datasets', which upgrades huggingface-hub past 1.0 "
            "and makes transformers 4.48.3 unimportable"
        )
    return f"{len(INSTALL_CELL_GUARDS)} notebooks carry hub/datasets bounds"


def tree():
    expected = [
        "configs/protocol.yaml",
        "routerguard/__init__.py",
        "routerguard/env.py",
        "routerguard/models.py",
        "routerguard/data.py",
        "routerguard/stats.py",
        "routerguard/run.py",
        "notebooks/01_setup_and_load_stats.ipynb",
        # Step 2
        "routerguard/moe.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_equivalence.py",
        "tests/test_smoke.py",
        # Step 3
        "routerguard/balance.py",
        "routerguard/prune.py",
        "routerguard/eval.py",
        "notebooks/02_olmoe_balance_and_baselines.ipynb",
        "tests/test_step3.py",
        "tools/preflight_step3.py",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    assert not missing, f"missing: {missing}"
    return f"{len(expected)} project files present"


print(f"pre-flight: {ROOT}")
check("modules compile", modules)
check("protocol.yaml", protocol)
check("notebook 01", notebook)
check("install pins", install_pins)
check("file tree", tree)
print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("all Step-1 pre-flight checks passed")
