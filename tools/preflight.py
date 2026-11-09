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
        # Step 2 (added once the equivalence gate is in place)
        "routerguard/moe.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_equivalence.py",
        "tests/test_smoke.py",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    assert not missing, f"missing: {missing}"
    return f"{len(expected)} Step-1 files present"


print(f"pre-flight: {ROOT}")
check("modules compile", modules)
check("protocol.yaml", protocol)
check("notebook 01", notebook)
check("file tree", tree)
print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("all Step-1 pre-flight checks passed")
