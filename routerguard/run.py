"""Run bookkeeping: run ids, resolved config, git sha, env dump, metrics files.

Every experiment writes into ``results/<run_id>/`` so a number in the write-up can
be traced back to the exact protocol, code revision, and environment that made it.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "protocol.yaml"
DEFAULT_RESULTS = PROJECT_ROOT / "results"


def git_sha(short: bool = True, cwd: str | Path | None = None) -> str:
    cwd = str(cwd or PROJECT_ROOT)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short" if short else "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001 - not a git checkout is fine
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_protocol(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load ``protocol.yaml`` and apply overrides; refuses to silently mutate a frozen protocol."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyyaml is required: pip install pyyaml") from exc

    path = Path(path or DEFAULT_PROTOCOL)
    if not path.exists():
        raise FileNotFoundError(f"protocol not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle) or {}
    if overrides:
        if protocol.get("frozen", False):
            print(
                "[run] WARNING: overriding fields of a frozen protocol; "
                "the resolved config is saved with this run and must be reported"
            )
        protocol = _deep_merge(protocol, overrides)
    protocol["_protocol_path"] = str(path)
    return protocol


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml

        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except ImportError:  # pragma: no cover
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _jsonable(obj: Any) -> Any:
    import torch

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclass
class Run:
    tag: str
    run_id: str
    dir: Path
    protocol: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics_dir(self) -> Path:
        return self.dir / "metrics"

    def path(self, *parts: str) -> Path:
        target = self.dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save_json(self, name: str, payload: Any) -> Path:
        target = self.metrics_dir / (name if name.endswith(".json") else f"{name}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        self.metrics[name] = payload
        return target

    def write_text(self, name: str, text: str) -> Path:
        target = self.dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def record_env(self, model: Any = None, extra: dict[str, Any] | None = None) -> Path:
        from . import env as rgenv

        return self.write_text("env.txt", rgenv.env_text(model, extra))

    def __str__(self) -> str:
        return f"Run({self.run_id}, tag={self.tag}, dir={self.dir})"


def new_run(
    tag: str,
    protocol: dict[str, Any] | None = None,
    protocol_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    results_root: str | Path | None = None,
    seed: int | None = None,
) -> Run:
    """Create ``results/<timestamp>_<sha>_<tag>/`` with resolved config + env + sha."""
    if protocol is None:
        protocol = load_protocol(protocol_path, overrides)
    elif overrides:
        protocol = _deep_merge(protocol, overrides)

    seed = int(seed if seed is not None else protocol.get("calibration", {}).get("seed", 42))
    set_seed(seed)

    root = Path(results_root or DEFAULT_RESULTS)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{git_sha()}_{tag}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run = Run(tag=tag, run_id=run_id, dir=run_dir, protocol=protocol, seed=seed)
    _dump_yaml(run_dir / "config.resolved.yaml", protocol)
    run.write_text("git_sha.txt", f"{git_sha(short=False)}\n")
    run.write_text(
        "cmdline.txt",
        " ".join([sys.executable, *sys.argv]) + f"\ncwd={os.getcwd()}\n",
    )
    run.record_env()
    return run


# Gate semantics, stated once so the labels cannot be misread later.
#   constraint -> measured value must satisfy the direction against the threshold
#   decision   -> the threshold selects a branch; there is no "pass"/"fail"
GATE_SPECS: dict[str, dict[str, Any]] = {
    "gini_skip_balance": {
        "kind": "decision",
        "outcome_if_high": "proceed_with_balancing",
        "outcome_if_low": "pivot_to_redundancy_only",
        "meaning": "baseline Gini below this means load balancing has nothing to fix",
    },
    "max_kl_nats": {
        "kind": "constraint",
        "direction": "<=",
        "meaning": "router-distribution shift on eval must stay under this (nats)",
    },
    "min_gain_over_frequency_pts": {
        "kind": "constraint",
        "direction": ">=",
        "meaning": "redundancy pruning must beat frequency pruning by this many ppl points",
    },
    "min_balance_transfer_ratio": {
        "kind": "constraint",
        "direction": ">=",
        "meaning": "fraction of the calibration-time balance gain that must survive on eval",
    },
}


def summarize_gates(protocol: dict[str, Any], measured: dict[str, float]) -> dict[str, Any]:
    """Evaluate measured values against ``protocol.yaml`` gates.

    A constraint reports pass/fail. A decision gate reports which branch was taken,
    because for e.g. ``gini_skip_balance`` a *high* value is the good outcome.
    """
    out: dict[str, Any] = {}
    for name, threshold in (protocol.get("gates") or {}).items():
        spec = GATE_SPECS.get(name, {"kind": "constraint", "direction": "<="})
        value = measured.get(name)
        entry: dict[str, Any] = {
            "threshold": threshold,
            "measured": value,
            "kind": spec["kind"],
            "meaning": spec.get("meaning"),
        }
        if value is None:
            entry["status"] = "not_measured"
            out[name] = entry
            continue

        if spec["kind"] == "decision":
            branch = spec["outcome_if_high"] if value >= threshold else spec["outcome_if_low"]
            entry["status"] = branch
            entry["detail"] = f"{name}={value:.4f} {'>=' if value >= threshold else '<'} {threshold} -> {branch}"
        else:
            direction = spec["direction"]
            passed = value <= threshold if direction == "<=" else value >= threshold
            entry["status"] = "pass" if passed else "fail"
            entry["direction"] = direction
            entry["detail"] = f"{name}={value:.4f} {direction} {threshold} -> {'pass' if passed else 'fail'}"
        out[name] = entry
    return out


def format_gates(gates: dict[str, Any]) -> str:
    """One line per gate, safe when a gate was not measured in this step."""
    lines = []
    for name, entry in gates.items():
        if entry.get("measured") is None:
            lines.append(f"  {name}: not measured in this step (threshold {entry['threshold']})")
        else:
            lines.append(f"  {entry.get('detail', name)}")
    return "\n".join(lines)
