"""Environment pinning, fallback chain, and hard MoE-structure assertions.

Why this module exists
----------------------
transformers v5 fuses routed experts into 3D tensors (``experts.gate_up_proj``),
which breaks per-expert forward hooks AND makes bitsandbytes unable to quantize
them (transformers issue #1849). RouterGuard needs one ``nn.Module`` per expert,
so we pin 4.48.3 and *assert* the structure at load time instead of hoping.
"""

from __future__ import annotations

import json
import platform
import sys
from typing import Any

REQUIRED_TRANSFORMERS = "4.48.3"
FALLBACK_TRANSFORMERS = ("4.49.0", "4.51.3")
ALLOWED_TRANSFORMERS = (REQUIRED_TRANSFORMERS,) + FALLBACK_TRANSFORMERS
MIN_TORCH = "2.2.0"

# transformers 4.48.3 requires huggingface-hub>=0.24,<1.0. A modern `datasets` wants
# hub>=1.31 and will happily upgrade it during `pip install datasets`, after which
# transformers cannot even be imported. This is silent, session-wasting, and easy to
# hit in Colab, so the range is asserted rather than assumed.
HUB_MIN, HUB_MAX = "0.24.0", "1.0.0"
DATASETS_MIN, DATASETS_MAX = "2.19.0", "5.0.0"

INSTALL_PACKAGES = (
    f"transformers=={REQUIRED_TRANSFORMERS}",
    f"huggingface-hub>={HUB_MIN},<{HUB_MAX}",
    f"datasets>={DATASETS_MIN},<{DATASETS_MAX}",
    "accelerate",
    "bitsandbytes>=0.43.3",
    "pyyaml",
    "matplotlib",
    "sentencepiece",
)

INSTALL_HINT = (
    "pip install -q " + " ".join(f"'{p}'" for p in INSTALL_PACKAGES) + "   # then RESTART the runtime"
)


def _vtuple(s: str) -> tuple[int, ...]:
    out = []
    for part in str(s).split(".")[:3]:
        num = "".join(ch for ch in part if ch.isdigit())
        if num == "":
            break
        out.append(int(num))
    return tuple(out)


# package name -> importable module name (they differ: pyyaml ships module "yaml")
PACKAGE_MODULES = {
    "torch": "torch",
    "transformers": "transformers",
    "huggingface-hub": "huggingface_hub",
    "tokenizers": "tokenizers",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    "numpy": "numpy",
    "pyyaml": "yaml",
}


def versions() -> dict[str, str]:
    """Installed versions of the packages this project depends on."""
    info: dict[str, str] = {"python": platform.python_version()}
    for package, module_name in PACKAGE_MODULES.items():
        try:
            module = __import__(module_name)
            info[package] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001 - report, never crash the dump
            info[package] = f"missing ({type(exc).__name__})"
    return info


def gpu_info() -> dict[str, Any]:
    """GPU name / memory / compute capability, without requiring CUDA."""
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda"] = torch.version.cuda
        if not info["cuda_available"]:
            return info
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "name": props.name,
                "total_gb": round(props.total_memory / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def assert_env(strict: bool = True) -> dict[str, Any]:
    """Verify pins and return a report. Raises on a disallowed transformers version.

    strict=True (default) rejects anything outside ALLOWED_TRANSFORMERS.
    A fallback version is accepted but flagged so runs record which one was used.
    """
    report: dict[str, Any] = {"versions": versions(), "gpu": gpu_info()}
    tf = report["versions"]["transformers"]
    if tf.startswith("missing"):
        raise RuntimeError(f"transformers is not importable.\nInstall with:\n  {INSTALL_HINT}")

    if tf not in ALLOWED_TRANSFORMERS:
        msg = (
            f"transformers=={tf} is not supported by RouterGuard.\n"
            f"  required: {REQUIRED_TRANSFORMERS}\n"
            f"  fallbacks: {', '.join(FALLBACK_TRANSFORMERS)}\n"
            "v5.x fuses experts into 3D tensors, which breaks per-expert hooks and\n"
            "bitsandbytes quantization of experts (issue #1849).\n"
            f"Fix:\n  {INSTALL_HINT}"
        )
        if strict:
            raise RuntimeError(msg)
        report.setdefault("warnings", []).append(msg)

    report["transformers_is_primary"] = tf == REQUIRED_TRANSFORMERS
    if tf in FALLBACK_TRANSFORMERS:
        report.setdefault("warnings", []).append(
            f"running on fallback transformers {tf}; results must be reported with this version"
        )

    hub = report["versions"].get("huggingface-hub", "")
    if hub and not hub.startswith("missing"):
        if not (_vtuple(HUB_MIN) <= _vtuple(hub) < _vtuple(HUB_MAX)):
            raise RuntimeError(
                f"huggingface-hub=={hub} is incompatible with transformers {tf} "
                f"(needs >={HUB_MIN},<{HUB_MAX}).\n"
                "This is almost always caused by a package such as `datasets` upgrading the\n"
                "hub as a side effect of installation; transformers then fails to import.\n"
                f"Fix:\n  {INSTALL_HINT}"
            )
    datasets_v = report["versions"].get("datasets", "")
    if datasets_v and not datasets_v.startswith("missing"):
        if _vtuple(datasets_v) >= _vtuple(DATASETS_MAX) or _vtuple(datasets_v) < _vtuple(DATASETS_MIN):
            raise RuntimeError(
                f"datasets=={datasets_v} is outside the range this project supports "
                f">={DATASETS_MIN},<{DATASETS_MAX} (newer releases require huggingface-hub>=1.x,\n"
                f"which breaks transformers {tf}).\nFix:\n  {INSTALL_HINT}"
            )

    torch_v = report["versions"]["torch"]
    if not torch_v.startswith("missing") and _vtuple(torch_v) < _vtuple(MIN_TORCH):
        raise RuntimeError(f"torch>={MIN_TORCH} required, found {torch_v}")
    if strict and not report["gpu"]["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available. This project is executed on Colab GPUs; "
            "set strict=False only for CPU smoke tests."
        )
    return report


def find_moe_blocks(model: Any) -> list[tuple[str, Any]]:
    """All modules that look like a sparse MoE block (``.gate`` + ``.experts``)."""
    blocks: list[tuple[str, Any]] = []
    for name, mod in model.named_modules():
        if hasattr(mod, "experts") and hasattr(mod, "gate"):
            blocks.append((name, mod))
    return blocks


def _fused_expert_params(model: Any) -> list[tuple[str, tuple[int, ...]]]:
    """State-dict entries whose expert weights are >=3D (transformers v5 fusion)."""
    out: list[tuple[str, tuple[int, ...]]] = []
    try:
        state = model.state_dict()
    except Exception:  # noqa: BLE001
        return out
    for key, tensor in state.items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf == "experts" and getattr(tensor, "ndim", 0) >= 3:
            out.append((key, tuple(tensor.shape)))
    return out


def assert_moe_structure(
    model: Any,
    expect_layers: int | None = None,
    expect_experts: int | None = None,
) -> dict[str, Any]:
    """Hard-assert one ``nn.ModuleList`` entry per routed expert, per MoE layer."""
    import torch.nn as nn

    blocks = find_moe_blocks(model)
    if not blocks:
        fused = _fused_expert_params(model)
        raise RuntimeError(
            "No MoE block with a per-expert nn.ModuleList was found.\n"
            f"  fused expert params (3D+): {fused[:4]}{' ...' if len(fused) > 4 else ''}\n"
            "This is the transformers v5 fused-experts layout. Per-expert hooks and\n"
            f"bitsandbytes expert quantization do not work on it.\nInstall:\n  {INSTALL_HINT}"
        )

    cfg = getattr(model, "config", None)
    num_experts = getattr(cfg, "num_experts", None)
    top_k = getattr(cfg, "num_experts_per_tok", None)
    norm_topk_prob = getattr(cfg, "norm_topk_prob", None)
    counts: list[int] = []
    shared: list[str] = []

    for name, block in blocks:
        if not isinstance(block.experts, nn.ModuleList):
            fused = _fused_expert_params(model)
            if fused:
                raise RuntimeError(
                    f"{name}.experts is {type(block.experts).__name__} with fused 3D+ weights "
                    f"{fused[:2]}: this is the transformers v5 fused-experts layout.\n"
                    "Per-expert hooks and bitsandbytes expert quantization do not work on it.\n"
                    f"Install:\n  {INSTALL_HINT}"
                )
            raise TypeError(
                f"{name}.experts is {type(block.experts).__name__}, expected nn.ModuleList. "
                "RouterGuard cannot hook or prune fused experts."
            )
        counts.append(len(block.experts))
        if not isinstance(block.gate, nn.Linear):
            raise TypeError(f"{name}.gate is {type(block.gate).__name__}, expected nn.Linear")
        if block.gate.out_features != len(block.experts):
            raise ValueError(
                f"{name}: gate.out_features={block.gate.out_features} != "
                f"len(experts)={len(block.experts)}"
            )
        if hasattr(block, "shared_expert"):
            shared.append(name)

    if len(set(counts)) != 1:
        raise ValueError(f"inconsistent expert counts across layers: {sorted(set(counts))}")
    n_experts = counts[0]
    if expect_experts is not None and n_experts != expect_experts:
        raise ValueError(f"expected {expect_experts} experts/layer, found {n_experts}")
    if num_experts is not None and num_experts != n_experts:
        raise ValueError(f"config.num_experts={num_experts} != len(experts)={n_experts}")
    if expect_layers is not None and len(blocks) != expect_layers:
        raise ValueError(f"expected {expect_layers} MoE layers, found {len(blocks)}")

    return {
        "num_moe_layers": len(blocks),
        "num_experts_per_layer": n_experts,
        "top_k": top_k,
        "norm_topk_prob": norm_topk_prob,
        "routed_experts_total": n_experts * len(blocks),
        "has_shared_expert": bool(shared),
        "shared_expert_layers": len(shared),
        "layer_names": [n for n, _ in blocks],
    }


def assert_gates_unquantized(model: Any) -> dict[str, Any]:
    """After 4/8-bit loading, routers must still be full-precision floats.

    If ``llm_int8_skip_modules`` fails to match, the gate becomes a bitsandbytes
    layer and router logits get quantized noise -> silently wrong load statistics.
    """
    import torch.nn as nn

    checked: list[str] = []
    for name, block in find_moe_blocks(model):
        for attr in ("gate", "shared_expert_gate"):
            gate = getattr(block, attr, None)
            if gate is None:
                continue
            weight = getattr(gate, "weight", None)
            if weight is None:
                continue
            if type(gate) is not nn.Linear:
                raise TypeError(
                    f"{name}.{attr} is {type(gate).__name__}, not nn.Linear: it was quantized. "
                    "Add it to llm_int8_skip_modules."
                )
            if weight.dtype not in (None,) and not weight.is_floating_point():
                raise TypeError(f"{name}.{attr}.weight dtype {weight.dtype} is not floating point")
            if weight.element_size() < 2:
                raise TypeError(f"{name}.{attr}.weight is {weight.dtype} (sub-16-bit): router is quantized")
            checked.append(f"{name}.{attr}")
    return {"gates_checked": len(checked), "unquantized": True}


def env_report(model: Any = None) -> dict[str, Any]:
    report: dict[str, Any] = {"versions": versions(), "gpu": gpu_info()}
    if model is not None:
        try:
            report["moe_structure"] = assert_moe_structure(model)
        except Exception as exc:  # noqa: BLE001 - report failures, don't mask them
            report["moe_structure"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            report["model_config"] = {
                k: getattr(model.config, k)
                for k in ("num_hidden_layers", "hidden_size", "num_experts",
                          "num_experts_per_tok", "norm_topk_prob", "torch_dtype")
                if hasattr(model.config, k)
            }
        except Exception:  # noqa: BLE001
            pass
    return report


def env_text(model: Any = None, extra: dict[str, Any] | None = None) -> str:
    """Human-readable dump written next to every run's metrics."""
    report = env_report(model)
    if extra:
        report["extra"] = extra
    lines = [
        "RouterGuard environment",
        "=" * 60,
        f"python              {report['versions']['python']}",
        f"executable          {sys.executable}",
        f"platform            {platform.platform()}",
        "-" * 60,
    ]
    for key, value in report["versions"].items():
        if key == "python":
            continue
        lines.append(f"{key:<20}{value}")
    lines.append("-" * 60)
    gpu = report["gpu"]
    if gpu.get("cuda_available"):
        lines.append(
            f"gpu                 {gpu.get('name')} ({gpu.get('total_gb')} GB, cc {gpu.get('compute_capability')})"
        )
        lines.append(f"cuda                {gpu.get('torch_cuda')}")
    else:
        lines.append(f"gpu                 none ({gpu.get('error', 'cuda unavailable')})")
    if "moe_structure" in report:
        lines.append("-" * 60)
        lines.append(json.dumps(report["moe_structure"], indent=2, default=str))
    if "warnings" in report:
        lines.append("-" * 60)
        for warning in report["warnings"]:
            lines.append(f"WARNING: {warning}")
    return "\n".join(lines) + "\n"


def write_env(path: str, model: Any = None, extra: dict[str, Any] | None = None) -> str:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = env_text(model, extra)
    target.write_text(text, encoding="utf-8")
    return str(target)
