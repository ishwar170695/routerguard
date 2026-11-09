"""Model loading: OLMoE (dev, bf16/8-bit/4-bit) and Qwen1.5-MoE (target, NF4 4-bit).

Both loaders return ``(model, tokenizer, LoadInfo)`` and *verify* the per-expert
``nn.ModuleList`` layout before handing the model back (see ``env.assert_moe_structure``).

Precision policy
----------------
* OLMoE bf16 is 13.8 GB of weights -> needs an A100/L4-class card. T4 (Turing,
  cc 7.5) has no native bf16 and 16 GB, so ``auto`` picks 8-bit there.
* Qwen1.5-MoE is always 4-bit NF4 with bf16 compute, gates kept in full precision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from . import env as rgenv

OLMOE_ID = "allenai/OLMoE-1B-7B-0924"
QWEN_ID = "Qwen/Qwen1.5-MoE-A2.7B"

QWEN_SKIP_MODULES = ["gate", "shared_expert_gate", "lm_head"]
OLMOE_SKIP_MODULES = ["gate", "lm_head"]


@dataclass
class LoadInfo:
    model_id: str
    precision: str
    compute_dtype: str
    total_params: int
    moe_params: int
    active_params_estimate: int
    gpu: str
    gpu_total_gb: float
    mem_alloc_gb: float
    mem_reserved_gb: float
    structure: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.model_id} [{self.precision}/{self.compute_dtype}] "
            f"{self.total_params/1e9:.2f}B total, ~{self.active_params_estimate/1e9:.2f}B active, "
            f"{self.mem_alloc_gb:.2f} GB allocated on {self.gpu}"
        )


def gpu_info() -> dict[str, Any]:
    return rgenv.gpu_info()


def choose_precision(gpu: dict[str, Any] | None = None) -> str:
    """Pick bf16 / 8bit / 4bit for OLMoE based on the actual card."""
    gpu = gpu or gpu_info()
    if not gpu.get("cuda_available"):
        return "bf16"  # CPU smoke test; caller decides whether that is sane
    cc = tuple(int(x) for x in str(gpu.get("compute_capability", "0.0")).split("."))
    total = float(gpu.get("total_gb", 0.0))
    if cc >= (8, 0):
        # 13.8 GB of bf16 weights + activations; keep headroom.
        return "bf16" if total >= 20.0 else "8bit"
    return "8bit"  # T4 / V100-era: no native bf16, and 13.8 GB does not fit comfortably


def _count_params(model: torch.nn.Module, blocks: list[tuple[str, Any]]) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    moe = 0
    for _, block in blocks:
        moe += sum(p.numel() for p in block.gate.parameters())
        moe += sum(p.numel() for p in block.experts.parameters())
        shared = getattr(block, "shared_expert", None)
        if shared is not None:
            moe += sum(p.numel() for p in shared.parameters())
            gate = getattr(block, "shared_expert_gate", None)
            if gate is not None:
                moe += sum(p.numel() for p in gate.parameters())
    return total, moe


def _active_estimate(model: torch.nn.Module, blocks: list[tuple[str, Any]], total: int) -> int:
    """Rough active-parameter count: everything except the unselected routed experts."""
    routed = 0
    selected = 0
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or 0)
    for _, block in blocks:
        per_expert = sum(p.numel() for p in block.experts[0].parameters())
        n = len(block.experts)
        routed += per_expert * n
        selected += per_expert * min(top_k, n)
    return total - routed + selected


def _finalize(model, tokenizer, model_id: str, precision: str, notes: list[str]) -> LoadInfo:
    model.eval()
    # Router logits are only needed for inspection; never cache KV in eval loops.
    try:
        model.config.use_cache = False
    except Exception:  # noqa: BLE001
        pass
    try:
        model.config.output_router_logits = False
    except Exception:  # noqa: BLE001
        pass

    structure = rgenv.assert_moe_structure(model)
    blocks = rgenv.find_moe_blocks(model)
    if precision in ("4bit", "8bit"):
        rgenv.assert_gates_unquantized(model)

    total, moe = _count_params(model, blocks)
    gpu = gpu_info()
    info = LoadInfo(
        model_id=model_id,
        precision=precision,
        compute_dtype=str(next(model.parameters()).dtype),
        total_params=total,
        moe_params=moe,
        active_params_estimate=_active_estimate(model, blocks, total),
        gpu=str(gpu.get("name", "cpu")),
        gpu_total_gb=float(gpu.get("total_gb", 0.0)),
        mem_alloc_gb=round(torch.cuda.memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0.0,
        mem_reserved_gb=round(torch.cuda.memory_reserved() / 1024**3, 3) if torch.cuda.is_available() else 0.0,
        structure=structure,
        notes=notes,
    )
    return info


def load_olmoe(
    model_id: str = OLMOE_ID,
    precision: str = "auto",
    device: str = "cuda",
    double_quant: bool = False,
    cache_dir: str | None = None,
    attn_implementation: str | None = None,
) -> tuple[Any, Any, LoadInfo]:
    """Load OLMoE-1B-7B. ``precision`` in {auto, bf16, 8bit, 4bit}."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    notes: list[str] = []
    if precision == "auto":
        precision = choose_precision()
        notes.append(f"auto-selected precision '{precision}'")
    if precision not in ("bf16", "8bit", "4bit"):
        raise ValueError(f"unsupported precision {precision!r}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    kwargs: dict[str, Any] = {"cache_dir": cache_dir}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    if precision == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map={"": device}, **kwargs
        )
    else:
        from transformers import BitsAndBytesConfig

        if precision == "8bit":
            quant = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=OLMOE_SKIP_MODULES,
            )
        else:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=double_quant,
                llm_int8_skip_modules=OLMOE_SKIP_MODULES,
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map={"": device}, **kwargs
        )
        notes.append("quantized load; compare only against unpruned OLMoE at the same precision")
    return model, tokenizer, _finalize(model, tokenizer, model_id, precision, notes)


def load_qwen(
    model_id: str = QWEN_ID,
    device: str = "cuda",
    double_quant: bool = False,
    cache_dir: str | None = None,
    attn_implementation: str | None = None,
) -> tuple[Any, Any, LoadInfo]:
    """Load Qwen1.5-MoE-A2.7B in NF4 with bf16 compute and full-precision routers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    notes = [
        "NF4 4-bit, bf16 compute",
        f"skipped from quantization: {QWEN_SKIP_MODULES}",
        "routed experts only: the always-on shared expert is never pruned",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=double_quant,
        llm_int8_skip_modules=QWEN_SKIP_MODULES,
    )
    kwargs: dict[str, Any] = {"cache_dir": cache_dir}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={"": device}, **kwargs
    )
    info = _finalize(model, tokenizer, model_id, "4bit", notes)
    if not info.structure.get("has_shared_expert"):
        raise RuntimeError("Qwen model has no shared_expert: wrong checkpoint or transformers version")
    return model, tokenizer, info


def free(*objects: Any) -> None:
    """Drop references and empty the CUDA cache between Colab stages."""
    import gc

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
