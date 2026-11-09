"""Shared test fixtures: tiny OLMoE / Qwen1.5-MoE models built from the *real* upstream
classes, so the "original" forward in every equivalence test is literally
transformers' own code, not a reimplementation. No downloads, no GPU.
"""

import os

import pytest
import torch

TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_experts=8,
    num_experts_per_tok=3,
    max_position_embeddings=64,
)

FLAVORS = ["olmoe", "qwen"]


def build_model(flavor: str, norm_topk_prob: bool = False, dtype=torch.float32, seed: int = 0):
    torch.manual_seed(seed)
    if flavor == "olmoe":
        from transformers import OlmoeConfig, OlmoeForCausalLM

        cfg = OlmoeConfig(**TINY, norm_topk_prob=norm_topk_prob)
        model = OlmoeForCausalLM(cfg)
    elif flavor == "qwen":
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

        cfg = Qwen2MoeConfig(
            **TINY,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=32,
            norm_topk_prob=norm_topk_prob,
            decoder_sparse_step=1,
        )
        model = Qwen2MoeForCausalLM(cfg)
    else:
        raise ValueError(flavor)
    model = model.to(dtype).eval()
    return model


def make_ids(batch: int = 2, seq: int = 16, vocab: int = 128, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=generator)


@pytest.fixture(params=FLAVORS)
def flavor(request):
    return request.param


@pytest.fixture
def model(flavor):
    return build_model(flavor)


@pytest.fixture
def ids():
    return make_ids()


REAL_MODELS = os.environ.get("ROUTERGUARD_REAL_MODELS", "") == "1"
requires_gpu_and_weights = pytest.mark.skipif(
    not REAL_MODELS or not torch.cuda.is_available(),
    reason="set ROUTERGUARD_REAL_MODELS=1 on a GPU runtime to run real-checkpoint tests",
)
