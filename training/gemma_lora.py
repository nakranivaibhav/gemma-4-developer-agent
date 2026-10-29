"""Gemma 4 + LoRA pieces shared by SFT (training/sft_train.py) and RL (training/rl/), so the two cannot drift.

- Base weights: the int4 compressed-tensors checkpoint unpacked to bf16 (run_compressed=False): the exact values the
  scorer's int4 kernel computes with. Ship the adapter separately; never merge it into the int4 weights.
- Attention: E2B global layers have head_dim 512 and 1 KV head; SDPA with GQA has no fused kernel for that and falls
  back to "math" (31 GB at 32k). GQA is turned off (transformers repeats the KV head) and the forward must run under
  `sdpa_ctx()` (memory-efficient kernel only, never a silent fallback).
- Final hidden state: read from the final norm through a hook; output_hidden_states=True would keep every layer's
  output on the GPU. The LM head is the tied input embedding.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch


def patch_attention():
    import transformers.integrations.sdpa_attention as sa
    sa.use_gqa_in_sdpa = lambda attention_mask, key: False


def sdpa_ctx():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])


def offload_frozen_parts(model):
    """E4B: move frozen weights the text forward does not need on the GPU to CPU (~6.5 GB).
    embed_tokens_per_layer (262k x 42 layers x 256, 5.6 GB bf16) is a plain lookup with no gradient:
    it runs on CPU and only its output (seq x 10752) goes to the GPU. The audio and vision towers
    are never called on text-only input."""
    lm = model.model.language_model
    ple = lm.embed_tokens_per_layer.to("cpu").requires_grad_(False)
    ple.register_forward_pre_hook(lambda m, args: (args[0].to("cpu"),))
    ple.register_forward_hook(lambda m, args, out: out.to("cuda"))
    for name in ("audio_tower", "vision_tower"):
        if getattr(model.model, name, None) is not None:
            getattr(model.model, name).to("cpu")
    torch.cuda.empty_cache()
    print(f"offloaded per-layer embeddings + towers to CPU | {torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU")


def load_base(model_dir: str | Path, offload_ple: bool = False):
    from transformers import AutoModelForCausalLM, CompressedTensorsConfig

    patch_attention()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )
    model.config.use_cache = False
    if offload_ple:
        offload_frozen_parts(model)
    return model


def base_of(model):
    """The transformers model under a PeftModel (or the model itself)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


@contextmanager
def final_hidden(model):
    """with final_hidden(model) as grab: model(...); h = grab["h"]  -> output of the final norm (B, T, H)."""
    grab = {}
    hook = base_of(model).model.language_model.norm.register_forward_hook(lambda m, a, o: grab.__setitem__("h", o))
    try:
        yield grab
    finally:
        hook.remove()


def lm_head_weight(model) -> torch.Tensor:
    return model.get_output_embeddings().weight
