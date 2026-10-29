"""Per-block torch.compile for the Anima DiT, with Comfy Kitchen attention support.

Based on AnimaBlockCompile from comfyui-anima-blockcompile, Copyright (c) 2026 Seunghyun Ji, MIT License
(LICENSE-anima-blockcompile): torch.compile each transformer block through ComfyUI core's
set_torch_compile_wrapper. Added here: CK attention routing.

Anima calls attention with tensor containers, and CK's container path (prequantize_int8_attention)
passes raw data pointers that dynamo can't trace ("Cannot access data pointer of Tensor"). The plain
CK function goes through the comfy_kitchen::int8_attention custom op instead, which compiles as an
opaque call and gives bit-identical output. Covers --use-ck-attention and the ModelAttentionBackend node.
"""

import logging

import comfy.ldm.modules.attention as comfy_attention
from comfy_api.torch_helpers import set_torch_compile_wrapper

logger = logging.getLogger(__name__)


def _skip_transformer_options_guards(guard_entries):
    # transformer_options is a fresh dict every step; guarding on it would recompile every block each step
    return [("transformer_options" not in entry.name) for entry in guard_entries]


class KludgeAnimaBlockCompile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "anima/kludge"
    DESCRIPTION = ("torch.compile (inductor) the Anima DiT one transformer block at a time. "
                   "Works with Comfy Kitchen attention. Turns off dynamic VRAM for this model.")
    EXPERIMENTAL = True

    def patch(self, model):
        # dynamic VRAM keeps weights symbolic-shaped under dynamo and the block matmuls fail the meta check
        m = model.clone(disable_dynamic=True)

        ck = comfy_attention.attention_comfy_kitchen_int8
        override = m.model_options["transformer_options"].get("optimized_attention_override")
        if (override is None and comfy_attention.optimized_attention is ck) or \
                getattr(override, "container_function", None) is ck.container_function:
            m.set_model_optimized_attention(ck.__wrapped__)
            logger.info("KludgeAnimaBlockCompile: routing Comfy Kitchen attention through its custom op")

        blocks = m.get_model_object("diffusion_model").blocks
        set_torch_compile_wrapper(
            model=m,
            keys=[f"diffusion_model.blocks.{i}" for i in range(len(blocks))],
            backend="inductor",
            options={"guard_filter_fn": _skip_transformer_options_guards},
        )
        logger.info("KludgeAnimaBlockCompile: per-block compile of %d blocks", len(blocks))
        return (m,)
