"""Anima 2.9B LoRA remap.

Anima 2.9B is the 28-block Anima (2B) with 12 blocks inserted at these new indices. The other 28
positions hold the original blocks in order, so a 2B LoRA fits once its block indices are moved:
2B block i -> LEGACY_BLOCK_MAP[i]. Same table as the community *_29b LoRA conversions.

The node sits after any LoRA loader (LoRA Manager, core LoraLoader, ...) and moves the patches that
came from 2B LoRAs. Patches of one LoRA file share a single `loaded_keys` set (comfy.lora.load_lora),
so each LoRA is judged on its own: one that only touches blocks < 28 is treated as a 2B LoRA, one that
reaches block 28+ is already a 2.9B LoRA and is left alone.
"""

import logging
import re
import uuid

logger = logging.getLogger(__name__)

INSERTED_BLOCKS = (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)
LEGACY_BLOCK_MAP = tuple(i for i in range(40) if i not in INSERTED_BLOCKS)
BLOCK_KEY = re.compile(r"^(diffusion_model\.blocks\.)(\d+)(\..+)$")


def _is_anima_29b(model):
    config = getattr(model.model.model_config, "unet_config", {})
    return config.get("image_model") == "anima" and config.get("num_blocks") == 40


def _lora_id(patch):
    # (strength_patch, adapter, strength_model, offset, function); raw tensor patches have no loaded_keys
    return id(getattr(patch[1], "loaded_keys", None))


class KludgeAnima29BLoraRemap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "remap"
    CATEGORY = "anima/kludge"
    DESCRIPTION = ("Moves patches of 2B (28-block) Anima LoRAs onto the matching blocks of an Anima 2.9B model. "
                   "Place after the LoRA loaders. LoRAs already made for 2.9B are left as they are.")

    def remap(self, model):
        if not _is_anima_29b(model):
            return (model,)

        max_block = {}
        for key, patches in model.patches.items():
            match = BLOCK_KEY.match(key)
            if match is None:
                continue
            for p in patches:
                lora = _lora_id(p)
                max_block[lora] = max(max_block.get(lora, -1), int(match.group(2)))
        legacy = {lora for lora, top in max_block.items() if top < 28}
        if not legacy:
            return (model,)

        m = model.clone()
        patches_out = {}
        moved = 0
        for key, patches in model.patches.items():
            match = BLOCK_KEY.match(key)
            for p in patches:
                target = key
                if match is not None and _lora_id(p) in legacy:
                    target = f"{match.group(1)}{LEGACY_BLOCK_MAP[int(match.group(2))]}{match.group(3)}"
                    moved += 1
                patches_out.setdefault(target, []).append(p)
        m.patches = patches_out
        m.patches_uuid = uuid.uuid4()
        logger.info("KludgeAnima29BLoraRemap: moved %d patches from %d 2B LoRA(s) to 2.9B block positions", moved, len(legacy))
        return (m,)
