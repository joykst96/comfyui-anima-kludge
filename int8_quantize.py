"""Runtime int8 quantization of an Anima model (no file written).

Goes right after the diffusion model loader. Re-reads the source file the loader used, quantizes the
2-D linears inside the DiT blocks (attention, MLP, adaLN modulation) to int8 with a per-output-channel
absmax scale, and builds a new model through ComfyUI's normal loader, so LoRA, the 2.9B remap,
block compile and dynamic VRAM work exactly as with a pre-quantized int8 file. llm_adapter, embedders
and the final layer stay in their source dtype.

convrot: block-Hadamard rotation (largest of 256/64/16 dividing K) before quantizing, stored as
int8_tensorwise + convrot so the runtime rotates activations to match. rowwise: no rotation.
"""

import json
import logging
import re

import torch
import comfy.model_management
import comfy.sd
import comfy.utils
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

logger = logging.getLogger(__name__)

BLOCK_LINEAR = re.compile(r"(?:^|\.)blocks\.\d+\..+\.weight$")
CONVROT_GROUPSIZES = (256, 64, 16)
MIN_GEMM = 256  # a GEMM whose small side is under this never beats bf16, int8 is pure overhead


def _quant_config(cfg):
    return torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)


def _quantize(w, groupsize, device):
    w = w.to(device, torch.float32)
    if groupsize:
        w = _rotate_weight(w, _build_hadamard(groupsize, device=device, dtype=torch.float32), groupsize)
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q.cpu(), scale.cpu()


def _load_quantized(unet_path, model_options, fmt, disable_dynamic=False):
    sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
    device = comfy.model_management.get_torch_device()
    out, count = {}, 0
    for key, w in sd.items():
        if not BLOCK_LINEAR.search(key) or "llm_adapter" in key or w.ndim != 2 or min(w.shape) < MIN_GEMM:
            out[key] = w
            continue
        groupsize = next((g for g in CONVROT_GROUPSIZES if w.shape[1] % g == 0), None) if fmt == "convrot" else None
        if fmt == "convrot" and groupsize is None:
            out[key] = w
            continue
        base = key[:-len(".weight")]
        out[key], out[f"{base}.weight_scale"] = _quantize(w, groupsize, device)
        cfg = {"format": "int8_tensorwise"}
        if groupsize:
            cfg.update(convrot=True, convrot_groupsize=groupsize)
        out[f"{base}.comfy_quant"] = _quant_config(cfg)
        count += 1
    del sd
    logger.info("KludgeAnimaInt8Quantize: %s, %d layers quantized", fmt, count)
    # cached_patcher_init is how ComfyUI rebuilds the model, e.g. clone(disable_dynamic=True) in block compile
    model = comfy.sd.load_diffusion_model_state_dict(out, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)
    model.cached_patcher_init = (_load_quantized, (unet_path, model_options, fmt))
    return model


class KludgeAnimaInt8Quantize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "Anima model straight from Load Diffusion Model (no LoRA or patches yet)."}),
            "format": (["convrot", "rowwise"], {"default": "convrot"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "quantize"
    CATEGORY = "anima/kludge"
    DESCRIPTION = ("Quantizes an Anima model to int8 at load time, same result as a pre-quantized int8 file. "
                   "Put it right after Load Diffusion Model, before LoRA loaders. Nothing is written to disk.")

    def quantize(self, model, format):
        if model.model.model_config.unet_config.get("image_model") != "anima":
            raise ValueError("KludgeAnimaInt8Quantize: not an Anima model")
        if model.patches:
            raise ValueError("KludgeAnimaInt8Quantize: place it directly after the model loader, before LoRA loaders")
        init = getattr(model, "cached_patcher_init", None)
        if not init or init[0] is not comfy.sd.load_diffusion_model:
            raise ValueError("KludgeAnimaInt8Quantize: needs a model from Load Diffusion Model (UNETLoader)")
        unet_path, model_options = init[1][0], init[1][1]
        return (_load_quantized(unet_path, model_options, format),)
