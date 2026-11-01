# comfyui-anima-kludge

Makeshift helper nodes for Anima (2B / 2.9B). Category `anima/kludge`.

```
Load Diffusion Model → Anima INT8 Quantize → LoRA loader(s) → Anima 2.9B LoRA Remap
  → Anima SPEED → Anima Spectrum → Anima Block Compile → KSampler
```

Every node is optional; keep the order.

## Nodes

**Anima INT8 Quantize** (`KludgeAnimaInt8Quantize`) — quantizes a bf16 Anima model to int8 while
loading, nothing written to disk. `convrot` (default, lower error) or `rowwise` (no rotation).
Same result as a pre-quantized `int8_tensorwise` file. Put it right after Load Diffusion Model.

**Anima 2.9B LoRA Remap** (`KludgeAnima29BLoraRemap`) — lets 2B Anima LoRAs work on Anima 2.9B by
moving their patches onto the 2.9B blocks they came from. Put it after the LoRA loaders (LoRA
Manager, core LoRA loader, ...). LoRAs already made for 2.9B and 2B models are left alone.

**Anima Spectrum** (`KludgeAnimaSpectrum`) — Spectrum feature forecasting: on most steps the
transformer blocks are skipped and their output is predicted from earlier steps. A model patch, so the
normal KSampler stays; the step count is read from the sampler, so there is no steps input to keep in
sync. Defaults match ComfyUI-Spectrum-KSampler (30 steps: 17 full model calls, 13 predicted).

**Anima SPEED** (`KludgeAnimaSPEED`) — SPD multi-resolution sampling: the early steps run on a
half-size latent, which is then spectrally expanded to full size at sigma 0.7. Unlike the original
SPEED sampler (Euler only) it wraps whichever sampler the KSampler uses. Changes the composition. With
Anima Spectrum on the same model it is the full SPEED; it pays off at large sizes (1536²), at 1024²
Spectrum alone is faster.

**Anima Block Compile** (`KludgeAnimaBlockCompile`) — `torch.compile` per transformer block, also
with Comfy Kitchen attention (`--use-ck-attention`). Turns off dynamic VRAM for the model and uses a
little more attention memory. The first run after a server start or resolution change compiles.

Update risks and checks for each node: [MAINTENANCE.md](MAINTENANCE.md).

## License

MIT, Copyright (c) 2026 joykst96 (see LICENSE). Code only; no model weights are included.
`block_compile.py` is based on AnimaBlockCompile by Seunghyun Ji, MIT License
(see LICENSE-anima-blockcompile). `spectrum.py` and `speed.py` follow ComfyUI-Spectrum-KSampler by
sorryhyun, MIT License (see LICENSE-spectrum-ksampler).

## Thanks

These projects were referenced while making this pack. Thank you to their authors.

- [comfyui-anima-blockcompile](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-blockcompile)
  by Seunghyun Ji: the per-block compile node this pack's Block Compile is built on.
- [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) by sorryhyun:
  the Spectrum and SPEED samplers that the Spectrum and SPEED nodes follow and were checked against.
  Papers: Spectrum (Han et al., CVPR 2026, arXiv:2603.01623), SPD (Xiao et al., arXiv:2605.18736).
- [ComfyUI-EasyUseAnima](https://github.com/n0va39/ComfyUI-EasyUseAnima) by n0va39: its 2.9B LoRA
  support uses the same 2B → 2.9B block table, which helped confirm the mapping. Its AiO sampler
  showed how Spectrum and SPEED are used in practice.
- [comfy-model-tools](https://github.com/Comfy-Org/comfy-model-tools) by Comfy-Org: the int8 ConvRot
  quantizer whose output the INT8 Quantize node was checked against.
- [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast) by BobJohnson24: referenced
  for how LoRAs can be handled on int8 models.
- [ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) by willmiao: the LoRA loader
  this pack was tested with.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and Comfy Kitchen: the compile wrapper,
  LoRA patching and int8 attention this pack relies on.
