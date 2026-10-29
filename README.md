# comfyui-anima-kludge

Makeshift helper nodes for Anima (2B / 2.9B). Category `anima/kludge`.

```
Load Diffusion Model → Anima INT8 Quantize → LoRA loader(s) → Anima 2.9B LoRA Remap → Anima Block Compile → KSampler
```

Every node is optional; keep the order.

## Nodes

**Anima INT8 Quantize** (`KludgeAnimaInt8Quantize`) — quantizes a bf16 Anima model to int8 while
loading, nothing written to disk. `convrot` (default, lower error) or `rowwise` (no rotation).
Same result as a pre-quantized `int8_tensorwise` file. Put it right after Load Diffusion Model.

**Anima 2.9B LoRA Remap** (`KludgeAnima29BLoraRemap`) — lets 2B Anima LoRAs work on Anima 2.9B by
moving their patches onto the 2.9B blocks they came from. Put it after the LoRA loaders (LoRA
Manager, core LoRA loader, ...). LoRAs already made for 2.9B and 2B models are left alone.

**Anima Block Compile** (`KludgeAnimaBlockCompile`) — `torch.compile` per transformer block, also
with Comfy Kitchen attention (`--use-ck-attention`). Turns off dynamic VRAM for the model and uses a
little more attention memory. The first run after a server start or resolution change compiles.

Update risks and checks for each node: [MAINTENANCE.md](MAINTENANCE.md).

## License

MIT, Copyright (c) 2026 joykst96 (see LICENSE). Code only; no model weights are included.
`block_compile.py` is based on AnimaBlockCompile by Seunghyun Ji, MIT License
(see LICENSE-anima-blockcompile).

## Thanks

These projects were referenced while making this pack. Thank you to their authors.

- [comfyui-anima-blockcompile](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-blockcompile)
  by Seunghyun Ji: the per-block compile node this pack's Block Compile is built on.
- [ComfyUI-EasyUseAnima](https://github.com/n0va39/ComfyUI-EasyUseAnima) by n0va39: its 2.9B LoRA
  support uses the same 2B → 2.9B block table, which helped confirm the mapping.
- [comfy-model-tools](https://github.com/Comfy-Org/comfy-model-tools) by Comfy-Org: the int8 ConvRot
  quantizer whose output the INT8 Quantize node was checked against.
- [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast) by BobJohnson24: referenced
  for how LoRAs can be handled on int8 models.
- [ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) by willmiao: the LoRA loader
  this pack was tested with.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and Comfy Kitchen: the compile wrapper,
  LoRA patching and int8 attention this pack relies on.
