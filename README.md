# comfyui-anima-kludge

Makeshift helper nodes for Anima (2B / 2.9B). Category `anima/kludge`.

```
Load Diffusion Model → Anima INT8 Quantize → LoRA loader(s) → Anima 2.9B LoRA Remap → Anima Block Compile → KSampler
```

Every node is optional; keep the order.

## Anima INT8 Quantize (kludge) — `KludgeAnimaInt8Quantize`

Quantizes a bf16 Anima model (2B or 2.9B) to int8 while loading; nothing is written to disk.
Put it right after Load Diffusion Model, before any LoRA loader. It re-reads the file the loader
used and quantizes the 2-D linears inside the DiT blocks (attention, MLP, adaLN modulation) with a
per-output-channel absmax scale; llm_adapter, embedders and the final layer keep their dtype.

- `convrot`: block-Hadamard rotation before quantizing (lower weight error). Default.
- `rowwise`: no rotation, slightly faster.

Output is bit-identical to a pre-quantized `int8_tensorwise` file of the same format, and the
model goes through ComfyUI's normal loader, so LoRA (requantized on load), the 2.9B remap and
block compile work as usual. Costs a few seconds on first load (more with Block Compile, which
rebuilds the model once with dynamic VRAM off).

## Anima Block Compile (kludge) — `KludgeAnimaBlockCompile`

`torch.compile` (inductor) per transformer block via ComfyUI's `set_torch_compile_wrapper`.
Unlike the original AnimaBlockCompile it also works with Comfy Kitchen attention
(`--use-ck-attention` or the Model Attention Backend node): CK attention is routed through the
`comfy_kitchen::int8_attention` custom op instead of the container path dynamo can't trace.
Output without compile is bit-identical to default CK. Turns off dynamic VRAM for the model.
First generation after a server start or resolution change includes the compile.

## Anima 2.9B LoRA Remap (kludge) — `KludgeAnima29BLoraRemap`

Lets 2B (28-block) Anima LoRAs work on Anima 2.9B. Place it after the LoRA loaders (LoRA Manager,
core LoRA loader, ...). It moves patches of 2B LoRAs from block i to the inherited 2.9B block
(0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39);
the 12 inserted blocks get no LoRA. Each LoRA is judged separately: one that reaches block 28+ is
already a 2.9B LoRA and is left alone. Does nothing on 2B models.
Result is bit-identical to a pre-converted `*_29b` LoRA file.

## License

MIT, Copyright (c) 2026 joykst96 (see LICENSE). Code only; no model weights are included.

`block_compile.py` is based on AnimaBlockCompile from
[comfyui-anima-blockcompile](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-blockcompile)
by Seunghyun Ji, MIT License (see LICENSE-anima-blockcompile).

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
