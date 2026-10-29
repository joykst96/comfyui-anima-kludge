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

## Known risks and what to check after ComfyUI updates

Tested with ComfyUI 0.43.0, comfy-kitchen 0.2.40, torch 2.14.1+cu130 (2026-10-29).
Recheck after updating any of these. The quickest check for each node is listed with it.

### Block Compile — the stopgap one, steps off ComfyUI's direction

- Turns off dynamic VRAM, ComfyUI's current memory system. The core TorchCompileModel node does
  the same today, but ComfyUI's own speed path under dynamic VRAM is the block CUDA graphs in
  `comfy/model_prefetch.py`, so this node lives outside that.
- The CK fix bypasses the `AttentionTensorContainer` path ComfyUI is moving to. That path frees
  q/k/v right after quantizing them; the plain path used here keeps them a bit longer
  (same output, a little more temporary memory).
- Relies on internals: `comfy.ldm.modules.attention.attention_comfy_kitchen_int8`, its
  `__wrapped__` and `container_function`, and the module-level `optimized_attention`.
- Retire or patch when:
  - the original AnimaBlockCompile works with `--use-ck-attention` (container path became
    compile-safe) → delete the CK routing block;
  - ComfyUI offers compile with dynamic VRAM on → follow that instead of `disable_dynamic=True`.
- Check: CK + this node must compile without the "Cannot access data pointer of Tensor" error,
  and its uncompiled output must stay bit-identical to plain CK.

### 2.9B LoRA Remap — fits the standard LoRA flow, but can fail silently

- Relies on internals: `ModelPatcher.patches` entries being
  `(strength_patch, adapter, strength_model, offset, function)`, adapters of one LoRA file sharing
  one `loaded_keys` set (`comfy/lora.py`), keys named `diffusion_model.blocks.N....`, and
  `unet_config` `image_model` / `num_blocks`.
- If the `loaded_keys` sharing changes, LoRAs can't be told apart and a native 2.9B LoRA could
  be moved by mistake without any error.
- The block table comes from the 2.9B model layout, not from ComfyUI; it only changes if a new
  2.9B base with a different insertion pattern appears.
- Check: on a 2.9B model, 2B LoRA + this node must be bit-identical to the same LoRA converted
  to `*_29b`, and a `*_29b` LoRA + this node must be unchanged.

### INT8 Quantize — standard format, small own code

- Output is the standard `int8_tensorwise` (+ `convrot`) format loaded by ComfyUI's normal
  loader. If ComfyUI adds its own load-time int8 quantization, this node can simply be retired.
- Uses private comfy-kitchen helpers `_build_hadamard` / `_rotate_weight`
  (`comfy_kitchen.tensor.int8_utils`; they lived in `comfy_kitchen.tensor.int8` before). If they
  move, the import fails. The rotation must match what ComfyUI applies at runtime, so keep using
  comfy-kitchen's helpers rather than a local Hadamard.
- Quantizes in fp32 on purpose: comfy-kitchen's public `TensorWiseINT8Layout.quantize` gave
  slightly higher weight error in testing and isn't bit-identical to comfy-model-tools files.
- Relies on internals: `model.cached_patcher_init == (comfy.sd.load_diffusion_model,
  (unet_path, model_options))` from Load Diffusion Model, and ComfyUI calling the factory again
  with `disable_dynamic=True` (block compile does this).
- Layer selection mirrors comfy-model-tools defaults for Anima (2-D block linears, min side
  ≥ 256, llm_adapter excluded). If that tool changes its defaults, fresh files will differ from
  this node; that's expected, not a bug.
- Check: node output must be bit-identical to a file made by comfy-model-tools
  (`quant_int8_convrot.py`, default options) for the same model.

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
