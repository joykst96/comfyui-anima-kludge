# Maintenance notes

Known risks and what to check after updating ComfyUI, comfy-kitchen or torch.

Tested with ComfyUI 0.43.0, comfy-kitchen 0.2.40, torch 2.14.1+cu130 (2026-10-29).
Recheck after updating any of these. The quickest check for each node is listed with it.

## Block Compile — the stopgap one, steps off ComfyUI's direction

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

## 2.9B LoRA Remap — fits the standard LoRA flow, but can fail silently

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

## INT8 Quantize — standard format, small own code

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
