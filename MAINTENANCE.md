# Maintenance notes

Known risks and what to check after updating ComfyUI, comfy-kitchen or torch.

Tested with ComfyUI 0.43.0, comfy-kitchen 0.2.40, torch 2.14.1+cu130 (2026-10-29; Spectrum and SPEED 2026-11-01).
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

## Spectrum — re-implements the end of the Anima forward

Tested 2026-11-01 against ComfyUI-Spectrum-KSampler 2.13.0.

- On predicted steps the node runs its own copy of the tail of `MiniTrainDIT._forward`
  (`comfy/ldm/cosmos/predict2.py`): `t_embedder[0]` → cast → `t_embedder[1]` → `t_embedding_norm`
  → `final_layer(x, emb, adaln_lora_B_T_3D=...)` → `unpatchify` → crop. If ComfyUI changes that
  tail (new modulation, another final layer argument), predicted steps go wrong silently.
- The feature is caught with a pre-hook on `final_layer`, inside a `DIFFUSION_MODEL` wrapper. Works
  with per-block compile; the core TorchCompileModel (whole model compiled) traces the wrapper and is
  not supported. If Anima moves to the memory compiler (`comfy/model_prefetch.py`), the copy is
  already made under `pause_malloc_graph`.
- Relies on `transformer_options` keys `sample_sigmas`, `sigmas`, `cond_or_uncond`, `uuids`, and on
  the flow model getting sigma 1.0 as 0.99997 (matched with a 1e-4 tolerance). If steps stop being
  found, every call runs the full model: the log line then shows few or no predicted steps.
- ControlNet-style `control` input → full model calls only.
- Samplers (2026-11-01, 3 scenes each, 100% crops): euler, er_sde, euler_ancestral, dpmpp_2m, uni_pc,
  heun, dpmpp_sde clean. dpmpp_2m_sde(_gpu) gets iridescent blotches and dpmpp_3m_sde breaks up; the
  original DiT Spectrum Patch does the same, so it is the method, not this code: a predicted step
  can't react to the noise injected the step before, and the multistep correction amplifies that.
  The node checks `sampler_function.__name__` against `UNSUPPORTED_SAMPLERS` and stays off for them;
  a renamed or new SDE multistep sampler would slip through. Multi-call samplers: off-grid
  (midpoint) calls always run the full model, so dpmpp_sde gains little (18 vs 20 s); heun predicts
  26 of 59 calls.
- Extra VRAM (memtrace peak, 2B, batch 1): +512 MB at 1024², +1152 MB at 1536² with
  `history_size` 100; +256 / +576 MB with 8, which looked the same at 100% (25-28 dB vs 100).
- Also checked: 2.9B with a 2B LoRA + Remap + compile (LoRA applied), batch 2, CFG 1, 20 and 50
  steps (`20 steps, 14 full, 6 predicted`; `50 steps, 22 full, 28 predicted`), hires 2-pass on the
  same patched model (each pass gets its own forecast), five 1536-area buckets.
- Check: with defaults and 30 steps the log says `30 steps, 17 full model calls, 13 predicted`. Next
  to the original DiT Spectrum Patch (same settings, er_sde) the images look the same and both sit
  in the same PSNR range against no Spectrum (19-26 dB, 3 prompts x 2 seeds); kludge vs original is
  25-33 dB. Not bit-identical: the original rounds the forecast to bf16.

## SPEED — swaps the sampler function for one call

- Wraps `SAMPLER_SAMPLE`; when the sampler is a `KSAMPLER` it temporarily replaces
  `sampler_function` with a two-stage function (low-res sigmas, expand, re-spaced full-res sigmas)
  and restores it afterwards. Relies on the k-diffusion call signature
  `(model, x, sigmas, extra_args, callback, disable, **options)`, on `extra_args["seed"]` and
  `extra_args["model_options"]["transformer_options"]["sample_sigmas"]`, and on the callback dict's
  `"i"`.
- Turns itself off (with a log line) for inpainting masks, non-k-diffusion samplers and partial
  denoise (first sigma below `model_sampling.sigma_max`). Without that check a hires pass at denoise
  0.5 (starts at sigma 0.75) shrank and re-expanded the upscaled image and came out blurred.
- The two-stage function keeps the sampler's `__name__` (functools.wraps) so Spectrum's sampler
  check still works when SPEED is outside it.
- Conditioning that depends on the latent size (area conds, concat latents) is not handled at the
  low-res stage.
- Block Compile sees two latent sizes, so torch.compile switches to dynamic shapes on the first run
  (longer first compile).
- Measured with CK + int8 + Block Compile, er_sde 30 steps (no patch → Spectrum / SPEED /
  SPEED + Spectrum): 2B 1024² 6.58 → 3.98 / 5.28 / 4.95 s, 2B 1536² 17.53 → 10.43 / 10.23 / 9.10 s,
  2.9B 1024² 9.24 → 5.52 / 7.73 / 6.62 s. At 1024² the low-res stage is too small to use the GPU well
  and only 2 of the 13 full-res steps get predicted, so Spectrum alone wins there. 1536-area
  buckets (1536², 1344x1728, 1728x1344, 1152x2048, 2048x1152): 16.3-16.6 → 9.7-10.0 / - /
  9.0-9.5 s, all coherent.
- Works with every k-diffusion sampler tried (the 8 above for Spectrum, plus dpmpp_2m_sde and
  dpmpp_3m_sde_gpu with SPEED alone); multistep samplers restart their history at the handoff.
  With 20 steps the full-res stage has 8 steps and Spectrum predicts none of them.
- Check: SPEED + Spectrum with `euler` must stay close to the original `KSampler (Spectrum + SPD /
  SPEED)` with `adaptive_smc_alpha` 0 (33-35 dB, 3 prompts x 2 seeds; same composition), and the log
  shows `64x64 for 17 steps, then 128x128 from sigma 0.6970 (aligned 0.8215)` at 1024², 30 steps,
  simple scheduler.
