"""Spectrum feature forecasting for the Anima DiT as a MODEL patch.

Spectrum (Han et al., CVPR 2026, arXiv:2603.01623): the feature entering final_layer changes smoothly
with the step, so on most steps all transformer blocks are skipped and that feature is predicted from
earlier steps (Chebyshev ridge fit blended with a first-order Taylor step). Only t_embedder, final_layer
and unpatchify run on a predicted step.

Schedule rule, forecaster math and defaults follow ComfyUI-Spectrum-KSampler by sorryhyun, MIT License
(LICENSE-spectrum-ksampler). Different here: the step count and step index come from the sampler's own
sigmas (no steps input), and every sampler call gets fresh forecasters (hires pass, SPEED stages).
"""

import contextlib
import logging
import math

import torch
from comfy.patcher_extension import WrappersMP

try:
    from comfy.model_prefetch import pause_malloc_graph
except ImportError:
    pause_malloc_graph = contextlib.nullcontext

logger = logging.getLogger(__name__)

STATE_KEY = "kludge_spectrum"
# SDE multistep samplers: a predicted step can't react to the noise injected the step before, and their
# multistep correction amplifies the mismatch (iridescent blotches; 3m_sde breaks up). Same with the
# original Spectrum patch. Spectrum stays off for these; er_sde, euler_ancestral, dpmpp_sde are fine.
UNSUPPORTED_SAMPLERS = {f"sample_dpmpp_{s}" for s in ("2m_sde", "2m_sde_gpu", "2m_sde_heun", "2m_sde_heun_gpu",
                                                      "3m_sde", "3m_sde_gpu")}
# set by the SPEED node on its low-resolution stage: those steps always run the full model
LOWRES_KEY = "kludge_spd_lowres"


class _Forecaster:
    """Ridge regression on Chebyshev T_0..T_m of the step index, blended with a Taylor step by `blend`.
    The prediction is a weighted sum of the stored features, so no fp32 copy of the history is made."""

    def __init__(self, degree, lam, blend, total, history):
        self.degree, self.lam, self.blend, self.total, self.history = degree, lam, blend, total, history
        self.t, self.h = [], []

    def update(self, t, h):
        self.t.append(float(t))
        self.h.append(h)
        if len(self.t) > self.history:
            del self.t[0], self.h[0]

    def ready(self):
        return len(self.t) >= max(2, self.degree + 2)

    def _design(self, t):
        tau = (2.0 * t / self.total - 1.0).reshape(-1, 1)
        cols = [torch.ones_like(tau), tau]
        for _ in range(2, self.degree + 1):
            cols.append(2 * tau * cols[-1] - cols[-2])
        return torch.cat(cols[:self.degree + 1], 1)

    @torch.no_grad()
    def predict(self, t_star):
        t = torch.tensor(self.t, dtype=torch.float64)
        x = self._design(t)
        a = x.T @ x + self.lam * torch.eye(x.shape[1], dtype=torch.float64)
        x_star = self._design(torch.tensor([float(t_star)], dtype=torch.float64))[0]
        c = x @ torch.linalg.solve(a, x_star)
        if self.blend < 1.0:
            k = (t_star - self.t[-1]) / max(self.t[-1] - self.t[-2], 1e-8)
            taylor = torch.zeros_like(c)
            taylor[-1], taylor[-2] = 1 + k, -k
            c = (1 - self.blend) * taylor + self.blend * c
        out = torch.zeros(self.h[0].shape, dtype=torch.float32, device=self.h[0].device)
        for ck, hk in zip(c.tolist(), self.h):
            out.add_(hk, alpha=ck)
        return out.to(self.h[0].dtype)


class _Run:
    """Forecast state for one sigma schedule (one sampler call, or one SPEED stage)."""

    def __init__(self, cfg, sigmas):
        self.cfg, self.sigmas = cfg, sigmas
        self.steps = len(sigmas) - 1
        self.window = cfg["window_size"]
        self.consec_cached = 0
        self.step, self.mode = -1, "actual"
        self.forecasters = {}
        self.actual = self.forecast = 0

    def step_of(self, sigma):
        """Index of `sigma` in the schedule, None for an off-grid evaluation (midpoint solvers).
        The model gets sigma 1.0 as 0.99997 (flow model_sampling clamp), hence the tolerance."""
        for i, s in enumerate(self.sigmas[:-1]):
            if abs(s - sigma) <= 1e-4:
                return i
        return None

    def begin_step(self, step, key):
        """Advance the window rule to `step` and decide its mode (same rule as ComfyUI-Spectrum-KSampler)."""
        if step == self.step:
            return
        if self.mode == "actual":
            if self.step >= self.cfg["warmup_steps"]:
                self.window = round(self.window + self.cfg["flex_window"], 3)
            self.consec_cached = 0
        else:
            self.consec_cached += 1
        self.step = step
        forecaster = self.forecasters.get(key)
        cache = (self.cfg["warmup_steps"] <= step < self.steps - self.cfg["tail_actual_steps"]
                 and forecaster is not None and forecaster.ready()
                 and (self.consec_cached + 1) % max(1, math.floor(self.window)) != 0)
        self.mode = "cached" if cache else "actual"

    def forecaster(self, key, feat):
        f = self.forecasters.get(key)
        if f is not None and f.h[0].shape != feat.shape:
            f = None
        if f is None:
            cfg = self.cfg
            f = self.forecasters[key] = _Forecaster(cfg["cheby_degree"], cfg["ridge_lambda"], cfg["blend_w"],
                                                    self.steps, cfg["history_size"])
        return f


class _State:
    """Lives in transformer_options for one sampler call; holds the run of the current schedule."""

    def __init__(self, cfg):
        self.cfg, self.run, self._src = cfg, None, None
        self.done = []

    def run_for(self, sample_sigmas):
        if sample_sigmas is None:
            return None
        if sample_sigmas is not self._src:
            sigmas = [float(s) for s in sample_sigmas.tolist()]
            if self.run is None or sigmas != self.run.sigmas:
                if self.run is not None:
                    self.done.append(self.run)
                    self.run.forecasters = {}
                self.run = _Run(self.cfg, sigmas)
            self._src = sample_sigmas
        return self.run

    def finish(self):
        runs = self.done + ([self.run] if self.run is not None else [])
        for r in runs:
            r.forecasters = {}
        self.run = self._src = None
        return runs


def _batch_key(transformer_options, x):
    uuids = transformer_options.get("uuids") or ()
    cond_or_uncond = transformer_options.get("cond_or_uncond") or ()
    replaced = tuple(sorted(map(str, transformer_options.get("patches_replace", {}).get("dit", {}))))
    return tuple(cond_or_uncond), tuple(map(str, uuids)), replaced, tuple(x.shape)


def _head(dit, feat, timesteps, x_shape):
    """The part of MiniTrainDIT._forward after the blocks, fed with a predicted feature."""
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    t_emb, adaln_lora = dit.t_embedder[1](dit.t_embedder[0](timesteps).to(feat.dtype))
    t_emb = dit.t_embedding_norm(t_emb)
    out = dit.unpatchify(dit.final_layer(feat, t_emb, adaln_lora_B_T_3D=adaln_lora))
    return out[:, :, :x_shape[-3], :x_shape[-2], :x_shape[-1]]


def _diffusion_model_wrapper(executor, x, timesteps, context, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options", {})
    state = transformer_options.get(STATE_KEY)
    if state is None or transformer_options.get(LOWRES_KEY) or kwargs.get("control") is not None:
        return executor(x, timesteps, context, *args, **kwargs)
    run = state.run_for(transformer_options.get("sample_sigmas"))
    sigmas = transformer_options.get("sigmas")
    step = run.step_of(sigmas.flatten()[0].item()) if run is not None and sigmas is not None else None
    if step is None:
        return executor(x, timesteps, context, *args, **kwargs)

    dit = executor.class_obj
    key = _batch_key(transformer_options, x)
    run.begin_step(step, key)
    forecaster = run.forecasters.get(key)
    if run.mode == "cached" and forecaster is not None and forecaster.ready():
        run.forecast += 1
        return _head(dit, forecaster.predict(step), timesteps, x.shape)

    captured = []

    def capture(module, inputs):
        # the copy must not come from the memory compiler's pool
        with pause_malloc_graph():
            captured.append(inputs[0].detach().clone())

    handle = dit.final_layer.register_forward_pre_hook(capture)
    try:
        out = executor(x, timesteps, context, *args, **kwargs)
    finally:
        handle.remove()
    run.actual += 1
    if len(captured) == 1:
        f = run.forecaster(key, captured[0])
        if f.t and f.t[-1] == step:  # re-evaluation at the same sigma (Heun etc.): keep the latest
            f.h[-1] = captured[0]
        else:
            f.update(step, captured[0])
    return out


def _sampler_sample_wrapper(cfg):
    def wrapper(executor, guider, sigmas, extra_args, *args, **kwargs):
        name = getattr(getattr(executor.class_obj, "sampler_function", None), "__name__", "")
        if name in UNSUPPORTED_SAMPLERS:
            logger.warning("KludgeAnimaSpectrum: off for %s (SDE multistep samplers get color blotches with "
                           "Spectrum); use er_sde, euler, dpmpp_2m or euler_ancestral", name[len("sample_"):])
            return executor(guider, sigmas, extra_args, *args, **kwargs)
        transformer_options = extra_args["model_options"].setdefault("transformer_options", {})
        state = transformer_options[STATE_KEY] = _State(cfg)
        try:
            return executor(guider, sigmas, extra_args, *args, **kwargs)
        finally:
            transformer_options.pop(STATE_KEY, None)
            for r in state.finish():
                logger.info("KludgeAnimaSpectrum: %d steps, %d full model calls, %d predicted",
                            r.steps, r.actual, r.forecast)
    return wrapper


class KludgeAnimaSpectrum:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "warmup_steps": ("INT", {"default": 6, "min": 0, "max": 100,
                                     "tooltip": "First steps that always run the full model."}),
            "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.05,
                                      "tooltip": "Start window: one full model call every N steps."}),
            "flex_window": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 5.0, "step": 0.05,
                                      "tooltip": "Window growth after each full call. Higher is faster."}),
            "tail_actual_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                          "tooltip": "Last steps that always run the full model."}),
            "blend_w": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "1.0 = pure Chebyshev fit, 0.0 = pure Taylor step."}),
            "cheby_degree": ("INT", {"default": 3, "min": 1, "max": 8}),
            "ridge_lambda": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
            "history_size": ("INT", {"default": 100, "min": 3, "max": 1000,
                                     "tooltip": "Full-call features kept per sampler call (VRAM)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "anima/kludge"
    DESCRIPTION = ("Spectrum: skip the transformer blocks on most steps and predict their output. "
                   "Step count comes from the sampler, so it works with any sampler, scheduler and denoise.")
    EXPERIMENTAL = True

    def patch(self, model, **cfg):
        if cfg["history_size"] < cfg["cheby_degree"] + 2:
            raise ValueError("history_size must be at least cheby_degree + 2")
        dit = model.get_model_object("diffusion_model")
        for name in ("final_layer", "t_embedder", "t_embedding_norm", "unpatchify"):
            if not hasattr(dit, name):
                raise ValueError(f"KludgeAnimaSpectrum needs an Anima / Cosmos DiT model (no {name})")
        m = model.clone()
        m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, STATE_KEY, _diffusion_model_wrapper)
        m.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, STATE_KEY, _sampler_sample_wrapper(cfg))
        return (m,)
