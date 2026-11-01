"""SPEED: SPD multi-resolution sampling for Anima as a MODEL patch.

SPD (Xiao et al., arXiv:2605.18736): the early, noise-dominated steps run on a low-resolution latent (DCT
low-pass); when sigma drops to the handoff value the latent is spectrally expanded to full size (its DCT
block embedded in the full grid, the new high frequencies filled with sigma-scaled noise) and the
remaining sigmas are re-spaced to the aligned sigma. With KludgeAnimaSpectrum on the same model the
full-resolution part is forecast, which is SPEED (Spectrum + SPD) as in ComfyUI-Spectrum-KSampler.

Expansion geometry follows networks/spd_core.py of anima_lora by Seunghyun Ji, MIT License
(LICENSE-anima-lora), as vendored in ComfyUI-Spectrum-KSampler (LICENSE-spectrum-ksampler). Different here: instead of a custom Euler loop, the node wraps
the chosen sampler and calls it twice (low-res sigmas, then the re-spaced full-res sigmas), so any
k-diffusion sampler works. Multistep samplers restart their history at the handoff.
"""

import functools
import logging
import math

import torch
import comfy.samplers
from comfy.patcher_extension import WrappersMP

from .spectrum import LOWRES_KEY

logger = logging.getLogger(__name__)

PATCH = 2  # Anima patch_spatial: latent H/W stay even
_DCT = {}


def _dct_matrix(n, device):
    m = _DCT.get((n, device))
    if m is None:
        k = torch.arange(n, device=device, dtype=torch.float32)
        m = torch.cos(math.pi * k[:, None] * (2 * k[None, :] + 1) / (2 * n))
        m[0] *= 1.0 / math.sqrt(n)
        m[1:] *= math.sqrt(2.0 / n)
        _DCT[(n, device)] = m
    return m


def _dct2(x):
    return _dct_matrix(x.shape[-2], x.device) @ x @ _dct_matrix(x.shape[-1], x.device).T


def _idct2(x):
    return _dct_matrix(x.shape[-2], x.device).T @ x @ _dct_matrix(x.shape[-1], x.device)


def _snap(v):
    return max(PATCH, int(round(v / PATCH)) * PATCH)


def _lowpass(x, scale):
    h, w = min(_snap(x.shape[-2] * scale), x.shape[-2]), min(_snap(x.shape[-1] * scale), x.shape[-1])
    return _idct2(_dct2(x.float())[..., :h, :w]).to(x.dtype)


def _expand(x, sigma, size, gen):
    """Low-res latent at `sigma` -> full-size latent at the aligned sigma (paper Eq. iii, 5-6)."""
    h_lo, w_lo = x.shape[-2:]
    h_hi, w_hi = size
    r = math.sqrt(h_hi * w_hi / (h_lo * w_lo))  # orthonormal DCT: signal amplitude ratio of the two grids
    sigma_new = r * sigma / (1.0 + (r - 1.0) * sigma)
    kappa = r / (1.0 + (r - 1.0) * sigma)
    coef = torch.randn((*x.shape[:-2], h_hi, w_hi), generator=gen, device=x.device, dtype=torch.float32) * sigma
    coef[..., :h_lo, :w_lo] = _dct2(x.float())
    return (_idct2(coef) * kappa).to(x.dtype), sigma_new


def _with_schedule(extra_args, sigmas, lowres):
    model_options = dict(extra_args["model_options"])
    transformer_options = dict(model_options.get("transformer_options", {}))
    transformer_options["sample_sigmas"] = sigmas
    if lowres:
        transformer_options[LOWRES_KEY] = True
    model_options["transformer_options"] = transformer_options
    return {**extra_args, "model_options": model_options}


def _two_stage(sampler_function, k, scale):
    @functools.wraps(sampler_function)  # keeps __name__ for KludgeAnimaSpectrum's sampler check
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
        size = x.shape[-2:]
        x_lo = sampler_function(model, _lowpass(x, scale), sigmas[:k + 1],
                                extra_args=_with_schedule(extra_args, sigmas[:k + 1], True),
                                callback=callback, disable=disable, **options)
        gen = torch.Generator(device=x.device).manual_seed(int(extra_args.get("seed") or 0) + 10_000)
        sigma = float(sigmas[k])
        x_hi, sigma_new = _expand(x_lo, sigma, size, gen)
        rest = sigmas[k:] * (sigma_new / sigma)
        rest[0] = sigma_new
        logger.info("KludgeAnimaSPEED: %dx%d for %d steps, then %dx%d from sigma %.4f (aligned %.4f)",
                    x_lo.shape[-1], x_lo.shape[-2], k, size[-1], size[-2], sigma, sigma_new)

        def callback_hi(d):
            callback({**d, "i": d["i"] + k})

        return sampler_function(model, x_hi, rest, extra_args=_with_schedule(extra_args, rest, False),
                                callback=callback_hi if callback is not None else None, disable=disable, **options)
    return sample


def _sampler_sample_wrapper(scale, handoff_sigma):
    def wrapper(executor, guider, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None,
                disable_pbar=False):
        sampler = executor.class_obj
        values = sigmas.tolist()
        k = next((i for i, s in enumerate(values) if s <= handoff_sigma), len(values) - 1)
        sigma_max = float(guider.model_patcher.get_model_object("model_sampling").sigma_max)
        reason = None
        if not isinstance(sampler, comfy.samplers.KSAMPLER):
            reason = f"sampler {type(sampler).__name__} is not a k-diffusion sampler"
        elif denoise_mask is not None:
            reason = "inpainting mask"
        elif values[0] < 0.999 * sigma_max:
            # img2img / hires pass: shrinking the image latent and expanding it again only blurs it
            reason = f"partial denoise (starts at sigma {values[0]:.4f})"
        elif k == 0:
            reason = f"schedule starts at sigma {values[0]:.4f}, below the handoff"
        elif k >= len(values) - 2:
            reason = "no full-resolution steps left after the handoff"
        if reason is not None:
            logger.info("KludgeAnimaSPEED: off for this sampler call (%s)", reason)
            return executor(guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        original = sampler.sampler_function
        sampler.sampler_function = _two_stage(original, k, scale)
        try:
            return executor(guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        finally:
            sampler.sampler_function = original
    return wrapper


class KludgeAnimaSPEED:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05,
                                "tooltip": "Size of the low-resolution stage (fraction of width/height). "
                                           "0.5 is the tested point; 1.0 turns the node off."}),
            "handoff_sigma": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Switch to full resolution when sigma drops to this. "
                                                   "0.7 is the tested point."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "anima/kludge"
    DESCRIPTION = ("SPEED (SPD): run the early steps on a low-resolution latent, then expand it to full size. "
                   "Works with any sampler. Add Anima Spectrum to forecast the full-resolution steps too.")
    EXPERIMENTAL = True

    def patch(self, model, scale, handoff_sigma):
        if scale >= 1.0 or not 0.0 < handoff_sigma < 1.0:
            return (model,)
        m = model.clone()
        m.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, "kludge_speed", _sampler_sample_wrapper(scale, handoff_sigma))
        return (m,)
