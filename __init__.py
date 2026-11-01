from .block_compile import KludgeAnimaBlockCompile
from .int8_quantize import KludgeAnimaInt8Quantize
from .lora_remap import KludgeAnima29BLoraRemap
from .spectrum import KludgeAnimaSpectrum
from .speed import KludgeAnimaSPEED

NODE_CLASS_MAPPINGS = {
    "KludgeAnimaInt8Quantize": KludgeAnimaInt8Quantize,
    "KludgeAnima29BLoraRemap": KludgeAnima29BLoraRemap,
    "KludgeAnimaSpectrum": KludgeAnimaSpectrum,
    "KludgeAnimaSPEED": KludgeAnimaSPEED,
    "KludgeAnimaBlockCompile": KludgeAnimaBlockCompile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KludgeAnimaInt8Quantize": "Anima INT8 Quantize (kludge)",
    "KludgeAnima29BLoraRemap": "Anima 2.9B LoRA Remap (kludge)",
    "KludgeAnimaSpectrum": "Anima Spectrum (kludge)",
    "KludgeAnimaSPEED": "Anima SPEED (kludge)",
    "KludgeAnimaBlockCompile": "Anima Block Compile (kludge)",
}
