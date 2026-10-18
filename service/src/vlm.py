"""Local Qwen-VL wrapper. One model instance per process."""
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

_cache = {}


def backends(model):
    """Qwen: force mem-efficient SDPA (math fallback OOMs). Gemma: its masks need the math kernel (short seqs, fine)."""
    b = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
    return b + [SDPBackend.MATH] if "gemma" in type(model).__name__.lower() else b


def load(model_path, quant="none"):
    key = (model_path, quant)
    if key not in _cache:
        kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        if quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        elif quant == "8bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        _cache.clear()
        torch.cuda.empty_cache()
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kw).eval()
        proc = AutoProcessor.from_pretrained(model_path)
        _cache[key] = (model, proc)
    return _cache[key]


@torch.inference_mode()
def generate(model_path, quant, system, content, max_new_tokens=512):
    """content: list of str | PIL.Image, interleaved."""
    model, proc = load(model_path, quant)
    parts = [{"type": "image", "image": c} if not isinstance(c, str) else {"type": "text", "text": c}
             for c in content]
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": parts}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    with sdpa_kernel(backends(model)):  # math fallback OOMs
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True), inputs["input_ids"].shape[1]


def free():
    _cache.clear()
    torch.cuda.empty_cache()
