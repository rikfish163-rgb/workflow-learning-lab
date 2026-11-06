"""Pinned NF4 loading recipe, shared by CPU checks and the separate GPU probe."""

SKIP_MODULES = ["model.visual", "lm_head"]


def load_nf4(path, device):
    if device not in ("cpu", "cuda:0"):
        raise ValueError("only an explicit CPU check or one admitted logical GPU is allowed")
    import torch
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=SKIP_MODULES,
    )
    return AutoModelForImageTextToText.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": device}, quantization_config=config,
    ).eval()


def inspect_nf4(model, device):
    import bitsandbytes as bnb

    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    if not layers or not getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("model was not quantized")
    if any("visual" in n or n == "lm_head" for n, _ in layers):
        raise ValueError("protected vision/head modules were quantized")
    if not any("linear_attn.in_proj_qkv" in n for n, _ in layers):
        raise ValueError("linear-attention projections were not quantized")
    if not any("self_attn.q_proj" in n for n, _ in layers):
        raise ValueError("full-attention projections were not quantized")
    if any(str(p.device) != device for p in model.parameters()):
        raise ValueError("unexpected parameter device or offload")
    for name, layer in layers:
        state = layer.weight.quant_state
        if state is None or state.quant_type != "nf4" or not state.nested:
            raise ValueError(f"wrong quantization state: {name}")
    return {"linear4bit_layers": len(layers), "quantized_module_names": [n for n, _ in layers],
            "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
            "quantization_config": model.config.quantization_config.to_dict()}
