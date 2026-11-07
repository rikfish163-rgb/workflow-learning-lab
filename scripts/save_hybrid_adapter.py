"""Native PEFT export of clean, prepared Qwen3.5 text LoRA, with exact verification.

Caller owns training completion, provenance, GPU admission and output registration.
This helper loads no model and starts no GPU work. Failed saves retain their
partial directory for inspection; an existing path is never overwritten/resumed.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

from scripts.qwen38_training import PROJECTIONS
from scripts.training_numerics import tensor_digest


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_hybrid_adapter(model, output_dir, *, expected_base_model_name_or_path):
    """Return completion evidence only after native files/config/tensors verify.

    source_adapter_sha256 uses full named trainable parameters, matching the
    train_groups final digest. PEFT-state digests use native serialized names.
    Positive rank is dynamic; the prepared alpha=2*rank recipe is required.
    """
    path = Path(output_dir)
    if os.path.lexists(path):
        raise FileExistsError('adapter output path already exists; no overwrite or resume')
    _require(path.is_absolute() and path.parent.is_dir(), 'absolute output path with existing parent required')
    _require(isinstance(expected_base_model_name_or_path, str) and expected_base_model_name_or_path.strip(),
             'explicit expected base model name/path required')
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    _require(isinstance(model, PeftModel) and set(model.peft_config) == {'default'}
             and model.active_adapters == ['default']
             and type(model.get_base_model()).__name__ == 'Qwen3_5ForConditionalGeneration',
             'single active prepared Qwen3.5 text LoRA required')
    config = model.peft_config['default']
    text = model.get_base_model().config.text_config
    _require(len(text.layer_types) == text.num_hidden_layers and set(text.layer_types) == set(PROJECTIONS),
             'hybrid text attention architecture required')
    targets = {f'model.language_model.layers.{index}.{PROJECTIONS[kind][0]}.{projection}'
               for index, kind in enumerate(text.layer_types) for projection in PROJECTIONS[kind][1]}
    configured_targets = config.target_modules
    _require(isinstance(configured_targets, set)
             and all(isinstance(name, str) and name for name in configured_targets),
             'explicit native LoRA target set required')
    resolved_targets = {name for name, _ in model.get_base_model().named_modules()
                        if any(name == target or name.endswith('.' + target) for target in configured_targets)}
    _require(type(config) is LoraConfig and config.peft_type == 'LORA' and config.task_type == 'CAUSAL_LM'
             and config.base_model_name_or_path == expected_base_model_name_or_path
             and type(config.r) is int and config.r > 0 and config.lora_alpha == 2 * config.r
             and config.lora_dropout == 0 and config.bias == 'none' and resolved_targets == targets
             and not config.modules_to_save and not config.rank_pattern and not config.alpha_pattern
             and not config.use_dora and not config.use_rslora and not config.lora_bias,
             'adapter configuration differs from prepared text LoRA recipe')
    expected_names = {f'base_model.model.{target}.lora_{kind}.default.weight'
                      for target in targets for kind in ['A', 'B']}
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    _require(set(params) == expected_names and all(p.grad is None for p in model.parameters())
             and len({str(p.device) for p in params.values()}) == 1
             and all(str(p.device) in ['cpu', 'cuda:0'] and p.dtype == torch.float32
                     and bool(torch.isfinite(p).all()) for p in params.values()),
             'clean finite FP32 text LoRA tensors only; no dirty/frozen gradients')
    for target in targets:
        module = model.get_submodule(f'base_model.model.{target}')
        _require(not module.merged and not module.disable_adapters,
                 'merged or disabled adapters cannot be exported as this recipe')
        _require(tuple(module.lora_A['default'].weight.shape) == (config.r, module.in_features)
                 and tuple(module.lora_B['default'].weight.shape) == (module.out_features, config.r),
                 'live LoRA tensor shapes differ from configured rank')
    source = {name: p.detach().cpu().clone() for name, p in params.items()}
    source_sha = tensor_digest(source)
    state = get_peft_model_state_dict(model, adapter_name='default', save_embedding_layers=False)
    expected_state = {name.replace('.default.', '.'): value for name, value in source.items()}
    _require(state.keys() == expected_state.keys()
             and all(value.shape == expected_state[name].shape and value.dtype == expected_state[name].dtype
                     and torch.equal(value.detach().cpu(), expected_state[name]) for name, value in state.items()),
             'native PEFT state contains missing/extra/non-LoRA tensors')
    state_sha = tensor_digest(expected_state)
    before_config = copy.deepcopy(config.to_dict())
    # mkdir is exclusive even if another writer creates the path after preflight.
    path.mkdir(exist_ok=False)
    try:
        model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False,
                              selected_adapters=['default'])
    finally:
        # Native PEFT writes inference_mode=True but restores it only on success.
        config.inference_mode = before_config['inference_mode']
    _require(config.to_dict() == before_config, 'native export changed the live adapter configuration')
    entries = {entry.name: entry for entry in path.iterdir()}
    required = {'adapter_model.safetensors', 'adapter_config.json'}
    _require(required <= entries.keys() <= required | {'README.md'}
             and all(not entry.is_symlink() and entry.is_file() for entry in entries.values()),
             'unexpected export files; only native adapter/config/model-card files allowed')
    saved_config = json.loads(entries['adapter_config.json'].read_text())
    config_fields = {'base_model_name_or_path': expected_base_model_name_or_path,
                     'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': config.r,
                     'lora_alpha': config.lora_alpha, 'lora_dropout': 0, 'bias': 'none',
                     'inference_mode': True}
    _require(all(saved_config.get(key) == value for key, value in config_fields.items())
             and saved_config.get('inference_mode') is True
             and isinstance(saved_config.get('target_modules'), list)
             and set(saved_config['target_modules']) == configured_targets
             and not saved_config.get('modules_to_save') and not saved_config.get('rank_pattern')
             and not saved_config.get('alpha_pattern') and not saved_config.get('use_dora')
             and not saved_config.get('use_rslora') and not saved_config.get('lora_bias'),
             'saved adapter configuration differs from native inference export intent')
    saved = load_file(str(entries['adapter_model.safetensors']), device='cpu')
    _require(saved.keys() == expected_state.keys()
             and all(value.shape == expected_state[name].shape and value.dtype == expected_state[name].dtype
                     and torch.equal(value, expected_state[name]) for name, value in saved.items()),
             'saved safetensors names/shapes/dtypes/values differ from live native PEFT state')
    saved_sha = tensor_digest(saved)
    _require(saved_sha == state_sha and tensor_digest(params) == source_sha
             and all(p.grad is None for p in model.parameters()), 'adapter values/gradients changed during export')
    files = {name: {'sha256': _file_digest(entry), 'bytes': entry.stat().st_size}
             for name, entry in sorted(entries.items())}
    return {'status': 'complete', 'complete': True, 'adapter_saved': True, 'output_directory': str(path),
            'format': 'safetensors', 'tensor_count': len(saved),
            'total_parameter_count': sum(value.numel() for value in saved.values()),
            'adapter_tensor_bytes': sum(value.numel() * value.element_size() for value in saved.values()),
            'adapter_dtypes': sorted({str(value.dtype) for value in saved.values()}),
            'source_adapter_sha256': source_sha, 'source_peft_state_sha256': state_sha,
            'saved_peft_state_sha256': saved_sha, 'live_config_unchanged': True,
            'config': config_fields | {'target_module_count': len(targets)},
            'files': files, 'total_file_bytes': sum(item['bytes'] for item in files.values())}
