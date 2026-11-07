"""Verified native PEFT inference reload on an already loaded Qwen3.5 NF4 base.

No model loading, admission, HTTP server or experiment authorization here.
The caller must bind the completed export record and base provenance in its
separate service plan and measure full-model inference memory before evaluation.
"""
import json
from pathlib import Path

from scripts.qwen38_training import PROJECTIONS
from scripts.save_hybrid_adapter import _file_digest, _require
from scripts.training_numerics import tensor_digest


def load_hybrid_adapter(base, adapter_dir, *, saved_record, expected_base_model_name_or_path):
    path, saved = Path(adapter_dir), saved_record
    _require(path.is_absolute() and not path.is_symlink() and path.is_dir()
        and saved.get('output_directory') == str(path) and saved.get('status') == 'complete'
        and saved.get('complete') is saved.get('adapter_saved') is True
        and saved.get('format') == 'safetensors', 'completed local native adapter export required')
    files = saved.get('files', {})
    required = {'adapter_model.safetensors', 'adapter_config.json'}
    _require(required <= files.keys() <= required | {'README.md'}
        and {p.name for p in path.iterdir()} == set(files), 'adapter inventory differs')
    for name, expected in files.items():
        p = path / name
        _require(p.is_file() and not p.is_symlink() and _file_digest(p) == expected['sha256']
            and p.stat().st_size == expected['bytes'], 'adapter file differs: ' + name)
    _require(type(base).__name__ == 'Qwen3_5ForConditionalGeneration'
        and base.config.model_type == 'qwen3_5' and not hasattr(base, 'peft_config')
        and getattr(base, 'is_loaded_in_4bit', False) is True
        and base.name_or_path == expected_base_model_name_or_path, 'unadapted matching NF4 base required')
    text = base.config.text_config
    _require(len(text.layer_types) == text.num_hidden_layers and set(text.layer_types) == set(PROJECTIONS),
        'hybrid text architecture required')
    config = json.loads((path / 'adapter_config.json').read_text())
    rank = config.get('r')
    expected_config = {'base_model_name_or_path': expected_base_model_name_or_path,
        'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': rank, 'lora_alpha': 2 * rank if type(rank) is int else None,
        'lora_dropout': 0, 'bias': 'none', 'inference_mode': True}
    _require(type(rank) is int and rank > 0 and config.get('inference_mode') is True
        and all(config.get(k) == v == saved['config'].get(k) for k, v in expected_config.items())
        and not any(config.get(k) for k in ['modules_to_save', 'rank_pattern', 'alpha_pattern',
            'use_dora', 'use_rslora', 'lora_bias']), 'adapter inference config differs')
    targets = {f'model.language_model.layers.{i}.{PROJECTIONS[kind][0]}.{projection}'
        for i, kind in enumerate(text.layer_types) for projection in PROJECTIONS[kind][1]}
    configured = config.get('target_modules')
    _require(isinstance(configured, list) and configured
        and all(isinstance(n, str) and n for n in configured), 'explicit attention target set required')
    resolved = {name for name, _ in base.named_modules()
        if any(name == target or name.endswith('.' + target) for target in configured)}
    _require(resolved == targets and saved['config']['target_module_count'] == len(targets),
        'adapter attention coverage differs')
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    tensors = load_file(str(path / 'adapter_model.safetensors'), device='cpu')
    expected_names = {f'base_model.model.{name}.lora_{kind}.weight' for name in targets for kind in ['A', 'B']}
    _require(tensors.keys() == expected_names and len(tensors) == saved['tensor_count']
        and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in tensors.values()),
        'adapter tensor names/dtype/finite values differ')
    for name in targets:
        module = base.get_submodule(name)
        _require(tuple(tensors[f'base_model.model.{name}.lora_A.weight'].shape) == (rank, module.in_features)
            and tuple(tensors[f'base_model.model.{name}.lora_B.weight'].shape) == (module.out_features, rank),
            'adapter tensor shape differs')
    _require(sum(v.numel() for v in tensors.values()) == saved['total_parameter_count']
        and sum(v.numel() * v.element_size() for v in tensors.values()) == saved['adapter_tensor_bytes']
        and tensor_digest(tensors) == saved['source_peft_state_sha256'] == saved['saved_peft_state_sha256']
        and tensor_digest({n.replace('.lora_A.', '.lora_A.default.').replace('.lora_B.', '.lora_B.default.'): v
            for n, v in tensors.items()}) == saved['source_adapter_sha256'], 'adapter saved-state digest differs')
    before = {n: (id(p), p.data_ptr(), p.dtype, p.device) for n, p in base.named_parameters()}
    _require(all(str(p.device) in ('cpu', 'cuda:0') for p in base.parameters()), 'unsupported base placement')
    # Native PEFT autocast=True preserves the exported FP32 adapter. False
    # loads into BF16 on this NF4 base and rounds values (CPU regression).
    model = PeftModel.from_pretrained(base, path, adapter_name='default', is_trainable=False,
        autocast_adapter_dtype=True, local_files_only=True)
    model.eval()
    model.config.use_cache = model.config.text_config.use_cache = True
    actual = get_peft_model_state_dict(model, adapter_name='default', save_embedding_layers=False)
    _require(actual.keys() == tensors.keys()
        and all(v.dtype == tensors[n].dtype and torch.equal(v.detach().cpu(), tensors[n]) for n, v in actual.items())
        and model.active_adapters == ['default'] and not model.is_gradient_checkpointing
        and all(not p.requires_grad for p in model.parameters()), 'native inference reload differs')
    after = {n.replace('.base_layer.', '.'): (id(p), p.data_ptr(), p.dtype, p.device)
        for n, p in base.named_parameters() if '.lora_' not in n}
    _require(before == after, 'reload changed NF4/base parameter storage')
    for name in targets:
        module = base.get_submodule(name)
        _require(not module.merged and not module.disable_adapters, 'inference adapter disabled or merged')
    for name, expected in files.items():
        _require(_file_digest(path / name) == expected['sha256'], 'adapter changed during reload')
    return model, {'status': 'verified', 'adapter_state_sha256': tensor_digest(actual),
        'source_adapter_sha256': saved['source_adapter_sha256'], 'tensor_count': len(actual),
        'base_parameter_storage_unchanged': True, 'all_parameters_frozen': True,
        'active_adapters': ['default'], 'merged': False, 'use_cache': True,
        'adapter_dtypes': sorted({str(v.dtype) for v in actual.values()}),
        'files': files, 'scope': 'Native adapter reload only; no full27B memory or task-learning claim.'}
