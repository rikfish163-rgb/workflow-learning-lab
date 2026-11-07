"""Replace Qwen3.5 decoder checkpoints with attention/MLP checkpoints temporarily.

Intended for training inside ActivationOffload so each smaller boundary can be
saved on CPU. Computation shapes, parameters, and attention algorithms are kept.
CUDA long-context feasibility must be measured independently.
"""
from contextlib import contextmanager


@contextmanager
def sublayer_checkpoint_context(model):
    import torch
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    marker = '_wll_sublayer_checkpoint_active'
    if hasattr(model, marker):
        raise ValueError('sublayer checkpoint context already active')
    layers = [m for m in model.modules() if isinstance(m, Qwen3_5DecoderLayer)]
    if not layers or not model.training or any(not m.gradient_checkpointing for m in layers):
        raise ValueError('training Qwen3.5 decoder with existing layer checkpoints required')
    if model.config.use_cache or model.config.text_config.use_cache:
        raise ValueError('cache must be disabled')
    modules = []
    for layer in layers:
        attention = layer.linear_attn if layer.block_type == 'linear_attention' else layer.self_attn
        modules.extend([attention, layer.mlp])
    if len({id(m) for m in modules}) != len(modules):
        raise ValueError('shared sublayers are unsupported')
    stats = {'decoder_layers': len(layers), 'wrapped_sublayers': len(modules),
        'calls': 0, 'evaluations': 0}
    missing = object()
    restore = []

    def wrapped(forward):
        def evaluated(*args, **kwargs):
            stats['evaluations'] += 1
            return forward(*args, **kwargs)

        def call(*args, **kwargs):
            if any(kwargs.get(key) is not None for key in ['past_key_values', 'cache_params']):
                raise ValueError('mutable cache is incompatible with recomputation')
            if kwargs.get('use_cache') or not model.training or not torch.is_grad_enabled():
                raise ValueError('training with gradients and no cache required')
            stats['calls'] += 1
            # This runtime saves only positional checkpoint inputs through saved_tensors_hooks.
            # Flatten kwargs and nested structures too, avoiding retained GPU tensor closures.
            flat, spec = tree_flatten((args, kwargs))

            def run(*values):
                positional, keywords = tree_unflatten(list(values), spec)
                return evaluated(*positional, **keywords)

            return checkpoint(run, *flat, use_reentrant=False, preserve_rng_state=True)

        return call

    def replace(module, name, value):
        restore.append((module, name, module.__dict__.get(name, missing)))
        setattr(module, name, value)

    try:
        replace(model, marker, True)
        for layer in layers:
            replace(layer, 'gradient_checkpointing', False)
        for module in modules:
            replace(module, 'forward', wrapped(module.forward))
        yield stats
    finally:
        for module, name, previous in reversed(restore):
            if previous is missing:
                delattr(module, name)
            else:
                setattr(module, name, previous)
