"""Text-only LoRA preparation for the inference-compatible Qwen3.5 wrapper.

Base weights keep their loaded dtype and storage, including frozen NF4 weights.
In particular, the large BF16 embedding/head/vision parameters are not upcast.
This helper does not load a model or acquire a GPU; a GPU caller needs the
project admission guard and a separately registered memory-tested configuration.
"""
import inspect

from scripts.target_logits import target_logit_inputs

PROJECTIONS = {
    'full_attention': ('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
    'linear_attention': ('linear_attn', ('in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj')),
}


def prepare_hybrid_lora(model, *, rank):
    if type(rank) is not int or rank < 1:
        raise ValueError('positive integer LoRA rank required')
    if (type(model).__name__ != 'Qwen3_5ForConditionalGeneration'
            or model.config.model_type != 'qwen3_5' or hasattr(model, 'peft_config')):
        raise ValueError('unadapted Qwen3.5 conditional model required')
    text = model.config.text_config
    if (text.model_type != 'qwen3_5_text' or len(text.layer_types) != text.num_hidden_layers
            or set(text.layer_types) != set(PROJECTIONS)
            or 'logits_to_keep' not in inspect.signature(model.forward).parameters
            or 'shift_labels' not in inspect.signature(model.loss_function).parameters):
        raise ValueError('hybrid architecture and native shifted loss required')
    import torch
    from peft import LoraConfig, get_peft_model

    modules = dict(model.named_modules())
    targets = []
    for index, kind in enumerate(text.layer_types):
        attention, names = PROJECTIONS[kind]
        for name in names:
            target = f'model.language_model.layers.{index}.{attention}.{name}'
            if target not in modules or not isinstance(modules[target], torch.nn.Linear):
                raise ValueError('missing text projection: ' + target)
            targets.append(target)
    # Non-reentrant checkpointing supports frozen inputs without adding an
    # input-gradient hook. Native norm implementations retain their own casts.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = text.use_cache = False
    adapted = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank,
        target_modules=targets, lora_dropout=0, bias='none', task_type='CAUSAL_LM'),
        autocast_adapter_dtype=True)
    adapted.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    adapted.train()
    return adapted


def target_batch(inputs):
    if 'labels' not in inputs or {'shift_labels', 'logits_to_keep'}.intersection(inputs):
        raise ValueError('full labels and no pre-existing target selection required')
    # Keep full labels: Trainer counts targets over the complete accumulation
    # group, including its final partial group. Only vocabulary projection shrinks.
    return {**inputs, **target_logit_inputs(inputs['labels'])}
