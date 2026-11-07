"""Frozen output-head CPU storage with native checkpointed projection.

The decoder, vocabulary projection and input gradient still compute on the
admitted device. Only persistent head storage moves. Non-reentrant checkpointing
recreates its temporary device weight for native Linear backward, then releases
it before decoder backward. CPU validation copies the weight to exercise lifetime
semantics. No custom derivative, global patch, CUDA initialization or GPU admission
is performed here; a GPU caller requires a separately registered experiment.
"""


def prepare_output_head_cpu_storage(model):
    import torch
    from torch.utils.checkpoint import checkpoint

    if not hasattr(model, 'peft_config'):
        raise ValueError('prepared text LoRA model required')
    base = model.get_base_model()
    if type(base).__name__ != 'Qwen3_5ForConditionalGeneration':
        raise ValueError('Qwen3.5 conditional model required')
    if not hasattr(base, '_wll_text_cpu_storage'):
        raise ValueError('text CPU storage must be installed first')
    if hasattr(base, '_wll_output_head_cpu_storage'):
        raise ValueError('output-head CPU storage already installed')
    head = base.get_output_embeddings()
    if (type(head) is not torch.nn.Linear or head.bias is not None
            or 'forward' in head.__dict__ or hasattr(head, '_hf_hook')
            or getattr(head, '_compiled_call_impl', None) is not None):
        raise ValueError('unmodified bias-free native Linear head required')
    if head.weight.requires_grad or not head.weight.is_floating_point():
        raise ValueError('unquantized frozen head required')
    embedding = base.get_input_embeddings().weight
    if (embedding.device == head.weight.device
            and embedding.untyped_storage().data_ptr() == head.weight.untyped_storage().data_ptr()):
        raise ValueError('tied embedding/output head is unsupported')
    execution_device = head.weight.device
    if (str(execution_device) not in ('cpu', 'cuda:0')
            or str(execution_device) != base._wll_text_cpu_storage['execution_device']
            or any(p.device != execution_device for p in model.parameters() if p.requires_grad)):
        raise ValueError('CPU validation or single admitted execution device required')
    size = head.weight.numel() * head.weight.element_size()
    stats = {'execution_device': str(execution_device), 'cpu_modules': ['lm_head'],
        'cpu_parameter_bytes': size,
        'released_cuda_parameter_bytes': size if execution_device.type == 'cuda' else 0,
        'dtype': str(head.weight.dtype), 'shape': list(head.weight.shape),
        'projection': 'native_linear_nonreentrant_checkpoint', 'custom_backward': False}
    head.to('cpu')

    def project(value):
        if (head.weight.device.type != 'cpu' or head.weight.requires_grad
                or value.device != execution_device):
            raise ValueError('frozen CPU head and original execution device required')
        temporary = head.weight.to(value.device, copy=value.device.type == 'cpu', non_blocking=False)
        return torch.nn.functional.linear(value, temporary)

    def forward(value):
        if torch.is_grad_enabled() and value.requires_grad:
            # Projection is deterministic and consumes no random numbers.
            return checkpoint(project, value, use_reentrant=False, preserve_rng_state=False)
        return project(value)

    head.forward = forward
    base._wll_output_head_cpu_storage = stats
    return dict(stats)
