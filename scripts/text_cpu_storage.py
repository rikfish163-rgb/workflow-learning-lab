"""Keep frozen text lookup and unused vision weights on CPU for text-only training.

This changes persistent base-weight placement and must be registered separately
from experiments that prohibited CPU weight offload. Decoder and output-head
computation stay on the admitted device; embedding lookup is an exact gather.
The installation lasts for this model's lifetime, avoiding a restore allocation
during an OOM unwind. No global runtime or installed-library code is changed.
"""


def prepare_text_cpu_storage(model):
    import torch

    if not hasattr(model, 'peft_config'):
        raise ValueError('prepared text LoRA model required')
    base = model.get_base_model()
    if type(base).__name__ != 'Qwen3_5ForConditionalGeneration':
        raise ValueError('Qwen3.5 conditional model required')
    if hasattr(base, '_wll_text_cpu_storage'):
        raise ValueError('CPU text storage already installed')
    embedding, vision, head = base.get_input_embeddings(), base.model.visual, base.get_output_embeddings()
    if type(embedding) is not torch.nn.Embedding or 'forward' in embedding.__dict__:
        raise ValueError('unmodified embedding lookup required')
    moved = [*embedding.parameters(), *vision.parameters()]
    if any(p.requires_grad for p in moved):
        raise ValueError('embedding and vision must be frozen')
    if embedding.weight.untyped_storage().data_ptr() == head.weight.untyped_storage().data_ptr():
        raise ValueError('tied embedding/output head is unsupported')
    if any(hasattr(m, '_hf_hook') for m in [embedding, *vision.modules()]):
        raise ValueError('existing device dispatch hooks are unsupported')
    execution_device = head.weight.device
    if str(execution_device) not in ('cpu', 'cuda:0'):
        raise ValueError('CPU validation or one admitted logical GPU required')
    if any(p.device != execution_device for p in moved):
        raise ValueError('different initial storage placement')
    stats = {'execution_device': str(execution_device),
        'cpu_modules': ['model.language_model.embed_tokens', 'model.visual'],
        'cpu_parameter_bytes': sum(p.numel() * p.element_size() for p in moved),
        'released_cuda_parameter_bytes': sum(p.numel() * p.element_size() for p in moved
            if p.device.type == 'cuda'), 'text_only': True}
    forbidden = {getattr(base.config, name) for name in
        ['image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id']
        if getattr(base.config, name, None) is not None}
    original = embedding.forward
    embedding.to('cpu')
    vision.to('cpu')

    def lookup(input):
        cpu_ids = input.to('cpu', non_blocking=False)
        if forbidden and bool(torch.isin(cpu_ids, torch.tensor(sorted(forbidden), dtype=cpu_ids.dtype)).any()):
            raise ValueError('text-only CPU lookup rejects vision tokens')
        return original(cpu_ids).to(execution_device, non_blocking=False)

    def text_only(module, args, kwargs):
        if any(kwargs.get(k) is not None for k in
            ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw']):
            raise ValueError('text-only CPU storage rejects multimodal arguments')

    def no_vision(module, args, kwargs):
        raise ValueError('text-only model cannot execute the offloaded vision module')

    embedding.forward = lookup
    base.register_forward_pre_hook(text_only, with_kwargs=True)
    vision.register_forward_pre_hook(no_vision, with_kwargs=True)
    base._wll_text_cpu_storage = stats
    return dict(stats)
