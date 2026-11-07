"""Bound reference delta-rule temporaries without dropping tokens or state gradients.

This is an isolated training-process experiment. Keep the context active through
backward: decoder checkpoint recomputation calls the same module-level function.
Installed Transformers files are never modified. Packed sequences are unsupported.
"""
from contextlib import contextmanager

_ACTIVE = False


def streamed_delta(reference, query, key, value, g, beta, *, window_tokens=512,
                   chunk_size=64, initial_state=None, output_final_state=False,
                   use_qk_l2norm_in_kernel=False, stats=None, **kwargs):
    import torch
    from torch.utils.checkpoint import checkpoint

    if (not isinstance(chunk_size, int) or chunk_size <= 0
            or not isinstance(window_tokens, int) or window_tokens <= 0 or window_tokens % chunk_size):
        raise ValueError('window must be a positive multiple of the reference chunk size')
    if kwargs.get('cu_seqlens') is not None:
        raise ValueError('packed sequences are not supported')
    tensors = (query, key, value, g, beta)
    if (query.ndim != 4 or query.shape != key.shape or value.ndim != 4
            or query.shape[:3] != value.shape[:3] or g.shape != beta.shape
            or g.shape != query.shape[:3] or query.shape[1] == 0
            or len({x.device for x in tensors}) != 1):
        raise ValueError('different or empty delta-rule input shapes/devices')
    if stats is not None:
        stats['calls'] += 1
        stats['max_input_tokens'] = max(stats['max_input_tokens'], query.shape[1])

    def one_window(q, k, v, decay, rate, state):
        if stats is not None:
            stats['reference_calls'] += 1
            stats['max_reference_tokens'] = max(stats['max_reference_tokens'], q.shape[1])
        return reference(q, k, v, decay, rate, chunk_size=chunk_size,
            initial_state=state, output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)

    outputs, state = [], initial_state
    for start in range(0, query.shape[1], window_tokens):
        xs = [x[:, start:start + window_tokens] for x in tensors]
        needs_grad = any(x.requires_grad for x in xs) or (state is not None and state.requires_grad)
        if torch.is_grad_enabled() and needs_grad:
            output, state = checkpoint(one_window, *xs, state, use_reentrant=False)
        else:
            output, state = one_window(*xs, state)
        outputs.append(output)
    return torch.cat(outputs, dim=1), state if output_final_state else None


@contextmanager
def streamed_delta_context(*, window_tokens=512):
    global _ACTIVE
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    if _ACTIVE or not isinstance(window_tokens, int) or window_tokens <= 0 or window_tokens % 64:
        raise ValueError('nested installation or invalid reference window')
    original = qwen.torch_chunk_gated_delta_rule
    stats = {'window_tokens': window_tokens, 'calls': 0, 'reference_calls': 0,
        'max_input_tokens': 0, 'max_reference_tokens': 0}

    def wrapped(*args, **kwargs):
        return streamed_delta(original, *args, window_tokens=window_tokens, stats=stats, **kwargs)

    _ACTIVE = True
    qwen.torch_chunk_gated_delta_rule = wrapped
    try:
        yield stats
    finally:
        qwen.torch_chunk_gated_delta_rule = original
        _ACTIVE = False
