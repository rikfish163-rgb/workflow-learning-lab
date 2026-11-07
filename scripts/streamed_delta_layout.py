"""Experimental windowing variant that restores reference input-gradient strides.

The forward calculation is unchanged. CUDA whole-model numerical equivalence is
not established by this adapter; the previous failed tolerance remains binding.
"""
from contextlib import contextmanager
from functools import lru_cache

from scripts.streamed_delta import streamed_delta, streamed_delta_context


@lru_cache(maxsize=1)
def _layout_function():
    import torch

    class NativeGradientLayout(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, gradient):
            if gradient is None:
                return None
            return gradient.transpose(1, 2).contiguous().transpose(1, 2)

    return NativeGradientLayout


def _aligned(values):
    function = _layout_function()
    return [function.apply(value) if value.requires_grad else value for value in values]


def native_layout_streamed_delta(reference, query, key, value, g, beta, **kwargs):
    return streamed_delta(reference, *_aligned([query, key, value, g, beta]), **kwargs)


@contextmanager
def native_layout_streamed_context(*, window_tokens=512):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    # Reuse the original installation guard, reference binding, stats and cleanup.
    with streamed_delta_context(window_tokens=window_tokens) as stats:
        installed = qwen.torch_chunk_gated_delta_rule

        def wrapped(query, key, value, g, beta, **kwargs):
            return installed(*_aligned([query, key, value, g, beta]), **kwargs)

        qwen.torch_chunk_gated_delta_rule = wrapped
        try:
            yield stats
        finally:
            qwen.torch_chunk_gated_delta_rule = installed
