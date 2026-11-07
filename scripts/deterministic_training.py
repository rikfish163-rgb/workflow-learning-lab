"""Strict, explicitly selected SDPA backend for a bounded numerical diagnostic."""
import os
import sys
from contextlib import contextmanager

BACKENDS = ['MATH', 'FLASH_ATTENTION']
WORKSPACE = ':4096:8'


def configure_deterministic_environment(*, environ=None, modules=None):
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    if 'torch' in modules:
        raise ValueError('configure cuBLAS before importing torch')
    if environ.get('CUBLAS_WORKSPACE_CONFIG', WORKSPACE) != WORKSPACE:
        raise ValueError('conflicting cuBLAS workspace configuration')
    environ['CUBLAS_WORKSPACE_CONFIG'] = WORKSPACE
    return {'CUBLAS_WORKSPACE_CONFIG': WORKSPACE}


def settings():
    import torch

    cuda = torch.backends.cuda
    return {
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'reduced_precision_math_sdp': cuda.fp16_bf16_reduction_math_sdp_allowed(),
        'sdp_enabled': {'MATH': cuda.math_sdp_enabled(),
            'FLASH_ATTENTION': cuda.flash_sdp_enabled(),
            'EFFICIENT_ATTENTION': cuda.mem_efficient_sdp_enabled(),
            'CUDNN_ATTENTION': cuda.cudnn_sdp_enabled()},
    }


@contextmanager
def deterministic_attention(backend):
    if backend not in BACKENDS or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != WORKSPACE:
        raise ValueError('explicit supported backend and cuBLAS workspace required')
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    before = settings()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        with sdpa_kernel(getattr(SDPBackend, backend)):
            yield settings()
    finally:
        torch.use_deterministic_algorithms(before['deterministic_algorithms'], warn_only=before['warn_only'])
        torch.backends.cudnn.deterministic = before['cudnn_deterministic']
        torch.backends.cudnn.benchmark = before['cudnn_benchmark']
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(before['reduced_precision_math_sdp'])
