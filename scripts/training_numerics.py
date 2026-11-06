"""Sequential control/repeat/offload updates from identical adapters and RNG.

Only adapter, buffer, and RNG snapshots are retained on CPU. The frozen model
stays on its device; no second pretrained model is loaded. Results quantify
differences rather than silently accepting them under a chosen tolerance.
"""
import hashlib
import json
import math
from contextlib import nullcontext

from scripts.activation_offload import ActivationOffload
from scripts.probe_qwen38_training import train_case

ARMS = ['control', 'repeat_control', 'offload']


def tensor_digest(values):
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_metrics(reference, candidate):
    import torch

    if not reference or reference.keys() != candidate.keys():
        raise ValueError('comparison names differ or are empty')
    rows, ref_squared, delta_squared, equal = [], 0., 0., True
    for name in reference:
        a, b = reference[name].detach().cpu(), candidate[name].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('comparison shape/dtype differs or contains nonfinite values')
        same = torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
        equal &= same
        a, b = a.double(), b.double()
        delta = a - b
        ref_squared += float(a.square().sum())
        delta_squared += float(delta.square().sum())
        rows.append({'name': name, 'max_abs_error': float(delta.abs().max()), 'bitwise_equal': same})
    return {'tensor_count': len(rows), 'bitwise_equal': equal,
        'max_abs_error': max(r['max_abs_error'] for r in rows),
        'global_relative_l2_error': math.sqrt(delta_squared / ref_squared) if ref_squared else
            (0. if not delta_squared else None),
        'reference_l2_norm': math.sqrt(ref_squared), 'difference_l2_norm': math.sqrt(delta_squared),
        'largest_differences': sorted(rows, key=lambda r: r['max_abs_error'], reverse=True)[:5]}


def paired_updates(model, ids, case, progress):
    import torch

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not params or any('lora_' not in n for n in params):
        raise ValueError('only adapter parameters may be trainable')
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}
    buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(ids.device).clone() if ids.device.type == 'cuda' else None
    rngs = {'cpu': cpu_rng, **({'cuda': cuda_rng} if cuda_rng is not None else {})}
    adapter_sha, rng_sha = tensor_digest(initial), tensor_digest(rngs)
    pointer_dtype = [(p, p.data_ptr(), p.dtype) for p in model.parameters() if not p.requires_grad]

    def restore():
        with torch.no_grad():
            for name, parameter in params.items():
                parameter.copy_(initial[name])
                parameter.grad = None
            current = dict(model.named_buffers())
            if current.keys() != buffers.keys():
                raise ValueError('model buffers changed identity')
            for name, value in buffers.items():
                current[name].copy_(value)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, ids.device)

    outcomes, comparisons, reference_gradients, reference_parameters = [], {}, None, None
    try:
        for arm in ARMS:
            restore()
            if tensor_digest(params) != adapter_sha:
                raise ValueError('adapter restore differs')
            current_rngs = {'cpu': torch.get_rng_state()}
            if cuda_rng is not None:
                current_rngs['cuda'] = torch.cuda.get_rng_state(ids.device)
            arm_rng_sha = tensor_digest(current_rngs)
            if arm_rng_sha != rng_sha:
                raise ValueError('RNG restore differs')
            captured = {n: [] for n in params}
            hooks = [p.register_hook(lambda grad, name=n, captured=captured:
                captured[name].append(grad.detach().cpu().clone())) for n, p in params.items()]
            offload = ActivationOffload(model) if arm == 'offload' else None
            try:
                with offload if offload is not None else nullcontext():
                    result = train_case(model, ids, case,
                        lambda phase, details, arm=arm: progress(arm, phase, details))
            finally:
                for handle in hooks:
                    handle.remove()
            if any(len(values) != 2 for values in captured.values()):
                raise ValueError('incomplete gradient collection')
            if any(p.data_ptr() != ptr or p.dtype != dtype or p.grad is not None for p, ptr, dtype in pointer_dtype):
                raise ValueError('frozen storage/dtype/gradient changed')
            gradients = [{n: values[i] for n, values in captured.items()} for i in range(2)]
            final = {n: p.detach().cpu().clone() for n, p in params.items()}
            result.update(arm=arm, initial_adapter_sha256=adapter_sha, initial_rng_sha256=arm_rng_sha,
                final_adapter_sha256=tensor_digest(final),
                gradient_sha256_by_step=[tensor_digest(g) for g in gradients],
                offload=dict(offload.stats) if offload else None)
            if arm == 'control':
                reference_gradients, reference_parameters = gradients, final
            else:
                comparisons[arm] = {
                    'losses_bitwise_equal': result['losses'] == outcomes[0]['losses'],
                    'loss_abs_errors': [abs(a-b) for a, b in zip(outcomes[0]['losses'], result['losses'], strict=True)],
                    'gradients': [tensor_metrics(a, b) for a, b in zip(reference_gradients, gradients, strict=True)],
                    'parameters': tensor_metrics(reference_parameters, final)}
            outcomes.append(result)
        return {'arms': outcomes, 'comparisons': comparisons,
            'input_sha256': tensor_digest({'input_ids': ids}), 'case': case}
    finally:
        restore()
