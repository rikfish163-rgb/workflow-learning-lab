"""Registered saved-activation offload probe; same full synthetic inputs and 20GiB cap."""
import argparse
import gc
import importlib.metadata
import json
import os
import resource
import sys
import time
from pathlib import Path

from scripts.activation_offload import ActivationOffload
from scripts.probe_allocator import allocator_details, configure_allocator, memory
from scripts.probe_chunked_prefill import CAP, PYTHON, ROOT, digest, require
from scripts.probe_qwen38 import MODEL, RUNTIME, WEIGHT_PROOF, verify_model_provenance
from scripts.probe_qwen38_training import train_case
from wll.storage import atomic_json

ID = 'qwen38-27b-training-offload-001'
PLAN = 'artifacts/qwen38-27b-training-offload-plan-001.json'
DESIGN = 'artifacts/qwen38-27b-training-offload-design-001.json'
FIXTURE = 'artifacts/activation-offload-tiny-model-001'
DOWNLOAD = 'artifacts/qwen38-27b-download-plan-001.json'
CASES = [{'input_tokens': n, 'target_tokens': t}
    for n, t in [(256, 32), (2048, 128), (8192, 256), (16384, 512)]]


def preflight(sha):
    require(digest(ROOT / PLAN) == sha, 'offload registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['design'] == DESIGN
        and plan['design_sha256'] == digest(ROOT / DESIGN), 'unregistered offload probe')
    required = {'scripts/activation_offload.py', 'scripts/probe_qwen38_training_offload.py',
        'scripts/run_qwen38_training_offload_probe.py', 'scripts/probe_qwen38_training.py',
        'scripts/qwen38_training.py', 'scripts/nf4_loader.py', 'scripts/target_logits.py',
        'scripts/gpu_idle_watch.py', 'wll/gpu_policy.py', 'wll/gpu_idle.py', 'wll/storage.py'}
    require(required <= plan['source_sha256'].keys(), 'missing executable binding')
    require({RUNTIME, WEIGHT_PROOF, DOWNLOAD, 'config/gpu-policy.json',
        'artifacts/activation-offload-CPU-verification-001.json'} <= plan['input_sha256'].keys(),
        'missing offload provenance')
    for field in ('source_sha256', 'input_sha256'):
        for name, expected in plan[field].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and digest(ROOT / name) == expected, 'bound offload input changed: ' + name)
    audit = json.loads((ROOT / RUNTIME).read_text())
    require(sys.prefix == audit['runtime']['python_prefix'] == str(Path(PYTHON).parent.parent),
        'different Python environment')
    for name, expected in {**audit['runtime_sha256'], **plan['runtime_sha256']}.items():
        require(digest(name) == expected, 'offload runtime changed: ' + name)
    for name, version in audit['runtime']['packages'].items():
        require(importlib.metadata.version(name) == version, 'different package: ' + name)
    policy = json.loads((ROOT / 'config/gpu-policy.json').read_text())
    require(policy['idle_seconds'] == 180 and set(policy['allowed_gpus']) == {'8', '9'},
        'updated single GPU idle policy required')
    verify_model_provenance(MODEL, json.loads((ROOT / DOWNLOAD).read_text()),
        json.loads((ROOT / WEIGHT_PROOF).read_text()))
    design = json.loads((ROOT / DESIGN).read_text())
    require(design['cases'] == CASES and design['rank'] == 8 and design['steps_per_case'] == 2
        and design['learning_rate'] == 1e-4 and design['optimizer'] == 'AdamW_foreach_false'
        and design['weight_decay'] == 0 and design['max_grad_norm'] == 1.0
        and design['allocator_cap_bytes'] == CAP and design['max_gpu_job_seconds'] == 3600
        and design['automatic_retry'] is False and design['automatic_truncation'] is False
        and design['maximum_attempts'] == 1 and design['save_adapter'] is False
        and design['seed'] == 17 and design['input_seed_text'] == 'This is a synthetic training memory probe. '
        and design['offload'] == 'synchronous_activation_only'
        and design['cuda_numerical_check'] == {'input_tokens': 129, 'target_tokens': 8, 'rank': 4,
            'optimizer_steps': 2, 'rtol': 0, 'atol': 0}, 'offload recipe differs')
    return design


def compare_updates(model, ids, case):
    """Paired tiny-model check on the caller's device, including every LoRA gradient."""
    import copy
    from contextlib import nullcontext

    import torch

    candidate = copy.deepcopy(model)
    gradients, outcomes = [], []
    offload = ActivationOffload(candidate)
    for current, context in [(model, nullcontext()), (candidate, offload)]:
        captured = {}
        handles = []
        for name, parameter in current.named_parameters():
            if parameter.requires_grad:
                captured[name] = []
                handles.append(parameter.register_hook(
                    lambda grad, name=name, captured=captured: captured[name].append(grad.detach().cpu().clone())))
        try:
            with context:
                outcomes.append(train_case(current, ids, case, lambda *_: None))
        finally:
            for handle in handles:
                handle.remove()
        gradients.append(captured)
    require(outcomes[0]['losses'] == outcomes[1]['losses'], 'offload changed tiny losses')
    compared = 0
    for name, values in gradients[0].items():
        require(len(values) == len(gradients[1][name]) == 2, 'missing paired gradient')
        for a, b in zip(values, gradients[1][name], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            compared += 1
    for (name, a), (other, b) in zip(model.named_parameters(), candidate.named_parameters(), strict=True):
        require(name == other, 'parameter names differ')
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    require(offload.stats['copied_bytes'] > 0, 'activation copies missing')
    return {'input_tokens': case['input_tokens'], 'optimizer_steps': 2, 'compared_gradients': compared,
        'losses': outcomes[0]['losses'], 'max_loss_gradient_parameter_error': 0,
        'offload': dict(offload.stats), 'device': str(ids.device)}


def assess_complete(cases):
    require(len(cases) == len(CASES), 'all registered training contexts required')
    for result, expected in zip(cases, CASES, strict=True):
        require(all(result.get(k) == v for k, v in expected.items())
            and result['optimizer_steps'] == 2 and result['supervised_tokens'] == 2 * expected['target_tokens']
            and result['finite_gradients'] is result['finite_parameters'] is True
            and result['changed_adapter_tensors'] > 0
            and result['changed_attention_types'] == ['linear_attn', 'self_attn']
            and result['peak_reserved_bytes'] <= CAP
            and result['offload']['gpu_to_cpu_bytes'] > 0 and result['offload']['cpu_to_gpu_bytes'] > 0,
            'incomplete or invalid offloaded training measurement')
    return {'contexts_passed': [case['input_tokens'] for case in cases],
        'synthetic_training_probe_complete': True, 'task_learning_demonstrated': False,
        'scope': 'Synthetic lengths/target counts only; no real trajectory fit or learning claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    allocator = configure_allocator('expandable')
    design = preflight(args.plan_sha256)
    if args.check:
        print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'id': ID, 'cases': CASES}))
        return
    path = ROOT / f'artifacts/{ID}.json'
    require(not path.exists(), 'attempt exists; no retry')
    from wll.gpu_policy import acquire_gpu, verify_cuda

    state = {'id': ID, 'phase': 'waiting_admission', 'pid': os.getpid(),
        'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
        'started_epoch': time.time(), 'plan_sha256': args.plan_sha256, 'allocator_environment': allocator,
        'cases': [], 'unattempted_cases': CASES, 'attempted_measurements': 0,
        'training': True, 'synthetic_only': True, 'benchmark_evaluation': False,
        'adapter_saved': False, 'task_learning_demonstrated': False}
    atomic_json(path, state)
    try:
        with acquire_gpu() as admission:
            import torch
            from transformers import AutoTokenizer

            from scripts.nf4_loader import inspect_nf4, load_nf4
            from scripts.qwen38_training import prepare_hybrid_lora

            verify_cuda(torch)
            torch.set_num_threads(2)
            torch.manual_seed(17)
            state.update(phase='cuda_numerical_check', gpu=admission)
            atomic_json(path, state)
            small = prepare_hybrid_lora(load_nf4(ROOT / FIXTURE, 'cuda:0'), rank=4)
            with torch.no_grad():
                for name, parameter in small.named_parameters():
                    if 'lora_B' in name:
                        parameter.normal_(std=.02)
            small_ids = torch.randint(3, 50, (1, 129), device='cuda:0')
            state['cuda_numerical_check'] = compare_updates(small, small_ids,
                {'input_tokens': 129, 'target_tokens': 8})
            del small, small_ids
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(17)
            state['phase'] = 'loading'
            atomic_json(path, state)
            start = time.monotonic()
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
            model = load_nf4(MODEL, 'cuda:0')
            state['quantization_actual'] = inspect_nf4(model, 'cuda:0')
            require(state['quantization_actual']['linear4bit_layers'] == 496, 'different quantized model')
            base = [(p, p.data_ptr(), p.dtype) for p in model.parameters()]
            model = prepare_hybrid_lora(model, rank=8)
            require(all(p.data_ptr() == pointer and p.dtype == dtype and not p.requires_grad
                for p, pointer, dtype in base), 'base storage or dtype changed')
            adapters = [p for p in model.parameters() if p.requires_grad]
            require(len(adapters) == 608 and all(p.dtype == torch.float32 for p in adapters),
                'unexpected trainable adapter set')
            torch.cuda.synchronize()
            state.update(phase='probing', load_and_prepare_seconds=time.monotonic() - start,
                base_storage_dtype_preserved=True, trainable_adapter_tensors=len(adapters),
                trainable_adapter_parameters=sum(p.numel() for p in adapters), loading_memory=memory(torch),
                actual_allocator=allocator_details(torch, 'expandable', require_segments=True))
            require(state['loading_memory']['reserved_bytes.all.peak'] <= CAP, 'loading exceeds allocator cap')
            atomic_json(path, state)
            seed = tokenizer(design['input_seed_text'], add_special_tokens=False)['input_ids']
            require(seed, 'empty synthetic seed')
            for index, case in enumerate(CASES):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                state.update(current_case=case, attempted_measurements=index + 1, unattempted_cases=CASES[index + 1:])

                def progress(phase, details):
                    state.update(current_operation=phase, current_step=details, checked_epoch=time.time(),
                        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                    atomic_json(path, state)

                ids = torch.tensor([(seed * (case['input_tokens'] // len(seed) + 1))[:case['input_tokens']]],
                    device='cuda:0', dtype=torch.long)
                offload = ActivationOffload(model)
                try:
                    with offload:
                        result = train_case(model, ids, case, progress)
                finally:
                    state['current_offload_counters'] = dict(offload.stats)
                    atomic_json(path, state)
                result['offload'] = dict(offload.stats)
                result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                torch.cuda.synchronize()
                result.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), after_update_memory=memory(torch))
                require(result['peak_reserved_bytes'] <= CAP, 'training exceeds allocator cap')
                state['cases'].append(result)
                atomic_json(path, state)
                del ids, result
            state.update(phase='complete', result=assess_complete(state['cases']), finished_epoch=time.time())
            atomic_json(path, state)
    except BaseException as exc:
        state.update(phase='failed', error_type=type(exc).__name__, error=str(exc)[:2000],
            finished_epoch=time.time())
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            try:
                state['failure_memory'] = memory(torch)
            except Exception as secondary:
                state['failure_memory_error'] = type(secondary).__name__
        state['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        atomic_json(path, state)
        raise


if __name__ == '__main__':
    main()
