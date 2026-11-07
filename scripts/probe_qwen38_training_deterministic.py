"""Registered strict MATH/FLASH diagnostics, each with identical three-arm starts."""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

from scripts.deterministic_training import (
    BACKENDS,
    configure_deterministic_environment,
    deterministic_attention,
)
from scripts.probe_allocator import configure_allocator, memory
from scripts.probe_chunked_prefill import CAP, ROOT, digest, require
from scripts.probe_qwen38 import MODEL
from scripts.probe_qwen38_training_numerics import CASE
from scripts.probe_qwen38_training_numerics import assess_complete as assess_comparison
from scripts.probe_qwen38_training_numerics import preflight as parent_preflight
from scripts.training_numerics import ARMS, paired_updates
from wll.storage import atomic_json

ID = 'qwen38-27b-training-deterministic-001'
PLAN = 'artifacts/qwen38-27b-training-deterministic-plan-001.json'
PARENT_SHA = 'e3e8bbb42b8c69c6366a3c34ddbaf7b62aa0c7e743428678d21c3e7fbebe8ee8'


def preflight(sha):
    require(digest(ROOT / PLAN) == sha, 'deterministic registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['case'] == CASE
        and plan['arms'] == ARMS and plan['backends'] == BACKENDS
        and plan['steps_per_arm'] == 2 and plan['rank'] == 8
        and plan['max_gpu_job_seconds'] == 600 and plan['maximum_attempts'] == 1
        and plan['automatic_retry'] is False and plan['allocator_cap_bytes'] == CAP
        and plan['parent_plan_sha256'] == PARENT_SHA, 'different deterministic design')
    require({'scripts/deterministic_training.py', 'scripts/probe_qwen38_training_deterministic.py',
        'scripts/run_qwen38_training_deterministic_probe.py'} <= plan['source_sha256'].keys(),
        'missing deterministic source binding')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in plan[field].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and digest(ROOT / name) == expected, 'bound deterministic file changed: ' + name)
    for name, expected in plan['runtime_sha256'].items():
        require(Path(name).is_absolute() and digest(name) == expected, 'deterministic runtime changed: ' + name)
    return parent_preflight(PARENT_SHA)


def assess_complete(rows):
    require([r['backend'] for r in rows] == BACKENDS, 'both ordered backends required')
    results = {}
    inputs, adapters, rngs = set(), set(), set()
    for row in rows:
        backend, actual, comparison = row['backend'], row['settings'], row['comparison']
        require(actual == {'deterministic_algorithms': True, 'warn_only': False,
            'cudnn_deterministic': True, 'cudnn_benchmark': False,
            'float32_matmul_precision': 'highest', 'reduced_precision_math_sdp': False,
            'sdp_enabled': {'MATH': backend == 'MATH', 'FLASH_ATTENTION': backend == 'FLASH_ATTENTION',
                'EFFICIENT_ATTENTION': False, 'CUDNN_ATTENTION': False}}, 'strict backend policy differs')
        results[backend] = assess_comparison(comparison)
        inputs.add(comparison['input_sha256'])
        adapters.update(a['initial_adapter_sha256'] for a in comparison['arms'])
        rngs.update(a['initial_rng_sha256'] for a in comparison['arms'])
    require(len(inputs) == len(adapters) == len(rngs) == 1, 'different backend initial state or input')
    return {'numerical_diagnostic_complete': True, 'by_backend': results,
        'task_learning_demonstrated': False,
        'scope': 'Within-backend repeat/offload equality only;256-token synthetic updates,not long-context training.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    allocator = configure_allocator('expandable')
    environment = configure_deterministic_environment()
    design = preflight(args.plan_sha256)
    if args.check:
        print(json.dumps({'preflight': 'passed', 'id': ID, 'uses_gpu': False}))
        return
    path = ROOT / f'artifacts/{ID}.json'
    require(not path.exists(), 'attempt exists; no retry')
    from wll.gpu_policy import acquire_gpu, verify_cuda

    state = {'id': ID, 'phase': 'waiting_admission', 'pid': os.getpid(),
        'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
        'plan_sha256': args.plan_sha256, 'started_epoch': time.time(), 'allocator_environment': allocator,
        'deterministic_environment': environment, 'backends': [], 'unattempted_backends': BACKENDS,
        'synthetic_only': True, 'training': True, 'benchmark_evaluation': False,
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
            state.update(phase='loading', gpu=admission)
            atomic_json(path, state)
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
            model = load_nf4(MODEL, 'cuda:0')
            require(inspect_nf4(model, 'cuda:0')['linear4bit_layers'] == 496, 'wrong quantized model')
            model = prepare_hybrid_lora(model, rank=8)
            require(sum(p.requires_grad for p in model.parameters()) == 608, 'wrong adapter coverage')
            seed = tokenizer(design['input_seed_text'], add_special_tokens=False)['input_ids']
            ids = torch.tensor([(seed * (256 // len(seed) + 1))[:256]], device='cuda:0', dtype=torch.long)
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            def progress(arm, phase, details):
                state.update(current_arm=arm, current_operation=phase, current_step=details,
                    checked_epoch=time.time())
                atomic_json(path, state)

            for index, backend in enumerate(BACKENDS):
                state.update(phase='comparing', current_backend=backend,
                    unattempted_backends=BACKENDS[index + 1:])
                with deterministic_attention(backend) as actual:
                    state['current_settings'] = actual
                    atomic_json(path, state)
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state(cuda_rng)
                    comparison = paired_updates(model, ids, CASE, progress)
                    state['backends'].append({'backend': backend, 'settings': actual, 'comparison': comparison})
                    atomic_json(path, state)
            torch.cuda.synchronize()
            state['memory'] = memory(torch)
            require(state['memory']['reserved_bytes.all.peak'] <= CAP, 'allocator cap exceeded')
            state.update(phase='complete', result=assess_complete(state['backends']), finished_epoch=time.time(),
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            atomic_json(path, state)
    except BaseException as exc:
        state.update(phase='failed', error_type=type(exc).__name__, error=str(exc)[:2000], finished_epoch=time.time())
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            try:
                state['failure_memory'] = memory(torch)
            except Exception as secondary:
                state['failure_memory_error'] = type(secondary).__name__
        atomic_json(path, state)
        raise


if __name__ == '__main__':
    main()
