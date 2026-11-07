"""One registered CUDA gate for the actual pooled-token SFT update recipe.

Only synthetic inputs; no research adapter is exported. The native objective
comparison and repeat must pass before the full 16K accumulation stress case.
"""
import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

from scripts.activation_offload import ActivationOffload
from scripts.deterministic_training import (
    configure_deterministic_environment,
    deterministic_attention,
)
from scripts.flat_training_numerics import streamed_delta_context
from scripts.hybrid_target_updates import train_groups, validate_groups
from scripts.output_head_cpu_storage import prepare_output_head_cpu_storage
from scripts.probe_allocator import configure_allocator, memory
from scripts.probe_chunked_prefill import CAP, ROOT, digest, require
from scripts.probe_qwen38 import MODEL
from scripts.probe_qwen38_training_head_storage import inspect_storage, validate_storage
from scripts.probe_qwen38_training_head_storage import (
    preflight as parent_preflight,
)
from scripts.probe_qwen38_training_streamed import SETTINGS, TOLERANCE, WINDOW
from scripts.text_cpu_storage import prepare_text_cpu_storage
from wll.storage import atomic_json

ID = 'qwen38-27b-hybrid-updates-001'
PLAN = 'artifacts/qwen38-27b-hybrid-updates-plan-001.json'
PARENT_PLAN = 'artifacts/qwen38-27b-training-head-storage-plan-001.json'
PARENT_SHA = '2091f7f15eb05ac1069a40b722d9292f065db5577c4aba5c6d5beab8ec9795bc'
COMPARISON = [[[2048, 64], [2048, 128]], [[2048, 96]]]
LONG = [[[16384, 512], [16384, 256]], [[16384, 146]]]
ARMS = ['reference', 'repeat_reference', 'candidate']
CPU_PROOF = 'artifacts/hybrid-update-numerics-CPU-verification-001.json'
HEAD_PROOF = 'artifacts/head-storage-training-integration-001.json'
DESIGN = {'comparison_schedule': COMPARISON, 'long_schedule': LONG, 'arms': ARMS,
    'window_tokens': WINDOW, 'tolerance': TOLERANCE, 'deterministic_settings': SETTINGS,
    'seed': 17, 'rank': 8, 'learning_rate': 1e-4, 'max_grad_norm': 1.0,
    'max_gpu_job_seconds': 1800, 'maximum_attempts': 1, 'automatic_retry': False,
    'allocator_cap_bytes': CAP, 'required_idle_seconds': 180,
    'synthetic_only': True, 'adapter_saved': False, 'parent_plan_sha256': PARENT_SHA}


def preflight(sha):
    require(digest(ROOT / PLAN) == sha, 'registered hybrid plan changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['id'] == ID and plan['state'] == 'registered'
        and all(plan.get(k) == v for k, v in DESIGN.items()), 'different hybrid update design')
    required = {'scripts/probe_qwen38_hybrid_updates.py', 'scripts/run_qwen38_hybrid_probe.py',
        'scripts/hybrid_target_updates.py', 'scripts/hybrid_update_numerics.py',
        'scripts/gpu_idle_watch.py', 'wll/gpu_policy.py', 'wll/gpu_idle.py'}
    require(required <= plan['source_sha256'].keys(), 'missing hybrid source binding')
    require({PARENT_PLAN, CPU_PROOF, HEAD_PROOF, 'config/gpu-policy.json'}
        <= plan['input_sha256'].keys(), 'missing hybrid prerequisite binding')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in plan[field].items():
            p = Path(name)
            require(not p.is_absolute() and '..' not in p.parts
                and not (ROOT / p).is_symlink() and digest(ROOT / p) == expected,
                'bound hybrid file changed: ' + name)
    require(plan['input_sha256'][PARENT_PLAN] == PARENT_SHA, 'parent plan changed')
    proof = json.loads((ROOT / CPU_PROOF).read_text())
    require(proof['status'] == 'passed' and proof['GPU_jobs_added'] == 0
        and proof['cuda_initialized'] is False, 'real CPU numerical verification required')
    bindings = {**plan['source_sha256'], **plan['input_sha256']}
    require(all(bindings.get(n) == h for n, h in proof['source_and_evidence_sha256'].items()),
        'numerical CPU evidence not frozen')
    previous = json.loads((ROOT / HEAD_PROOF).read_text())
    require(previous['phase'] == 'complete' and previous['all_resources_released'] is True
        and previous['plan_sha256'] == PARENT_SHA, 'prior 16K probe must complete and release')
    return parent_preflight(PARENT_SHA)


def build_groups(seed, layout):
    require(isinstance(seed, list) and seed
        and all(type(t) is int and t >= 0 for t in seed), 'nonempty valid synthetic seed required')
    groups = []
    for group in layout:
        rows = []
        for length, targets in group:
            require(type(length) is int and type(targets) is int and 0 < targets < length <= 16384,
                'invalid synthetic lengths')
            ids = (seed * (length // len(seed) + 1))[:length]
            rows.append({'input_ids': ids, 'attention_mask': [1] * length,
                'labels': [-100] * (length - targets) + ids[-targets:]})
        groups.append(rows)
    validate_groups(groups, max_length=16384)
    return groups


def validate_update_measurement(result, expected):
    totals = [sum(t for _, t in g) for g in expected]
    microbatches = sum(map(len, expected))
    require(result['schedule'] == expected and result['optimizer_steps'] == len(expected)
        and result['supervised_tokens'] == sum(totals)
        and result['target_tokens_per_update'] == totals
        and result['examples_per_update'] == list(map(len, expected))
        and result['finite_gradients'] is result['finite_parameters'] is True
        and result['adapter_saved'] is False and result['changed_adapter_tensors'] == 608
        and result['changed_attention_types'] == ['linear_attn', 'self_attn']
        and all(len(result[k]) == len(expected) and all(type(x) in (int, float)
            and math.isfinite(x) and x >= 0 for x in result[k])
            for k in ['losses', 'gradient_norms_before_clipping'])
        and 0 < result['memory']['reserved_bytes.all.peak'] <= CAP
        and result['offload']['gpu_to_cpu_bytes'] > 0 and result['offload']['cpu_to_gpu_bytes'] > 0,
        'incomplete or invalid full-context hybrid update measurement')
    stats = result['streaming']
    require(stats['window_tokens'] == WINDOW
        and stats['max_input_tokens'] == max(n for g in expected for n, _ in g)
        and stats['max_reference_tokens'] == WINDOW
        and stats['reference_calls'] > stats['calls'] > 0
        and stats['sublayer_checkpoint'] == {'decoder_layers': 64, 'wrapped_sublayers': 128,
            'calls': 128 * microbatches, 'evaluations': 256 * microbatches},
        'incomplete full-history streaming or checkpoint coverage')


def assess_complete(state):
    from scripts.hybrid_update_numerics import assess_hybrid_comparison

    require(state['deterministic_settings'] == SETTINGS and state['synthetic_only'] is True
        and state['adapter_saved'] is False, 'different synthetic scope/settings')
    validate_storage(state['cpu_storage'])
    validate_storage(state['final_cpu_storage'])
    require(state['cpu_storage'] == state['final_cpu_storage'], 'CPU placement changed')
    comparison = assess_hybrid_comparison(state['comparison'])
    validate_comparison_scope(state['comparison'])
    for arm in state['comparison']['arms']:
        validate_update_measurement(arm, COMPARISON)
    validate_update_measurement(state['long_case'], LONG)
    require(state['long_case']['initial_adapter_sha256'] ==
        state['comparison']['arms'][0]['initial_adapter_sha256'], 'long case must start from restored adapters')
    return {'comparison': comparison, 'synthetic_updates': 8, 'synthetic_microbatches': 12,
        'contexts_passed': [2048, 16384], 'synthetic_update_recipe_CUDA_verified': True,
        'real_workflow_training': False, 'task_learning_demonstrated': False,
        'scope': 'Synthetic numerical/memory gate, variable token counts and last short group; no adapter export.'}


def validate_comparison_scope(comparison):
    require([a['arm'] for a in comparison['arms']] == ARMS
        and comparison['trainable_adapter_tensors'] == 608
        and comparison['max_length'] == 16384 and comparison['window_tokens'] == WINDOW
        and comparison['schedule'] == [[{'input_tokens': n, 'target_tokens': t} for n, t in g]
            for g in COMPARISON], 'different comparison schedule or adapter coverage')


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
        'plan_sha256': args.plan_sha256, 'started_epoch': time.time(),
        'allocator_environment': allocator, 'deterministic_environment': environment,
        'comparison': None, 'long_case': None, 'synthetic_only': True, 'training': True,
        'benchmark_evaluation': False, 'adapter_saved': False, 'task_learning_demonstrated': False}
    atomic_json(path, state)
    try:
        with acquire_gpu() as admission:
            import torch
            from transformers import AutoTokenizer

            from scripts.hybrid_update_numerics import (
                assess_hybrid_comparison,
                compare_hybrid_updates,
            )
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
            text = prepare_text_cpu_storage(model)
            head = prepare_output_head_cpu_storage(model)
            placement = {**text, 'cpu_modules': text['cpu_modules'] + head['cpu_modules'],
                'cpu_parameter_bytes': text['cpu_parameter_bytes'] + head['cpu_parameter_bytes'],
                'released_cuda_parameter_bytes': text['released_cuda_parameter_bytes']
                    + head['released_cuda_parameter_bytes'], 'head': head}
            state['cpu_storage'] = inspect_storage(model, placement)
            gc.collect()
            torch.cuda.empty_cache()
            state['loading_memory'] = memory(torch)
            require(state['loading_memory']['reserved_bytes.all.peak'] <= CAP, 'loading exceeds cap')
            atomic_json(path, state)
            seed = tokenizer(design['input_seed_text'], add_special_tokens=False)['input_ids']

            def progress(arm, phase, details):
                state.update(current_arm=arm, current_operation=phase, current_step=details,
                    checked_epoch=time.time())
                atomic_json(path, state)

            with deterministic_attention('FLASH_ATTENTION') as actual:
                require(actual == SETTINGS, 'strict Flash setup differs')
                state.update(phase='comparing', deterministic_settings=actual)
                atomic_json(path, state)
                comparison = compare_hybrid_updates(model, build_groups(seed, COMPARISON),
                    progress, max_length=16384, window_tokens=WINDOW)
                for arm in comparison['arms']:
                    arm['schedule'] = COMPARISON
                state['comparison'] = comparison
                atomic_json(path, state)
                state['comparison_assessment'] = assess_hybrid_comparison(comparison)
                validate_comparison_scope(comparison)
                for arm in comparison['arms']:
                    validate_update_measurement(arm, COMPARISON)
                state.update(phase='long_context', current_arm='candidate')
                atomic_json(path, state)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                with streamed_delta_context(model, window_tokens=WINDOW) as streaming, ActivationOffload(model) as offload:
                    result = train_groups(model, build_groups(seed, LONG), max_length=16384,
                        progress=lambda phase, details: progress('candidate', phase, details))
                torch.cuda.synchronize()
                result.update(schedule=LONG, seconds=time.monotonic() - started,
                    memory=memory(torch), offload=dict(offload.stats), streaming=dict(streaming))
                state['long_case'] = result
                atomic_json(path, state)
                validate_update_measurement(result, LONG)
            state['final_cpu_storage'] = inspect_storage(model, placement)
            state.update(phase='complete', result=assess_complete(state), finished_epoch=time.time(),
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
