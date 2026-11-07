"""Compare full-state512-token windows, then measure full8K/16K synthetic training."""
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
from scripts.probe_allocator import configure_allocator, memory
from scripts.probe_chunked_prefill import CAP, ROOT, digest, require
from scripts.probe_qwen38 import MODEL
from scripts.probe_qwen38_training import train_case
from scripts.probe_qwen38_training_deterministic import preflight as parent_preflight
from scripts.streamed_delta import streamed_delta_context
from scripts.streamed_training_numerics import ARMS, compare_streamed_updates
from wll.storage import atomic_json

ID = 'qwen38-27b-training-streamed-001'
PLAN = 'artifacts/qwen38-27b-training-streamed-plan-001.json'
PARENT_SHA = '58823adf2d010314b3ac9fec997210a6872fa898d57c610f8bc9cdc04bc73839'
CASE = {'input_tokens': 2048, 'target_tokens': 128}
CASES = [{'input_tokens': 8192, 'target_tokens': 256}, {'input_tokens': 16384, 'target_tokens': 512}]
WINDOW = 512
TOLERANCE = {'loss_abs': 1e-5, 'gradient_global_relative_l2': 1e-4,
    'gradient_max_abs': 1e-3, 'parameter_max_abs': 5e-6}
SETTINGS = {'deterministic_algorithms': True, 'warn_only': False,
    'cudnn_deterministic': True, 'cudnn_benchmark': False,
    'float32_matmul_precision': 'highest', 'reduced_precision_math_sdp': False,
    'sdp_enabled': {'MATH': False, 'FLASH_ATTENTION': True,
        'EFFICIENT_ATTENTION': False, 'CUDNN_ATTENTION': False}}


def preflight(sha):
    require(digest(ROOT / PLAN) == sha, 'streamed registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['comparison_case'] == CASE
        and plan['arms'] == ARMS and plan['cases'] == CASES and plan['window_tokens'] == WINDOW
        and plan['tolerance'] == TOLERANCE and plan['deterministic_settings'] == SETTINGS
        and plan['steps_per_arm_or_case'] == 2 and plan['rank'] == 8
        and plan['max_gpu_job_seconds'] == 1800 and plan['maximum_attempts'] == 1
        and plan['automatic_retry'] is False and plan['allocator_cap_bytes'] == CAP
        and plan['parent_plan_sha256'] == PARENT_SHA, 'different streamed design')
    require({'scripts/streamed_delta.py', 'scripts/streamed_training_numerics.py',
        'scripts/probe_qwen38_training_streamed.py', 'scripts/run_qwen38_training_streamed_probe.py'}
        <= plan['source_sha256'].keys(), 'missing streamed source binding')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in plan[field].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and digest(ROOT / name) == expected, 'bound streamed file changed: ' + name)
    return parent_preflight(PARENT_SHA)


def validate_measurement(result, expected, *, streamed):
    require(all(result[k] == v for k, v in expected.items())
        and result['optimizer_steps'] == 2 and result['supervised_tokens'] == 2*expected['target_tokens']
        and result['finite_gradients'] is result['finite_parameters'] is True
        and result['changed_adapter_tensors'] == 608
        and result['changed_attention_types'] == ['linear_attn', 'self_attn']
        and len(result['losses']) == 2 and all(math.isfinite(x) for x in result['losses'])
        and 0 < result['memory']['reserved_bytes.all.peak'] <= CAP
        and result['offload']['gpu_to_cpu_bytes'] > 0 and result['offload']['cpu_to_gpu_bytes'] > 0,
        'invalid full-context training measurement')
    if streamed:
        stats = result['streaming']
        require(stats['window_tokens'] == WINDOW and stats['max_input_tokens'] == expected['input_tokens']
            and stats['max_reference_tokens'] == WINDOW and stats['reference_calls'] > stats['calls'] > 0,
            'full input and bounded reference windows required')


def assess_comparison(result):
    arms, comparisons = result['arms'], result['comparisons']
    require(result['case'] == CASE and result['window_tokens'] == WINDOW
        and [a['arm'] for a in arms] == ARMS and set(comparisons) == {'repeat_reference', 'streamed'},
        'incomplete streamed comparison')
    for key in ['initial_adapter_sha256', 'initial_rng_sha256']:
        require(len({a[key] for a in arms}) == 1, 'different initial state')
    for arm in arms:
        validate_measurement(arm, CASE, streamed=arm['arm'] == 'streamed')
        require(len(arm['gradient_sha256_by_step']) == 2, 'incomplete gradient hashes')
    equivalent = {}
    for index, key in enumerate(['repeat_reference', 'streamed'], 1):
        comp, arm = comparisons[key], arms[index]
        require(comp['losses_bitwise_equal'] == (arms[0]['losses'] == arm['losses'])
            and comp['loss_abs_errors'] == [abs(a-b) for a, b in zip(arms[0]['losses'], arm['losses'], strict=True)]
            and len(comp['gradients']) == 2, 'loss or gradient evidence disagrees')
        require(comp['parameters']['bitwise_equal'] ==
            (arms[0]['final_adapter_sha256'] == arm['final_adapter_sha256']), 'parameter hash disagrees')
        for step, gradient in enumerate(comp['gradients']):
            require(gradient['bitwise_equal'] ==
                (arms[0]['gradient_sha256_by_step'][step] == arm['gradient_sha256_by_step'][step]),
                'gradient hash disagrees')
        for metric in [*comp['gradients'], comp['parameters']]:
            require(metric['tensor_count'] == 608
                and all(isinstance(metric[k], (int, float)) and math.isfinite(metric[k]) and metric[k] >= 0
                    for k in ['max_abs_error', 'global_relative_l2_error']), 'invalid numerical metric')
            if metric['bitwise_equal']:
                require(metric['max_abs_error'] == metric['global_relative_l2_error'] == 0.,
                    'bitwise equality disagrees with measured error')
        equivalent[key] = comp['losses_bitwise_equal'] and comp['parameters']['bitwise_equal'] and \
            all(g['bitwise_equal'] for g in comp['gradients'])
        require(max(comp['loss_abs_errors']) <= TOLERANCE['loss_abs']
            and comp['parameters']['max_abs_error'] <= TOLERANCE['parameter_max_abs']
            and all(g['global_relative_l2_error'] <= TOLERANCE['gradient_global_relative_l2']
                and g['max_abs_error'] <= TOLERANCE['gradient_max_abs'] for g in comp['gradients']),
            'numerical difference exceeds registered tolerance; no long-context continuation')
    require(equivalent['repeat_reference'], 'reference is not repeatable; stop before long-context continuation')
    return {'within_registered_tolerance': True, 'repeat_reference_bitwise_equal': True,
        'streamed_bitwise_equal': equivalent['streamed'], 'tolerance': TOLERANCE,
        'task_learning_demonstrated': False}


def assess_complete(state):
    require(state['deterministic_settings'] == SETTINGS and len(state['cases']) == len(CASES),
        'strict Flash settings and both long contexts required')
    comparison = assess_comparison(state['comparison'])
    for result, expected in zip(state['cases'], CASES, strict=True):
        validate_measurement(result, expected, streamed=True)
    return {'comparison': comparison, 'contexts_passed': [2048, 8192, 16384],
        'synthetic_training_probe_complete': True, 'task_learning_demonstrated': False,
        'scope': 'Synthetic full inputs and two updates per arm/case; no real trajectory training or learning claim.'}


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
        'deterministic_environment': environment, 'comparison': None, 'cases': [], 'unattempted_cases': CASES,
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

            def input_ids(length):
                return torch.tensor([(seed * (length // len(seed) + 1))[:length]], device='cuda:0', dtype=torch.long)

            def progress(arm, phase, details):
                state.update(current_arm=arm, current_operation=phase, current_step=details,
                    checked_epoch=time.time())
                atomic_json(path, state)

            with deterministic_attention('FLASH_ATTENTION') as actual:
                require(actual == SETTINGS, 'strict Flash setup differs')
                state.update(phase='comparing', deterministic_settings=actual, current_case=CASE)
                atomic_json(path, state)
                state['comparison'] = compare_streamed_updates(model, input_ids(2048), CASE,
                    progress, window_tokens=WINDOW)
                atomic_json(path, state)
                state['comparison_assessment'] = assess_comparison(state['comparison'])
                atomic_json(path, state)
                # The comparison restores initial adapters. Long-context cases then update sequentially.
                for index, case in enumerate(CASES):
                    state.update(phase='training', current_case=case, unattempted_cases=CASES[index + 1:])
                    atomic_json(path, state)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    ids = input_ids(case['input_tokens'])
                    with streamed_delta_context(window_tokens=WINDOW) as streaming, ActivationOffload(model) as offload:
                        result = train_case(model, ids, case, lambda phase, details: progress('streamed', phase, details))
                    result.update(memory=memory(torch), streaming=dict(streaming), offload=dict(offload.stats))
                    state['cases'].append(result)
                    atomic_json(path, state)
                    validate_measurement(result, case, streamed=True)
                    del ids
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
