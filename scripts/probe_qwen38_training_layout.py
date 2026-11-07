"""Compare full-state512-token windows, then measure full8K/16K synthetic training."""
import argparse
import gc
import json
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
from scripts.probe_qwen38_training_streamed import (
    CASE,
    CASES,
    SETTINGS,
    TOLERANCE,
    WINDOW,
    assess_comparison,
    assess_complete,
    validate_measurement,
)
from scripts.probe_qwen38_training_streamed import (
    preflight as parent_preflight,
)
from scripts.streamed_delta_layout import native_layout_streamed_context as streamed_delta_context
from scripts.streamed_training_numerics_layout import ARMS, compare_streamed_updates
from wll.storage import atomic_json

ID = 'qwen38-27b-training-layout-001'
PLAN = 'artifacts/qwen38-27b-training-layout-plan-001.json'
PARENT_SHA = '035f8614d3f6821a1d6969c8d2f273934e9e88ba75fbd9223553414299876608'


def preflight(sha):
    require(digest(ROOT / PLAN) == sha, 'streamed registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['comparison_case'] == CASE
        and plan['arms'] == ARMS and plan['cases'] == CASES and plan['window_tokens'] == WINDOW
        and plan['tolerance'] == TOLERANCE and plan['deterministic_settings'] == SETTINGS
        and plan['steps_per_arm_or_case'] == 2 and plan['rank'] == 8
        and plan['max_gpu_job_seconds'] == 1800 and plan['maximum_attempts'] == 1
        and plan['automatic_retry'] is False and plan['allocator_cap_bytes'] == CAP
        and plan['parent_plan_sha256'] == PARENT_SHA
        and plan['required_idle_seconds'] == 180
        and plan['variant'] == 'native_input_gradient_layout', 'different streamed design')
    require({'scripts/streamed_delta_layout.py', 'scripts/streamed_training_numerics_layout.py',
        'scripts/probe_qwen38_training_layout.py', 'scripts/run_qwen38_training_layout_probe.py'}
        <= plan['source_sha256'].keys(), 'missing streamed source binding')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in plan[field].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and digest(ROOT / name) == expected, 'bound streamed file changed: ' + name)
    return parent_preflight(PARENT_SHA)


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
        'variant': 'native_input_gradient_layout', 'synthetic_only': True, 'training': True, 'benchmark_evaluation': False,
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
