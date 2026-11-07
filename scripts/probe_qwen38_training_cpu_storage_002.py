"""Verify frozen CPU weight placement against the prior GPU reference, then full8K/16K."""
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
from scripts.flat_training_numerics import (
    ARMS,
    compare_streamed_updates,
    streamed_delta_context,
)
from scripts.probe_allocator import configure_allocator, memory
from scripts.probe_chunked_prefill import CAP, ROOT, digest, require
from scripts.probe_qwen38 import MODEL
from scripts.probe_qwen38_training import train_case
from scripts.probe_qwen38_training_flat import preflight as parent_preflight
from scripts.probe_qwen38_training_streamed import (
    CASE,
    CASES,
    SETTINGS,
    TOLERANCE,
    WINDOW,
    validate_measurement,
)
from scripts.probe_qwen38_training_sublayer import (
    assess_comparison,
    validate_sublayer,
)
from scripts.probe_qwen38_training_sublayer import (
    assess_complete as parent_assess_complete,
)
from scripts.text_cpu_storage import prepare_text_cpu_storage
from wll.storage import atomic_json

ID = 'qwen38-27b-training-cpu-storage-002'
PLAN = 'artifacts/qwen38-27b-training-cpu-storage-plan-002.json'
PARENT_SHA = '4d73310bdff1b0e58683cbebf4f2b7be1dc11e357467a941f6d6c9ed2bf0c265'


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
        and plan['cpu_modules'] == CPU_MODULES and plan['expected_cpu_parameter_bytes'] == CPU_BYTES
        and plan['reference_signature'] == REFERENCE and plan['reference_input_sha256'] == INPUT_SHA
        and plan['variant'] == 'frozen_embedding_and_vision_CPU_storage', 'different streamed design')
    require({'scripts/sublayer_checkpoint_flat.py', 'scripts/flat_training_numerics.py',
        'scripts/probe_qwen38_training_cpu_storage_002.py', 'scripts/run_qwen38_training_cpu_storage_probe_002.py',
        'scripts/text_cpu_storage.py'}
        <= plan['source_sha256'].keys(), 'missing streamed source binding')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in plan[field].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and digest(ROOT / name) == expected, 'bound streamed file changed: ' + name)
    require(PRIOR_RESULT in plan['input_sha256'], 'prior result binding required')
    assess_reference(json.loads((ROOT / PRIOR_RESULT).read_text())['comparison'])
    return parent_preflight(PARENT_SHA)


CPU_MODULES = ['model.language_model.embed_tokens', 'model.visual']
CPU_BYTES = 3464256992
INPUT_SHA = '0b4544b5fde0aa51b3aec827a063669430b53ab3ee9cb341d159725ec0400a34'
PRIOR_RESULT = 'artifacts/flat-training-result-001/remote-artifacts/qwen38-27b-training-flat-001.json'
REFERENCE = {
    'losses': [5.30125617980957, 4.903193473815918],
    'initial_adapter_sha256': '95aa9a71d761da51c46cf8452f2cbb51c6b4ac8a6335759fae368bb8980c9ef6',
    'initial_rng_sha256': '4794a8ca11eccb143204224cd0a6f5d914c161fdafa887b931085eaaf1d49818',
    'gradient_sha256_by_step': [
        '3019298ac401bd35c8a700ae7b70bfc0bd2d7d61b5d7d042b288111646ac68b0',
        '2d00b5f3cce05cdc22a5264e719e61c842289faac547ff2a99dc1dec25043bdf'],
    'final_adapter_sha256': '1d28e0993788d7c615cc8d3584fbc451295b1c5702629531f4b6d20626370257',
}


def assess_reference(comparison):
    reference = comparison['arms'][0]
    require(reference['arm'] == 'reference' and comparison['input_sha256'] == INPUT_SHA
        and all(reference.get(k) == v for k, v in REFERENCE.items()),
        'prior all-GPU reference differs')
    return {'all_GPU_reference_bitwise_equal': True}


def validate_storage(stats):
    expected = {name: {'bytes': size, 'dtype': ['torch.bfloat16'], 'device': ['cpu'], 'trainable': 0}
        for name, size in zip(CPU_MODULES, [2542796800, 921460192], strict=True)}
    require(stats.get('execution_device') == 'cuda:0' and stats.get('cpu_modules') == CPU_MODULES
        and stats.get('cpu_parameter_bytes') == stats.get('released_cuda_parameter_bytes') == CPU_BYTES
        and stats.get('text_only') is True and stats.get('layout') == expected
        and stats.get('other_parameters_on_cuda0') is True
        and stats.get('trainable_tensors_on_cuda0') == 608
        and stats.get('buffers_on_expected_devices') is True,
        'CPU storage differs from registered placement')


def inspect_storage(model, stats):
    base = model.get_base_model()

    def on_cpu(name):
        return any(name == prefix or name.startswith(prefix + '.') for prefix in CPU_MODULES)

    layout = {}
    for name in CPU_MODULES:
        params = list(base.get_submodule(name).parameters())
        layout[name] = {'bytes': sum(p.numel() * p.element_size() for p in params),
            'dtype': sorted({str(p.dtype) for p in params}),
            'device': sorted({str(p.device) for p in params}),
            'trainable': sum(p.requires_grad for p in params)}
    result = {**stats, 'layout': layout,
        'other_parameters_on_cuda0': all(str(p.device) == 'cuda:0'
            for n, p in base.named_parameters() if not on_cpu(n)),
        'trainable_tensors_on_cuda0': sum(p.requires_grad and str(p.device) == 'cuda:0'
            for p in base.parameters()),
        'buffers_on_expected_devices': all(str(b.device) == ('cpu' if on_cpu(n) else 'cuda:0')
            for n, b in base.named_buffers())}
    validate_storage(result)
    return result


def assess_complete(state):
    validate_storage(state['cpu_storage'])
    validate_storage(state['final_cpu_storage'])
    require(state['cpu_storage'] == state['final_cpu_storage'], 'CPU storage changed during training')
    return {**parent_assess_complete(state), **assess_reference(state['comparison']),
        'cpu_storage_verified': True}


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
        'variant': 'frozen_embedding_and_vision_CPU_storage', 'synthetic_only': True, 'training': True, 'benchmark_evaluation': False,
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
            state['before_cpu_storage_memory'] = memory(torch)
            placement = prepare_text_cpu_storage(model)
            state['cpu_storage'] = inspect_storage(model, placement)
            gc.collect()
            torch.cuda.empty_cache()
            state['after_cpu_storage_memory'] = memory(torch)
            atomic_json(path, state)
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
                state['comparison_assessment'] = {**assess_comparison(state['comparison']),
                    **assess_reference(state['comparison'])}
                atomic_json(path, state)
                # The comparison restores initial adapters. Long-context cases then update sequentially.
                for index, case in enumerate(CASES):
                    state.update(phase='training', current_case=case, unattempted_cases=CASES[index + 1:])
                    atomic_json(path, state)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    ids = input_ids(case['input_tokens'])
                    with streamed_delta_context(model, window_tokens=WINDOW) as streaming, ActivationOffload(model) as offload:
                        result = train_case(model, ids, case, lambda phase, details: progress('streamed', phase, details))
                    result.update(memory=memory(torch), streaming=dict(streaming), offload=dict(offload.stats))
                    state['cases'].append(result)
                    atomic_json(path, state)
                    validate_measurement(result, case, streamed=True)
                    validate_sublayer(result)
                    del ids
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
