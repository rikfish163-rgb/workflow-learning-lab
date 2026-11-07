"""Separately registered allocator telemetry on the existing synthetic9B probe."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

from scripts.probe_chunked_prefill import (
    CAP,
    ROOT,
    assess_pair,
    digest,
    generation_options,
    require,
    validate_case,
)
from scripts.probe_chunked_prefill import preflight as parent_preflight
from wll.storage import atomic_json

ID = 'qwen35-9b-allocator-diagnostic-001'
PLAN = 'artifacts/qwen35-allocator-plan-001.json'
PARENT_SHA = '69ba61b739bf382385e71710f7d5be03f59954e4f2c0d9d262b97942b84fafa7'
CONFIGS = {'default': 'backend:native,expandable_segments:False',
           'expandable': 'backend:native,expandable_segments:True'}
ENV_KEYS = ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING')
METRICS = ('allocated_bytes', 'reserved_bytes', 'active_bytes', 'inactive_split_bytes')
COUNTERS = ('num_alloc_retries', 'num_ooms', 'num_device_alloc', 'num_device_free')


def configure_allocator(mode, *, environ=None, modules=None):
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    require(mode in CONFIGS and 'torch' not in modules, 'explicit mode before torch import required')
    expected = CONFIGS[mode]
    require(all(environ.get(k) in (None, expected) for k in ENV_KEYS[:2]),
            'conflicting inherited allocator configuration')
    require(environ.get(ENV_KEYS[2]) in (None, '0'), 'caching must remain enabled')
    environ['PYTORCH_ALLOC_CONF'] = expected
    return {k: environ.get(k) for k in ENV_KEYS}


def select_memory(raw):
    keys = [f'{metric}.all.{period}' for metric in METRICS for period in ('current', 'peak')]
    keys += list(COUNTERS)
    require(all(k in raw and type(raw[k]) is int and raw[k] >= 0 for k in keys),
            'missing or invalid native allocator telemetry')
    require(all(raw[f'{m}.all.peak'] >= raw[f'{m}.all.current'] for m in METRICS),
            'allocator peak smaller than current value')
    return {k: raw[k] for k in keys}


def counter_delta(before, after):
    result = {k: after[k] - before[k] for k in COUNTERS}
    require(all(type(v) is int and v >= 0 for v in result.values()), 'allocator counters decreased')
    return result


def memory(torch):
    require(torch.cuda.memory.get_allocator_backend() == 'native', 'native allocator required')
    return select_memory(torch.cuda.memory_stats(0))


def allocator_details(torch, mode, *, require_segments=False):
    snapshot = torch.cuda.memory._snapshot()
    settings = snapshot['allocator_settings']
    enabled = mode == 'expandable'
    require(torch.cuda.memory.get_allocator_backend() == 'native'
            and settings['expandable_segments'] is enabled
            and settings['PYTORCH_CUDA_ALLOC_CONF'] == CONFIGS[mode],
            'actual allocator settings differ from registration')
    segments = snapshot['segments']
    require(all(segment['device'] == 0 for segment in segments), 'unexpected allocation device')
    expanded = sum(segment.get('is_expandable', False) for segment in segments)
    if require_segments:
        require(segments and (expanded > 0 if enabled else expanded == 0),
                'allocated segments do not confirm requested allocator mode')
    return {'backend': 'native', 'settings': settings, 'segment_count': len(segments),
            'expandable_segment_count': expanded}


def preflight(expected_sha):
    require(digest(ROOT / PLAN) == expected_sha, 'allocator registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['id'] == ID and plan['status'] == 'registered'
            and plan['parent_plan_sha256'] == PARENT_SHA
            and plan['allocator_configs'] == CONFIGS and plan['mode_order'] == list(CONFIGS)
            and plan['max_gpu_job_seconds'] == 1200 and plan['attempts_per_mode'] == 1,
            'unexpected allocator experiment')
    required = {'scripts/probe_allocator.py', 'scripts/run_allocator_diagnostic.py'}
    require(required <= plan['source_sha256'].keys(), 'launch source binding missing')
    for field in ('source_sha256', 'input_sha256'):
        for name, sha in plan[field].items():
            require(digest(ROOT / name) == sha, f'bound file changed: {name}')
    design = parent_preflight(PARENT_SHA)
    return plan, design


def measure(model, ids, mask, arm, chunk_size, pad_token_id):
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = memory(torch)
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**generation_options(ids, mask, chunk_size, pad_token_id))
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    after = memory(torch)  # Cache/output still alive, before diagnostic copies.
    result = {'arm': arm, 'input_tokens': ids.shape[1], 'chunk_size': chunk_size,
              'seconds': elapsed, 'peak_allocated_bytes': after['allocated_bytes.all.peak'],
              'peak_reserved_bytes': after['reserved_bytes.all.peak'],
              'produced_tokens': output.sequences.shape[1] - ids.shape[1],
              'output_token_ids': output.sequences[0, ids.shape[1]:].tolist(),
              'logit_steps': len(output.logits),
              'all_logits_finite': all(bool(torch.isfinite(x).all()) for x in output.logits),
              'before_generation': before, 'after_generation': after,
              'generation_counter_delta': counter_delta(before, after)}
    last_logits = output.logits[-1].detach().float().cpu()
    result['last_logits'] = {'minimum': last_logits.min().item(),
                            'maximum': last_logits.max().item(), 'l2_norm': last_logits.norm().item()}
    del output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result['after_cleanup'] = memory(torch)
    result['including_cleanup_counter_delta'] = counter_delta(before, result['after_cleanup'])
    return result, last_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=list(CONFIGS), required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    env = configure_allocator(args.mode)
    plan, design = preflight(args.plan_sha256)
    if args.check:
        print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'mode': args.mode,
                          'plan_sha256': args.plan_sha256, 'allocator_environment': env}))
        return
    path = ROOT / f'artifacts/{ID}-{args.mode}.json'
    if path.exists():
        raise FileExistsError('allocator attempt exists; no automatic retry')
    from wll.gpu_policy import acquire_gpu, verify_cuda

    state = {'id': f'{ID}-{args.mode}', 'mode': args.mode, 'phase': 'waiting_admission',
             'started_epoch': time.time(), 'pid': os.getpid(),
             'start_ticks': Path('/proc/self/stat').read_text().split()[21],
             'plan_sha256': args.plan_sha256, 'allocator_environment': env,
             'attempted_measurements': 0, 'cases': [], 'pairs': [],
             'benchmark_evaluation': False, 'training': False}
    atomic_json(path, state)
    try:
        with acquire_gpu() as admission:
            import torch
            from transformers import AutoTokenizer

            from scripts.nf4_loader import inspect_nf4, load_nf4
            from scripts.text_chunked_prefill import install_text_chunked_prefill

            verify_cuda(torch)
            torch.set_num_threads(2)
            torch.manual_seed(17)
            state.update(phase='loading', gpu=admission,
                         allocator_backend=torch.cuda.memory.get_allocator_backend())
            require(state['allocator_backend'] == 'native', 'different allocator backend')
            atomic_json(path, state)
            tokenizer = AutoTokenizer.from_pretrained(design['model_directory'],
                local_files_only=True, trust_remote_code=False)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            model = load_nf4(design['model_directory'], 'cuda:0')
            state['quantization_actual'] = inspect_nf4(model, 'cuda:0')
            install_text_chunked_prefill(model)
            torch.cuda.synchronize()
            state.update(phase='probing', load_seconds=time.monotonic() - start,
                         loading_memory=memory(torch),
                         actual_allocator=allocator_details(torch, args.mode, require_segments=True))
            require(state['loading_memory']['reserved_bytes.all.peak'] <= CAP, 'load exceeded cap')
            atomic_json(path, state)
            seed_ids = tokenizer(design['input_seed_text'], add_special_tokens=False)['input_ids']
            for length, order in zip(design['input_lengths'], design['arm_order_by_context'], strict=True):
                values = (seed_ids * (length // len(seed_ids) + 1))[:length]
                ids = torch.tensor([values], dtype=torch.long, device='cuda:0')
                mask, logits = torch.ones_like(ids), {}
                for arm in order:
                    state.update(current_case={'input_tokens': length, 'arm': arm},
                                 attempted_measurements=state['attempted_measurements'] + 1)
                    atomic_json(path, state)
                    measured, last = measure(model, ids, mask, arm, design['arms'][arm],
                                             tokenizer.eos_token_id)
                    state['cases'].append(measured)
                    atomic_json(path, state)
                    validate_case(measured)
                    require(measured['generation_counter_delta']['num_ooms'] == 0,
                            'allocator reported an OOM')
                    logits[arm] = last
                pair = assess_pair(state['cases'][-2:])
                difference = (logits['full'] - logits['chunked']).abs()
                pair['last_logits_max_absolute_difference'] = difference.max().item()
                pair['last_logits_mean_absolute_difference'] = difference.mean().item()
                state['pairs'].append(pair)
                atomic_json(path, state)
                del ids, mask, logits, last, difference
                gc.collect()
                torch.cuda.empty_cache()
            state.update(phase='complete', finished_epoch=time.time(),
                         completion_gate_pending_release=True,
                         within_mode_32k_memory_benefit=state['pairs'][-1]['strict_memory_benefit'])
            atomic_json(path, state)
    except BaseException as exc:
        state.update(phase='failed', error_type=type(exc).__name__, error=str(exc)[:1200],
                     finished_epoch=time.time())
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            try:
                state['failure_memory'] = memory(torch)
            except Exception as diagnostic_error:
                state['failure_memory_error'] = type(diagnostic_error).__name__
        atomic_json(path, state)
        raise


if __name__ == '__main__':
    main()
