"""One bounded27B same-state control/repeat/offload numerical diagnostic."""
import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

from scripts.probe_allocator import configure_allocator, memory
from scripts.probe_chunked_prefill import CAP, ROOT, digest, require
from scripts.probe_qwen38 import MODEL
from scripts.probe_qwen38_training_offload import preflight as parent_preflight
from scripts.training_numerics import ARMS, paired_updates
from wll.storage import atomic_json

ID = 'qwen38-27b-training-numerics-001'
PLAN = 'artifacts/qwen38-27b-training-numerics-plan-001.json'
PARENT_SHA = '280ee60edcbf4c4ad033b1de382667c89f44fdf7f54a730194a036d662dcafd0'
CASE = {'input_tokens': 256, 'target_tokens': 32}


def preflight(sha):
    require(digest(ROOT / PLAN) == sha, 'numerical registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['case'] == CASE
        and plan['arms'] == ARMS and plan['steps_per_arm'] == 2 and plan['rank'] == 8
        and plan['max_gpu_job_seconds'] == 600 and plan['maximum_attempts'] == 1
        and plan['automatic_retry'] is False and plan['allocator_cap_bytes'] == CAP
        and plan['parent_plan_sha256'] == PARENT_SHA, 'different numerical design')
    require({'scripts/training_numerics.py', 'scripts/probe_qwen38_training_numerics.py',
        'scripts/run_qwen38_training_numerics_probe.py'} <= plan['source_sha256'].keys(),
        'missing numerical source binding')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in plan[field].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and digest(ROOT / name) == expected, 'bound numerical file changed: ' + name)
    return parent_preflight(PARENT_SHA)


def assess_complete(result):
    arms, comparisons = result['arms'], result['comparisons']
    require(result['case'] == CASE and [a['arm'] for a in arms] == ARMS
        and set(comparisons) == {'repeat_control', 'offload'}, 'incomplete numerical arms')
    for field in ['initial_adapter_sha256', 'initial_rng_sha256']:
        require(len({a[field] for a in arms}) == 1, 'different initial state')
    for arm in arms:
        require(all(arm[k] == v for k, v in CASE.items()) and arm['optimizer_steps'] == 2
            and arm['supervised_tokens'] == 64 and arm['finite_gradients'] is arm['finite_parameters'] is True
            and len(arm['losses']) == len(arm['gradient_sha256_by_step']) == 2
            and all(math.isfinite(loss) for loss in arm['losses'])
            and arm['changed_attention_types'] == ['linear_attn', 'self_attn'], 'invalid numerical arm')
    require(arms[2]['offload']['gpu_to_cpu_bytes'] > 0 and arms[2]['offload']['cpu_to_gpu_bytes'] > 0,
        'real CUDA transfers required')
    conclusions = {}
    for name, comparison in comparisons.items():
        index = ARMS.index(name)
        require(len(comparison['gradients']) == 2 and comparison['parameters']['tensor_count'] == 608
            and all(g['tensor_count'] == 608 for g in comparison['gradients']), 'incomplete parameter coverage')
        require(comparison['parameters']['bitwise_equal'] ==
            (arms[0]['final_adapter_sha256'] == arms[index]['final_adapter_sha256']), 'parameter hash disagrees')
        for step, gradient in enumerate(comparison['gradients']):
            require(gradient['bitwise_equal'] ==
                (arms[0]['gradient_sha256_by_step'][step] == arms[index]['gradient_sha256_by_step'][step]),
                'gradient hash disagrees')
        conclusions[name + '_bitwise_equal'] = comparison['losses_bitwise_equal'] and \
            comparison['parameters']['bitwise_equal'] and all(g['bitwise_equal'] for g in comparison['gradients'])
    return {**conclusions, 'numerical_diagnostic_complete': True, 'task_learning_demonstrated': False,
        'scope': 'Same initial27B adapters/RNG,256 input and32 target tokens,two updates. Not long-context or task validation.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    allocator = configure_allocator('expandable')
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
        'comparison': None, 'synthetic_only': True, 'training': True,
        'benchmark_evaluation': False, 'adapter_saved': False, 'task_learning_demonstrated': False}
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
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            state.update(phase='comparing', float32_matmul_precision=torch.get_float32_matmul_precision(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())

            def progress(arm, phase, details):
                state.update(current_arm=arm, current_operation=phase, current_step=details,
                    checked_epoch=time.time())
                atomic_json(path, state)

            state['comparison'] = paired_updates(model, ids, CASE, progress)
            torch.cuda.synchronize()
            state['memory'] = memory(torch)
            require(state['memory']['reserved_bytes.all.peak'] <= CAP, 'allocator cap exceeded')
            state.update(phase='complete', result=assess_complete(state['comparison']), finished_epoch=time.time(),
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
