"""One registered NF4 synthetic GPU comparison; no task or learning claims."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

from wll.storage import atomic_json

ROOT = Path(__file__).resolve().parent.parent
ID = 'qwen35-9b-chunked-prefill-001'
PLAN = 'artifacts/qwen35-chunked-prefill-plan-001.json'
DESIGN = 'artifacts/qwen35-chunked-prefill-design-001.json'
PYTHON = '/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python'
CAP = 20 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_registration(root, expected_sha):
    root = Path(root)
    require(digest(root / PLAN) == expected_sha, 'registered plan changed')
    plan = json.loads((root / PLAN).read_text())
    design = json.loads((root / DESIGN).read_text())
    require(plan['id'] == design['id'] == ID and plan['status'] == 'registered',
            'unregistered experiment')
    require(plan['design'] == DESIGN and plan['design_sha256'] == digest(root / DESIGN),
            'original design changed')
    required = {'scripts/probe_chunked_prefill.py', 'scripts/run_chunked_prefill_probe.py',
                'scripts/gpu_idle_watch.py', 'wll/gpu_idle.py', 'wll/storage.py'}
    require(required <= plan['source_sha256'].keys(), 'missing launch dependency binding')
    for field in ('source_sha256', 'input_sha256'):
        require(all(plan[field].get(name) == sha for name, sha in design[field].items()),
                'original design bindings changed')
        for name, sha in plan[field].items():
            require(digest(root / name) == sha, f'bound file changed: {name}')
    require(design['input_lengths'] == [2048, 8192, 16384, 32768]
            and design['new_tokens'] == 8 and design['arms'] == {'full': None, 'chunked': 512}
            and design['max_gpu_job_seconds'] == 1200 and design['allocator_cap_gib'] == 20
            and design['maximum_attempts'] == 1 and design['automatic_retry'] is False,
            'unexpected experiment configuration')
    return plan, design


def preflight(expected_sha):
    from artifacts.probe_qwen35_nf4_001 import preflight as nf4_preflight

    plan, design = load_registration(ROOT, expected_sha)
    previous = nf4_preflight(plan['input_sha256'][
        'artifacts/qwen35-9b-nf4-feasibility-plan-001.json'])
    require(previous['model_directory'] == design['model_directory']
            and previous['revision'] == design['model_revision'], 'different model identity')
    return design


def validate_case(result):
    require(result['produced_tokens'] == 8 and len(result['output_token_ids']) == 8
            and result['logit_steps'] == 8 and result['all_logits_finite'] is True,
            'incomplete or nonfinite generation')
    require(result['peak_reserved_bytes'] <= CAP, 'allocator cap exceeded')


def assess_pair(cases):
    require(len(cases) == 2 and {r['arm'] for r in cases} == {'full', 'chunked'},
            'two different arms required')
    full = next(r for r in cases if r['arm'] == 'full')
    chunked = next(r for r in cases if r['arm'] == 'chunked')
    for result in cases:
        validate_case(result)
    require(full['input_tokens'] == chunked['input_tokens'], 'different context lengths')
    require(full['output_token_ids'] == chunked['output_token_ids'], 'greedy token mismatch')
    saved = full['peak_reserved_bytes'] - chunked['peak_reserved_bytes']
    return {'input_tokens': full['input_tokens'], 'identical_output_tokens': True,
            'chunked_reserved_bytes_saved': saved, 'strict_memory_benefit': saved > 0}


def generation_options(ids, mask, chunk_size, pad_token_id):
    return {'input_ids': ids, 'attention_mask': mask, 'do_sample': False,
            'use_cache': True, 'min_new_tokens': 8, 'max_new_tokens': 8,
            'prefill_chunk_size': chunk_size, 'pad_token_id': pad_token_id,
            'logits_to_keep': 1, 'disable_compile': True,
            'return_dict_in_generate': True, 'output_logits': True}


def measure(model, ids, mask, arm, chunk_size, pad_token_id):
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**generation_options(ids, mask, chunk_size, pad_token_id))
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    # Peak values precede diagnostic copies/reductions. No cache/output survives
    # this call on CUDA; only one final-token logit vector is retained on CPU.
    result = {'arm': arm, 'input_tokens': ids.shape[1], 'chunk_size': chunk_size,
              'seconds': elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'produced_tokens': output.sequences.shape[1] - ids.shape[1],
              'output_token_ids': output.sequences[0, ids.shape[1]:].tolist(),
              'logit_steps': len(output.logits),
              'all_logits_finite': all(bool(torch.isfinite(x).all()) for x in output.logits)}
    last_logits = output.logits[-1].detach().float().cpu()
    result['last_logits'] = {'minimum': last_logits.min().item(),
                            'maximum': last_logits.max().item(),
                            'l2_norm': last_logits.norm().item()}
    del output
    return result, last_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    design = preflight(args.plan_sha256)
    if args.check:
        print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'id': ID,
                          'model_directory': design['model_directory'],
                          'plan_sha256': args.plan_sha256}))
        return
    result_path = ROOT / f'artifacts/{ID}.json'
    if result_path.exists():
        raise FileExistsError('an attempt exists; no automatic retry')
    from wll.gpu_policy import acquire_gpu, verify_cuda

    state = {'phase': 'waiting_admission', 'started_epoch': time.time(),
             'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().split()[21],
             'plan_sha256': args.plan_sha256, 'benchmark_evaluation': False, 'training': False,
             'attempted_measurements': 0, 'cases': [], 'pairs': []}
    atomic_json(result_path, state)
    try:
        with acquire_gpu() as admission:
            import torch
            from transformers import AutoTokenizer

            from scripts.nf4_loader import inspect_nf4, load_nf4
            from scripts.text_chunked_prefill import install_text_chunked_prefill

            verify_cuda(torch)
            torch.set_num_threads(2)
            torch.manual_seed(17)
            state.update(phase='loading', gpu=admission)
            atomic_json(result_path, state)
            tokenizer = AutoTokenizer.from_pretrained(design['model_directory'],
                local_files_only=True, trust_remote_code=False)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            model = load_nf4(design['model_directory'], 'cuda:0')
            state['quantization_actual'] = inspect_nf4(model, 'cuda:0')
            install_text_chunked_prefill(model)
            torch.cuda.synchronize()
            state.update(phase='probing', load_seconds=time.monotonic() - start,
                         load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         load_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            require(state['load_peak_reserved_bytes'] <= CAP, 'model loading exceeded cap')
            atomic_json(result_path, state)
            seed_ids = tokenizer(design['input_seed_text'], add_special_tokens=False)['input_ids']
            for length, order in zip(design['input_lengths'], design['arm_order_by_context'], strict=True):
                values = (seed_ids * (length // len(seed_ids) + 1))[:length]
                ids = torch.tensor([values], dtype=torch.long, device='cuda:0')
                mask = torch.ones_like(ids)
                logits = {}
                for arm in order:
                    state.update(current_case={'input_tokens': length, 'arm': arm},
                                 attempted_measurements=state['attempted_measurements'] + 1)
                    atomic_json(result_path, state)
                    measured, last = measure(model, ids, mask, arm, design['arms'][arm],
                                             tokenizer.eos_token_id)
                    state['cases'].append(measured)
                    atomic_json(result_path, state)
                    validate_case(measured)
                    logits[arm] = last
                pair = assess_pair(state['cases'][-2:])
                difference = (logits['full'] - logits['chunked']).abs()
                pair['last_logits_max_absolute_difference'] = difference.max().item()
                pair['last_logits_mean_absolute_difference'] = difference.mean().item()
                state['pairs'].append(pair)
                atomic_json(result_path, state)
                del ids, mask, logits, last, difference
                gc.collect()
                torch.cuda.empty_cache()
            state.update(phase='complete', finished_epoch=time.time(),
                         completion_gate_pending_release=True,
                         memory_benefit_gate=state['pairs'][-1]['strict_memory_benefit'])
            atomic_json(result_path, state)
    except BaseException as exc:
        state.update(phase='failed', error_type=type(exc).__name__, error=str(exc)[:1200],
                     finished_epoch=time.time())
        atomic_json(result_path, state)
        raise


if __name__ == '__main__':
    main()
