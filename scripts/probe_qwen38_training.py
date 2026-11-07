"""One registered NF4 LoRA memory/update probe, with synthetic text only."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

from scripts.probe_allocator import allocator_details, configure_allocator, memory
from scripts.probe_chunked_prefill import CAP, ROOT, digest, require
from scripts.probe_qwen38 import MODEL
from scripts.probe_qwen38 import preflight as parent_preflight
from wll.storage import atomic_json

ID = 'qwen38-27b-training-feasibility-001'
PLAN = 'artifacts/qwen38-27b-training-feasibility-plan-001.json'
DESIGN = 'artifacts/qwen38-27b-training-feasibility-design-001.json'
PARENT_SHA = 'b896c7311e52cde4a7594e95afa2e9657c2530a6787e896d32e7226b7d14b352'
CASES = [{'input_tokens': n, 'target_tokens': t}
    for n, t in [(256, 32), (2048, 128), (8192, 256), (16384, 512)]]


def preflight(sha):
    require(digest(ROOT / PLAN) == sha, 'training probe registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['design'] == DESIGN
        and plan['design_sha256'] == digest(ROOT / DESIGN), 'unregistered training probe')
    for field in ('source_sha256', 'input_sha256'):
        for name, expected in plan[field].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and digest(ROOT / name) == expected, 'bound training input changed: ' + name)
    for name, expected in plan['runtime_sha256'].items():
        require(digest(name) == expected, 'training runtime changed: ' + name)
    design = json.loads((ROOT / DESIGN).read_text())
    require(design['cases'] == CASES and design['rank'] == 8 and design['steps_per_case'] == 2
        and design['learning_rate'] == 1e-4 and design['optimizer'] == 'AdamW_foreach_false'
        and design['weight_decay'] == 0 and design['max_grad_norm'] == 1.0
        and design['allocator_cap_bytes'] == CAP and design['max_gpu_job_seconds'] == 3600
        and design['automatic_retry'] is False and design['automatic_truncation'] is False
        and design['maximum_attempts'] == 1 and design['save_adapter'] is False
        and design['seed'] == 17 and design['input_seed_text'] == 'This is a synthetic training memory probe. '
        and design['parent_plan_sha256'] == PARENT_SHA, 'training recipe differs')
    parent = parent_preflight(PARENT_SHA)
    require(parent['model_directory'] == str(MODEL), 'different pretrained weights')
    return design


def train_case(model, ids, case, progress):
    import torch

    from scripts.qwen38_training import target_batch

    require(ids.ndim == 2 and tuple(ids.shape) == (1, case['input_tokens'])
        and 0 < case['target_tokens'] < case['input_tokens'], 'invalid synthetic case')
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    require(params and all('lora_' in name and 'visual' not in name for name in params), 'text adapters only')
    initial = {name: p.detach().cpu().clone() for name, p in params.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=1e-4, weight_decay=0, foreach=False)
    labels = ids.clone()
    labels[:, :-case['target_tokens']] = -100
    batch = target_batch({'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': labels})
    losses, norms = [], []
    started = time.monotonic()
    for index in range(2):
        optimizer.zero_grad(set_to_none=True)
        progress('forward', {'optimizer_step': index})
        output = model(**batch)
        require(tuple(output.logits.shape[:2]) == (1, case['target_tokens']), 'target projection differs')
        require(bool(torch.isfinite(output.loss)), 'nonfinite loss')
        progress('backward', {'optimizer_step': index, 'loss': float(output.loss.detach())})
        output.loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values()),
            'missing or nonfinite adapter gradient')
        require(all(p.grad is None for p in model.parameters() if not p.requires_grad), 'base gradient exists')
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0, error_if_nonfinite=True)
        progress('update', {'optimizer_step': index})
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in params.values()), 'nonfinite adapter parameter')
        losses.append(float(output.loss.detach()))
        norms.append(float(norm))
        del output
    optimizer.zero_grad(set_to_none=True)
    changed = [name for name, p in params.items() if not torch.equal(initial[name], p.detach().cpu())]
    kinds = sorted({kind for name in changed for kind in ('linear_attn', 'self_attn') if kind in name})
    require(changed and kinds == ['linear_attn', 'self_attn'], 'both attention types must update')
    return {**case, 'optimizer_steps': 2, 'supervised_tokens': 2 * case['target_tokens'],
        'losses': losses, 'gradient_norms_before_clipping': norms, 'finite_gradients': True,
        'finite_parameters': True, 'changed_adapter_tensors': len(changed),
        'changed_attention_types': kinds, 'seconds': time.monotonic() - started}


def assess_complete(cases):
    require(len(cases) == len(CASES), 'all registered training contexts required')
    for result, expected in zip(cases, CASES, strict=True):
        require(all(result.get(k) == v for k, v in expected.items())
            and result['optimizer_steps'] == 2 and result['supervised_tokens'] == 2 * expected['target_tokens']
            and result['finite_gradients'] is result['finite_parameters'] is True
            and result['changed_adapter_tensors'] > 0
            and result['changed_attention_types'] == ['linear_attn', 'self_attn']
            and result['peak_reserved_bytes'] <= CAP, 'incomplete or invalid training measurement')
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
            state.update(phase='loading', gpu=admission)
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
                    state.update(current_operation=phase, current_step=details, checked_epoch=time.time())
                    atomic_json(path, state)

                ids = torch.tensor([(seed * (case['input_tokens'] // len(seed) + 1))[:case['input_tokens']]],
                    device='cuda:0', dtype=torch.long)
                result = train_case(model, ids, case, progress)
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
        atomic_json(path, state)
        raise


if __name__ == '__main__':
    main()
