"""One complete, separately registered success-trajectory SFT baseline pilot."""
import argparse
import gc
import json
import math
import os
import re
import resource
import sys
import time
from pathlib import Path

from scripts.activation_offload import ActivationOffload
from scripts.b3_training_data import SCHEDULE_SHA256, load_verified_groups
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
from scripts.probe_qwen38_hybrid_updates import preflight as parent_preflight
from scripts.probe_qwen38_training_head_storage import inspect_storage, validate_storage
from scripts.probe_qwen38_training_streamed import SETTINGS, WINDOW
from scripts.text_cpu_storage import prepare_text_cpu_storage
from wll.storage import atomic_json

ID = 'qwen38-b3-sft-001'
PLAN = 'artifacts/qwen38-b3-sft-plan-001.json'
PARENT_SHA = '1483493d8d8c49ec08d20f4631a6f7d39ec8526c483ba9d3db130ee93da247ab'
GATE = 'artifacts/hybrid-updates-integration-001.json'
GATE_RELEASE = 'artifacts/hybrid-updates-release-001.json'
SCHEDULE_DIR = 'artifacts/b3-training-schedule-preparation-001'
OUTPUT = f'artifacts/{ID}/adapter'
CPU_PROOFS = (
    'artifacts/b3-training-data-CPU-verification-001.json',
    'artifacts/b3-training-data-CPU-verification-002.json',
    'artifacts/hybrid-adapter-save-CPU-verification-001.json',
    'artifacts/gpu-idle-watch-b3-CPU-verification-001.json',
)
DESIGN = {'parent_plan_sha256': PARENT_SHA, 'seed': 17, 'epochs': 1,
    'microbatch_size': 1, 'gradient_accumulation_steps': 4, 'optimizer_steps': 86,
    'examples': 344, 'input_tokens': 2028299, 'supervised_tokens': 11085,
    'max_length': 16384, 'rank': 8, 'alpha': 16, 'dropout': 0,
    'learning_rate': 1e-4, 'weight_decay': 0, 'max_grad_norm': 1.0,
    'optimizer': 'AdamW', 'betas': [0.9, 0.999], 'epsilon': 1e-8,
    'foreach': False, 'learning_rate_schedule': 'constant',
    'objective': 'FP32 per-token CE, FP64 sum divided by whole-update supervised-token count',
    'max_gpu_job_seconds': 21600, 'maximum_attempts': 1, 'automatic_retry': False,
    'required_idle_seconds': 180, 'allocator_cap_bytes': CAP, 'output': OUTPUT,
    'window_tokens': WINDOW, 'deterministic_settings': SETTINGS,
    'schedule_sha256': SCHEDULE_SHA256, 'save_policy': 'final_only_after_all86updates',
    'early_stopping': False, 'resume': False, 'method_experiences_used': 0,
    'baseline': 'B3_success_only_SFT', 'benchmark_evaluation': False}


def preflight(sha):
    require(digest(ROOT / PLAN) == sha, 'B3 registered plan changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['id'] == ID and plan['state'] == 'registered'
        and all(plan.get(k) == v for k, v in DESIGN.items()), 'different B3 design')
    require({'scripts/train_qwen38_b3.py', 'scripts/run_qwen38_b3_training.py',
        'scripts/b3_training_data.py', 'scripts/hybrid_target_updates.py', 'scripts/save_hybrid_adapter.py',
        'wll/gpu_policy.py', 'scripts/gpu_idle_watch.py', 'scripts/gpu_idle_watch_b3.py'} <= plan['source_sha256'].keys(),
        'missing B3 source binding')
    require({GATE, GATE_RELEASE, 'config/gpu-policy.json',
        *[f'{SCHEDULE_DIR}/{n}' for n in SCHEDULE_SHA256]} <= plan['input_sha256'].keys(),
        'missing B3 input binding')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in plan[field].items():
            p = Path(name)
            require(not p.is_absolute() and '..' not in p.parts and not (ROOT / p).is_symlink()
                and digest(ROOT / p) == expected, 'bound B3 file changed: ' + name)
    validate_cpu_proofs({**plan['source_sha256'], **plan['input_sha256']})
    require(all(plan['input_sha256'][f'{SCHEDULE_DIR}/{n}'] == h for n, h in SCHEDULE_SHA256.items()),
        'prepared full corpus schedule changed')
    gate = json.loads((ROOT / GATE).read_text())
    release = json.loads((ROOT / GATE_RELEASE).read_text())
    require(gate['phase'] == 'complete' and gate['complete'] is gate['all_resources_released'] is True
        and gate['plan_sha256'] == PARENT_SHA and gate['probe']['gate_passed'] is True
        and gate['release_sha256'] == digest(ROOT / GATE_RELEASE)
        and release['released'] is release['complete'] is True
        and release['controller_released'] is release['heartbeat_absent'] is True,
        'completed, audited and released synthetic recipe prerequisite required')
    parent_preflight(PARENT_SHA)
    return plan


def validate_cpu_proofs(bindings, *, root=ROOT):
    proofs = []
    for name in CPU_PROOFS:
        require(name in bindings and digest(root / name) == bindings[name], 'CPU proof missing or changed')
        proof = json.loads((root / name).read_text())
        for source, expected in proof['source_and_evidence_sha256'].items():
            require(bindings.get(source) == expected and digest(root / source) == expected,
                'CPU evidence/source binding differs: ' + source)
        proofs.append(proof)
    data, archive, saved, watcher = proofs
    require(data['phase'] == 'cpu_regeneration_verified_not_training_authorization'
        and data['new_gpu_jobs'] == 0 and data['torch_cuda_initialized'] is False
        and data['training_started'] is False and data['examples'] == 344
        and data['optimizer_updates'] == 86 and data['input_tokens_one_epoch'] == 2028299
        and data['supervised_tokens_one_epoch'] == 11085
        and all(check['exit_code'] == 0 for check in data['checks'].values()), 'CPU data proof incomplete')
    recovered = archive['recovered_exact_stdin_path']
    require(archive['phase'] == 'cpu_verification_script_archived_no_rerun'
        and archive['prior_verification_path'] == CPU_PROOFS[0]
        and archive['prior_verification_sha256'] == bindings[CPU_PROOFS[0]]
        and bindings.get(recovered) == archive['recovered_exact_stdin_sha256'] == digest(root / recovered),
        'CPU command archive binding differs')
    require(saved['status'] == 'passed' and saved['actual_CPU_checks'] == 39
        and saved['GPU_jobs_added'] == 0 and saved['cuda_initialized'] is False,
        'CPU native adapter export proof incomplete')
    require(watcher['status'] == 'passed' and watcher['actual_CPU_checks'] == 81
        and watcher['GPU_jobs_added'] == watcher['GPU_queries'] == watcher['monitor_processes_started'] == 0
        and watcher['torch_imported'] is False, 'CPU watcher proof incomplete')


def validate_data_summary(data):
    schedule = ROOT / SCHEDULE_DIR / 'update-groups.json'
    require(digest(schedule) == SCHEDULE_SHA256['update-groups.json']
        and data['order_sha256'] == json.loads(schedule.read_text())['order_sha256']
        and data['schedule_files_sha256'] == SCHEDULE_SHA256
        and data['examples'] == 344 and data['optimizer_updates'] == 86
        and data['input_tokens_one_epoch'] == 2028299 and data['supervised_tokens_one_epoch'] == 11085,
        'B3 data identity or coverage differs')


def validate_training_result(result, totals):
    schedule_path = ROOT / SCHEDULE_DIR / 'update-groups.json'
    require(digest(schedule_path) == SCHEDULE_SHA256['update-groups.json']
        and totals == [g['supervised_tokens'] for g in json.loads(schedule_path.read_text())['groups']],
        'registered per-update denominators differ')
    require(len(totals) == 86 and sum(totals) == 11085
        and result['optimizer_steps'] == 86 and result['supervised_tokens'] == 11085
        and result['target_tokens_per_update'] == totals and result['examples_per_update'] == [4] * 86
        and result['finite_gradients'] is result['finite_parameters'] is True
        and result['changed_adapter_tensors'] == 608
        and result['changed_attention_types'] == ['linear_attn', 'self_attn']
        and result['adapter_saved'] is False
        and result['initial_adapter_sha256'] != result['final_adapter_sha256']
        and all(len(result[k]) == 86 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
            for v in result[k]) for k in ['losses', 'gradient_norms_before_clipping']),
        'incomplete or invalid B3 training result')
    mem = result['memory']
    require(0 < mem['allocated_bytes.all.peak'] <= mem['reserved_bytes.all.peak'] <= CAP
        and mem['num_ooms'] == 0, 'B3 memory cap or OOM evidence differs')
    stats = result['streaming']
    require(stats['window_tokens'] == WINDOW and stats['max_input_tokens'] == 14996
        and stats['max_reference_tokens'] == WINDOW and stats['reference_calls'] > stats['calls'] > 0
        and stats['sublayer_checkpoint'] == {'decoder_layers': 64, 'wrapped_sublayers': 128,
            'calls': 44032, 'evaluations': 88064}
        and result['offload']['gpu_to_cpu_bytes'] > 0 and result['offload']['cpu_to_gpu_bytes'] > 0,
        'full B3 history, checkpoint or offload coverage differs')


def record_progress(state, event_path, phase, details):
    # train_groups calls update immediately BEFORE optimizer.step(). Only the
    # following group's first forward proves the preceding update completed.
    lower = state.get('completed_update_lower_bound', 0)
    if phase == 'forward' and details['microbatch'] == 0:
        lower = max(lower, details['optimizer_step'])
    event = {'epoch': time.time(), 'phase': phase, 'details': dict(details),
        'completed_update_lower_bound': lower, 'update_callback_precedes_optimizer_step': True}
    with event_path.open('a') as stream:
        stream.write(json.dumps(event, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    state.update(current_operation=phase, current_step=details, checked_epoch=event['epoch'],
        completed_update_lower_bound=lower)


def validate_saved_adapter(saved, final_sha, output):
    require(saved.get('status') == 'complete' and saved.get('complete') is saved.get('adapter_saved') is True
        and saved.get('tensor_count') == 608 and saved.get('total_parameter_count') == 23760896
        and saved.get('adapter_tensor_bytes') == 95043584 and saved.get('adapter_dtypes') == ['torch.float32']
        and saved.get('format') == 'safetensors' and saved.get('output_directory') == str(output)
        and saved.get('live_config_unchanged') is True
        and all(isinstance(saved.get(k), str) and re.fullmatch('[a-f0-9]{64}', saved[k])
            for k in ['source_adapter_sha256', 'source_peft_state_sha256', 'saved_peft_state_sha256'])
        and saved['source_adapter_sha256'] == final_sha
        and saved['source_peft_state_sha256'] == saved['saved_peft_state_sha256'],
        'saved adapter differs from final trained weights')
    files = saved.get('files')
    required = {'adapter_model.safetensors', 'adapter_config.json'}
    require(isinstance(files, dict) and required <= files.keys() <= required | {'README.md'}
        and output.is_dir() and not output.is_symlink()
        and {p.name for p in output.iterdir()} == set(files), 'saved adapter file inventory differs')
    for name, metadata in files.items():
        p = output / name
        require(not p.is_symlink() and p.is_file() and digest(p) == metadata.get('sha256')
            and p.stat().st_size == metadata.get('bytes'), 'saved adapter file changed: ' + name)
    require(files['adapter_model.safetensors']['bytes'] >= saved['adapter_tensor_bytes'] + 8,
        'saved adapter shorter than declared tensor payload and header')


def assess_complete(state):
    require(state['synthetic_only'] is False and state['baseline'] == 'B3_success_only_SFT'
        and state['benchmark_evaluation'] is False and state['method_experiences_used'] == 0
        and state['adapter_saved'] is True and state['task_learning_demonstrated'] is False,
        'B3 completion scope differs')
    validate_training_result(state['training_result'], state['target_tokens_per_update'])
    validate_storage(state['cpu_storage'])
    validate_storage(state['final_cpu_storage'])
    validate_data_summary(state['data'])
    require(state['cpu_storage'] == state['final_cpu_storage']
        and state['deterministic_settings'] == SETTINGS, 'B3 placement changed')
    validate_saved_adapter(state['saved_adapter'], state['training_result']['final_adapter_sha256'], ROOT / OUTPUT)
    return {'training_complete': True, 'baseline': 'B3_success_only_SFT', 'optimizer_steps': 86,
        'supervised_tokens': 11085, 'examples': 344, 'adapter_saved': True,
        'method_experiences_used': 0, 'task_learning_demonstrated': False,
        'scope': 'One-epoch success-trajectory baseline pilot; no heldout evaluation or recovery-method gain.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    allocator = configure_allocator('expandable')
    environment = configure_deterministic_environment()
    preflight(args.plan_sha256)
    if args.check:
        groups, data = load_verified_groups(ROOT, MODEL)
        validate_groups(groups, max_length=16384)
        print(json.dumps({'preflight': 'passed', 'id': ID, 'uses_gpu': False,
            'examples': data['examples'], 'updates': len(groups)}))
        return
    path, output = ROOT / f'artifacts/{ID}.json', ROOT / OUTPUT
    events = ROOT / f'artifacts/{ID}-events.jsonl'
    require(not path.exists() and not output.parent.exists() and not output.parent.is_symlink()
        and not events.exists() and not events.is_symlink(),
        'B3 attempt or output exists; no automatic retry or resume')
    from wll.gpu_policy import acquire_gpu, verify_cuda

    state = {'id': ID, 'phase': 'waiting_admission', 'pid': os.getpid(),
        'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
        'plan_sha256': args.plan_sha256, 'started_epoch': time.time(),
        'allocator_environment': allocator, 'deterministic_environment': environment,
        'synthetic_only': False, 'training': True, 'baseline': 'B3_success_only_SFT',
        'benchmark_evaluation': False, 'adapter_saved': False, 'task_learning_demonstrated': False,
        'method_experiences_used': 0, 'training_result': None, 'saved_adapter': None,
        'completed_update_lower_bound': 0, 'progress_journal': str(events.relative_to(ROOT)),
        'progress_scope': 'Update callback precedes optimizer.step; only next group start establishes prior completion.'}
    atomic_json(path, state)
    try:
        with events.open('x'):
            pass
        with acquire_gpu() as admission:
            state.update(phase='preparing_data', gpu=admission)
            atomic_json(path, state)
            # Importing the tokenizer imports torch. Admission must precede it.
            groups, state['data'] = load_verified_groups(ROOT, MODEL)
            state['target_tokens_per_update'] = validate_groups(groups, max_length=16384)
            import torch

            from scripts.nf4_loader import inspect_nf4, load_nf4
            from scripts.qwen38_training import prepare_hybrid_lora
            from scripts.save_hybrid_adapter import save_hybrid_adapter

            verify_cuda(torch)
            torch.set_num_threads(2)
            torch.manual_seed(17)
            state['phase'] = 'loading'
            atomic_json(path, state)
            model = load_nf4(MODEL, 'cuda:0')
            require(inspect_nf4(model, 'cuda:0')['linear4bit_layers'] == 496, 'wrong NF4 base')
            model = prepare_hybrid_lora(model, rank=8)
            require(model.peft_config['default'].base_model_name_or_path == str(MODEL),
                'prepared base identity differs; stop before training')
            require(sum(p.requires_grad for p in model.parameters()) == 608
                and sum(p.numel() for p in model.parameters() if p.requires_grad) == 23760896,
                'wrong LoRA coverage')
            text, head = prepare_text_cpu_storage(model), prepare_output_head_cpu_storage(model)
            placement = {**text, 'cpu_modules': text['cpu_modules'] + head['cpu_modules'],
                'cpu_parameter_bytes': text['cpu_parameter_bytes'] + head['cpu_parameter_bytes'],
                'released_cuda_parameter_bytes': text['released_cuda_parameter_bytes']
                    + head['released_cuda_parameter_bytes'], 'head': head}
            state['cpu_storage'] = inspect_storage(model, placement)
            state['loading_memory'] = memory(torch)
            require(state['loading_memory']['reserved_bytes.all.peak'] <= CAP, 'loading exceeds cap')
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            def progress(phase, details):
                record_progress(state, events, phase, details)
                atomic_json(path, state)

            state['phase'] = 'training'
            started = time.monotonic()
            with deterministic_attention('FLASH_ATTENTION') as settings:
                require(settings == SETTINGS, 'strict Flash settings differ')
                state['deterministic_settings'] = settings
                atomic_json(path, state)
                with streamed_delta_context(model, window_tokens=WINDOW) as streaming, ActivationOffload(model) as offload:
                    result = train_groups(model, groups, max_length=16384, progress=progress)
            torch.cuda.synchronize()
            result.update(seconds=time.monotonic() - started, memory=memory(torch),
                streaming=dict(streaming), offload=dict(offload.stats))
            state['training_result'] = result
            atomic_json(path, state)
            validate_training_result(result, state['target_tokens_per_update'])
            state['completed_update_lower_bound'] = 86
            state['progress_journal_sha256'] = digest(events)
            state['final_cpu_storage'] = inspect_storage(model, placement)
            state['phase'] = 'saving_adapter'
            atomic_json(path, state)
            output.parent.mkdir(mode=0o700)
            state['saved_adapter'] = save_hybrid_adapter(model, output,
                expected_base_model_name_or_path=str(MODEL))
            state['adapter_saved'] = True
            state['after_save_memory'] = memory(torch)
            require(state['after_save_memory']['reserved_bytes.all.peak'] <= CAP, 'saving exceeds cap')
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
