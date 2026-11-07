"""Audit the original27-task B3 collection before private success-only SFT export."""
import argparse
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import scripts.run_qwen38_distill_baseline_002 as base
from scripts.audit_qwen38_generation import audit_events, parse_events
from scripts.encode_no_thinking_targets import audit_tokens
from scripts.load_memory_projection_v2 import project_file, snapshot_files
from scripts.report_timed_tasks import audit_run
from scripts.serve_qwen38_idle180 import DOWNLOAD
from scripts.serve_qwen38_idle180 import preflight as service_preflight
from wll.cli import read_episodes
from wll.gpu_idle import validate_ticket
from wll.prompts import initial_messages
from wll.protocol import fingerprint
from wll.storage import atomic_json
from wll.training import export_first_attempt_examples

CORPUS = 'artifacts/qwen38-distill-baseline-corpus-001'
LAUNCH = 'artifacts/distill-baseline-launch-002.json'
require = base.require
AUX_SOURCES = ['scripts/audit_distill_baseline.py', 'scripts/audit_qwen38_generation.py',
    'scripts/encode_no_thinking_targets.py', 'scripts/load_memory_projection_v2.py']


def validate_tokenizer_files(model, download):
    model = Path(model).resolve(strict=True)
    require(model == Path(download['model_directory']).resolve(strict=True), 'registered tokenizer directory required')

    def relevant(name):
        return Path(name).name in {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'} or name.endswith('.jinja')

    expected = {f['name']: f['sha256'] for f in download['files'] if relevant(f['name'])}
    actual = {str(p.relative_to(model)): base.digest(p) for p in model.rglob('*')
        if p.is_file() and relevant(str(p))}
    require({'tokenizer.json', 'tokenizer_config.json'} <= expected.keys() and actual == expected,
        'tokenizer files differ from registered model download')
    return actual


def validate_completion(state, launch, sha):
    require(state['phase'] == 'complete' and type(state['completed_batches']) is int
        and state['completed_batches'] == 5 and state['current_batch'] == 4
        and state['plan_sha256'] == launch['plan_sha256'] == sha
        and state['pid'] == launch['controller_pid']
        and state['start_ticks'] == launch['controller_start_ticks']
        and launch['controller_uses_GPU'] is False, 'complete five-batch controller identity required')


def curate_rows(episodes, tasks, plan):
    require(tasks and len({t['task_id'] for t in tasks}) == len(tasks)
        and [asdict(e.task) for e in episodes] == tasks, 'complete ordered original tasks required')
    for episode in episodes:
        require(episode.task.split == 'train' and type(episode.success) is bool
            and episode.branch == 'first' and episode.replayed_steps == 0
            and episode.experience_id is None and episode.seed == plan['seed']
            and episode.metadata.get('prompt_version') == plan['prompt_version'],
            'known no-memory train first attempts required')
    rows = export_first_attempt_examples(episodes)
    for episode in episodes:
        history = initial_messages(episode.task, plan['prompt_version'])
        selected = {r['step_index']: r for r in rows if r['task_id'] == episode.task.task_id}
        for index, step in enumerate(episode.steps):
            if index in selected:
                require(selected[index]['messages'][:-1] == history,
                    'export history differs from actual serving request; refuse silent rewriting')
            history = history + [{'role': 'assistant', 'content': step.raw_response},
                                 {'role': 'user', 'content': step.observation}]
    return {'attempts': len(episodes), 'successes': sum(e.success for e in episodes),
        'scenario_count': len({e.task.scenario_id for e in episodes}),
        'successful_scenarios': sorted({e.task.scenario_id for e in episodes if e.success}),
        'exported_targets': len(rows),
        'excluded_error_targets': sum(s.observation.startswith(('Execution failed', 'Generation failed:'))
            for e in episodes if e.success for s in e.steps),
        'failed_task_ids_excluded': [e.task.task_id for e in episodes if not e.success],
        'episode_hashes': [fingerprint(e.to_dict()) for e in episodes],
        'all_exported_histories_match_actual_requests': True if rows else None}, rows


def container_absent(filter_value):
    require(not subprocess.check_output(['docker', 'ps', '-aq', '--filter', filter_value],
        text=True, timeout=15).strip(), 'recorded container remains active or retained')


def audit_stage(root, plan, tasks, index):
    stage = root / 'artifacts' / base.ID / f'batch-{index}'
    before = snapshot_files(stage)
    health, release = base.read(stage / 'health.json'), base.read(stage / 'release.json')
    base.validate_health(health, plan, index)
    activation = base.read(stage / 'activation.json')
    base.validate_live_service(activation, health)
    expected = base.expected_protocol(plan, base.batch_rows(tasks, index), health)
    require(expected == base.read(stage / 'expected.json'), 'external task protocol differs')
    report = audit_run(stage / 'run', expected)
    require(report == base.read(stage / 'report.json') and report['valid_for_comparison']
        and report['completed'] == len(plan['batches'][index]) and report['unknown_outcomes'] == 0,
        'complete raw-audited batch required')
    require(release['owned'] and release['released'] and not release['model_pids'], 'batch not released')
    job = base.read(stage / 'job.json')
    require(job == base.make_job(index, job['owner'], base.digest(root / base.PLAN)), 'job differs')
    remote = stage / 'remote-artifacts'
    manifest = base.read(remote / 'manifest.json')
    require(not manifest['missing'], 'remote evidence missing')
    require(set(manifest['sha256']) == {f"gpu-idle-job-{job['id']}.json",
        f"gpu-idle-job-{job['id']}.log", f"gpu-idle-ticket-{job['id']}.consumed.json",
        *[f'gpu9-admission-{pid}.json' for pid in activation['model_pids']]}, 'remote file coverage differs')
    for name, sha in manifest['sha256'].items():
        require(Path(name).name == name and base.digest(remote / name) == sha, 'remote evidence changed')
    info = base.read(remote / f"gpu-idle-job-{job['id']}.json")
    require(info == release['job'] and info['status'] == 'finished', 'GPU job not terminal or differs')
    require(len(activation['model_pids']) == 1, 'one model process required')
    pid = activation['model_pids'][0]
    admission = base.read(remote / f'gpu9-admission-{pid}.json')
    ticket = base.read(remote / f"gpu-idle-ticket-{job['id']}.consumed.json")
    require(ticket == info['ticket'] == admission['ticket'] and ticket['job_id'] == job['id'], 'ticket differs')
    validate_ticket(ticket, plan['gpu_policy'], info['gpu_index'], plan['gpu_policy_sha256'],
        ticket['boot_id'], ticket['qualified_at_monotonic'])
    require(admission['mode'] == 'continuous_idle_only' and admission['required_idle_seconds'] == 180
        and admission['allocator_limit_mib'] == 20480 and admission['pid'] == pid
        and admission['gpu_uuid'] == ticket['gpu_uuid'] == health['backend']['gpu_policy']['gpu_uuid']
        and admission['physical_gpu_index'] == info['gpu_index'] == health['backend']['gpu_policy']['physical_gpu_index']
        and admission['snapshot']['idle'] and not admission['snapshot']['has_compute_process'],
        'single-device admission differs')
    cleanup = base.read(stage / 'local-cleanup.json')
    containers = [json.loads(line) for line in (stage / 'containers.jsonl').read_text().splitlines()]
    ids = [r['id'] for r in containers]
    require(len(ids) == len(set(ids)) == len(plan['batches'][index])
        and cleanup['local_client_exited'] and cleanup['recorded_containers_absent']
        and set(cleanup['checked_identifiers']) == set(ids), 'task container identity or cleanup differs')
    preflight = base.read(stage / 'cpu-environment-preflight.json')
    require(preflight['phase'] == 'complete' and preflight['container_absent']
        and preflight['task_descriptions_match'] and preflight['model_generation_calls'] == 0
        and preflight['uses_gpu'] is False and preflight['task_ids'] == plan['batches'][index]
        and preflight['finished_epoch'] < info['started_epoch'], 'CPU preflight differs')
    for identifier in ids:
        container_absent('id=' + identifier)
    container_absent('name=^/' + preflight['container_name'] + '$')
    fresh = base.remote_queue('status', job, [pid])
    require(fresh['monitor_live'] and fresh['released'] and fresh['job'] == info and not fresh['model_pids'], 'fresh release differs')
    episodes = read_episodes(stage / 'run/episodes.jsonl')
    log = remote / f"gpu-idle-job-{job['id']}.log"
    ledger = audit_events([e.to_dict() for e in episodes], parse_events(log.read_text()),
        plan, health['model'], report['generation_counters_from_summary'])
    require(ledger['returned_token_cost'] == report['returned_response_cost'], 'returned costs differ')
    require(snapshot_files(stage) == before, 'batch changed during audit')
    return {'batch': index, 'report': report, 'release': release, 'generation': ledger,
        'GPU_job_seconds': info['finished_epoch'] - info['started_epoch'], 'physical_gpu': info['gpu_index'],
        'admission_idle_seconds': ticket['idle_seconds'], 'task_container_ids': ids,
        'preflight_container': preflight['container_name'], 'fresh_release': fresh,
        'input_sha256': {str((stage / name).relative_to(root)): sha for name, sha in before.items()}}, episodes


def audit_complete(root, sha):
    root = Path(root).resolve(strict=True)
    plan, tasks = base.load_registration(root, root / base.PLAN, sha)
    auxiliary = {name: base.digest(root / name) for name in AUX_SOURCES}
    output = root / 'artifacts' / base.ID
    state, launch = base.read(output / 'status.json'), base.read(root / LAUNCH)
    validate_completion(state, launch, sha)
    proc = Path('/proc') / str(state['pid'])
    if proc.exists():
        fields = (proc / 'stat').read_text().split(')', 1)[1].split()
        require(fields[19] != state['start_ticks'] or fields[0] in {'Z', 'X'}, 'controller still active')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', plan['port']))
    before = snapshot_files(output)
    audits, episodes = [], []
    for index in range(5):
        audit, batch = audit_stage(root, plan, tasks, index)
        audits.append(audit)
        episodes.extend(batch)
    result = base.combine_batches([a['report'] for a in audits], [a['release'] for a in audits])
    require(result == state['result'] == base.read(output / 'report.json'), 'full collection summary differs')
    ids = [c for a in audits for c in a['task_container_ids']]
    require(len(ids) == len(set(ids)) == 27
        and len({a['preflight_container'] for a in audits}) == 5, 'cross-batch environment reused')
    corpus, rows = curate_rows(episodes, tasks, plan)
    require(corpus['scenario_count'] == 9 and corpus['successes'] == result['successes'], 'corpus scope differs')
    tokens = {k: sum(a['generation']['returned_token_cost'][k] for a in audits)
        for k in ('prompt_tokens', 'completion_tokens')}
    require(tokens == result['returned_response_cost'], 'all-batch generation accounting differs')
    require(snapshot_files(output) == before and base.load_registration(root, root / base.PLAN, sha) == (plan, tasks),
        'registered collection changed during audit')
    require(auxiliary == {name: base.digest(root / name) for name in AUX_SOURCES}, 'auditor sources changed')
    return {'phase': 'complete', 'checked_epoch': time.time(), 'plan_sha256': sha,
        'source_sha256': auxiliary,
        'auditor_sha256': base.digest(__file__), 'encoder_sha256': base.digest(root / 'scripts/encode_no_thinking_targets.py'),
        'result': result, 'batches': audits, 'curation': corpus,
        'generated_requests': sum(a['generation']['generated_requests'] for a in audits),
        'context_rejections': sum(a['generation']['context_rejections'] for a in audits),
        'collection_GPU_job_seconds': sum(a['GPU_job_seconds'] for a in audits),
        'prior_infrastructure_GPU_job_seconds': plan['prior_infrastructure_cost_seconds'],
        'total_GPU_job_seconds_including_prior_infrastructure': sum(a['GPU_job_seconds'] for a in audits)
            + plan['prior_infrastructure_cost_seconds'],
        'all_resources_released': True, 'unknown_generated_response_cost': False,
        'new_training_started': False, 'accepted_experiences': 0,
        'input_sha256': {**{str((output / n).relative_to(root)): h for n, h in before.items()},
                         LAUNCH: base.digest(root / LAUNCH), base.PLAN: sha},
        'scope': 'Original train/distill B3 success-only corpus; failures retained and billed. '
            'No length selection or truncation. Not recovery-method data, heldout evaluation or training gain.'}, rows, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--tokenizer-path')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only visibility required')
    target = base.ROOT / CORPUS
    if args.tokenizer_path:
        proof = base.read(target / 'audit.json')
        require(proof['phase'] == 'complete' and proof['plan_sha256'] == args.plan_sha256
            and proof['auditor_sha256'] == base.digest(__file__)
            and proof['encoder_sha256'] == base.digest(base.ROOT / 'scripts/encode_no_thinking_targets.py')
            and proof['sft_sha256'] == base.digest(target / 'sft.jsonl'), 'audited corpus or encoder changed')
        plan, _ = base.load_registration(base.ROOT, base.ROOT / base.PLAN, args.plan_sha256)
        # CPU-only checks of the same interpreter, tokenizer runtime and model provenance.
        service_preflight(plan['service_plan_sha256'])
        for name, sha in {**proof['input_sha256'], **proof['source_sha256']}.items():
            require(base.digest(project_file(base.ROOT, name)) == sha, 'collected provenance changed: ' + name)
        require(plan['input_sha256'][DOWNLOAD] == base.digest(base.ROOT / DOWNLOAD), 'model download binding differs')
        tokenizer_files = validate_tokenizer_files(args.tokenizer_path, base.read(base.ROOT / DOWNLOAD))
        require(not (target / 'token-audit.json').exists(), 'token audit already exists')
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, trust_remote_code=False)
        rows = [json.loads(line) for line in (target / 'sft.jsonl').read_text().splitlines()]
        result = audit_tokens(rows, tokenizer)
        result.update(sft_sha256=proof['sft_sha256'], encoder_sha256=proof['encoder_sha256'],
            tokenizer_path=str(Path(args.tokenizer_path).resolve()),
            tokenizer_files_sha256=tokenizer_files)
        atomic_json(target / 'token-audit.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in {'samples', 'tokenizer_files_sha256'}}))
        return
    require(not target.exists(), 'corpus output exists; do not overwrite')
    proof, rows, episodes = audit_complete(base.ROOT, args.plan_sha256)
    target.mkdir()
    for name, records in [('sft.jsonl', rows), ('episodes.jsonl', [e.to_dict() for e in episodes])]:
        with (target / name).open('x') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    proof.update(sft_sha256=base.digest(target / 'sft.jsonl'), episodes_sha256=base.digest(target / 'episodes.jsonl'))
    atomic_json(target / 'audit.json', proof)
    print(json.dumps({'phase': 'complete', **proof['curation']}))


if __name__ == '__main__':
    main()
