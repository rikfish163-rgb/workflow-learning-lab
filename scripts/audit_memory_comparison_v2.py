"""Reconcile raw three-arm requests, service events, admission and released resources."""
import subprocess

from scripts.audit_qwen38_followups import audit_events
from scripts.audit_qwen38_generation import parse_events
from scripts.report_memory_comparison import summarize_episodes
from scripts.run_memory_comparison import audit_batch, branch_schedule
from scripts.run_qwen38_memory_comparison_v2 import (
    ID,
    digest,
    expected_protocols,
    make_job,
    read,
    remote_queue,
    require,
    task_offset,
    validate_health,
    validate_live_service,
)
from wll.cli import read_episodes
from wll.gpu_idle import validate_ticket
from wll.protocol import fingerprint


def audit_generation(stage, tasks, candidates, expected, plan, index, events):
    report = audit_batch(stage / 'run', tasks, candidates, expected, task_offset=task_offset(plan, index))
    require(report['valid_for_aggregation'] and report == read(stage / 'report.json'), 'complete raw-audited batch required')
    requests = [row for trial in report['trials'] for row in trial['audit']['requests']]
    ledger = audit_events(requests, events, plan, report['reported_context_rejections'])
    require(ledger['returned_token_cost'] == report['returned_response_cost']
        and ledger['phase_response_cost']['draft'] == {'prompt_tokens': 0, 'completion_tokens': 0}
        and ledger['allocator_retries'] == 0, 'aggregate cost, hidden writer or allocator retry differs')
    return ledger


def container_absent(identifier):
    return not subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'id=' + identifier],
        text=True, timeout=15).strip()


def audit_stage(root, plan, inputs, tasks, candidates, index, stage):
    subset = [t for t in tasks if t.task_id in plan['batches'][index]]
    health = read(stage / 'health.json')
    validate_health(health, plan, index)
    expected = expected_protocols(subset, candidates, plan, health)
    require(read(stage / 'expected.json') == expected, 'external branch protocol differs')
    release, cleanup = read(stage / 'release.json'), read(stage / 'local-cleanup.json')
    require(release['owned'] and release['released'] and cleanup['local_client_exited']
        and cleanup['recorded_containers_absent'], 'resource release unconfirmed')
    remote = stage / 'remote-artifacts'
    manifest = read(remote / 'manifest.json')
    require(not manifest['missing'], 'remote service evidence missing')
    for name, sha in manifest['sha256'].items():
        require('/' not in name and name not in {'.', '..'} and digest(remote / name) == sha, 'remote service evidence changed')
    job_id = f'{ID}-b{index}'
    job, activation = read(stage / 'job.json'), read(stage / 'activation.json')
    require(job == make_job(index, job['owner'], job['plan_sha256']), 'job configuration differs')
    validate_live_service(activation, health)
    known = activation['model_pids']
    require(len(known) == 1, 'one model process required')
    info = read(remote / f'gpu-idle-job-{job_id}.json')
    require(info['status'] == 'finished' and info == release['job']
        and info['job_id'] == job_id and info['queue_sha256'] == digest(stage / 'job.json')
        and 0 <= info['finished_epoch'] - info['started_epoch'] <= plan['gpu_seconds_per_batch'] + 90,
        'finished job identity or duration differs')
    ticket = read(remote / f'gpu-idle-ticket-{job_id}.consumed.json')
    admission = read(remote / f'gpu9-admission-{known[0]}.json')
    validate_ticket(ticket, plan['gpu_policy'], info['gpu_index'], plan['gpu_policy_sha256'],
        ticket['boot_id'], ticket['qualified_at_monotonic'])
    gpu = health['backend']['gpu_policy']
    require(ticket == info['ticket'] == admission['ticket'] and ticket['job_id'] == job_id
        and admission['pid'] == known[0] and admission['mode'] == 'ten_minute_idle_only'
        and admission['physical_gpu_index'] == gpu['physical_gpu_index'] == info['gpu_index']
        and admission['gpu_uuid'] == gpu['gpu_uuid'] and admission['allocator_limit_mib'] == 20480
        and admission['snapshot']['idle'] is True and not admission['snapshot']['has_compute_process'],
        'fresh one-device admission differs')
    containers = read(stage / 'containers.json')
    require([(r['task_id'], r['arm']) for r in containers] == branch_schedule(subset, task_offset(plan, index))
        and len(containers) == len({r['id'] for r in containers}) == len(subset) * 3
        and all(container_absent(r['id']) for r in containers), 'fresh task containers missing or remain active')
    preflight = read(stage / 'cpu-environment-preflight.json')
    require(preflight['phase'] == 'complete' and preflight['container_absent'] and preflight['task_descriptions_match']
        and preflight['model_generation_calls'] == 0 and preflight['uses_gpu'] is False
        and preflight['task_ids'] == plan['batches'][index]
        and preflight['finished_epoch'] < info['started_epoch'], 'pre-admission CPU preflight differs')
    require(not subprocess.check_output(['docker', 'ps', '-aq', '--filter',
        'name=^/' + preflight['container_name'] + '$'], text=True, timeout=15).strip(), 'preflight container remains')
    fresh = remote_queue('status', job, known)
    require(fresh['released'] and fresh['job'] == info and not fresh['model_pids'], 'fresh GPU release unconfirmed')
    log = remote / f'gpu-idle-job-{job_id}.log'
    ledger = audit_generation(stage, subset, candidates, expected, plan, index, parse_events(log.read_text()))
    return {'audit': 'passed', 'script_sha256': digest(__file__), 'report_sha256': digest(stage / 'report.json'),
        'server_log_sha256': digest(log), 'job_wall_seconds': info['finished_epoch'] - info['started_epoch'],
        'single_GPU': info['gpu_index'], 'admission_idle_seconds': ticket['idle_seconds'],
        'task_containers_absent': len(subset) * 3, 'CPU_preflight_container_absent': True, 'GPU_released': True, **ledger,
        'scope': 'One complete bounded batch. Shared control billed once. Generation time overlaps service wall time; '
            'selection waits for all three batches and is confined to reused train utility.'}


def combine_study(root, plan, inputs, tasks, candidates, output):
    audits, episodes, resources, reports = [], [], [], []
    for index in range(len(plan['batches'])):
        stage = output / f'batch-{index}'
        audit = audit_stage(root, plan, inputs, tasks, candidates, index, stage)
        require(audit == read(stage / 'generation-audit.json'), 'saved batch audit differs')
        audits.append(audit)
        reports.append(read(stage / 'report.json'))
        resources.extend(read(stage / 'containers.json'))
        subset = [t for t in tasks if t.task_id in plan['batches'][index]]
        for task_id, arm in branch_schedule(subset, task_offset(plan, index)):
            branch, = read_episodes(stage / 'run/trials' / task_id / arm / 'episodes.jsonl')
            episodes.append(branch)
    require(len({r['id'] for r in resources}) == 27, 'environment reused across batches')
    summary = summarize_episodes(tasks, candidates, episodes, thresholds=plan['selection_thresholds'])
    require(summary['observed_step_cost'] == {k: sum(r['observed_step_cost'][k] for r in reports)
        for k in ('steps', 'prompt_tokens', 'completion_tokens')}, 'shared-control step accounting differs')
    selected = [arm for arm, passes in summary['candidate_gate_passes'].items() if passes]
    projections = {a: {'experience_id': candidates[a].experience_id,
        'rendered_sha256': inputs['candidates'][a]['rendered_sha256'],
        'raw_candidate_fingerprint': fingerprint(inputs['candidates'][a]['candidate'])} for a in selected}
    return {**summary, 'status': 'complete', 'selection_rule_activated': True,
        'utility_selected_projections': projections, 'accepted_experiences': 0,
        'raw_candidate_content_certified': False, 'full_hint_writer_instruction_violation_retained': True,
        'proposal_specific_utility_evidence': 'proposal_hint_memory' in selected and summary['source_contrast_positive'],
        'generated_requests': sum(a['generated_requests'] for a in audits),
        'context_rejections': sum(a['context_rejections'] for a in audits),
        'returned_response_cost': {k: sum(a['returned_token_cost'][k] for a in audits)
            for k in ('prompt_tokens', 'completion_tokens')},
        'GPU_job_seconds': sum(a['job_wall_seconds'] for a in audits),
        'generation_seconds': sum(a['generation_seconds'] for a in audits),
        'peak_reserved_bytes': max(a['peak_reserved_bytes'] for a in audits),
        'physical_GPU_by_batch': [a['single_GPU'] for a in audits],
        'unknown_branches': 0, 'unattempted_branches': 0, 'unknown_generated_response_cost': False,
        'all_resources_released': True, 'automatic_training': False,
        'batch_audit_sha256': [digest(output / f'batch-{i}/generation-audit.json') for i in range(len(plan['batches']))],
        'scope': 'Complete reused train utility for unchanged render projections. Empirical gates are correlated '
            'and do not establish familywise confidence, held-out efficacy, full writer compliance or parameter learning. '
            'Any selected projection requires separate content/training registration; no raw candidate promoted here. '
            'Earlier sources, critics and writer costs remain separate historical costs.'}
