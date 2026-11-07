"""Audit the 002 + partial 003 + 004 B3 collection before private SFT export.

This is a CPU controller. The audit reads the stored journals, checks current
release through the existing read-only status operation, and never queues work.
The historical 003 report remains incomplete: its startup failure is retained
as a separate operational record, not as an actor outcome.
"""

import argparse
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

import scripts.audit_distill_baseline as parent_audit
import scripts.run_qwen38_distill_baseline_004 as base
from scripts.audit_qwen38_generation import audit_events, parse_events
from scripts.encode_no_thinking_targets import audit_tokens
from scripts.load_memory_projection_v2 import project_file, snapshot_files
from scripts.report_timed_tasks import audit_run
from scripts.serve_qwen38_idle180 import DOWNLOAD
from scripts.serve_qwen38_idle180 import preflight as service_preflight
from wll.cli import read_episodes
from wll.gpu_idle import validate_ticket
from wll.storage import atomic_json

CORPUS = 'artifacts/qwen38-distill-baseline-corpus-004'
LAUNCH = 'artifacts/distill-baseline-launch-004.json'
PRIOR_INFRASTRUCTURE_SECONDS = 78.92770290374756
PARTIAL_JOB_SECONDS = 1214.3943722248077
require = base.require
curate_rows = parent_audit.curate_rows
validate_tokenizer_files = parent_audit.validate_tokenizer_files
container_absent = parent_audit.container_absent
AUX_SOURCES = {
    'scripts/audit_distill_baseline.py': 'c739641b3605d08da8d2617d71665f0d1889628d1d568e5dc3cefe5f392db74d',
    'scripts/audit_distill_baseline_003.py': '21bf0e70000bf5d050f50d84952ae93b42d59e9ef942328b653efb5137f5a9c6',
    'scripts/audit_qwen38_generation.py': '0636e02018c340d4eda04a53d7ae23d5e5376b3e148cf62531740f88f668f667',
    'scripts/encode_no_thinking_targets.py': 'b56cb336ab991024e2d4ffaa77409cb1f061dc5c43d40fcedba4fc196f81a3f8',
    'scripts/load_memory_projection_v2.py': '4630bd5c263079275efd841f55ba7a9a9030261cd1b47abad37b57d7cc47f717',
}


def validate_hashes(root, hashes):
    root = Path(root).resolve(strict=True)
    for name, sha in hashes.items():
        require(base.digest(project_file(root, name)) == sha, 'source or evidence changed: ' + name)


def validate_completion(state, launch, sha):
    require(state['phase'] == 'complete' and type(state['completed_batches']) is int
        and state['completed_batches'] == 6 and state['retained_batches'] == 4
        and state['new_completed_batches'] == 2 and state['current_batch'] == 4
        and state['plan_sha256'] == launch['plan_sha256'] == sha
        and launch['id'] == base.ID and state['pid'] == launch['controller_pid']
        and state['start_ticks'] == launch['controller_start_ticks']
        and launch['controller_uses_GPU'] is False, 'complete six-segment controller identity required')


def read_controller_snapshot(root, output):
    """Bind the parsed state and launch record to snapshots taken before parsing."""
    before = snapshot_files(output)
    launch_path = Path(root) / LAUNCH
    launch_sha = base.digest(launch_path)
    state_raw = (output / 'status.json').read_bytes()
    launch_raw = launch_path.read_bytes()
    require(hashlib.sha256(state_raw).hexdigest() == before.get('status.json')
        and hashlib.sha256(launch_raw).hexdigest() == launch_sha, 'controller snapshot changed')
    return json.loads(state_raw), json.loads(launch_raw), before, launch_sha


def validate_freeze_check(freeze, plan, plan_sha, preflight, job_started):
    """Require the local and remote source checks before CPU and GPU work."""
    remote = freeze['remote']
    require(freeze['plan_sha256'] == plan_sha and freeze['local_files_match'] is True
        and remote['preflight'] == 'passed' and remote['uses_gpu'] is False
        and type(remote['tasks']) is int and remote['tasks'] == 27
        and type(remote['source_bindings']) is int
        and remote['source_bindings'] == len(plan['source_sha256']), 'freeze preflight differs')
    times = (freeze['checked_epoch'], preflight['started_epoch'], preflight['finished_epoch'], job_started)
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in times)
        and times[0] <= times[1] <= times[2] < times[3], 'freeze or CPU preflight timing differs')


def validate_token_corpus(root):
    """Reject symlinked corpus directories, inputs, or token audit destination."""
    root = Path(root).resolve(strict=True)
    artifacts = root / 'artifacts'
    target = root / CORPUS
    require(artifacts.is_dir() and not artifacts.is_symlink() and target.is_dir()
        and not target.is_symlink() and target.resolve(strict=True).parent == artifacts,
        'corpus directory differs or is a symlink')
    for name in ('audit.json', 'sft.jsonl', 'episodes.jsonl'):
        path = target / name
        require(not path.is_symlink() and path.is_file() and path.resolve(strict=True).parent == target,
            'corpus input missing or is a symlink: ' + name)
    token_output = target / 'token-audit.json'
    require(not token_output.exists() and not token_output.is_symlink(), 'corpus token audit already exists')
    return target


def validate_partial_evidence(ids, evidence, report, episodes, attempts, steps, responses, ledger, seconds):
    """Bind the incomplete 003 batch to its immutable pre-actor incident."""
    base.validate_resume_evidence(evidence, ids)
    require(evidence['plan_sha256'] == base.PREVIOUS_SHA
        and evidence['completed_prior_tasks'] == 18 and evidence['new_completed_tasks'] == 5
        and evidence['new_successes'] == 5 and evidence['total_known_outcomes'] == 23
        and evidence['total_known_successes'] == 19 and evidence['total_known_failures'] == 4,
        'historical known outcomes differ')
    require(report == evidence['partial_report'] and report['audit'] == 'passed'
        and report['status'] == 'incomplete' and report['scheduled'] == report['attempted'] == 6
        and report['completed'] == report['successes'] == 5 and report['unknown_outcomes'] == 1
        and report['unattempted'] == 0 and report['valid_for_comparison'] is False,
        'partial report was changed or incorrectly marked complete')
    require([e.task.task_id for e in episodes] == ids[18:23]
        and all(e.success is True for e in episodes)
        and [a['task_id'] for a in attempts] == ids[18:24]
        and attempts[-1] == evidence['unknown_original_attempt'], 'partial attempt identity differs')
    require(steps and responses and all(row['task_id'] in ids[18:23] for row in steps + responses),
        'unknown startup has an actor action or response')
    require(ledger == evidence['generation'] and ledger['generated_requests'] == 69
        and ledger['context_rejections'] == 0 and ledger['unknown_generated_response_cost'] is False
        and len(ledger['requests']) == 69
        and all(row['task_id'] in ids[18:23] for row in ledger['requests'])
        and ledger['returned_token_cost'] == report['returned_response_cost'],
        'partial generation ledger differs or has an unknown task request')
    require(type(seconds) in (int, float) and math.isfinite(seconds)
        and seconds == evidence['GPU_job_seconds'] == PARTIAL_JOB_SECONDS,
        'partial mixed-job wall cost differs')
    return dict(attempts[-1])


def validate_merged(tasks, audits, episode_groups, result, prior_seconds):
    """Independently check the six segments, all actor outcomes, and cost views."""
    ids = [t['task_id'] for t in tasks]
    require(len(ids) == len(set(ids)) == 27 and len(audits) == len(episode_groups) == 6,
        'original 27 tasks and six segments required')
    require(type(prior_seconds) in (int, float) and math.isfinite(prior_seconds)
        and prior_seconds == PRIOR_INFRASTRUCTURE_SECONDS, 'prior infrastructure costs differ')
    segments = [ids[:6], ids[6:12], ids[12:18], ids[18:23], ids[23:24], ids[24:]]
    totals = {'observed_step_cost': dict.fromkeys(('steps', 'prompt_tokens', 'completion_tokens'), 0),
              'returned_response_cost': dict.fromkeys(('prompt_tokens', 'completion_tokens'), 0),
              'generation_counters_from_summary': dict.fromkeys(('generation_calls',
                  'generation_calls_without_response', 'generation_calls_with_unusable_usage'), 0)}
    successes = operational = generated = rejected = 0
    scenarios, seconds = set(), []
    for index, (expected_ids, item, episodes) in enumerate(zip(segments, audits, episode_groups, strict=True)):
        report, release, ledger = item['report'], item['release'], item['generation']
        partial = index == 3
        require([e.task.task_id for e in episodes] == expected_ids
            and all(type(e.success) is bool for e in episodes), 'missing, duplicate or unknown actor outcome')
        completed_successes = sum(e.success for e in episodes)
        group_scenarios = {e.task.task_id.rsplit('_', 1)[0] for e in episodes if e.success}
        require(report['audit'] == 'passed' and report['status'] == ('incomplete' if partial else 'complete')
            and report['scheduled'] == report['attempted'] == (6 if partial else len(expected_ids))
            and report['completed'] == len(expected_ids) and report['successes'] == completed_successes
            and report['unknown_outcomes'] == int(partial) and report['unattempted'] == 0
            and report['valid_for_comparison'] is (not partial)
            and report['successful_scenarios'] == sorted(group_scenarios), 'segment result differs')
        require(release['released'] is True and release['owned'] is True and not release['model_pids'],
            'segment release missing')
        require(ledger['unknown_generated_response_cost'] is False
            and ledger['returned_token_cost'] == report['returned_response_cost'], 'generation costs differ')
        for field, costs in totals.items():
            for key in costs:
                value = report[field][key]
                require(type(value) is int and value >= 0, 'missing or invalid cost')
                costs[key] += value
        for name in ('generated_requests', 'context_rejections'):
            require(type(ledger[name]) is int and ledger[name] >= 0, 'missing generation accounting')
        counters = report['generation_counters_from_summary']
        require(counters['generation_calls'] == ledger['generated_requests'] + ledger['context_rejections']
            and counters['generation_calls_without_response'] == ledger['context_rejections']
            and counters['generation_calls_with_unusable_usage'] == 0, 'unaccounted generation call')
        wall = item['GPU_job_seconds']
        require(type(wall) in (int, float) and math.isfinite(wall) and wall > 0,
            'missing or invalid GPU job wall cost')
        require(not partial or wall == PARTIAL_JOB_SECONDS, 'partial mixed-job wall cost differs')
        successes += completed_successes
        operational += report['attempted']
        generated += ledger['generated_requests']
        rejected += ledger['context_rejections']
        scenarios.update(group_scenarios)
        seconds.append(wall)
    require(sum(item['report']['successes'] for item in audits[:3]) == 14
        and audits[3]['report']['successes'] == 5, 'known original outcomes differ')
    require(operational == 28 and result['completed'] == 27 and result['successes'] == successes
        and result['success_rate'] == successes / 27 and result['scenario_count'] == 9
        and result['successful_scenarios'] == sorted(scenarios) and result['unknown_outcomes'] == 0
        and result['prior_environment_startup_attempts_without_actor'] == 1
        and result['operational_attempt_records'] == operational
        and result['automatic_training_started'] is False and result['accepted_experiences'] == 0
        and all(result[k] == v for k, v in totals.items()), 'merged summary or total costs differ')
    return {'known_actor_outcomes': 27, 'operational_attempt_records': operational,
        'generated_requests': generated, 'context_rejections': rejected,
        'collection_GPU_job_seconds': sum(seconds), 'prior_infrastructure_GPU_job_seconds': prior_seconds,
        'total_GPU_job_seconds_including_prior_infrastructure': sum(seconds) + prior_seconds}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit_segment(root, plan, tasks, module, index, partial=False):
    stage = root / 'artifacts' / module.ID / f'batch-{index}'
    before = snapshot_files(stage)
    health, release = base.read(stage / 'health.json'), base.read(stage / 'release.json')
    module.validate_health(health, plan, index)
    activation = base.read(stage / 'activation.json')
    module.validate_live_service(activation, health)
    batch_tasks = module.batch_rows(tasks, index)
    expected = module.expected_protocol(plan, batch_tasks, health)
    require(expected == base.read(stage / 'expected.json'), 'external task protocol differs')
    report = audit_run(stage / 'run', expected)
    require(report == base.read(stage / 'report.json'), 'raw-audited report differs')
    if not partial:
        require(report['audit'] == 'passed' and report['status'] == 'complete'
            and report['valid_for_comparison'] is True and report['completed'] == len(batch_tasks)
            and report['unknown_outcomes'] == 0, 'complete raw-audited segment required')
    require(release['owned'] is True and release['released'] is True and not release['model_pids']
        and not release['model_processes'] and release['monitor_live'] is True, 'segment not released')
    job = base.read(stage / 'job.json')
    plan_sha = base.digest(root / module.PLAN)
    require(job == module.make_job(index, job['owner'], plan_sha), 'job identity differs')
    remote = stage / 'remote-artifacts'
    manifest = base.read(remote / 'manifest.json')
    pids = activation['model_pids']
    require(len(pids) == 1 and not manifest['missing'], 'single admitted process and all remote evidence required')
    expected_files = {f"gpu-idle-job-{job['id']}.json", f"gpu-idle-job-{job['id']}.log",
                      f"gpu-idle-ticket-{job['id']}.consumed.json", f'gpu9-admission-{pids[0]}.json'}
    require(set(manifest['sha256']) == expected_files, 'remote file coverage differs')
    for name, sha in manifest['sha256'].items():
        require(Path(name).name == name and base.digest(remote / name) == sha, 'remote evidence changed')
    info = base.read(remote / f"gpu-idle-job-{job['id']}.json")
    require(info == release['job'] and info['status'] == 'finished' and info['job_id'] == job['id']
        and info['queue_sha256'] == base.digest(stage / 'job.json')
        and all(activation['job'][k] == info[k] for k in
            ('job_id', 'gpu_index', 'queue_sha256', 'started_epoch', 'ticket', 'process_group', 'start_ticks')),
        'GPU job identity or terminal evidence differs')
    start, finish = info['started_epoch'], info['finished_epoch']
    require(type(start) in (int, float) and type(finish) in (int, float)
        and math.isfinite(start) and math.isfinite(finish) and finish > start, 'invalid GPU job timing')
    pid = pids[0]
    admission = base.read(remote / f'gpu9-admission-{pid}.json')
    ticket = base.read(remote / f"gpu-idle-ticket-{job['id']}.consumed.json")
    require(ticket == info['ticket'] == admission['ticket'] and ticket['job_id'] == job['id'], 'ticket differs')
    validate_ticket(ticket, plan['gpu_policy'], info['gpu_index'], plan['gpu_policy_sha256'],
        ticket['boot_id'], ticket['qualified_at_monotonic'])
    require(admission['mode'] == 'continuous_idle_only' and admission['required_idle_seconds'] == 180
        and admission['allocator_limit_mib'] == 20480 and admission['pid'] == pid
        and admission['gpu_uuid'] == ticket['gpu_uuid'] == health['backend']['gpu_policy']['gpu_uuid']
        and admission['physical_gpu_index'] == info['gpu_index'] == health['backend']['gpu_policy']['physical_gpu_index']
        and admission['snapshot']['idle'] is True and not admission['snapshot']['has_compute_process'],
        'single-device admission differs')
    cleanup = base.read(stage / 'local-cleanup.json')
    ids = [row['id'] for row in read_lines(stage / 'containers.jsonl')]
    count = 5 if partial else len(batch_tasks)
    require(len(ids) == len(set(ids)) == count and cleanup['local_client_exited'] is True
        and cleanup['recorded_containers_absent'] is True
        and set(cleanup['checked_identifiers']) == set(ids), 'container identity or cleanup differs')
    preflight = base.read(stage / 'cpu-environment-preflight.json')
    require(preflight['phase'] == 'complete' and preflight['container_absent'] is True
        and preflight['task_descriptions_match'] is True and preflight['model_generation_calls'] == 0
        and preflight['uses_gpu'] is False and preflight['task_ids'] == [t['task_id'] for t in batch_tasks]
        and preflight['finished_epoch'] < start, 'CPU environment preflight differs')
    validate_freeze_check(base.read(stage / 'freeze-check.json'), plan, plan_sha, preflight, start)
    for identifier in ids:
        container_absent('id=' + identifier)
    container_absent('name=^/' + preflight['container_name'] + '$')
    fresh = module.remote_queue('status', job, [pid])
    require(fresh['monitor_live'] is True and fresh['released'] is True and fresh['job'] == info
        and not fresh['model_pids'] and not fresh['model_processes'], 'fresh release differs')
    episodes = read_episodes(stage / 'run/episodes.jsonl')
    log = remote / f"gpu-idle-job-{job['id']}.log"
    ledger = audit_events([e.to_dict() for e in episodes], parse_events(log.read_text()),
        plan, health['model'], report['generation_counters_from_summary'])
    require(ledger['returned_token_cost'] == report['returned_response_cost']
        and ledger['unknown_generated_response_cost'] is False, 'returned generation costs differ')
    unknown = None
    if partial:
        unknown = validate_partial_evidence(plan['task_ids'], base.read(root / base.INCIDENT), report, episodes,
            read_lines(stage / 'run/attempts.jsonl'), read_lines(stage / 'run/steps.jsonl'),
            read_lines(stage / 'run/responses.jsonl'), ledger, finish - start)
    require(snapshot_files(stage) == before, 'segment changed during audit')
    return {'study': module.ID, 'batch': index, 'report': report, 'release': release, 'generation': ledger,
        'GPU_job_seconds': finish - start, 'physical_gpu': info['gpu_index'],
        'admission_idle_seconds': ticket['idle_seconds'], 'task_container_ids': ids,
        'preflight_container': preflight['container_name'], 'fresh_release': fresh,
        'unknown_startup_attempt': unknown,
        'input_sha256': {str((stage / name).relative_to(root)): sha for name, sha in before.items()}}, episodes


def audit_complete(root, sha, expected_auditor_sha):
    root = Path(root).resolve(strict=True)
    auditor_sha = base.digest(__file__)
    require(auditor_sha == expected_auditor_sha, 'auditor source changed')
    validate_hashes(root, AUX_SOURCES)
    plan, tasks = base.load_registration(root, root / base.PLAN, sha)
    auxiliary = dict(AUX_SOURCES)
    output = root / 'artifacts' / base.ID
    state, launch, before, launch_sha = read_controller_snapshot(root, output)
    validate_completion(state, launch, sha)
    proc = Path('/proc') / str(state['pid'])
    if proc.exists():
        fields = (proc / 'stat').read_text().split(')', 1)[1].split()
        require(fields[19] != state['start_ticks'] or fields[0] in {'Z', 'X'}, 'controller still active')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', plan['port']))
    incident_sha = base.digest(root / base.INCIDENT)
    require(incident_sha == base.INCIDENT_SHA, '003 incident evidence changed')
    original_plan = base.read(root / base.ORIGINAL_PLAN)
    previous_plan = base.read(root / base.PREVIOUS_PLAN)
    schedule = [(base.original, original_plan, i, False) for i in range(3)]
    schedule += [(base.previous, previous_plan, 3, True), (base, plan, 3, False), (base, plan, 4, False)]
    audits, groups = [], []
    for module, segment_plan, index, partial in schedule:
        item, episodes = audit_segment(root, segment_plan, tasks, module, index, partial)
        audits.append(item)
        groups.append(episodes)
    result = base.combine_batches([a['report'] for a in audits], [a['release'] for a in audits])
    require(result == state['result'] == base.read(output / 'report.json'), 'controller summary differs')
    totals = validate_merged(tasks, audits, groups, result, plan['prior_infrastructure_cost_seconds'])
    ids = [identifier for item in audits for identifier in item['task_container_ids']]
    require(len(ids) == len(set(ids)) == 27 and len({a['preflight_container'] for a in audits}) == 6,
        'cross-segment environment reused')
    container_absent('name=^/wll-')
    episodes = [episode for group in groups for episode in group]
    corpus, rows = curate_rows(episodes, tasks, plan)
    require(corpus['attempts'] == 27 and corpus['scenario_count'] == 9
        and corpus['successes'] == result['successes'], 'curated corpus differs from all known outcomes')
    input_hashes = {name: digest for item in audits for name, digest in item['input_sha256'].items()}
    input_hashes.update({str((output / name).relative_to(root)): digest for name, digest in before.items()})
    input_hashes.update({LAUNCH: launch_sha, base.PLAN: sha, base.INCIDENT: incident_sha})
    validate_hashes(root, input_hashes)
    validate_hashes(root, auxiliary)
    for module, _, index, _ in schedule:
        stage = root / 'artifacts' / module.ID / f'batch-{index}'
        stage_hashes = {str((stage / name).relative_to(root)): value for name, value in snapshot_files(stage).items()}
        require(stage_hashes == {name: value for name, value in input_hashes.items()
            if name.startswith(str(stage.relative_to(root)) + '/')}, 'segment files changed during audit')
    require(snapshot_files(output) == before and base.digest(__file__) == auditor_sha
        and base.load_registration(root, root / base.PLAN, sha) == (plan, tasks),
        'registered collection or auditor changed during audit')
    return {'phase': 'complete', 'checked_epoch': time.time(), 'plan_sha256': sha,
        'source_sha256': auxiliary, 'auditor_sha256': auditor_sha,
        'encoder_sha256': base.digest(root / 'scripts/encode_no_thinking_targets.py'),
        'result': result, 'segments': audits, 'curation': corpus, **totals,
        'prior_unknown_startup_attempt': audits[3]['unknown_startup_attempt'],
        'all_resources_released': True, 'unknown_generated_response_cost': False,
        'new_training_started': False, 'accepted_experiences': 0, 'input_sha256': input_hashes,
        'scope': 'Private B3 train/distill, 27 known first actor outcomes plus a separate pre-actor startup '
            'failure. The incomplete 003 report and all 69 generation events and its whole mixed-job '
            'wall are retained. Cost views overlap. Context-rejection CPU tokenization is not measured. '
            'No truncation, training, recovery-method, heldout or efficacy claim.'}, rows, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--auditor-sha256', required=True)
    parser.add_argument('--tokenizer-path')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only visibility required')
    target = base.ROOT / CORPUS
    if args.tokenizer_path:
        target = validate_token_corpus(base.ROOT)
        corpus_hashes = {name: base.digest(target / name) for name in ('audit.json', 'sft.jsonl', 'episodes.jsonl')}
        proof = base.read(target / 'audit.json')
        require(proof['phase'] == 'complete' and proof['plan_sha256'] == args.plan_sha256
            and proof['auditor_sha256'] == args.auditor_sha256 == base.digest(__file__)
            and proof['encoder_sha256'] == base.digest(base.ROOT / 'scripts/encode_no_thinking_targets.py')
            and proof['sft_sha256'] == base.digest(target / 'sft.jsonl')
            and proof['episodes_sha256'] == base.digest(target / 'episodes.jsonl'), 'audited corpus changed')
        plan, _ = base.load_registration(base.ROOT, base.ROOT / base.PLAN, args.plan_sha256)
        service_preflight(plan['service_plan_sha256'])
        validate_hashes(base.ROOT, {**proof['input_sha256'], **proof['source_sha256']})
        require(plan['input_sha256'][DOWNLOAD] == base.digest(base.ROOT / DOWNLOAD), 'model download differs')
        tokenizer_files = validate_tokenizer_files(args.tokenizer_path, base.read(base.ROOT / DOWNLOAD))
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, trust_remote_code=False)
        require(validate_token_corpus(base.ROOT) == target
            and corpus_hashes == {name: base.digest(target / name) for name in corpus_hashes},
            'corpus changed before token audit')
        rows = read_lines(target / 'sft.jsonl')
        result = audit_tokens(rows, tokenizer)
        require(validate_token_corpus(base.ROOT) == target
            and corpus_hashes == {name: base.digest(target / name) for name in corpus_hashes},
            'corpus changed during token audit')
        result.update(sft_sha256=proof['sft_sha256'], encoder_sha256=proof['encoder_sha256'],
            tokenizer_path=str(Path(args.tokenizer_path).resolve()), tokenizer_files_sha256=tokenizer_files)
        atomic_json(target / 'token-audit.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in {'samples', 'tokenizer_files_sha256'}}))
        return
    require(not target.exists(), 'corpus output exists; do not overwrite')
    proof, rows, episodes = audit_complete(base.ROOT, args.plan_sha256, args.auditor_sha256)
    target.mkdir(mode=0o700)
    for name, records in [('sft.jsonl', rows), ('episodes.jsonl', [e.to_dict() for e in episodes])]:
        with (target / name).open('x') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    proof.update(sft_sha256=base.digest(target / 'sft.jsonl'), episodes_sha256=base.digest(target / 'episodes.jsonl'))
    atomic_json(target / 'audit.json', proof)
    print(json.dumps({'phase': 'complete', **{key: proof['curation'][key] for key in
        ('attempts', 'successes', 'scenario_count', 'exported_targets', 'excluded_error_targets')}}))


if __name__ == '__main__':
    main()
