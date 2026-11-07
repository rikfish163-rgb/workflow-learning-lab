"""One exclusive four-batch parent; no restart or downstream launch.

Actor execution and resource cleanup belong to the registered coordinator. This
parent authenticates its closed evidence before applying the pure source gate.
"""
import argparse
import copy
import fcntl
import json
import math
import os
import re
import signal
import stat
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-codeact-within-source-001'
OUTPUT = ROOT / 'artifacts' / ID
PRIVATE = Path('/home/hetaisheng/.local/share/workflow-learning-lab/codeact-within-source-001')
ERROR = 'codeact_source_study_invalid'
_COUNTS = ('generation_attempts', 'generation_returns', 'prompt_tokens', 'completion_tokens',
           'actions', 'api_request_dispatches', 'budget_rejections')
_WRITER = {'messages_before', 'raw_response', 'action', 'observation', 'error_index', 'target_index'}


def _require(value):
    if not value:
        raise ValueError(ERROR)


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _identity(who):
    _require(type(who) is dict and set(who) == {'pid', 'start_ticks'}
             and all(type(v) is int and v > 0 for v in who.values()))
    return who


def _dependencies():
    from scripts import codeact_within_registration_001 as registration
    from scripts import run_codeact_within_source_001 as coordinator
    from scripts.codeact_within_sources_001 import qualify_sources
    return SimpleNamespace(registration=registration, coordinator=coordinator, qualify_sources=qualify_sources)


def _inspect_batch(deps, plan, sha, index, result, parent):
    """Inspect only closed, accepted batches; coordinator authenticates trace semantics."""
    c, core = deps.coordinator, deps.coordinator._lifecycle()
    stage, private, snapshots = OUTPUT / f'batch-{index}', PRIVATE / f'batch-{index}', []
    _require(c.verify_batch_report(stage, plan, index, sha) is True)

    def read(path, expected=None, public=False):
        raw, digest = c._read(path, expected=expected, public=public, maximum=128_000_000)
        snapshots.append((path, digest, public))
        return json.loads(raw, object_pairs_hook=c._pairs, parse_constant=c._reject_number), digest

    lifecycle, life_sha = read(stage / 'lifecycle.json')
    _require(_same(lifecycle, result) and lifecycle['accepted'] is True
             and lifecycle['error_type'] is None and lifecycle['automatic_retry'] is False
             and lifecycle['primary_failure_phase'] is lifecycle['primary_failure_errno'] is None
             and _same(lifecycle['journal_write_failure'], {'failed': False, 'errno': None})
             and _number(lifecycle['finished_epoch']))
    job, _ = read(stage / 'job.json')
    _require(_same(job, core.queue.make_job(index, job.get('owner'), sha)))
    health, _ = read(stage / 'health.json')
    activation, _ = read(stage / 'activation.json')
    final_health, _ = read(stage / 'final-health.json')
    core.validate_health(health, plan, 'B3')
    core.validate_health(final_health, plan, 'B3')
    _require(_same(final_health['backend'], health['backend']))
    request, request_sha = read(stage / 'request.json')
    _require(_same(request, {'plan_sha256': sha, 'batch_index': index, 'expected_backend': health['backend']}))
    evidence, _ = read(stage / 'remote-evidence.json')
    gpu = core.verify_gpu_evidence(evidence, job, activation, health, plan)
    _require(_same(gpu, lifecycle['gpu_costs'])
             and all(_number(gpu[k]) for k in ('gpu_job_wall_seconds', 'model_loading_seconds'))
             and gpu['model_loading_seconds'] <= gpu['gpu_job_wall_seconds'] <= 7200
             and type(gpu['gpu_index']) is int and gpu['gpu_index'] in (8, 9)
             and type(gpu['model_pid']) is int and gpu['model_pid'] > 0)
    released, _ = read(stage / 'GPU-release.json')
    _require(all(released.get(k) is True for k in ('owned', 'released', 'heartbeat_absent', 'monitor_live')))
    report, report_sha = read(stage / 'run/report.json')
    binding, _ = read(stage / 'client-report-binding.json')
    _require(_same(binding, {'sha256': report_sha, 'request_sha256': request_sha})
             and type(report['slots']) is list and len(report['slots']) == 3)
    event_raw, event_sha = c._read(stage / 'run/events.jsonl', expected=report['events_sha256'], public=True)
    snapshots.append((stage / 'run/events.jsonl', event_sha, True))
    _require(event_raw and event_raw.endswith(b'\n'))
    events = [json.loads(line, object_pairs_hook=c._pairs, parse_constant=c._reject_number)
              for line in event_raw.splitlines()]
    intents = [row for row in events if row.get('event') == 'container_intent']
    _require(len(intents) == 3)
    guarded, _ = read(stage / 'watchdog.json')
    guard_launch, _ = read(stage / 'guard-launch.json')
    recovered, _ = read(stage / 'CPU-recovery.json')
    client = _identity(guarded['client_identity'])
    _require(guarded.get('client_exit_code') == 0 and type(guarded.get('client_exit_code')) is int
             and guarded['phase'] == 'finished'
             and _same(guard_launch, {'identity': _identity(guarded['watchdog'])})
             and type(guarded['client_seconds']) is int and guarded['client_seconds'] == plan['client_seconds'] == 6000
             and _same(recovered['controller'], parent)
             and _same(guarded['parent'], parent)
             and _same(guarded['command'], core.watchdog.client_command(stage, sha)))
    for record in (guarded, recovered):
        _require(record['study_id'] == ID and record['plan_sha256'] == sha
                 and _same(record['client_release'], {**client, 'verified_exited': True}))
        release = record['container_release']
        _require(release['verified'] is True and release['errors'] == []
                 and type(release['containers']) is list and len(release['containers']) == 3)
        for offset, item in enumerate(release['containers']):
            intent = intents[offset]
            _require(item['task_id'] == intent['task_id'] == plan['batches'][index][offset]
                     and item['arm'] == 'first_attempt' and _same(item['intent'], intent['intent'])
                     and intent['intent']['owner_pid'] == client['pid']
                     and type(intent['intent']['owner_pid']) is int
                     and intent['intent']['owner_start_ticks'] == client['start_ticks']
                     and type(intent['intent']['owner_start_ticks']) is int
                     and intent['intent']['image'] == plan['image']
                     and item['creation_confirmed'] is item['exact_owned_name_absent'] is True
                     and (item['removed_container_id'] is None
                          or item['removed_container_id'] == report['slots'][offset]['owner']['container_id']))
    envelopes, public = [], []
    for offset, row in enumerate(report['slots']):
        source_index = index * 3 + offset
        _require(row['task_id'] == plan['task_ids'][source_index]
                 and row['arm'] == 'first_attempt' and row['phase'] == 'source')
        outcome = {key: row[key] for key in c._OUTCOME_FIELDS}
        _require(c._outcome_valid(outcome) and c._owner_verified(row['owner'], outcome, row['observed_model'])
                 and outcome['costs_known'] is True and all(type(v) is int and v >= 0 for v in outcome['counts'].values()))
        owner = row['owner']
        _require(type(owner['phase_seconds']) is dict and set(owner['phase_seconds']) == {'startup', 'operation', 'cleanup'}
                 and all(_number(v) for v in owner['phase_seconds'].values()) and _number(owner['wall_seconds']))
        bound = row['private_bindings']['episode_result']
        _require(bound['file'] == f'episode-{offset:02}.json')
        episode, _ = read(private / bound['file'], expected=bound['sha256'])
        _require(type(episode) is dict and set(episode) == {'private', 'outcome'} and _same(episode['outcome'], outcome))
        for key, filename in (('actor_trace', f'actor-{offset:02}.jsonl'), ('wire_trace', f'wire-{offset:02}.jsonl')):
            binding = row['private_bindings'][key]
            _require(binding['file'] == filename)
            _, digest = c._read(private / filename, expected=binding['sha256'], maximum=128_000_000)
            snapshots.append((private / filename, digest, False))
        envelopes.append(dict(source_index=source_index, task_id=row['task_id'], **episode,
                              verified=dict(owner_lifecycle=True, container_release=True,
                                            model_lease_release=True, all_costs=True)))
        public.append(dict(source_index=source_index, task_id=row['task_id'], batch_index=index,
                           status=row['status'], attempted=True, auth_complete=True,
                           model_reported_complete=outcome['model_reported_complete'],
                           official_success=outcome['official_success'], costs_known=True, counts=copy.deepcopy(outcome['counts']),
                           actor_wall_seconds=outcome['actor_wall_seconds'], world_wall_seconds=owner['wall_seconds'],
                           phase_seconds=copy.deepcopy(owner['phase_seconds'])))
    for path, digest, is_public in snapshots:
        c._read(path, expected=digest, public=is_public, maximum=128_000_000)
    summary = dict(batch_index=index, status='complete', accepted=True, auth_complete=True, lifecycle_sha256=life_sha,
                   report_sha256=report_sha, gpu_costs=gpu,
                   release=dict(cpu_client=True, containers=True, model_lease=True))
    return envelopes, public, summary


def _preserve_partial(deps, plan, sha, index, report):
    """Keep observed costs separate from authenticated closed-episode totals."""
    c, stage = deps.coordinator, OUTPUT / f'batch-{index}'
    report['batches'][index]['partial_metadata_status'] = 'unavailable_or_invalid'
    try:
        raw, report_sha = c._read(stage / 'run/report.json')
        batch = json.loads(raw, object_pairs_hook=c._pairs, parse_constant=c._reject_number)
        request, request_sha = c._read_json(stage / 'request.json')
        planned = [dict(task_id=t, arm='first_attempt', phase='source') for t in plan['batches'][index]]
        _require(batch['id'] == ID and batch['protocol'] == c.BATCH_PROTOCOL and batch['plan_sha256'] == sha
                 and type(batch['batch_index']) is int and batch['batch_index'] == index
                 and batch['request_sha256'] == request_sha and request['plan_sha256'] == sha
                 and type(request['batch_index']) is int and request['batch_index'] == index
                 and _same(batch['planned_slots'], planned) and type(batch['slots']) is list
                 and len(batch['slots']) <= 3 and _same(batch['unattempted_slots'], planned[len(batch['slots']):]))
        events, event_sha = c._read(stage / 'run/events.jsonl', expected=batch['events_sha256'], public=True)
        _require(events and events.endswith(b'\n'))
        event_rows = [json.loads(line, object_pairs_hook=c._pairs, parse_constant=c._reject_number)
                      for line in events.splitlines()]
        _require(all(type(row.get('event_index')) is int and row['event_index'] == n
                     and row.get('task_id') in plan['batches'][index] for n, row in enumerate(event_rows)))
        rows = copy.deepcopy(report['sources'][index * 3:index * 3 + 3])
        for offset, row in enumerate(batch['slots']):
            _require(all(row[k] == planned[offset][k] for k in ('task_id', 'arm', 'phase'))
                     and row['status'] in {'scored', 'known_unscored', 'unknown'})
            progress = row['observed_model']
            _require(type(progress) is dict and set(progress) == {
                'generation_attempts', 'generation_returns', 'prompt_tokens', 'completion_tokens'}
                     and all(type(v) is int and v >= 0 for v in progress.values()))
            outcome = {key: row[key] for key in c._OUTCOME_FIELDS}
            valid = c._outcome_valid(outcome)
            observed = {key: (outcome['counts'].get(key) if valid else None) for key in _COUNTS}
            for key, value in progress.items():
                observed[key] = value if observed[key] is None else max(value, observed[key])
            if valid:
                for key, value in outcome['known_costs'].items():
                    observed[key] = value if observed[key] is None else max(value, observed[key])
            owner_api = row['owner'].get('api_attempts_observed') if type(row['owner']) is dict else None
            if type(owner_api) is int and owner_api >= 0:
                old_api = observed['api_request_dispatches']
                observed['api_request_dispatches'] = owner_api if old_api is None else max(old_api, owner_api)
            bound = False
            if valid and c._owner_verified(row['owner'], outcome, progress):
                try:
                    values = {}
                    for key, filename in (('actor_trace', f'actor-{offset:02}.jsonl'),
                                          ('wire_trace', f'wire-{offset:02}.jsonl'),
                                          ('episode_result', f'episode-{offset:02}.json')):
                        binding = row['private_bindings'][key]
                        _require(binding['file'] == filename)
                        values[key], _ = c._read(PRIVATE / f'batch-{index}' / filename,
                                                expected=binding['sha256'], maximum=128_000_000)
                    episode = json.loads(values['episode_result'], object_pairs_hook=c._pairs, parse_constant=c._reject_number)
                    _require(set(episode) == {'private', 'outcome'} and _same(episode['outcome'], outcome))
                    c._verify_private(values['actor_trace'], values['wire_trace'], episode, request['expected_backend'])
                    bound = True
                except Exception:
                    pass
            rows[offset].update(status='observed_' + row['status'], attempted=True, auth_complete=False,
                                model_reported_complete=outcome['model_reported_complete'] if bound else None,
                                official_success=outcome['official_success'] if bound else None,
                                costs_known=None, reported_costs_known=outcome['costs_known'] if valid else None,
                                counts=copy.deepcopy(outcome['counts']) if valid else None,
                                observed_unverified_costs=observed)
        for row in rows[len(batch['slots']):]:
            row.update(status='unattempted', attempted=False, auth_complete=False, costs_known=None,
                       official_success=None, model_reported_complete=None, counts=None)
        c._read(stage / 'run/report.json', expected=report_sha)
        c._read(stage / 'run/events.jsonl', expected=event_sha, public=True)
        report['sources'][index * 3:index * 3 + 3] = rows
        totals = {}
        for key in _COUNTS:
            values = [row['observed_unverified_costs'][key] for row in rows
                      if row.get('observed_unverified_costs', {}).get(key) is not None]
            totals[key] = sum(values) if values else None
        report['batches'][index].update(auth_complete=False, report_sha256=report_sha,
                                       partial_metadata_status='observed_unverified',
                                       observed_unverified_costs=totals)
    except Exception:
        pass  # Missing or invalid metadata remains explicitly unknown, never zero.


def _costs(report):
    rows = [row for row in report['sources'] if row['costs_known'] is True and row['auth_complete'] is True]
    batches = [row for row in report['batches'] if row['status'] == 'complete' and row['auth_complete'] is True]
    full = len(rows) == 12 and len(batches) == 4
    subtotal = {key: sum(row['counts'][key] for row in rows) for key in _COUNTS}
    for key in ('actor_wall_seconds', 'world_wall_seconds'):
        subtotal[key] = sum(row[key] for row in rows)
    for phase in ('startup', 'operation', 'cleanup'):
        subtotal[phase + '_seconds'] = sum(row['phase_seconds'][phase] for row in rows)
    gpu = sum(row['gpu_costs']['gpu_job_wall_seconds'] for row in batches)
    loading = sum(row['gpu_costs']['model_loading_seconds'] for row in batches)
    return dict(all_costs_known=full, closed_known_episode_count=len(rows), closed_known_episode_subtotal=subtotal,
                closed_known_gpu_subtotal=gpu, gpu_job_wall_seconds=gpu if full else None,
                model_loading_seconds=loading if full else None, CUDA_seconds=None,
                scope='GPU wall includes model loading; actor/world and GPU timings overlap.')


def run_study(sha, *, _deps=None):
    _require(type(sha) is str and re.fullmatch('[a-f0-9]{64}', sha) is not None)
    deps = _deps or _dependencies()
    reg, c = deps.registration, deps.coordinator
    plan = reg.load_registration(ROOT, sha, private=True)
    reg.check_predecessor(ROOT, plan)
    _require(plan['id'] == ID and plan['task_ids'] == list(reg.TASK_IDS) and len(reg.TASK_IDS) == 12
             and plan['batches'] == reg.BATCHES == [plan['task_ids'][i:i + 3] for i in range(0, 12, 3)]
             and plan['private_directory'] == str(PRIVATE))
    core = c._lifecycle()
    preflight = core.remote({'operation': 'preflight', 'sha': sha})
    _require(type(preflight) is dict and preflight.get('registration') == 'passed' and preflight.get('monitor_live') is True)
    lock_fd = os.open(ROOT / 'artifacts/foundation-pilot-001.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(lock_fd, 'a+') as lock:
        info = os.fstat(lock.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
                 and stat.S_IMODE(info.st_mode) & 0o022 == 0)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        core.validate_directory(PRIVATE)
        c.io.reserve(OUTPUT)
        parent = _identity(core.identity(os.getpid()))
        intent = dict(id=ID, plan_sha256=sha, parent=parent, started_epoch=time.time(), automatic_retry=False)
        intent_sha = c._write_private(PRIVATE / 'study-intent.json', intent)
        report = dict(id=ID, plan_sha256=sha, status='running', parent=parent,
                      study_intent_sha256=intent_sha, started_epoch=intent['started_epoch'], fixed_denominator=12,
                      batches=[dict(batch_index=i, status='unattempted', accepted=None, auth_complete=False,
                                    gpu_costs=None, release=None) for i in range(4)],
                      sources=[dict(source_index=i, task_id=task_id, batch_index=i // 3, status='unattempted', attempted=False,
                                    auth_complete=False, model_reported_complete=None, official_success=None, costs_known=None, counts=None,
                                    actor_wall_seconds=None, world_wall_seconds=None, phase_seconds=None)
                               for i, task_id in enumerate(plan['task_ids'])],
                      unattempted_batches=list(range(4)), qualification=None, writer_inputs=None,
                      automatic_retry=False, utility_evaluated=False)
        journal_fd = os.open(OUTPUT / 'events.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(journal_fd, 'w', encoding='ascii') as journal:
            sequence = 0
            def save(event, **fields):
                nonlocal sequence
                value = dict(event=event, event_index=sequence, epoch=time.time(), plan_sha256=sha, **fields)
                journal.write(json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                sequence += 1
                report['costs'] = _costs(report)
                c.io.checkpoint(OUTPUT / 'report.json', report)
            results, envelopes = [], []
            save('study_start', parent=parent)
            for index in range(4):
                report['unattempted_batches'].remove(index)
                report['batches'][index]['status'] = 'running'
                for row in report['sources'][index * 3:index * 3 + 3]:
                    row.update(status='unknown', attempted=None)
                save('batch_start', batch_index=index)
                try:
                    result = c.run_stage(plan, sha, index)
                    _require(type(result) is dict and type(result.get('accepted')) is bool)
                    results.append(result)
                    if result['accepted'] is not True:
                        raise ValueError(ERROR)
                    group, public, batch = _inspect_batch(deps, plan, sha, index, result, parent)
                    envelopes.extend(group)
                    report['sources'][index * 3:index * 3 + 3] = public
                    report['batches'][index] = batch
                except (Exception, KeyboardInterrupt, SystemExit) as exc:
                    report['status'] = 'interrupted_no_retry' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'incomplete_no_retry'
                    report['batches'][index].update(status='unknown', accepted=False, auth_complete=False)
                    _preserve_partial(deps, plan, sha, index, report)
                    save('batch_unknown', batch_index=index)
                    break
                save('batch_complete', batch_index=index)
            if all(batch['status'] == 'complete' for batch in report['batches']):
                try:
                    envelopes = []
                    for index, result in enumerate(results):
                        # Invalidate first: a failed final reread cannot retain prior authentication.
                        report['batches'][index].update(status='unknown', accepted=False, auth_complete=False)
                        for row in report['sources'][index * 3:index * 3 + 3]:
                            row.update(status='unknown', auth_complete=False, costs_known=None,
                                       official_success=None, model_reported_complete=None)
                        group, public, batch = _inspect_batch(deps, plan, sha, index, result, parent)
                        envelopes.extend(group)
                        report['sources'][index * 3:index * 3 + 3], report['batches'][index] = public, batch
                    qualified = deps.qualify_sources(envelopes)
                    _require(type(qualified) is dict and set(qualified) == {'public', 'writer_inputs'}
                             and type(qualified['public']['qualified']) is bool and qualified['public']['episode_count'] == 12)
                    report['qualification'] = qualified['public']
                    if qualified['public']['qualified']:
                        rows = qualified['writer_inputs']
                        _require(type(rows) is list and len(rows) >= 20 and all(type(row) is dict and set(row) == _WRITER for row in rows))
                        digest = c._write_private(PRIVATE / 'writer-inputs.json', rows)
                        report['writer_inputs'] = dict(file='writer-inputs.json', sha256=digest, count=len(rows))
                    else:
                        _require(qualified['writer_inputs'] == [])
                    report['status'] = 'complete'
                except (Exception, KeyboardInterrupt, SystemExit) as exc:
                    report['status'] = 'interrupted_no_retry' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'incomplete_no_retry'
                    report['qualification'] = report['writer_inputs'] = None
            report['finished_epoch'] = time.time()
            save('study_stop', status=report['status'])
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(argv)
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        try:
            report = run_study(args.plan_sha256)
            result = dict(id=ID, status=report['status'], fixed_denominator=12)
            code = 0 if report['status'] == 'complete' else 1
        except Exception:
            result, code = dict(id=ID, status='failed', error_code=ERROR), 1
        print(json.dumps(result, sort_keys=True))
        return code
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
