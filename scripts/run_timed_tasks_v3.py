"""One fixed six-task dev batch with durable task/request/container intent.

The parent owns registration, preparation, watchdogs and independently verified
process/container release. No task discovery, GPU acquisition, retry or resume
occurs here. Train and the full-dev V2 entrypoint remain unchanged.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from scripts import run_timed_tasks_v2 as v2
from scripts.documentation_feedback import DocumentationFeedbackWorld
from scripts.run_timed_tasks import (
    BackendMismatch,
    CheckedModel,
    CleanupTrackedWorld,
    RolloutDeadline,
    RolloutTimedOut,
    wait_for_idle,
)
from wll.model import LocalModel
from wll.protocol import Task, fingerprint
from wll.runner import run_episode
from wll.storage import atomic_json, require_space

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE, BATCH_COUNT = 6, 10
SEED_PROTOCOL = 'wll-appworld-seed-v1'
FEEDBACK_SOURCES = {
    'scripts/documentation_feedback.py': 'e7eb24a8c87a5304815514203c483cfd4c2fe5bd32b0f82e5cd07b9de032b5b2',
    'scripts/probe_observation_visibility.py': 'd9bd7d258eabdc006a9aa470eff2cc0d06bf14e8437c4b1c74d3e6287129e301',
}
UNKNOWN_SEED = {'requested': 100, 'effective': None, 'verified': False}


class EnvironmentMismatch(ValueError):
    pass


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feedback_recipe():
    for name, sha in FEEDBACK_SOURCES.items():
        v2._bound_bytes(ROOT / name, sha, 'observation feedback source')
    return {'kind': 'silent_documentation_print_hint_v1', 'source_sha256': dict(FEEDBACK_SOURCES)}


def environment_identity(image):
    if type(image) is not str or re.fullmatch('sha256:[0-9a-f]{64}', image) is None:
        raise ValueError('immutable image SHA256 required')
    return fingerprint({'image': image, 'observation_feedback': feedback_recipe(),
                        'environment_seed': 100, 'seed_protocol': SEED_PROTOCOL})


def client_identity():
    return {'pid': os.getpid(), 'start_ticks': int(
        Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19])}


def select_batch(tasks, batch_index, data_dir):
    if type(batch_index) is not int or not 0 <= batch_index < BATCH_COUNT:
        raise ValueError('batch index must be an integer from0 through9')
    tasks = list(tasks)
    v2.validate_dev_ids([getattr(task, 'task_id', None) for task in tasks], data_dir)
    if any(not isinstance(task, Task) or task.split != 'dev'
           or task.scenario_id != task.task_id.rsplit('_', 1)[0]
           or type(task.instruction) is not str or not task.instruction
           or type(task.supervisor) is not str for task in tasks):
        raise ValueError('complete dev task metadata/scenario/split required')
    return tasks[batch_index*BATCH_SIZE:min((batch_index+1)*BATCH_SIZE, 57)]


def load_prepared_tasks(path, *, expected_sha256, data_dir, deps_dir, image):
    """Read private, parent-prepared metadata; perform no Docker discovery."""
    raw = v2._bound_bytes(path, expected_sha256, 'prepared task metadata')
    value = json.loads(raw)
    expected_id = environment_identity(image)
    if (value.get('split') != 'dev' or value.get('image') != image
            or value.get('data_dir') != str(Path(data_dir).resolve())
            or value.get('deps_dir') != str(Path(deps_dir).resolve())
            or value.get('load_ground_truth') is not False
            or value.get('official_dev_sha256') != v2.OFFICIAL_DEV_SHA256
            or value.get('preparation_sha256') != v2.PREPARATION_SHA256
            or value.get('seed_protocol') != SEED_PROTOCOL
            or type(value.get('environment_seed')) is not int or value['environment_seed'] != 100
            or value.get('observation_feedback') != feedback_recipe()
            or value.get('environment_id') != expected_id):
        raise ValueError('prepared metadata environment/provenance differs')
    try:
        tasks = [Task(**row) for row in value['tasks']]
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid prepared task records') from exc
    select_batch(tasks, 0, data_dir)  # Validate all57 before selecting any batch.
    return tasks, value


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, value):
    atomic_json(path, value)
    _sync_directory(Path(path).parent)


def _append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False)+'\n')
    stream.flush()
    os.fsync(stream.fileno())


def _intent(value, identity, image):
    if (type(value) is not dict or set(value) != {'container_name', 'image', 'owner_pid',
            'owner_start_ticks', 'ownership_token'}
            or type(value['ownership_token']) is not str
            or re.fullmatch('[0-9a-f]{32}', value['ownership_token']) is None
            or value['container_name'] != 'wll-'+value['ownership_token'][:16]
            or value['image'] != image
            or type(value['owner_pid']) is not int or value['owner_pid'] != identity['pid']
            or type(value['owner_start_ticks']) is not int
            or value['owner_start_ticks'] != identity['start_ticks']):
        raise EnvironmentMismatch('container ownership intent differs')
    return dict(value)


class _IntentModel:
    def __init__(self, model, emit, task_id, call_counter):
        self.model, self.emit, self.task_id, self.call_counter = model, emit, task_id, call_counter

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, messages, seed):
        self.call_counter[0] += 1
        call = {'task_id': self.task_id, 'call_id': self.call_counter[0], 'seed': seed}
        self.emit('generation_attempt', **call)  # fsync before LocalModel's HTTP call.
        try:
            result = self.model.generate(messages, seed)
        except BaseException as exc:
            self.emit('generation_failed', **call, error_type=type(exc).__name__)
            raise
        valid = isinstance(result, dict) and all(type(result.get(k)) is int and result[k] >= 0
                                                 for k in ('prompt_tokens', 'completion_tokens'))
        self.emit('generation_returned', **call, usage_valid=valid,
            prompt_tokens=result['prompt_tokens'] if valid else None,
            completion_tokens=result['completion_tokens'] if valid else None)
        return result


def _summary(manifest, records, counters, status, wall_seconds):
    count = len(manifest['batch_task_ids'])
    complete = sum(r['status'] == 'complete' for r in records)
    successes = sum(r['status'] == 'complete' and r['success'] is True for r in records)
    unknown = count-complete
    started = {r['task_id'] for r in records}
    return {'status': status, 'arm': manifest['arm'], 'batch_index': manifest['batch_index'],
        'scheduled': count, 'attempted': len(records), 'completed': complete, 'successes': successes,
        'unknown_outcomes': unknown, 'unattempted': count-len(records),
        'unattempted_task_ids': [name for name in manifest['batch_task_ids'] if name not in started],
        'valid_for_comparison': unknown == 0, 'full_dev_evaluation_complete': False,
        'comparison_scope': 'This fixed batch only; the full-dev paired estimate requires all57 known pairs.',
        'full_schedule': {'task_ids': [t['task_id'] for t in manifest['tasks']], 'task_count': 57,
            'batch_size': BATCH_SIZE, 'batch_count': BATCH_COUNT,
            'slice': manifest['slice'], 'batch_task_ids': manifest['batch_task_ids']},
        'success_rate_completed_only': successes/complete if complete else None,
        'success_rate_all_scheduled': successes/count if unknown == 0 else None,
        'success_bounds_all_scheduled': [successes/count, (successes+unknown)/count],
        'unknown_additional_generation_cost': bool(counters['generation_calls_without_response']
            or counters['generation_calls_with_unusable_usage']),
        **counters, 'wall_seconds': wall_seconds,
        'cost_note': 'Returned-response and observed-step costs overlap; do not add them. '
            'Intent without usable returned usage has unknown possible additional generation cost.'}


def run_batch(tasks, world_factory, model, *, batch_index, arm, data_dir, output, expected_backend,
              image, environment_id, prepared_manifest_sha256=None, before_task=None,
              max_steps=40, rollout_seconds=600, seed=17, environment_seed=100,
              prompt_version=v2.DEV_PROMPT):
    """world_factory(task, on_container_intent) must return the frozen feedback wrapper."""
    tasks = list(tasks)
    selected = select_batch(tasks, batch_index, data_dir)
    v2._validate_dev_recipe(environment_seed=environment_seed, max_steps=max_steps,
        rollout_seconds=rollout_seconds, seed=seed, model=model, prompt_version=prompt_version)
    if (arm not in ('B0', 'B3') or not isinstance(expected_backend, dict)
            or expected_backend.get('arm') != arm or environment_id != environment_identity(image)
            or (prepared_manifest_sha256 is not None and re.fullmatch('[0-9a-f]{64}',
                prepared_manifest_sha256) is None)):
        raise ValueError('registered arm/backend/environment/metadata binding required')
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    _sync_directory(output.parent)
    identity = client_identity()
    manifest = {'status': 'running', 'arm': arm, 'batch_index': batch_index,
        'slice': [batch_index*6, min(batch_index*6+6, 57)],
        'tasks': [asdict(task) for task in tasks], 'batch_task_ids': [t.task_id for t in selected],
        'client_identity': identity, 'started_epoch': time.time(), 'model_id': model.model_id,
        'expected_backend': expected_backend, 'image': image, 'environment_id': environment_id,
        'data_dir': str(Path(data_dir).resolve()),
        'observation_feedback': feedback_recipe(), 'seed_protocol': SEED_PROTOCOL,
        'environment_seed': dict(UNKNOWN_SEED), 'prepared_manifest_sha256': prepared_manifest_sha256,
        'max_steps': max_steps, 'rollout_seconds': rollout_seconds, 'seed': seed,
        'max_tokens': model.max_tokens, 'temperature': model.temperature, 'prompt_version': prompt_version,
        'official_dev_sha256': v2.OFFICIAL_DEV_SHA256, 'preparation_sha256': v2.PREPARATION_SHA256,
        'script_sha256': file_sha256(__file__),
        'deadline_scope': 'rollout only; parent owns hard watchdog and independent release verification'}
    _atomic(output / 'manifest.json', manifest)
    counters = {'generation_calls': 0, 'generation_calls_without_response': 0,
        'generation_calls_with_unusable_usage': 0,
        'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0},
        'observed_step_cost': {'steps': 0, 'prompt_tokens': 0, 'completion_tokens': 0}}
    records, used_names, used_tokens, call_counter = [], set(), set(), [0]
    started, terminal = time.monotonic(), 'interrupted'
    try:
        with ExitStack() as stack:
            streams = {name: stack.enter_context((output / (name+'.jsonl')).open('x'))
                for name in ('events', 'attempts', 'episodes', 'steps', 'responses', 'feedback-events')}
            _sync_directory(output)
            event_index = 0

            def emit(event, **details):
                nonlocal event_index
                _append(streams['events'], {'event_index': event_index, 'event': event,
                                            'epoch': time.time(), **details})
                event_index += 1

            emit('batch_started', client_identity=identity, arm=arm, batch_index=batch_index,
                 batch_task_ids=manifest['batch_task_ids'])
            for task in selected:
                emit('task_started', task_id=task.task_id)  # Before health/world/model side effects.
                record = {'task_id': task.task_id, 'status': 'in_progress', 'success': None,
                    'started_epoch': time.time(), 'cleanup_verified': None,
                    'environment_seed': dict(UNKNOWN_SEED), 'feedback_events': 0}
                records.append(record)
                _atomic(output / 'summary.json', _summary(manifest, records, counters, 'running',
                                                          time.monotonic()-started))
                deadline, world, abort, seen_feedback = RolloutDeadline(rollout_seconds), None, False, 0

                def container_intent(value, record=record, task_id=task.task_id):
                    value = _intent(value, identity, image)
                    if ('container_intent' in record or value['container_name'] in used_names
                            or value['ownership_token'] in used_tokens):
                        raise EnvironmentMismatch('duplicate container intent')
                    emit('container_intent', task_id=task_id, container_intent=value)
                    record['container_intent'] = value
                    used_names.add(value['container_name'])
                    used_tokens.add(value['ownership_token'])

                def on_response(value, task_id=task.task_id):
                    _append(streams['responses'], {'task_id': task_id, **value})

                checked = CheckedModel(_IntentModel(model, emit, task.task_id, call_counter),
                                       expected_backend, counters, on_response)
                try:
                    waiting = time.monotonic()
                    try:
                        if before_task is not None:
                            before_task(task)
                    finally:
                        record['backend_wait_seconds'] = time.monotonic()-waiting
                    startup = time.monotonic()
                    try:
                        world = CleanupTrackedWorld(world_factory(task, container_intent), deadline)
                        if ('container_intent' not in record
                                or not isinstance(world.world, DocumentationFeedbackWorld)):
                            raise EnvironmentMismatch('owned seeded feedback world required')
                        record['environment_seed'] = v2._validate_seed(world.environment_seed_provenance)
                        emit('world_ready', task_id=task.task_id, environment_seed=record['environment_seed'])
                    finally:
                        record['startup_seconds'] = time.monotonic()-startup

                    def observe(index, step, replayed, world=world, task_id=task.task_id):
                        nonlocal seen_feedback
                        assert not replayed
                        current = world.feedback_events
                        if current != seen_feedback:
                            _append(streams['feedback-events'], {'task_id': task_id, 'index': index,
                                'feedback_events': current, 'action_sha256': fingerprint(step.action),
                                'augmented_observation_sha256': fingerprint(step.observation)})
                            seen_feedback = current
                        _append(streams['steps'], {'task_id': task_id, 'index': index,
                                                  'replayed': False, 'step': asdict(step)})
                        counters['observed_step_cost']['steps'] += 1
                        for key in ('prompt_tokens', 'completion_tokens'):
                            counters['observed_step_cost'][key] += getattr(step, key)

                    with deadline:
                        episode = run_episode(task, lambda world=world: world, checked, max_steps=max_steps,
                                              seed=seed, on_step=observe, prompt_version=prompt_version)
                    episode.metadata.update(environment_id=environment_id,
                        environment_seed=dict(record['environment_seed']),
                        observation_feedback=manifest['observation_feedback'], feedback_events=world.feedback_events,
                        rollout_deadline_seconds=rollout_seconds, arm=arm, batch_index=batch_index)
                    if type(episode.success) is not bool:
                        raise TypeError('official outcome must be boolean')
                    _append(streams['episodes'], episode.to_dict())
                    record.update(status='complete', success=episode.success,
                                  episode_hash=fingerprint(episode.to_dict()))
                except RolloutTimedOut as exc:
                    record.update(status='timed_out', error_type=type(exc).__name__)
                except Exception as exc:
                    record.update(status='infrastructure_error', error_type=type(exc).__name__)
                    abort = world is None or isinstance(exc, (BackendMismatch, v2.SeedMismatch, EnvironmentMismatch))
                except BaseException as exc:
                    record.update(status='interrupted', error_type=type(exc).__name__)
                    raise
                finally:
                    if world is not None:
                        try:
                            world.close()
                        except Exception as exc:
                            record.update(status='infrastructure_error', success=None,
                                          cleanup_error_type=type(exc).__name__)
                            abort = True
                        record['cleanup_verified'] = world.closed
                        record['feedback_events'] = getattr(world, 'feedback_events', 0)
                    record['finished_epoch'] = time.time()
                    _append(streams['attempts'], record)
                    _atomic(output / 'summary.json', _summary(manifest, records, counters, 'running',
                                                              time.monotonic()-started))
                if abort:
                    manifest['stop_reason'] = 'startup_backend_environment_or_cleanup_failure'
                    break
        terminal = 'complete' if len(records) == len(selected) and all(
            r['status'] == 'complete' for r in records) else 'incomplete'
    finally:
        manifest['status'] = terminal
        if len(records) == len(selected) and all(r['environment_seed']['verified'] is True for r in records):
            manifest['environment_seed'] = {'requested': 100, 'effective': 100, 'verified': True}
        _atomic(output / 'manifest.json', manifest)
        _atomic(output / 'summary.json', _summary(manifest, records, counters, terminal,
                                                  time.monotonic()-started))
    return _summary(manifest, records, counters, terminal, time.monotonic()-started)


def _read_journal(path):
    rows, errors = [], []
    if not path.is_file() or path.is_symlink():
        return rows, [path.name+':missing_or_symlink']
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):
            errors.append(path.name+':truncated_tail')
            break
        try:
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError()
            rows.append(value)
        except (ValueError, UnicodeError):
            errors.append(path.name+':malformed_row')
            break
    return rows, errors


def reconcile_batch(output, *, terminated_identity):
    """Read durable evidence only after the parent verifies this client exited.

    This does not inspect processes, Docker labels/IDs or GPU state and never
    certifies release. Damaged journals make later start status uncertain.
    """
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    canonical = select_batch([Task(**row) for row in manifest['tasks']],
                             manifest['batch_index'], manifest['data_dir'])
    if ([task.task_id for task in canonical] != manifest['batch_task_ids']
            or manifest['environment_id'] != environment_identity(manifest['image'])):
        raise ValueError('durable batch manifest selection/environment differs')
    events, errors = _read_journal(output / 'events.jsonl')
    identity = manifest['client_identity']
    if (type(identity) is not dict or set(identity) != {'pid', 'start_ticks'}
            or any(type(identity[k]) is not int or identity[k] <= 0 for k in identity)
            or type(terminated_identity) is not dict or terminated_identity.get('verified_exited') is not True
            or any(type(terminated_identity.get(k)) is not int for k in identity)
            or {k: terminated_identity.get(k) for k in identity} != identity
            or not events or events[0].get('event') != 'batch_started'
            or events[0].get('client_identity') != identity
            or events[0].get('batch_task_ids') != manifest['batch_task_ids']
            or events[0].get('arm') != manifest['arm']
            or events[0].get('batch_index') != manifest['batch_index']):
        raise ValueError('verified terminated client identity differs from durable batch start')
    selected = manifest['batch_task_ids']
    started, ready, intents, calls, returned = [], set(), [], {}, {}
    for index, event in enumerate(events):
        try:
            if type(event.get('event_index')) is not int or event['event_index'] != index:
                raise ValueError()
            kind, task_id = event['event'], event.get('task_id')
            if kind == 'batch_started':
                if index != 0:
                    raise ValueError()
            elif kind == 'task_started':
                if len(started) >= len(selected) or task_id != selected[len(started)]:
                    raise ValueError()
                started.append(task_id)
            elif not started or task_id != started[-1]:
                raise ValueError()
            elif kind == 'container_intent':
                value = _intent(event['container_intent'], identity, manifest['image'])
                if any(value['container_name'] == old['container_name'] for old in intents):
                    raise ValueError()
                intents.append(value)
            elif kind == 'world_ready':
                v2._validate_seed(event['environment_seed'])
                ready.add(task_id)
            elif kind == 'generation_attempt':
                number = event['call_id']
                if (task_id not in ready or type(number) is not int or number != len(calls)+1
                        or event['seed'] != 17+sum(c['task_id'] == task_id for c in calls.values())):
                    raise ValueError()
                calls[number] = event
            elif kind in ('generation_returned', 'generation_failed'):
                number = event['call_id']
                if number not in calls or calls[number]['task_id'] != task_id:
                    raise ValueError()
                if kind == 'generation_returned':
                    if number in returned or type(event['usage_valid']) is not bool or (
                        event['usage_valid'] and any(type(event[k]) is not int or event[k] < 0
                            for k in ('prompt_tokens', 'completion_tokens'))):
                        raise ValueError()
                    returned[number] = event
            else:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            errors.append('events.jsonl:invalid_sequence_or_record')
            break
    attempts, issues = _read_journal(output / 'attempts.jsonl')
    errors.extend(issues)
    records = {name: {'task_id': name, 'status': 'lost_final_record', 'success': None,
                     'cleanup_verified': None} for name in started}
    for index, record in enumerate(attempts):
        valid = (index < len(started) and record.get('task_id') == started[index]
                 and record.get('status') in {'complete', 'timed_out', 'infrastructure_error', 'interrupted'})
        if record.get('status') == 'complete':
            try:
                v2._validate_seed(record.get('environment_seed'))
                valid = (valid and type(record.get('success')) is bool
                    and record.get('cleanup_verified') is True
                    and type(record.get('episode_hash')) is str
                    and re.fullmatch('[0-9a-f]{64}', record['episode_hash']) is not None)
            except v2.SeedMismatch:
                valid = False
        else:
            valid = valid and record.get('success') is None
        if not valid:
            errors.append('attempts.jsonl:invalid_final_record')
            break
        records[record['task_id']] = record
    episodes, issues = _read_journal(output / 'episodes.jsonl')
    errors.extend(issues)
    by_task = {}
    for episode in episodes:
        name = episode.get('task', {}).get('task_id')
        if name not in started or name in by_task:
            errors.append('episodes.jsonl:invalid_or_duplicate_task')
            continue
        by_task[name] = episode
    for name, record in records.items():
        episode = by_task.get(name)
        if record['status'] == 'complete' and (episode is None
                or fingerprint(episode) != record['episode_hash']
                or episode.get('success') is not record['success']):
            errors.append('attempts.jsonl:episode_commit_mismatch')
            records[name] = {'task_id': name, 'status': 'invalid_final_evidence',
                             'success': None, 'cleanup_verified': None}
    counters = {'generation_calls': len(calls),
        'generation_calls_without_response': len(calls.keys()-returned.keys()),
        'generation_calls_with_unusable_usage': sum(not r['usage_valid'] for r in returned.values()),
        'returned_response_cost': {key: sum(r[key] for r in returned.values() if r['usage_valid'])
                                  for key in ('prompt_tokens', 'completion_tokens')},
        'observed_step_cost': {'steps': 0, 'prompt_tokens': 0, 'completion_tokens': 0}}
    steps, issues = _read_journal(output / 'steps.jsonl')
    errors.extend(issues)
    observed_indices = {}
    for row in steps:
        step = row.get('step', {})
        name, index = row.get('task_id'), row.get('index')
        if (name not in started or type(index) is not int or not 0 <= index < 40
                or index != observed_indices.get(name, 0)
                or any(type(step.get(k)) is not int or step[k] < 0
                       for k in ('prompt_tokens', 'completion_tokens'))):
            errors.append('steps.jsonl:invalid_cost')
            break
        observed_indices[name] = index+1
        counters['observed_step_cost']['steps'] += 1
        for key in ('prompt_tokens', 'completion_tokens'):
            counters['observed_step_cost'][key] += step[key]
    # Even currently unused journal data must not hide corruption.
    for name in ('responses', 'feedback-events'):
        _, issues = _read_journal(output / (name+'.jsonl'))
        errors.extend(issues)
    result = _summary(manifest, list(records.values()), counters, 'reconciled', None)
    result.update(journal_integrity_verified=not errors, reconciliation_errors=errors,
        container_intents=intents, cleanup_release_independently_verified=False,
        client_identity=identity, start_status_unknown_task_ids=[],
        attempted_is_lower_bound=bool(errors), event_count_lower_bound=len(events))
    if errors:
        result.update(valid_for_comparison=False, unattempted=None, unattempted_task_ids=[],
            start_status_unknown_task_ids=[name for name in selected if name not in started],
            unknown_additional_generation_cost=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['B0', 'B3'], required=True)
    parser.add_argument('--batch-index', type=int, required=True)
    for name in ('task-manifest', 'task-manifest-sha256', 'data-dir', 'deps-dir', 'image'):
        parser.add_argument('--'+name, required=True)
    for name in ('backend-file', 'endpoint', 'model', 'output'):
        parser.add_argument('--'+name)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU client requires CUDA_VISIBLE_DEVICES empty')
    tasks, metadata = load_prepared_tasks(args.task_manifest,
        expected_sha256=args.task_manifest_sha256, data_dir=args.data_dir,
        deps_dir=args.deps_dir, image=args.image)
    selected = select_batch(tasks, args.batch_index, args.data_dir)
    if args.check:
        print(json.dumps({'uses_gpu': False, 'full_task_count': 57, 'scheduled': len(selected),
                          'arm': args.arm, 'batch_index': args.batch_index}))
        return
    if not all((args.backend_file, args.endpoint, args.model, args.output)):
        raise ValueError('backend/endpoint/model/output required for a batch attempt')
    health = json.loads(Path(args.backend_file).read_text())
    if (health.get('model') != args.model or health.get('ready') is not True
            or health.get('max_context') != 16384 or health.get('max_output_tokens') != 1024
            or health.get('backend', {}).get('arm') != args.arm):
        raise ValueError('health record differs from fixed dev service')
    from scripts.owned_appworld_v3 import OwnedAppWorldEnvironmentV3

    def factory(task, callback):
        world = OwnedAppWorldEnvironmentV3(task.task_id, args.data_dir, 40, args.image,
            args.deps_dir, environment_seed=100, on_container_intent=callback)
        return DocumentationFeedbackWorld(world)

    model = LocalModel(args.endpoint, args.model, max_tokens=1024, temperature=.7)
    result = run_batch(tasks, factory, model, batch_index=args.batch_index, arm=args.arm,
        data_dir=args.data_dir, output=args.output, expected_backend=health['backend'],
        image=args.image, environment_id=metadata['environment_id'],
        prepared_manifest_sha256=args.task_manifest_sha256,
        before_task=lambda task: wait_for_idle(args.endpoint, args.model, health['backend'], seconds=180))
    print(json.dumps({key: result[key] for key in ('status', 'arm', 'batch_index', 'scheduled',
                     'attempted', 'completed', 'unknown_outcomes', 'valid_for_comparison')}), flush=True)
    if not result['valid_for_comparison']:
        raise SystemExit(1)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
