"""Independently audit one fixed dev batch after the caller verifies client exit.

The expected manifest must be built from frozen metadata and verified health,
not from this run. This module does not inspect processes, Docker or GPUs and
never certifies their release. Private task descriptions and traces are not
included in its returned report.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from scripts import run_timed_tasks_v2 as v2
from scripts import run_timed_tasks_v3 as v3
from scripts.report_b3_dev_pairs import load_official_ids
from wll.protocol import fingerprint

EXPECTED_FIELDS = {
    'arm', 'batch_index', 'slice', 'tasks', 'batch_task_ids', 'model_id', 'expected_backend',
    'image', 'environment_id', 'data_dir', 'observation_feedback', 'seed_protocol',
    'prepared_manifest_sha256', 'max_steps', 'rollout_seconds', 'seed', 'max_tokens',
    'temperature', 'prompt_version', 'official_dev_sha256', 'preparation_sha256',
    'script_sha256', 'deadline_scope',
}
JOURNALS = ('events', 'attempts', 'episodes', 'steps', 'responses', 'feedback-events')
UNKNOWN_SEED = {'requested': 100, 'effective': None, 'verified': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _snapshot(directory):
    directory = Path(directory)
    require(not directory.is_symlink() and directory.is_dir(), 'run directory missing or symlinked')
    result = {}
    for name in ('manifest.json', 'summary.json', *(item + '.jsonl' for item in JOURNALS)):
        path = directory / name
        require(not path.is_symlink(), 'symlinked run evidence')
        if path.exists():
            require(path.is_file(), 'non-file run evidence')
            result[name] = _sha(path.read_bytes())
    require('manifest.json' in result, 'manifest missing')
    return result


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _seed(value):
    require(type(value) is dict and set(value) == {'requested', 'effective', 'verified'}
            and type(value['requested']) is int and value['requested'] == 100
            and type(value['verified']) is bool, 'invalid environment seed evidence')
    require((value['verified'] is False and value['effective'] is None)
            or (value['verified'] is True and type(value['effective']) is int and value['effective'] == 100),
            'environment seed mismatch')
    return dict(value)


def _validate_manifest(manifest, expected, terminated):
    require(type(expected) is dict and set(expected) == EXPECTED_FIELDS, 'complete external expected manifest required')
    require(type(manifest) is dict and EXPECTED_FIELDS <= set(manifest)
            and set(manifest) <= EXPECTED_FIELDS | {'status', 'client_identity', 'started_epoch',
                                                     'environment_seed', 'stop_reason'}
            and all(manifest[name] == expected[name] and type(manifest[name]) is type(expected[name])
                    for name in EXPECTED_FIELDS), 'run manifest differs from external expectation')
    ids = load_official_ids()
    tasks = expected['tasks']
    require(type(tasks) is list and len(tasks) == 57 and all(type(t) is dict for t in tasks)
            and [task.get('task_id') for task in tasks] == ids
            and all(set(task) == {'task_id', 'scenario_id', 'split', 'instruction', 'supervisor'}
                    and task.get('scenario_id') == task['task_id'].rsplit('_', 1)[0]
                    and task.get('split') == 'dev' and type(task.get('instruction')) is str
                    and bool(task['instruction']) and type(task.get('supervisor')) is str for task in tasks),
            'full official dev task order differs')
    index = expected['batch_index']
    require(type(index) is int and 0 <= index < 10, 'invalid fixed batch index')
    start, end = index * 6, min(index * 6 + 6, 57)
    require(expected['slice'] == [start, end] and expected['batch_task_ids'] == ids[start:end]
            and expected['arm'] in ('B0', 'B3') and type(expected['expected_backend']) is dict
            and expected['expected_backend'].get('arm') == expected['arm']
            and type(expected['model_id']) is str and expected['model_id'], 'fixed slice or backend differs')
    require(type(expected['prepared_manifest_sha256']) is str
            and re.fullmatch('[0-9a-f]{64}', expected['prepared_manifest_sha256']) is not None
            and expected['official_dev_sha256'] == v2.OFFICIAL_DEV_SHA256
            and expected['preparation_sha256'] == v2.PREPARATION_SHA256
            and expected['script_sha256'] == v3.file_sha256(v3.__file__)
            and expected['observation_feedback'] == v3.feedback_recipe()
            and expected['seed_protocol'] == v3.SEED_PROTOCOL
            and expected['environment_id'] == v3.environment_identity(expected['image']),
            'prepared metadata, source or environment binding differs')
    require(type(expected['seed']) is int and expected['seed'] == 17
            and type(expected['max_steps']) is int and expected['max_steps'] == 40
            and type(expected['max_tokens']) is int and expected['max_tokens'] == 1024
            and type(expected['rollout_seconds']) is int and expected['rollout_seconds'] == 600
            and type(expected['temperature']) is float and expected['temperature'] == .7
            and expected['prompt_version'] == 'appworld-v3'
            and expected['deadline_scope'] == 'rollout only; parent owns hard watchdog and independent release verification',
            'fixed generation or deadline recipe differs')
    identity = manifest.get('client_identity')
    require(type(identity) is dict and set(identity) == {'pid', 'start_ticks'}
            and all(type(identity[k]) is int and identity[k] > 0 for k in identity)
            and type(terminated) is dict and terminated.get('verified_exited') is True
            and all(type(terminated.get(k)) is int and terminated[k] == identity[k] for k in identity),
            'verified terminated client identity differs')
    require(manifest.get('status') in ('running', 'complete', 'incomplete', 'interrupted')
            and _number(manifest.get('started_epoch')) and manifest['started_epoch'] > 0,
            'invalid manifest status or time')
    _seed(manifest.get('environment_seed'))
    return ids[start:end]


def _invalid_report(expected, hashes, errors):
    return {'audit': 'incomplete_evidence', 'arm': expected['arm'], 'batch_index': expected['batch_index'],
        'scheduled': len(expected['batch_task_ids']), 'attempted': None, 'completed': 0,
        'unknown_outcomes': len(expected['batch_task_ids']), 'unattempted': None, 'valid_for_comparison': False,
        'journal_integrity_verified': False, 'reconciliation_errors': errors,
        'normalized_outcomes': [{'task_id': task_id, 'status': 'unknown', 'success': None,
                                 'generation_seed': 17, 'environment_seed': dict(UNKNOWN_SEED)}
                                for task_id in expected['batch_task_ids']],
        'costs': {'generation_calls': None, 'generation_calls_without_response': None,
                  'generation_calls_with_unusable_usage': None, 'returned_response_cost': None,
                  'observed_step_cost': None, 'unknown_additional_generation_cost': True,
                  'observed_attempt_wall_seconds': None, 'observed_backend_wait_seconds': None,
                  'observed_startup_seconds': None, 'timing_coverage': None,
                  'costs_are_lower_bounds': True},
        'cleanup_release_independently_verified': False, 'run_files_sha256': hashes,
        'scope': 'Journal damaged or missing; every task is unknown, including start status. '
                 'Client exit is supplied by the caller; process, container and GPU release are not audited.'}


def _read_rows(directory):
    result, errors = {}, []
    for name in JOURNALS:
        rows, issues = v3._read_journal(directory / (name + '.jsonl'))
        result[name] = rows
        errors.extend(issues)
    return result, errors


def _unchanged(directory, before, expected):
    require(_snapshot(directory) == before and v3.file_sha256(v3.__file__) == expected['script_sha256']
            and v3.feedback_recipe() == expected['observation_feedback']
            and [task['task_id'] for task in expected['tasks']] == load_official_ids(),
            'run or source evidence changed during audit')


def _events(rows, manifest):
    selected = manifest['batch_task_ids']
    require(rows and rows[0].get('event') == 'batch_started'
            and rows[0].get('client_identity') == manifest['client_identity']
            and rows[0].get('arm') == manifest['arm'] and rows[0].get('batch_index') == manifest['batch_index']
            and rows[0].get('batch_task_ids') == selected, 'batch-start event differs')
    started, intents, ready, calls, finishes = [], {}, {}, [], {}
    used_names, used_tokens = set(), set()
    for index, event in enumerate(rows):
        require(type(event) is dict and type(event.get('event_index')) is int
                and event['event_index'] == index and _number(event.get('epoch')), 'invalid event index or time')
        kind, task_id = event.get('event'), event.get('task_id')
        if index == 0:
            continue
        if kind == 'task_started':
            require(len(started) < len(selected) and task_id == selected[len(started)]
                    and not any(call['call_id'] not in finishes for call in calls),
                    'task order differs or previous call unresolved')
            started.append(task_id)
            continue
        require(started and task_id == started[-1], 'event task differs from active task')
        if kind == 'container_intent':
            value = v3._intent(event.get('container_intent'), manifest['client_identity'], manifest['image'])
            require(task_id not in intents and task_id not in ready
                    and value['container_name'] not in used_names and value['ownership_token'] not in used_tokens,
                    'duplicate or reordered container intent')
            intents[task_id] = value
            used_names.add(value['container_name'])
            used_tokens.add(value['ownership_token'])
        elif kind == 'world_ready':
            require(task_id in intents and task_id not in ready, 'world lacks unique owned intent')
            ready[task_id] = v2._validate_seed(event.get('environment_seed'))
        elif kind == 'generation_attempt':
            number = event.get('call_id')
            task_calls = [call for call in calls if call['task_id'] == task_id]
            require(task_id in ready and type(number) is int and number == len(calls) + 1
                    and type(event.get('seed')) is int and event['seed'] == 17 + len(task_calls)
                    and len(task_calls) < 40 and not any(call['call_id'] not in finishes for call in calls)
                    and not any(finishes[call['call_id']]['event'] == 'generation_failed'
                                or (finishes[call['call_id']]['event'] == 'generation_returned'
                                    and finishes[call['call_id']]['usage_valid'] is False)
                                for call in task_calls), 'invalid generation attempt or seed')
            calls.append(event)
        elif kind in ('generation_returned', 'generation_failed'):
            number = event.get('call_id')
            require(type(number) is int and calls and number == calls[-1]['call_id']
                    and number not in finishes and calls[-1]['task_id'] == task_id
                    and event.get('seed') == calls[-1]['seed'], 'generation outcome differs from attempt')
            if kind == 'generation_returned':
                require(type(event.get('usage_valid')) is bool, 'missing usage validity')
                if event['usage_valid']:
                    require(all(type(event.get(k)) is int and event[k] >= 0
                                for k in ('prompt_tokens', 'completion_tokens')), 'invalid returned usage')
                else:
                    require(event.get('prompt_tokens') is None and event.get('completion_tokens') is None,
                            'invalid unusable-usage event')
            else:
                require(type(event.get('error_type')) is str and event['error_type'], 'missing generation error')
            finishes[number] = event
        else:
            raise ValueError('unknown durable event')
    return started, intents, ready, calls, finishes


def _attempts(rows, started, intents, ready):
    require(len(rows) <= len(started) and len(started) - len(rows) <= 1,
            'final attempt records are not a prefix of started tasks')
    result = {}
    for index, row in enumerate(rows):
        require(row.get('task_id') == started[index] and row.get('status') in {
            'complete', 'timed_out', 'infrastructure_error', 'interrupted'}, 'invalid final attempt order')
        task_id, status = row['task_id'], row['status']
        seed = _seed(row.get('environment_seed'))
        if task_id in ready:
            require(seed == ready[task_id], 'attempt seed differs from world-ready evidence')
        else:
            require(seed == UNKNOWN_SEED, 'seed verified without world-ready evidence')
        require(_number(row.get('started_epoch')) and _number(row.get('finished_epoch'))
                and row['finished_epoch'] >= row['started_epoch'], 'invalid attempt timing')
        for key in ('backend_wait_seconds', 'startup_seconds'):
            require(key not in row or _number(row[key]), 'invalid observed time')
        require(type(row.get('feedback_events')) is int and row['feedback_events'] >= 0,
                'invalid feedback event count')
        if 'container_intent' in row:
            require(row['container_intent'] == intents.get(task_id), 'attempt container differs from intent')
        if status == 'complete':
            require(type(row.get('success')) is bool and row.get('cleanup_verified') is True
                    and task_id in ready and row.get('container_intent') == intents.get(task_id)
                    and type(row.get('episode_hash')) is str
                    and re.fullmatch('[0-9a-f]{64}', row['episode_hash']) is not None,
                    'completed attempt lacks verified outcome or cleanup')
        else:
            cleanup = row.get('cleanup_verified')
            require(row.get('success') is None and (cleanup is None or type(cleanup) is bool),
                    'unknown attempt has a claimed success or invalid cleanup')
        result[task_id] = row
    return result


def _responses(rows, started, calls, finishes, attempts, expected_backend):
    valid = {task_id: [] for task_id in started}
    for call in calls:
        end = finishes.get(call['call_id'])
        if end is not None and end['event'] == 'generation_returned' and end['usage_valid']:
            valid[call['task_id']].append(end)
    grouped = {task_id: [] for task_id in started}
    expected_hash = fingerprint(expected_backend)
    order = []
    for row in rows:
        name = row.get('task_id')
        require(type(name) is str and name in grouped, 'response belongs to an unstarted task')
        order.append(started.index(name))
        index = len(grouped[name])
        require(index < len(valid[name]), 'response has no matching returned event')
        event = valid[name][index]
        require(type(row.get('seed')) is int and row['seed'] == event['seed']
                and type(row.get('backend_matches')) is bool and type(row.get('backend_sha256')) is str
                and re.fullmatch('[0-9a-f]{64}', row['backend_sha256']) is not None
                and row['backend_matches'] == (row['backend_sha256'] == expected_hash)
                and all(type(row.get(k)) is int and row[k] >= 0 and row[k] == event[k]
                        for k in ('prompt_tokens', 'completion_tokens')),
                'response differs from durable returned event or backend')
        require(not grouped[name] or grouped[name][-1]['backend_matches'] is True,
                'response continued after backend mismatch')
        grouped[name].append(row)
    require(order == sorted(order), 'response order differs')
    for name, attempt in attempts.items():
        if attempt['status'] == 'complete':
            require(len(grouped[name]) == len(valid[name])
                    and all(row['backend_matches'] is True for row in grouped[name]),
                    'completed task lacks matching responses')
    return grouped


def _steps(rows, started, responses):
    grouped = {task_id: [] for task_id in started}
    order = []
    for row in rows:
        name, step = row.get('task_id'), row.get('step')
        require(type(name) is str and name in grouped and type(step) is dict, 'step belongs to an unstarted task')
        order.append(started.index(name))
        index = len(grouped[name])
        require(type(row.get('index')) is int and row['index'] == index and index < 40
                and row.get('replayed') is False and index < len(responses[name])
                and responses[name][index]['backend_matches'] is True
                and all(type(step.get(k)) is int and step[k] >= 0
                        and step[k] == responses[name][index][k]
                        for k in ('prompt_tokens', 'completion_tokens'))
                and _number(step.get('elapsed_seconds')) and type(step.get('action')) is str
                and type(step.get('observation')) is str, 'invalid step or response association')
        grouped[name].append(step)
    require(order == sorted(order), 'step order differs')
    return grouped


def _feedback(rows, started, steps, attempts):
    grouped = {task_id: [] for task_id in started}
    order = []
    for row in rows:
        name, index = row.get('task_id'), row.get('index')
        require(type(name) is str and name in grouped and type(index) is int and 0 <= index < 40,
                'invalid feedback task or step')
        order.append(started.index(name))
        previous = grouped[name]
        require(type(row.get('feedback_events')) is int and row['feedback_events'] == len(previous) + 1
                and (not previous or previous[-1]['index'] < index), 'feedback counter or order differs')
        require(all(type(row.get(key)) is str and re.fullmatch('[0-9a-f]{64}', row[key]) is not None
                    for key in ('action_sha256', 'augmented_observation_sha256')),
                'invalid feedback hash')
        if index < len(steps[name]):
            step = steps[name][index]
            require(row.get('action_sha256') == fingerprint(step['action'])
                    and row.get('augmented_observation_sha256') == fingerprint(step['observation']),
                    'feedback hashes differ from step')
        else:
            require(index == len(steps[name]) and (name not in attempts or attempts[name]['status'] != 'complete'),
                    'feedback refers to an absent completed step')
        previous.append(row)
    require(order == sorted(order), 'feedback order differs')
    for name, attempt in attempts.items():
        observed = len(grouped[name])
        require(attempt['feedback_events'] == observed if attempt['status'] == 'complete'
                else observed <= attempt['feedback_events'] <= observed + 1,
                'attempt feedback count differs')
    return grouped


def _episodes(rows, manifest, started, ready, attempts, steps, feedback):
    tasks = {task['task_id']: task for task in manifest['tasks']}
    by_task, order = {}, []
    for episode in rows:
        task = episode.get('task')
        name = task.get('task_id') if type(task) is dict else None
        require(type(name) is str and name in started and name not in by_task, 'episode task missing or duplicated')
        order.append(started.index(name))
        meta = episode.get('metadata')
        require(task == tasks[name] and episode.get('model_id') == manifest['model_id']
                and type(episode.get('seed')) is int and episode['seed'] == 17
                and type(episode.get('success')) is bool and episode.get('branch') == 'first'
                and episode.get('experience_id') is None and type(episode.get('replayed_steps')) is int
                and episode['replayed_steps'] == 0
                and episode.get('steps') == steps[name] and _number(episode.get('elapsed_seconds'))
                and type(meta) is dict and meta.get('environment_id') == manifest['environment_id']
                and meta.get('environment_seed') == ready.get(name)
                and meta.get('observation_feedback') == manifest['observation_feedback']
                and meta.get('feedback_events') == len(feedback[name])
                and meta.get('rollout_deadline_seconds') == manifest['rollout_seconds']
                and meta.get('arm') == manifest['arm'] and meta.get('batch_index') == manifest['batch_index']
                and all(type(meta.get(k)) is type(manifest[k]) and meta.get(k) == manifest[k] for k in
                        ('max_steps', 'prompt_version', 'max_tokens', 'temperature')),
                'episode identity, metadata or steps differ')
        if steps[name]:
            require(meta.get('backend') == manifest['expected_backend'], 'episode backend differs')
        by_task[name] = episode
    require(order == sorted(order), 'episode order differs')
    for name, attempt in attempts.items():
        if attempt['status'] == 'complete':
            require(name in by_task and by_task[name]['success'] is attempt['success']
                    and fingerprint(by_task[name]) == attempt['episode_hash'],
                    'completed attempt and episode differ')
    return by_task


def audit_batch(run_dir, *, expected_manifest, terminated_identity):
    """Recompute this batch's known outcomes and costs from durable journals."""
    directory = Path(run_dir)
    before = _snapshot(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    selected = _validate_manifest(manifest, expected_manifest, terminated_identity)
    rows, issues = _read_rows(directory)
    if issues or not rows['events']:
        _unchanged(directory, before, expected_manifest)
        return _invalid_report(expected_manifest, before, issues or ['events.jsonl:missing_batch_start'])
    reconciled = v3.reconcile_batch(directory, terminated_identity=terminated_identity)
    if not reconciled['journal_integrity_verified']:
        raise ValueError('durable journal reconciliation failed')
    started, intents, ready, calls, finishes = _events(rows['events'], manifest)
    attempts = _attempts(rows['attempts'], started, intents, ready)
    responses = _responses(rows['responses'], started, calls, finishes, attempts, manifest['expected_backend'])
    steps = _steps(rows['steps'], started, responses)
    for name, attempt in attempts.items():
        if attempt['status'] == 'complete':
            require(len(steps[name]) == len(responses[name]), 'completed response lacks a committed step')
    feedback = _feedback(rows['feedback-events'], started, steps, attempts)
    _episodes(rows['episodes'], manifest, started, ready, attempts, steps, feedback)

    returned = [event for event in finishes.values() if event['event'] == 'generation_returned']
    usable = [event for event in returned if event['usage_valid']]
    no_reply = len(calls) - len(returned)
    unusable = sum(not event['usage_valid'] for event in returned)
    costs = {'generation_calls': len(calls), 'generation_calls_without_response': no_reply,
             'generation_calls_with_unusable_usage': unusable,
             'returned_response_cost': {key: sum(event[key] for event in usable)
                                        for key in ('prompt_tokens', 'completion_tokens')},
             'observed_step_cost': {'steps': sum(len(value) for value in steps.values()),
                 **{key: sum(step[key] for group in steps.values() for step in group)
                    for key in ('prompt_tokens', 'completion_tokens')}},
             'unknown_additional_generation_cost': bool(no_reply or unusable),
             'observed_attempt_wall_seconds': sum(row['finished_epoch'] - row['started_epoch']
                                                  for row in attempts.values()),
             'observed_backend_wait_seconds': sum(row.get('backend_wait_seconds', 0) for row in attempts.values()),
             'observed_startup_seconds': sum(row.get('startup_seconds', 0) for row in attempts.values()),
             'timing_coverage': {'started_tasks': len(started), 'finalized_attempts': len(attempts),
                 'attempts_with_backend_wait': sum('backend_wait_seconds' in row for row in attempts.values()),
                 'attempts_with_startup': sum('startup_seconds' in row for row in attempts.values())},
             'costs_are_lower_bounds': bool(no_reply or unusable or len(attempts) < len(started)
                 or any('backend_wait_seconds' not in row or 'startup_seconds' not in row
                        for row in attempts.values()))}
    require(all(reconciled.get(key) == costs[key] for key in (
        'generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage',
        'returned_response_cost', 'observed_step_cost')), 'independent costs differ from reconciliation')
    outcomes = []
    for name in selected:
        attempt = attempts.get(name)
        status = ('unattempted' if name not in started else
                  'complete' if attempt is not None and attempt['status'] == 'complete' else 'unknown')
        outcomes.append({'task_id': name, 'status': status,
                         'success': attempt['success'] if status == 'complete' else None,
                         'generation_seed': 17, 'environment_seed': dict(ready.get(name, UNKNOWN_SEED))})
    completed = sum(row['status'] == 'complete' for row in outcomes)
    if manifest['environment_seed']['verified'] is True:
        require(len(started) == len(selected) and len(ready) == len(selected),
                'manifest claims verified seeds without durable evidence')
    if manifest['status'] == 'complete':
        require(completed == len(selected) and manifest['environment_seed']['verified'] is True,
                'manifest claims completion without all outcomes and verified seeds')
    elif manifest['status'] == 'incomplete':
        require(completed < len(selected), 'incomplete manifest contradicts all completed outcomes')
    require(reconciled['scheduled'] == len(selected) and reconciled['attempted'] == len(started)
            and reconciled['completed'] == completed
            and reconciled['unattempted'] == len(selected) - len(started), 'reconciliation differs')
    _unchanged(directory, before, expected_manifest)
    return {'audit': 'passed', 'arm': manifest['arm'], 'batch_index': manifest['batch_index'],
            'scheduled': len(selected), 'attempted': len(started), 'completed': completed,
            'unknown_outcomes': len(selected) - completed, 'unattempted': len(selected) - len(started),
            'valid_for_comparison': completed == len(selected), 'journal_integrity_verified': True,
            'normalized_outcomes': outcomes, 'costs': costs,
            'cleanup_release_independently_verified': False, 'run_files_sha256': before,
            'scope': 'Fixed dev batch only; caller supplies client-exit verification. Process, '
                     'container and GPU release and full-schedule paired comparison require separate audit. '
                     'V3 and fixed feedback source hashes are checked; the caller must independently bind '
                     'all other sources, prepared metadata and the actual service health to its frozen plan. '
                     'Attempt wall includes startup, waiting and rollout; returned-response and observed-step '
                     'token views overlap. Do not add overlapping costs.'}
