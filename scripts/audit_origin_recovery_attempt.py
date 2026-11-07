"""Independently audit one private origin-attempt directory without executing it.

The caller supplies the private Task and externally bound source/proof objects.
Only hashes and classifications leave this function. It neither discovers data
nor proves admission, service/process/container release, or training readiness.
All twelve source audits, source/proof file admission, and serving provenance
remain parent-controller obligations. ``audit_sha256`` fingerprints this report
without its own field; it is not the byte hash of a subsequently saved JSON file.
"""
import copy
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict
from pathlib import Path

from scripts.documentation_feedback import FEEDBACK
from scripts.probe_observation_visibility import printable_documentation_call
from scripts.run_timed_tasks_v3 import SEED_PROTOCOL, environment_identity, feedback_recipe
from wll.prompts import initial_messages
from wll.protocol import Episode, Step, Task, fingerprint
from wll.runner import extract_code

ROOT = Path(__file__).resolve().parents[1]
DESIGN = 'artifacts/qwen38-origin-recovery-design-001.json'
DESIGN_SHA256 = '5563a02031871d1ac34a50e3b2c685485b3d1e4e45babc269021c814ac35e46e'
RUNTIME_SHA256 = '3a966b652074b48edbaee5449abc8e5b080fd749a4fb66b92122b289ea5342d0'
ENGINE_SHA256 = '04eb4c807b92a4d44f373ba4fd2c2e4d4014c1d194dfdb1e1294e731a61b6a5b'
JOURNALS = ('events', 'attempts', 'episodes', 'steps', 'responses', 'feedback-events')


class _Invalid(ValueError):
    """Codes deliberately contain no private task, action, response or exception text."""


def _check(condition, code):
    if not condition:
        raise _Invalid(code)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _fp(value):
    # Preserve protocol.fingerprint's serialization, rejecting non-standard JSON.
    json.dumps(value, allow_nan=False)
    return fingerprint(value)


def _same(left, right):
    # Python equality aliases True/1 and False/0. Durable protocol types do not.
    return _fp(left) == _fp(right)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _check(key not in value, 'duplicate_json_key')
        value[key] = item
    return value


def _json(raw):
    def invalid_constant(value):
        raise _Invalid('nonfinite_json')
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=invalid_constant)


def _read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        _check(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'unsafe_file')
        return stream.read()


def _digest(path):
    return hashlib.sha256(_read(path)).hexdigest()


def _load(directory):
    if not directory.is_dir() or directory.is_symlink():
        return {}, {}, ['unsafe_directory']
    data, hashes, errors = {}, {}, []
    for name in ('manifest', *JOURNALS):
        path = directory / (name + ('.json' if name == 'manifest' else '.jsonl'))
        try:
            raw = _read(path)
            hashes[path.name] = hashlib.sha256(raw).hexdigest()
            if name == 'manifest':
                data[name] = _json(raw)
                _check(type(data[name]) is dict, 'manifest_schema')
            else:
                _check(not raw or raw.endswith(b'\n'), 'truncated_journal')
                data[name] = [_json(line) for line in raw.splitlines()]
                _check(all(type(row) is dict for row in data[name]), 'journal_schema')
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
            data.pop(name, None)
            errors.append('unreadable_' + name)
    return data, hashes, errors


def _semantic(backend):
    _check(type(backend) is dict and backend.get('arm') == 'B3'
           and type(backend.get('gpu_policy')) is dict, 'backend_schema')
    gpu = backend['gpu_policy']
    _check(_integer(backend.get('loading_peak_reserved_bytes'), 1)
           and type(gpu.get('physical_gpu_index')) is int
           and type(gpu.get('gpu_uuid')) is str and bool(gpu['gpu_uuid']), 'backend_schema')
    value = copy.deepcopy(backend)
    del value['loading_peak_reserved_bytes']
    del value['gpu_policy']['physical_gpu_index']
    del value['gpu_policy']['gpu_uuid']
    return _fp(value)


def _seed(value):
    return (type(value) is dict and set(value) == {'requested', 'effective', 'verified'}
            and type(value['requested']) is int and value['requested'] == 100
            and type(value['effective']) is int and value['effective'] == 100
            and value['verified'] is True)


def _step(value):
    _check(type(value) is dict and set(value) == set(Step.__dataclass_fields__), 'step_schema')
    _check(all(type(value[key]) is str for key in ('action', 'observation', 'raw_response'))
           and all(_integer(value[key]) for key in ('prompt_tokens', 'completion_tokens'))
           and _seconds(value['elapsed_seconds']), 'step_schema')
    _check(extract_code(value['raw_response']) == value['action'], 'raw_action')


def _episode(value, *, task, model_id, backend, metadata_sha, image, environment_id,
             replayed=0, max_steps=40, branch='first', semantic_only=False):
    _check(type(value) is dict and set(value) == set(Episode.__dataclass_fields__), 'episode_schema')
    _check(_same(value['task'], asdict(task)) and value['model_id'] == model_id
           and type(value['seed']) is int and value['seed'] == 17
           and value['experience_id'] is None and value['branch'] == branch
           and type(value['replayed_steps']) is int and value['replayed_steps'] == replayed
           and type(value['success']) is bool and _seconds(value['elapsed_seconds'])
           and type(value['steps']) is list and replayed <= len(value['steps']) <= max_steps
           and value['termination'] in ('completed', 'step_budget'), 'episode_identity')
    _check(value['termination'] != 'step_budget' or len(value['steps']) == max_steps,
           'episode_termination')
    for step in value['steps']:
        _step(step)
    meta = value['metadata']
    _check(type(meta) is dict and type(meta.get('max_steps')) is int and meta['max_steps'] == max_steps
           and type(meta.get('max_tokens')) is int and meta['max_tokens'] == 1024
           and type(meta.get('temperature')) is float and meta['temperature'] == .7
           and meta.get('prompt_version') == 'appworld-v3' and meta.get('image') == image
           and meta.get('environment_id') == environment_id and _seed(meta.get('environment_seed'))
           and meta.get('observation_feedback') == feedback_recipe()
           and meta.get('prepared_manifest_sha256') == metadata_sha
           and meta.get('backend_semantic_sha256') == _semantic(backend)
           and _sha(meta.get('expected_backend_sha256')), 'episode_metadata')
    if len(value['steps']) > replayed:
        observed = meta.get('backend')
        _check(_fp(observed) == meta['expected_backend_sha256']
               and _semantic(observed) == _semantic(backend), 'episode_backend')
        if not semantic_only:
            _check(_same(observed, backend) and meta['expected_backend_sha256'] == _fp(backend),
                   'episode_backend')
    else:
        _check(meta.get('backend') is None, 'episode_backend')
        if not semantic_only:
            _check(meta['expected_backend_sha256'] == _fp(backend), 'episode_backend')


def _prefix(episode):
    for index, step in enumerate(episode['steps']):
        if step['observation'].startswith('Execution failed'):
            return ('no_budget' if index == 39 else 'candidate'), episode['steps'][:index + 1]
    return 'no_error', []


def _bindings(task, kind, model_id, backend, metadata_sha, source, source_sha, proof, proof_sha):
    raw = _read(ROOT / DESIGN)
    _check(hashlib.sha256(raw).hexdigest() == DESIGN_SHA256, 'design_source')
    design = _json(raw)
    _check(_digest(ROOT / 'scripts/run_origin_recovery_attempt.py') == RUNTIME_SHA256, 'runtime_source')
    _check(kind in ('source', 'prefix_check', 'control', 'hint')
           and isinstance(task, Task) and task.task_id in design['selection']['task_ids']
           and task.scenario_id == task.task_id.rsplit('_', 1)[0] and task.split == 'train'
           and type(task.instruction) is str and bool(task.instruction)
           and type(task.supervisor) is str and type(model_id) is str and bool(model_id)
           and _sha(metadata_sha), 'input_identity')
    _semantic(backend)
    image, environment_id = design['protocol']['image'], design['protocol']['environment_id']
    _check(environment_identity(image) == environment_id, 'environment_source')
    classification, prefix = None, []
    if kind == 'source':
        _check(all(value is None for value in (source, source_sha, proof, proof_sha)), 'source_inputs')
    else:
        _check(isinstance(source, Episode) and _sha(source_sha) and _fp(source.to_dict()) == source_sha,
               'source_binding')
        value = source.to_dict()
        _episode(value, task=task, model_id=model_id, backend=backend, metadata_sha=metadata_sha,
                 image=image, environment_id=environment_id, semantic_only=True)
        classification, prefix = _prefix(value)
        if kind == 'prefix_check':
            _check(proof is None and proof_sha is None, 'proof_inputs')
        else:
            _check(classification == 'candidate' and type(proof) is dict and _sha(proof_sha)
                   and _fp(proof) == proof_sha, 'proof_binding')
            expected = {'kind': 'prefix_check', 'status': 'eligible', 'source_sha256': source_sha,
                'prefix_sha256': _fp(prefix), 'prefix_length': len(prefix), 'task_id': task.task_id,
                'task_sha256': _fp(asdict(task)), 'model_id': model_id, 'backend_semantic_sha256': _semantic(backend),
                'image': image, 'environment_id': environment_id, 'seed': 17,
                'prepared_manifest_sha256': metadata_sha}
            _check(all(proof.get(key) == value for key, value in expected.items())
                   and type(proof.get('seed')) is int and type(proof.get('prefix_length')) is int
                   and proof.get('prefix_success') is False and proof.get('prefix_completed') is False
                   and proof.get('local_cleanup_complete') is True and _seed(proof.get('environment_seed'))
                   and _sha(proof.get('backend_sha256')) and proof.get('total_resources_released') is None,
                   'proof_binding')
    return design, classification, prefix


def _costs(data, invalid):
    events, record = data.get('events', []), {}
    if data.get('attempts') and len(data['attempts']) > 1:
        record = data['attempts'][-1]
    costs = {'generation': {'attempted': 0, 'returned': 0, 'prompt_tokens': 0,
        'completion_tokens': 0, 'unknown_additional_cost': bool(invalid)},
        'lower_bound_only': bool(invalid),
        'note': 'Returned generation usage overlaps Step usage. Historical replay Step costs are excluded; '
                'execute, startup, cleanup and rollout/wall views must not be added together.'}
    for name in ('replay', 'new_execution'):
        costs[name] = {'attempted': 0, 'returned': 0, 'elapsed_seconds': 0.0,
                       'unknown_additional_cost': bool(invalid)}
    for row in events:
        event = row.get('event')
        if event == 'generation_attempt':
            costs['generation']['attempted'] += 1
        if event == 'generation_returned':
            costs['generation']['returned'] += 1
            if row.get('usage_valid') is True and all(_integer(row.get(key)) for key in
                                                    ('prompt_tokens', 'completion_tokens')):
                for key in ('prompt_tokens', 'completion_tokens'):
                    costs['generation'][key] += row[key]
            else:
                costs['generation']['unknown_additional_cost'] = True
        for prefix, name in (('replay', 'replay'), ('execute', 'new_execution')):
            if event == prefix + '_attempt':
                costs[name]['attempted'] += 1
            if event == prefix + '_returned':
                costs[name]['returned'] += 1
            if event in (prefix + '_returned', prefix + '_failed') and _seconds(row.get('elapsed_seconds')):
                costs[name]['elapsed_seconds'] += row['elapsed_seconds']
    for name in ('generation', 'replay', 'new_execution'):
        costs[name]['unknown_additional_cost'] |= costs[name]['attempted'] != costs[name]['returned']
    for key in ('startup_seconds', 'cleanup_seconds', 'rollout_seconds', 'wall_seconds'):
        value = record.get(key)
        costs[key] = value if _seconds(value) else None
    return costs


def _full_history(backend, prompt_tokens):
    """Source-bound serving contract + actual usage, not a token-ID attestation.

    The parent must admit the real backend/service and its frozen source map.
    Without that evidence a reconstructed client transcript alone is insufficient.
    """
    sources, recipe = backend.get('source_sha256'), backend.get('recipe')
    return (type(sources) is dict and sources.get('scripts/serve_qwen38.py') == ENGINE_SHA256
            and _digest(ROOT / 'scripts/serve_qwen38.py') == ENGINE_SHA256
            and type(recipe) is dict and type(recipe.get('max_context')) is int
            and recipe['max_context'] == 16384 and type(recipe.get('max_output_tokens')) is int
            and recipe['max_output_tokens'] == 1024 and _integer(prompt_tokens, 1)
            and prompt_tokens + 1024 <= 16384)


def _audit(data, *, task, kind, model_id, backend, metadata_sha, source, source_sha, proof, proof_sha):
    design, classification, prefix = _bindings(task, kind, model_id, backend, metadata_sha,
                                               source, source_sha, proof, proof_sha)
    image, env = design['protocol']['image'], design['protocol']['environment_id']
    hint = design['protocol']['hint']
    manifest = data['manifest']
    shared = {'kind': kind, 'task_id': task.task_id, 'task_sha256': _fp(asdict(task)),
        'model_id': model_id, 'backend_sha256': _fp(backend), 'backend_semantic_sha256': _semantic(backend),
        'image': image, 'environment_id': env, 'seed': 17, 'prepared_manifest_sha256': metadata_sha,
        'source_sha256': source_sha}
    _check(all(manifest.get(key) == value for key, value in shared.items()), 'manifest_binding')
    _check(manifest.get('attempt_script_sha256') == RUNTIME_SHA256
           and manifest.get('preparation_script_sha256') == design['selection']['component_sha256']
           and manifest.get('prefix_proof_sha256') == proof_sha
           and manifest.get('hint_sha256') == (_fp(hint) if kind == 'hint' else None)
           and type(manifest.get('max_steps')) is int and manifest['max_steps'] == 40
           and type(manifest.get('seed')) is int and type(manifest.get('max_tokens')) is int
           and manifest['max_tokens'] == 1024 and type(manifest.get('temperature')) is float
           and manifest['temperature'] == .7 and manifest.get('prompt_version') == 'appworld-v3'
           and type(manifest.get('environment_seed_requested')) is int
           and manifest['environment_seed_requested'] == 100 and manifest.get('seed_protocol') == SEED_PROTOCOL
           and manifest.get('observation_feedback') == feedback_recipe()
           and type(manifest.get('deadline_seconds')) is int
           and manifest['deadline_seconds'] == (120 if kind == 'prefix_check' else 600)
           and manifest.get('automatic_retry') is False
           and manifest.get('parent_must_verify_source_admission_and_release') is True,
           'manifest_recipe')
    identity = manifest.get('client_identity')
    _check(type(identity) is dict and set(identity) == {'pid', 'start_ticks'}
           and all(_integer(value, 1) for value in identity.values()), 'client_identity')
    attempts = data['attempts']
    _check(len(attempts) == 2 and attempts[0] == {'kind': kind, 'task_id': task.task_id, 'status': 'started'},
           'attempt_terminal')
    record = attempts[-1]
    _check(all(record.get(key) == value for key, value in shared.items())
           and type(record.get('seed')) is int and record.get('total_resources_released') is None,
           'record_binding')
    _check(manifest.get('status') == record.get('status') and record.get('status') != 'unknown'
           and not any(key in record for key in ('error_type', 'cleanup_error_type')), 'unknown_outcome')
    events = data['events']
    for index, event in enumerate(events):
        _check(type(event.get('event_index')) is int and event['event_index'] == index
               and _seconds(event.get('epoch')), 'event_sequence')
    cursor = 0

    def take(name, details):
        nonlocal cursor
        _check(cursor < len(events), 'missing_event')
        row = events[cursor]
        cursor += 1
        _check(row.get('event') == name and set(row) == {'event', 'event_index', 'epoch', *details},
               'event_schema')
        _check(_same({key: row[key] for key in details}, details), 'event_binding')
        return row

    take('attempt_started', {'kind': kind, 'task_id': task.task_id, 'client_identity': identity})
    skipped = kind == 'prefix_check' and classification in ('no_error', 'no_budget')
    if skipped:
        _check(record['status'] == classification and record.get('success') is None
               and type(record.get('prefix_length')) is int and record['prefix_length'] == len(prefix)
               and record.get('prefix_sha256') == (_fp(prefix) if prefix else None)
               and record.get('local_cleanup_complete') is False
               and all(not data[name] for name in ('steps', 'episodes', 'responses', 'feedback-events'))
               and cursor == len(events), 'ineligible_record')
        result = {'source_classification': classification, 'no_world_created': True,
                  'requests': [], 'targets': [], 'episode_sha256': None}
    else:
        _check(record.get('local_cleanup_complete') is True and _seed(record.get('environment_seed')),
               'incomplete_cleanup_or_seed')
        _check(cursor < len(events), 'missing_container_intent')
        intent = events[cursor].get('container_intent')
        _check(type(intent) is dict and set(intent) == {'container_name', 'image', 'owner_pid',
                'owner_start_ticks', 'ownership_token'}
               and type(intent['ownership_token']) is str
               and re.fullmatch('[0-9a-f]{32}', intent['ownership_token']) is not None
               and intent['container_name'] == 'wll-' + intent['ownership_token'][:16]
               and intent['image'] == image and type(intent['owner_pid']) is int
               and intent['owner_pid'] == identity['pid'] and type(intent['owner_start_ticks']) is int
               and intent['owner_start_ticks'] == identity['start_ticks'], 'container_intent')
        take('container_intent', {'task_id': task.task_id, 'container_intent': intent})
        take('world_ready', {'task_id': task.task_id, 'environment_seed': record['environment_seed']})
        _check(len(data['episodes']) == 1, 'episode_count')
        episode = data['episodes'][0]
        _episode(episode, task=task, model_id=model_id, backend=backend, metadata_sha=metadata_sha,
                 image=image, environment_id=env, replayed=len(prefix),
                 max_steps=len(prefix) if kind == 'prefix_check' else 40,
                 branch='first' if kind == 'source' else 'recovery')
        _check(record.get('episode_sha256') == _fp(episode), 'episode_hash')
        _check(_same(episode['steps'][:len(prefix)], prefix), 'replay_step_copy')
        _check(len(data['steps']) == len(episode['steps']), 'step_count')
        student = initial_messages(task, 'appworld-v3')
        requests, targets, expected_feedback = [], [], []
        hint_index = len(student) + 2 * len(prefix)
        new_count = 0
        for index, step in enumerate(episode['steps']):
            replayed = index < len(prefix)
            row = data['steps'][index]
            _check(_same(row, {'task_id': task.task_id, 'index': index, 'replayed': replayed,
                'raw_action_match': step['raw_response'] == step['action'],
                'raw_action_target_filter_passed': not replayed and step['raw_response'] == step['action'],
                'step': step}) and type(row.get('index')) is int
                and all(type(row.get(key)) is bool for key in
                    ('replayed', 'raw_action_match', 'raw_action_target_filter_passed')), 'step_journal')
            if not replayed:
                _check(1 <= step['prompt_tokens'] <= 15360 and 1 <= step['completion_tokens'] <= 1024,
                       'serving_token_budget')
                teacher = copy.deepcopy(student)
                if kind == 'hint':
                    teacher.insert(hint_index, {'role': 'user', 'content': hint})
                call = {'call_id': new_count + 1, 'seed': 17 + index,
                    'hint_index': hint_index if kind == 'hint' else None,
                    'teacher_messages_sha256': _fp(teacher), 'student_messages_sha256': _fp(student)}
                _check(cursor < len(events), 'missing_request')
                _check(all(events[cursor].get(key) == value for key, value in call.items())
                       and type(events[cursor].get('call_id')) is int
                       and type(events[cursor].get('seed')) is int
                       and (events[cursor].get('hint_index') is None or
                            type(events[cursor]['hint_index']) is int), 'request_messages')
                take('generation_attempt', call)
                take('generation_returned', {**call, 'usage_valid': True,
                    'raw_response': step['raw_response'], 'raw_response_sha256': _fp(step['raw_response']),
                    'prompt_tokens': step['prompt_tokens'], 'completion_tokens': step['completion_tokens']})
                _check(events[cursor - 1]['usage_valid'] is True
                       and type(events[cursor - 1]['prompt_tokens']) is int
                       and type(events[cursor - 1]['completion_tokens']) is int, 'response_usage')
                _check(new_count < len(data['responses']), 'missing_response')
                response = data['responses'][new_count]
                _check(_same(response, {'task_id': task.task_id, 'seed': 17 + index,
                    'backend_matches': True, 'backend_sha256': _fp(backend),
                    'prompt_tokens': step['prompt_tokens'], 'completion_tokens': step['completion_tokens']})
                    and response.get('backend_matches') is True
                    and type(response.get('seed')) is int
                    and all(type(response.get(key)) is int for key in ('prompt_tokens', 'completion_tokens')),
                    'response_backend_or_usage')
                requests.append({**call, 'step_index': index, 'teacher_verified': True,
                                 'student_verified': True, 'backend_sha256': _fp(backend)})
                full_history = _full_history(backend, step['prompt_tokens'])
                nonempty = bool(step['action'].strip())
                no_error = not step['observation'].startswith(('Execution failed', 'Generation failed',
                                                              'Infrastructure error'))
                raw_match = step['raw_response'] == step['action']
                target_sha = _fp(step['action'])
                targets.append({'task_id': task.task_id, 'episode_sha256': _fp(episode), 'step_index': index,
                    'target_sha256': target_sha, 'sample_sha256': _fp({
                        'student_messages_sha256': call['student_messages_sha256'], 'target_sha256': target_sha}),
                    'student_messages_sha256': call['student_messages_sha256'], 'messages_reconstructed': True,
                    'full_history': full_history, 'nonempty': nonempty, 'no_error': no_error,
                    'raw_action_match': raw_match,
                    'eligible': episode['success'] and full_history and nonempty and no_error and raw_match})
                new_count += 1
            action_event = 'replay' if replayed else 'execute'
            take(action_event + '_attempt', {'index': index, 'action': step['action'],
                                              'action_sha256': _fp(step['action'])})
            _check(cursor < len(events), 'missing_execution_return')
            elapsed = events[cursor].get('elapsed_seconds')
            _check(_seconds(elapsed) and (replayed or step['elapsed_seconds'] >= elapsed), 'execute_time')
            take(action_event + '_returned', {'index': index, 'elapsed_seconds': elapsed,
                'observation': step['observation'], 'observation_sha256': _fp(step['observation'])})
            student.extend([{'role': 'assistant', 'content': step['raw_response'] or step['action']},
                            {'role': 'user', 'content': step['observation']}])
            if step['observation'] == 'Execution successful.\n' + FEEDBACK:
                try:
                    printable_documentation_call(step['action'])
                except ValueError:
                    pass
                else:
                    expected_feedback.append({'task_id': task.task_id, 'index': index, 'replayed': replayed,
                        'feedback_events': len(expected_feedback) + 1, 'action_sha256': _fp(step['action']),
                        'augmented_observation_sha256': _fp(step['observation'])})
        _check(cursor == len(events) and new_count == len(data['responses']), 'extra_or_missing_events')
        _check(_same(data['feedback-events'], expected_feedback), 'feedback_journal')
        if kind == 'prefix_check':
            completed = episode['termination'] == 'completed'
            _check(record.get('prefix_success') is episode['success']
                   and record.get('prefix_completed') is completed and record.get('success') is None
                   and record['status'] == ('already_terminal' if completed or episode['success'] else 'eligible'),
                   'prefix_status')
        else:
            _check(record['status'] == 'complete' and record.get('success') is episode['success'], 'outcome')
        if kind == 'source':
            classification, found = _prefix(episode)
        else:
            found = prefix
        _check(type(record.get('prefix_length')) is int and record['prefix_length'] == len(found)
               and record.get('prefix_sha256') == (_fp(found) if found else None), 'prefix_summary')
        if kind == 'source':
            _check(record.get('source_classification') == classification, 'source_classification')
        result = {'source_classification': classification, 'no_world_created': False,
            'requests': requests, 'targets': targets, 'episode_sha256': _fp(episode)}
        for key in ('startup_seconds', 'cleanup_seconds', 'rollout_seconds', 'wall_seconds'):
            _check(_seconds(record.get(key)), 'time_cost')
        _check(record['rollout_seconds'] <= manifest['deadline_seconds']
               and episode['elapsed_seconds'] <= record['rollout_seconds']
               and sum(record[key] for key in ('startup_seconds', 'cleanup_seconds', 'rollout_seconds'))
               <= record['wall_seconds'], 'time_scope')
    costs = _costs(data, False)
    generation = record.get('generation', {})
    _check(_same(generation, {'generation_calls': costs['generation']['attempted'],
        'generation_calls_without_response': 0, 'generation_calls_with_unusable_usage': 0,
        'returned_response_cost': {key: costs['generation'][key] for key in ('prompt_tokens', 'completion_tokens')},
        'unknown_additional_cost': False}), 'generation_cost')
    for key, name in (('replay', 'replay'), ('new_actions', 'new_execution')):
        expected = costs[name]
        if key == 'replay':
            expected = {'expected': len(prefix), **expected}
        _check(_same(record.get(key), expected) and record[key].get('unknown_additional_cost') is False,
               'execution_cost')
    _check(_seconds(record.get('wall_seconds')), 'wall_cost')
    return {**shared, **result, 'status': record['status'], 'success': record['success'],
        'environment_seed': record['environment_seed'], 'local_cleanup_complete': record['local_cleanup_complete'],
        'prefix_length': record['prefix_length'], 'prefix_sha256': record['prefix_sha256'],
        'prefix_proof_sha256': proof_sha, 'proof_sha256': _fp(record) if kind == 'prefix_check' else None,
        'prefix_success': record.get('prefix_success'), 'prefix_completed': record.get('prefix_completed'),
        'client_identity': identity, 'costs': costs}


def audit_attempt(directory, *, task, kind, model_id, expected_backend, prepared_manifest_sha256,
                  source_episode=None, source_sha256=None, prefix_proof=None, prefix_proof_sha256=None):
    """Return a fail-closed local-record audit; never run a world/model or write files.

    Pass externally admitted ``Episode``/prefix-record objects and their protocol
    fingerprints. Their full prior journals must also be audited by the parent.
    Each response must commit to the *full* current backend; only the source and
    proof cross-job comparison drops the three explicitly enumerated fields.
    Known failures remain valid observations. Infrastructure unknowns, partial
    journals, pending execution and incomplete local cleanup are rejected while
    preserving readable generation/execute cost lower bounds. Targets are only
    local candidates; no paired-help, future training or release claim is made.
    """
    data, hashes, errors = _load(Path(directory))
    report = {'audit_valid': False, 'known_outcome': False, 'status': 'unknown', 'success': None,
        'kind': kind, 'task_id': getattr(task, 'task_id', None), 'model_id': model_id,
        'prepared_manifest_sha256': prepared_manifest_sha256, 'source_sha256': source_sha256,
        'prefix_proof_sha256': prefix_proof_sha256, 'local_cleanup_complete': False,
        'no_world_created': None, 'requests': [], 'targets': [], 'episode_sha256': None,
        'total_resources_released': None, 'training_ready': False,
        'scope': 'local record consistency only; synthetic CPU verification is not an actor result',
        'parent_must_verify_source_and_proof_admission': True,
        'external_release_verified': False, 'serving_evidence_scope':
            'full_history requires source-bound no-truncation serving contract plus returned usage; '
            'no per-request token IDs or independently authenticated server receipt are recorded',
        'target_hash_definition': 'protocol.fingerprint(action text); sample hash binds student request hash and target hash',
        'design_sha256': DESIGN_SHA256, 'runtime_sha256': RUNTIME_SHA256,
        'auditor_sha256': _digest(Path(__file__)), 'input_file_sha256': hashes}
    records = data.get('attempts', [])
    if len(records) > 1:
        report['recorded_status'] = records[-1].get('status')
        report['local_cleanup_complete'] = records[-1].get('local_cleanup_complete') is True
    if not errors:
        try:
            report.update(_audit(data, task=task, kind=kind, model_id=model_id, backend=expected_backend,
                metadata_sha=prepared_manifest_sha256, source=source_episode, source_sha=source_sha256,
                proof=prefix_proof, proof_sha=prefix_proof_sha256))
        except _Invalid as exc:
            errors.append(str(exc))
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, RecursionError):
            errors.append('invalid_schema_or_binding')
    if errors:
        report['costs'] = _costs(data, True)
    else:
        report.update(audit_valid=True, known_outcome=True)
    report['errors'] = errors
    report['audit_sha256'] = _fp(report)
    return report
