"""Read-only reconstruction of one current utility branch's private evidence.

Only identity, counts, booleans and hashes leave this module. Public events must
be authenticated by the caller; neither files nor recorded owner attestations
reproduce scoring, tokenization, server billing or fresh physical release.
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
from types import SimpleNamespace

from scripts import recovery_evidence_utility as utility
from scripts import run_recovery_evidence_utility_branch as branch
from wll.prompts import initial_messages
from wll.protocol import fingerprint
from wll.runner import extract_code

MAX_FILE_BYTES = 64 * 1024**2
MAX_TRACE_EVENTS = 4096
_ERROR = 'coverage_contract_branch_audit_invalid'
_RECORD_KEYS = {'task_id', 'scenario_id', 'arm', 'outcome', 'valid_for_continuation',
                'released', 'protocol_sha256'}
_COUNTS = ('generation_attempts', 'generation_returns', 'generation_returns_with_unusable_usage',
           'context_count_attempts', 'context_count_returns', 'context_rejections',
           'execution_attempts', 'execution_returns', 'evaluation_attempts', 'evaluation_returns',
           'generation_calls_without_response')
_FLAGS = ('unknown_additional_response_cost', 'unknown_additional_context_count_cost',
          'unknown_additional_cost')
_TRACE_FIELDS = {
    **{kind + '_attempt': set() for kind in ('completed', 'api_count')},
    **{kind + '_returned': {'result'} for kind in ('completed', 'api_count', 'execution')},
    **{kind + '_failed': {'error_type'} for kind in ('completed', 'api_count', 'execution', 'generation')},
    'context_count_attempt': {'messages'}, 'context_count_returned': {'tokens'},
    'context_budget_rejected': {'tokens'}, 'generation_attempt': {'messages', 'seed'},
    'generation_returned': {'response'}, 'execution_attempt': {'code'},
    'step_recorded': {'step_index', 'step', 'replayed'},
}
_PUBLIC_FIELDS = {
    'world_startup_attempt': set(), 'container_intent': {'intent'},
    'owner_api_count_attempt': set(), 'owner_api_count_returned': {'value'},
    'world_ready': {'container_id', 'environment_seed'}, 'world_operation_attempt': set(),
    'evaluation_attempt': set(), 'evaluation_returned': {'valid_boolean'},
    'world_operation_returned': set(), 'world_failed': {'error_type'},
    'world_cleanup_attempt': set(), 'world_cleanup_returned': {'verified'},
    'response_checked': {'seed', 'backend_matches', 'backend_sha256', 'prompt_tokens', 'completion_tokens'},
}
_STEP_FIELDS = {'action', 'observation', 'prompt_tokens', 'completion_tokens',
                'elapsed_seconds', 'raw_response'}
_SCOPE = (
    'Local immutable-file and prompt/response/step reconstruction only. Authenticated public events '
    'are required for continuation. Evaluation booleans and cleanup are recorded owner attestations, '
    'not independent scoring or fresh physical process/container verification. Token counts are '
    'reconciled as recorded, not retokenized. Unknown-prefix report-only costs remain attestations; '
    'visible usage is preserved without adding overlapping step/trace costs. Server billing, '
    'physical release, wall-clock enforcement, content certification, utility acceptance and '
    'training remain external.'
)


def _require(condition):
    if not condition:
        raise ValueError(_ERROR)


def _count(value):
    return type(value) is int and value >= 0


def _time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _error_type(value):
    return value is None or (type(value) is str and re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,127}', value))


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(_same(value[key], item) for key, item in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected, strict=True))
    return value == expected


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory(path):
    raw = os.fspath(path)
    parsed = Path(raw)
    _require(type(raw) is str and parsed.is_absolute() and str(parsed) == raw
             and '..' not in parsed.parts and '\0' not in raw)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open('/', flags)
    try:
        for part in parsed.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        _require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _snapshot(path):
    """Anchor all reads, reject links, bound bytes and recheck every open inode."""
    directory, opened = _directory(path), []
    try:
        before = _identity(os.fstat(directory))
        names = set(os.listdir(directory))
        _require('report.json' in names and names <= {'actor.jsonl', 'episode.json', 'report.json'})
        data = {}
        for name in sorted(names):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
            opened.append((name, fd, _identity(os.fstat(fd))))
            info = os.fstat(fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1
                and info.st_size <= MAX_FILE_BYTES)
            chunks, total = [], 0
            while True:
                chunk = os.read(fd, min(65536, MAX_FILE_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                _require(total <= MAX_FILE_BYTES)
            data[name] = b''.join(chunks)
        for name, fd, identity in opened:
            _require(_identity(os.fstat(fd)) == identity
                and _identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) == identity)
        _require(_identity(os.fstat(directory)) == before)
        current = _directory(path)
        try:
            _require(_identity(os.fstat(current)) == before)
        finally:
            os.close(current)
        return data
    finally:
        for _, fd, _ in opened:
            os.close(fd)
        os.close(directory)


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value)
            value[key] = item
        return value
    def constant(_value):
        raise ValueError(_ERROR)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _projection(row):
    value = {'event': row['event']}
    for key in ('seed', 'tokens', 'error_type', 'step_index', 'replayed'):
        if key in row:
            value[key] = row[key]
    for key in ('messages', 'response', 'code', 'result', 'step'):
        if key in row:
            value[key + '_sha256'] = fingerprint(row[key])
    return value


class _Incomplete(Exception):
    pass


class _Stopped(Exception):
    pass


def _actor(rows, task, experience, protocol, arm, backend):
    _require(len(rows) <= MAX_TRACE_EVENTS)
    projected, usage, unusable = [], {'prompt_tokens': 0, 'completion_tokens': 0}, 0
    for index, row in enumerate(rows):
        _require(type(row) is dict and type(row.get('index')) is int and row['index'] == index
            and row.get('event') in _TRACE_FIELDS
            and set(row) == {'index', 'event'} | _TRACE_FIELDS[row['event']])
        if 'error_type' in row:
            _require(row['error_type'] is not None and _error_type(row['error_type']))
        projected.append(_projection(row))
        if row['event'] == 'generation_returned':
            response = row['response']
            valid = type(response) is dict
            for key in usage:
                value = response.get(key) if type(response) is dict else None
                if _count(value):
                    usage[key] += value
                else:
                    valid = False
            unusable += int(not valid)
            if valid:
                projected.append({'event': 'response_checked', 'seed': 17 + sum(
                    old['event'] == 'generation_attempt' for old in rows[:index]) - 1,
                    'backend_matches': response.get('backend') == backend,
                    'backend_sha256': fingerprint(response.get('backend')),
                    **{key: response[key] for key in usage}})
    cursor = 0
    state = {'complete': False, 'steps': [], 'done': False, 'termination': None,
             'backend_seen': False, 'projected': projected, 'usage': usage, 'unusable': unusable}
    messages = initial_messages(task, protocol['prompt_version'])
    if experience is not None:
        messages.append({'role': 'user', 'content':
            'A previous training experience may apply; check its conditions:\n' + experience.render()})

    def take(kind):
        nonlocal cursor
        if cursor == len(rows):
            raise _Incomplete()
        row = rows[cursor]
        _require(row['event'] == kind)
        cursor += 1
        return row

    def rpc(kind, **fields):
        take_row = take(kind + '_attempt')
        _require({key: value for key, value in take_row.items() if key not in {'index', 'event'}} == fields)
        if cursor < len(rows) and rows[cursor]['event'] == kind + '_failed':
            take(kind + '_failed')
            raise _Stopped()
        return take(kind + '_returned')['result']

    def completed():
        value = rpc('completed')
        _require(type(value) is bool)
        state['done'] = value
        return value

    def step(response, action, observation):
        row = take('step_recorded')
        value = row['step']
        _require(type(row['step_index']) is int and row['step_index'] == len(state['steps'])
            and row['replayed'] is False and type(value) is dict and set(value) == _STEP_FIELDS
            and _time(value['elapsed_seconds'])
            and _same(value, {'action': action, 'observation': observation,
                'prompt_tokens': response['prompt_tokens'], 'completion_tokens': response['completion_tokens'],
                'elapsed_seconds': value['elapsed_seconds'], 'raw_response': response['text']}))
        state['steps'].append(value)

    try:
        if completed():
            raise _Stopped()  # Runtime rejects an already-complete initial world.
        done, last_api = completed(), 0
        for index in range(40):
            if done:
                break
            _require(take('context_count_attempt')['messages'] == messages)
            tokens = take('context_count_returned')['tokens']
            _require(type(tokens) is int and tokens > 0)
            if index == 0 and tokens != protocol['initial_input_tokens'][task.task_id][arm]:
                raise _Stopped()
            if tokens > 15360:
                _require(take('context_budget_rejected')['tokens'] == tokens)
                state['termination'] = 'context_budget'
                break
            request = take('generation_attempt')
            _require(type(request['seed']) is int and request['seed'] == 17 + index
                and request['messages'] == messages)
            if cursor < len(rows) and rows[cursor]['event'] == 'generation_failed':
                take('generation_failed')
                raise _Stopped()
            response = take('generation_returned')['response']
            if (type(response) is not dict or any(not _count(response.get(key)) for key in usage)
                    or response.get('backend') != backend or type(response.get('text')) is not str
                    or response['prompt_tokens'] != tokens or response['prompt_tokens'] > 15360
                    or response['completion_tokens'] > 1024
                    or (response.get('format_error') is not None and type(response['format_error']) is not str)):
                raise _Stopped()
            state['backend_seen'] = True
            if response.get('format_error'):
                step(response, '', 'Generation failed: ' + response['format_error'])
                state['termination'] = 'generation_format_error'
                break
            action = extract_code(response['text'])
            observation = rpc('execution', code=action)
            _require(type(observation) is str)
            count = rpc('api_count')
            _require(_count(count) and count >= last_api)
            last_api = count
            step(response, action, observation)
            messages.extend([{'role': 'assistant', 'content': response['text']},
                             {'role': 'user', 'content': observation}])
            done = completed()
        state['termination'] = 'completed' if done else (state['termination'] or 'step_budget')
        state['complete'] = True
    except (_Incomplete, _Stopped):
        pass
    _require(cursor == len(rows))
    return state


def _episode(value, actor, task, experience, protocol, arm, backend):
    keys = {'task', 'model_id', 'seed', 'steps', 'success', 'termination', 'experience_id',
            'branch', 'replayed_steps', 'elapsed_seconds', 'metadata'}
    metadata = {key: copy.deepcopy(protocol[key]) for key in utility.EPISODE_SETTINGS}
    metadata.update(arm=arm, world_completed=actor['done'])
    if actor['backend_seen']:
        metadata['backend'] = backend
    _require(actor['complete'] and type(value) is dict and set(value) == keys
        and value['task'] == asdict(task) and value['model_id'] == utility.MODEL_ID
        and type(value['seed']) is int and value['seed'] == 17
        and _same(value['steps'], actor['steps']) and type(value['success']) is bool
        and value['termination'] == actor['termination']
        and value['experience_id'] == (experience.experience_id if experience is not None else None)
        and value['branch'] == 'first' and type(value['replayed_steps']) is int
        and value['replayed_steps'] == 0 and _same(value['metadata'], metadata)
        and _time(value['elapsed_seconds'])
        and sum(step['elapsed_seconds'] for step in actor['steps']) <= value['elapsed_seconds'] + 1e-6)


def _owner(value, protocol):
    if value == {}:
        return
    _require(type(value) is dict and set(value) == {'status', 'safe_to_continue', 'creation_verified',
        'cleanup_verified', 'journal_failed', 'unknown_additional_cost', 'container_intents', 'container_id',
        'creator_exit_code', 'owned_containers_absent', 'api_attempts_observed', 'phase_seconds',
        'error_type', 'cleanup_error_type', 'wall_seconds'}
        and value['status'] in {'complete', 'unknown'}
        and all(type(value[key]) is bool for key in ('safe_to_continue', 'creation_verified',
            'cleanup_verified', 'journal_failed', 'unknown_additional_cost'))
        and type(value['container_intents']) is list and len(value['container_intents']) <= 1
        and (value['container_id'] is None or utility._sha(value['container_id']))
        and (value['creator_exit_code'] is None or type(value['creator_exit_code']) is int)
        and type(value['owned_containers_absent']) is list
        and len(value['owned_containers_absent']) <= 1
        and all(type(item) is bool for item in value['owned_containers_absent'])
        and (value['api_attempts_observed'] is None or _count(value['api_attempts_observed']))
        and _error_type(value['error_type']) and _error_type(value['cleanup_error_type'])
        and type(value['phase_seconds']) is dict and set(value['phase_seconds']) <= {'startup', 'operation', 'cleanup'}
        and all(_time(item) for item in value['phase_seconds'].values()) and _time(value['wall_seconds']))
    for intent in value['container_intents']:
        _require(type(intent) is dict and set(intent) == {'container_name', 'image', 'owner_pid',
            'owner_start_ticks', 'ownership_token'}
            and type(intent['ownership_token']) is str and re.fullmatch('[0-9a-f]{32}', intent['ownership_token'])
            and intent['container_name'] == 'wll-' + intent['ownership_token'][:16]
            and intent['image'] == protocol['environment_image']
            and type(intent['owner_pid']) is int and intent['owner_pid'] > 0
            and type(intent['owner_start_ticks']) is int and intent['owner_start_ticks'] > 0)
    if value['creation_verified']:
        _require(len(value['container_intents']) == 1 and utility._sha(value['container_id']))
    if value['cleanup_verified']:
        _require(value['creation_verified'] and type(value['creator_exit_code']) is int
            and value['owned_containers_absent'] == [True])
    if value['safe_to_continue'] or value['status'] == 'complete':
        _require(value['safe_to_continue'] and value['status'] == 'complete'
            and value['creation_verified'] and value['cleanup_verified'] and not value['journal_failed']
            and not value['unknown_additional_cost'] and value['error_type'] is None
            and value['cleanup_error_type'] is None
            and set(value['phase_seconds']) == {'startup', 'operation', 'cleanup'}
            and sum(value['phase_seconds'].values()) <= value['wall_seconds'] + 1e-6)


def _cost(report, rows, actor, episode):
    value, owner = report['cost'], report['owner']
    _require(type(value) is dict and set(value) == set(_COUNTS) | set(_FLAGS) | {
        'generation_usage_observed', 'api_attempts_observed'}
        and all(_count(value[key]) for key in _COUNTS)
        and all(type(value[key]) is bool for key in _FLAGS)
        and type(value['generation_usage_observed']) is dict
        and set(value['generation_usage_observed']) == {'prompt_tokens', 'completion_tokens'}
        and all(_count(item) for item in value['generation_usage_observed'].values())
        and (value['api_attempts_observed'] is None or _count(value['api_attempts_observed'])))
    counts = {event: sum(row['event'] == event for row in rows) for event in _TRACE_FIELDS}
    for kind in ('generation', 'context_count', 'execution'):
        _require(value[kind + '_returns'] <= value[kind + '_attempts'] <= 40
            and value[kind + '_attempts'] <= counts[kind + '_attempt']
            and value[kind + '_returns'] >= counts[kind + '_returned'])
    _require(value['generation_attempts'] >= counts['generation_returned'] + counts['generation_failed']
        and value['generation_returns_with_unusable_usage'] >= actor['unusable']
        and value['generation_returns_with_unusable_usage'] <= value['generation_returns']
        and value['context_rejections'] == counts['context_budget_rejected'] <= 1
        and value['evaluation_returns'] <= value['evaluation_attempts'] <= 1
        and value['generation_calls_without_response'] == value['generation_attempts'] - value['generation_returns']
        and value['unknown_additional_response_cost'] is (value['generation_attempts'] != value['generation_returns']
            or value['generation_returns_with_unusable_usage'] > 0)
        and value['unknown_additional_context_count_cost'] is (
            value['context_count_attempts'] != value['context_count_returns'])
        and value['unknown_additional_cost'] is (owner.get('unknown_additional_cost', True)
            or value['unknown_additional_response_cost'] or value['unknown_additional_context_count_cost']))
    _require(all(value['generation_usage_observed'][key] >= item for key, item in actor['usage'].items()))
    observed = [row['result'] for row in rows if row['event'] == 'api_count_returned']
    _require(all(_count(item) for item in observed) and observed == sorted(observed))
    if owner.get('api_attempts_observed') is not None:
        _require(value['api_attempts_observed'] == owner['api_attempts_observed'])
    if observed:
        _require(value['api_attempts_observed'] is not None and value['api_attempts_observed'] >= max(observed))
    if not report['journal_failed']:
        _require(value['generation_attempts'] == counts['generation_attempt']
            and value['generation_returns'] == counts['generation_returned']
            and value['generation_usage_observed'] == actor['usage']
            and value['generation_returns_with_unusable_usage'] == actor['unusable'])
    known = (owner.get('safe_to_continue') is True and not report['journal_failed'] and episode is not None
        and not value['unknown_additional_cost'] and value['execution_attempts'] == value['execution_returns']
        and value['evaluation_attempts'] == value['evaluation_returns'] == 1)
    _require((report['status'] == 'complete') is known)
    if known:
        _require(actor['complete'] and all(value[kind + '_attempts'] == counts[kind + '_attempt']
            and value[kind + '_returns'] == counts[kind + '_returned']
            for kind in ('generation', 'context_count', 'execution')))
    return known


def _public(events, report, actor, known):
    _require(type(events) is list and len(events) <= 2 * MAX_TRACE_EVENTS)
    normalized, last_index = [], None
    indexed = bool(events and 'index' in events[0])
    for raw in events:
        _require(type(raw) is dict and ('index' in raw) is indexed)
        row = dict(raw)
        if indexed:
            index = row.pop('index')
            _require(_count(index) and (last_index is None or index > last_index))
            last_index = index
        kind = row.get('event')
        _require(kind in _TRACE_FIELDS or kind in _PUBLIC_FIELDS)
        if kind in _PUBLIC_FIELDS:
            _require(set(row) == {'event'} | _PUBLIC_FIELDS[kind])
        normalized.append(row)
    private = [row for row in normalized if row['event'] in _TRACE_FIELDS or row['event'] == 'response_checked']
    _require(_same(private, actor['projected'][:len(private)]))
    if not report['journal_failed']:
        _require(_same(private, actor['projected']))
    owner, cost = report['owner'], report['cost']
    def selected(kind):
        return [row for row in normalized if row['event'] == kind]
    returned = selected('owner_api_count_returned')
    values = [row['value'] for row in returned]
    _require(all(_count(value) for value in values) and values == sorted(values)
        and all(type(row['valid_boolean']) is bool for row in selected('evaluation_returned'))
        and all(type(row['verified']) is bool for row in selected('world_cleanup_returned'))
        and all(_error_type(row['error_type']) for row in selected('world_failed')))
    if values:
        _require(cost['api_attempts_observed'] is not None and cost['api_attempts_observed'] >= max(values))
    if not report['journal_failed']:
        _require(len(selected('evaluation_attempt')) == cost['evaluation_attempts']
            and len(selected('evaluation_returned')) == cost['evaluation_returns'])
    if known:
        intent = owner['container_intents'][0]
        expected = [{'event': 'world_startup_attempt'}, {'event': 'container_intent', 'intent': intent},
            {'event': 'owner_api_count_attempt'}, {'event': 'owner_api_count_returned', 'value': 0},
            {'event': 'world_ready', 'container_id': owner['container_id'],
             'environment_seed': {'requested': 100, 'effective': 100, 'verified': True}},
            {'event': 'world_operation_attempt'}, *actor['projected'],
            {'event': 'evaluation_attempt'}, {'event': 'owner_api_count_attempt'},
            {'event': 'owner_api_count_returned', 'value': owner['api_attempts_observed']},
            {'event': 'evaluation_returned', 'valid_boolean': True},
            {'event': 'world_operation_returned'}, {'event': 'world_cleanup_attempt'},
            {'event': 'world_cleanup_returned', 'verified': True}]
        _require(_same(normalized, expected) and cost['api_attempts_observed'] == max(values))
    elif owner and not report['journal_failed']:
        _require(selected('container_intent') == [{'event': 'container_intent', 'intent': value}
            for value in owner['container_intents']]
            and selected('world_cleanup_returned') == [{'event': 'world_cleanup_returned',
                'verified': owner['cleanup_verified']}])


def audit_branch(directory, task, experience, protocol, *, arm, expected_backend,
                 expected_public_events=None):
    """Audit recorded evidence without returning actor material or authorizing release.

    ``expected_public_events`` is an authenticated ordered list of dictionaries
    shaped ``{'event': name, **fields}``, optionally each with an increasing
    outer ``index``. No event sequence means no accepted lifecycle or outcome.
    """
    try:
        branch.verify_sources()
        branch._admit(task, experience, SimpleNamespace(model_id=utility.MODEL_ID,
            max_tokens=1024, temperature=.7, timeout=180), arm, protocol, expected_backend)
        raw = _snapshot(directory)
        files = {name: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
                 for name, value in raw.items()}
        report = _json(raw['report.json'])
        _require(type(report) is dict and set(report) == {'status', 'record', 'cost', 'journal_failed',
            'private_files', 'owner', 'error_type', 'automatic_retry', 'accepted_experiences',
            'training_authorized', 'cost_note'} and report['status'] in {'complete', 'unknown'}
            and type(report['journal_failed']) is bool and _error_type(report['error_type'])
            and report['automatic_retry'] is False and type(report['accepted_experiences']) is int
            and report['accepted_experiences'] == 0 and report['training_authorized'] is False
            and type(report['private_files']) is dict
            and set(report['private_files']) <= set(raw) - {'report.json'}
            and all(_same(value, files[name]) for name, value in report['private_files'].items()))
        _owner(report['owner'], protocol)
        _require(not report['owner'].get('journal_failed', False) or report['journal_failed'])
        actor_raw = raw.get('actor.jsonl', b'')
        _require(not actor_raw or actor_raw.endswith(b'\n'))
        rows = [_json(line) for line in actor_raw.splitlines()]
        actor = _actor(rows, task, experience, protocol, arm, expected_backend)
        episode = _json(raw['episode.json']) if 'episode.json' in raw else None
        if episode is not None:
            _episode(episode, actor, task, experience, protocol, arm, expected_backend)
        known = _cost(report, rows, actor, episode)
        record = report['record']
        _require(type(record) is dict and set(record) == _RECORD_KEYS
            and record['task_id'] == task.task_id and record['scenario_id'] == task.scenario_id
            and record['arm'] == arm and record['protocol_sha256'] == fingerprint(protocol)
            and type(record['valid_for_continuation']) is bool and record['valid_for_continuation'] is known
            and type(record['released']) is bool and record['released'] is (
                report['owner'].get('cleanup_verified') is True)
            and (record['outcome'] is None if not known else
                type(record['outcome']) is bool and record['outcome'] is episode['success']))
        if known:
            _require(set(report['private_files']) == set(raw) - {'report.json'})
        if expected_public_events is not None:
            _public(expected_public_events, report, actor, known)
        accepted_lifecycle = known and expected_public_events is not None
        result_record = dict(record)
        if not accepted_lifecycle:
            result_record.update(outcome=None, valid_for_continuation=False)
        if expected_public_events is None:
            result_record['released'] = False
        return {'audit': 'passed', 'status': 'complete' if accepted_lifecycle else 'unknown',
            'actor_reconstruction_passed': actor['complete'] and episode is not None,
            'record': result_record, 'cost': copy.deepcopy(report['cost']), 'private_files': files,
            'reason': 'public_evidence_missing' if expected_public_events is None else
                None if accepted_lifecycle else 'branch_unknown', 'scope': _SCOPE}
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError, UnicodeError,
            RecursionError, OverflowError, MemoryError):
        raise ValueError(_ERROR) from None
