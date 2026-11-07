"""One exclusive, fixed five-case completion batch; no GPU or queue ownership.

The caller binds source provenance and owns the model service. Every attempted
arm delegates world lifecycle/scoring to run_completion_arm; this layer owns
public metadata and private trace writers. Any uncertainty stops the batch.
"""

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

from scripts.completion_appworld import CompletionAppWorldEnvironment
from scripts.completion_arm import run_completion_arm
from scripts.completion_check_messages import build_check_messages
from scripts.completion_intervention import MAX_NEW_CALLS
from scripts.completion_replay_probe import validate_boundary
from scripts.completion_tool_instructions import get_tool_instructions
from scripts.completion_trace import PrivateTrace
from scripts.control_plane_diagnostics import validate_directory
from wll.protocol import Task
from wll.runner import extract_code

_RESULT_FIELDS = frozenset({
    'status', 'arm', 'task_id', 'accepted', 'safe_to_continue', 'terminal_score',
    'creation_verified', 'cleanup_verified', 'unknown_additional_cost', 'journal_failed',
    'container_intents', 'replay', 'intervention', 'api_attempts_observed', 'replay_api_attempts',
    'generation_usage_observed', 'operation_intents', 'operation_returns', 'phase_seconds',
    'container_id', 'owned_containers_absent', 'creator_exit_code', 'cleanup_error_type',
    'error_type', 'wall_seconds',
})
_EVENT_FIELDS = frozenset({
    'phase', 'arm', 'task_id', 'index', 'seed', 'messages_sha256', 'expected_backend_sha256',
    'response_sha256', 'backend_sha256', 'action_sha256', 'observation_sha256',
    'expected_sha256', 'actual_sha256', 'request_sha256', 'elapsed_seconds', 'error_type',
    'usage_valid', 'prompt_tokens', 'completion_tokens', 'api_calls', 'value', 'reason',
    'termination', 'ready_for_scoring', 'container_id', 'environment_seed', 'intent',
    'verified', 'valid_text', 'valid_boolean', 'matches_expected', 'completed', 'terminal_score',
    'client_identity', 'planned_arms', 'metadata_sha256', 'image', 'tool_instructions_sha256',
    'result', 'original_index', 'initial_context_tokens', 'status', 'safe_to_continue',
    'unattempted',
})


def _plain(value, depth=0):
    if depth > 100:
        raise ValueError('batch_value_depth')
    kind = type(value)
    if value is None or kind in (str, int, bool):
        return value
    if kind is float and math.isfinite(value):
        return value
    if kind is list:
        return [_plain(item, depth + 1) for item in value]
    if kind is dict and all(type(key) is str for key in value):
        return {key: _plain(item, depth + 1) for key, item in value.items()}
    raise ValueError('batch_non_plain_value')


def _bytes(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':'), sort_keys=True) + '\n').encode('ascii')


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _write(fd, raw):
    remaining = memoryview(raw)
    while remaining:
        count = os.write(fd, remaining)
        if count <= 0:
            raise OSError('batch_write_no_progress')
        remaining = remaining[count:]
    os.fsync(fd)


class _Events:
    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
        self.index = 0
        self.failed = False

    def emit(self, kind, **fields):
        try:
            if (self.failed or not re.fullmatch('[a-z][a-z0-9_]{0,127}', kind)
                    or set(fields) - _EVENT_FIELDS):
                raise ValueError('invalid_batch_event')
            row = _plain(fields)
            row.update(event_index=self.index, event=kind, epoch=time.time())
            _write(self.fd, _bytes(row))
            self.index += 1
        except BaseException:
            self.failed = True
            raise

    def close(self):
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)


def _canonical(path):
    raw = os.fspath(path)
    parsed = Path(raw)
    if not parsed.is_absolute() or str(parsed) != raw or '..' in parsed.parts or '\0' in raw:
        raise ValueError('canonical_absolute_batch_path_required')
    return parsed


def _preflight(cases, tasks, metadata, expected_backend, image, before_arm, world_factory):
    if (type(cases) is not list or len(cases) != 5 or type(tasks) is not dict
            or type(metadata) is not dict or type(expected_backend) is not dict or not expected_backend
            or type(image) is not str or re.fullmatch('sha256:[a-f0-9]{64}', image) is None
            or (before_arm is not None and not callable(before_arm))
            or (world_factory is not None and not callable(world_factory))):
        raise ValueError('invalid_batch_input')
    cases, metadata, backend = _plain(cases), _plain(metadata), _plain(expected_backend)
    selected = {}
    for case in cases:
        if (type(case) is not dict or set(case) != {
                'task_id', 'original_index', 'steps', 'boundary_index', 'initial_context_tokens'}
                or type(case['task_id']) is not str
                or re.fullmatch('[A-Za-z0-9_-]{1,128}', case['task_id']) is None
                or type(case['original_index']) is not int or not 0 <= case['original_index'] < 11
                or type(case['initial_context_tokens']) is not dict
                or set(case['initial_context_tokens']) != {'A', 'B'}
                or any(type(n) is not int or n < 0 for n in case['initial_context_tokens'].values())):
            raise ValueError('invalid_batch_case')
        task = tasks.get(case['task_id'])
        if (type(task) is not Task or task.task_id != case['task_id'] or task.split != 'train'
                or any(type(value) is not str for value in asdict(task).values())):
            raise ValueError('invalid_batch_task')
        selected[task.task_id] = Task(**asdict(task))
        steps, boundary = case['steps'], case['boundary_index']
        if (type(steps) is not list or any(type(step) is not dict
                or set(step) not in ({'action', 'observation'}, {'action', 'observation', 'raw_response'})
                or any(type(value) is not str for value in step.values()) for step in steps)):
            raise ValueError('invalid_batch_actor_fields')
        validate_boundary(steps, boundary)
        if (boundary != len(steps) - 1 or boundary + MAX_NEW_CALLS > 40
                or steps[boundary]['observation'] != ''
                or extract_code(steps[boundary].get('raw_response', steps[boundary]['action'])) != steps[boundary]['action']):
            raise ValueError('invalid_batch_boundary')
        prefix = steps[:boundary]
        for arm in ('A', 'B'):
            build_check_messages(task, prefix, steps[boundary].get('raw_response', steps[boundary]['action']),
                                 arm=arm, tool_instructions=get_tool_instructions())
    indices = [case['original_index'] for case in cases]
    if len(selected) != 5 or indices != sorted(set(indices)):
        raise ValueError('invalid_batch_order')
    if world_factory is None:
        for key in ('data_dir', 'deps_dir'):
            if type(metadata.get(key)) is not str or not Path(metadata[key]).is_absolute():
                raise ValueError('missing_batch_runtime_directory')
    return cases, selected, metadata, backend


def _unknown(record, exc):
    if record.get('status') != 'unknown':
        record.setdefault('observed_status', record.get('status'))
    record.update(status='unknown', accepted=False, safe_to_continue=False,
                  unknown_additional_cost=True, batch_error_type=type(exc).__name__)


def run_batch(*, cases, tasks, metadata, output, private_dir, expected_backend,
              model, image, before_arm=None, world_factory=None):
    """Run exactly five supplied cases once, in ascending original-index order.

    before_arm() is a caller-owned health check. An injected world_factory takes
    (task, *, on_container_intent); the default uses metadata data_dir/deps_dir.
    Both paths must return a new world for each call. No source provenance is
    inferred here. Public/private output directories must both be absent.
    """
    cases, tasks, metadata, backend = _preflight(
        cases, tasks, metadata, expected_backend, image, before_arm, world_factory)
    output, private_dir = _canonical(output), _canonical(private_dir)
    if output == private_dir:
        raise ValueError('batch_paths_must_differ')
    output.mkdir()
    validate_directory(private_dir.parent)
    private_dir.mkdir(mode=0o700)
    validate_directory(private_dir)
    planned = [{'task_id': case['task_id'], 'arm': arm, 'original_index': case['original_index']}
               for case in cases for arm in (('A', 'B') if case['original_index'] % 2 == 0 else ('B', 'A'))]
    report = {'status': 'stopped_unknown', 'safe_to_continue': False, 'planned_arms': planned,
              'arms': [], 'unattempted': _plain(planned), 'metadata_sha256': _sha(metadata),
              'expected_backend_sha256': _sha(backend), 'image': image,
              'tool_instructions_sha256': hashlib.sha256(get_tool_instructions().encode()).hexdigest(),
              'journal_failed': False}
    events = trace = None
    interrupt = None
    seen_worlds = []
    try:
        events = _Events(output/'events.jsonl')
        ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
        events.emit('batch_started', client_identity={'pid': os.getpid(), 'start_ticks': ticks},
                    planned_arms=planned, metadata_sha256=report['metadata_sha256'], image=image,
                    expected_backend_sha256=report['expected_backend_sha256'],
                    tool_instructions_sha256=report['tool_instructions_sha256'])
        for case in cases:
            slots = [slot for slot in planned if slot['task_id'] == case['task_id']]
            if any(value > 15360 for value in case['initial_context_tokens'].values()):
                events.emit('case_context_ineligible', task_id=case['task_id'],
                            original_index=case['original_index'], initial_context_tokens=case['initial_context_tokens'])
                for slot in slots:
                    report['arms'].append({**slot, 'status': 'context_ineligible', 'accepted': False,
                        'safe_to_continue': True, 'attempted': False, 'terminal_score': None,
                        'cleanup_not_required': True, 'initial_context_tokens': _plain(case['initial_context_tokens'])})
                continue
            for slot in slots:
                record = {**slot, 'status': 'unknown', 'accepted': False, 'safe_to_continue': False,
                    'attempted': True, 'terminal_score': None, 'trace_closed': False,
                    'generation_usage_observed': {'prompt_tokens': 0, 'completion_tokens': 0},
                    'api_attempts_observed': None, 'unknown_additional_cost': True}

                def emit(kind, *, _slot=slot, _record=record, **fields):
                    if any(key in fields and fields[key] != _slot[key] for key in ('task_id', 'arm')):
                        raise ValueError('batch_event_slot_mismatch')
                    if kind == 'generation_returned':
                        for key in ('prompt_tokens', 'completion_tokens'):
                            value = fields.get(key)
                            if type(value) is int and value >= 0:
                                _record['generation_usage_observed'][key] += value
                    if kind == 'api_count_returned' and type(fields.get('value')) is int:
                        previous = _record['api_attempts_observed']
                        if fields['value'] >= 0 and (previous is None or fields['value'] >= previous):
                            _record['api_attempts_observed'] = fields['value']
                    if kind == 'evaluation_returned' and type(fields.get('terminal_score')) is bool:
                        _record['terminal_score'] = fields['terminal_score']
                    try:
                        events.emit(kind, **{'task_id': _slot['task_id'], 'arm': _slot['arm'], **fields})
                    except BaseException:
                        _record['journal_failed'] = True
                        events.failed = True
                        raise

                try:
                    events.emit('arm_started', **slot)
                    if before_arm is not None:
                        events.emit('before_arm_attempt', **slot)
                        before_arm()
                        events.emit('before_arm_returned', **slot)
                    trace_path = private_dir/f"{slot['original_index']:02d}-{slot['task_id']}-{slot['arm']}.jsonl"
                    trace = PrivateTrace(trace_path)

                    def factory(*, on_container_intent, _task=tasks[slot['task_id']]):
                        task = _task
                        if world_factory is None:
                            world = CompletionAppWorldEnvironment(task.task_id, metadata['data_dir'], 40,
                                image=image, deps_dir=metadata['deps_dir'], environment_seed=100,
                                on_container_intent=on_container_intent)
                        else:
                            world = world_factory(task, on_container_intent=on_container_intent)
                        if any(world is old for old in seen_worlds):
                            raise ValueError('batch_world_reused')
                        seen_worlds.append(world)
                        return world

                    result = run_completion_arm(tasks[slot['task_id']], _plain(case['steps']),
                        case['boundary_index'], model, arm=slot['arm'], expected_backend=_plain(backend),
                        tool_instructions=get_tool_instructions(), image=image, world_factory=factory,
                        trace=trace, emit=emit)
                    if (type(result) is not dict or set(result) - _RESULT_FIELDS
                            or result.get('task_id') != slot['task_id'] or result.get('arm') != slot['arm']
                            or type(result.get('accepted')) is not bool
                            or type(result.get('safe_to_continue')) is not bool
                            or result.get('status') not in {'scored', 'unknown', 'observation_mismatch', 'early_completed'}
                            or (result['safe_to_continue'] and (result['status'] == 'unknown'
                                or result.get('cleanup_verified') is not True
                                or result.get('journal_failed') is not False))
                            or (result['accepted'] and (result['status'] != 'scored'
                                or type(result.get('terminal_score')) is not bool))):
                        raise ValueError('invalid_arm_result')
                    record.update(_plain(result))
                except BaseException as exc:
                    _unknown(record, exc)
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        interrupt = exc
                finally:
                    if trace is not None:
                        try:
                            trace.close()
                            record['trace_closed'] = True
                            record['private_trace_sha256'] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
                            record['private_trace_file'] = trace_path.name
                        except BaseException as exc:
                            _unknown(record, exc)
                            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                                interrupt = exc
                        trace = None
                try:
                    events.emit('arm_returned', task_id=slot['task_id'], arm=slot['arm'], result=record)
                except BaseException as exc:
                    _unknown(record, exc)
                    record['journal_failed'] = True
                    events.failed = True
                report['arms'].append(record)
                if not record['safe_to_continue']:
                    break
            if report['arms'] and not report['arms'][-1]['safe_to_continue']:
                break
        consumed = {(row['task_id'], row['arm']) for row in report['arms']}
        report['unattempted'] = [slot for slot in planned if (slot['task_id'], slot['arm']) not in consumed]
        report['safe_to_continue'] = not report['unattempted'] and all(row['safe_to_continue'] for row in report['arms'])
        report['status'] = 'complete' if report['safe_to_continue'] else 'stopped_unknown'
        events.emit('batch_finished', status=report['status'], safe_to_continue=report['safe_to_continue'],
                    unattempted=report['unattempted'])
    except BaseException as exc:
        report.update(status='stopped_unknown', safe_to_continue=False, error_type=type(exc).__name__)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupt = exc
    finally:
        consumed = {(row['task_id'], row['arm']) for row in report['arms']}
        report['unattempted'] = [slot for slot in planned if (slot['task_id'], slot['arm']) not in consumed]
        if events is not None:
            report['journal_failed'] = events.failed
            try:
                events.close()
            except BaseException as exc:
                report.update(journal_failed=True, safe_to_continue=False, status='stopped_unknown',
                              error_type=type(exc).__name__)
    try:
        fd = os.open(output/'report.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        try:
            _write(fd, _bytes(report))
        finally:
            os.close(fd)
    except BaseException as exc:
        report.update(status='stopped_unknown', safe_to_continue=False, report_write_error=type(exc).__name__)
    if interrupt is not None:
        raise interrupt
    return report


def aggregate_pair_summary(records, all11ids):
    """Metadata accounting only; provenance and external release remain caller-owned."""
    if (type(records) is not list or type(all11ids) is not list or len(all11ids) != 11
            or any(type(tid) is not str for tid in all11ids) or len(set(all11ids)) != 11):
        raise ValueError('invalid_pair_denominator')
    slots = {}
    for record in records:
        if (type(record) is not dict or record.get('task_id') not in all11ids
                or record.get('arm') not in ('A', 'B') or type(record.get('accepted')) is not bool
                or (record.get('terminal_score') is not None and type(record['terminal_score']) is not bool)):
            raise ValueError('invalid_pair_record')
        key = (record['task_id'], record['arm'])
        if key in slots:
            raise ValueError('duplicate_pair_slot')
        slots[key] = record
    result = {'denominator': 11, 'complete_pairs': 0, 'help': 0, 'harm': 0,
              'both_success': 0, 'both_failure': 0, 'context_ineligible_pairs': 0,
              'historical_mismatch_pairs': 0, 'incomplete_pairs': 0,
              'unattempted_arms': 22 - len(slots)}
    for tid in all11ids:
        pair = [slots.get((tid, arm)) for arm in ('A', 'B')]
        if all(pair) and all(row.get('status') == 'context_ineligible' and row['accepted'] is False
                             and row.get('attempted') is False for row in pair):
            result['context_ineligible_pairs'] += 1
        elif all(pair) and all(row.get('status') == 'historical_mismatch' and row['accepted'] is False
                               and row.get('attempted') is False for row in pair):
            result['historical_mismatch_pairs'] += 1
        elif all(pair) and all(row['accepted'] is True and row.get('status') == 'scored'
                and row.get('cleanup_verified') is True and row.get('trace_closed') is True
                and row.get('safe_to_continue') is True and row.get('attempted') is True
                and row.get('journal_failed') is False
                and type(row.get('terminal_score')) is bool for row in pair):
            a, b = (row['terminal_score'] for row in pair)
            result['complete_pairs'] += 1
            result['help' if not a and b else 'harm' if a and not b else 'both_success' if a else 'both_failure'] += 1
        else:
            result['incomplete_pairs'] += 1
    return result
