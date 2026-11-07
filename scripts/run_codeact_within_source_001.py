"""Bounded source batch client over the existing authenticated GPU lifecycle.

Only trusted repository code is compiled for the isolated lifecycle. Model
responses, actor histories and task descriptions stay in separate private files.
Importing this module starts no model, task, container, queue or registration.
"""

import argparse
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

from artifacts import prepare_b3_dev_metadata_001 as io
from scripts import codeact_model_client_001 as model_client
from scripts import native_appworld_model_client as wire
from scripts.codeact_model_client_001 import CodeActModel
from scripts.codeact_owned_environment_001 import CodeActOwnedEnvironment
from scripts.codeact_owned_worker_001 import IMAGE_SOURCES, MODULES
from scripts.codeact_within_episode_001 import _response_valid, _worker_valid, run_episode
from scripts.completion_batch import _Events
from scripts.completion_trace import PrivateTrace
from scripts.fresh_completion_world import run_owned_world
from scripts.owned_appworld_v3 import _validate_intent
from wll.protocol import Task

ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-codeact-within-source-001'
OUTPUT = ROOT / 'artifacts' / ID
PRIVATE = Path('/home/hetaisheng/.local/share/workflow-learning-lab/codeact-within-source-001')
ENDPOINT = 'http://127.0.0.1:18992'
MODEL_ID = 'qwen38-b3-matched-b3-001'
BATCH_PROTOCOL = 'wll-codeact-within-source-batch-v1'
_FROZEN_SHA256 = '8e66d9af9562145d8b0cb38dec87078544bda15c82db6e65d933f8a76bc394af'
_ERROR = 'codeact_source_batch_invalid'
_CORE = None

_COUNTER_CHILD = r'''import json,sys
from scripts.coverage_contract_token_endpoint import count_tokens
p=json.load(sys.stdin)
n=count_tokens(p['endpoint'],p['messages'],model_id=p['model_id'],expected_backend=p['expected_backend'])
sys.stdout.write(json.dumps({'input_tokens':n},allow_nan=False))
'''
_OWNER_FIELDS = ('status', 'safe_to_continue', 'creation_verified', 'cleanup_verified',
    'journal_failed', 'unknown_additional_cost', 'api_attempts_observed', 'container_id',
    'creator_exit_code', 'owned_containers_absent', 'phase_seconds', 'wall_seconds',
    'error_type', 'cleanup_error_type')
_OWNER_EVENTS = {'container_intent': {'intent'}, 'owner_api_count_attempt': set(),
    'owner_api_count_returned': {'value'}, 'world_startup_attempt': set(),
    'world_ready': {'container_id', 'environment_seed'}, 'world_operation_attempt': set(),
    'world_operation_returned': set(), 'world_failed': {'error_type'},
    'world_cleanup_attempt': set(), 'world_cleanup_returned': {'verified'}}
_SAFE_TERMINATIONS = frozenset({'completed', 'initial_completion', 'generation_budget', 'action_budget',
    'actor_wall', 'generation_length', 'generation_unknown_finish', 'generation_format_error', 'context_budget'})
_KNOWN_UNSCORED = frozenset({'execution_timeout', 'execution_api_budget_rejected'})
_FAULT_TERMINATIONS = frozenset({'model_failure', 'invalid_model_response', 'execution_failure',
    'invalid_worker_result', 'execution_unknown', 'completion_failure', 'record_failure',
    'controller_failure', 'clock_failure', 'scoring_failure', 'initial_world_binding_failure'})
_COST_KEYS = frozenset({'prompt_tokens', 'completion_tokens', 'api_request_dispatches', 'budget_rejections'})
_COUNT_KEYS = _COST_KEYS | {'generation_attempts', 'generation_returns', 'actions'}
_OUTCOME_FIELDS = ('counts', 'known_costs', 'termination', 'model_reported_complete',
                   'official_success', 'costs_known', 'actor_wall_seconds')


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _registration():
    return importlib.import_module('scripts.codeact_within_registration_001')


def _isolated_lifecycle(*, _queue=None, _watchdog=None):
    registration = _registration()
    queue = _queue or importlib.import_module('scripts.codeact_within_gpu_queue_001')
    watchdog = _watchdog or importlib.import_module('scripts.codeact_within_watchdog_001')
    _require(queue.ID == registration.ID == ID and queue.PORT == 18992
             and queue.GPU_SECONDS == 7200 and queue.CLIENT_SECONDS == 6000
             and queue.BATCH_COUNT == 4 and queue.TASKS_PER_BATCH == 3
             and queue.MODEL_IDS['B3'] == MODEL_ID)
    path = Path(__file__).with_name('run_verification_comparison_v2.py')
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _FROZEN_SHA256)
    source = raw.decode('utf-8')
    for old, new, count in (
        ('18984', '18992', 4),
        ('scripts.verification_watchdog_v2 import watch', 'scripts.codeact_within_watchdog_001 import watch', 1),
        ('type(index) is int and index in (0, 1)', 'type(index) is int and 0 <= index < 4', 1),
    ):
        _require(source.count(old) == count)
        source = source.replace(old, new)
    first = "        batch_report = json.loads((stage / 'run/report.json').read_text())\n"
    last = "        io.write_new(stage / 'client-report-binding.json', {\n"
    _require(source.count(first) == source.count(last) == 1)
    start, stop = source.index(first), source.index(last)
    _require(start < stop)
    source = source[:start] + '        verify_batch_report(stage, plan, index, sha)\n' + source[stop:]
    spec = importlib.util.spec_from_file_location('scripts._codeact_within_source_lifecycle', path)
    _require(spec is not None and spec.loader is not None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), 'exec', dont_inherit=True), module.__dict__)
    module.queue, module.watchdog = queue, watchdog
    module.ID, module.OUTPUT, module.PRIVATE, module.IMAGE = ID, OUTPUT, PRIVATE, registration.IMAGE
    old = 'from scripts import completion_gpu_queue as q'
    _require(module.OLD_REMOTE_SOURCE.count(old) == 1)
    module.REMOTE_SOURCE = module.OLD_REMOTE_SOURCE.replace(
        old, 'from scripts import codeact_within_gpu_queue_001 as q')
    module.verify_batch_report = verify_batch_report
    return module


def _lifecycle():
    global _CORE
    if _CORE is None:
        _CORE = _isolated_lifecycle()
    return _CORE


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(_ERROR)
        result[key] = value
    return result


def _reject_number(_value):
    raise ValueError(_ERROR)


def count_tokens_bounded(messages, *, deadline, expected_backend, tmp_dir, _clock=time.monotonic, _run=subprocess.run):
    _require(type(deadline) in (int, float) and math.isfinite(deadline)
             and type(expected_backend) is dict and bool(expected_backend)
             and type(tmp_dir) is str and Path(tmp_dir).is_absolute())
    payload = json.dumps({'endpoint': ENDPOINT + '/v1', 'model_id': MODEL_ID,
                          'messages': messages, 'expected_backend': expected_backend},
                         ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode('ascii')
    _require(len(payload) <= 2_000_000)
    remaining = deadline - _clock()
    if remaining <= 0:
        raise TimeoutError('codeact_count_deadline')
    result = _run([sys.executable, '-B', '-c', _COUNTER_CHILD], input=payload,
                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=ROOT,
                  env={'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                       'TMPDIR': tmp_dir, 'LANG': 'C.UTF-8'}, timeout=min(30.0, remaining), check=False)
    if _clock() >= deadline:
        raise TimeoutError('codeact_count_deadline')
    _require(type(result.returncode) is int and result.returncode == 0
             and type(result.stdout) is bytes and len(result.stdout) <= 1024)
    value = json.loads(result.stdout, object_pairs_hook=_pairs, parse_constant=_reject_number)
    _require(type(value) is dict and set(value) == {'input_tokens'}
             and type(value['input_tokens']) is int and value['input_tokens'] > 0)
    return value['input_tokens']


def _read(path, *, expected=None, maximum=32_000_000, public=False):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        mode = stat.S_IMODE(before.st_mode)
        _require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                 and (mode == 0o600 if not public else mode & 0o022 == 0)
                 and before.st_nlink == 1)
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    _require(len(raw) <= maximum and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))
    digest = hashlib.sha256(raw).hexdigest()
    _require(expected is None or digest == expected)
    return raw, digest


def _read_json(path, expected=None):
    raw, digest = _read(path, expected=expected)
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_reject_number), digest


def _write_private(path, value):
    raw = (json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False,
                      separators=(',', ':')) + '\n').encode('ascii')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    io.sync_directory(Path(path).parent)
    return hashlib.sha256(raw).hexdigest()


def _trace_actor(trace, event, **fields):
    if 'index' in fields:
        _require('step_index' not in fields)
        fields['step_index'] = fields.pop('index')
    trace.record(event, **fields)


def _trace_wire(trace, value):
    _require(type(value) is dict and type(value.get('event')) is str)
    fields = dict(value)
    event = fields.pop('event')
    _require('index' not in fields)
    trace.record(event, **fields)


def _health_check(plan, expected, tunnel, deadline):
    core = _lifecycle()
    if time.monotonic() >= deadline:
        raise TimeoutError('codeact_batch_deadline')
    _require(core.live(tunnel) and core.identity(tunnel['pid']) == tunnel)
    health = core.health()
    core.validate_health(health, plan, 'B3')
    _require(health.get('backend') == expected and core.live(tunnel)
             and core.identity(tunnel['pid']) == tunnel and time.monotonic() < deadline)


def _scoped_events(events, task_id):
    def emit(event, **fields):
        _require(event in _OWNER_EVENTS and set(fields) == _OWNER_EVENTS[event])
        events.emit(event, task_id=task_id, arm='first_attempt', phase='source', **fields)
    return emit


class _CheckedModel:
    def __init__(self, delegate, plan, expected, tunnel, deadline, events, task_id, progress):
        self.delegate, self.plan, self.expected, self.tunnel = delegate, plan, expected, tunnel
        self.deadline, self.events, self.task_id, self.progress = deadline, events, task_id, progress

    def generate(self, messages, seed):
        index = self.progress['generation_attempts']
        self.progress['generation_attempts'] += 1
        _health_check(self.plan, self.expected, self.tunnel, self.deadline)
        self.events.emit('generation_attempt', task_id=self.task_id, arm='first_attempt',
                         phase='source', index=index, seed=seed)
        response = self.delegate.generate(messages, seed=seed)
        _require(type(response) is dict and type(response.get('prompt_tokens')) is int
                 and type(response.get('completion_tokens')) is int
                 and response['prompt_tokens'] > 0 and response['completion_tokens'] > 0)
        self.progress['generation_returns'] += 1
        self.progress['prompt_tokens'] += response['prompt_tokens']
        self.progress['completion_tokens'] += response['completion_tokens']
        self.events.emit('generation_returned', task_id=self.task_id, arm='first_attempt',
                         phase='source', index=index, seed=seed, prompt_tokens=response['prompt_tokens'],
                         completion_tokens=response['completion_tokens'])
        _health_check(self.plan, self.expected, self.tunnel, self.deadline)
        return response


def _outcome_valid(outcome):
    if (type(outcome) is not dict or set(outcome) != set(_OUTCOME_FIELDS)
            or type(outcome['counts']) is not dict or set(outcome['counts']) != _COUNT_KEYS
            or type(outcome['known_costs']) is not dict or set(outcome['known_costs']) != _COST_KEYS
            or type(outcome['costs_known']) is not bool
            or (outcome['model_reported_complete'] is not None and type(outcome['model_reported_complete']) is not bool)
            or (outcome['official_success'] is not None and type(outcome['official_success']) is not bool)):
        return False
    counts, known = outcome['counts'], outcome['known_costs']
    if (any(type(v) is not int or v < 0 for v in known.values())
            or any(v is not None and (type(v) is not int or v < 0) for v in counts.values())
            or any(type(counts[k]) is not int for k in ('generation_attempts', 'generation_returns', 'actions'))
            or counts['generation_returns'] > counts['generation_attempts']
            or counts['actions'] > counts['generation_returns'] or counts['generation_attempts'] > 40
            or counts['actions'] > 40):
        return False
    if any(counts[k] is not None and counts[k] < known[k] for k in _COST_KEYS):
        return False
    wall = outcome['actor_wall_seconds']
    if wall is not None and (type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0):
        return False
    if outcome['costs_known'] and (wall is None or any(counts[k] != known[k] for k in _COST_KEYS)):
        return False
    return type(outcome['termination']) is str and outcome['termination'] in (
        _SAFE_TERMINATIONS | _KNOWN_UNSCORED | _FAULT_TERMINATIONS)


def _owner_projection(owner):
    _require(type(owner) is dict and all(key in owner for key in _OWNER_FIELDS))
    projected = {key: copy.deepcopy(owner[key]) for key in _OWNER_FIELDS}
    for key in ('error_type', 'cleanup_error_type'):
        value = projected[key]
        if value is not None and value not in {'TimeoutError', 'RuntimeError', 'ValueError',
                'TraceWriteError', 'RolloutTimedOut', 'OSError', 'OtherException'}:
            projected[key] = 'OtherException'
    return projected


def _owner_verified(owner, outcome, progress):
    if (type(owner) is not dict or set(owner) != set(_OWNER_FIELDS)
            or type(progress) is not dict or set(progress) != {
                'generation_attempts', 'generation_returns', 'prompt_tokens', 'completion_tokens'}
            or any(type(value) is not int or value < 0 for value in progress.values())):
        return False
    return (owner['status'] == 'complete' and owner['safe_to_continue'] is True
            and owner['creation_verified'] is owner['cleanup_verified'] is True
            and owner['journal_failed'] is owner['unknown_additional_cost'] is False
            and type(owner['container_id']) is str
            and re.fullmatch('[a-f0-9]{64}', owner['container_id']) is not None
            and type(owner['creator_exit_code']) is int
            and type(owner['owned_containers_absent']) is list
            and len(owner['owned_containers_absent']) == 1 and owner['owned_containers_absent'][0] is True
            and owner['error_type'] is owner['cleanup_error_type'] is None
            and type(owner['api_attempts_observed']) is int
            and owner['api_attempts_observed'] == outcome['counts']['api_request_dispatches']
            and outcome['costs_known'] is True
            and all(outcome['counts'][k] == progress[k] for k in progress))


def _binding(path):
    _, digest = _read(path, maximum=128_000_000)
    return {'file': Path(path).name, 'sha256': digest}


def _run_one(plan, task_id, offset, expected, tunnel, deadline, events, private):
    progress = {'generation_attempts': 0, 'generation_returns': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
    holder, owner, result = {}, None, None
    actor_path, wire_path = private / f'actor-{offset:02}.jsonl', private / f'wire-{offset:02}.jsonl'
    episode_path = private / f'episode-{offset:02}.json'
    bindings = {'actor_trace': None, 'wire_trace': None, 'episode_result': None}
    event = _scoped_events(events, task_id)
    try:
        with PrivateTrace(actor_path) as actor, PrivateTrace(wire_path) as wire:
            def world_factory(*, on_container_intent):
                _health_check(plan, expected, tunnel, deadline)
                world = CodeActOwnedEnvironment(task_id, plan['data_dir'], 40, plan['image'], plan['deps_dir'],
                    on_container_intent=on_container_intent,
                    sources={name: ROOT / (name.replace('.', '/') + '.py') for name in MODULES},
                    source_sha256={name: plan['source_sha256'][name.replace('.', '/') + '.py'] for name in MODULES},
                    image_source_sha256={name: plan['source_sha256'][name] for name in IMAGE_SOURCES})
                holder['world'] = world
                return world

            def operation(world, _event):
                description = holder['world'].description
                _require(type(description) is dict and set(description) == {'instruction', 'supervisor'}
                         and all(type(v) is str for v in description.values()))
                task = Task(task_id, task_id.rsplit('_', 1)[0], 'train',
                            description['instruction'], description['supervisor'])
                model_deadline = min(deadline, time.monotonic() + 1500)
                def counter(messages, *, deadline):
                    return count_tokens_bounded(messages, deadline=deadline, expected_backend=expected,
                                                tmp_dir=plan['tmp_dir'])
                delegate = CodeActModel(ENDPOINT, MODEL_ID, expected_backend=expected, count_tokens=counter,
                    record=lambda value: _trace_wire(wire, value), deadline=model_deadline,
                    tmp_dir=plan['tmp_dir'], timeout=180, temperature=0.7)
                model = _CheckedModel(delegate, plan, expected, tunnel, deadline, events, task_id, progress)
                return run_episode(task, world, model, record=lambda name, **fields:
                                   _trace_actor(actor, name, **fields), seed=17, experience=None)

            owner = run_owned_world(image=plan['image'], world_factory=world_factory, emit=event,
                                    operation=operation, operation_timeout=1500)
            result = owner.get('operation_result') if type(owner) is dict else None
        bindings['actor_trace'], bindings['wire_trace'] = _binding(actor_path), _binding(wire_path)
        if type(result) is dict:
            _write_private(episode_path, result)
            bindings['episode_result'] = _binding(episode_path)
        _health_check(plan, expected, tunnel, deadline)
        projected_owner = _owner_projection(owner)
        outcome = result.get('outcome') if type(result) is dict and set(result) == {'private', 'outcome'} else None
        valid = _outcome_valid(outcome) and _owner_verified(projected_owner, outcome, progress)
        if valid and outcome['termination'] in _SAFE_TERMINATIONS and type(outcome['official_success']) is bool:
            status = 'scored'
        elif valid and outcome['termination'] in _KNOWN_UNSCORED and outcome['official_success'] is None:
            status = 'known_unscored'
        else:
            status = 'unknown'
        return {'task_id': task_id, 'arm': 'first_attempt', 'phase': 'source', 'status': status,
                **(copy.deepcopy(outcome) if _outcome_valid(outcome) else dict.fromkeys(_OUTCOME_FIELDS)),
                'owner': projected_owner, 'observed_model': dict(progress), 'private_bindings': bindings}
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        for key, path in (('actor_trace', actor_path), ('wire_trace', wire_path), ('episode_result', episode_path)):
            if path.exists():
                try:
                    bindings[key] = _binding(path)
                except Exception:
                    pass
        projected_owner = None
        if type(owner) is dict:
            try:
                projected_owner = _owner_projection(owner)
            except Exception:
                pass
        return {'task_id': task_id, 'arm': 'first_attempt', 'phase': 'source', 'status': 'unknown',
                **dict.fromkeys(_OUTCOME_FIELDS), 'owner': projected_owner,
                'observed_model': dict(progress), 'private_bindings': bindings}


def run_client(sha, request_path, request_sha):
    registration = _registration()
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and type(sha) is str
             and re.fullmatch('[a-f0-9]{64}', sha) is not None
             and type(request_sha) is str and re.fullmatch('[a-f0-9]{64}', request_sha) is not None)
    paths = [OUTPUT / f'batch-{index}' / 'request.json' for index in range(4)]
    path = Path(request_path)
    _require(path in paths)
    index = paths.index(path)
    request, _ = _read_json(path, request_sha)
    _require(type(request) is dict and set(request) == {'plan_sha256', 'batch_index', 'expected_backend'}
             and request['plan_sha256'] == sha and type(request['batch_index']) is int
             and request['batch_index'] == index and type(request['expected_backend']) is dict
             and bool(request['expected_backend']))
    plan = registration.load_registration(ROOT, sha, private=True)
    registration.check_predecessor(ROOT, plan)
    _require(plan['task_ids'] == list(registration.TASK_IDS) and plan['batches'] == registration.BATCHES
             and len(plan['batches']) == 4 and all(len(batch) == 3 for batch in plan['batches'])
             and plan['private_directory'] == str(PRIVATE) and os.environ.get('TMPDIR') == plan['tmp_dir']
             and plan['client_seconds'] == 6000 and plan['model_start_seconds'] == 900
             and plan['cleanup_seconds'] == 180 and plan['gpu_seconds'] == 7200)
    tunnel_json, _ = _read_json(path.parent / 'tunnel.json')
    tunnel = tunnel_json.get('identity') if type(tunnel_json) is dict else None
    _require(type(tunnel) is dict and set(tunnel) == {'pid', 'start_ticks'}
             and all(type(v) is int and v > 0 for v in tunnel.values()))
    deadline = time.monotonic() + 6000
    _health_check(plan, request['expected_backend'], tunnel, deadline)
    stage, private = path.parent, PRIVATE / f'batch-{index}'
    io.reserve(stage / 'run')
    io.reserve(private)
    _lifecycle().validate_directory(private)
    planned = [{'task_id': task_id, 'arm': 'first_attempt', 'phase': 'source'}
               for task_id in plan['batches'][index]]
    rows, events, failed = [], None, False
    try:
        events = _Events(stage / 'run/events.jsonl')
        for offset, slot in enumerate(planned):
            row = _run_one(plan, slot['task_id'], offset, request['expected_backend'], tunnel,
                           deadline, events, private)
            rows.append(row)
            if row['status'] == 'unknown':
                break
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        failed = True
    finally:
        if events is not None:
            try:
                events.close()
            except BaseException:
                failed = True
            failed = failed or events.failed
    events_sha = None
    try:
        _, events_sha = _read(stage / 'run/events.jsonl', public=True)
    except Exception:
        failed = True
    complete = not failed and len(rows) == 3 and all(row['status'] != 'unknown' for row in rows)
    report = {'id': ID, 'protocol': BATCH_PROTOCOL, 'plan_sha256': sha, 'request_sha256': request_sha,
              'batch_index': index, 'status': 'complete' if complete else 'unknown',
              'safe_to_continue': complete, 'planned_slots': planned, 'slots': rows,
              'unattempted_slots': planned[len(rows):], 'journal_failed': failed, 'events_sha256': events_sha}
    io.write_new(stage / 'run/report.json', report)
    return 0 if complete else 1


def _verify_events(raw, slots, image):
    _require(raw and raw.endswith(b'\n'))
    rows = [json.loads(line, object_pairs_hook=_pairs, parse_constant=_reject_number)
            for line in raw.splitlines()]
    _require(all(type(row) is dict and row.get('event_index') == i and type(row.get('event_index')) is int
                 and type(row.get('epoch')) in (int, float) and math.isfinite(row['epoch'])
                 and row['epoch'] >= 0 for i, row in enumerate(rows)))
    position, names = 0, set()
    for slot in slots:
        start = position
        while position < len(rows) and rows[position].get('task_id') == slot['task_id']:
            position += 1
        selected = rows[start:position]
        _require(selected and all(row.get('arm') == 'first_attempt' and row.get('phase') == 'source'
                                  for row in selected))
        fixed = [row for row in selected if row.get('event') in _OWNER_EVENTS]
        kinds = [row['event'] for row in fixed]
        _require(kinds == ['world_startup_attempt', 'container_intent', 'owner_api_count_attempt',
                 'owner_api_count_returned', 'world_ready', 'world_operation_attempt',
                 'owner_api_count_attempt', 'owner_api_count_returned', 'world_operation_returned',
                 'world_cleanup_attempt', 'world_cleanup_returned'])
        _require(all(set(row) == {'event_index', 'epoch', 'event', 'task_id', 'arm', 'phase'}
                     | _OWNER_EVENTS[row['event']] for row in fixed))
        actor = [row for row in selected if row.get('event') in {'generation_attempt', 'generation_returned'}]
        _require(selected == fixed[:6] + actor + fixed[6:])
        intent = fixed[1]['intent']
        _validate_intent(intent)
        _require(intent['image'] == image and intent['container_name'] not in names)
        names.add(intent['container_name'])
        ready = fixed[4]
        seed = ready['environment_seed']
        _require(ready['container_id'] == slot['owner']['container_id']
                 and type(seed) is dict and set(seed) == {'requested', 'effective', 'verified'}
                 and all(type(seed[key]) is int and seed[key] == 100 for key in ('requested', 'effective'))
                 and seed['verified'] is True
                 and type(fixed[3]['value']) is int and fixed[3]['value'] == 0
                 and type(fixed[7]['value']) is int
                 and fixed[7]['value'] == slot['owner']['api_attempts_observed']
                 and fixed[-1]['verified'] is True)
        attempts = returns = prompt = completion = 0
        open_index = None
        for row in actor:
            is_attempt = row['event'] == 'generation_attempt'
            fields = {'index', 'seed'} if is_attempt else {'index', 'seed', 'prompt_tokens', 'completion_tokens'}
            _require(set(row) == {'event_index', 'epoch', 'event', 'task_id', 'arm', 'phase'} | fields
                     and type(row['index']) is int and type(row['seed']) is int
                     and row['seed'] == 17 + row['index'])
            if is_attempt:
                _require(open_index is None and row['index'] == attempts)
                open_index, attempts = attempts, attempts + 1
            else:
                _require(open_index == row['index'] and type(row['prompt_tokens']) is int
                         and row['prompt_tokens'] > 0 and type(row['completion_tokens']) is int
                         and row['completion_tokens'] > 0)
                returns += 1
                prompt += row['prompt_tokens']
                completion += row['completion_tokens']
                open_index = None
        _require({'generation_attempts': attempts, 'generation_returns': returns,
                  'prompt_tokens': prompt, 'completion_tokens': completion} == slot['observed_model']
                 and (open_index is None or slot['termination'] == 'context_budget'))
    _require(position == len(rows))


def _trace_rows(raw):
    _require(not raw or raw.endswith(b'\n'))
    rows = [json.loads(line, object_pairs_hook=_pairs, parse_constant=_reject_number)
            for line in raw.splitlines()]
    _require(all(type(row) is dict and type(row.get('index')) is int and row['index'] == index
                 for index, row in enumerate(rows)))
    return rows


def _same(left, right):
    return wire._encoded(left) == wire._encoded(right)


class _TraceCursor:
    def __init__(self, raw):
        self.rows = _trace_rows(raw)
        self.position = 0

    def take(self, event, fields):
        _require(self.position < len(self.rows))
        row = self.rows[self.position]
        self.position += 1
        _require(row.get('event') == event and set(row) == {'index', 'event'} | set(fields))
        _require(all(_same(row[key], value) for key, value in fields.items()))
        return row

    def take_dynamic(self, event, fields):
        _require(self.position < len(self.rows))
        row = self.rows[self.position]
        self.position += 1
        _require(row.get('event') == event and set(row) == {'index', 'event'} | set(fields))
        return row

    def done(self):
        _require(self.position == len(self.rows))


def _verify_private(actor_raw, wire_raw, episode, expected_backend):
    """Authenticate histories and receipts without executing actor code.

    Completion comes from the source-bound trusted runner outcome; these logs
    do not independently prove the worker's completion probe.
    """
    private, outcome = episode['private'], episode['outcome']
    _require(type(private) is dict and set(private) == {'messages', 'steps'}
             and type(private['steps']) is list and len(private['steps']) <= 40)
    steps = private['steps']
    history = copy.deepcopy(steps[0]['messages_before'] if steps else private['messages'])
    actor, model = _TraceCursor(actor_raw), _TraceCursor(wire_raw)
    actor.take('actor_start', {'messages': history, 'seed': 17})
    last_request = 0
    returns = actions = 0
    measured = dict.fromkeys(_COST_KEYS, 0)
    for index, step in enumerate(steps):
        _require(type(step) is dict and set(step) == {'index', 'seed', 'messages_before',
                 'response', 'action', 'execution'} and type(step['index']) is int
                 and step['index'] == index and type(step['seed']) is int and step['seed'] == 17 + index
                 and _same(step['messages_before'], history))
        actor.take('generation_intent', {'step_index': index, 'seed': step['seed'], 'messages': history})
        response = step['response']
        if response is None:
            _require(index == len(steps) - 1 and step['action'] is None and step['execution'] is None
                     and outcome['termination'] in {'actor_wall', 'context_budget'})
            actor.take('generation_not_dispatched', {'step_index': index, 'reason': outcome['termination']})
            if outcome['termination'] == 'actor_wall':
                continue
        else:
            _require(_response_valid(response))
            returns += 1
            measured['prompt_tokens'] += response['prompt_tokens']
            measured['completion_tokens'] += response['completion_tokens']
            actor.take('generation_return', {'step_index': index, 'response': response})
        _, messages_sha = wire._message_payload(history)
        model.take('count_intent', {'messages': history, 'messages_sha256': messages_sha, 'seed': step['seed']})
        counted = model.take_dynamic('count_returned', {'input_tokens', 'messages_sha256'})
        count = counted['input_tokens']
        _require(type(count) is int and count > 0 and counted['messages_sha256'] == messages_sha)
        if response is None:
            _require(count > 15360)
            model.take('generation_failed', {'stage': 'count', 'transport_started': False,
                       'observed_usage': None, 'known_not_dispatched': True, 'unknown_cost': False})
            continue
        _require(model_client._usage({'usage': {'prompt_tokens': response.get('prompt_tokens'),
                    'completion_tokens': response.get('completion_tokens')}}) is not None
                 and count == response['prompt_tokens'] and count <= 15360)
        request = {'model': MODEL_ID, 'messages': history, 'seed': step['seed'],
                   'max_tokens': 1024, 'temperature': 0.7}
        model.take('generation_intent', {'request': request, 'request_sha256': wire._digest(request),
                                        'messages_sha256': messages_sha, 'seed': step['seed']})
        raw_reply = model.take_dynamic('generation_raw_reply', {'raw_reply_base64'})['raw_reply_base64']
        _require(type(raw_reply) is str)
        reply = wire._json(base64.b64decode(raw_reply, validate=True))
        _require(type(reply) is dict and set(reply) == model_client._REPLY_KEYS
                 and reply['model'] == MODEL_ID and _same(reply['backend'], expected_backend)
                 and type(reply['choices']) is list and len(reply['choices']) == 1
                 and type(reply['choices'][0]) is dict and set(reply['choices'][0]) == {'message', 'finish_reason'}
                 and type(reply['choices'][0]['message']) is dict
                 and set(reply['choices'][0]['message']) == {'role', 'content'}
                 and reply['choices'][0]['message']['role'] == 'assistant'
                 and type(reply['choices'][0]['message']['content']) is str
                 and reply['choices'][0]['finish_reason'] in {'stop', 'length', 'unknown'}
                 and model_client._usage(reply) is not None and reply['usage']['prompt_tokens'] == count
                 and (reply['format_error'] is None or type(reply['format_error']) is str)
                 and model_client._finite(reply['generation_seconds'])
                 and type(reply['request_id']) is int and reply['request_id'] > last_request)
        model_client._memory(reply['memory'])
        reconstructed = {'text': reply['choices'][0]['message']['content'], **reply['usage'],
                         **{key: reply[key] for key in ('format_error', 'backend', 'generation_seconds',
                                                         'request_id', 'memory')},
                         'finish_reason': reply['choices'][0]['finish_reason']}
        _require(_same(reconstructed, response))
        model.take('generation_validated', {'response': response})
        last_request = reply['request_id']
        if step['action'] is not None:
            _require(type(step['action']) is str and type(step['execution']) is dict
                     and type(step['execution'].get('observation')) is str
                     and _worker_valid(step['execution'], measured))
            actions += 1
            measured['api_request_dispatches'] = step['execution']['api']['request_dispatches']
            measured['budget_rejections'] = step['execution']['api']['budget_rejections']
            actor.take('action_intent', {'step_index': index, 'action': step['action']})
            actor.take('action_result', {'step_index': index, 'result': step['execution']})
            history.extend([{'role': 'assistant', 'content': response['text']},
                            {'role': 'user', 'content': step['execution']['observation']}])
        else:
            _require(step['execution'] is None and index == len(steps) - 1)
    _require(_same(history, private['messages']))
    counts = outcome['counts']
    pending_wall = bool(steps and steps[-1]['response'] is None and outcome['termination'] == 'actor_wall')
    _require(counts['generation_attempts'] == len(steps) - int(pending_wall)
             and counts['generation_returns'] == returns and counts['actions'] == actions
             and _same(measured, outcome['known_costs']))
    actor.take('actor_stop', {'termination': outcome['termination'],
               'model_reported_complete': outcome['model_reported_complete'],
               'known_costs': outcome['known_costs'], 'generation_attempts': counts['generation_attempts'],
               'generation_returns': counts['generation_returns'], 'actions': counts['actions']})
    actor.done()
    model.done()


def verify_batch_report(stage, plan, index, sha):
    _require(type(index) is int and 0 <= index < 4 and type(sha) is str
             and re.fullmatch('[a-f0-9]{64}', sha) is not None)
    stage = Path(stage)
    report, _ = _read_json(stage / 'run/report.json')
    request, request_sha = _read_json(stage / 'request.json')
    _require(type(request) is dict and set(request) == {'plan_sha256', 'batch_index', 'expected_backend'}
             and request['plan_sha256'] == sha and type(request['batch_index']) is int
             and request['batch_index'] == index and type(request['expected_backend']) is dict
             and bool(request['expected_backend']))
    planned = [{'task_id': task_id, 'arm': 'first_attempt', 'phase': 'source'}
               for task_id in plan['batches'][index]]
    _require(type(report) is dict and set(report) == {'id', 'protocol', 'plan_sha256', 'request_sha256',
             'batch_index', 'status', 'safe_to_continue', 'planned_slots', 'slots', 'unattempted_slots',
             'journal_failed', 'events_sha256'}
             and report['id'] == ID and report['protocol'] == BATCH_PROTOCOL and report['plan_sha256'] == sha
             and report['request_sha256'] == request_sha and type(report['batch_index']) is int
             and report['batch_index'] == index and report['status'] == 'complete'
             and report['safe_to_continue'] is True and report['journal_failed'] is False
             and report['planned_slots'] == planned and report['unattempted_slots'] == []
             and type(report['slots']) is list and len(report['slots']) == len(planned) == 3)
    private = PRIVATE / f'batch-{index}'
    _lifecycle().validate_directory(private)
    expected_names = set()
    for offset, (slot, row) in enumerate(zip(planned, report['slots'], strict=True)):
        _require(type(row) is dict and set(row) == {'task_id', 'arm', 'phase', 'status',
                 'owner', 'observed_model', 'private_bindings'} | set(_OUTCOME_FIELDS)
                 and all(row[key] == slot[key] for key in slot)
                 and row['status'] in {'scored', 'known_unscored'}
                 and type(row['owner']) is dict and type(row['observed_model']) is dict
                 and set(row['observed_model']) == {'generation_attempts', 'generation_returns',
                                                    'prompt_tokens', 'completion_tokens'})
        outcome = {key: row[key] for key in _OUTCOME_FIELDS}
        _require(_outcome_valid(outcome) and _owner_verified(row['owner'], outcome, row['observed_model']))
        _require((row['status'] == 'scored' and outcome['termination'] in _SAFE_TERMINATIONS
                  and type(outcome['official_success']) is bool)
                 or (row['status'] == 'known_unscored' and outcome['termination'] in _KNOWN_UNSCORED
                     and outcome['official_success'] is None))
        bindings = row['private_bindings']
        _require(type(bindings) is dict and set(bindings) == {'actor_trace', 'wire_trace', 'episode_result'})
        private_raw = {}
        for key, filename in (('actor_trace', f'actor-{offset:02}.jsonl'),
                              ('wire_trace', f'wire-{offset:02}.jsonl'),
                              ('episode_result', f'episode-{offset:02}.json')):
            binding = bindings[key]
            _require(type(binding) is dict and set(binding) == {'file', 'sha256'}
                     and binding['file'] == filename and type(binding['sha256']) is str
                     and re.fullmatch('[a-f0-9]{64}', binding['sha256']) is not None)
            raw, _ = _read(private / filename, expected=binding['sha256'], maximum=128_000_000)
            private_raw[key] = raw
            expected_names.add(filename)
            if key == 'episode_result':
                episode = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_reject_number)
                _require(type(episode) is dict and set(episode) == {'private', 'outcome'}
                         and _same(episode['outcome'], outcome))
        try:
            _verify_private(private_raw['actor_trace'], private_raw['wire_trace'], episode,
                            request['expected_backend'])
        except Exception:
            raise ValueError(_ERROR) from None
    _require(set(os.listdir(private)) == expected_names)
    _require(type(report['events_sha256']) is str and re.fullmatch('[a-f0-9]{64}', report['events_sha256']))
    event_raw, _ = _read(stage / 'run/events.jsonl', expected=report['events_sha256'], public=True)
    _verify_events(event_raw, report['slots'], plan['image'])
    return True


def run_stage(plan, sha, index):
    return _lifecycle().run_stage(plan, sha, index)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--client-request', required=True)
    parser.add_argument('--client-request-sha256', required=True)
    args = parser.parse_args(argv)
    return run_client(args.plan_sha256, args.client_request, args.client_request_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
