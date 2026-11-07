"""One-shot private writer coordinator; no Task world, scorer or training route.

The source qualification is recomputed in one bounded trusted CPU child. Its
private bundle and each full writer input stay in private files/pipes. Later
bounded children recheck frozen bytes rather than mining histories again.
Import and help perform no admission, service, private-data or model operation.
"""

import argparse
import ast
import hashlib
import importlib
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from scripts import native_experience_content_001 as content
from scripts import native_experience_registration_001 as registration
from scripts import native_experience_writer_001 as writer
from scripts.native_appworld_model_client_004 import NativeModel
from scripts.native_appworld_worker_client import _decode, _encode, _exclusive
from scripts.verification_token_counter import _message_payload

_ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-native-experience-writer-001'
PLAN = 'artifacts/qwen38-native-experience-writer-plan-001.json'
SOURCE6_SHA256 = 'b1ecbefadae2c4257d7baca09265948d8ca8d6b0a0928d644f2d87b0a320e4c3'
LIFECYCLE_SHA256 = 'c9cac394a244e8764d3bbd7143d3b44e0db00e20cc5f6016d84504d89bbee7f6'
PARENT_SECONDS = 94200
VERIFICATION_SECONDS = 180
MAX_VERIFICATION_BYTES = 32_000_000
_ERROR = 'native_experience_writer_coordinator_failed'
_SOURCE_CORE = None
SOURCE_GATE_KIND = 'ordinary_control_recovery_gate'
_MANIFEST_FIELDS = frozenset({
    'private', 'source_gate_kind', 'primary_data_feasibility_gate', 'primary_candidate_student_target_sha256',
    'ordinary_control_targets', 'ordinary_control_target_sha256', 'ordinary_targets_scope',
    'source_plan_sha256', 'source_result_sha256', 'private_files_sha256', 'public_context_binding',
    'writer_messages_sha256', 'scope',
})
_TARGET_FIELDS = frozenset({
    'task_id', 'target_sha256', 'action', 'student_messages', 'source_messages_sha256',
    'student_messages_sha256', 'new_generation_index',
})
_VERIFY_CODE = r'''
import json,sys
try:
    raw=sys.stdin.buffer.read(1000001)
    if len(raw)>1000000: raise ValueError()
    spec=json.loads(raw)
    from pathlib import Path
    root=Path(spec['root'])
    if not root.is_absolute() or '..' in root.parts: raise ValueError()
    sys.path.insert(0,str(root))
    from scripts.run_native_experience_writer_001 import _verification_payload,_encode
    answer=_encode(_verification_payload(spec))
    if len(answer)>32000000: raise ValueError()
    sys.stdout.buffer.write(answer)
except BaseException:
    sys.stderr.write('native_experience_verification_failed\n')
    raise SystemExit(2)
'''


class NativeExperienceWriterError(RuntimeError):
    """Fixed content-free failure; all histories and costs remain private."""


class _BudgetStop(Exception):
    pass


def _require(value):
    if not value:
        raise NativeExperienceWriterError(_ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _control_manifest(manifest, source):
    """Check the complete private source binding without rewriting its contents."""
    _require(type(source) is dict and source.get('source_gate_kind') == SOURCE_GATE_KIND
             and type(manifest) is dict and set(manifest) == _MANIFEST_FIELDS
             and manifest['private'] is True and manifest['source_gate_kind'] == SOURCE_GATE_KIND
             and manifest['source_plan_sha256'] == source['plan_sha256'] == registration.SOURCE_PLAN_SHA
             and registration.public._sha(source['result_sha256'])
             and manifest['source_result_sha256'] == source['result_sha256']
             and _hash(_encode(manifest)) == source['private_input_manifest_sha256']
             and type(manifest['private_files_sha256']) is dict
             and type(manifest['ordinary_targets_scope']) is str and type(manifest['scope']) is str)
    inputs = source['writer_inputs']
    _require(type(inputs) is list and all(type(item) is dict and set(item) == {
        'source_index', 'source_scenario', 'messages_sha256'} for item in inputs)
        and manifest['writer_messages_sha256'] == [item['messages_sha256'] for item in inputs])
    context = manifest['public_context_binding']
    _require(type(context) is dict and set(context) == {'path', 'sha256'}
             and type(context['path']) is str and registration.public._sha(context['sha256']))
    primary, primary_hashes = manifest['primary_data_feasibility_gate'], manifest['primary_candidate_student_target_sha256']
    _require(type(primary) is dict and set(primary) == {
        'status', 'complete_help_scenarios', 'distinct_student_targets', 'source_denominator'}
        and primary['status'] in {'passed', 'insufficient'} and type(primary['source_denominator']) is int
        and primary['source_denominator'] == 12 and type(primary['complete_help_scenarios']) is int
        and 0 <= primary['complete_help_scenarios'] <= 12 and type(primary['distinct_student_targets']) is int
        and type(primary_hashes) is list and all(registration.public._sha(value) for value in primary_hashes)
        and primary_hashes == sorted(set(primary_hashes))
        and len(primary_hashes) == primary['distinct_student_targets']
        and len(primary_hashes) <= 24 * primary['complete_help_scenarios']
        and (primary['status'] == 'passed') == (
            primary['complete_help_scenarios'] >= 4 and primary['distinct_student_targets'] >= 20))
    targets, hashes = manifest['ordinary_control_targets'], manifest['ordinary_control_target_sha256']
    _require(type(targets) is list and type(hashes) is list
             and all(type(row) is dict and set(row) == _TARGET_FIELDS
                     and registration.public._sha(row['target_sha256'])
                     and registration.public._sha(row['source_messages_sha256'])
                     and row['student_messages_sha256'] == row['source_messages_sha256']
                     and type(row['task_id']) is str and type(row['action']) is str
                     and type(row['student_messages']) is list
                     and type(row['new_generation_index']) is int and 0 <= row['new_generation_index'] < 24
                     for row in targets)
             and all(registration.public._sha(value) for value in hashes)
             and hashes == sorted(set(hashes))
             and hashes == sorted(row['target_sha256'] for row in targets))
    return len(hashes)


def _queue():
    return importlib.import_module('scripts.native_experience_gpu_queue_001')


def _make_job(owner, sha):
    return _queue().make_job(0, owner, sha)


def _source_core():
    global _SOURCE_CORE
    if _SOURCE_CORE is not None:
        return _SOURCE_CORE
    _require(_hash(registration.public_read(_ROOT, 'scripts/run_native_source_pilot_006.py')) == SOURCE6_SHA256)
    from scripts import run_native_source_pilot_006 as source

    _SOURCE_CORE = source._isolated()
    return _SOURCE_CORE


def unit_command(unit, arguments, *, root=_ROOT, log_path):
    """Retain source-parent CPU/network controls, with a bounded writer CLI."""
    command = _source_core().unit_command('pilot', unit, arguments, root=root, log_path=log_path)
    old = '--property=RuntimeMaxSec=1123800'
    _require(command.count(old) == 1)
    command[command.index(old)] = '--property=RuntimeMaxSec=' + str(PARENT_SECONDS)
    locations = [i for i, value in enumerate(command) if 'scripts.run_native_source_pilot_006' in value]
    _require(len(locations) == 1)
    i = locations[0]
    command[i] = command[i].replace('scripts.run_native_source_pilot_006',
                                   'scripts.run_native_experience_writer_001')
    return command


def _services(root):
    """Clone authenticated helpers into writer globals; mutate no predecessor."""
    core, queue = _source_core(), _queue()
    _require(queue.ID == registration.ID == ID and queue.PLAN == registration.PLAN == PLAN)
    registry = SimpleNamespace(**{**vars(registration), 'static_backend': core.registration.static_backend})
    episode = core.episode._get_core()
    reader_scope = dict(episode.__dict__)
    reader_scope.update(PRIVATE_RUN_ROOT=Path(registration.PRIVATE_RUN_ROOT),
                        registration=registry, queue=queue, REMOTE=queue.REMOTE)
    for name in ('_private_directory', '_read_private', '_backend', '_queue_binding'):
        function = getattr(episode, name)
        reader_scope[name] = FunctionType(function.__code__, reader_scope, name,
                                          function.__defaults__, function.__closure__)
    reader = SimpleNamespace(**reader_scope)
    raw = registration.public_read(_ROOT, 'scripts/run_native_source_pilot.py')
    _require(_hash(raw) == LIFECYCLE_SHA256)
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == '_Runtime')
    status = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'unit_status')
    durations = [n for n in ast.walk(status) if isinstance(n, ast.Constant) and n.value == 900]
    _require(len(durations) == 1)
    durations[0].value = PARENT_SECONDS
    methods = {'__init__', 'now', 'epoch', 'wait', 'owner', '_write', 'record', 'write', 'read',
               'queue', 'heartbeat', 'monitor', 'port_owners', 'ready', 'unit_status', 'stop_unit', 'close'}
    node.body = [part for part in node.body if isinstance(part, ast.FunctionDef) and part.name in methods]
    scope = dict(core.__dict__)
    scope.update(registration=registry, queue=queue, episode=reader)
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, str(_ROOT / 'scripts/run_native_source_pilot.py'), 'exec'), scope)
    return scope['_Runtime'](root), reader


def _verification_payload(spec):
    """Fixed child implementation. Its return value must stay on a private pipe."""
    root, sha = Path(spec['root']), spec['plan_sha256']
    if spec['action'] == 'preflight':
        _require(set(spec) == {'action', 'root', 'plan_sha256'})
        plan, bundle = registration.private_preflight(root, sha)
        remote = registration.load_registration(_queue().REMOTE, sha, private=False)
        _require(_encode(remote) == _encode(plan))
        return {'plan': plan, 'bundle': bundle}
    _require(spec['action'] == 'hash'
             and set(spec) == {'action', 'root', 'plan_sha256', 'manifest', 'input'})
    plan = registration.load_registration(root, sha, private=False)
    _service, reader = _services(root)
    binding = spec['manifest']
    _require(binding['path'] == str(Path(plan['private_run_root']) / 'input-manifest.json'))
    raw = reader._read_private(binding['path'], maximum=MAX_VERIFICATION_BYTES)
    _require(_hash(raw) == binding['sha256'])
    manifest = _decode(raw)
    source = plan['source']
    _control_manifest(manifest, source)
    from scripts import native_source_episode_006 as episode6

    source_reader = episode6._get_core()
    for path, digest in manifest['private_files_sha256'].items():
        _require(_hash(source_reader._read_private(path, maximum=64_000_000)) == digest)
    context = manifest['public_context_binding']
    _require(_hash(registration.public_read(root, context['path'])) == context['sha256'])
    binding = spec['input']
    index = binding['source_index']
    _require(type(index) is int and binding['path'] == str(
        Path(plan['private_run_root']) / f'writer-input-{index:02d}.json'))
    raw = reader._read_private(binding['path'], maximum=MAX_VERIFICATION_BYTES)
    _require(_hash(raw) == binding['sha256'])
    item = _decode(raw)
    metadata = {key: item[key] for key in ('source_index', 'source_scenario', 'messages_sha256')}
    _require(set(item) == {'source_index', 'source_scenario', 'messages', 'messages_sha256'}
             and metadata in source['writer_inputs']
             and all(item[key] == binding[key] for key in metadata)
             and _message_payload(item['messages'])[1] == item['messages_sha256'])
    return {'input': item}


class _Runtime:
    def __init__(self, root):
        self.root = Path(root)
        self.service, self.reader = _services(root)
        self.verification_audits, self.verification_spent = [], 0.0

    def __getattr__(self, name):
        return getattr(self.service, name)

    def _verify(self, spec, *, gpu_remaining):
        remaining = VERIFICATION_SECONDS - self.verification_spent
        if gpu_remaining is not None:
            remaining = min(remaining, gpu_remaining - 150)
        _require(remaining > 0)
        spec = {**spec, 'root': str(self.root)}
        payload = _encode(spec)
        _require(len(payload) <= 1_000_000)
        env = {key: os.environ[key] for key in (
            'PATH', 'HOME', 'USER', 'LOGNAME', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS') if key in os.environ}
        env.update(CUDA_VISIBLE_DEVICES='', TMPDIR=registration.TMPDIR,
                   PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='17', LANG='C.UTF-8',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   DO_NOT_TRACK='1', TOKENIZERS_PARALLELISM='false')
        audit = {'stage': spec.get('action'), 'wall_seconds': None, 'returned': False,
                 'timeout_seconds': remaining, 'child_returncode': None,
                 'stdout_bytes': None, 'stdout_sha256': None, 'stderr_bytes': None, 'stderr_sha256': None}
        started, stdout, stderr = self.now(), None, None
        try:
            result = subprocess.run([registration.PYTHON, '-s', '-P', '-B', '-c', _VERIFY_CODE],
                input=payload, capture_output=True, timeout=remaining,
                check=False, env=env)
            stdout, stderr, audit['child_returncode'] = result.stdout, result.stderr, result.returncode
            _require(result.returncode == 0 and type(stdout) is bytes
                     and len(stdout) <= MAX_VERIFICATION_BYTES and self.now() - started <= remaining)
            decoded = _decode(stdout)
            audit['returned'] = True
            return decoded
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.output, error.stderr
            raise NativeExperienceWriterError(_ERROR) from None
        except Exception:
            raise NativeExperienceWriterError(_ERROR) from None
        finally:
            audit['wall_seconds'] = self.now() - started
            self.verification_spent += audit['wall_seconds']
            for name, data in (('stdout', stdout), ('stderr', stderr)):
                if type(data) is bytes:
                    audit[name + '_bytes'], audit[name + '_sha256'] = len(data), _hash(data)
            self.verification_audits.append(audit)

    def preflight(self, sha):
        _require(socket.gethostname() == 'amax' and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
        answer = self._verify({'action': 'preflight', 'plan_sha256': sha}, gpu_remaining=None)
        plan, bundle = answer['plan'], answer['bundle']
        self.service.plan = plan
        return plan, bundle

    def begin(self, plan, sha, slots):
        service = self.service
        service.private = Path(plan['private_run_root'])
        service.fd = self.reader._private_directory(service.private)
        with _exclusive(service.fd, 'study-intent.json') as file:
            service._write(file, {'plan_sha256': sha, 'sources': slots, 'automatic_retry': False,
                                  'pid': os.getpid(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
        service.journal = _exclusive(service.fd, 'study-journal.jsonl')
        os.fsync(service.fd)

    def preflight_failure(self, report):
        descriptor = self.reader._private_directory(Path(registration.PRIVATE_RUN_ROOT))
        try:
            with _exclusive(descriptor, 'preflight-failure.json') as file:
                self.service._write(file, report)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def binding(self, job, service):
        state, _ready, identity = self.reader._queue_binding(
            {'job': job, 'expected_backend': service['backend']}, _queue().queue_operation)
        _require(identity == service['identity'])
        return state

    def check_input(self, sha, manifest, item_binding, *, gpu_remaining):
        return self._verify({'action': 'hash', 'plan_sha256': sha,
                             'manifest': manifest, 'input': item_binding},
                            gpu_remaining=gpu_remaining)['input']

    def model(self, plan, service, *, deadline):
        return NativeModel(f'http://127.0.0.1:{registration.PORT}/v1', plan['model_id'],
            expected_backend=service['backend'], max_tokens=128, context_tokens=16384,
            temperature=0.7, timeout=120, deadline=deadline, _clock=self.now)

    def reviewer_model(self, plan, service, *, deadline):
        return NativeModel(f'http://127.0.0.1:{registration.PORT}/v1', plan['model_id'],
            expected_backend=service['backend'], max_tokens=256, context_tokens=16384,
            temperature=0.7, timeout=120, deadline=deadline, _clock=self.now)

    def release(self, job, known_pids):
        return _queue().fenced_release(self.service, job, known_pids)


class _BoundModel:
    def __init__(self, model, digest, check):
        self.model, self.digest, self.check = model, digest, check
        self.max_tokens = model.max_tokens

    def check_health(self):
        self.check()
        return self.model.check_health()

    def count_input_tokens(self, messages):
        self.check()
        _require(_message_payload(messages)[1] == self.digest)
        return self.model.count_input_tokens(messages)

    def generate(self, messages, seed):
        self.check()
        _require(_message_payload(messages)[1] == self.digest)
        return self.model.generate(messages, seed=seed)


def _qualified(plan, bundle):
    _require(plan['id'] == ID and plan['status'] == 'registered' and plan['model_arm'] == 'B3'
             and plan['automatic_retry'] is False and plan['writer_authorized'] is True
             and all(plan[key] is False for key in ('training_authorized', 'utility_authorized', 'dev_or_test_authorized'))
             and plan['writer'] == registration.writer_policy() and type(bundle) is dict
             and set(bundle) == {'private', 'source_gate_kind', 'gate', 'writer_inputs', 'input_manifest',
                                 'input_manifest_sha256'}
             and bundle['private'] is True and bundle['source_gate_kind'] == SOURCE_GATE_KIND)
    gate, items = bundle['gate'], bundle['writer_inputs']
    _require(type(gate) is dict and set(gate) == {'status', 'source_denominator', 'complete_recovery_scenarios',
                                                  'distinct_student_targets'}
             and gate['status'] == 'passed' and type(gate['source_denominator']) is int
             and gate['source_denominator'] == 12 and type(gate['complete_recovery_scenarios']) is int
             and 4 <= gate['complete_recovery_scenarios'] <= 12 and type(gate['distinct_student_targets']) is int
             and 20 <= gate['distinct_student_targets'] <= 24 * gate['complete_recovery_scenarios']
             and bundle['input_manifest_sha256'] == plan['source']['private_input_manifest_sha256']
             and type(items) is list and len(items) == gate['complete_recovery_scenarios']
             and _control_manifest(bundle['input_manifest'], plan['source']) == gate['distinct_student_targets'])
    metadata = [{key: item[key] for key in ('source_index', 'source_scenario', 'messages_sha256')} for item in items]
    _require(metadata == plan['source']['writer_inputs']
             and [v['source_index'] for v in metadata] == sorted({v['source_index'] for v in metadata})
             and len({v['source_scenario'] for v in metadata}) == len(metadata)
             and all(type(item['source_index']) is int and 0 <= item['source_index'] < 12
                     and _message_payload(item['messages'])[1] == item['messages_sha256'] for item in items))
    return metadata


def run_writer(plan_sha, root=_ROOT, _runtime=None):
    """Run one registered writer job once; never resume or fill missing inputs."""
    _require(type(plan_sha) is str and len(plan_sha) == 64 and all(c in '0123456789abcdef' for c in plan_sha))
    runtime = _runtime or _Runtime(root)
    parent_started = runtime.now()
    report = {'id': ID, 'plan_sha256': plan_sha, 'status': 'unknown', 'sources': [],
              'costs': {'source_verifications': runtime.verification_audits, 'writer_calls': [],
                        'review_calls': [], 'GPU_jobs': []},
              'automatic_retry': False, 'training_jobs': 0, 'utility_or_dev_or_test_calls': 0}
    begun = preflight_complete = enqueued = False
    heartbeat, job, known_pids = None, None, []
    admission_start = admission_epoch = None
    try:
        plan, bundle = runtime.preflight(plan_sha)
        metadata = _qualified(plan, bundle)
        preflight_complete = True
        report['sources'] = [{**row, 'status': 'unstarted', 'review_status': 'unstarted'} for row in metadata]
        runtime.begin(plan, plan_sha, report['sources'])
        begun = True
        manifest = runtime.write('input-manifest.json', bundle['input_manifest'])
        bindings = []
        for item in bundle['writer_inputs']:
            binding = runtime.write(f"writer-input-{item['source_index']:02d}.json", item)
            bindings.append({**next(v for v in metadata if v['source_index'] == item['source_index']), **binding})
        runtime.record({'event': 'writer_inputs_frozen', 'manifest': manifest, 'inputs': bindings})
        runtime.monitor(empty_port=True)
        job = _make_job(runtime.owner(), plan_sha)
        admission_start, admission_epoch = runtime.now(), runtime.epoch()
        runtime.record({'event': 'GPU_enqueue_intent', 'job': job})
        enqueued = True
        state = runtime.queue('enqueue', job)
        heartbeat = runtime.heartbeat(job)
        service = None
        while service is None:
            heartbeat.check()
            runtime.monitor()
            _source_core()._queue_state(state, runtime)
            _require(state.get('released') is False)
            info = state.get('job')
            if type(info) is dict and info.get('status') == 'running':
                if runtime.epoch() - info['started_epoch'] >= plan['writer']['model_ready_seconds']:
                    raise _BudgetStop('model_ready_budget')
            if runtime.now() - admission_start >= plan['admission_wait_seconds']:
                raise _BudgetStop('admission_timeout')
            service = runtime.ready(job, state)
            if (type(info) is dict and info.get('status') == 'running'
                    and runtime.epoch() - info['started_epoch'] >= plan['writer']['model_ready_seconds']):
                raise _BudgetStop('model_ready_budget')
            if service is None:
                runtime.wait(5)
                state = runtime.queue('pulse', job)
        known_pids = list(state['model_pids'])
        started_epoch = state['job']['started_epoch']
        runtime.record({'event': 'GPU_ready', 'job': job, 'service': service})

        def remaining():
            return 7200 - (runtime.epoch() - started_epoch)

        def guard():
            heartbeat.check()
            if remaining() <= 150:
                raise _BudgetStop('GPU_cap')
            latest = runtime.binding(job, service)
            _source_core()._queue_state(latest, runtime)
            _require(latest.get('released') is False and latest.get('job', {}).get('status') == 'running')

        for slot, binding in zip(report['sources'], bindings, strict=True):
            guard()
            item = runtime.check_input(plan_sha, manifest, binding, gpu_remaining=remaining())
            _require(all(item[key] == binding[key] for key in ('source_index', 'source_scenario', 'messages_sha256'))
                     and _message_payload(item['messages'])[1] == binding['messages_sha256'])
            guard()
            if remaining() < plan['writer']['per_call_seconds'] + 150:
                raise _BudgetStop('GPU_cap')
            model = _BoundModel(runtime.model(plan, service, deadline=runtime.now() + 300),
                                item['messages_sha256'], guard)
            entry = {'source_index': item['source_index'], 'attempted': False,
                     'wall_seconds': None, 'result': None, 'unknown_cost': True}
            report['costs']['writer_calls'].append(entry)
            started = runtime.now()
            slot['status'] = 'running'
            try:
                runtime.record({'event': 'writer_call_intent', 'source_index': item['source_index'],
                                'messages_sha256': item['messages_sha256'], 'seed': 17 + item['source_index']})
                entry['attempted'] = True
                result = writer.write_once(item['messages'], model, seed=17 + item['source_index'],
                    record=lambda event, index=item['source_index']: runtime.record(
                        {'event': 'writer_event', 'source_index': index, 'writer_event': event}))
                entry['result'], entry['unknown_cost'] = result, result['cost']['unknown_cost']
                slot['status'] = result['status']
                _require(result['status'] in {'candidate_unvalidated', 'abstained', 'unknown'}
                         and result['accepted'] is False and result['training_authorized'] is False)
                binding_result = runtime.write(f"writer-result-{item['source_index']:02d}.json", result)
                runtime.record({'event': 'writer_call_finished', 'source_index': item['source_index'],
                                'result_binding': binding_result})
            finally:
                entry['wall_seconds'] = runtime.now() - started
            if entry['wall_seconds'] > plan['writer']['per_call_seconds']:
                raise _BudgetStop('writer_call_budget')
            if slot['status'] == 'unknown' or entry['unknown_cost']:
                break
            if slot['status'] != 'candidate_unvalidated':
                slot['review_status'] = 'not_applicable'
                continue
            guard()
            candidate = result['candidate']
            review_entry = {'source_index': item['source_index'], 'attempted': False,
                            'wall_seconds': None, 'result': None, 'unknown_cost': True}
            report['costs']['review_calls'].append(review_entry)
            review_started = runtime.now()
            try:
                slot['review_status'] = 'running'
                screened = content.screen(item['messages'], candidate)
                reviewed = {'screen': screened, 'messages_sha256': None, 'result': None,
                            'status': 'rejected', 'candidate_sha256': _hash(_encode(candidate)),
                            'insertion_sha256': _hash(content.insertion(candidate).encode('utf-8'))}
                if screened['status'] == 'passed':
                    messages = content.build_review_messages(item['messages'], candidate)
                    reviewed['messages_sha256'] = _message_payload(messages)[1]
                    review_policy = plan['writer']['content_review']
                    if remaining() < review_policy['per_call_seconds'] + 150:
                        raise _BudgetStop('GPU_cap')
                    review_model = _BoundModel(runtime.reviewer_model(plan, service,
                        deadline=runtime.now() + review_policy['per_call_seconds']),
                        reviewed['messages_sha256'], guard)
                    runtime.record({'event': 'review_call_intent', 'source_index': item['source_index'],
                                    'messages_sha256': reviewed['messages_sha256'],
                                    'candidate_sha256': reviewed['candidate_sha256'],
                                    'insertion_sha256': reviewed['insertion_sha256'],
                                    'seed': 117 + item['source_index']})
                    review_entry['attempted'] = True
                    review_result = content.review_once(messages, review_model,
                        seed=117 + item['source_index'], record=lambda event, index=item['source_index']:
                        runtime.record({'event': 'review_event', 'source_index': index, 'review_event': event}))
                    reviewed['result'], reviewed['status'] = review_result, review_result['status']
                    _require(review_result['status'] in {
                        'plausible_pending_utility', 'rejected', 'abstained', 'unknown'}
                        and review_result['accepted'] is False and review_result['content_certified'] is False
                        and review_result['training_authorized'] is False)
                    review_entry['unknown_cost'] = review_result['cost']['unknown_cost']
                else:
                    _require(screened['status'] == 'rejected')
                    review_entry['unknown_cost'] = False
                review_entry['result'] = reviewed
                slot['review_status'] = reviewed['status']
                review_binding = runtime.write(f"review-result-{item['source_index']:02d}.json", reviewed)
                runtime.record({'event': 'review_call_finished', 'source_index': item['source_index'],
                                'result_binding': review_binding})
            finally:
                review_entry['wall_seconds'] = runtime.now() - review_started
            if review_entry['wall_seconds'] > plan['writer']['content_review']['per_call_seconds']:
                raise _BudgetStop('review_call_budget')
            if slot['review_status'] == 'unknown' or review_entry['unknown_cost']:
                break
        else:
            report['status'] = 'complete'
    except _BudgetStop as error:
        report['status'] = 'stopped_' + str(error)
    except BaseException:
        report['status'] = 'unknown'
    finally:
        if heartbeat is not None:
            try:
                heartbeat.stop()
            except BaseException:
                report['status'] = 'unknown'
        if enqueued:
            release = {'verified': False, 'GPU_seconds': None}
            try:
                release = runtime.release(job, known_pids)
                _require(release['verified'] is True)
                if release.get('runtime_overrun') is True:
                    report['status'] = 'unknown'
            except BaseException:
                report['status'] = 'unknown'
            finally:
                report['costs']['GPU_jobs'].append({'job': job, 'release': release,
                    'admission_to_release_seconds': runtime.now() - admission_start,
                    'admission_wait_seconds': max(0, (release.get('state', {}).get('job') or {}).get(
                        'started_epoch', runtime.epoch()) - admission_epoch) if release.get('verified') else None,
                    'unknown_cost': release.get('verified') is not True})
        report['costs']['parent_wall_seconds'] = runtime.now() - parent_started
        for slot in report['sources']:
            if slot['status'] == 'running':
                slot['status'] = 'unknown'
            if slot.get('review_status') == 'running':
                slot['review_status'] = 'unknown'
        if begun:
            try:
                runtime.record({'event': 'study_stop', 'report': report})
                runtime.write('study-result.json', report)
            except BaseException:
                report['status'] = 'unknown'
        elif not preflight_complete:
            try:
                runtime.preflight_failure(report)
            except BaseException:
                pass
        runtime.close()
    return report


def launch_writer(plan_sha, *, root=_ROOT):
    """Launch a bounded persistent parent; private qualification precedes enqueue."""
    _require(socket.gethostname() == 'amax' and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    runtime = _Runtime(root)
    plan = registration.load_registration(root, plan_sha, private=False)
    descriptor = runtime.reader._private_directory(Path(plan['private_run_root']))
    try:
        unit = 'wll-native-source-experience-writer-' + runtime.owner() + '.service'
        log = Path(plan['private_run_root']) / 'writer-parent.log'
        with _exclusive(descriptor, log.name):
            pass
        command = unit_command(unit, ['--run', '--plan-sha256', plan_sha], root=root, log_path=log)
        with _exclusive(descriptor, 'parent-launch-intent.json') as file:
            runtime.service._write(file, {'unit': unit, 'argv': command, 'plan_sha256': plan_sha})
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=10, check=False)
        _require(result.returncode == 0)
        spec = {'kind': 'pilot', 'unit': unit, 'command': command}
        runtime.service.units[unit] = spec
        state = runtime.unit_status(spec)
        _require(state['verified'] is True)
        return {'unit': unit, 'plan_sha256': plan_sha,
                'status': 'launch_returned_not_completion', 'main_pid': state['main_pid']}
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument('--run', action='store_true')
    choices.add_argument('--launch', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.launch:
            launch_writer(args.plan_sha256)
            return 0

        def interrupted(_signum, _frame):
            raise NativeExperienceWriterError(_ERROR)

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupted)
        return 0 if run_writer(args.plan_sha256)['status'] == 'complete' else 1
    except BaseException:
        sys.stderr.write(_ERROR + '\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
