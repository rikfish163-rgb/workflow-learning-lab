"""Frozen reviewed memory versus no memory on fresh native train attempts.

This executes the registered V9/27 cohort, not a report-only approximation.
Each arm has a fresh episode process and worker; the trusted scorer starts only
after both have closed. A normal registered actor budget stop with a saved,
closed, known-cost policy receipt and separate boolean score is a known task
failure/success. Incomplete execution/save/score evidence remains incomplete or
unknown and prevents full-cohort estimation. Actor completion is kept separately.
First metadata uses the full remaining 120-second setup budget through controller
and worker006. RPC60 remains the default; explicit actor/replay overrides retain
their existing phase and worker deadlines. Actor and model004 remain unchanged.

The legacy argument named review_hint is used only for the exact frozen ordinary
memory at user-message index 2, with no prefix or replay. There is no source-error
review intervention, bundle rewrite, retry, writer, training, dev or test route.
Import/help start no private preflight, world, model, scorer, queue or GPU work.
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
from dataclasses import asdict
from pathlib import Path
from types import FunctionType, SimpleNamespace

from scripts import native_appworld_actor_004 as actor
from scripts import native_appworld_controller_006 as controller
from scripts import native_experience_utility_001 as utility
from scripts.native_appworld_model_client_004 import NativeModel
from scripts.native_appworld_worker_client import _DIR_FLAGS, _decode, _encode, _exclusive, _private
from scripts.native_appworld_worker_client_006 import NativeWorkerClient
from scripts.verification_token_counter import _message_payload

_ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-native-experience-utility-001'
PLAN = 'artifacts/qwen38-native-experience-utility-plan-001.json'
SOURCE4_SHA256 = '76e3ae6db7607a28744369e3eb3b8d8e4deab5850e98881964147d4e8d3cc6aa'
LIFECYCLE_SHA256 = 'c9cac394a244e8764d3bbd7143d3b44e0db00e20cc5f6016d84504d89bbee7f6'
PARENT_SECONDS = 14 * (86400 + 7200) + 600
MAX_DOCUMENT = 32_000_000
_CORE = None
_ERROR = 'native_experience_utility_coordinator_failed'
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
    from scripts.run_native_experience_utility_001 import _verification_payload,_encode
    result=_encode(_verification_payload(spec))
    if len(result)>32000000: raise ValueError()
    sys.stdout.buffer.write(result)
except BaseException:
    sys.stderr.write('native_experience_utility_verification_failed\n')
    raise SystemExit(2)
'''


class NativeExperienceUtilityError(RuntimeError):
    """Content-free infrastructure failure; private evidence stays private."""


class _BudgetStop(Exception):
    pass


def _require(value):
    if not value:
        raise NativeExperienceUtilityError(_ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _registry():
    return importlib.import_module('scripts.native_experience_utility_registration_001')


def _queue():
    return importlib.import_module('scripts.native_experience_utility_gpu_queue_001')


def _core():
    global _CORE
    if _CORE is None:
        from scripts import run_native_source_pilot_004 as source

        _require(_hash(_registry().public_read(_ROOT, 'scripts/run_native_source_pilot_004.py')) == SOURCE4_SHA256)
        _CORE = source._isolated()
    return _CORE


def _make_job(batch, owner, sha):
    return _queue().make_job(batch, owner, sha)


def _experiment(index, arm, owner):
    return f'native-experience-utility-001-{index:02d}-{arm}-{owner}'


def _reader(prefix, registry, queue):
    source = _core().episode._get_core()
    scope = dict(source.__dict__)
    scope.update(PRIVATE_RUN_ROOT=Path(prefix), registration=registry, queue=queue, REMOTE=queue.REMOTE)
    for name in ('_private_directory', '_read_private', '_backend', '_queue_binding'):
        fn = getattr(source, name)
        scope[name] = FunctionType(fn.__code__, scope, name, fn.__defaults__, fn.__closure__)
    return SimpleNamespace(**scope)


def unit_command(kind, unit, arguments, *, root=_ROOT, log_path):
    _require(kind in {'pilot', 'episode', 'scorer'})
    command = _core().unit_command(kind, unit, arguments, root=root, log_path=log_path)
    if kind == 'scorer':
        return command
    previous = 'scripts.run_native_source_pilot_004' if kind == 'pilot' else 'scripts.native_source_episode_004'
    positions = [i for i, text in enumerate(command) if previous in text]
    _require(len(positions) == 1)
    i = positions[0]
    command[i] = command[i].replace(previous, 'scripts.run_native_experience_utility_001')
    if kind == 'episode':
        command.insert(i + 1, '--episode')
    else:
        old = '--property=RuntimeMaxSec=282600'
        _require(command.count(old) == 1)
        command[command.index(old)] = '--property=RuntimeMaxSec=' + str(PARENT_SECONDS)
    return command


def _services(root):
    registry, queue, core = _registry(), _queue(), _core()
    _require(registry.ID == queue.ID == ID and registry.PLAN == queue.PLAN == PLAN)
    facade = SimpleNamespace(**{**vars(registry), 'static_backend': core.registration.static_backend})
    reader = _reader(registry.PRIVATE_RUN_ROOT, facade, queue)
    raw = registry.public_read(_ROOT, 'scripts/run_native_source_pilot.py')
    _require(_hash(raw) == LIFECYCLE_SHA256)
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == '_Runtime')
    status = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'unit_status')
    constants = [n for n in ast.walk(status) if isinstance(n, ast.Constant) and n.value == 900]
    _require(len(constants) == 1)

    class Duration(ast.NodeTransformer):
        def visit_Constant(self, value):
            if value is constants[0]:
                return ast.IfExp(test=ast.Compare(left=ast.Subscript(value=ast.Name(id='spec', ctx=ast.Load()),
                    slice=ast.Constant(value='kind'), ctx=ast.Load()), ops=[ast.Eq()],
                    comparators=[ast.Constant(value='pilot')]), body=ast.Constant(value=PARENT_SECONDS),
                    orelse=ast.Constant(value=900))
            return value

    node = Duration().visit(node)
    allowed = {'__init__', 'now', 'epoch', 'wait', 'owner', '_write', 'record', 'write', 'read',
               'queue', 'heartbeat', 'monitor', 'port_owners', 'ready', 'launch', 'unit_status',
               'stop_unit', 'artifacts', 'create_score_directory', 'score_result', 'cleanup_worker', 'close'}
    node.body = [fn for fn in node.body if isinstance(fn, ast.FunctionDef) and fn.name in allowed]
    scope = dict(core.__dict__)
    scope.update(registration=facade, queue=queue, episode=reader, unit_command=unit_command)
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, str(_ROOT / 'scripts/run_native_source_pilot.py'), 'exec'), scope)
    return scope['_Runtime'](root), reader


def _validate_bundle(plan, bundle):
    _require(type(bundle) is dict and bundle.get('private') is True
             and bundle.get('status') == 'plausible_pending_utility'
             and bundle.get('source_gate_kind') == 'ordinary_control_recovery_gate')
    for key in ('source_plan_sha256', 'source_result_sha256', 'writer_plan_sha256',
                'writer_result_sha256', 'review_manifest_sha256', 'experience_bundle_id',
                'source_scenario_ids', 'source_gate_kind'):
        _require(bundle[key] == plan['source'][key])
    text = bundle['insertion_text']
    _require(type(text) is str and 1 <= len(text.encode('utf-8')) <= 65536
             and bundle['experience_bundle_id'] == 'native-experience-001:' + _hash(text.encode('utf-8'))
             and type(bundle['private_files_sha256']) is dict and bool(bundle['private_files_sha256']))
    return text


def _verification_payload(spec):
    registry, root, sha = _registry(), Path(spec['root']), spec['plan_sha256']
    if spec['action'] == 'preflight':
        _require(set(spec) == {'action', 'root', 'plan_sha256'})
        plan, bundle = registry.private_preflight(root, sha)
        _require(_encode(registry.load_registration(_queue().REMOTE, sha, private=False)) == _encode(plan))
        _validate_bundle(plan, bundle)
        return {'plan': plan, 'bundle': bundle}
    _require(spec['action'] == 'hash' and set(spec) == {'action', 'root', 'plan_sha256', 'binding'})
    plan = registry.load_registration(root, sha, private=False)
    _service, reader = _services(root)
    binding = spec['binding']
    _require(binding['path'] == str(Path(registry.PRIVATE_RUN_ROOT) / 'reviewed-bundle.json'))
    raw = reader._read_private(binding['path'], maximum=MAX_DOCUMENT)
    _require(_hash(raw) == binding['sha256'])
    bundle = _decode(raw)
    _validate_bundle(plan, bundle)
    general = _reader(registry.PRIVATE_STATE_ROOT,
                      SimpleNamespace(**{**vars(registry), 'static_backend': _core().registration.static_backend}), _queue())
    for path, sha in bundle['private_files_sha256'].items():
        _require(_hash(general._read_private(path, maximum=64_000_000)) == sha)
    return {'bundle': bundle}


class _Runtime:
    def __init__(self, root):
        self.root = Path(root)
        self.service, self.reader = _services(root)
        self.verification_audits = []

    def __getattr__(self, key):
        return getattr(self.service, key)

    def _verify(self, spec, *, gpu_remaining=None):
        registry = _registry()
        timeout = 180 if gpu_remaining is None else min(180, gpu_remaining - 150)
        _require(timeout > 0)
        spec = {**spec, 'root': str(self.root)}
        payload = _encode(spec)
        _require(len(payload) <= 1_000_000)
        env = {key: os.environ[key] for key in (
            'PATH', 'HOME', 'USER', 'LOGNAME', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS') if key in os.environ}
        env.update(CUDA_VISIBLE_DEVICES='', TMPDIR=registry.TMPDIR, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONHASHSEED='17', LANG='C.UTF-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', TOKENIZERS_PARALLELISM='false')
        audit = {'stage': spec['action'], 'timeout_seconds': timeout, 'wall_seconds': None,
                 'returned': False, 'child_returncode': None}
        started, stdout, stderr = self.now(), None, None
        try:
            child = subprocess.run([registry.PYTHON, '-s', '-P', '-B', '-c', _VERIFY_CODE],
                input=payload, capture_output=True, timeout=timeout, check=False, env=env)
            stdout, stderr, audit['child_returncode'] = child.stdout, child.stderr, child.returncode
            _require(child.returncode == 0 and type(stdout) is bytes and len(stdout) <= MAX_DOCUMENT
                     and self.now() - started <= timeout)
            result = _decode(stdout)
            audit['returned'] = True
            return result
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.output, error.stderr
            raise NativeExperienceUtilityError(_ERROR) from None
        except Exception:
            raise NativeExperienceUtilityError(_ERROR) from None
        finally:
            audit['wall_seconds'] = self.now() - started
            for name, value in (('stdout', stdout), ('stderr', stderr)):
                if type(value) is bytes:
                    audit[name + '_bytes'], audit[name + '_sha256'] = len(value), _hash(value)
            self.verification_audits.append(audit)

    def preflight(self, sha):
        _require(socket.gethostname() == 'amax' and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
        result = self._verify({'action': 'preflight', 'plan_sha256': sha})
        self.service.plan = result['plan']
        return result['plan'], result['bundle']

    def begin(self, plan, sha, rows, bundle):
        self.service.private = Path(plan['private_run_root'])
        self.service.fd = self.reader._private_directory(self.service.private)
        with _exclusive(self.service.fd, 'study-intent.json') as file:
            self.service._write(file, {'plan_sha256': sha, 'summaries': rows, 'automatic_retry': False,
                'reviewed_bundle_sha256': _hash(_encode(bundle) + b'\n'), 'pid': os.getpid(),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
        self.service.journal = _exclusive(self.service.fd, 'study-journal.jsonl')
        os.fsync(self.service.fd)

    def check_bundle(self, sha, binding, *, gpu_remaining):
        return self._verify({'action': 'hash', 'plan_sha256': sha, 'binding': binding},
                            gpu_remaining=gpu_remaining)['bundle']

    def binding(self, job, service):
        state, _ready, identity = self.reader._queue_binding(
            {'job': job, 'expected_backend': service['backend']}, _queue().queue_operation)
        _require(identity == service['identity'])
        return state

    def release(self, job, pids):
        return _queue().fenced_release(self.service, job, pids)

    def preflight_failure(self, report):
        fd = self.reader._private_directory(Path(_registry().PRIVATE_RUN_ROOT))
        try:
            with _exclusive(fd, 'preflight-failure.json') as file:
                self.service._write(file, report)
            os.fsync(fd)
        finally:
            os.close(fd)


def _limits(policy):
    return actor.ActorLimits(max_generations=policy['max_generations'], max_actions=policy['max_actions'],
        max_api_requests=policy['max_api_requests'], max_input_tokens=policy['max_input_tokens'],
        max_new_tokens=policy['max_new_tokens'], max_prompt_tokens_total=policy['max_prompt_tokens_total'],
        max_completion_tokens_total=policy['max_completion_tokens_total'], wall_seconds=policy['actor_seconds'])


class _BoundModel:
    def __init__(self, model, check):
        self.model, self.check, self.max_tokens = model, check, model.max_tokens

    @property
    def deadline(self):
        return self.model.deadline

    @deadline.setter
    def deadline(self, value):
        self.model.deadline = value

    def check_health(self):
        self.check()
        return self.model.check_health()

    def count_input_tokens(self, messages):
        self.check()
        return self.model.count_input_tokens(messages)

    def generate(self, messages, seed):
        self.check()
        return self.model.generate(messages, seed=seed)


def _create_episode_directory(reader, private_root, run_dir):
    """Create exactly one private unit directory and verify its live identity."""
    parent_fd = reader._private_directory(private_root)
    run_fd = current_fd = None
    try:
        name = Path(run_dir).name
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        run_fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
        opened = _private(run_fd, directory=True)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        current_fd = reader._private_directory(Path(run_dir))
        current = _private(current_fd, directory=True)
        _require((opened.st_dev, opened.st_ino) == (linked.st_dev, linked.st_ino)
                 == (current.st_dev, current.st_ino))
    finally:
        if current_fd is not None:
            os.close(current_fd)
        if run_fd is not None:
            os.close(run_fd)
        os.close(parent_fd)


def run_episode(config_path, config_sha256, *, root=_ROOT, _load_plan=None, _controller=None,
                _model_factory=None, _worker_factory=None, _queue_status=None):
    """One separately isolated clean first attempt; never loads ground truth."""
    registry = _registry()
    _service, reader = _services(root)
    raw = reader._read_private(config_path, maximum=MAX_DOCUMENT)
    _require(_hash(raw) == config_sha256)
    config = _decode(raw)
    _require(set(config) == {'plan_sha256', 'pair_index', 'arm', 'run_dir', 'job',
                              'expected_backend', 'bundle_binding'})
    plan = (_load_plan or registry.load_registration)(root, config['plan_sha256'], private=False)
    index, arm, job = config['pair_index'], config['arm'], config['job']
    _require(type(index) is int and 0 <= index < 27 and arm in {'control', 'experience'}
             and job == _queue().make_job(index // 2, job['owner'], config['plan_sha256'])
             and config['run_dir'] == str(Path(registry.PRIVATE_RUN_ROOT) / f'unit-{index:02d}-{arm}-{job["owner"]}'))
    binding = config['bundle_binding']
    _require(binding['path'] == str(Path(registry.PRIVATE_RUN_ROOT) / 'reviewed-bundle.json'))
    raw = reader._read_private(binding['path'], maximum=MAX_DOCUMENT)
    _require(_hash(raw) == binding['sha256'])
    bundle = _decode(raw)
    text = _validate_bundle(plan, bundle)
    intent = _decode(reader._read_private(Path(registry.PRIVATE_RUN_ROOT) / 'study-intent.json', maximum=MAX_DOCUMENT))
    _require(intent['plan_sha256'] == config['plan_sha256'] and intent['reviewed_bundle_sha256'] == binding['sha256'])
    reader._backend(plan, config['expected_backend'])
    initial = reader._queue_binding(config, _queue_status or _queue().queue_operation)[2]
    _create_episode_directory(reader, Path(registry.PRIVATE_RUN_ROOT), config['run_dir'])

    def check():
        _require(reader._queue_binding(config, _queue_status or _queue().queue_operation)[2] == initial)

    factory = _model_factory or NativeModel
    model = factory(f'http://127.0.0.1:{registry.PORT}/v1', plan['model_id'],
        expected_backend=config['expected_backend'], max_tokens=1024, context_tokens=16384,
        temperature=0.7, timeout=120, deadline=_service.now() + 780)
    model = _BoundModel(model, check)
    task_id, _scenario, seed = plan['expected_pairs'][index]
    experiment = _experiment(index, arm, job['owner'])

    def worker(task, exp, path, **kwargs):
        _require(task == task_id and exp == experiment and Path(path) == Path(config['run_dir']) / 'worker'
                 and kwargs == {'max_actions': 60, 'max_api_requests': 60, 'rpc_timeout': 60, 'wall_seconds': 780})
        return (_worker_factory or NativeWorkerClient)(task, exp, path, **kwargs)

    result = (_controller or controller.run_task)(task_id, experiment, config['run_dir'],
        limits=_limits(plan['utility']), seed=seed, model=model, token_counter=model.count_input_tokens,
        rpc_timeout=60, worker_wall_seconds=780, setup_seconds=120, cleanup_seconds=60,
        is_policy_episode=True, source_metadata=None, prefix_records=None,
        review_hint=text if arm == 'experience' else None, _worker_factory=worker)
    _require(type(result) is dict and result.get('task_id') == task_id)
    return result


def _summaries(plan):
    return [{'task_id': task, 'scenario_id': scene, 'seed': seed, 'arm': arm, 'split': 'train',
        'status': 'unstarted', 'closed': False, 'branch': 'first', 'replayed_steps': 0, 'success': None,
        'outcome_unknown': False, 'unknown_cost': False, 'is_policy_episode': False,
        'worker_exit_code': None, 'score_status': 'not_scored', 'identity': dict(plan['identity']),
        'experience_bundle_id': plan['source']['experience_bundle_id'] if arm == 'experience' else None}
        for task, scene, seed in plan['expected_pairs'] for arm in ('control', 'experience')]


def _run_arm(runtime, plan, sha, bundle, job, service, index, arm, row, guard, report, expected_metadata):
    run_dir = Path(plan['private_run_root']) / f'unit-{index:02d}-{arm}-{job["owner"]}'
    experiment, task = _experiment(index, arm, job['owner']), plan['expected_pairs'][index][0]
    config = {'plan_sha256': sha, 'pair_index': index, 'arm': arm, 'run_dir': str(run_dir),
              'job': job, 'expected_backend': service['backend'], 'bundle_binding': bundle}
    binding = runtime.write(f'config-{index:02d}-{arm}.json', config)
    unit = f'wll-native-source-utility-{index:02d}-{arm}-{job["owner"]}.service'
    parent, started, charged = None, runtime.now(), False
    cost = {'pair_index': index, 'arm': arm, 'parent_wall_seconds': None, 'unknown_cost': True}
    try:
        parent, _wall = _core()._unit(runtime, 'episode', unit,
            ['--config', binding['path'], '--config-sha256', binding['sha256']], guard=guard)
        evidence = runtime.artifacts(run_dir)
        receipt, journal = evidence['receipt'], evidence['journal']
        cost.update(_core()._costs(receipt, journal))
        report['costs']['episodes'].append(cost)
        charged = True
        _require(receipt.get('status') == 'saved_and_closed' and receipt.get('task_id') == task
                 and receipt.get('experiment_name') == experiment and receipt.get('is_policy_episode') is True
                 and type(receipt.get('worker_exit_code')) is int and receipt['worker_exit_code'] == 0
                 and receipt.get('replay_outcome') is None and cost['counts_complete'] is True)
        intent = _core()._one(journal, 'controller_intent')
        _require(intent['seed'] == 17 and intent['limits'] == asdict(_limits(plan['utility']))
                 and intent['review_hint_present'] is (arm == 'experience'))
        metadata = _core()._one(journal, 'metadata_result')['reply']['result']
        metadata_sha = _hash(_encode(metadata))
        _require(expected_metadata is None or metadata_sha == expected_metadata)
        memory = report['_insertion_text'] if arm == 'experience' else None
        messages = actor.build_actor_messages(metadata, review_hint=memory)
        context = {'metadata_sha256': metadata_sha, 'messages_sha256': _message_payload(messages)[1]}
        _core().actor_evidence(receipt, journal, expected_context=context, review_hint=memory)
        _require(_core()._one(journal, 'actor_start')['seed'] == 17)
        _require(runtime.cleanup_worker(run_dir, task, experiment)['verified'] is True)
        _require(runtime.unit_status(parent)['main_pid'] == 0)
        row.update(closed=True, is_policy_episode=True, worker_exit_code=0)
        score_dir = Path(plan['private_run_root']) / f'score-{index:02d}-{arm}-{job["owner"]}'
        runtime.create_score_directory(score_dir)
        score_unit = f'wll-native-source-utility-{index:02d}-score-{arm}-{job["owner"]}.service'
        score_parent, score_started = None, runtime.now()
        score_cost = {'pair_index': index, 'arm': arm, 'attempted': True, 'unknown_cost': True, 'wall_seconds': None}
        report['costs']['scorers'].append(score_cost)
        try:
            score_parent, _ = _core()._unit(runtime, 'scorer', score_unit,
                ['--receipt', evidence['receipt_binding']['path'], '--receipt-sha256', evidence['receipt_binding']['sha256'],
                 '--out-dir', str(score_dir), '--parent-unit-name', parent['unit']], guard=guard)
            score = runtime.score_result(score_dir)
            _require(score['status'] == 'scored' and type(score['success']) is bool
                     and score['is_policy_episode'] is True
                     and score['receipt_sha256'] == evidence['receipt_binding']['sha256']
                     and score['manifest_sha256'] == receipt['snapshot']['manifest_sha256'])
            score_cost['unknown_cost'] = False
        finally:
            score_cost['wall_seconds'] = runtime.now() - score_started
            if score_parent is not None:
                _require(runtime.stop_unit(score_parent)['verified'] is True)
        row.update(status='known', score_status='scored', success=score['success'],
                   outcome_unknown=False, unknown_cost=False)
        execution = {'pair_index': index, 'arm': arm, 'receipt_binding': evidence['receipt_binding'],
                     'metadata_sha256': metadata_sha,
                     'actor_status': receipt['actor_outcome']['status'],
                     'model_reported_complete': receipt['actor_outcome']['status'] == 'model_reported_complete'}
        report['executions'].append(execution)
        return metadata_sha
    except BaseException:
        row.update(status='unknown', outcome_unknown=True, unknown_cost=True,
                   score_status='unknown' if row['closed'] else 'not_scored', success=None)
        raise
    finally:
        if not charged:
            try:
                partial = runtime.artifacts(run_dir)
                cost.update(_core()._costs(partial['receipt'], partial['journal']))
            except Exception:
                receipt, journal = {}, []
                try:
                    raw = runtime.read(run_dir / 'controller.jsonl')
                    journal = _core()._journal(raw[:raw.rfind(b'\n') + 1])
                except Exception:
                    pass
                try:
                    receipt = _decode(runtime.read(run_dir / 'execution-receipt.json'))
                except Exception:
                    pass
                cost.update(_core()._costs(receipt, journal))
            cost.update(unknown_cost=True, counts_complete=False)
            report['costs']['episodes'].append(cost)
        try:
            _require(runtime.cleanup_worker(run_dir, task, experiment)['verified'] is True)
        finally:
            if parent is not None:
                _require(runtime.stop_unit(parent)['verified'] is True)
            cost['parent_wall_seconds'] = runtime.now() - started


def _registered(plan, bundle):
    reg = _registry()
    _require(plan['id'] == ID and plan['status'] == 'registered' and plan['utility'] == reg.utility_policy()
             and plan['automatic_retry'] is False and plan['utility_authorized'] is True
             and all(plan[key] is False for key in ('writer_authorized', 'training_authorized', 'dev_or_test_authorized')))
    text = _validate_bundle(plan, bundle)
    pairs = plan['expected_pairs']
    _require(len(pairs) == 27 and len({p[0] for p in pairs}) == 27
             and len({p[1] for p in pairs}) == 9 and all(p[2] == 17 for p in pairs)
             and plan['batches'] == [list(range(i, min(i + 2, 27))) for i in range(0, 27, 2)]
             and plan['branch_orders'] == [['control', 'experience'] if i % 2 == 0 else ['experience', 'control'] for i in range(27)])
    utility.evaluate_utility(pairs, [], identity=plan['identity'], source_scenario_ids=bundle['source_scenario_ids'],
                             experience_bundle_id=bundle['experience_bundle_id'])
    return text


def run_utility(plan_sha, root=_ROOT, _runtime=None):
    runtime = _runtime or _Runtime(root)
    report = {'id': ID, 'plan_sha256': plan_sha, 'status': 'unknown', 'summaries': [], 'executions': [],
              'planned_pairs': 27, 'planned_summaries': 54,
              'utility': None, 'costs': {'episodes': [], 'scorers': [], 'GPU_jobs': [],
                                       'source_verifications': runtime.verification_audits},
              'automatic_retry': False, 'writer_calls': 0, 'training_jobs': 0, 'dev_or_test_calls': 0}
    begun, preflight_done, plan = False, False, None
    parent_started = runtime.now()
    try:
        plan, reviewed = runtime.preflight(plan_sha)
        report['_insertion_text'] = _registered(plan, reviewed)
        report['summaries'] = _summaries(plan)
        preflight_done = True
        runtime.begin(plan, plan_sha, report['summaries'], reviewed)
        begun = True
        bundle = runtime.write('reviewed-bundle.json', reviewed)
        for batch_index, indices in enumerate(plan['batches']):
            heartbeat, enqueued, pids = None, False, []
            job = _make_job(batch_index, runtime.owner(), plan_sha)
            started, admission_epoch = runtime.now(), runtime.epoch()
            try:
                runtime.monitor(empty_port=True)
                runtime.record({'event': 'GPU_enqueue_intent', 'job': job})
                enqueued = True
                state = runtime.queue('enqueue', job)
                heartbeat = runtime.heartbeat(job)
                service = None
                while service is None:
                    heartbeat.check()
                    runtime.monitor()
                    _core()._queue_state(state, runtime)
                    _require(state.get('released') is False)
                    info = state.get('job')
                    if runtime.now() - started >= 86400:
                        raise _BudgetStop('admission_timeout')
                    if type(info) is dict and info.get('status') == 'running' and runtime.epoch() - info['started_epoch'] >= 600:
                        raise _BudgetStop('model_ready_budget')
                    service = runtime.ready(job, state)
                    if type(info) is dict and info.get('status') == 'running' and runtime.epoch() - info['started_epoch'] >= 600:
                        raise _BudgetStop('model_ready_budget')
                    if service is None:
                        runtime.wait(5)
                        state = runtime.queue('pulse', job)
                pids, gpu_started = list(state['model_pids']), state['job']['started_epoch']
                runtime.record({'event': 'GPU_ready', 'job': job, 'service': service})

                def remaining(gpu_started=gpu_started):
                    return 7200 - (runtime.epoch() - gpu_started)

                def guard(heartbeat=heartbeat, job=job, service=service, remaining=remaining):
                    heartbeat.check()
                    if remaining() <= 150:
                        raise _BudgetStop('GPU_cap')
                    latest = runtime.binding(job, service)
                    _core()._queue_state(latest, runtime)
                    _require(latest.get('released') is False and latest.get('job', {}).get('status') == 'running')

                for index in indices:
                    guard()
                    if remaining() < 2 * (900 + 120 + 180) + 150:
                        raise _BudgetStop('GPU_cap')
                    metadata = None
                    for arm in plan['branch_orders'][index]:
                        guard()
                        checked = runtime.check_bundle(plan_sha, bundle, gpu_remaining=remaining())
                        _require(_encode(checked) == _encode(reviewed))
                        guard()
                        row = next(v for v in report['summaries'] if v['task_id'] == plan['expected_pairs'][index][0] and v['arm'] == arm)
                        metadata = _run_arm(runtime, plan, plan_sha, bundle, job, service, index,
                                            arm, row, guard, report, metadata)
            finally:
                heartbeat_failure = None
                if heartbeat is not None:
                    try:
                        heartbeat.stop()
                    except BaseException as error:
                        report['status'] = 'unknown'
                        heartbeat_failure = error
                if enqueued:
                    release = {'verified': False, 'GPU_seconds': None}
                    try:
                        release = runtime.release(job, pids)
                        _require(release['verified'] is True and release.get('runtime_overrun') is not True)
                    finally:
                        report['costs']['GPU_jobs'].append({'job': job, 'release': release,
                            'admission_to_release_seconds': runtime.now() - started,
                            'admission_wait_seconds': max(0, (release.get('state', {}).get('job') or {}).get(
                                'started_epoch', runtime.epoch()) - admission_epoch) if release.get('verified') else None,
                            'unknown_cost': release.get('verified') is not True})
                if heartbeat_failure is not None:
                    raise heartbeat_failure
        report['status'] = 'complete'
    except _BudgetStop as error:
        report['status'] = 'stopped_' + str(error)
    except BaseException:
        report['status'] = 'unknown'
    finally:
        report.pop('_insertion_text', None)
        if preflight_done:
            validation_started = runtime.now()
            try:
                report['utility'] = utility.evaluate_utility(plan['expected_pairs'], report['summaries'],
                    identity=plan['identity'], source_scenario_ids=reviewed['source_scenario_ids'],
                    experience_bundle_id=reviewed['experience_bundle_id'])
                costs = report['costs']
                verified = (report['status'] == 'complete' and len(costs['GPU_jobs']) == 14
                    and len(costs['episodes']) == len(costs['scorers']) == 54
                    and all(v.get('unknown_cost') is False and v.get('counts_complete') is True
                            for v in costs['episodes'])
                    and all(v.get('unknown_cost') is False for v in costs['scorers'])
                    and all(v.get('unknown_cost') is False and v['release'].get('verified') is True
                            and v['release'].get('runtime_overrun') is False for v in costs['GPU_jobs']))
                report['utility']['global_execution_and_costs_verified'] = verified
                if not verified:
                    # Keep every actual task score and the complete denominator.
                    # An unfinished lifecycle cannot certify an effect or gain.
                    report['utility'].update(status='not_evaluable',
                        reason='global_execution_or_cost_evidence_incomplete',
                        passed=None, checks=None, statistics=None)
            except Exception:
                report['status'] = 'unknown'
            finally:
                report['costs']['utility_validation_wall_seconds'] = runtime.now() - validation_started
        report['costs']['parent_wall_seconds'] = runtime.now() - parent_started
        if begun:
            try:
                runtime.record({'event': 'study_stop', 'report': report})
                runtime.write('study-result.json', report)
            except BaseException:
                report['status'] = 'unknown'
                if report['utility'] is not None:
                    report['utility'].update(status='not_evaluable',
                        reason='study_persistence_incomplete', passed=None, checks=None, statistics=None,
                        global_execution_and_costs_verified=False)
        elif not preflight_done:
            try:
                runtime.preflight_failure(report)
            except BaseException:
                pass
        runtime.close()
    return report


def launch_utility(plan_sha, *, root=_ROOT):
    registry = _registry()
    _require(socket.gethostname() == 'amax' and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    runtime = _Runtime(root)
    plan = registry.load_registration(root, plan_sha, private=False)
    fd = runtime.reader._private_directory(Path(plan['private_run_root']))
    try:
        unit = 'wll-native-source-experience-utility-' + runtime.owner() + '.service'
        log = Path(plan['private_run_root']) / 'utility-parent.log'
        with _exclusive(fd, log.name):
            pass
        command = unit_command('pilot', unit, ['--run', '--plan-sha256', plan_sha], root=root, log_path=log)
        with _exclusive(fd, 'parent-launch-intent.json') as file:
            runtime.service._write(file, {'unit': unit, 'plan_sha256': plan_sha, 'argv': command})
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=10, check=False)
        _require(result.returncode == 0)
        spec = {'kind': 'pilot', 'unit': unit, 'command': command}
        runtime.service.units[unit] = spec
        status = runtime.unit_status(spec)
        _require(status['verified'] is True)
        return {'unit': unit, 'main_pid': status['main_pid'], 'status': 'launch_returned_not_completion'}
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--config')
    parser.add_argument('--config-sha256')
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--run', action='store_true')
    operation.add_argument('--launch', action='store_true')
    operation.add_argument('--episode', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.episode:
            _require(args.config is not None and args.config_sha256 is not None)
            return 0 if run_episode(args.config, args.config_sha256)['status'] == 'saved_and_closed' else 1
        _require(args.plan_sha256 is not None)
        if args.launch:
            launch_utility(args.plan_sha256)
            return 0

        def interrupted(_signum, _frame):
            raise NativeExperienceUtilityError(_ERROR)

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupted)
        return 0 if run_utility(args.plan_sha256)['status'] == 'complete' else 1
    except BaseException:
        sys.stderr.write(_ERROR + '\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
