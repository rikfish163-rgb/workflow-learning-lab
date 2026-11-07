"""Synthetic lifecycle only; no real Task, private history, worker or GPU."""

import copy
import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import native_experience_utility_registration_001 as registration
from scripts import run_native_experience_utility_001 as runner
from scripts.native_appworld_worker_client import _decode, _encode

SHA = 'a' * 64
TEXT = 'When a write is unconfirmed, read current state before choosing a correction.'


def plan_and_bundle():
    pairs = [[f'{scene:07x}_{version}', f'{scene:07x}', 17]
             for scene in range(1, 10) for version in (1, 2, 3)]
    bundle_id = 'native-experience-001:' + runner._hash(TEXT.encode())
    origins = [f'{scene:07x}' for scene in range(100, 112)]
    source = {'source_plan_sha256': 'b' * 64, 'source_result_sha256': 'c' * 64,
              'writer_plan_sha256': 'd' * 64, 'writer_result_sha256': 'e' * 64,
              'review_manifest_sha256': 'f' * 64, 'experience_bundle_id': bundle_id,
              'source_scenario_ids': origins, 'source_gate_kind': 'ordinary_control_recovery_gate'}
    bundle = {'private': True, 'status': 'plausible_pending_utility', **source,
              'insertion_text': TEXT, 'private_files_sha256': {'/synthetic/private/review.json': 'f' * 64}}
    plan = {'id': runner.ID, 'status': 'registered', 'utility': registration.utility_policy(),
            'expected_pairs': pairs, 'batches': [list(range(i, min(i + 2, 27))) for i in range(0, 27, 2)],
            'branch_orders': [['control', 'experience'] if i % 2 == 0 else ['experience', 'control'] for i in range(27)],
            'identity': {'model_id': registration.MODEL_ID, 'scaffold_id': 'native004-memory-slot2',
                         'backend_id': 'synthetic-B3', 'budget_id': 'fixed-600-60'},
            'source': source, 'model_arm': 'B3', 'model_id': registration.MODEL_ID,
            'private_run_root': '/synthetic/private', 'automatic_retry': False,
            'utility_authorized': True, 'writer_authorized': False, 'training_authorized': False,
            'dev_or_test_authorized': False, 'admission_wait_seconds': 86400,
            'gpu_seconds': 7200, 'total_gpu_seconds': 100800}
    return plan, bundle


class Runtime:
    def __init__(self, *, failure=None, budget_stop=False):
        self.plan, self.bundle = plan_and_bundle()
        self.failure, self.budget_stop = failure, budget_stop
        self.clock, self.calls, self.files, self.units, self.evidence = 0.0, [], {}, {}, {}
        self.verification_audits, self.begun = [], False
        spec = importlib.util.spec_from_file_location('_utility_fixture', runner._ROOT / 'tests/test_run_native_source_pilot.py')
        self.fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fixture)
        self.fixture.registration = SimpleNamespace(
            TASK_IDS=[row[0] for row in self.plan['expected_pairs']], MODEL_ID=registration.MODEL_ID)

    def now(self):
        return self.clock

    def epoch(self):
        return 10000 + self.clock

    def owner(self):
        return f'{self.batch + 1:032x}' if hasattr(self, 'batch') else '1' * 32

    def wait(self, seconds):
        self.clock += seconds

    def preflight(self, sha):
        self.calls.append(('preflight', sha))
        self.verification_audits.append({'stage': 'preflight', 'wall_seconds': 1})
        if self.failure == 'preflight':
            raise ValueError('private synthetic qualification failure')
        return copy.deepcopy(self.plan), copy.deepcopy(self.bundle)

    def begin(self, plan, sha, rows, bundle):
        if self.begun:
            raise FileExistsError('exclusive intent')
        self.begun = True

    def record(self, event):
        self.calls.append(('record', copy.deepcopy(event)))
        if self.failure == 'study_stop' and event.get('event') == 'study_stop':
            raise OSError('private synthetic study journal failure')

    def write(self, name, value):
        path = '/synthetic/private/' + name
        if path in self.files:
            raise FileExistsError('exclusive output')
        self.files[path] = _encode(value) + b'\n'
        return {'path': path, 'sha256': runner._hash(self.files[path])}

    def read(self, path):
        return self.files[str(path)]

    def check_bundle(self, sha, binding, *, gpu_remaining):
        self.calls.append(('verify_bundle',))
        self.verification_audits.append({'stage': 'hash', 'wall_seconds': 0.1})
        if self.failure == 'bundle_changed':
            raise ValueError('private synthetic changed bundle')
        return copy.deepcopy(self.bundle)

    def monitor(self, *, empty_port=False):
        self.calls.append(('monitor', empty_port))

    def queue(self, operation, job, known_pids=()):
        self.calls.append(('queue', operation, job['batch_index']))
        if operation == 'enqueue':
            self.batch, self.started = job['batch_index'], self.epoch()
        return {'owned': True, 'monitor_live': True, 'epoch': self.epoch(), 'released': False,
                'admission_cancelled': False, 'model_pids': [123],
                'job': {'status': 'running', 'started_epoch': self.started}}

    def heartbeat(self, job):
        def stop():
            if self.failure == 'heartbeat_stop' or self.failure == 'heartbeat_stop_last' and self.batch == 13:
                raise RuntimeError('private synthetic heartbeat failure')

        return SimpleNamespace(check=lambda: None, stop=stop)

    def ready(self, job, state):
        return {'backend': {'synthetic': True}, 'identity': ('synthetic',)}

    def binding(self, job, service):
        return self.queue('status', job)

    def release(self, job, known_pids):
        self.calls.append(('release', job['batch_index']))
        if self.failure == 'release' or self.failure == 'release_last' and self.batch == 13:
            raise RuntimeError('private synthetic release failure')
        return {'verified': True, 'GPU_seconds': self.epoch() - self.started,
                'runtime_overrun': False, 'state': {'job': {'started_epoch': self.started}}}

    def launch(self, kind, unit, arguments, *, log_name):
        self.calls.append(('launch', kind, unit, list(arguments)))
        result = {'kind': kind, 'unit': unit, 'command': []}
        self.units[unit] = result
        if kind == 'episode':
            config = _decode(self.read(arguments[arguments.index('--config') + 1]))
            index, arm = config['pair_index'], config['arm']
            task = self.plan['expected_pairs'][index][0]
            metadata = {'instruction': 'Synthetic ' + task, 'supervisor': {},
                        'app_descriptions': {'synthetic': 'CPU fixture public API'}}
            context = {'metadata_sha256': self.fixture._digest(metadata)}
            receipt, journal = self.fixture._fixture_episode(index, 'first_attempt', context, None, no_boundary=True)
            receipt['experiment_name'] = runner._experiment(index, arm, config['job']['owner'])
            for row in journal:
                if row['event'] in {'actor_start', 'generation_intent'} and arm == 'experience':
                    row['messages'].insert(2, {'role': 'user', 'content': TEXT})
                if row['event'] == 'actor_start':
                    row['seed'], row['review_hint_index'] = 17, 2 if arm == 'experience' else None
                if row['event'] == 'generation_intent':
                    row['max_new_tokens'] = 1024
            if self.budget_stop:
                outcome = receipt['actor_outcome']
                outcome.update(status='incomplete', stop_reason='wall_seconds')
                outcome['cost'] = dict.fromkeys(outcome['cost'], 0)
                journal = [v for v in journal if v['event'] in {'metadata_result', 'actor_start'}]
                journal.append({'event': 'actor_stop', 'outcome': copy.deepcopy(outcome)})
            controller = {'event': 'controller_intent', 'task_id': task,
                          'experiment_name': receipt['experiment_name'], 'seed': 17,
                          'limits': runner._limits(self.plan['utility']).__dict__,
                          'review_hint_present': arm == 'experience', 'is_policy_episode': True}
            journal.insert(0, controller)
            journal.extend([{'event': 'actor_result', 'outcome': copy.deepcopy(receipt['actor_outcome'])},
                            {'event': 'controller_stop', 'result': copy.deepcopy(receipt)}])
            receipt['snapshot'] = {'manifest_sha256': 'e' * 64}
            if self.failure == 'episode':
                receipt['status'] = 'unknown'
                receipt['actor_outcome']['unknown_cost'] = True
            self.evidence[config['run_dir']] = {'receipt': receipt, 'journal': journal,
                'receipt_binding': {'path': config['run_dir'] + '/execution-receipt.json',
                                    'sha256': self.fixture._digest(receipt)}}
            result.update(index=index, arm=arm)
        return result

    def unit_status(self, spec):
        failed = self.failure == 'episode' and spec['kind'] == 'episode'
        return {'verified': True, 'running': False, 'main_pid': 0, 'exit_code': int(failed), 'success': not failed}

    def stop_unit(self, spec):
        self.calls.append(('stop', spec['unit']))
        return {'verified': True}

    def artifacts(self, path):
        return copy.deepcopy(self.evidence[str(path)])

    def cleanup_worker(self, run_dir, task_id, experiment):
        self.calls.append(('worker_closed', str(run_dir)))
        return {'verified': True}

    def create_score_directory(self, path):
        self.calls.append(('score_dir', str(path)))

    def score_result(self, path):
        arm = Path(path).name.split('-')[2]
        run = '/synthetic/private/' + Path(path).name.replace('score-', 'unit-', 1)
        if self.failure == 'scorer':
            raise RuntimeError('private synthetic scorer failure')
        return {'status': 'scored', 'success': arm == 'experience' and not self.budget_stop,
                'receipt_sha256': self.evidence[run]['receipt_binding']['sha256'],
                'manifest_sha256': 'e' * 64, 'is_policy_episode': True}

    def preflight_failure(self, report):
        self.calls.append(('preflight_failed',))

    def close(self):
        self.calls.append(('close',))


@pytest.fixture(autouse=True)
def fake_jobs(monkeypatch):
    monkeypatch.setattr(runner, '_make_job', lambda batch, owner, sha: {
        'id': runner.ID + '-' + str(batch), 'batch_index': batch, 'owner': owner})


def test_full_actual_execution_schedule_uses_54_fresh_parents_then_separate_scorers():
    runtime = Runtime()
    result = runner.run_utility(SHA, _runtime=runtime)
    assert result['status'] == 'complete'
    assert result['utility']['planned_pairs'] == result['utility']['complete_pairs'] == 27
    assert result['utility']['planned_scenarios'] == 9 and result['utility']['passed'] is True
    assert len([v for v in runtime.calls if v[:2] == ('launch', 'episode')]) == 54
    assert len([v for v in runtime.calls if v[:2] == ('launch', 'scorer')]) == 54
    assert len([v for v in runtime.calls if v[0] == 'release']) == 14
    for i, call in enumerate(runtime.calls):
        if call[:2] == ('launch', 'scorer'):
            args = call[3]
            parent = args[args.index('--parent-unit-name') + 1]
            receipt = args[args.index('--receipt') + 1]
            assert runtime.units[parent]['kind'] == 'episode'
            assert ('worker_closed', str(Path(receipt).parent)) in runtime.calls[:i]
    configs = [_decode(raw) for path, raw in runtime.files.items() if '/config-' in path]
    assert [v['arm'] for v in configs] == [arm for order in runtime.plan['branch_orders'] for arm in order]
    assert len({v['run_dir'] for v in configs}) == 54
    for config in configs:
        evidence = runtime.evidence[config['run_dir']]
        start = runner._core()._one(evidence['journal'], 'actor_start')['messages']
        assert start[2:] == ([{'role': 'user', 'content': TEXT}] if config['arm'] == 'experience' else [])
    assert runtime.bundle['insertion_text'] == TEXT
    assert result['training_jobs'] == result['writer_calls'] == result['dev_or_test_calls'] == 0


@pytest.mark.parametrize('failure', ['preflight', 'bundle_changed', 'episode', 'scorer', 'heartbeat_stop', 'release'])
def test_unknown_never_becomes_false_or_subset_estimate_and_preserves_cleanup(failure):
    runtime = Runtime(failure=failure)
    result = runner.run_utility(SHA, _runtime=runtime)
    assert result['status'] == 'unknown'
    assert result['utility'] is None or result['utility']['status'] == 'not_evaluable'
    if result['utility'] is not None:
        assert result['utility']['planned_summaries'] == 54 and result['utility']['statistics'] is None
    assert len([v for v in runtime.calls if v[0] == 'release']) == (failure != 'preflight')
    if failure in {'episode', 'scorer'}:
        assert result['costs']['episodes'] and result['costs']['episodes'][0]['parent_wall_seconds'] is not None
    assert 'private synthetic' not in str(result)


def test_closed_scored_budget_stops_are_known_failures_on_full_denominator():
    result = runner.run_utility(SHA, _runtime=Runtime(budget_stop=True))
    assert result['status'] == 'complete' and result['utility']['complete_pairs'] == 27
    assert all(v['status'] == 'known' and v['success'] is False for v in result['summaries'])
    assert all(not v['model_reported_complete'] for v in result['executions'])


@pytest.mark.parametrize('failure', ['release_last', 'heartbeat_stop_last', 'study_stop'])
def test_last_cleanup_unknown_blocks_positive_effect_without_relabelling_known_scores(failure):
    result = runner.run_utility(SHA, _runtime=Runtime(failure=failure))
    assert result['status'] == 'unknown'
    assert len(result['summaries']) == 54
    assert all(row['status'] == 'known' and type(row['success']) is bool for row in result['summaries'])
    assert result['utility']['planned_pairs'] == result['utility']['complete_pairs'] == 27
    assert result['utility']['status'] == 'not_evaluable'
    assert result['utility']['passed'] is None and result['utility']['statistics'] is None
    assert result['utility']['global_execution_and_costs_verified'] is False


def test_private_gate_bundle_identity_and_once_only_intent_cannot_be_bypassed():
    runtime = Runtime()
    runtime.bundle['insertion_text'] += ' changed'
    assert runner.run_utility(SHA, _runtime=runtime)['status'] == 'unknown'
    assert not any(v[0] == 'queue' for v in runtime.calls)
    runtime = Runtime()
    runner.run_utility(SHA, _runtime=runtime)
    before = len([v for v in runtime.calls if v[0] == 'launch'])
    again = runner.run_utility(SHA, _runtime=runtime)
    assert again['status'] == 'unknown'
    assert len([v for v in runtime.calls if v[0] == 'launch']) == before


def test_default_controller_and_worker_bind_metadata006_without_changing_actor_or_model():
    from scripts import native_appworld_actor_004 as actor
    from scripts import native_appworld_controller_006 as controller
    from scripts.native_appworld_model_client_004 import NativeModel
    from scripts.native_appworld_worker_client_006 import NativeWorkerClient

    assert runner.controller is controller
    assert runner.NativeWorkerClient is NativeWorkerClient
    assert runner.actor is actor and runner.NativeModel is NativeModel


def test_bundle_gate_kind_is_bound_to_verified_ordinary_control_recovery():
    plan, bundle = plan_and_bundle()
    assert runner._validate_bundle(plan, bundle) == TEXT
    bundle['source_gate_kind'] = 'hint_causal_help_gate'
    with pytest.raises(runner.NativeExperienceUtilityError):
        runner._validate_bundle(plan, bundle)
    plan['source']['source_gate_kind'] = bundle['source_gate_kind']
    with pytest.raises(runner.NativeExperienceUtilityError):
        runner._validate_bundle(plan, bundle)


@pytest.mark.parametrize('arm', ['control', 'experience'])
def test_episode_entry_uses_same_seed_budget_and_exact_memory_without_replay(monkeypatch, tmp_path, arm):
    from scripts import native_appworld_controller_006 as controller

    assert runner.controller is controller
    plan, bundle = plan_and_bundle()
    clock = [0.0]
    registry = SimpleNamespace(PRIVATE_RUN_ROOT=str(tmp_path), PORT=18991)
    job = {'owner': 'b' * 32}
    binding = {'path': str(tmp_path / 'reviewed-bundle.json'), 'sha256': runner._hash(_encode(bundle) + b'\n')}
    config = {'plan_sha256': SHA, 'pair_index': 0, 'arm': arm,
              'run_dir': str(tmp_path / ('unit-00-' + arm + '-' + job['owner'])),
              'job': job, 'expected_backend': {'synthetic': True}, 'bundle_binding': binding}
    config_path = str(tmp_path / 'config.json')
    documents = {binding['path']: _encode(bundle) + b'\n', config_path: _encode(config) + b'\n',
                 str(tmp_path / 'study-intent.json'): _encode({
                     'plan_sha256': SHA, 'reviewed_bundle_sha256': binding['sha256']}) + b'\n'}
    reader = SimpleNamespace(_read_private=lambda path, **_kw: documents[str(path)],
        _private_directory=lambda path: os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC),
        _backend=lambda *_args: None, _queue_binding=lambda *_args: ({}, {}, ('fixed',)))
    monkeypatch.setattr(runner, '_registry', lambda: registry)
    monkeypatch.setattr(runner, '_services', lambda _root: (SimpleNamespace(now=lambda: clock[0]), reader))
    monkeypatch.setattr(runner, '_queue', lambda: SimpleNamespace(
        make_job=lambda *_args: job, queue_operation=lambda *_args: None))
    model_calls, worker_calls, controller_calls, requests = [], [], [], []
    cost = dict.fromkeys(('action_attempts', 'api_invocations', 'upstream_requests'), 0)

    class Model:
        def __init__(self, *args, **kwargs):
            model_calls.append((args, kwargs))
            self.max_tokens, self.deadline = kwargs['max_tokens'], kwargs['deadline']

        def check_health(self):
            clock[0] += 10
            return {'scope': 'synthetic_CPU_health'}

        def count_input_tokens(self, messages):
            return 15361

        def generate(self, *_args, **_kwargs):
            pytest.fail('over-context CPU fixture must not generate')

    class Worker:
        unit_name = 'synthetic-worker006'
        released = False

        def request(self, request, *, timeout):
            requests.append((request['op'], timeout))
            if request['op'] == 'metadata':
                assert timeout >= 80
                clock[0] += 80
            return {'ok': True, 'cost': cost.copy(), 'result': {
                'instruction': 'Synthetic', 'supervisor': {}, 'app_descriptions': {}}}

        def wait(self, *, timeout):
            requests.append(('wait', timeout))
            return 0

        def snapshot(self, *, timeout):
            requests.append(('snapshot', timeout))
            return {'unit_name': self.unit_name, 'cost': cost.copy()}

        def release(self):
            self.released = True

        def abort(self):
            pytest.fail('successful metadata fixture must close normally')

    instance = Worker()

    def worker(*args, **kwargs):
        worker_calls.append((args, kwargs))
        return instance

    core = controller._isolated()
    execute_original = core.run_task

    def execute(task, experiment, run_dir, **kwargs):
        controller_calls.append((task, experiment, run_dir, kwargs))
        return execute_original(task, experiment, run_dir, **kwargs,
            _clock=lambda: clock[0], _inspect=lambda *_a, **_kw: {'scope': 'synthetic_CPU_profile'})

    monkeypatch.setattr(core, 'run_task', execute)
    monkeypatch.setattr(core, 'PRIVATE_STATE_ROOT', tmp_path)
    monkeypatch.setattr(core.socket, 'gethostname', lambda: 'amax')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(controller, '_CORE', core)
    monkeypatch.setattr(runner, 'NativeWorkerClient', worker)

    result = runner.run_episode(config_path, runner._hash(documents[config_path]),
        _load_plan=lambda *_a, **_kw: plan, _model_factory=Model)
    assert result['status'] == 'saved_and_closed'
    args = controller_calls[0][3]
    assert args['review_hint'] == (TEXT if arm == 'experience' else None)
    assert args['source_metadata'] is args['prefix_records'] is None
    assert args['seed'] == 17 and args['is_policy_episode'] is True
    assert args['limits'] == runner._limits(plan['utility'])
    assert len(worker_calls) == len(model_calls) == 1 and model_calls[0][1]['max_tokens'] == 1024
    assert requests == [('metadata', 110.0), ('save', 60), ('close', 60), ('wait', 60), ('snapshot', 60)]
    assert instance.released and result['elapsed_seconds'] == 90
    assert result['actor_outcome']['cost'] == {'generations': 0, 'prompt_tokens': 0, 'completion_tokens': 0, **cost}
    assert result['actor_outcome']['status'] == 'incomplete' and result['worker_exit_code'] == 0
    assert Path(config['run_dir']).stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        runner.run_episode(config_path, runner._hash(documents[config_path]),
                           _load_plan=lambda *_a, **_kw: plan, _model_factory=Model)
    assert len(worker_calls) == len(model_calls) == 1


def test_episode_directory_rejects_preexisting_symlink(tmp_path):
    target = tmp_path / 'synthetic-other'
    target.mkdir(mode=0o700)
    path = tmp_path / ('unit-00-control-' + 'b' * 32)
    path.symlink_to(target, target_is_directory=True)
    reader = SimpleNamespace(_private_directory=lambda value: os.open(
        value, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC))
    with pytest.raises(FileExistsError):
        runner._create_episode_directory(reader, tmp_path, path)
    assert path.is_symlink() and list(target.iterdir()) == []


def test_episode_directory_rejects_changed_namespace_identity(tmp_path):
    other = tmp_path / 'synthetic-other'
    other.mkdir(mode=0o700)
    path = tmp_path / ('unit-00-control-' + 'b' * 32)

    def open_changed(value):
        chosen = tmp_path if Path(value) == tmp_path else other
        return os.open(chosen, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)

    reader = SimpleNamespace(_private_directory=open_changed)
    with pytest.raises(runner.NativeExperienceUtilityError):
        runner._create_episode_directory(reader, tmp_path, path)
    assert path.is_dir() and list(other.iterdir()) == []


def test_default_namespace_and_child_commands_keep_existing_resource_controls():
    from scripts import native_source_episode_004 as source_episode

    runtime = runner._Runtime(runner._ROOT)
    assert runtime.reader.PRIVATE_RUN_ROOT == Path(registration.PRIVATE_RUN_ROOT)
    assert source_episode._get_core().PRIVATE_RUN_ROOT != runtime.reader.PRIVATE_RUN_ROOT
    unit = 'wll-native-source-utility-00-control-' + 'a' * 32 + '.service'
    for kind, memory, duration in [('episode', '6G', 900), ('scorer', '3G', 120),
                                    ('pilot', '6G', runner.PARENT_SECONDS)]:
        command = runner.unit_command(kind, unit, [], root=Path('/synthetic/root'), log_path='/synthetic/private/log')
        assert '--property=MemoryMax=' + memory in command
        assert '--property=RuntimeMaxSec=' + str(duration) in command
        assert '--property=NoNewPrivileges=yes' in command
        if kind == 'episode':
            assert '--episode' in command and 'scripts.run_native_experience_utility_001' in ' '.join(command)
        if kind == 'scorer':
            assert 'scripts.native_appworld_scorer' in ' '.join(command)


def test_verification_budget_is_per_pass_and_private_output_never_enters_argv_or_audit(monkeypatch):
    runtime = runner._Runtime(runner._ROOT)
    clock, calls = {'now': 10.0}, []
    monkeypatch.setattr(runtime, 'now', lambda: clock['now'])

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        clock['now'] += 100
        return SimpleNamespace(returncode=0, stdout=_encode({'bundle': 'private synthetic text'}), stderr=b'')

    monkeypatch.setattr(runner.subprocess, 'run', run)
    for _ in range(2):
        result = runtime._verify({'action': 'hash', 'plan_sha256': SHA,
                                  'binding': {'private': 'private synthetic input'}})
        assert result == {'bundle': 'private synthetic text'}
    assert [v['wall_seconds'] for v in runtime.verification_audits] == [100, 100]
    assert all(v[1]['timeout'] == 180 and v[1]['env']['CUDA_VISIBLE_DEVICES'] == '' for v in calls)
    assert 'private synthetic' not in repr([v[0] for v in calls])

    def timeout(argv, **kwargs):
        clock['now'] += 180
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'], output=b'private synthetic output')

    monkeypatch.setattr(runner.subprocess, 'run', timeout)
    with pytest.raises(runner.NativeExperienceUtilityError):
        runtime._verify({'action': 'hash', 'plan_sha256': SHA})
    assert runtime.verification_audits[-1]['wall_seconds'] == 180
    assert runtime.verification_audits[-1]['returned'] is False
    assert 'private synthetic' not in str(runtime.verification_audits)


def test_help_cannot_initialize_execution_or_private_preflight(monkeypatch):
    monkeypatch.setattr(runner, '_Runtime', lambda *_a: pytest.fail('unexpected runtime'))
    with pytest.raises(SystemExit) as caught:
        runner.main(['--help'])
    assert caught.value.code == 0
