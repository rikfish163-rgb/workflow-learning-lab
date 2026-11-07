"""Container-only trusted RPC for within-workflow CodeAct execution.

Import is inert. Real AppWorld imports occur only behind the container guard.
Typed execute metadata belongs to the parent; actor_observation is the sole
actor projection. Evaluation is terminal and exposes only an official boolean.
"""
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import verification_readonly_registry_v3 as registry
from scripts.codeact_execution_observer_001 import _ERROR_TYPES
from scripts.codeact_within_worker_state_001 import CodeActWithinWorkerState

PROTOCOL, SEED_PROTOCOL = 'wll-codeact-within-worker-v1', 'wll-appworld-seed-v1'
STATUSES = ('success', 'error_before_exec', 'error_in_exec', 'policy_rejected',
            'unobserved', 'timeout', 'unknown', 'api_budget_rejected')
SCORING_SOURCES = {
    'evaluator.py': 'bde9deb3b1e6ac0fa9819013729c0e817a97c90f579108fa032a90bba0ca51cb',
    'task.py': 'f999c02316639eae33316ad5bac52c58c217a80ee553f38b3084eb5ea18c5d7e',
}
_FIELDS = {'list': {'split'}, 'describe': {'task_id'},
           'start': {'task_id', 'max_steps', 'environment_seed'}, 'execute': {'code'},
           **{op: set() for op in ('capabilities', 'completed', 'api_count', 'evaluate', 'close')}}


def _require(value):
    if not value:
        raise ValueError('codeact_worker_invalid')


def capabilities():
    return {'protocol': PROTOCOL, 'seed_protocol': SEED_PROTOCOL, 'environment_seed': 100,
            'api_limit': 500, 'execution_statuses': list(STATUSES),
            'actor_observation': 'observation_only_unchanged_v1',
            'evaluation': 'trusted_terminal_same_worker_v1',
            'source_verification': 'verification_readonly_registry_v3.verify_installed_sources',
            'registry_sha256': registry.capabilities()['registry_sha256'],
            'runtime_source_sha256': dict(registry.SOURCE_SHA256),
            'scoring_source_sha256': dict(SCORING_SOURCES)}


def actor_observation(result):
    """Strict, pure projection; never stringify trusted parent metadata."""
    _require(type(result) is dict and set(result) == {'observation', 'execution', 'api'}
             and type(result['observation']) is str)
    execution, api = result['execution'], result['api']
    _require(type(execution) is dict and set(execution) == {'status', 'error_type', 'hook_calls'}
             and type(execution['status']) is str and execution['status'] in STATUSES
             and type(execution['hook_calls']) is int and execution['hook_calls'] >= 0
             and (execution['error_type'] is None or type(execution['error_type']) is str)
             and type(api) is dict and set(api) == {'attempts', 'request_dispatches', 'budget_rejections', 'limit'}
             and all(type(v) is int and v >= 0 for v in api.values()) and api['limit'] == 500
             and api['request_dispatches'] <= api['limit']
             and api['attempts'] == api['request_dispatches'] + api['budget_rejections'])
    status, error, calls = execution['status'], execution['error_type'], execution['hook_calls']
    if status in {'error_before_exec', 'error_in_exec'}:
        _require(calls == 1 and error in set(_ERROR_TYPES.values()) | {'other'})
    elif status == 'api_budget_rejected':
        _require(calls <= 1 and error == 'ApiBudgetExceeded' and api['budget_rejections'] > 0)
    elif status == 'timeout':
        _require(calls == 1 and error in {None, 'TimeoutError'})
    else:
        _require(error is None and (status == 'unknown' or calls == int(status == 'success')))
    return result['observation']


class CodeActWithinWorker:
    """Trusted runtime injection for the Docker entrypoint or synthetic tests."""

    def __init__(self, runtime):
        self.runtime, self.world, self.state = runtime, None, None
        self._started = self._terminal = self._failed = self._closed = self._close_failed = False

    def reject(self):
        self._failed = self._terminal = True
        return {'ok': False, 'error': 'CodeActWorkerError'}

    def handle(self, request):
        try:
            return {'ok': True, 'result': self._dispatch(request)}
        except Exception:
            return self.reject()

    def _dispatch(self, request):
        _require(type(request) is dict and type(request.get('op')) is str)
        op = request['op']
        _require(op in _FIELDS and set(request) == {'op'} | _FIELDS[op])
        if op == 'capabilities':
            return capabilities()
        if op == 'close':
            return self.close()
        _require(not self._closed)
        if op == 'api_count':
            _require(self.state is not None)
            return self.state.api_count()
        _require(not self._terminal and not self._failed)
        if op in {'list', 'describe'}:
            _require(not self._started)
            if op == 'list':
                _require(request['split'] in {'train', 'dev', 'test_normal', 'test_challenge'})
                return self.runtime.list_task_ids(request['split'])
            _require(type(request['task_id']) is str and bool(request['task_id']))
            task = self.runtime.task_loader(task_id=request['task_id'], load_ground_truth=False)
            return {'instruction': task.instruction, 'supervisor': str(task.supervisor)}
        if op == 'start':
            return self._start(request)
        _require(self.world is not None and self.state is not None)
        if op == 'execute':
            _require(type(request['code']) is str)
            request['code'].encode('utf-8')
            result = self.state.execute(request['code'])
            actor_observation(result)  # Validate; send the full result only to the parent RPC.
            if result['execution']['status'] in {'unknown', 'timeout', 'api_budget_rejected'}:
                self._failed = self._terminal = True
            return result
        if op == 'completed':
            return self.state.completed()
        self._terminal = True  # Lock before prepare, imports, or evaluation can fail.
        self.state.prepare_terminal()
        tracker = self.runtime.evaluate_task(task_id=self.world.task_id,
            experiment_name=self.world.experiment_name, suppress_errors=True, save_report=False)
        total, expected, success = tracker.total_count, tracker.num_tests, tracker.success
        _require(type(total) is int and type(expected) is int and expected > 0 and 0 <= total <= expected
                 and type(success) is bool and (not success or total == expected))
        return success

    def _start(self, request):
        _require(not self._started and type(request['environment_seed']) is int
                 and request['environment_seed'] == 100 and type(request['task_id']) is str
                 and bool(request['task_id']) and type(request['max_steps']) is int
                 and request['max_steps'] > 0)
        self._started = True
        self.world = self.runtime.world_factory(task_id=request['task_id'], experiment_name='isolated',
            max_interactions=request['max_steps'], random_seed=100, load_ground_truth=False,
            timeout_seconds=30, raise_on_unsafe_syntax=True, raise_on_unsafe_execution=True,
            include_direct_functions=False, import_utils=False)
        try:
            _require(type(self.world.random_seed) is int and self.world.random_seed == 100
                     and self.world.task.ground_truth is None)
            self.state = CodeActWithinWorkerState(self.world, registry.ALLOWED_APIS,
                expected_function=self.runtime.expected_function, api_limit=500,
                argument_policy=registry.validate_inspection_arguments,
                response_policy=registry.validate_inspection_response)
        except BaseException:
            self.close()
            raise
        return {'protocol': SEED_PROTOCOL, 'started': True, 'task_id': request['task_id'],
                'environment_seed': {'requested': 100, 'effective': 100, 'verified': True}}

    def close(self):
        if not self._closed:
            self._closed = self._terminal = True
            try:
                if self.state is not None:
                    self.state.prepare_terminal()
            except BaseException:
                self._close_failed = True
            finally:
                try:
                    if self.world is not None:
                        self.world.close()
                except BaseException:
                    self._close_failed = True
                finally:
                    try:
                        self.runtime.close_all()
                    except BaseException:
                        self._close_failed = True
        if self._close_failed:
            raise RuntimeError('codeact_worker_close_failed')
        return True


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result)
        result[key] = value
    return result


def _reject_number(_):
    raise ValueError('codeact_worker_invalid')


def serve(worker, stdin, stdout, stderr):
    exit_code = 0
    try:
        while line := stdin.readline(4_000_001):
            try:
                _require(len(line.encode('utf-8')) <= 4_000_000 and line.endswith('\n'))
                request = json.loads(line, object_pairs_hook=_pairs,
                                     parse_float=_reject_number, parse_constant=_reject_number)
                with contextlib.redirect_stdout(stderr):
                    reply = worker.handle(request)
            except Exception:
                request, reply = None, worker.reject()
            stdout.write(json.dumps(reply, ensure_ascii=True, allow_nan=False) + '\n')
            stdout.flush()
            if type(request) is dict and request.get('op') == 'close':
                break
    finally:
        try:
            with contextlib.redirect_stdout(stderr):
                worker.close()
        except BaseException:
            exit_code = 1
    return exit_code


def _load_runtime():
    if (os.getuid() != 10001 or os.environ.get('APPWORLD_ROOT') != '/workspace'
            or not Path('/.dockerenv').is_file() or os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''):
        raise RuntimeError('codeact_worker_container_required')
    import appworld
    from appworld import AppWorld, load_task_ids
    from appworld.task import Task

    root = Path(appworld.__file__).parent
    registry.verify_installed_sources(root)
    for name, expected in SCORING_SOURCES.items():
        _require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected)

    def evaluate_task(**kwargs):
        from appworld.evaluator import evaluate_task as official

        return official(**kwargs)

    return SimpleNamespace(world_factory=AppWorld, task_loader=Task.load, list_task_ids=load_task_ids,
                           evaluate_task=evaluate_task, expected_function=AppWorld._shell_run_cell,
                           close_all=AppWorld.close_all)


def main():
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = _load_runtime()
        return serve(CodeActWithinWorker(runtime), sys.stdin, sys.stdout, sys.stderr)
    except Exception:
        sys.stderr.write('codeact_within_worker_failed\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
