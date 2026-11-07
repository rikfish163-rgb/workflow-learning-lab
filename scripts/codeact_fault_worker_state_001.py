"""Trusted CodeAct cell metadata and a whole-episode Requester dispatch cap.

Instantiate only in the future constrained Docker worker. This module executes
no code itself. Observation text is preserved and never parsed for error labels.
"""
import hashlib
import inspect
import threading
from types import MethodType

from scripts.codeact_execution_observer_001 import ExecutionObserver
from scripts.completion_actor_boundary import actor_code_restriction
from scripts.completion_worker_state import CompletionWorkerState


class ApiBudgetExceeded(RuntimeError):
    """A dispatch was blocked before the original Requester was called."""

    def __init__(self):
        super().__init__('codeact_api_budget_rejected')


def _unknown(calls=0):
    return {'status': 'unknown', 'error_type': None, 'hook_calls': calls}


POLICY = 'controlled_single_transient_request_refusal_v1'
POLICY_SHA256 = hashlib.sha256(POLICY.encode('ascii')).hexdigest()


class FaultExecutionObserver(ExecutionObserver):
    def __init__(self, world, *, expected_function, state):
        super().__init__(world, expected_function=expected_function)
        self._fault_state = state

    def _capture(self, code):
        result = super()._capture(code)
        try:
            injected = self._fault_state._injected_exception
            if injected is not None and (result.error_before_exec is injected or result.error_in_exec is injected):
                self._fault_state._injected_error_observed = True
        except Exception:
            # A malformed result cannot change the cell's real outcome.
            pass
        return result


class CodeActFaultWorkerState(CompletionWorkerState):
    def __init__(self, world, allowed_apis, *, expected_function, api_limit=500,
                 argument_policy=None, response_policy=None):
        if (not inspect.isfunction(expected_function) or type(api_limit) is not int
                or not 1 <= api_limit <= 500):
            raise ValueError('codeact_within_configuration')
        self._terminal = self._fault = self._restored = False
        self._injected_refusals = 0
        self._fault_marker = None
        self._injected_exception = None
        self._injected_error_observed = False
        self._fault_policy_sha256 = POLICY_SHA256
        self._last_execution = _unknown()
        self._api_limit = api_limit
        self._meter = {'request_dispatches': 0, 'budget_rejections': 0}
        self._budget_lock = threading.RLock()
        self._expected_function = expected_function
        self._observer = FaultExecutionObserver(world, expected_function=expected_function, state=self)
        self._observer.install()
        try:
            super().__init__(world, allowed_apis, argument_policy=argument_policy,
                             response_policy=response_policy)
            self._base_request, self._base_counter = self._request, self._counter
            self._base_code = self._base_request.__code__
            captured = inspect.getclosurevars(self._base_request).nonlocals
            self._original_request = captured['original_request']
            if (not isinstance(self._original_request, MethodType)
                    or self._original_request.__self__ is not self._requester):
                raise ValueError('codeact_within_request_binding')
            self._request_function = self._original_request.__func__

            def guarded_request(_app_name, _api_name, *args, **kwargs):
                with self._budget_lock:
                    self._verify()
                    if (self._terminal or self._fault) and not self._status_probe[0]:
                        raise RuntimeError('codeact_within_unavailable')
                    counted = _app_name != 'admin' and not self._status_probe[0]
                    if counted:
                        if self._meter['request_dispatches'] >= self._api_limit:
                            self._meter['budget_rejections'] += 1
                            raise ApiBudgetExceeded()
                        self._meter['request_dispatches'] += 1
                        if (_app_name not in ('api_docs', 'supervisor', 'admin')
                                and self._injected_refusals == 0):
                            # The underlying counter counts attempts, including failures
                            # before HTTP; the original Requester is deliberately not called.
                            self._counter[0] += 1
                            self._injected_refusals = 1
                            self._fault_marker = self._counter[0]
                            self._injected_exception = Exception('Transient tool failure: request was not executed.')
                            self._verify()
                            raise self._injected_exception
                    try:
                        result = self._base_request(_app_name, _api_name, *args, **kwargs)
                    except BaseException:
                        # Preserve the original exception even if integrity also fails.
                        try:
                            self._verify()
                        except BaseException:
                            pass
                        raise
                    self._verify()
                    return result

            self._guard = guarded_request
            self._guard_code = guarded_request.__code__
            self._request = self._requester.request = guarded_request
            self._verify()
        except BaseException:
            self._observer.restore()
            raise

    @property
    def last_execution(self):
        return dict(self._last_execution)

    def _verify(self):
        try:
            super()._verify()
            captured = inspect.getclosurevars(self._base_request).nonlocals
            guard_captured = inspect.getclosurevars(self._guard).nonlocals
            intact = (self._request is self._guard and self._base_request.__code__ is self._base_code
                      and self._guard.__code__ is self._guard_code
                      and set(guard_captured) == {'self'} and guard_captured['self'] is self
                      and self._counter is self._base_counter
                      and captured['counter'] is self._counter
                      and captured['status_probe'] is self._status_probe
                      and captured['original_request'] is self._original_request
                      and self._original_request.__self__ is self._requester
                      and self._original_request.__func__ is self._request_function
                      and inspect.getattr_static(type(self._requester), 'request') is self._request_function
                      and type(self._counter) is list and len(self._counter) == 1
                      and type(self._counter[0]) is int and self._counter[0] >= 0
                      and type(self._status_probe) is list and len(self._status_probe) == 1
                      and type(self._status_probe[0]) is bool
                      and type(self._api_limit) is int and 1 <= self._api_limit <= 500
                      and type(self._meter) is dict
                      and type(self._injected_refusals) is int and self._injected_refusals in (0, 1)
                      and self._injected_refusals <= self._counter[0]
                      and ((self._injected_refusals == 0 and self._fault_marker is None)
                           or (self._injected_refusals == 1 and type(self._fault_marker) is int
                               and 1 <= self._fault_marker <= self._counter[0]))
                      and type(self._injected_error_observed) is bool
                      and ((self._injected_refusals == 0 and self._injected_exception is None
                            and self._injected_error_observed is False)
                           or (self._injected_refusals == 1 and type(self._injected_exception) is Exception
                               and self._injected_exception.args == ('Transient tool failure: request was not executed.',)))
                      and self._observer._fault_state is self
                      and self._fault_policy_sha256 == POLICY_SHA256
                      and set(self._meter) == {'request_dispatches', 'budget_rejections'}
                      and all(type(v) is int and v >= 0 for v in self._meter.values())
                      and self._counter[0] == self._meter['request_dispatches'] <= self._api_limit)
            if self._restored:
                intact = (intact and '_shell_run_cell' not in vars(self._world)
                          and inspect.getattr_static(type(self._world), '_shell_run_cell')
                          is self._expected_function)
            else:
                intact = intact and self._observer.verify()
            if not intact:
                raise ValueError
        except BaseException:
            self._fault = True
            self._last_execution = _unknown()
            raise RuntimeError('codeact_within_integrity') from None

    def fault_status(self):
        self._verify()
        return {'injected_refusals': self._injected_refusals,
                'injected_at_request_attempt': self._fault_marker,
                'injected_error_observed': self._injected_error_observed,
                'policy_sha256': self._fault_policy_sha256,
                'backend_dispatched_for_fault': False}

    def api_metadata(self):
        self._verify()
        dispatched, rejected = self._meter['request_dispatches'], self._meter['budget_rejections']
        return {'attempts': dispatched + rejected, 'request_dispatches': dispatched,
                'budget_rejections': rejected, 'limit': self._api_limit}

    def inspect(self, text):
        if self._terminal or self._fault:
            raise RuntimeError('codeact_within_unavailable')
        return super().inspect(text)

    def completed(self):
        if self._terminal or self._fault:
            raise RuntimeError('codeact_within_unavailable')
        return super().completed()

    def execute(self, code):
        if self._terminal or self._fault:
            raise RuntimeError('codeact_within_unavailable')
        self._verify()
        reason = actor_code_restriction(code)
        if reason:
            self._last_execution = {'status': 'policy_rejected', 'error_type': None, 'hook_calls': 0}
            return {'observation': 'Execution failed. Policy restriction: ' + reason,
                    'execution': self.last_execution, 'api': self.api_metadata()}
        rejected_before = self._meter['budget_rejections']
        self._last_execution = _unknown()
        execution = None
        try:
            self._observer.begin()
        except BaseException:
            self._fault = True
            raise
        try:
            observation = super().execute(code)
            execution = self._observer.finish()
            self._verify()
            if type(observation) is not str:
                raise RuntimeError('codeact_within_observation')
        except BaseException:
            if execution is None:
                try:
                    execution = self._observer.finish()
                except BaseException:
                    execution = _unknown()
            self._last_execution = _unknown(execution['hook_calls'])
            self._fault = True
            raise
        if (self._meter['budget_rejections'] > rejected_before
                and execution['status'] not in {'timeout', 'unknown'}):
            execution = {'status': 'api_budget_rejected', 'error_type': 'ApiBudgetExceeded',
                         'hook_calls': execution['hook_calls']}
        self._last_execution = dict(execution)
        if execution['status'] in {'timeout', 'unknown', 'api_budget_rejected'}:
            self._fault = True
        return {'observation': observation, 'execution': self.last_execution, 'api': self.api_metadata()}

    def prepare_terminal(self):
        """Restore our cell hook before trusted terminal scoring/cleanup."""
        self._terminal = True
        try:
            self._verify()
        finally:
            if not self._restored:
                self._observer.restore()
                self._restored = True
        self._verify()
        return {'execution': self.last_execution, 'api': self.api_metadata()}
