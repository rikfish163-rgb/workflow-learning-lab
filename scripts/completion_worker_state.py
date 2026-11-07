"""Trusted inspection state, instantiated outside the actor's REPL namespace.

Counts public Requester.request attempts, including failures before HTTP dispatch
and track=False requests. Only the trusted completed() status probe is excluded.
This is a different metric from the mutable upstream HTTP request history.
Docker and upstream runtime restrictions remain required; the actor syntax
restriction is a constrained research subset, not a general Python sandbox.
"""

from scripts.completion_actor_boundary import actor_code_restriction, assigned_actor_names
from scripts.completion_check_protocol import (
    CheckProtocolError,
    dispatch_check,
    resolve_check_request,
)


class CompletionWorkerState:
    def __init__(self, world, allowed_apis, *, argument_policy=None, response_policy=None):
        # Configuration faults must fail startup, never become actor refusals.
        resolve_check_request({'check': 'Validate configuration.', 'tool': None}, {}, allowed_apis)
        if ((argument_policy is None) != (response_policy is None)
                or (argument_policy is not None
                    and (not callable(argument_policy) or not callable(response_policy)))):
            raise ValueError('invalid_inspection_policy')
        namespace = world.shell.user_ns
        if (type(namespace) is not dict
                or not all(type(key) is str for key in namespace)
                or world.include_direct_functions is not False
                or namespace.get('apis') is not world.apis
                or namespace.get('requester') is not world.requester):
            raise ValueError('unsupported_worker_namespace')
        self._world = world
        self._namespace = namespace
        self._apis = world.apis
        self._requester = world.requester
        self._initial_names = frozenset(namespace)
        self._actor_names = set()
        self._allowed_apis = allowed_apis
        self._argument_policy = argument_policy
        self._response_policy = response_policy
        original_request = world.requester.request
        counter = [0]
        status_probe = [False]

        def counted_request(_app_name, _api_name, *args, **kwargs):
            if _app_name != 'admin' and not status_probe[0]:
                counter[0] += 1
            return original_request(_app_name, _api_name, *args, **kwargs)

        self._request = counted_request
        self._counter = counter
        self._status_probe = status_probe
        self._requester.request = counted_request
        for name in ('requester', 'Requester', 'ApiCollection'):
            namespace.pop(name, None)

    def _verify(self):
        if (self._world.requester is not self._requester
                or self._requester.request is not self._request
                or self._world.shell.user_ns is not self._namespace
                or self._world.apis is not self._apis
                or self._namespace.get('apis') is not self._apis
                or any(name in self._namespace
                       for name in ('requester', 'Requester', 'ApiCollection'))
                or not all(type(key) is str for key in self._namespace)):
            raise RuntimeError('worker_binding_changed')

    def api_count(self):
        self._verify()
        return self._counter[0]

    def completed(self):
        self._verify()
        self._status_probe[0] = True
        try:
            result = self._world.task_completed()
        finally:
            self._status_probe[0] = False
        self._verify()
        if type(result) is not bool:
            raise TypeError('invalid_completion_status')
        return result

    def execute(self, code):
        self._verify()
        reason = actor_code_restriction(code)
        if reason:
            return 'Execution failed. Policy restriction: ' + reason
        assigned = assigned_actor_names(code)
        before = frozenset(self._namespace)
        result = self._world.execute(code)
        self._verify()
        # A static assignment in an untaken branch is not provenance for an
        # already-existing framework name. No initial framework keys qualify.
        self._actor_names.intersection_update(self._namespace)
        self._actor_names.update(
            name for name in assigned if name not in before
            and name not in self._initial_names and name in self._namespace)
        return result

    def inspect(self, text):
        self._verify()
        before = self.api_count()
        namespace = {name: self._namespace[name] for name in self._actor_names
                     if name in self._namespace}

        def request(app, api, *, track, raise_on_failure, **arguments):
            if self._argument_policy is not None:
                self._argument_policy(app, api, arguments)
            response = self._request(app, api, track=track,
                                     raise_on_failure=raise_on_failure, **arguments)
            if self._response_policy is not None:
                self._response_policy(app, api, response)
            return response

        try:
            result = dispatch_check(text, namespace=namespace,
                                    allowed_apis=self._allowed_apis,
                                    request=request, requests_count=self.api_count)
        except CheckProtocolError as exc:
            # A post-dispatch exception is unknown, never a zero-cost refusal.
            if self.api_count() != before:
                raise
            return {'status': 'rejected', 'api_calls': 0, 'observation': str(exc)}
        self._verify()
        return result
