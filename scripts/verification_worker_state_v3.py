"""V3 checker worker with isolated frozen state and V2 failure classification.

The frozen V1 state is loaded into a separate module namespace and bound to the
V3 registry and policy. The classifier below retains the V2 distinction between
known returned failures and unknown incomplete or untrusted operations. Only a
constrained worker may instantiate this state around a real AppWorld instance.
"""

import hashlib
import importlib.util
from pathlib import Path

from scripts import verification_program_v3 as policy
from scripts import verification_readonly_registry_v3 as registry
from scripts.completion_check_protocol import CheckProtocolError


def _load_state():
    source = Path(__file__).with_name('verification_worker_state.py')
    spec = importlib.util.spec_from_file_location('scripts._verification_worker_state_v3_core', source)
    if spec is None or spec.loader is None:
        raise ImportError('frozen verification worker state unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.registry = registry
    module.policy = policy
    return module


_CORE = _load_state()
MAX_REQUESTS = _CORE.MAX_REQUESTS
MAX_TRANSACTION_REQUESTS = _CORE.MAX_TRANSACTION_REQUESTS
MAX_OBSERVATION_BYTES = _CORE.MAX_OBSERVATION_BYTES


def _sha(name):
    return hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()


def capabilities():
    return {
        'protocol': 'wll-executable-verification-v3',
        'readonly_registry': registry.capabilities(),
        'registry_source_sha256': _sha('verification_readonly_registry_v3.py'),
        'condition_registry_sha256': _sha('condition_readonly_registry.py'),
        'completion_registry_sha256': _sha('completion_readonly_registry.py'),
        'program_v3_sha256': _sha('verification_program_v3.py'),
        'program_core_sha256': _sha('verification_program.py'),
        'worker_state_v3_sha256': _sha('verification_worker_state_v3.py'),
        'worker_state_core_sha256': _sha('verification_worker_state.py'),
        'worker_state_v2_sha256': _sha('verification_worker_state_v2.py'),
        'base_worker_sha256': _sha('completion_worker_state.py'),
        'actor_boundary_sha256': _sha('completion_actor_boundary.py'),
        'check_protocol_sha256': _sha('completion_check_protocol.py'),
        'worker_entrypoint_v3_sha256': _sha('verification_appworld_worker_v3.py'),
        'worker_entrypoint_core_sha256': _sha('completion_appworld_worker.py'),
        'max_requests': MAX_REQUESTS,
        'max_transaction_requests': MAX_TRANSACTION_REQUESTS,
        'max_observation_bytes': MAX_OBSERVATION_BYTES,
        'failure_semantics': 'known_returned_execution_failure_without_cause_or_task_score',
        'semantics': 'clause_scoped_model_claim_not_official_task_success',
    }


class VerificationWorkerStateV3(_CORE.VerificationWorkerState):
    """Use exactly the reviewed registry; preserve V2's conservative failures."""

    def __init__(self, world, allowed_apis, *, argument_policy=None, response_policy=None):
        if (type(allowed_apis) is not frozenset or allowed_apis != registry.ALLOWED_APIS
                or argument_policy is not registry.validate_inspection_arguments
                or response_policy is not registry.validate_inspection_response):
            raise ValueError('verification_v3_registry_binding_changed')
        super().__init__(world, registry.ALLOWED_APIS, argument_policy=argument_policy,
                         response_policy=response_policy)

    def inspect(self, text):
        if (self._allowed_apis != registry.ALLOWED_APIS
                or self._argument_policy is not registry.validate_inspection_arguments
                or self._response_policy is not registry.validate_inspection_response):
            raise RuntimeError('verification_v3_registry_binding_changed')
        execute = self._world.execute
        argument_policy = self._argument_policy
        returned = False
        output = None
        invalid_result = False
        result = None
        argument_fault = False

        def observe_arguments(*args):
            nonlocal argument_fault
            try:
                return argument_policy(*args)
            except CheckProtocolError:
                raise
            except Exception:
                argument_fault = True
                raise

        def observe_return(source):
            nonlocal returned, output, invalid_result, result
            output = execute(source)
            returned = True
            if type(output) is str and output == 'Execution successful.':
                try:
                    result = policy.validate_result(self._namespace.get(policy.RESULT_NAME))
                except policy.VerificationProgramError:
                    invalid_result = True
            return output

        try:
            self._world.execute = observe_return
            if argument_policy is not None:
                self._argument_policy = observe_arguments
            answer = super().inspect(text)
        finally:
            self._world.execute = execute
            self._argument_policy = argument_policy
        complete_requests = answer['api_calls'] == len(answer['observations'])
        if (answer['execution_status'] == 'unknown' and returned
                and type(output) is str and complete_requests and not argument_fault):
            if output.startswith('Execution failed.'):
                answer['execution_status'] = 'failed'
                answer['reason'] = 'checker_execution_failed'
            elif output == 'Execution successful.':
                available = {f'query:{item["index"]}' for item in answer['observations']}
                invalid_reference = (result is not None and any(
                    ref not in available for ref in result['evidence']))
                if invalid_result or invalid_reference:
                    answer['execution_status'] = 'failed'
                    answer['reason'] = 'checker_result_invalid'
        return answer
