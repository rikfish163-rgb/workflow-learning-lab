"""One inspection followed by at most three corrections, without evaluation.

The lifecycle owner must supply a fresh world whose complete visible prefix was
already exactly replayed, a bound model, a durable event writer and a deadline.
World.inspect must enforce the frozen read-only policy in its trusted worker;
this client neither asserts that a given backend is read-only nor owns cleanup.
"""

import time

from scripts.completion_boundary import inspect_completion_action
from scripts.completion_check_messages import build_check_messages
from scripts.completion_check_protocol import CheckProtocolError, parse_check_request
from scripts.run_timed_tasks import RolloutTimedOut
from wll.model import ContextBudgetExceeded
from wll.protocol import fingerprint
from wll.runner import extract_code

MAX_NEW_CALLS = 4
MAX_ENVIRONMENT_STEPS = 40
CORRECTION_PROMPT = (
    'The inspection phase has ended. The pending completion action is still unexecuted. '
    'Use the visible history and inspection response to make a justified correction if needed, '
    'then complete the task. Resume the original Python code-output rules. '
    'At most three new code responses remain, including completion; no extra inspection budget exists.'
)


class _OperationFailed(Exception):
    pass


def run_intervention(task, prefix_steps, pending_response, world, model, *, arm,
                     tool_instructions, expected_backend, emit):
    """Return execution/cost metadata only; never return or request a score.

    Deadline, interrupt and event-writer errors propagate to the lifecycle owner,
    which must retain durable progress and unknown costs. Ordinary backend errors
    stop this arm as unknown. Known format rejection is not retried. No pending
    source action is implicitly executed, including after a skipped inspection.
    """
    messages = build_check_messages(task, prefix_steps, pending_response, arm=arm,
                                    tool_instructions=tool_instructions)
    if (inspect_completion_action(extract_code(pending_response))['status'] != 'candidate'
            or len(prefix_steps) + MAX_NEW_CALLS > MAX_ENVIRONMENT_STEPS
            or type(expected_backend) is not dict or not expected_backend or not callable(emit)):
        raise ValueError('invalid intervention boundary, budget, backend or journal')
    result = {'termination': 'unknown', 'ready_for_scoring': False,
              'generation_attempts': 0, 'generation_returns': 0,
              'new_executions': 0, 'execution_returns': 0, 'api_calls': 0,
              'prompt_tokens': 0, 'completion_tokens': 0,
              'unknown_additional_cost': False, 'operation_seconds': {}, 'elapsed_seconds': 0.0}
    started = time.monotonic()

    def fail(reason, *, unknown_cost=True):
        result.update(error_type=reason, unknown_additional_cost=unknown_cost)
        raise _OperationFailed

    def call(kind, operation, **fields):
        emit(kind + '_attempt', **fields)
        if kind == 'generation':
            result['generation_attempts'] += 1
        if kind in ('inspection', 'execution'):
            result['new_executions'] += 1
        before = time.monotonic()
        try:
            value = operation()
        except (RolloutTimedOut, KeyboardInterrupt, SystemExit):
            raise
        except ContextBudgetExceeded:
            result['termination'] = 'context_budget'
            result['unknown_additional_cost'] = True
            elapsed = time.monotonic() - before
            result['operation_seconds'][kind] = result['operation_seconds'].get(kind, 0) + elapsed
            emit(kind + '_failed', error_type='ContextBudgetExceeded',
                 elapsed_seconds=elapsed, **fields)
            raise _OperationFailed from None
        except Exception as exc:
            elapsed = time.monotonic() - before
            result['operation_seconds'][kind] = result['operation_seconds'].get(kind, 0) + elapsed
            emit(kind + '_failed', error_type=type(exc).__name__, elapsed_seconds=elapsed, **fields)
            fail(type(exc).__name__)
        elapsed = time.monotonic() - before
        result['operation_seconds'][kind] = result['operation_seconds'].get(kind, 0) + elapsed
        if kind == 'generation':
            result['generation_returns'] += 1
        if kind in ('inspection', 'execution'):
            result['execution_returns'] += 1
        metadata = {}
        if kind == 'generation' and type(value) is dict:
            valid = {key: type(value.get(key)) is int and value[key] >= 0
                     for key in ('prompt_tokens', 'completion_tokens')}
            metadata = {'usage_valid': all(valid.values()),
                        'prompt_tokens': value['prompt_tokens'] if valid['prompt_tokens'] else None,
                        'completion_tokens': value['completion_tokens'] if valid['completion_tokens'] else None,
                        'backend_sha256': fingerprint(value.get('backend')),
                        'response_sha256': fingerprint(value['text'])
                        if type(value.get('text')) is str else None}
        elif kind == 'execution' and type(value) is str:
            metadata = {'observation_sha256': fingerprint(value)}
        elif kind == 'inspection' and type(value) is dict:
            metadata = {'observation_sha256': fingerprint(value['observation'])
                        if type(value.get('observation')) is str else None,
                        'api_calls': value.get('api_calls') if type(value.get('api_calls')) is int else None}
        elif kind in ('completed', 'api_count'):
            metadata = {'value': value if type(value) in (bool, int) else None}
        emit(kind + '_returned', elapsed_seconds=elapsed, **fields, **metadata)
        return value

    def completed():
        value = call('completed', world.completed)
        if type(value) is not bool:
            fail('InvalidCompletedType')
        return value

    def count():
        value = call('api_count', world.api_count)
        if type(value) is not int or value < 0:
            fail('InvalidApiCount')
        return value

    def generate():
        index = result['generation_attempts']
        # The backend receives fresh containers so it cannot alter stored history.
        response = call('generation', lambda: model.generate(
            [dict(message) for message in messages], seed=17 + len(prefix_steps) + index),
            index=index, seed=17 + len(prefix_steps) + index,
            expected_backend_sha256=fingerprint(expected_backend),
            messages_sha256=fingerprint(messages))
        if type(response) is not dict:
            fail('InvalidGenerationUsage')
        valid_usage = True
        for key in ('prompt_tokens', 'completion_tokens'):
            if type(response.get(key)) is int and response[key] >= 0:
                result[key] += response[key]
            else:
                valid_usage = False
        if not valid_usage:
            fail('InvalidGenerationUsage')
        if response.get('backend') != expected_backend:
            fail('BackendMismatch', unknown_cost=False)
        if response['prompt_tokens'] > 15360 or response['completion_tokens'] > 1024:
            fail('ReturnedTokenBudgetExceeded', unknown_cost=False)
        if type(response.get('text')) is not str:
            fail('InvalidGenerationText', unknown_cost=False)
        if response.get('format_error'):
            result.update(termination='generation_format_error', ready_for_scoring=True)
            raise _OperationFailed
        return response['text']

    try:
        if completed():
            result['termination'] = 'early_completed'
            return result
        before = count()
        inspection = generate()
        try:
            parsed = parse_check_request(inspection)
        except CheckProtocolError as exc:
            emit('inspection_rejected', reason=str(exc))
            result.update(termination='inspection_format_error', ready_for_scoring=True)
            return result
        if parsed['tool'] is None:
            observation = 'No inspection API requested; no inspection evidence obtained.'
            emit('inspection_skipped')
        else:
            reply = call('inspection', lambda: world.inspect(inspection),
                         request_sha256=fingerprint(inspection))
            after = count()
            if after >= before:
                result['api_calls'] += after - before
            if (type(reply) is dict and reply.get('status') == 'rejected'
                    and type(reply.get('observation')) is str
                    and type(reply.get('api_calls')) is int and reply['api_calls'] == 0
                    and after == before):
                # A trusted worker may reject an unlisted endpoint/variable before dispatch.
                # It is a known policy stop, not an infrastructure-unknown exclusion.
                result.update(termination='inspection_request_rejected', ready_for_scoring=True)
                return result
            if (type(reply) is not dict or reply.get('status') != 'observed'
                    or type(reply.get('observation')) is not str
                    or type(reply.get('api_calls')) is not int or reply['api_calls'] <= 0
                    or after - before != reply['api_calls']):
                fail('InvalidInspectionEvidence')
            before = after
            observation = reply['observation']
            if completed():
                fail('InspectionChangedCompletion')
        messages.extend([{'role': 'assistant', 'content': inspection},
                         {'role': 'user', 'content': observation + '\n\n' + CORRECTION_PROMPT}])
        for _ in range(MAX_NEW_CALLS - 1):
            raw = generate()
            action = extract_code(raw)
            observation = call('execution', lambda code=action: world.execute(code),
                               action_sha256=fingerprint(action))
            after = count()
            if after >= before:
                result['api_calls'] += after - before
            if type(observation) is not str or after < before:
                fail('InvalidExecutionEvidence')
            before = after
            messages.extend([{'role': 'assistant', 'content': raw},
                             {'role': 'user', 'content': observation}])
            if completed():
                result.update(termination='completed', ready_for_scoring=True)
                return result
        result.update(termination='generation_budget', ready_for_scoring=True)
        return result
    except _OperationFailed:
        return result
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
