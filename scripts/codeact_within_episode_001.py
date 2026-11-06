"""One bounded private CodeAct attempt over an owner-supplied fresh world.

The owner creates, authenticates and closes the world, model and private durable
recorder. This module sends code only to ``world.execute``; it executes none on
the host. It neither mines targets nor decides source qualification.
"""

import copy
import math
import time

from wll.model import ContextBudgetExceeded
from wll.prompts import initial_messages
from wll.runner import extract_code

MAX_GENERATIONS = MAX_ACTIONS = 40
ACTOR_SECONDS = 1200
MAX_INPUT, MAX_OUTPUT, MAX_API = 15360, 1024, 500
_ERROR_TYPES = frozenset({'Exception', 'ArithmeticError', 'AssertionError', 'AttributeError',
    'ImportError', 'IndexError', 'KeyError', 'LookupError', 'NameError', 'OverflowError',
    'RuntimeError', 'SyntaxError', 'TypeError', 'UnboundLocalError', 'ValueError',
    'ZeroDivisionError', 'other'})


class _RecordFailure(Exception):
    pass


class _ClockFailure(Exception):
    pass


def _emit(record, event, **fields):
    try:
        record(event, **copy.deepcopy(fields))
    except Exception:
        raise _RecordFailure from None


def _response_valid(value):
    if (type(value) is not dict or type(value.get('text')) is not str
            or type(value.get('prompt_tokens')) is not int or not 1 <= value['prompt_tokens'] <= MAX_INPUT
            or type(value.get('completion_tokens')) is not int or not 1 <= value['completion_tokens'] <= MAX_OUTPUT
            or type(value.get('finish_reason')) is not str
            or value['finish_reason'] not in {'stop', 'length', 'unknown'}
            or 'format_error' not in value
            or (value['format_error'] is not None and type(value['format_error']) is not str)):
        return False
    try:
        value['text'].encode('utf-8')
    except UnicodeError:
        return False
    return True


def _worker_valid(value, previous):
    if (type(value) is not dict or set(value) != {'observation', 'execution', 'api'}
            or type(value['observation']) is not str):
        return False
    execution, api = value['execution'], value['api']
    if (type(execution) is not dict or set(execution) != {'status', 'error_type', 'hook_calls'}
            or type(execution['status']) is not str or type(execution['hook_calls']) is not int
            or execution['hook_calls'] < 0 or type(api) is not dict
            or set(api) != {'attempts', 'request_dispatches', 'budget_rejections', 'limit'}
            or any(type(v) is not int or v < 0 for v in api.values())
            or api['limit'] != MAX_API or api['request_dispatches'] > MAX_API
            or api['attempts'] != api['request_dispatches'] + api['budget_rejections']
            or api['request_dispatches'] < previous['api_request_dispatches']
            or api['budget_rejections'] < previous['budget_rejections']):
        return False
    status, error, hooks = execution['status'], execution['error_type'], execution['hook_calls']
    if status == 'success':
        valid = error is None and hooks == 1
    elif status in {'error_before_exec', 'error_in_exec'}:
        valid = type(error) is str and error in _ERROR_TYPES and hooks == 1
    elif status in {'policy_rejected', 'unobserved'}:
        valid = (error is None and hooks == 0
                 and api['request_dispatches'] == previous['api_request_dispatches'])
    elif status == 'timeout':
        valid = (error is None or (type(error) is str and error == 'TimeoutError')) and hooks == 1
    elif status == 'unknown':
        valid = error is None
    elif status == 'api_budget_rejected':
        valid = type(error) is str and error == 'ApiBudgetExceeded' and hooks in (0, 1)
    else:
        return False
    if api['budget_rejections'] > previous['budget_rejections']:
        valid = valid and status in {'timeout', 'unknown', 'api_budget_rejected'}
    elif status == 'api_budget_rejected':
        return False
    return valid


def run_episode(task, world, model, *, record, seed=17, experience=None, _clock=time.monotonic):
    if (type(seed) is not int or seed != 17 or not callable(record) or not callable(_clock)
            or (experience is not None and type(experience) is not str)
            or any(not callable(getattr(world, name, None)) for name in ('api_count', 'completed', 'execute', 'evaluate'))
            or not callable(getattr(model, 'generate', None))):
        raise ValueError('codeact_episode_configuration')
    messages = initial_messages(task, 'appworld-v3')
    if experience:
        messages.append({'role': 'user', 'content':
                         'A previous training experience may apply; check its conditions:\n' + experience})
    steps = []
    attempts = returns = actions = 0
    known = {'prompt_tokens': 0, 'completion_tokens': 0,
             'api_request_dispatches': 0, 'budget_rejections': 0}
    unknown_model = unknown_api = journal_failed = clock_failed = False
    reported = None
    start = last_clock = elapsed = None

    def now():
        nonlocal last_clock
        try:
            value = _clock()
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or (last_clock is not None and value < last_clock)):
                raise ValueError
            last_clock = float(value)
            return last_clock
        except Exception:
            raise _ClockFailure from None

    def outcome(termination, official=None):
        counts = {'generation_attempts': attempts, 'generation_returns': returns,
                  'prompt_tokens': None if unknown_model else known['prompt_tokens'],
                  'completion_tokens': None if unknown_model else known['completion_tokens'],
                  'actions': actions,
                  'api_request_dispatches': None if unknown_api else known['api_request_dispatches'],
                  'budget_rejections': None if unknown_api else known['budget_rejections']}
        return {'private': {'messages': copy.deepcopy(messages), 'steps': copy.deepcopy(steps)},
                'outcome': {'counts': counts, 'known_costs': dict(known), 'termination': termination,
                            'model_reported_complete': reported, 'official_success': official,
                            'costs_known': not (unknown_model or unknown_api or journal_failed or clock_failed),
                            'actor_wall_seconds': elapsed}}

    def finish(termination, *, score=False):
        nonlocal journal_failed, clock_failed, elapsed
        if start is not None and not clock_failed:
            try:
                elapsed = now() - start
            except _ClockFailure:
                clock_failed, score, termination = True, False, 'clock_failure'
        try:
            _emit(record, 'actor_stop', termination=termination, model_reported_complete=reported,
                  known_costs=known, generation_attempts=attempts, generation_returns=returns, actions=actions)
        except _RecordFailure:
            journal_failed = True
            return outcome('record_failure')
        if not score:
            return outcome(termination)
        try:
            official = world.evaluate()
            if type(official) is not bool:
                raise ValueError('codeact_episode_score')
        except Exception:
            return outcome('scoring_failure')
        return outcome(termination, official)

    try:
        start = now()
        _emit(record, 'actor_start', messages=messages, seed=seed)
        initial_api = world.api_count()
        if type(initial_api) is not int or initial_api != 0:
            if type(initial_api) is int and 0 < initial_api <= MAX_API:
                known['api_request_dispatches'] = initial_api
            unknown_api = True
            return finish('initial_world_binding_failure')
        done = world.completed()
        if type(done) is not bool:
            unknown_api = True
            return finish('completion_failure')
        reported = False
        if done:
            return finish('initial_completion', score=True)

        while True:
            if now() - start >= ACTOR_SECONDS:
                return finish('actor_wall', score=True)
            if attempts >= MAX_GENERATIONS:
                return finish('generation_budget', score=True)
            if actions >= MAX_ACTIONS:
                return finish('action_budget', score=True)
            index = attempts
            step = {'index': index, 'seed': seed + index, 'messages_before': copy.deepcopy(messages),
                    'response': None, 'action': None, 'execution': None}
            steps.append(step)
            _emit(record, 'generation_intent', index=index, seed=seed + index, messages=messages)
            try:
                admitted = now() - start < ACTOR_SECONDS
            except _ClockFailure:
                _emit(record, 'generation_not_dispatched', index=index, reason='clock_failure')
                raise
            if not admitted:
                _emit(record, 'generation_not_dispatched', index=index, reason='actor_wall')
                return finish('actor_wall', score=True)
            attempts += 1
            try:
                response = model.generate(copy.deepcopy(messages), seed=seed + index)
            except ContextBudgetExceeded:
                _emit(record, 'generation_not_dispatched', index=index, reason='context_budget')
                return finish('context_budget', score=True)
            except Exception:
                unknown_model = True
                return finish('model_failure')
            returns += 1
            step['response'] = copy.deepcopy(response)
            if not _response_valid(response):
                unknown_model = True
                _emit(record, 'generation_return', index=index, response=response)
                return finish('invalid_model_response')
            known['prompt_tokens'] += response['prompt_tokens']
            known['completion_tokens'] += response['completion_tokens']
            _emit(record, 'generation_return', index=index, response=response)
            if response['format_error'] is not None:
                return finish('generation_format_error', score=True)
            if response['finish_reason'] != 'stop':
                reason = 'generation_length' if response['finish_reason'] == 'length' else 'generation_unknown_finish'
                return finish(reason, score=True)

            action = extract_code(response['text'])
            step['action'] = action
            _emit(record, 'action_intent', index=index, action=action)
            actions += 1
            reported = None
            try:
                executed = world.execute(action)
            except Exception:
                unknown_api, reported = True, None
                return finish('execution_failure')
            if not _worker_valid(executed, known):
                unknown_api, reported = True, None
                _emit(record, 'action_result', index=index, result=executed)
                return finish('invalid_worker_result')
            api = executed['api']
            known['api_request_dispatches'], known['budget_rejections'] = (
                api['request_dispatches'], api['budget_rejections'])
            step['execution'] = copy.deepcopy(executed)
            _emit(record, 'action_result', index=index, result=executed)
            messages.extend([{'role': 'assistant', 'content': response['text']},
                             {'role': 'user', 'content': executed['observation']}])
            current_api = world.api_count()
            if type(current_api) is not int or current_api != known['api_request_dispatches']:
                unknown_api, reported = True, None
                return finish('invalid_worker_result')
            status = executed['execution']['status']
            if status in {'timeout', 'unknown', 'api_budget_rejected'}:
                reported = None
                return finish('execution_' + status)
            done = world.completed()
            if type(done) is not bool:
                reported = None
                return finish('completion_failure')
            reported = done
            if done:
                return finish('completed', score=True)
    except _RecordFailure:
        journal_failed = True
        if start is not None:
            try:
                elapsed = now() - start
            except _ClockFailure:
                clock_failed = True
        return outcome('record_failure')
    except _ClockFailure:
        clock_failed = True
        return finish('clock_failure')
    except Exception:
        # A failed trusted count/clock/probe is not a known model failure.
        unknown_api, reported = True, None
        return finish('controller_failure')
