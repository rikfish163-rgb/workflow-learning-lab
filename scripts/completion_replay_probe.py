"""Replay actor-visible observations up to, but never through, a completion action.

This is a CPU diagnostic, not a rollout or an eligibility decision. The caller
owns a fresh constrained world, durable events, deadlines and verified cleanup.
No model, evaluator, task solution or hidden success label is accepted here.
"""

import time

from scripts.completion_boundary import inspect_completion_action
from scripts.run_timed_tasks import RolloutTimedOut
from wll.protocol import fingerprint


def validate_boundary(steps, boundary_index):
    """Check the entire supplied record before any world operation."""
    if (type(steps) is not list or not steps or type(boundary_index) is not int
            or not 0 <= boundary_index < len(steps)):
        raise ValueError('invalid completion boundary')
    if any(type(step) is not dict or type(step.get('action')) is not str
           or type(step.get('observation')) is not str for step in steps):
        raise ValueError('invalid actor-visible step')
    if inspect_completion_action(steps[boundary_index]['action'])['status'] != 'candidate':
        raise ValueError('completion action is not an isolated supported candidate')
    if any(inspect_completion_action(step['action'])['status'] != 'no_match'
           for step in steps[:boundary_index]):
        raise ValueError('earlier action has a completion signal or cannot be classified')


def replay_to_boundary(world, steps, boundary_index, *, emit):
    """Replay only preceding whole actions and compare exact visible strings.

Ordinary world failures return unknown; a deadline or interruption propagates
to the owning caller. Event failures also propagate before further operations.
The syntactic candidate is withheld entirely and no evaluate() call is made.
Matching observations do not prove all hidden state or dynamic callees equal.
"""
    validate_boundary(steps, boundary_index)
    if not callable(emit):
        raise ValueError('durable event callback required')
    result = {'status': 'unknown', 'attempted': 0, 'returned': 0, 'matched': 0,
              'completed_checks': 0, 'boundary_completed': None,
              'elapsed_seconds': 0.0, 'execute_seconds': 0.0,
              'completed_seconds': 0.0, 'unknown_additional_cost': False}
    started = time.monotonic()

    def completed(index):
        emit('completed_attempt', index=index)
        result['completed_checks'] += 1
        before = time.monotonic()
        try:
            value = world.completed()
        except RolloutTimedOut:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - before
            result['completed_seconds'] += elapsed
            result['error_type'] = type(exc).__name__
            result['unknown_additional_cost'] = True
            emit('completed_failed', index=index, elapsed_seconds=elapsed,
                 error_type=type(exc).__name__)
            return None
        elapsed = time.monotonic() - before
        result['completed_seconds'] += elapsed
        valid = type(value) is bool
        emit('completed_returned', index=index, elapsed_seconds=elapsed,
             valid_boolean=valid, completed=value if valid else None)
        if not valid:
            result['error_type'] = 'InvalidCompletedType'
            return None
        if value:
            result.update(status='early_completed', boundary_completed=True)
        return value

    try:
        if completed(0) is not False:
            return result
        for index, step in enumerate(steps[:boundary_index]):
            emit('replay_attempt', index=index, action_sha256=fingerprint(step['action']))
            result['attempted'] += 1
            before = time.monotonic()
            try:
                observation = world.execute(step['action'])
            except RolloutTimedOut:
                raise
            except Exception as exc:
                elapsed = time.monotonic() - before
                result['execute_seconds'] += elapsed
                result['error_type'] = type(exc).__name__
                result['unknown_additional_cost'] = True
                emit('replay_failed', index=index, elapsed_seconds=elapsed,
                     error_type=type(exc).__name__)
                return result
            elapsed = time.monotonic() - before
            result['execute_seconds'] += elapsed
            result['returned'] += 1
            valid = type(observation) is str
            emit('replay_returned', index=index, elapsed_seconds=elapsed,
                 valid_text=valid, matches_expected=valid and observation == step['observation'],
                 observation_sha256=fingerprint(observation) if valid else None)
            if not valid:
                result['error_type'] = 'InvalidObservationType'
                return result
            if observation != step['observation']:
                result['status'] = 'observation_mismatch'
                emit('replay_mismatch', index=index,
                     expected_sha256=fingerprint(step['observation']),
                     actual_sha256=fingerprint(observation))
                return result
            result['matched'] += 1
            if completed(index + 1) is not False:
                return result
        result.update(status='matched', boundary_completed=False)
        return result
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
