"""Own one fresh completion-check arm from replay through terminal scoring.

Only the lifecycle owner sees the terminal boolean score. The actor receives a
trace wrapper with no evaluation operation. Unknowns stop further arms; cleanup
and creator exit are required even when a score was returned. Acknowledged
public events preserve lower-bound costs if an operation or deadline fails.
"""

import os
import time
from dataclasses import asdict
from pathlib import Path

from scripts.completion_check_messages import build_check_messages
from scripts.completion_intervention import MAX_NEW_CALLS, run_intervention
from scripts.completion_replay_probe import replay_to_boundary, validate_boundary
from scripts.completion_trace import TracedModel, TracedWorld
from scripts.documentation_feedback import DocumentationFeedbackWorld
from scripts.owned_appworld_v3 import (
    _validate_intent,
    inspect_owned_container,
    remove_owned_container,
)
from scripts.run_timed_tasks import RolloutDeadline
from wll.runner import extract_code

SCORABLE_TERMINATIONS = frozenset({
    'completed', 'generation_budget', 'generation_format_error',
    'inspection_format_error', 'inspection_request_rejected',
})


def run_completion_arm(task, steps, boundary_index, model, *, arm, expected_backend,
                       tool_instructions, image, world_factory, trace, emit,
                       deadline_factory=RolloutDeadline):
    """Run once. Caller owns trace/emit writers and must honor safe_to_continue.

world_factory must create a new owned seeded worker using on_container_intent;
the callback is mandatory before process creation. Inputs contain only actor
fields, with the withheld step observation empty. No retries or GPU control.
Observed token and API counts are lower bounds when unknown_additional_cost is
true, not zero estimates for unobserved work.
"""
    validate_boundary(steps, boundary_index)
    if (any(set(step) - {'action', 'observation', 'raw_response'} for step in steps)
            or boundary_index != len(steps) - 1 or steps[boundary_index]['observation'] != ''
            or boundary_index + MAX_NEW_CALLS > 40 or type(expected_backend) is not dict
            or not expected_backend or not callable(world_factory) or not callable(emit)):
        raise ValueError('invalid_completion_arm_input')
    prefix = [dict(step) for step in steps[:boundary_index]]
    pending = steps[boundary_index].get('raw_response', steps[boundary_index]['action'])
    build_check_messages(task, prefix, pending, arm=arm, tool_instructions=tool_instructions)
    if extract_code(pending) != steps[boundary_index]['action']:
        raise ValueError('pending_action_response_mismatch')
    result = {'status': 'unknown', 'arm': arm, 'task_id': task.task_id,
              'accepted': False, 'safe_to_continue': False, 'terminal_score': None,
              'creation_verified': False, 'cleanup_verified': False,
              'unknown_additional_cost': False, 'journal_failed': False,
              'container_intents': [], 'replay': None, 'intervention': None,
              'api_attempts_observed': None, 'replay_api_attempts': None,
              'generation_usage_observed': {'prompt_tokens': 0, 'completion_tokens': 0},
              'operation_intents': {}, 'operation_returns': {}, 'phase_seconds': {}}
    started = time.monotonic()
    raw_world = None
    phase = 'startup'
    interrupt = None

    def event(kind, **fields):
        # These returns have already been persisted by the private trace before
        # the public journal is attempted, so retain their known cost lower bound.
        if kind == 'api_count_returned' and type(fields.get('value')) is int:
            value = fields['value']
            previous = result['api_attempts_observed']
            if value >= 0 and (previous is None or value >= previous):
                result['api_attempts_observed'] = value
        if kind == 'generation_returned':
            for name in ('prompt_tokens', 'completion_tokens'):
                value = fields.get(name)
                if type(value) is int and value >= 0:
                    result['generation_usage_observed'][name] += value
        try:
            emit(kind, phase=phase, arm=arm, task_id=task.task_id, **fields)
        except BaseException:
            result['journal_failed'] = True
            raise
        key = phase + '.' + kind.rsplit('_', 1)[0]
        if kind.endswith('_attempt'):
            result['operation_intents'][key] = result['operation_intents'].get(key, 0) + 1
        elif kind.endswith('_returned'):
            result['operation_returns'][key] = result['operation_returns'].get(key, 0) + 1

    def remember_interrupt(exc):
        nonlocal interrupt
        if interrupt is None and isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupt = exc

    def ownership(value):
        _validate_intent(value)
        ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
        if (result['container_intents'] or value['image'] != image
                or value['owner_pid'] != os.getpid() or value['owner_start_ticks'] != ticks):
            raise ValueError('invalid_arm_ownership')
        result['container_intents'].append(dict(value))
        event('container_intent', intent=value)

    def count(world):
        event('api_count_attempt')
        value = world.api_count()
        if type(value) is not int or value < 0:
            raise ValueError('invalid_arm_counter')
        previous = result['api_attempts_observed']
        if previous is not None and value < previous:
            raise ValueError('decreasing_arm_counter')
        event('api_count_returned', value=value)
        return value

    phase_started = time.monotonic()
    try:
        trace.record('source', task=asdict(task), prefix=prefix, pending_response=pending,
                     arm=arm, expected_backend=expected_backend)
        event('startup_attempt')
        with deadline_factory(150):
            raw_world = world_factory(on_container_intent=ownership)
            if len(result['container_intents']) != 1:
                raise ValueError('missing_arm_ownership')
            identity = inspect_owned_container(result['container_intents'][0])
            if identity is None or raw_world.worker.proc.poll() is not None:
                raise ValueError('arm_worker_not_live')
            seed = raw_world.environment_seed_provenance
            if seed != {'requested': 100, 'effective': 100, 'verified': True}:
                raise ValueError('arm_seed_mismatch')
            result['creation_verified'] = True
            result['container_id'] = identity
            event('world_ready', container_id=identity, environment_seed=seed)
        result['phase_seconds']['startup'] = time.monotonic() - phase_started
        world = TracedWorld(DocumentationFeedbackWorld(raw_world), trace)
        traced_model = TracedModel(model, trace)
        phase, phase_started = 'replay', time.monotonic()
        with deadline_factory(600):
            if count(world) != 0:
                raise ValueError('fresh_arm_counter_not_zero')
            replay = replay_to_boundary(world, steps, boundary_index, emit=event)
            result['replay'] = replay
            result['phase_seconds']['replay'] = time.monotonic() - phase_started
            if replay['status'] == 'unknown':
                result['unknown_additional_cost'] = True
            else:
                result['replay_api_attempts'] = count(world)
                result['status'] = replay['status']
            if replay['status'] == 'matched':
                phase, phase_started = 'intervention', time.monotonic()
                intervention = run_intervention(
                    task, prefix, pending, world, traced_model, arm=arm,
                    tool_instructions=tool_instructions,
                    expected_backend=expected_backend, emit=event)
                result['intervention'] = intervention
                result['unknown_additional_cost'] = intervention['unknown_additional_cost']
                result['phase_seconds']['intervention'] = time.monotonic() - phase_started
                result['status'] = 'unknown'
                event('intervention_finished', termination=intervention['termination'],
                      ready_for_scoring=intervention['ready_for_scoring'])
                if intervention['ready_for_scoring'] is True:
                    if intervention['termination'] not in SCORABLE_TERMINATIONS:
                        raise ValueError('invalid_scorable_termination')
                    count(world)
                    phase, phase_started = 'evaluation', time.monotonic()
                    event('evaluation_attempt')
                    score = raw_world.evaluate()
                    if type(score) is not bool:
                        raise TypeError('invalid_terminal_score')
                    result['terminal_score'] = score
                    event('evaluation_returned', terminal_score=score)
                    result['status'] = 'scored'
                    result['phase_seconds']['evaluation'] = time.monotonic() - phase_started
                elif intervention['termination'] == 'early_completed':
                    result['status'] = 'early_completed'
    except BaseException as exc:
        result.update(status='unknown', error_type=type(exc).__name__, unknown_additional_cost=True)
        remember_interrupt(exc)
        try:
            event('arm_failed', error_type=type(exc).__name__)
        except BaseException as nested:
            remember_interrupt(nested)
    finally:
        result['phase_seconds'].setdefault(phase, time.monotonic() - phase_started)
        phase, phase_started = 'cleanup', time.monotonic()
        try:
            event('cleanup_attempt')
        except BaseException as exc:
            result['unknown_additional_cost'] = True
            remember_interrupt(exc)
        cleanup_succeeded = True
        try:
            if raw_world is not None:
                raw_world.close()
            else:
                for intent in result['container_intents']:
                    remove_owned_container(intent)
        except BaseException as exc:
            cleanup_succeeded = False
            result['cleanup_error_type'] = type(exc).__name__
            remember_interrupt(exc)
        try:
            result['owned_containers_absent'] = [inspect_owned_container(intent) is None
                                                 for intent in result['container_intents']]
        except BaseException as exc:
            cleanup_succeeded = False
            result.setdefault('cleanup_error_type', type(exc).__name__)
            remember_interrupt(exc)
        try:
            result['creator_exit_code'] = raw_world.worker.proc.poll() if raw_world else None
        except BaseException as exc:
            cleanup_succeeded = False
            result.setdefault('cleanup_error_type', type(exc).__name__)
            remember_interrupt(exc)
        result['cleanup_verified'] = (cleanup_succeeded and result['creation_verified']
            and result.get('creator_exit_code') is not None
            and result.get('owned_containers_absent') == [True])
        if not result['cleanup_verified']:
            result['unknown_additional_cost'] = True
        result['phase_seconds']['cleanup'] = time.monotonic() - phase_started
        try:
            event('cleanup_returned', verified=result['cleanup_verified'])
        except BaseException as exc:
            result['unknown_additional_cost'] = True
            remember_interrupt(exc)
    result['accepted'] = (result['status'] == 'scored' and result['cleanup_verified']
                          and not result['journal_failed'])
    result['safe_to_continue'] = (result['status'] in {'scored', 'observation_mismatch', 'early_completed'}
                                  and result['cleanup_verified'] and not result['journal_failed'])
    result['wall_seconds'] = time.monotonic() - started
    if interrupt is not None:
        raise interrupt
    return result
