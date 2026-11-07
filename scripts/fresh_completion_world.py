"""Own one fresh world and return an opaque, owner-only callback result.

The supplied factory must record its ownership intent before creating a worker.
The callback receives a narrow world facade and an event writer; it must send
only public metadata to the writer. Actor text and scores belong only in its
private records or return value. Callback outcome semantics remain the caller's
responsibility; ``safe_to_continue`` attests only this world's lifecycle.
``unknown_additional_cost`` covers lifecycle and world-operation uncertainty;
the caller must also account for any uncertainty in the opaque callback result.
"""

import copy
import math
import os
import re
import time
from pathlib import Path

from scripts.completion_trace import TraceWriteError
from scripts.owned_appworld_v3 import (
    _validate_intent,
    inspect_owned_container,
    remove_owned_container,
)
from scripts.run_timed_tasks import RolloutDeadline

_INTERRUPTS = (KeyboardInterrupt, SystemExit, GeneratorExit)


def _error_type(exc):
    name = type(exc).__name__
    return name if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}', name) else 'OtherException'


def run_owned_world(*, image, world_factory, emit, operation,
                    deadline_factory=RolloutDeadline, operation_timeout=1800):
    """Create, verify, operate once, and verify cleanup of one owned worker.

    ``operation(world, event)`` receives only ``execute``, ``completed``,
    ``api_count``, ``evaluate`` and seed provenance on its world facade.
    Evaluation snapshots the independently trusted API counter before it makes
    the worker terminal; no later facade operation or second evaluation is
    permitted. The returned operation value is never emitted or interpreted.
    """
    if (type(image) is not str or re.fullmatch(r'sha256:[0-9a-f]{64}', image) is None
            or any(not callable(value) for value in (world_factory, emit, operation, deadline_factory))
            or type(operation_timeout) not in (int, float) or not math.isfinite(operation_timeout)
            or operation_timeout <= 0):
        raise ValueError('invalid_owned_world_input')
    ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    result = {'status': 'unknown', 'safe_to_continue': False, 'creation_verified': False,
              'cleanup_verified': False, 'journal_failed': False, 'unknown_additional_cost': False,
              'container_intents': [], 'container_id': None, 'creator_exit_code': None,
              'owned_containers_absent': [], 'api_attempts_observed': None,
              'operation_result': None, 'phase_seconds': {}, 'error_type': None,
              'cleanup_error_type': None}
    started = time.monotonic()
    phase, phase_started = 'startup', started
    raw_world = None
    interrupted = None
    operation_returned = False

    def remember(exc, *, cleanup=False):
        nonlocal interrupted
        name = _error_type(exc)
        if result['error_type'] is None:
            result['error_type'] = name
        if cleanup and result['cleanup_error_type'] is None:
            result['cleanup_error_type'] = name
        if isinstance(exc, TraceWriteError):
            result['journal_failed'] = True
        if interrupted is None and isinstance(exc, _INTERRUPTS):
            interrupted = exc
        result['unknown_additional_cost'] = True

    def event(kind, **fields):
        try:
            emit(kind, **copy.deepcopy(fields))
        except BaseException as exc:
            result['journal_failed'] = True
            remember(exc)
            raise

    def ownership(intent):
        try:
            _validate_intent(intent)
            if (result['container_intents'] or intent['image'] != image
                    or intent['owner_pid'] != os.getpid() or intent['owner_start_ticks'] != ticks):
                raise ValueError('invalid_owned_world_intent')
            result['container_intents'].append(dict(intent))
            event('container_intent', intent=intent)
        except BaseException as exc:
            remember(exc)
            raise

    def record_count(value):
        if (type(value) is not int or value < 0
                or (result['api_attempts_observed'] is not None
                    and value < result['api_attempts_observed'])):
            raise ValueError('invalid_owned_world_counter')
        result['api_attempts_observed'] = value
        return value

    def trusted_count():
        event('owner_api_count_attempt')
        # Preserve a returned observation even if its public journal write fails.
        value = record_count(raw_world.api_count())
        event('owner_api_count_returned', value=value)
        return value

    class OwnerWorld:
        def __init__(self, raw, seed):
            self._raw = raw
            self._seed = dict(seed)
            self.terminal = False
            self.fault = False

        @property
        def environment_seed_provenance(self):
            return dict(self._seed)

        def _check(self):
            if self.terminal or self.fault or result['journal_failed']:
                exc = RuntimeError('owned_world_unavailable')
                self.fault = True
                remember(exc)
                raise exc

        def _call(self, name, *args):
            self._check()
            try:
                return getattr(self._raw, name)(*args)
            except BaseException as exc:
                self.fault = True
                remember(exc)
                raise

        def execute(self, code):
            return self._call('execute', code)

        def completed(self):
            return self._call('completed')

        def api_count(self):
            value = self._call('api_count')
            try:
                return record_count(value)
            except BaseException as exc:
                self.fault = True
                remember(exc)
                raise

        def evaluate(self):
            self._check()
            try:
                trusted_count()
            except BaseException as exc:
                self.fault = True
                remember(exc)
                raise
            self.terminal = True  # The worker is terminal even if evaluation raises.
            try:
                return self._raw.evaluate()
            except BaseException as exc:
                self.fault = True
                remember(exc)
                raise

    try:
        event('world_startup_attempt')
        with deadline_factory(150):
            raw_world = world_factory(on_container_intent=ownership)
            if result['error_type'] is not None:
                raise RuntimeError('owned_world_intent_failed')
            if len(result['container_intents']) != 1:
                raise ValueError('missing_owned_world_intent')
            identity = inspect_owned_container(result['container_intents'][0])
            if (type(identity) is not str or re.fullmatch('[0-9a-f]{64}', identity) is None
                    or raw_world.worker.proc.poll() is not None):
                raise ValueError('owned_world_not_live')
            seed = raw_world.environment_seed_provenance
            if (type(seed) is not dict or set(seed) != {'requested', 'effective', 'verified'}
                    or type(seed['requested']) is not int or seed['requested'] != 100
                    or type(seed['effective']) is not int or seed['effective'] != 100
                    or seed['verified'] is not True):
                raise ValueError('owned_world_seed_mismatch')
            result['creation_verified'] = True
            result['container_id'] = identity
            if trusted_count() != 0:
                raise ValueError('owned_world_not_fresh')
            event('world_ready', container_id=identity, environment_seed=dict(seed))
        result['phase_seconds']['startup'] = time.monotonic() - phase_started
        phase, phase_started = 'operation', time.monotonic()
        facade = OwnerWorld(raw_world, seed)
        with deadline_factory(operation_timeout):
            event('world_operation_attempt')
            result['operation_result'] = operation(facade, event)
            operation_returned = True
            if facade.fault or result['journal_failed']:
                raise RuntimeError('owned_world_operation_incomplete')
            if not facade.terminal:
                trusted_count()
            event('world_operation_returned')
    except BaseException as exc:
        remember(exc)
        try:
            event('world_failed', error_type=_error_type(exc))
        except BaseException as nested:
            remember(nested)
    finally:
        result['phase_seconds'].setdefault(phase, time.monotonic() - phase_started)
        phase_started = time.monotonic()
        cleanup_succeeded = True
        try:
            event('world_cleanup_attempt')
        except BaseException as exc:
            remember(exc, cleanup=True)
        try:
            if raw_world is not None:
                raw_world.close()
            else:
                for intent in result['container_intents']:
                    remove_owned_container(intent)
        except BaseException as exc:
            cleanup_succeeded = False
            remember(exc, cleanup=True)
            if raw_world is not None:
                for intent in result['container_intents']:
                    try:
                        remove_owned_container(intent)
                    except BaseException as nested:
                        remember(nested, cleanup=True)
        try:
            result['owned_containers_absent'] = [inspect_owned_container(intent) is None
                                                 for intent in result['container_intents']]
        except BaseException as exc:
            cleanup_succeeded = False
            remember(exc, cleanup=True)
        try:
            result['creator_exit_code'] = raw_world.worker.proc.poll() if raw_world is not None else None
        except BaseException as exc:
            cleanup_succeeded = False
            remember(exc, cleanup=True)
        result['cleanup_verified'] = (cleanup_succeeded and result['creation_verified']
            and type(result['creator_exit_code']) is int
            and result['owned_containers_absent'] == [True])
        if not result['cleanup_verified']:
            result['unknown_additional_cost'] = True
        result['phase_seconds']['cleanup'] = time.monotonic() - phase_started
        try:
            event('world_cleanup_returned', verified=result['cleanup_verified'])
        except BaseException as exc:
            remember(exc, cleanup=True)

    if (operation_returned and result['error_type'] is None and result['cleanup_verified']
            and not result['journal_failed']):
        result.update(status='complete', safe_to_continue=True)
    result['wall_seconds'] = time.monotonic() - started
    if interrupted is not None:
        raise interrupted
    return result
