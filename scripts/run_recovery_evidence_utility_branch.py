"""One private, fresh utility branch; no scheduler, registration or CLI.

The caller supplies authenticated preparation, an exact token counter, a durable
public metadata writer and a current owned-world factory. Raw actor material and
the owner-only Episode remain in a new private directory. A returned ``unknown``
never authorizes another branch; observed cost is a lower bound on failed calls.
"""

import copy
import hashlib
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path

from scripts import fresh_completion_world as owner
from scripts import recovery_evidence_utility as utility
from scripts.completion_trace import (
    PrivateTrace,
    TracedModel,
    TracedWorld,
    TraceWriteError,
    _exception_type,
    _parent_fd,
)
from scripts.run_observed_utility_branch import initial_actor_messages
from scripts.run_timed_tasks import CheckedModel, RolloutDeadline
from wll.experience import Experience
from wll.model import ContextBudgetExceeded
from wll.protocol import Task, fingerprint
from wll.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]
_INTERRUPTS = (KeyboardInterrupt, SystemExit, GeneratorExit)
SOURCES = {
    'wll/runner.py': 'b954a1000b2e81f9248254ee2aa448eebfa5b3babb013f652be3eb0ccb37c640',
    'scripts/run_timed_tasks.py': '5540b99b5b8ce36aa68b107195c6981dfca5e6ebed3b6c8f4787f1cd0e0ba303',
    'scripts/run_observed_utility_branch.py': '34e91ad7c4ac44f714135ee71e419f57c58de557f6028fe17ce3a821e581d419',
    'scripts/fresh_completion_world.py': 'd17336cd2445e96e9e08b731d07273b5d751bfd92bfce956da0399496ed98b8a',
    'scripts/completion_trace.py': 'e060b5f93eeaa9116d2d44a948f36d5a018937809fcf02b4733bcdd859abc474',
}


def verify_sources():
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha
           for name, sha in SOURCES.items()):
        raise ValueError('utility_branch_source_drift')


def _require(condition):
    if not condition:
        raise ValueError('utility_branch_invalid')


def _admit(task, experience, model, arm, protocol, expected_backend):
    _require(type(protocol) is dict and all(type(protocol.get(key)) is type(value)
        and protocol[key] == value for key, value in utility.FIXED.items() if key != 'design')
        and protocol.get('comparison_view') == utility.COMPARISON_VIEW
        and protocol.get('thresholds') == utility.THRESHOLDS
        and protocol.get('reviews') == {'candidate': True, 'placebo': True, 'placebo_unrelated': True})
    utility._validate_protocol_provenance(protocol, experience if arm == utility.ARMS[1] else None)
    _require(isinstance(task, Task) and task.task_id in utility.TASK_IDS and task.split == 'train'
        and task.scenario_id == task.task_id.rsplit('_', 1)[0]
        and set(protocol.get('task_sha256', {})) == set(utility.TASK_IDS)
        and protocol['task_sha256'][task.task_id] == fingerprint(asdict(task))
        and type(protocol.get('environment_id')) is str and bool(protocol['environment_id'].strip()))
    utility._contexts(protocol['initial_input_tokens'], protocol['lesson_message_tokens'])
    utility._backend(expected_backend)
    _require(utility.static_backend(expected_backend) == protocol.get('expected_backend_static')
        and model.model_id == utility.MODEL_ID and type(model.max_tokens) is int and model.max_tokens == 1024
        and type(model.temperature) in (int, float) and model.temperature == .7
        and type(model.timeout) in (int, float) and model.timeout == 180 and arm in utility.ARMS)
    if arm == utility.ARMS[0]:
        _require(experience is None)
    else:
        key = 'candidate' if arm == utility.ARMS[1] else 'placebo'
        _require(isinstance(experience, Experience)
            and experience.experience_id == protocol.get(key + '_id')
            and fingerprint(asdict(experience)) == protocol.get(key + '_sha256'))
        if key == 'placebo':
            provenance = protocol.get('placebo_provenance', {})
            _require(experience.source_scenarios == [] and experience.source_episode_hashes == [])
            utility._validate_placebo_provenance(provenance, protocol['design'], placebo=experience)


def _reserve(output):
    """Reserve one new leaf, without repairing permissions or following links."""
    directory = None
    try:
        output = Path(output)
        _require(output.is_absolute() and not output.is_relative_to(ROOT))
        directory, output = _parent_fd(output)
        os.mkdir(output.name, 0o700, dir_fd=directory)
        fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                     dir_fd=directory)
        try:
            os.fchmod(fd, 0o700)
            info = os.fstat(fd)
            _require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700)
            os.fsync(fd)
            os.fsync(directory)
        finally:
            os.close(fd)
        return output
    except (OSError, TypeError):
        raise ValueError('utility_branch_private_output_unavailable') from None
    finally:
        if directory is not None:
            os.close(directory)


def _write_private(path, value):
    directory, path = _parent_fd(path)
    fd = None
    try:
        raw = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True).encode('ascii') + b'\n'
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=directory)
        os.fchmod(fd, 0o600)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise ValueError('invalid_private_file')
        remaining = memoryview(raw)
        while remaining:
            count = os.write(fd, remaining)
            if count <= 0:
                raise OSError('private_write_no_progress')
            remaining = remaining[count:]
        os.fsync(fd)
        os.fsync(directory)
        return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    except Exception:
        raise TraceWriteError('utility_branch_private_write_failed') from None
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


class _RemoteContextFailure(RuntimeError):
    """A dispatched rejection has unknown response cost, unlike a local count."""


class _ForwardingTracedModel(TracedModel):
    def __getattr__(self, name):
        return getattr(self._model, name)


def run_branch(task, experience, world_factory, model, *, output, arm, protocol,
               expected_backend, count_tokens, emit, before_generation,
               deadline_factory=RolloutDeadline):
    """Run one first attempt, returning public metadata and private-file hashes.

    ``emit`` must durably persist each event before returning. ``count_tokens``
    must count the actual full messages with the registered tokenizer/template.
    The caller remains responsible for freezing these callback implementations,
    task loading, GPU admission/health and the later controls-first phase gate.
    """
    verify_sources()
    _admit(task, experience, model, arm, protocol, expected_backend)
    _require(all(callable(value) for value in (world_factory, count_tokens, emit,
                                              before_generation, deadline_factory)))
    output = _reserve(output)
    cost = {'generation_attempts': 0, 'generation_returns': 0,
        'generation_usage_observed': {'prompt_tokens': 0, 'completion_tokens': 0},
        'generation_returns_with_unusable_usage': 0, 'context_count_attempts': 0,
        'context_count_returns': 0, 'context_rejections': 0, 'execution_attempts': 0,
        'execution_returns': 0, 'evaluation_attempts': 0, 'evaluation_returns': 0,
        'api_attempts_observed': None}
    legacy = {'generation_calls': 0, 'generation_calls_without_response': 0,
              'generation_calls_with_unusable_usage': 0,
              'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0}}
    state = {'journal_failed': False, 'world_completed': None, 'episode': None,
             'private_files': {}, 'error_type': None}
    trace, owned, interrupted = None, None, None
    initial_sha = fingerprint(initial_actor_messages(task, experience, protocol['prompt_version']))

    def public(event, **fields):
        # Only this adapter's explicit projections and the trusted owner's
        # metadata reach the caller. Never forward arbitrary model/world fields.
        try:
            emit(event, **fields)
        except BaseException as exc:
            state['journal_failed'] = True
            if isinstance(exc, _INTERRUPTS):
                raise
            raise TraceWriteError('utility_branch_public_write_failed') from None
        if event == 'owner_api_count_returned':
            cost['api_attempts_observed'] = fields['value']

    class ActorJournal:
        def record(self, event, **fields):
            try:
                trace.record(event, **fields)
                safe = {}
                for key in ('seed', 'tokens', 'error_type', 'step_index', 'replayed'):
                    if key in fields:
                        safe[key] = fields[key]
                for key in ('messages', 'response', 'code', 'result', 'step'):
                    if key in fields:
                        safe[key + '_sha256'] = fingerprint(fields[key])
                public(event, **safe)
            except BaseException:
                state['journal_failed'] = True
                raise

    journal = ActorJournal()

    class ObservedModel:
        def __getattr__(self, name):
            return getattr(model, name)

        def generate(self, messages, seed):
            # Called only after TracedModel has fsynced its dispatch intent.
            cost['generation_attempts'] += 1
            try:
                response = model.generate(messages, seed=seed)
            except ContextBudgetExceeded:
                raise _RemoteContextFailure('dispatched_context_rejection') from None
            cost['generation_returns'] += 1
            usable = True
            for key in ('prompt_tokens', 'completion_tokens'):
                value = response.get(key) if type(response) is dict else None
                if type(value) is int and value >= 0:
                    cost['generation_usage_observed'][key] += value
                else:
                    usable = False
            if not usable:
                cost['generation_returns_with_unusable_usage'] += 1
            return response

    checked = CheckedModel(_ForwardingTracedModel(ObservedModel(), journal), expected_backend,
                           legacy, lambda value: public('response_checked', **value))

    class ContextCheckedModel:
        def __getattr__(self, name):
            return getattr(checked, name)

        def generate(self, messages, seed):
            first = cost['context_count_attempts'] == 0
            if first:
                _require(fingerprint(messages) == initial_sha)
            journal.record('context_count_attempt', messages=messages)
            cost['context_count_attempts'] += 1
            try:
                tokens = count_tokens(copy.deepcopy(messages))
            except ContextBudgetExceeded:
                raise ValueError('context_counter_failed') from None
            cost['context_count_returns'] += 1
            _require(type(tokens) is int and tokens > 0)
            journal.record('context_count_returned', tokens=tokens)
            if first:
                _require(tokens == protocol['initial_input_tokens'][task.task_id][arm])
            if tokens > protocol['max_input_tokens']:
                cost['context_rejections'] += 1
                journal.record('context_budget_rejected', tokens=tokens)
                raise ContextBudgetExceeded()
            try:
                before_generation()
            except ContextBudgetExceeded:
                raise ValueError('generation_preflight_failed') from None
            response = checked.generate(messages, seed=seed)
            _require(type(response['text']) is str and response['prompt_tokens'] <= protocol['max_input_tokens']
                and response['prompt_tokens'] == tokens
                and response['completion_tokens'] <= protocol['max_tokens']
                and (response.get('format_error') is None or type(response['format_error']) is str))
            return response

    def operation(world, _event):
        class ObservedWorld:
            def execute(self, code):
                cost['execution_attempts'] += 1
                result = world.execute(code)
                cost['execution_returns'] += 1
                _require(type(result) is str)
                return result

            def completed(self):
                value = world.completed()
                _require(type(value) is bool)
                state['world_completed'] = value
                return value

            def api_count(self):
                value = world.api_count()
                cost['api_attempts_observed'] = value
                return value

        actor_world = TracedWorld(ObservedWorld(), journal)

        class RolloutWorld:
            def execute(self, code):
                result = actor_world.execute(code)
                actor_world.api_count()
                return result

            def completed(self):
                return actor_world.completed()

            def evaluate(self):
                public('evaluation_attempt')
                cost['evaluation_attempts'] += 1
                score = world.evaluate()
                cost['evaluation_returns'] += 1
                public('evaluation_returned', valid_boolean=type(score) is bool)
                _require(type(score) is bool)
                return score

            def close(self):
                pass  # The owner alone closes the real worker, outside its deadline.

        def observe(index, step, replayed):
            _require(replayed is False)
            journal.record('step_recorded', step_index=index, step=asdict(step), replayed=False)

        _require(actor_world.completed() is False)
        episode = run_episode(task, RolloutWorld, ContextCheckedModel(), max_steps=40,
            seed=17, prefix=(), branch='first', on_step=observe,
            experience=experience.render() if experience is not None else None,
            experience_id=experience.experience_id if experience is not None else None,
            prompt_version=protocol['prompt_version'])
        episode.metadata.update({key: copy.deepcopy(protocol[key]) for key in utility.EPISODE_SETTINGS})
        episode.metadata.update(arm=arm, world_completed=state['world_completed'])
        state['private_files']['episode.json'] = _write_private(output / 'episode.json', episode.to_dict())
        state['episode'] = episode

    try:
        trace = PrivateTrace(output / 'actor.jsonl')
        owned = owner.run_owned_world(image=protocol['environment_image'], world_factory=world_factory,
            emit=public, operation=operation, deadline_factory=deadline_factory, operation_timeout=600)
    except BaseException as exc:
        state['error_type'] = _exception_type(exc)
        state['journal_failed'] |= isinstance(exc, TraceWriteError)
        if isinstance(exc, _INTERRUPTS):
            interrupted = exc
    finally:
        if trace is not None:
            try:
                trace.close()
                raw = (output / 'actor.jsonl').read_bytes()
                state['private_files']['actor.jsonl'] = {
                    'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            except BaseException as exc:
                state['journal_failed'] = True
                state['error_type'] = _exception_type(exc)
                if interrupted is None and isinstance(exc, _INTERRUPTS):
                    interrupted = exc
    owned = owned or {}
    journal_failed = state['journal_failed'] or owned.get('journal_failed', False)
    released = owned.get('cleanup_verified') is True
    cost.update(generation_calls_without_response=cost['generation_attempts'] - cost['generation_returns'],
        unknown_additional_response_cost=(cost['generation_attempts'] != cost['generation_returns']
            or cost['generation_returns_with_unusable_usage'] > 0),
        unknown_additional_context_count_cost=cost['context_count_attempts'] != cost['context_count_returns'])
    if owned.get('api_attempts_observed') is not None:
        cost['api_attempts_observed'] = owned['api_attempts_observed']
    cost['unknown_additional_cost'] = (owned.get('unknown_additional_cost', True)
        or cost['unknown_additional_response_cost'] or cost['unknown_additional_context_count_cost'])
    known = (owned.get('safe_to_continue') is True and not journal_failed and state['episode'] is not None
        and not cost['unknown_additional_cost'] and cost['execution_attempts'] == cost['execution_returns']
        and cost['evaluation_attempts'] == cost['evaluation_returns'] == 1)
    result = {'status': 'complete' if known else 'unknown', 'record': {'task_id': task.task_id,
        'scenario_id': task.scenario_id, 'arm': arm,
        'outcome': state['episode'].success if known else None,
        'valid_for_continuation': known, 'released': released, 'protocol_sha256': fingerprint(protocol)},
        'cost': cost, 'journal_failed': journal_failed, 'private_files': state['private_files'],
        'owner': {key: value for key, value in owned.items() if key != 'operation_result'},
        'error_type': state['error_type'], 'automatic_retry': False,
        'accepted_experiences': 0, 'training_authorized': False,
        'cost_note': 'Response, step and trace records overlap; costs are not additive. '
                     'Observed API/token counts are lower bounds when additional cost is unknown.'}
    try:
        _write_private(output / 'report.json', result)
    except BaseException as exc:
        result.update(status='unknown', journal_failed=True)
        result['record'].update(outcome=None, valid_for_continuation=False)
        result['cost']['unknown_additional_cost'] = True
        if interrupted is None and isinstance(exc, _INTERRUPTS):
            interrupted = exc
    if interrupted is not None:
        if isinstance(interrupted, SystemExit):
            raise SystemExit(interrupted.code if type(interrupted.code) is int else 1) from None
        raise type(interrupted)() from None
    return result
