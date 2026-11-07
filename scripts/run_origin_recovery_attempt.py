"""One origin recovery attempt; parent owns admission and independent audit.

No Docker, GPU, service, or queue is started by this module. The injected
world factory must create a seeded owned world and durably report its container
intent before creation. Output is private and contains task/trajectory data.
"""

import copy
import math
import re
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from scripts import prepare_origin_recovery_pilot as preparation
from scripts.documentation_feedback import DocumentationFeedbackWorld
from scripts.run_timed_tasks import CheckedModel, CleanupTrackedWorld, RolloutDeadline
from scripts.run_timed_tasks_v2 import _validate_seed
from scripts.run_timed_tasks_v3 import (
    SEED_PROTOCOL,
    _append,
    _atomic,
    _intent,
    _sync_directory,
    client_identity,
    environment_identity,
    feedback_recipe,
    file_sha256,
)
from wll.prompts import initial_messages
from wll.protocol import Episode, Step, Task, fingerprint
from wll.runner import extract_code, run_episode
from wll.storage import require_space

IMAGE = 'sha256:2f5a87274ea53b165a7f1ec656d2f6534aa0fbdd6479f4a77efbd89e39ef7cb2'
GENERIC_HINT = (
    'Review the most recent execution error using only the task, actions and observations already visible to you. '
    'If needed, inspect the relevant read-only documentation or current state before choosing one minimal, '
    'verifiable correction. Do not repeat state-changing operations merely to make their output visible. '
    'Continue the original task within the remaining budget and follow the existing output-format rules.'
)
PROMPT_VERSION = 'appworld-v3'
SEED = 17
ENVIRONMENT_SEED = 100
MAX_STEPS = 40


def _sha(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def backend_semantic_signature(backend):
    """Drop only job-specific GPU identity and model-loading peak memory.

    GPU admission and limits are independently checked by the parent. Every
    other model, host, policy, tokenizer and generation field remains bound.
    """
    if type(backend) is not dict or type(backend.get('gpu_policy')) is not dict:
        raise ValueError('complete backend and GPU policy required')
    gpu = backend['gpu_policy']
    if (type(backend.get('loading_peak_reserved_bytes')) is not int
            or backend['loading_peak_reserved_bytes'] <= 0
            or type(gpu.get('physical_gpu_index')) is not int
            or type(gpu.get('gpu_uuid')) is not str or not gpu['gpu_uuid']):
        raise ValueError('backend GPU identity or loading peak missing')
    result = copy.deepcopy(backend)
    del result['loading_peak_reserved_bytes']
    del result['gpu_policy']['physical_gpu_index']
    del result['gpu_policy']['gpu_uuid']
    return result


def _source_prefix(episode):
    first = next((index for index, step in enumerate(episode.steps)
                  if step.observation.startswith('Execution failed')), None)
    if first is None:
        return 'no_error', []
    prefix = episode.steps[:first + 1]
    return ('no_budget' if len(prefix) == MAX_STEPS else 'candidate'), prefix


def _source_binding(episode, supplied_hash, task, model, backend, environment_id, metadata_sha):
    if (not isinstance(episode, Episode) or not _sha(supplied_hash)
            or fingerprint(episode.to_dict()) != supplied_hash
            or episode.task != task or episode.model_id != model.model_id
            or type(episode.seed) is not int or episode.seed != SEED
            or episode.branch != 'first' or episode.replayed_steps != 0
            or episode.experience_id is not None or type(episode.success) is not bool
            or episode.termination not in ('completed', 'step_budget')
            or len(episode.steps) > MAX_STEPS
            or (not episode.steps and episode.termination != 'completed')):
        raise ValueError('source episode identity or hash mismatch')
    if any(not isinstance(step, Step) or type(step.action) is not str
           or type(step.observation) is not str or type(step.raw_response) is not str
           or extract_code(step.raw_response) != step.action
           or type(step.elapsed_seconds) not in (float, int)
           or not math.isfinite(step.elapsed_seconds) or step.elapsed_seconds < 0
           or type(step.prompt_tokens) is not int or step.prompt_tokens < 0
           or type(step.completion_tokens) is not int or step.completion_tokens < 0
           for step in episode.steps):
        raise ValueError('invalid source steps')
    meta = episode.metadata
    current_semantic_sha = fingerprint(backend_semantic_signature(backend))
    if (not isinstance(meta, dict) or meta.get('max_steps') != MAX_STEPS
            or meta.get('max_tokens') != 1024 or meta.get('temperature') != .7
            or meta.get('prompt_version') != PROMPT_VERSION
            or not _sha(meta.get('expected_backend_sha256'))
            or meta.get('backend_semantic_sha256') != current_semantic_sha
            or (episode.steps and (type(meta.get('backend')) is not dict
                or fingerprint(meta['backend']) != meta['expected_backend_sha256']
                or fingerprint(backend_semantic_signature(meta['backend'])) != current_semantic_sha))
            or (not episode.steps and meta.get('backend') is not None)
            or meta.get('image') != IMAGE
            or meta.get('environment_id') != environment_id
            or meta.get('prepared_manifest_sha256') != metadata_sha
            or meta.get('observation_feedback') != feedback_recipe()):
        raise ValueError('source model or environment binding mismatch')
    _validate_seed(meta.get('environment_seed'))
    return _source_prefix(episode)


def _proof_binding(proof, supplied_hash, *, task, model, backend, environment_id,
                   metadata_sha, source_sha, prefix):
    expected = {
        'kind': 'prefix_check', 'status': 'eligible', 'source_sha256': source_sha,
        'prefix_sha256': fingerprint([asdict(step) for step in prefix]),
        'prefix_length': len(prefix), 'task_sha256': fingerprint(asdict(task)),
        'model_id': model.model_id,
        'backend_semantic_sha256': fingerprint(backend_semantic_signature(backend)),
        'environment_id': environment_id, 'image': IMAGE, 'seed': SEED,
        'prepared_manifest_sha256': metadata_sha, 'prefix_success': False,
        'prefix_completed': False, 'local_cleanup_complete': True,
    }
    if (not isinstance(proof, dict) or not _sha(supplied_hash)
            or fingerprint(proof) != supplied_hash
            or not _sha(proof.get('backend_sha256'))
            or type(proof.get('prefix_length')) is not int
            or proof.get('prefix_success') is not False
            or proof.get('prefix_completed') is not False
            or proof.get('local_cleanup_complete') is not True
            or any(proof.get(key) != value for key, value in expected.items())):
        raise ValueError('independent prefix proof identity or hash mismatch')
    _validate_seed(proof.get('environment_seed'))


class _TrackedWorld(CleanupTrackedWorld):
    """Fsync replay and new-action intent/return before and after execution."""

    def __init__(self, world, deadline, emit, replay, new_actions, kind):
        super().__init__(world, deadline)
        self.emit, self.replay, self.new_actions, self.kind = emit, replay, new_actions, kind
        self.close_attempted = False
        self.checked_branch_start = False
        self.cleanup_started = None
        self.cleanup_seconds = None
        self.cleanup_error_type = None

    def close(self):
        self.close_attempted = True
        self.cleanup_started = time.monotonic()
        try:
            return super().close()
        except BaseException as exc:
            self.cleanup_error_type = type(exc).__name__
            raise
        finally:
            self.cleanup_seconds = time.monotonic() - self.cleanup_started

    def execute(self, action):
        replayed = self.replay['attempted'] < self.replay['expected']
        counters = self.replay if replayed else self.new_actions
        event = 'replay' if replayed else 'execute'
        index = counters['attempted'] + (0 if replayed else self.replay['expected'])
        self.emit(event + '_attempt', index=index, action=action, action_sha256=fingerprint(action))
        counters['attempted'] += 1
        started = time.monotonic()
        try:
            observation = self.world.execute(action)
            if type(observation) is not str:
                raise ValueError('environment observation must be text')
        except BaseException as exc:
            elapsed = time.monotonic() - started
            counters['elapsed_seconds'] += elapsed
            self.emit(event + '_failed', index=index, error_type=type(exc).__name__,
                      elapsed_seconds=elapsed)
            raise
        elapsed = time.monotonic() - started
        counters['elapsed_seconds'] += elapsed
        self.emit(event + '_returned', index=index, elapsed_seconds=elapsed, observation=observation,
                  observation_sha256=fingerprint(observation))
        counters['returned'] += 1
        return observation

    def completed(self):
        result = self.world.completed()
        if type(result) is not bool:
            raise ValueError('completed state must be boolean')
        if (self.kind in ('control', 'hint') and not self.checked_branch_start
                and self.replay['returned'] == self.replay['expected']):
            self.checked_branch_start = True
            if result:
                raise ValueError('replayed branch prefix already completed')
        return result


class _RequestModel:
    def __init__(self, model, emit, task, prefix, kind):
        self.model, self.emit, self.kind = model, emit, kind
        self.counter = 0
        expected = initial_messages(task, PROMPT_VERSION)
        for step in prefix:
            expected.extend([{'role': 'assistant', 'content': step.raw_response or step.action},
                             {'role': 'user', 'content': step.observation}])
        self.expected_prefix = expected

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, messages, seed):
        student = copy.deepcopy(messages)
        index = len(self.expected_prefix)
        if student[:index] != self.expected_prefix:
            raise ValueError('actor-visible replay prefix differs')
        teacher = copy.deepcopy(student)
        hint_index = None
        if self.kind == 'hint':
            teacher.insert(index, {'role': 'user', 'content': GENERIC_HINT})
            hint_index = index
        self.counter += 1
        call = {'call_id': self.counter, 'seed': seed, 'hint_index': hint_index,
                'teacher_messages_sha256': fingerprint(teacher),
                'student_messages_sha256': fingerprint(student)}
        self.emit('generation_attempt', **call)  # fsync must precede model call.
        try:
            response = self.model.generate(teacher, seed)
        except BaseException as exc:
            self.emit('generation_failed', **call, error_type=type(exc).__name__)
            raise
        valid = isinstance(response, dict) and all(
            type(response.get(key)) is int and response[key] >= 0
            for key in ('prompt_tokens', 'completion_tokens'))
        raw = response.get('text') if isinstance(response, dict) else None
        raw = raw if type(raw) is str else None
        self.emit('generation_returned', **call, usage_valid=valid,
                  raw_response=raw, raw_response_sha256=fingerprint(raw) if raw is not None else None,
                  prompt_tokens=response['prompt_tokens'] if valid else None,
                  completion_tokens=response['completion_tokens'] if valid else None)
        # CheckedModel expects a mapping. Preserve unusable usage accounting even
        # when a malformed transport returns a non-mapping value.
        return response if isinstance(response, dict) else {}


class _NoGeneration:
    max_tokens = 1024
    temperature = .7

    def __init__(self, model_id):
        self.model_id = model_id

    def generate(self, messages, seed):
        raise AssertionError('prefix check cannot generate')


def run_attempt(task, world_factory, model, *, kind, output, expected_backend,
                image, environment_id, prepared_manifest_sha256, source_episode=None,
                source_sha256=None, prefix_proof=None, prefix_proof_sha256=None):
    """Run one attempt. The parent verifies prepared metadata, source and release.

    Source/proof hashes use ``wll.protocol.fingerprint`` of their JSON objects;
    the caller must bind those hashes to independently audited durable files.
    Prefix checks may pass model=None; a static identity is derived from the
    source without querying a service. No prefix-check model is ever called.
    Request journals contain hashes, not message bodies. An independent
    auditor must reconstruct every role/content sequence from the private Task,
    prefix and steps and verify both teacher and student hashes.
    """
    if kind not in ('source', 'prefix_check', 'control', 'hint'):
        raise ValueError('invalid attempt kind')
    selected = preparation.prepare()['selected_ids']
    if (not isinstance(task, Task) or task.task_id not in selected or task.split != 'train'
            or task.scenario_id != task.task_id.rsplit('_', 1)[0]
            or type(task.instruction) is not str or not task.instruction
            or type(task.supervisor) is not str):
        raise ValueError('fixed origin train task required')
    if model is None and kind == 'prefix_check':
        if not isinstance(source_episode, Episode):
            raise ValueError('offline prefix check requires a source episode')
        model = _NoGeneration(source_episode.model_id)
    if (image != IMAGE or environment_id != environment_identity(image)
            or not _sha(prepared_manifest_sha256) or not isinstance(expected_backend, dict)
            or expected_backend.get('arm') != 'B3' or type(getattr(model, 'model_id', None)) is not str
            or not model.model_id or type(getattr(model, 'max_tokens', None)) is not int
            or model.max_tokens != 1024 or type(getattr(model, 'temperature', None)) is not float
            or model.temperature != .7):
        raise ValueError('fixed model, seed, image, backend and metadata binding required')
    semantic_backend_sha256 = fingerprint(backend_semantic_signature(expected_backend))

    classification, prefix = None, []
    if kind == 'source':
        if any(value is not None for value in (source_episode, source_sha256,
                                                prefix_proof, prefix_proof_sha256)):
            raise ValueError('source attempt cannot take prior outcome inputs')
    else:
        classification, prefix = _source_binding(source_episode, source_sha256, task, model,
                                                   expected_backend, environment_id,
                                                   prepared_manifest_sha256)
        if kind == 'prefix_check':
            if prefix_proof is not None or prefix_proof_sha256 is not None:
                raise ValueError('prefix check cannot take a prior proof')
        else:
            if classification != 'candidate':
                raise ValueError('source has no replayable budgeted prefix')
            _proof_binding(prefix_proof, prefix_proof_sha256, task=task, model=model,
                backend=expected_backend, environment_id=environment_id,
                metadata_sha=prepared_manifest_sha256, source_sha=source_sha256, prefix=prefix)

    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    _sync_directory(output.parent)
    identity = client_identity()
    observation_feedback = feedback_recipe()
    manifest = {'status': 'running', 'kind': kind, 'task_id': task.task_id,
        'task_sha256': fingerprint(asdict(task)), 'model_id': model.model_id,
        'attempt_script_sha256': file_sha256(__file__),
        'preparation_script_sha256': file_sha256(preparation.__file__),
        'backend_sha256': fingerprint(expected_backend), 'backend_semantic_sha256': semantic_backend_sha256,
        'client_identity': identity,
        'image': image, 'environment_id': environment_id, 'seed': SEED,
        'environment_seed_requested': ENVIRONMENT_SEED, 'seed_protocol': SEED_PROTOCOL,
        'observation_feedback': observation_feedback, 'max_steps': MAX_STEPS,
        'max_tokens': 1024, 'temperature': .7, 'prompt_version': PROMPT_VERSION,
        'deadline_seconds': 120 if kind == 'prefix_check' else 600,
        'deadline_scope': 'after world readiness through terminal evaluation; startup/cleanup separate',
        'drain_scope': 'caller-owned; not performed or measured by this component',
        'metadata_binding_scope': 'caller-supplied hash; parent validates prepared metadata',
        'request_evidence': 'hashes only; auditor reconstructs role/content from private Task and steps',
        'prepared_manifest_sha256': prepared_manifest_sha256, 'source_sha256': source_sha256,
        'prefix_proof_sha256': prefix_proof_sha256,
        'hint_sha256': fingerprint(GENERIC_HINT) if kind == 'hint' else None,
        'automatic_retry': False,
        'parent_must_verify_source_admission_and_release': True}
    _atomic(output / 'manifest.json', manifest)
    replay = {'expected': len(prefix), 'attempted': 0, 'returned': 0, 'elapsed_seconds': 0.0}
    new_actions = {'attempted': 0, 'returned': 0, 'elapsed_seconds': 0.0}
    counters = {'generation_calls': 0, 'generation_calls_without_response': 0,
        'generation_calls_with_unusable_usage': 0,
        'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0}}
    record = {'kind': kind, 'status': 'unknown', 'success': None,
        'task_id': task.task_id, 'task_sha256': fingerprint(asdict(task)), 'model_id': model.model_id,
        'backend_sha256': fingerprint(expected_backend), 'backend_semantic_sha256': semantic_backend_sha256,
        'image': image,
        'environment_id': environment_id, 'seed': SEED,
        'environment_seed': {'requested': 100, 'effective': None, 'verified': False},
        'prepared_manifest_sha256': prepared_manifest_sha256, 'source_sha256': source_sha256,
        'prefix_sha256': fingerprint([asdict(step) for step in prefix]) if prefix else None,
        'prefix_length': len(prefix), 'local_cleanup_complete': False,
        'total_resources_released': None, 'replay': replay, 'new_actions': new_actions, 'generation': counters,
        'cost_note': 'Replay and new-action times are separately measured execute wall time; replayed step '
            'tokens and elapsed values are historical. Returned generation usage overlaps the step view.'}
    if kind == 'source':
        record['source_classification'] = 'unknown'
    deadline = RolloutDeadline(120 if kind == 'prefix_check' else 600)
    world = None
    seen_feedback = 0
    rollout_started = None
    start = time.monotonic()
    with ExitStack() as stack:
        streams = {name: stack.enter_context((output / (name + '.jsonl')).open('x'))
                   for name in ('events', 'attempts', 'episodes', 'steps', 'responses', 'feedback-events')}
        _sync_directory(output)
        event_index = 0

        def emit(event, **details):
            nonlocal event_index
            _append(streams['events'], {'event_index': event_index, 'event': event,
                                        'epoch': time.time(), **details})
            event_index += 1

        emit('attempt_started', kind=kind, task_id=task.task_id, client_identity=identity)
        _append(streams['attempts'], {'kind': kind, 'task_id': task.task_id, 'status': 'started'})
        intent_seen = False

        def container_intent(value):
            nonlocal intent_seen
            if intent_seen:
                raise ValueError('duplicate container intent')
            value = _intent(value, identity, image)
            emit('container_intent', task_id=task.task_id, container_intent=value)
            intent_seen = True

        def on_response(value):
            _append(streams['responses'], {'task_id': task.task_id, **value})

        def observe(index, step, replayed):
            nonlocal seen_feedback
            current = world.feedback_events
            if current != seen_feedback:
                _append(streams['feedback-events'], {'task_id': task.task_id, 'index': index,
                    'replayed': replayed, 'feedback_events': current,
                    'action_sha256': fingerprint(step.action),
                    'augmented_observation_sha256': fingerprint(step.observation)})
                seen_feedback = current
            raw_match = step.raw_response == step.action
            _append(streams['steps'], {'task_id': task.task_id, 'index': index,
                'replayed': replayed, 'raw_action_match': raw_match,
                'raw_action_target_filter_passed': not replayed and raw_match, 'step': asdict(step)})

        try:
            if kind == 'prefix_check' and classification != 'candidate':
                record['status'] = classification
            else:
                startup = time.monotonic()
                try:
                    wrapped = world_factory(task, container_intent)
                    world = _TrackedWorld(wrapped, deadline, emit, replay, new_actions, kind)
                    if not intent_seen or not isinstance(wrapped, DocumentationFeedbackWorld):
                        raise ValueError('owned seeded feedback world required')
                    record['environment_seed'] = _validate_seed(world.environment_seed_provenance)
                    emit('world_ready', task_id=task.task_id,
                         environment_seed=record['environment_seed'])
                finally:
                    record['startup_seconds'] = time.monotonic() - startup
                request_model = _RequestModel(model, emit, task, prefix, kind)
                checked = CheckedModel(request_model, expected_backend, counters, on_response)
                effective_model = _NoGeneration(model.model_id) if kind == 'prefix_check' else checked
                rollout_started = time.monotonic()
                with deadline:
                    episode = run_episode(task, lambda: world, effective_model,
                        max_steps=len(prefix) if kind == 'prefix_check' else MAX_STEPS,
                        seed=SEED, prefix=prefix,
                        branch='first' if kind == 'source' else 'recovery',
                        on_step=observe, prompt_version=PROMPT_VERSION)
                episode.metadata.update(image=image, environment_id=environment_id,
                    environment_seed=record['environment_seed'], observation_feedback=observation_feedback,
                    expected_backend_sha256=fingerprint(expected_backend),
                    backend_semantic_sha256=semantic_backend_sha256,
                    prepared_manifest_sha256=prepared_manifest_sha256)
                if type(episode.success) is not bool:
                    raise ValueError('terminal evaluation must return a boolean')
                _append(streams['episodes'], episode.to_dict())
                record['episode_sha256'] = fingerprint(episode.to_dict())
                if (counters['generation_calls_without_response']
                        or counters['generation_calls_with_unusable_usage']
                        or episode.termination not in ('completed', 'step_budget')):
                    raise ValueError('incomplete generation cannot establish a known outcome')
                if kind == 'prefix_check':
                    record['prefix_success'] = episode.success
                    record['prefix_completed'] = episode.termination == 'completed'
                    record['status'] = ('eligible' if not episode.success and not record['prefix_completed']
                                        else 'already_terminal')
                else:
                    record.update(status='complete', success=episode.success)
                    if kind == 'source':
                        record['source_classification'], found = _source_prefix(episode)
                        record['prefix_length'] = len(found)
                        record['prefix_sha256'] = fingerprint([asdict(step) for step in found]) if found else None
        except Exception as exc:
            record.update(status='unknown', success=None, error_type=type(exc).__name__)
        finally:
            if world is not None:
                try:
                    if not world.close_attempted:
                        world.close()
                except Exception as exc:
                    record.update(status='unknown', success=None, cleanup_error_type=type(exc).__name__)
                record['cleanup_seconds'] = world.cleanup_seconds
                if world.cleanup_error_type:
                    record['cleanup_error_type'] = world.cleanup_error_type
                if rollout_started is not None and world.cleanup_started is not None:
                    record['rollout_seconds'] = world.cleanup_started - rollout_started
                record['local_cleanup_complete'] = world.closed
                if not world.closed:
                    record.update(status='unknown', success=None)
            if kind == 'source' and record['status'] == 'unknown':
                record['source_classification'] = 'unknown'
            record['wall_seconds'] = time.monotonic() - start
            record['generation']['unknown_additional_cost'] = bool(
                counters['generation_calls_without_response'] or counters['generation_calls_with_unusable_usage'])
            record['replay']['unknown_additional_cost'] = replay['attempted'] != replay['returned']
            record['new_actions']['unknown_additional_cost'] = new_actions['attempted'] != new_actions['returned']
            _append(streams['attempts'], record)
            manifest['status'] = record['status']
            _atomic(output / 'manifest.json', manifest)
    return record
