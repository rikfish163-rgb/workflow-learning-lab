"""Native CPU client for one fixed origin recovery schedule decision.

The parent owns registration, admission, hard watchdogs and physical release.
Private attempt files stay under the exclusive output directory. No retry or
state transition beyond this one already-decided batch is performed here.
"""
import hashlib
import json
import os
import stat
import time
from dataclasses import asdict
from pathlib import Path

from scripts import audit_origin_recovery_attempt as auditor
from scripts import origin_recovery_cohort as cohort
from scripts import origin_recovery_schedule as schedule
from scripts import prepare_origin_recovery_metadata as metadata
from scripts import run_origin_recovery_attempt as runtime
from scripts.documentation_feedback import DocumentationFeedbackWorld
from scripts.run_timed_tasks import RolloutDeadline, wait_for_idle
from scripts.run_timed_tasks_v2 import _validate_seed
from scripts.run_timed_tasks_v3 import _atomic, _sync_directory, client_identity
from wll.model import LocalModel
from wll.protocol import Episode, fingerprint
from wll.storage import require_space

ROOT = Path(__file__).resolve().parents[1]
PREPARED_SHA256 = '7ebc9cebe3422b1750bed40854c246b177095b68d2a8895a522a9c64a6d01a3f'
SERVICE_PLAN = ROOT / 'artifacts/qwen38-b3-service-plan-001.json'
SERVICE_PLAN_SHA256 = 'd7f56521d101394c8559ae0604680dc75d1bd4e346a96032c1c1c94ec676bad5'
MODEL_ID = 'qwen38-b3-matched-b3-001'
ADAPTER_SHA256 = '2058219dff9afb29b755f8d10ae3b505fbea1d3f71f1adf6b62b28725917848a'
IDLE_SECONDS = 180


def append_event(stream, event):
    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def _bound_file(path, expected):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('expected a regular evidence file')
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('bound evidence file changed')
    return raw


def _validate_backend(backend, state):
    plan = json.loads(_bound_file(SERVICE_PLAN, SERVICE_PLAN_SHA256))
    if (type(backend) is not dict or backend.get('arm') != 'B3'
            or backend.get('service_plan_sha256') != SERVICE_PLAN_SHA256
            or backend.get('server_sha256') != plan['source_sha256']['scripts/serve_qwen38_b3.py']
            or plan.get('model_ids', {}).get('B3') != MODEL_ID
            or fingerprint(backend.get('recipe')) != fingerprint(plan['recipe'])
            or fingerprint(backend.get('saved_adapter')) != fingerprint(plan['saved_adapter'])
            or fingerprint(backend.get('training_result')) != fingerprint(plan['training_result'])
            or fingerprint(backend.get('source_sha256')) != fingerprint(plan['source_sha256'])
            or plan['saved_adapter']['files']['adapter_model.safetensors']['sha256'] != ADAPTER_SHA256
            or fingerprint(runtime.backend_semantic_signature(backend)) != state['backend_semantic_sha256']):
        raise ValueError('frozen B3 service, recipe, adapter or semantic backend differs')
    adapter, saved = backend.get('adapter'), plan['saved_adapter']
    if (type(adapter) is not dict or adapter.get('status') != 'verified'
            or type(adapter.get('tensor_count')) is not int or adapter['tensor_count'] != 608
            or adapter.get('source_adapter_sha256') != saved['source_adapter_sha256']
            or adapter.get('adapter_state_sha256') != saved['saved_peft_state_sha256']
            or adapter.get('base_parameter_storage_unchanged') is not True
            or adapter.get('all_parameters_frozen') is not True
            or adapter.get('active_adapters') != ['default'] or adapter.get('merged') is not False
            or adapter.get('use_cache') is not True or adapter.get('adapter_dtypes') != ['torch.float32']
            or fingerprint(adapter.get('files')) != fingerprint(saved['files'])):
        raise ValueError('B3 adapter reload evidence differs')
    quant = backend.get('quantization')
    if (type(quant) is not dict or type(quant.get('linear4bit_layers')) is not int
            or quant['linear4bit_layers'] != 496 or quant.get('parameter_devices') != ['cuda:0']
            or type(quant.get('configuration')) is not dict
            or quant['configuration'].get('load_in_4bit') is not True
            or quant['configuration'].get('bnb_4bit_quant_type') != 'nf4'):
        raise ValueError('B3 NF4 backend differs')


def _load_prepared(prepared_dir, state):
    tasks, value = metadata.load_prepared(Path(prepared_dir) / 'tasks.json', PREPARED_SHA256)
    tasks = cohort.select_prepared(tasks)
    ids = schedule.frozen_ids()
    if (state['prepared_manifest_sha256'] != PREPARED_SHA256 or state['model_id'] != MODEL_ID
            or [task.task_id for task in tasks] != ids or value.get('image') != runtime.IMAGE
            or value.get('environment_id') != schedule.ENVIRONMENT_ID
            or {task.task_id: fingerprint(asdict(task)) for task in tasks} != state['task_sha256']):
        raise ValueError('prepared metadata, tasks or study identity differs')
    return {task.task_id: task for task in tasks}, value


def _audit_equal(old, fresh):
    if (old.get('audit_valid') is not True or fresh.get('audit_valid') is not True
            or old.get('audit_sha256') != fresh.get('audit_sha256')
            or fingerprint(old) != fingerprint(fresh)):
        raise ValueError('prior independent audit differs from durable files')


def _prior(state, prior_attempts, task, kind):
    if kind == 'source':
        return {}, None
    entry = prior_attempts.get(task.task_id)
    if type(entry) is not dict or type(entry.get('source')) is not dict:
        raise ValueError('prior source directory and backend required')
    source = entry['source']
    if set(source) != {'directory', 'backend'}:
        raise ValueError('invalid prior source locator')
    source_dir, source_backend = Path(source['directory']), source['backend']
    _validate_backend(source_backend, state)
    old_source = next((row for b in state['source_batches'] for row in b['attempts']
                       if row['task_id'] == task.task_id), None)
    if type(old_source) is not dict:
        raise ValueError('source audit missing from schedule')
    common = {'task': task, 'model_id': MODEL_ID, 'expected_backend': source_backend,
              'prepared_manifest_sha256': PREPARED_SHA256}
    fresh_source = auditor.audit_attempt(source_dir, kind='source', **common)
    _audit_equal(old_source, fresh_source)
    source_file_sha = fresh_source['input_file_sha256']['episodes.jsonl']
    source_rows = _bound_file(source_dir / 'episodes.jsonl', source_file_sha).splitlines()
    if len(source_rows) != 1:
        raise ValueError('source episode file differs')
    episode = Episode.from_dict(json.loads(source_rows[0]))
    source_sha = fingerprint(episode.to_dict())
    if source_sha != fresh_source['episode_sha256']:
        raise ValueError('source episode fingerprint differs')
    kwargs = {'source_episode': episode, 'source_sha256': source_sha}
    provenance = {'source_episode_file_sha256': source_file_sha,
                  'source_audit_sha256': fresh_source['audit_sha256'],
                  'source_backend_sha256': fingerprint(source_backend),
                  'prefix_attempt_file_sha256': None, 'prefix_audit_sha256': None}
    if kind != 'prefix_check':
        prefix = entry.get('prefix')
        if type(prefix) is not dict or set(prefix) != {'directory'}:
            raise ValueError('prior prefix directory required')
        prefix_dir = Path(prefix['directory'])
        old_prefix = next((row for row in state['prefix_checks'] if row['task_id'] == task.task_id), None)
        if type(old_prefix) is not dict:
            raise ValueError('prefix audit missing from schedule')
        fresh_prefix = auditor.audit_attempt(prefix_dir, kind='prefix_check', **common, **kwargs)
        _audit_equal(old_prefix, fresh_prefix)
        if fresh_prefix.get('status') != 'eligible':
            raise ValueError('prefix is not eligible')
        proof_file_sha = fresh_prefix['input_file_sha256']['attempts.jsonl']
        proof_rows = _bound_file(prefix_dir / 'attempts.jsonl', proof_file_sha).splitlines()
        if len(proof_rows) != 2:
            raise ValueError('prefix proof file differs')
        proof = json.loads(proof_rows[-1])
        proof_sha = fingerprint(proof)
        if proof_sha != fresh_prefix['proof_sha256']:
            raise ValueError('prefix proof fingerprint differs')
        kwargs.update(prefix_proof=proof, prefix_proof_sha256=proof_sha)
        provenance.update(prefix_attempt_file_sha256=proof_file_sha,
                          prefix_audit_sha256=fresh_prefix['audit_sha256'])
    return {'kwargs': kwargs, 'provenance': provenance}, source_backend


def _idle(callback, endpoint, backend):
    if callback is None:
        wait_for_idle(endpoint, MODEL_ID, backend, seconds=IDLE_SECONDS)
    else:
        with RolloutDeadline(IDLE_SECONDS):
            callback()


def _default_world(value):
    from scripts.owned_appworld_v3 import OwnedAppWorldEnvironmentV3

    def factory(task, on_container_intent):
        world = OwnedAppWorldEnvironmentV3(task.task_id, value['data_dir'], runtime.MAX_STEPS,
            runtime.IMAGE, value['deps_dir'], environment_seed=100, on_container_intent=on_container_intent)
        return DocumentationFeedbackWorld(world)
    return factory


def run_batch(*, state, decision, prepared_dir, output, expected_backend, model, world_factory=None,
              prior_attempts=None, endpoint=None, before_idle=None, after_idle=None):
    """Execute exactly one scheduler decision; never attest external release.

    Injectable worlds, models and waits are intended for synthetic CPU tests.
    Actor requests use the supplied model; production callers must supply the
    registered loopback LocalModel and independently authorize its service.
    """
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU client requires CUDA_VISIBLE_DEVICES explicitly empty')
    tasks, value = _load_prepared(prepared_dir, state)
    if fingerprint(schedule.next_action(state)) != fingerprint(decision):
        raise ValueError('decision differs from current fixed schedule')
    action = decision.get('action')
    if action not in ('launch_source_batch', 'launch_prefix_check', 'launch_branch_batch'):
        raise ValueError('schedule is not ready to launch this batch')
    items = decision['attempts']
    if not 1 <= len(items) <= (1 if action == 'launch_prefix_check' else 6):
        raise ValueError('invalid batch size')
    prefix_only = action == 'launch_prefix_check'
    if prefix_only:
        if model is not None or expected_backend is not None or before_idle is not None or after_idle is not None:
            raise ValueError('prefix checks must be offline and use source backend only')
    else:
        _validate_backend(expected_backend, state)
        if (getattr(model, 'model_id', None) != MODEL_ID or type(getattr(model, 'max_tokens', None)) is not int
                or model.max_tokens != 1024 or type(getattr(model, 'temperature', None)) is not float
                or model.temperature != .7):
            raise ValueError('fixed B3 model and decoding required')
        if before_idle is None or after_idle is None:
            if (not isinstance(model, LocalModel) or type(endpoint) is not str
                    or endpoint.rstrip('/') != model.endpoint):
                raise ValueError('production wait requires the same loopback model endpoint')
    prior_attempts = {} if prior_attempts is None else prior_attempts
    if type(prior_attempts) is not dict:
        raise ValueError('invalid prior attempt locators')
    # Validate the complete decision before any output, model wait, or world creation.
    for item in items:
        _prior(state, prior_attempts, tasks[item['task_id']], item['kind'])
    world_factory = world_factory or _default_world(value)
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    _sync_directory(output.parent)
    (output / 'attempts').mkdir(mode=0o700)
    (output / 'audits').mkdir(mode=0o700)
    _sync_directory(output)
    identity = client_identity()
    manifest = {'status': 'running', 'decision': decision, 'client_identity': identity,
        'prepared_manifest_sha256': PREPARED_SHA256, 'model_id': MODEL_ID,
        'state_sha256': fingerprint(state), 'automatic_retry': False,
        'external_release_verified': False}
    _atomic(output / 'manifest.json', manifest)
    event_fd = os.open(output / 'events.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    records, idle_waits, stop_reason = [], [], None
    with os.fdopen(event_fd, 'w', encoding='utf-8') as events:
        _sync_directory(output)
        index = 0

        def emit(event, **details):
            nonlocal index
            append_event(events, {'event_index': index, 'event': event, 'epoch': time.time(), **details})
            index += 1

        emit('batch_started', client_identity=identity, decision=decision)
        for position, item in enumerate(items):
            task, kind = tasks[item['task_id']], item['kind']
            try:
                previous, source_backend = _prior(state, prior_attempts, task, kind)
            except Exception as exc:
                emit('prior_failed', task_id=task.task_id, kind=kind, error_type=type(exc).__name__)
                stop_reason = 'prior_audit_failed'
                break
            backend = source_backend if prefix_only else expected_backend
            before_seconds = after_seconds = None
            if not prefix_only:
                emit('idle_started', task_id=task.task_id, kind=kind, phase='before', limit_seconds=IDLE_SECONDS)
                started = time.monotonic()
                try:
                    _idle(before_idle, endpoint, backend)
                except Exception as exc:
                    before_seconds = time.monotonic() - started
                    idle_waits.append({'task_id': task.task_id, 'kind': kind, 'phase': 'before',
                        'status': 'failed', 'elapsed_seconds': before_seconds, 'error_type': type(exc).__name__})
                    emit('idle_failed', task_id=task.task_id, kind=kind, phase='before',
                         elapsed_seconds=before_seconds, error_type=type(exc).__name__)
                    stop_reason = 'before_idle_failed'
                    break
                before_seconds = time.monotonic() - started
                idle_waits.append({'task_id': task.task_id, 'kind': kind, 'phase': 'before',
                                   'status': 'returned', 'elapsed_seconds': before_seconds})
                emit('idle_returned', task_id=task.task_id, kind=kind, phase='before', elapsed_seconds=before_seconds)
            relative = f'attempts/{position:02d}-{kind}-{task.task_id}'
            attempt_dir = output / relative
            emit('attempt_started', task_id=task.task_id, kind=kind, directory=relative)

            def owned_world(actual_task, callback, *, attempt_kind=kind):
                intent_seen = False

                def intent(value):
                    nonlocal intent_seen
                    callback(value)  # Child journal first; root index before Docker Popen.
                    emit('container_intent', task_id=actual_task.task_id, kind=attempt_kind, container_intent=value)
                    intent_seen = True
                world = world_factory(actual_task, intent)
                try:
                    if not intent_seen or not isinstance(world, DocumentationFeedbackWorld):
                        raise ValueError('owned feedback world required')
                    seed = _validate_seed(world.environment_seed_provenance)
                    emit('world_ready', task_id=actual_task.task_id, kind=attempt_kind, environment_seed=seed)
                    return world
                except BaseException:
                    world.close()
                    raise

            kwargs = previous.get('kwargs', {})
            runtime_error = None
            try:
                runtime.run_attempt(task, owned_world, model, kind=kind, output=attempt_dir,
                    expected_backend=backend, image=runtime.IMAGE, environment_id=schedule.ENVIRONMENT_ID,
                    prepared_manifest_sha256=PREPARED_SHA256, **kwargs)
            except Exception as exc:
                runtime_error = type(exc).__name__
            audit = auditor.audit_attempt(attempt_dir, task=task, kind=kind, model_id=MODEL_ID,
                expected_backend=backend, prepared_manifest_sha256=PREPARED_SHA256, **kwargs)
            audit_relative = f'audits/{position:02d}-{kind}-{task.task_id}.json'
            _atomic(output / audit_relative, audit)
            audit_file_sha = hashlib.sha256((output / audit_relative).read_bytes()).hexdigest()
            emit('attempt_audited', task_id=task.task_id, kind=kind, audit_sha256=audit['audit_sha256'],
                 audit_valid=audit['audit_valid'], runtime_error_type=runtime_error)
            record = {'task_id': task.task_id, 'kind': kind, 'directory': relative,
                'audit_path': audit_relative, 'audit_file_sha256': audit_file_sha, 'audit': audit,
                'prior_provenance': previous.get('provenance'), 'before_idle_seconds': before_seconds,
                'after_idle_seconds': None, 'status': 'complete' if audit['audit_valid'] is True else 'unknown'}
            records.append(record)
            if not prefix_only:
                emit('idle_started', task_id=task.task_id, kind=kind, phase='after', limit_seconds=IDLE_SECONDS)
                started = time.monotonic()
                try:
                    _idle(after_idle, endpoint, backend)
                except Exception as exc:
                    after_seconds = time.monotonic() - started
                    idle_waits.append({'task_id': task.task_id, 'kind': kind, 'phase': 'after',
                        'status': 'failed', 'elapsed_seconds': after_seconds, 'error_type': type(exc).__name__})
                    emit('idle_failed', task_id=task.task_id, kind=kind, phase='after',
                         elapsed_seconds=after_seconds, error_type=type(exc).__name__)
                    record.update(status='unknown', after_idle_seconds=after_seconds)
                    stop_reason = 'after_idle_failed'
                    break
                after_seconds = time.monotonic() - started
                idle_waits.append({'task_id': task.task_id, 'kind': kind, 'phase': 'after',
                                   'status': 'returned', 'elapsed_seconds': after_seconds})
                record['after_idle_seconds'] = after_seconds
                emit('idle_returned', task_id=task.task_id, kind=kind, phase='after', elapsed_seconds=after_seconds)
            if runtime_error is not None or audit['audit_valid'] is not True:
                record['status'] = 'unknown'
                stop_reason = 'attempt_or_audit_unknown'
                break
        result = {'status': 'complete' if stop_reason is None else 'unknown', 'stop_reason': stop_reason,
            'decision': decision, 'attempted': len(records), 'attempts': records,
            'unattempted': items[len(records):], 'idle_waits': idle_waits, 'external_release_verified': False,
            'cost_scope': 'Idle waits are separate from attempt, model loading, admission and GPU job costs.'}
        emit('batch_finished', status=result['status'], stop_reason=stop_reason, attempted=len(records))
        _atomic(output / 'result.json', result)
        manifest.update(status=result['status'], stop_reason=stop_reason)
        _atomic(output / 'manifest.json', manifest)
    return result
