"""Private, one-shot train metadata and CPU context preparation.

The only permitted worker RPCs are list(train) and describe for the frozen
eleven IDs. No world, model, actor execution or evaluation is authorized here.
Private task values never enter public reports, event records or errors.
"""

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from scripts import owned_appworld_v3 as owned
from scripts.verification_readonly_registry_v3 import verify_installed_sources
from scripts.verification_token_counter import MODEL_PATH, TOKENIZER_SHA256, count_input_tokens
from wll.prompts import initial_messages
from wll.protocol import Task, fingerprint

ROOT = Path(__file__).resolve().parents[1]
COHORT = 'artifacts/fresh-completion-cohort-CPU-001.json'
COHORT_SHA256 = '2c2364eacba1d02d5f292639ec560d703e6cb09026a2719f4e3f1855f51f0570'
BUILD = 'artifacts/verification-v3-image-build-001.json'
BUILD_SHA256 = '01f6a0ec016ddf6bf3334df009d8cfc2298fad3ef202778c68cce4ee856db5e0'
IMAGE = 'sha256:cb48da28204f6ae5aa0686acb6b10556bc76ed05b96e24cef7f40e274cf4265d'
DATA_DIR = '/media/hetaisheng/044A81D94A81C83E/wll-runtime/data'
DEPS_DIR = '/media/hetaisheng/044A81D94A81C83E/wll-runtime/deps'
OFFICIAL_TRAIN_SHA256 = '93d9fe71e7a2e3b7529803d4a20b604f4ebf5ae806f321081140238068189d37'
PRIVATE = Path('/home/hetaisheng/.local/share/workflow-learning-lab/fresh-completion-metadata-001')
METADATA_PROOF = ROOT / 'artifacts/fresh-completion-metadata-CPU-001.json'
CONTEXT_PROOF = ROOT / 'artifacts/fresh-completion-context-CPU-001.json'
TASK_IDS = ('22cc237_3', '27e1026_3', '287e338_3', '2a163ab_3', '60d0b5b_3',
            '692c77d_3', '82e2fac_3', 'b0a8eae_3', 'b7a9ee9_3', 'c901732_3', 'cf6abd2_3')
SOURCE_PATHS = (
    COHORT, BUILD, 'artifacts/partitions-v1/manifest.json',
    'artifacts/prepare_fresh_completion_metadata_CPU_001.py',
    'scripts/fresh_completion_metadata.py', 'tests/test_fresh_completion_metadata.py',
    'scripts/owned_appworld_v3.py', 'scripts/verification_token_counter.py',
    'scripts/verification_appworld_worker_v3.py', 'scripts/verification_readonly_registry_v3.py',
    'scripts/verification_program_v3.py', 'scripts/verification_worker_state_v3.py',
    'docker/Dockerfile.verification-v3', 'wll/appworld.py', 'wll/appworld_v2.py',
    'wll/prompts.py', 'wll/protocol.py',
)
_MAX_BYTES = 4_000_000
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_pairs)


def _path(value):
    raw = os.fspath(value)
    if (type(raw) is not str or not raw.startswith('/') or raw.startswith('//')
            or os.path.normpath(raw) != raw
            or '\x00' in raw or raw == '/'):
        raise ValueError('invalid_path')
    return Path(raw)


def _directory_fd(path):
    path = _path(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open('/', flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _private_directory(fd):
    info = os.fstat(fd)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError('invalid_private_directory')


def _private_read(path):
    path = _path(path)
    directory = _directory_fd(path.parent)
    fd = None
    try:
        _private_directory(directory)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1
                or not 0 < before.st_size <= _MAX_BYTES):
            raise ValueError('invalid_private_file')
        chunks = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
            if sum(map(len, chunks)) > _MAX_BYTES:
                raise ValueError('private_file_too_large')
        raw = b''.join(chunks)
        after = os.fstat(fd)
        if (len(raw) != before.st_size or (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino,
                after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError('private_file_changed')
        return raw
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def _utf8(value, *, nonempty=False):
    if type(value) is not str or (nonempty and not value.strip()):
        return False
    try:
        value.encode('utf-8')
    except UnicodeEncodeError:
        return False
    return True


def validate_manifest(value, *, data_dir, deps_dir, image):
    """Validate exact private fields without returning text in errors."""
    try:
        data, deps = str(_path(data_dir)), str(_path(deps_dir))
        if (type(value) is not dict or set(value) != {'version', 'cohort_sha256', 'split',
                'image', 'data_dir', 'deps_dir', 'load_ground_truth', 'tasks'}
                or type(value['version']) is not int or value['version'] != 1
                or value['cohort_sha256'] != COHORT_SHA256 or value['split'] != 'train'
                or image != IMAGE or value['image'] != image or value['data_dir'] != data
                or value['deps_dir'] != deps or value['load_ground_truth'] is not False
                or type(value['tasks']) is not list or len(value['tasks']) != len(TASK_IDS)):
            raise ValueError('invalid_manifest')
        tasks = []
        for task_id, row in zip(TASK_IDS, value['tasks'], strict=True):
            if (type(row) is not dict or set(row) != {'task_id', 'scenario_id', 'split',
                    'instruction', 'supervisor'} or row['task_id'] != task_id
                    or row['scenario_id'] != task_id.rsplit('_', 1)[0] or row['split'] != 'train'
                    or not _utf8(row['instruction'], nonempty=True) or not _utf8(row['supervisor'])):
                raise ValueError('invalid_task')
            tasks.append(Task(**row))
        return tasks
    except Exception:
        raise ValueError('fresh_metadata_invalid') from None


def load_tasks(path, expected_sha256, *, data_dir, deps_dir, image):
    """Load all eleven tasks from a hash-bound, owner-only file; never log values."""
    try:
        if type(expected_sha256) is not str or _SHA.fullmatch(expected_sha256) is None:
            raise ValueError('invalid_digest')
        raw = _private_read(path)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('digest_mismatch')
        return validate_manifest(_json(raw), data_dir=data_dir, deps_dir=deps_dir, image=image)
    except Exception:
        raise ValueError('fresh_metadata_invalid') from None


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources():
    return {name: _sha_file(ROOT / name) for name in SOURCE_PATHS}


def _preflight(data_dir, deps_dir, image):
    """Read only public source metadata and the official ID-only train list."""
    if (str(_path(data_dir)) != DATA_DIR or str(_path(deps_dir)) != DEPS_DIR or image != IMAGE
            or not Path(data_dir).is_dir() or not Path(deps_dir).is_dir()):
        raise ValueError('metadata_environment_changed')
    verify_installed_sources(Path(deps_dir) / 'appworld')
    sources = _sources()
    if sources[COHORT] != COHORT_SHA256 or sources[BUILD] != BUILD_SHA256:
        raise ValueError('metadata_source_binding_changed')
    cohort = _json((ROOT / COHORT).read_bytes())
    build = _json((ROOT / BUILD).read_bytes())
    partition_name = 'artifacts/partitions-v1/manifest.json'
    if (cohort.get('status') != 'proposed_not_registered'
            or cohort.get('selection', {}).get('proposed_ids') != list(TASK_IDS)
            or cohort.get('selection', {}).get('excluded_scenarios') != ['d0b1f43']
            or cohort.get('source_sha256', {}).get(partition_name) != sources[partition_name]
            or build.get('status') != 'passed' or build.get('exit_code') != 0
            or build.get('image') != IMAGE or build.get('GPU_jobs') != 0
            or build.get('model_calls') != 0 or build.get('sources_unchanged') is not True):
        raise ValueError('metadata_provenance_changed')
    for name, expected in build['source_sha256'].items():
        if sources.get(name) != expected:
            raise ValueError('metadata_image_source_changed')
    partition = _json((ROOT / partition_name).read_bytes())
    origin = partition['task_ids']['origin']
    selected = [task_id for task_id in origin if task_id.endswith('_3')
                and task_id.rsplit('_', 1)[0] != 'd0b1f43']
    if selected != list(TASK_IDS) or len({item.rsplit('_', 1)[0] for item in selected}) != 11:
        raise ValueError('metadata_cohort_changed')
    official_path = Path(data_dir) / 'datasets/train.txt'
    raw = official_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OFFICIAL_TRAIN_SHA256:
        raise ValueError('metadata_train_ids_changed')
    official = raw.decode('utf-8').splitlines()
    if len(official) != len(set(official)) or not set(TASK_IDS) <= set(official):
        raise ValueError('metadata_train_ids_invalid')
    return official, sources


class _Journal:
    """Private fsynced event journal; never accepts task values or raw errors."""

    def __init__(self, directory, name):
        self.path = directory / name
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                          | os.O_CLOEXEC, 0o600)
        os.fsync(self.fd)
        parent = _directory_fd(directory)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)

    def emit(self, event, **fields):
        allowed = {'op', 'task_id', 'intent', 'messages_sha256', 'input_tokens', 'error_type'}
        if (type(event) is not str or re.fullmatch('[a-z][a-z0-9_]{0,80}', event) is None
                or set(fields) - allowed):
            raise ValueError('metadata_event_invalid')
        raw = (json.dumps({'event': event, 'epoch': time.time(), **fields}, ensure_ascii=True,
                          allow_nan=False, sort_keys=True) + '\n').encode('ascii')
        pending = memoryview(raw)
        while pending:
            count = os.write(self.fd, pending)
            if count <= 0:
                raise OSError('metadata_event_write_failed')
            pending = pending[count:]
        os.fsync(self.fd)

    def close(self):
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)


def _write_exclusive(path, value, *, private=False):
    path = _path(path)
    directory = _directory_fd(path.parent)
    fd = None
    try:
        if private:
            _private_directory(directory)
        raw = (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                          indent=2) + '\n').encode('ascii')
        if len(raw) > _MAX_BYTES:
            raise ValueError('metadata_output_too_large')
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                     | os.O_CLOEXEC, 0o600 if private else 0o644, dir_fd=directory)
        pending = memoryview(raw)
        while pending:
            count = os.write(fd, pending)
            if count <= 0:
                raise OSError('metadata_write_failed')
            pending = pending[count:]
        os.fsync(fd)
        os.fsync(directory)
        return hashlib.sha256(raw).hexdigest(), len(raw)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def metadata_call(worker, op, emit, *, on_return=None, **kwargs):
    """Reject any RPC outside the fixed, ID-only list/describe protocol."""
    if not ((op == 'list' and kwargs == {'split': 'train'})
            or (op == 'describe' and set(kwargs) == {'task_id'}
                and type(kwargs['task_id']) is str and kwargs['task_id'] in TASK_IDS)):
        raise ValueError('metadata_rpc_forbidden')
    fields = {'op': op, **({'task_id': kwargs['task_id']} if op == 'describe' else {})}
    emit('metadata_call_attempt', **fields)
    result = worker.call(op, **kwargs)
    if on_return is not None:
        on_return()
    emit('metadata_call_returned', **fields)
    return result


def _inspect_isolation(intent, data_dir, deps_dir):
    identity = owned.inspect_owned_container(intent)
    if identity is None:
        raise ValueError('metadata_container_absent')
    raw = subprocess.run(['docker', 'inspect', '--type', 'container', identity], capture_output=True,
                         timeout=5, check=True).stdout
    containers = _json(raw)
    if type(containers) is not list or len(containers) != 1:
        raise ValueError('metadata_isolation_invalid')
    container = containers[0]
    host = container['HostConfig']
    expected = {'NetworkMode': 'none', 'ReadonlyRootfs': True, 'Privileged': False,
                'PidsLimit': 128, 'Memory': 3 * 1024**3, 'NanoCpus': 2 * 10**9,
                'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges']}
    mounts = {row['Destination']: row for row in container['Mounts'] if row['Type'] == 'bind'}
    wanted = {'/workspace/data': data_dir, '/opt/deps': deps_dir}
    if (container['Id'] != identity or container['Config']['User'] != '10001:10001'
            or any(host.get(key) != value for key, value in expected.items())
            or host.get('Devices') or host.get('DeviceRequests') or host.get('Runtime') == 'nvidia'
            or set(mounts) != set(wanted)
            or any(mounts[key]['Source'] != value or mounts[key]['RW'] is not False
                   for key, value in wanted.items())):
        raise ValueError('metadata_isolation_invalid')
    return {'container_id': identity, 'isolation_verified': True}


def _task(task_id, info):
    if (type(info) is not dict or set(info) != {'instruction', 'supervisor'}
            or not _utf8(info['instruction'], nonempty=True) or not _utf8(info['supervisor'])):
        raise ValueError('metadata_reply_invalid')
    return Task(task_id, task_id.rsplit('_', 1)[0], 'train', info['instruction'], info['supervisor'])


def _error_type(exc):
    name = type(exc).__name__
    return name if re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,127}', name) else 'OtherException'


def prepare(*, private_dir=PRIVATE, metadata_path=METADATA_PROOF, context_path=CONTEXT_PROOF,
            data_dir=DATA_DIR, deps_dir=DEPS_DIR, image=IMAGE, worker_factory=None,
            token_counter=count_input_tokens):
    """Prepare once, record failures and cleanup, and never retry an operation.

    Optional injected worker/counter are for synthetic CPU tests. The registered
    entrypoint uses the real owned worker and pinned CPU tokenizer defaults.
    """
    private_dir, metadata_path, context_path = map(_path, (private_dir, metadata_path, context_path))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('metadata_cuda_visibility_invalid')
    if any(os.path.lexists(path) for path in (private_dir, metadata_path, context_path)):
        raise FileExistsError('metadata_output_already_exists')
    if not callable(token_counter) or (worker_factory is not None and not callable(worker_factory)):
        raise ValueError('metadata_factory_invalid')
    started = time.monotonic()
    cleanup = {'creation_verified': False, 'creator_exit_code': None,
               'owned_containers_absent': [], 'verified': False}
    report = {'id': 'fresh-completion-metadata-CPU-001', 'status': 'failed',
              'cohort_sha256': COHORT_SHA256, 'task_ids': list(TASK_IDS), 'task_count': 0,
              'tasks_path': str(private_dir / 'tasks.json'), 'tasks_sha256': None,
              'tasks_bytes': None, 'image': image, 'data_dir': data_dir, 'deps_dir': deps_dir,
              'load_ground_truth': False, 'official_train_sha256': OFFICIAL_TRAIN_SHA256,
              'list_calls': 0, 'list_returns': 0, 'describe_calls': 0, 'describe_returns': 0,
              'world_start_calls': 0, 'execute_calls': 0, 'evaluate_calls': 0,
              'model_calls': 0, 'GPU_jobs': 0, 'cleanup': cleanup, 'source_sha256': {},
              'event_log_sha256': None, 'error_type': None, 'elapsed_seconds': None}
    journal = worker = None
    intents, tasks = [], []
    pending_interrupt = None
    failed = False
    cleanup_failed = False
    sources = {}

    def remember(exc):
        nonlocal failed, pending_interrupt
        failed = True
        if report['error_type'] is None:
            report['error_type'] = _error_type(exc)
        if pending_interrupt is None and isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            pending_interrupt = exc

    def own(intent):
        owned._validate_intent(intent)
        ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
        if (intents or intent['image'] != image or intent['owner_pid'] != os.getpid()
                or intent['owner_start_ticks'] != ticks):
            raise ValueError('metadata_intent_invalid')
        journal.emit('container_intent', intent=intent)
        intents.append(dict(intent))

    def returned(kind):
        report[kind + '_returns'] += 1

    try:
        parent = _directory_fd(private_dir.parent)
        try:
            os.mkdir(private_dir.name, 0o700, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)
        directory = _directory_fd(private_dir)
        try:
            _private_directory(directory)
        finally:
            os.close(directory)
        journal = _Journal(private_dir, 'metadata-events.jsonl')
        official, sources = _preflight(data_dir, deps_dir, image)
        report['source_sha256'] = sources
        factory = worker_factory or owned.OwnedAppWorldWorkerV3
        worker = factory(data_dir, image, deps_dir, on_container_intent=own)
        report['list_calls'] += 1
        listed = metadata_call(worker, 'list', journal.emit, on_return=lambda: returned('list'),
                               split='train')
        if type(listed) is not list or listed != official or len(intents) != 1:
            raise ValueError('metadata_train_list_invalid')
        isolation = _inspect_isolation(intents[0], data_dir, deps_dir)
        if (type(isolation) is not dict or isolation.get('isolation_verified') is not True
                or type(isolation.get('container_id')) is not str
                or _SHA.fullmatch(isolation['container_id']) is None
                or worker.proc.poll() is not None):
            raise ValueError('metadata_container_invalid')
        cleanup['creation_verified'] = True
        for task_id in TASK_IDS:
            report['describe_calls'] += 1
            info = metadata_call(worker, 'describe', journal.emit,
                                 on_return=lambda: returned('describe'), task_id=task_id)
            tasks.append(_task(task_id, info))
            report['task_count'] = len(tasks)
    except BaseException as exc:
        remember(exc)
    finally:
        if worker is not None:
            try:
                worker.close()
            except BaseException as exc:
                cleanup_failed = True
                remember(exc)
        for intent in intents:
            try:
                owned.remove_owned_container(intent)
                cleanup['owned_containers_absent'].append(owned.inspect_owned_container(intent) is None)
            except BaseException as exc:
                cleanup_failed = True
                remember(exc)
        if worker is not None:
            try:
                cleanup['creator_exit_code'] = worker.proc.poll()
            except BaseException as exc:
                cleanup_failed = True
                remember(exc)
        cleanup['verified'] = (not cleanup_failed and cleanup['creation_verified'] is True
            and type(cleanup['creator_exit_code']) is int
            and cleanup['owned_containers_absent'] == [True])
        if not cleanup['verified']:
            failed = True
            if report['error_type'] is None:
                report['error_type'] = 'MetadataCleanupUnverified'
        if journal is not None:
            try:
                journal.close()
                report['event_log_sha256'] = _sha_file(journal.path)
            except BaseException as exc:
                remember(exc)

    if not failed:
        try:
            if len(tasks) != len(TASK_IDS) or _sources() != sources:
                raise ValueError('metadata_sources_changed')
            verify_installed_sources(Path(deps_dir) / 'appworld')
            manifest = {'version': 1, 'cohort_sha256': COHORT_SHA256, 'split': 'train',
                        'image': image, 'data_dir': data_dir, 'deps_dir': deps_dir,
                        'load_ground_truth': False, 'tasks': [asdict(task) for task in tasks]}
            validate_manifest(manifest, data_dir=data_dir, deps_dir=deps_dir, image=image)
            digest, size = _write_exclusive(private_dir / 'tasks.json', manifest, private=True)
            if load_tasks(private_dir / 'tasks.json', digest, data_dir=data_dir,
                          deps_dir=deps_dir, image=image) != tasks:
                raise ValueError('metadata_round_trip_invalid')
            report.update(status='passed', tasks_sha256=digest, tasks_bytes=size)
        except BaseException as exc:
            remember(exc)
    report['elapsed_seconds'] = time.monotonic() - started
    metadata_sha, _ = _write_exclusive(metadata_path, report)
    if pending_interrupt is not None:
        raise pending_interrupt
    if report['status'] != 'passed':
        return {'metadata': report, 'context': None}

    context_started = time.monotonic()
    context = {'id': 'fresh-completion-context-CPU-001', 'status': 'failed',
               'metadata_proof_sha256': metadata_sha, 'tasks_sha256': report['tasks_sha256'],
               'cohort_sha256': COHORT_SHA256, 'image': image, 'model_path': MODEL_PATH,
               'tokenizer_files_sha256': dict(TOKENIZER_SHA256), 'prompt_version': 'appworld-v3',
               'input_limit': 15360, 'output_limit': 1024, 'rows': [], 'task_count': len(TASK_IDS),
               'initial_eligible_count': 0, 'context_count_calls': 0, 'context_count_returns': 0,
               'model_calls': 0, 'GPU_jobs': 0, 'CUDA_VISIBLE_DEVICES': '',
               'cuda_initialized': False, 'model_weights_loaded': False, 'source_sha256': sources,
               'event_log_sha256': None, 'error_type': None, 'elapsed_seconds': None}
    journal = None
    pending_interrupt = None
    try:
        journal = _Journal(private_dir, 'context-events.jsonl')
        for index, task in enumerate(tasks):
            messages = initial_messages(task, 'appworld-v3')
            messages_sha = fingerprint(messages)
            journal.emit('context_count_attempt', task_id=task.task_id, messages_sha256=messages_sha)
            context['context_count_calls'] += 1
            token_count = token_counter(messages)
            context['context_count_returns'] += 1
            if type(token_count) is not int or token_count < 1:
                raise ValueError('metadata_token_count_invalid')
            fits = token_count <= 15360
            context['rows'].append({'task_id': task.task_id, 'original_index': index,
                                    'messages_sha256': messages_sha, 'input_tokens': token_count,
                                    'fits_initial_input': fits})
            context['initial_eligible_count'] += int(fits)
            journal.emit('context_count_returned', task_id=task.task_id,
                         messages_sha256=messages_sha, input_tokens=token_count)
        if _sources() != sources:
            raise ValueError('metadata_sources_changed')
        context['status'] = 'passed'
    except BaseException as exc:
        context['error_type'] = _error_type(exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            pending_interrupt = exc
    finally:
        if journal is not None:
            try:
                journal.close()
                context['event_log_sha256'] = _sha_file(journal.path)
            except BaseException as exc:
                context.update(status='failed', error_type=_error_type(exc))
        context['elapsed_seconds'] = time.monotonic() - context_started
        _write_exclusive(context_path, context)
    if pending_interrupt is not None:
        raise pending_interrupt
    return {'metadata': report, 'context': context}
