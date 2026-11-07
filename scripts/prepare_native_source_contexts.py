"""Prepare the fixed native source pilot's initial contexts on CPU only.

The only public output is an allowlisted proof containing hashes and counts.
Task metadata and messages stay in memory; diagnostics and the exclusive
receipt stay in the private remote-backed state directory.
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import sys
from pathlib import Path

from scripts import native_source_registration as registration
from scripts.verification_token_counter import _message_payload

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path('/home/data4/hts/workflow-learning-lab/runtime-remote-001/workspace')
PRIVATE_DIR_NAME = 'native-source-context-preparation-CPU-001'
PUBLIC_PROOF = 'artifacts/native-source-initial-context-CPU-001.json'
PREPARATION = 'scripts/prepare_native_source_contexts.py'
PROOF_SOURCES = (registration.NATIVE_SOURCES | registration.PREPARATION_SOURCES |
                 {'scripts/native_source_registration.py'})
MAX_INPUT_TOKENS = 15360
_SHA = re.compile(r'[a-f0-9]{64}')
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_NEW_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


class ContextPreparationError(RuntimeError):
    """A content-free preparation failure; detailed diagnostics remain private."""


def _require(condition, code='context_preparation_failed'):
    if not condition:
        raise ContextPreparationError(code)


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def _source_hashes(root):
    return {name: hashlib.sha256(registration.public_read(root, name)).hexdigest()
            for name in sorted(PROOF_SOURCES)}


@contextlib.contextmanager
def _forbid_ground_truth(ground_truth_type, progress):
    original = vars(ground_truth_type).get('load')
    _require(original is not None)

    def forbidden(_cls, *_args, **_kwargs):
        progress['ground_truth_load_attempts'] += 1
        raise ContextPreparationError('ground_truth_forbidden')

    ground_truth_type.load = classmethod(forbidden)
    try:
        yield
    finally:
        ground_truth_type.load = original


def _task_metadata(task, task_id):
    _require(task.id == task_id and task.ground_truth is None)
    return {
        'instruction': task.instruction,
        'supervisor': {key: task.supervisor[key]
                       for key in ('first_name', 'last_name', 'email', 'phone_number')
                       if key in task.supervisor},
        'app_descriptions': task.app_descriptions,
    }


def prepare_contexts(root, *, task_type, ground_truth_type, build_messages, count_tokens, progress=None):
    """Measure all fixed tasks through injected trusted CPU-only dependencies."""
    progress = {} if progress is None else progress
    progress.update(stage='source_snapshot', task_metadata_loads=0, ground_truth_load_attempts=0)
    try:
        _require(len(registration.TASK_IDS) == len(set(registration.TASK_IDS)) == 12)
        sources = _source_hashes(root)
        rows = []
        with _forbid_ground_truth(ground_truth_type, progress):
            for task_id in registration.TASK_IDS:
                progress['stage'] = 'task_metadata'
                task = task_type.load(task_id=task_id, load_ground_truth=False)
                _require(progress['ground_truth_load_attempts'] == 0, 'ground_truth_forbidden')
                progress['task_metadata_loads'] += 1
                metadata = _task_metadata(task, task_id)
                metadata_raw = _encoded(metadata)
                progress['stage'] = 'initial_context'
                messages = build_messages(metadata, prefix_records=None, review_hint=None)
                messages_sha = _message_payload(messages)[1]
                _require(_encoded(metadata) == metadata_raw, 'metadata_changed')
                progress['stage'] = 'cpu_token_count'
                count = count_tokens(messages)
                _require(type(count) is int and count >= 1, 'invalid_token_count')
                _require(count <= MAX_INPUT_TOKENS, 'initial_context_over_limit')
                _require(_message_payload(messages)[1] == messages_sha
                         and _encoded(metadata) == metadata_raw, 'context_changed')
                rows.append({'task_id': task_id, 'input_tokens': count, 'initial_fits': True,
                             'metadata_sha256': hashlib.sha256(metadata_raw).hexdigest(),
                             'messages_sha256': messages_sha})
        progress['stage'] = 'source_recheck'
        _require(progress['ground_truth_load_attempts'] == 0 and _source_hashes(root) == sources,
                 'bound_source_changed')
        return {
            'id': registration.CONTEXT_PROOF_ID, 'status': 'measured_CPU_initial_contexts',
            'task_ids': list(registration.TASK_IDS), 'rows': rows, 'source_sha256': sources,
            'preparation': {'path': PREPARATION, 'sha256': sources[PREPARATION]},
            'model_path': registration.MODEL_PATH, 'model_config_sha256': registration.MODEL_CONFIG_SHA256,
            'tokenizer_files_sha256': dict(registration.TOKENIZER_SHA256), 'CUDA_VISIBLE_DEVICES': '',
            'cuda_initialized': False, 'model_weights_loaded': False, 'GPU_jobs': 0, 'model_calls': 0,
            'task_worlds_started': 0, 'task_metadata_loads': progress['task_metadata_loads'],
            'ground_truth_load_attempts': progress['ground_truth_load_attempts'],
        }
    except ContextPreparationError:
        raise
    except Exception:
        raise ContextPreparationError('context_preparation_failed') from None


def _check_public_proof(proof):
    fields = {'id', 'status', 'task_ids', 'rows', 'source_sha256', 'preparation', 'model_path',
              'model_config_sha256', 'tokenizer_files_sha256', 'CUDA_VISIBLE_DEVICES',
              'cuda_initialized', 'model_weights_loaded', 'GPU_jobs', 'model_calls',
              'task_worlds_started', 'task_metadata_loads', 'ground_truth_load_attempts'}
    _require(type(proof) is dict and set(proof) == fields)
    _require(proof['id'] == registration.CONTEXT_PROOF_ID
             and proof['status'] == 'measured_CPU_initial_contexts'
             and proof['task_ids'] == list(registration.TASK_IDS)
             and proof['model_path'] == registration.MODEL_PATH
             and proof['model_config_sha256'] == registration.MODEL_CONFIG_SHA256
             and proof['tokenizer_files_sha256'] == registration.TOKENIZER_SHA256
             and proof['CUDA_VISIBLE_DEVICES'] == '' and proof['cuda_initialized'] is False
             and proof['model_weights_loaded'] is False)
    _require(all(type(proof[key]) is int and proof[key] == 0 for key in (
        'GPU_jobs', 'model_calls', 'task_worlds_started', 'ground_truth_load_attempts'))
        and type(proof['task_metadata_loads']) is int and proof['task_metadata_loads'] == 12)
    sources = proof['source_sha256']
    _require(type(sources) is dict and set(sources) == PROOF_SOURCES
             and all(type(value) is str and _SHA.fullmatch(value) for value in sources.values())
             and proof['preparation'] == {'path': PREPARATION, 'sha256': sources[PREPARATION]})
    rows = proof['rows']
    _require(type(rows) is list and len(rows) == len(registration.TASK_IDS))
    for task_id, row in zip(registration.TASK_IDS, rows, strict=True):
        _require(type(row) is dict and set(row) == {'task_id', 'input_tokens', 'initial_fits',
                                                     'metadata_sha256', 'messages_sha256'}
                 and row['task_id'] == task_id and type(row['input_tokens']) is int
                 and 1 <= row['input_tokens'] <= MAX_INPUT_TOKENS and row['initial_fits'] is True
                 and all(type(row[key]) is str and _SHA.fullmatch(row[key])
                         for key in ('metadata_sha256', 'messages_sha256')))


def _exclusive_file(directory_fd, name):
    descriptor = os.open(name, _NEW_FLAGS, 0o600, dir_fd=directory_fd)
    try:
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, 'wb')
    except BaseException:
        os.close(descriptor)
        raise


def _receipt(file, event):
    file.write(_encoded(event) + b'\n')
    file.flush()
    os.fsync(file.fileno())


def _publish_proof(root, private_fd, raw):
    staging_name = 'context-proof.staged.json'
    with _exclusive_file(private_fd, staging_name) as staging:
        staging.write(raw)
        staging.flush()
        os.fsync(staging.fileno())
    os.fsync(private_fd)
    artifact_fd = registration._directory(Path(root) / 'artifacts')
    try:
        os.link(staging_name, Path(PUBLIC_PROOF).name, src_dir_fd=private_fd,
                dst_dir_fd=artifact_fd, follow_symlinks=False)
        os.unlink(staging_name, dir_fd=private_fd)
        os.fsync(private_fd)
        os.fsync(artifact_fd)
    finally:
        os.close(artifact_fd)


def _run_and_publish(root, private_fd, receipt, producer):
    progress = {'stage': 'initializing', 'task_metadata_loads': 0, 'ground_truth_load_attempts': 0}
    _receipt(receipt, {'status': 'started'})
    try:
        proof = producer(progress)
        _check_public_proof(proof)
        progress['stage'] = 'publication_source_recheck'
        _require(_source_hashes(root) == proof['source_sha256'], 'bound_source_changed')
        raw = _encoded(proof) + b'\n'
        digest = hashlib.sha256(raw).hexdigest()
        _receipt(receipt, {'status': 'ready_to_publish', 'proof_sha256': digest})
        _publish_proof(root, private_fd, raw)
        _receipt(receipt, {'status': 'published', 'proof_sha256': digest})
    except BaseException as error:
        safe_codes = {'initial_context_over_limit', 'invalid_token_count', 'ground_truth_forbidden',
                      'metadata_changed', 'context_changed', 'bound_source_changed',
                      'tokenizer_or_config_changed', 'tokenizer_or_runtime_changed'}
        reason = (error.args[0] if type(error) is ContextPreparationError and len(error.args) == 1
                  and type(error.args[0]) is str and error.args[0] in safe_codes
                  else 'context_preparation_failed')
        _receipt(receipt, {'status': 'failed', 'stage': progress['stage'],
                           'reason': reason,
                           'task_metadata_loads': progress['task_metadata_loads'],
                           'ground_truth_load_attempts': progress['ground_truth_load_attempts']})
        raise


def _host_preflight():
    _require(socket.gethostname() == 'amax' and ROOT == WORKSPACE and Path.cwd().resolve() == WORKSPACE
             and sys.executable == registration.PYTHON and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
             and os.environ.get('TMPDIR') == registration.TMPDIR,
             'unauthorized_preparation_environment')
    for path in (registration.PRIVATE_STATE_ROOT, registration.TMPDIR, registration.ASSET_ROOT):
        registration._private_directory(path)
    for name in ('data', 'deps'):
        descriptor = registration._directory(Path(registration.ASSET_ROOT) / name)
        os.close(descriptor)
    _require(shutil.disk_usage(registration.TMPDIR).free >= registration.MINIMUM_FREE_BYTES)


def _runtime_hashes():
    base = Path(registration.MODEL_PATH)
    return {name: registration._absolute_hash(str(base / name))
            for name in ('config.json', *sorted(registration.TOKENIZER_SHA256))}


def _cuda_uninitialized():
    torch = sys.modules.get('torch')
    return torch is None or not torch.cuda.is_initialized()


@contextlib.contextmanager
def _native_task_classes():
    root = Path(registration.ASSET_ROOT)
    deps = root / 'deps'
    os.environ['APPWORLD_ROOT'] = str(root)
    os.environ['APPWORLD_CACHE'] = registration.TMPDIR
    sys.path.insert(0, str(deps))
    import dotenv

    original = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *_args, **_kwargs: False
    try:
        import appworld
        from appworld.common.path_store import path_store
        from appworld.ground_truth import GroundTruth
        from appworld.task import Task

        _require(Path(appworld.__file__).resolve().parent == deps / 'appworld')
        path_store.update_root(str(root))
        _require(path_store.data == str(root / 'data'))
        yield Task, GroundTruth
    finally:
        dotenv.load_dotenv = original
        sys.path.remove(str(deps))


def _actual_preparation(progress):
    from scripts import native_appworld_actor as actor
    from scripts import native_verification_token_counter as counter

    _require(counter.MODEL_PATH == registration.MODEL_PATH and counter.PYTHON_PATH == registration.PYTHON
             and counter.TOKENIZER_SHA256 == registration.TOKENIZER_SHA256
             and counter.TMPDIR == registration.TMPDIR and _cuda_uninitialized())
    expected_runtime = {'config.json': registration.MODEL_CONFIG_SHA256, **registration.TOKENIZER_SHA256}
    _require(_runtime_hashes() == expected_runtime, 'tokenizer_or_config_changed')
    with _native_task_classes() as (task, ground_truth):
        proof = prepare_contexts(ROOT, task_type=task, ground_truth_type=ground_truth,
                                 build_messages=actor.build_actor_messages,
                                 count_tokens=counter.count_input_tokens, progress=progress)
    _require(_cuda_uninitialized() and _runtime_hashes() == expected_runtime,
             'tokenizer_or_runtime_changed')
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args(argv)
    _host_preflight()
    os.umask(0o077)
    root_fd = registration._directory(registration.PRIVATE_STATE_ROOT)
    try:
        os.mkdir(PRIVATE_DIR_NAME, 0o700, dir_fd=root_fd)
        os.fsync(root_fd)
        private_fd = os.open(PRIVATE_DIR_NAME, _DIR_FLAGS, dir_fd=root_fd)
    finally:
        os.close(root_fd)
    try:
        with _exclusive_file(private_fd, 'receipt.jsonl') as receipt:
            with _exclusive_file(private_fd, 'diagnostics.log') as log:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                _run_and_publish(ROOT, private_fd, receipt, _actual_preparation)
    finally:
        os.close(private_fd)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        sys.stderr.write('native_source_context_preparation_failed\n')
        raise SystemExit(1) from None
