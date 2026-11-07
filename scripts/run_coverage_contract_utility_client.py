"""One registered utility batch with private inputs and audited fresh branches.

The controller authenticates prior batch audits and physical releases, owns the
queue, service, and watchdog, and supplies a byte-bound prior ledger. This client
never treats an owner release attestation as an independent physical check.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import urllib.request
from dataclasses import asdict
from pathlib import Path

from scripts import recovery_evidence_utility as utility
from scripts import report_coverage_contract_utility_branch as audit
from scripts import run_coverage_contract_utility_batch as batch
from scripts import run_recovery_evidence_utility_branch as branch
from scripts.completion_batch import _bytes, _plain, _write
from scripts.completion_trace import _parent_fd
from scripts.documentation_feedback import DocumentationFeedbackWorld
from scripts.fresh_completion_metadata import OFFICIAL_TRAIN_SHA256, _directory_fd, _path
from scripts.owned_appworld_v3 import OwnedAppWorldEnvironmentV3, _validate_intent
from scripts.run_completion_lesson import _strict_json
from scripts.run_observed_utility_branch import initial_actor_messages
from scripts.verification_readonly_registry_v3 import SOURCE_SHA256 as RUNTIME_SOURCE_SHA256
from wll.experience import Experience
from wll.model import LocalModel
from wll.protocol import Task, fingerprint

ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-coverage-contract-utility-001'
PRIVATE_RUN_ROOT = Path('/home/hetaisheng/.local/share/workflow-learning-lab/coverage-contract-utility-001')
ENDPOINT = 'http://127.0.0.1:18990/v1'
MODEL_ID = utility.MODEL_ID
IMAGE = utility.FIXED['environment_image']
_ERROR = 'coverage_utility_client_invalid'
_LIMIT = 4_000_000
_SHA = re.compile('[a-f0-9]{64}\\Z')
_TASK_FIELDS = {'task_id', 'scenario_id', 'split', 'instruction', 'supervisor'}
_PAYLOAD_FIELDS = {'partition', 'tasks', 'candidate_provenance', 'candidate', 'placebo',
                   'placebo_provenance', 'messages'}


def _require(condition):
    if not condition:
        raise ValueError(_ERROR)


def _sha(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def _queue():
    from scripts import coverage_contract_gpu_queue
    return coverage_contract_gpu_queue


def _bound_private(path, expected, size=None):
    _require(_sha(expected) and (size is None or type(size) is int and 0 < size <= _LIMIT))
    directory, parsed = _parent_fd(path)
    fd = None
    try:
        fd = os.open(parsed.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                     dir_fd=directory)
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
            and stat.S_IMODE(before.st_mode) == 0o600 and before.st_nlink == 1
            and 0 < before.st_size <= _LIMIT and (size is None or before.st_size == size))
        chunks, total = [], 0
        while chunk := os.read(fd, 65536):
            total += len(chunk)
            _require(total <= _LIMIT)
            chunks.append(chunk)
        after = os.fstat(fd)
        def identity(item):
            return (item.st_dev, item.st_ino, item.st_uid, item.st_mode, item.st_nlink,
                    item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        raw = b''.join(chunks)
        _require(identity(before) == identity(after) and len(raw) == before.st_size
            and hashlib.sha256(raw).hexdigest() == expected)
        return raw
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def _proof(plan, name):
    _require(type(name) is str and name in plan['input_sha256'])
    raw = _queue().public_read(ROOT, name)
    _require(hashlib.sha256(raw).hexdigest() == plan['input_sha256'][name])
    value = _strict_json(raw)
    _require(type(value) is dict)
    return value


def load_prepared(plan):
    """Authenticate exact private files and reconstruct all initial actor messages."""
    try:
        _require(type(plan) is dict and plan.get('id') == ID and plan.get('task_ids') == list(utility.TASK_IDS))
        context, proof = _proof(plan, plan['context_proof']), _proof(plan, plan['protocol_proof'])
        _require(context['private_input_sha256'] == proof['private_input_sha256'] == plan['private_input_sha256']
            and context['private_input_bytes'] == plan['private_input_bytes']
            and proof['private_protocol_sha256'] == plan['private_protocol_sha256']
            and proof['private_protocol_bytes'] == plan['private_protocol_bytes']
            and proof['protocol_fingerprint'] == plan['protocol_fingerprint'])
        input_path, protocol_path = _path(context['private_input_path']), _path(proof['private_protocol_path'])
        _require(str(input_path) == plan['private_input_path'] and str(protocol_path) == plan['private_protocol_path'])
        raw = _bound_private(input_path, plan['private_input_sha256'], plan['private_input_bytes'])
        _require(raw.endswith(b'\n') and len(raw.splitlines()) == 1)
        row = _strict_json(raw)
        _require(type(row) is dict and set(row) == {'index', 'event', 'payload'}
            and type(row['index']) is int and row['index'] == 0 and row['event'] == 'utility_input_bundle')
        payload = row['payload']
        _require(type(payload) is dict and set(payload) == _PAYLOAD_FIELDS
            and type(payload['tasks']) is list and len(payload['tasks']) == len(utility.TASK_IDS))
        tasks = []
        for task_id, value in zip(utility.TASK_IDS, payload['tasks'], strict=True):
            _require(type(value) is dict and set(value) == _TASK_FIELDS
                and all(type(item) is str for item in value.values())
                and value['task_id'] == task_id and value['instruction'].strip())
            for item in value.values():
                item.encode('utf-8')
            tasks.append(Task(**value))
        partition = payload['partition']
        utility.validate_cohort(partition, tasks)
        candidate, placebo = Experience(**payload['candidate']), Experience(**payload['placebo'])
        protocol = _strict_json(_bound_private(protocol_path, plan['private_protocol_sha256'],
                                               plan['private_protocol_bytes']))
        _require(type(protocol) is dict and fingerprint(protocol) == plan['protocol_fingerprint']
            and protocol['environment_id'] == proof['environment_id']
            and protocol['environment_image'] == plan['image'] == IMAGE
            and protocol['expected_backend_static'] == plan['expected_backend_static']['B3'])
        utility._protocol(partition, tasks, protocol)
        utility._validate_protocol_provenance(protocol, candidate)
        utility._validate_placebo_provenance(payload['placebo_provenance'], protocol['design'], placebo=placebo)
        _require(payload['candidate_provenance'] == protocol['candidate_provenance']
            and payload['placebo_provenance'] == protocol['placebo_provenance']
            and fingerprint(payload['candidate_provenance']) == context['candidate_provenance_sha256']
            and fingerprint(asdict(candidate)) == context['candidate_sha256'] == protocol['candidate_sha256']
            and fingerprint(asdict(placebo)) == context['placebo_sha256'] == protocol['placebo_sha256']
            and placebo.experience_id == protocol['placebo_id']
            and placebo.source_scenarios == placebo.source_episode_hashes == []
            and context['placebo_provenance'] == payload['placebo_provenance']
            and context['task_ids'] == list(utility.TASK_IDS)
            and context['task_sha256'] == protocol['task_sha256'])
        messages, counts = payload['messages'], context['CPU_tokenization_proof']['results']
        expected_keys = {task.task_id + ':' + arm for task in tasks for arm in utility.ARMS}
        _require(type(messages) is dict and set(messages) == expected_keys and type(counts) is dict
            and expected_keys <= set(counts))
        for task in tasks:
            for arm, experience in zip(utility.ARMS, (None, candidate, placebo), strict=True):
                name = task.task_id + ':' + arm
                value, token_proof = messages[name], counts[name]
                _require(value == initial_actor_messages(task, experience, protocol['prompt_version'])
                    and type(token_proof) is dict and token_proof.get('kind') == 'messages'
                    and token_proof.get('content_sha256') == fingerprint(value)
                    and type(token_proof.get('input_tokens')) is int
                    and token_proof['input_tokens'] == protocol['initial_input_tokens'][task.task_id][arm])
        return {'partition': partition, 'tasks': tasks, 'candidate': candidate, 'placebo': placebo,
            'protocol': protocol, 'input_path': input_path, 'protocol_path': protocol_path}
    except Exception:
        raise ValueError(_ERROR) from None


def _digest_public(path):
    path = _path(path)
    directory = _directory_fd(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode))
        digest, length = hashlib.sha256(), 0
        while chunk := os.read(fd, 65536):
            length += len(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            and length == before.st_size)
        return digest.hexdigest()
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def _docs_files(root):
    directory = _directory_fd(root)
    found = set()
    try:
        for relative, dirs, files, fd in os.fwalk('.', dir_fd=directory, follow_symlinks=False):
            for name in dirs:
                _require(stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode))
            for name in files:
                if name.endswith('.json'):
                    _require(stat.S_ISREG(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode))
                    found.add(str(root / Path(relative) / name))
        return found
    finally:
        os.close(directory)


def verify_native_bindings(plan, input_path, protocol_path):
    """Check the registered public/runtime closure and only the two current private files."""
    try:
        bindings = plan['native_bindings']
        _require(type(bindings) is dict and type(plan['source_sha256']) is dict
            and type(plan['input_sha256']) is dict)
        for group in (plan['source_sha256'], plan['input_sha256']):
            _require(all(bindings.get(name) == sha for name, sha in group.items()))
        data, deps = _path(plan['data_dir']), _path(plan['deps_dir'])
        appworld, docs = deps / 'appworld', data / 'api_docs'
        docs_files = _docs_files(docs)
        _require(bool(docs_files) and docs_files <= bindings.keys())
        train, version = data / 'datasets/train.txt', data / 'version.txt'
        _require(bindings.get(str(train)) == OFFICIAL_TRAIN_SHA256)
        data_fd = _directory_fd(data)
        try:
            try:
                version_info = os.stat('version.txt', dir_fd=data_fd, follow_symlinks=False)
            except FileNotFoundError:
                version_info = None
            if version_info is not None:
                _require(stat.S_ISREG(version_info.st_mode) and str(version) in bindings)
        finally:
            os.close(data_fd)
        for relative, expected in RUNTIME_SOURCE_SHA256.items():
            name = os.path.normpath(str(appworld / relative))
            _require(bindings.get(name) == expected)
        private = {str(input_path): plan['private_input_sha256'], str(protocol_path): plan['private_protocol_sha256']}
        _require(len(private) == 2 and all(bindings.get(name) == sha for name, sha in private.items()))
        forbidden = {'tasks', 'dbs', 'ground_truth', 'groundtruth', 'solutions', 'task_solutions'}
        for name, expected in bindings.items():
            _require(type(name) is str and bool(name) and os.path.normpath(name) == name and _sha(expected))
            path = Path(name)
            if not path.is_absolute():
                _require(name in plan['source_sha256'] or name in plan['input_sha256'])
            elif name in private:
                _bound_private(path, expected)
            else:
                source = (path.is_relative_to(appworld) and path.suffix == '.py'
                    and not set(path.relative_to(appworld).parts) & forbidden)
                sibling = path == deps / 'fastapi_login/fastapi_login.py'
                _require(source or sibling or name in docs_files or path in {train, version})
                _require(_digest_public(path) == expected)
    except Exception:
        raise ValueError(_ERROR) from None


def _schedule():
    return [{'batch_index': index, 'phase': phase, 'phase_batch_index': number,
        'branches': [list(pair) for pair in values]}
        for index, (phase, number, values) in enumerate(
            [(phase, number, values) for phase in ('controls', 'prompted')
             for number, values in enumerate(utility.batch_schedules(phase))])]


def load_prior(plan_sha256, protocol, batch_index, path, expected_sha256):
    """Validate a controller-authenticated prior ledger, not physical release itself."""
    try:
        _require(_sha(plan_sha256) and type(batch_index) is int and 0 <= batch_index < len(_schedule()))
        start = sum(len(item['branches']) for item in _schedule()[:batch_index])
        if start == 0:
            _require(path is None and expected_sha256 is None)
            return []
        _require(_path(path) == PRIVATE_RUN_ROOT / f'prior-{batch_index:02d}.json')
        raw = _bound_private(path, expected_sha256)
        ledger = _strict_json(raw)
        _require(type(ledger) is dict and set(ledger) == {'study_id', 'plan_sha256',
            'protocol_fingerprint', 'records'} and ledger['study_id'] == ID
            and ledger['plan_sha256'] == plan_sha256 and ledger['protocol_fingerprint'] == fingerprint(protocol)
            and type(ledger['records']) is list and len(ledger['records']) == start)
        for row, (name, arm) in zip(ledger['records'], utility.branch_schedule(), strict=False):
            _require(type(row) is dict and set(row) == {'task_id', 'scenario_id', 'arm', 'outcome',
                'valid_for_continuation', 'released', 'protocol_sha256'}
                and row['task_id'] == name and row['scenario_id'] == name.rsplit('_', 1)[0]
                and row['arm'] == arm and type(row['outcome']) is bool
                and row['valid_for_continuation'] is True and row['released'] is True
                and row['protocol_sha256'] == fingerprint(protocol))
        return ledger['records']
    except Exception:
        raise ValueError(_ERROR) from None


def validate_health(health, plan):
    try:
        _require(type(health) is dict and health.get('model') == MODEL_ID and health.get('ready') is True
            and health.get('busy') is False and type(health.get('max_context')) is int
            and health['max_context'] == 16384 and type(health.get('max_output_tokens')) is int
            and health['max_output_tokens'] == 1024 and type(health.get('backend')) is dict)
        utility._backend(health['backend'])
        _require(utility.static_backend(health['backend']) == plan['expected_backend_static']['B3'])
        return copy.deepcopy(health['backend'])
    except Exception:
        raise ValueError(_ERROR) from None


def get_health():
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args, **_kwargs):
            return None
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open('http://127.0.0.1:18990/health', timeout=5) as response:
            raw = response.read(262145)
        _require(len(raw) <= 262144)
        return _strict_json(raw)
    except Exception:
        raise ValueError(_ERROR) from None


def _count_tokens(endpoint, messages, *, model_id, expected_backend):
    from scripts.coverage_contract_token_endpoint import count_tokens
    return count_tokens(endpoint, messages, model_id=model_id, expected_backend=expected_backend)


class Events:
    """Exclusive, fsynced metadata-only batch and branch event journal."""

    def __init__(self, path):
        directory, parsed = _parent_fd(path)
        self.fd, self.index, self.failed = None, 0, False
        try:
            self.fd = os.open(parsed.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                              0o600, dir_fd=directory)
            os.fchmod(self.fd, 0o600)
            info = os.fstat(self.fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1)
            os.fsync(self.fd)
            os.fsync(directory)
        except BaseException:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            raise
        finally:
            os.close(directory)

    def emit(self, event, **fields):
        try:
            fields_by_event = {
                'client_backend': {'backend', 'backend_sha256'},
                'batch_started': {'phase', 'batch_index', 'start_global_index', 'scheduled_branches', 'protocol_sha256'},
                'branch_attempt': {'global_index', 'task_id', 'arm', 'protocol_sha256', 'prior_records_sha256'},
                'branch_event': {'global_index', 'branch_event'},
                'branch_audited': {'global_index', 'record', 'cost', 'result_file'},
            }
            _require(not self.failed and event in fields_by_event and set(fields) == fields_by_event[event])
            if event == 'client_backend':
                _require(type(fields['backend']) is dict and fields['backend_sha256'] == fingerprint(fields['backend']))
            if 'global_index' in fields:
                _require(type(fields['global_index']) is int and 0 <= fields['global_index'] < 81)
            if event == 'batch_started':
                _require(fields['phase'] in {'controls', 'prompted'}
                    and all(type(fields[key]) is int and fields[key] >= 0
                            for key in ('batch_index', 'start_global_index', 'scheduled_branches'))
                    and _sha(fields['protocol_sha256']))
            if event == 'branch_attempt':
                _require(fields['task_id'] in utility.TASK_IDS and fields['arm'] in utility.ARMS
                    and _sha(fields['protocol_sha256']) and _sha(fields['prior_records_sha256']))
            if event == 'branch_event':
                value = fields['branch_event']
                _require(type(value) is dict and type(value.get('event')) is str)
                kind = value['event']
                if kind in audit._PUBLIC_FIELDS:
                    expected = audit._PUBLIC_FIELDS[kind]
                else:
                    _require(kind in audit._TRACE_FIELDS)
                    expected = {key + '_sha256' if key in {'messages', 'response', 'code', 'result', 'step'}
                                else key for key in audit._TRACE_FIELDS[kind]}
                _require(set(value) == {'event'} | expected)
                for key, item in value.items():
                    if key.endswith('_sha256') or key == 'container_id':
                        _require(_sha(item))
                    elif key in {'seed', 'tokens', 'step_index', 'value', 'prompt_tokens', 'completion_tokens'}:
                        _require(type(item) is int and item >= 0)
                    elif key in {'replayed', 'valid_boolean', 'verified', 'backend_matches'}:
                        _require(type(item) is bool)
                    elif key == 'error_type':
                        _require(type(item) is str and re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,127}', item))
                    elif key == 'environment_seed':
                        _require(type(item) is dict and item == {'requested': 100, 'effective': 100, 'verified': True}
                            and type(item['requested']) is int and type(item['effective']) is int
                            and item['verified'] is True)
                if kind == 'container_intent':
                    _validate_intent(value['intent'])
            if event == 'branch_audited':
                record, cost, result_file = fields['record'], fields['cost'], fields['result_file']
                _require(type(record) is dict and set(record) == {'task_id', 'scenario_id', 'arm', 'outcome',
                    'valid_for_continuation', 'released', 'protocol_sha256'}
                    and record['task_id'] in utility.TASK_IDS
                    and record['scenario_id'] == record['task_id'].rsplit('_', 1)[0]
                    and record['arm'] in utility.ARMS and (type(record['outcome']) is bool or record['outcome'] is None)
                    and type(record['valid_for_continuation']) is bool and type(record['released']) is bool
                    and _sha(record['protocol_sha256']))
                sanitized, incomplete = batch._cost(cost)
                _require(not incomplete and sanitized == cost and type(result_file) is dict
                    and set(result_file) == {'sha256', 'bytes'} and _sha(result_file['sha256'])
                    and type(result_file['bytes']) is int and result_file['bytes'] >= 0)
            row = {'index': self.index, 'event': event, **_plain(fields)}
            _write(self.fd, _bytes(row))
            self.index += 1
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)


def run_client(plan_sha256, batch_index, *, prior_sha256=None):
    """Run one registered batch; never start services, enqueue jobs, or retry."""
    events = None
    try:
        _require(_sha(plan_sha256) and type(batch_index) is int and 0 <= batch_index < len(_schedule()))
        plan = _queue().load_registration(ROOT, plan_sha256)
        _require(type(plan) is dict and plan.get('id') == ID and plan.get('batch_schedule') == _schedule()
            and plan.get('image') == IMAGE and plan.get('private_run_root') == str(PRIVATE_RUN_ROOT))
        directory = _directory_fd(PRIVATE_RUN_ROOT)
        try:
            info = os.fstat(directory)
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o700)
        finally:
            os.close(directory)
        prepared = load_prepared(plan)
        verify_native_bindings(plan, prepared['input_path'], prepared['protocol_path'])
        protocol = prepared['protocol']
        prior_path = None if batch_index == 0 else PRIVATE_RUN_ROOT / f'prior-{batch_index:02d}.json'
        prior = load_prior(plan_sha256, protocol, batch_index, prior_path, prior_sha256)
        row = _schedule()[batch_index]
        decision = utility.phase_decision(prepared['partition'], prepared['tasks'], protocol, prior)
        allowed = {'awaiting_controls'} if row['phase'] == 'controls' else {'ready_for_prompted', 'awaiting_prompted'}
        _require(decision['status'] in allowed)
        backend = validate_health(get_health(), plan)
        model = LocalModel(ENDPOINT, MODEL_ID, max_tokens=1024, temperature=.7, timeout=180)
        events = Events(PRIVATE_RUN_ROOT / f'batch-{batch_index:02d}-events.jsonl')
        events.emit('client_backend', backend=backend, backend_sha256=fingerprint(backend))

        def before_generation():
            _require(fingerprint(validate_health(get_health(), plan)) == fingerprint(backend))

        def token_count(messages):
            return _count_tokens(ENDPOINT, messages, model_id=MODEL_ID, expected_backend=backend)

        def run_one(task, experience, *, arm, protocol, output, emit):
            def world_factory(*, on_container_intent):
                world = OwnedAppWorldEnvironmentV3(task.task_id, plan['data_dir'], 40,
                    image=IMAGE, deps_dir=plan['deps_dir'], environment_seed=100,
                    on_container_intent=on_container_intent)
                return DocumentationFeedbackWorld(world)
            return branch.run_branch(task, experience, world_factory, model, output=output, arm=arm,
                protocol=protocol, expected_backend=backend, count_tokens=token_count,
                emit=emit, before_generation=before_generation)

        def audit_one(directory, task, experience, protocol, *, arm, expected_public_events):
            return audit.audit_branch(directory, task, experience, protocol, arm=arm,
                expected_backend=backend, expected_public_events=expected_public_events)

        result = batch.run_batch(prepared['partition'], prepared['tasks'], prepared['candidate'],
            prepared['placebo'], protocol, prior, phase=row['phase'], batch_index=row['phase_batch_index'],
            output=PRIVATE_RUN_ROOT / f'batch-{batch_index:02d}',
            run_one=run_one, audit_one=audit_one, emit=events.emit)
        return {'status': result['status'], 'study_id': ID, 'plan_sha256': plan_sha256,
            'batch_index': batch_index, 'protocol_fingerprint': fingerprint(protocol),
            'backend': backend, 'backend_sha256': fingerprint(backend), 'batch': result,
            'identity_scope': 'The client checks the backend snapshot and health; process and tunnel '
                'identity, queue ownership and heartbeat remain with the outer controller.',
            'prior_scope': 'The controller authenticates prior batch audits and fresh physical releases; '
                'this client checks only the bound ledger and its schedule prefix.'}
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise ValueError(_ERROR) from None
    finally:
        if events is not None:
            try:
                events.close()
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except BaseException:
                raise ValueError(_ERROR) from None


def main(argv=None):
    class Parser(argparse.ArgumentParser):
        def error(self, _message):
            raise ValueError(_ERROR)

    parser = Parser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--batch-index', type=int, required=True)
    parser.add_argument('--prior-sha256')
    try:
        args = parser.parse_args(argv)
        result = run_client(args.plan_sha256, args.batch_index, prior_sha256=args.prior_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result['status'] == 'complete' else 1
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        print(json.dumps({'status': 'error', 'error_type': 'Interrupted'}))
        return 1
    except BaseException:
        print(json.dumps({'status': 'error', 'error_type': 'ClientError'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
