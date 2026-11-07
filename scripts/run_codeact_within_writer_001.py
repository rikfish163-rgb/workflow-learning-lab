"""Unregistered CodeAct writer controller; import/help perform no operations.

The actual source must pass its independent gate and sealed-input registration.
This module has no Source export, Task, world, scorer, training or utility route.
Only private callbacks/files retain histories, candidates and service evidence.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from types import FunctionType

from scripts import codeact_within_experience_001 as experience
from scripts import codeact_within_writer_registration_001 as registration
from scripts import run_codeact_within_writer_cases_001 as cases
from scripts.native_appworld_model_client_004 import NativeModel
from scripts.native_appworld_worker_client import _decode, _encode, _exclusive
from scripts.verification_token_counter import _message_payload

ROOT = Path(__file__).resolve().parents[1]
ID, PLAN = registration.ID, registration.PLAN
PARENT_SECONDS = 94200
BASE_SHA256 = '1a738c877b1af6a53c1a9d2a95c476b5923c0d2e8381ffa8795895e86aff16b9'
_ERROR = 'codeact_within_writer_controller_failed'
_CORE = None


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _queue():
    from scripts import codeact_within_writer_gpu_queue_001
    return codeact_within_writer_gpu_queue_001


def _isolated(*, raw=None, _registration=None, _queue_module=None):
    path = ROOT / 'scripts/run_native_experience_writer_001.py'
    raw = path.read_bytes() if raw is None else raw
    _require(not path.is_symlink() and type(raw) is bytes and _hash(raw) == BASE_SHA256)
    spec = importlib.util.spec_from_file_location('scripts._codeact_within_writer_001_reuse', path)
    _require(spec is not None and spec.loader is not None)
    core = importlib.util.module_from_spec(spec)
    # Fixed human-authored repository source, never model-generated Python.
    exec(compile(raw, str(path), 'exec', dont_inherit=True), core.__dict__)
    core.registration = registration if _registration is None else _registration
    core._queue = (lambda: _queue()) if _queue_module is None else (lambda: _queue_module)
    core.ID, core.PLAN, core.PARENT_SECONDS = ID, PLAN, PARENT_SECONDS
    return core


def _reuse():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def unit_command(unit, arguments, *, root=ROOT, log_path):
    command = _reuse().unit_command(unit, arguments, root=root, log_path=log_path)
    old = 'scripts.run_native_experience_writer_001'
    positions = [i for i, value in enumerate(command) if old in value]
    _require(len(positions) == 1 and '--property=RuntimeMaxSec=94200' in command)
    i = positions[0]
    command[i] = command[i].replace(old, 'scripts.run_codeact_within_writer_001')
    return command


class _Runtime:
    def __init__(self, root):
        self.root = Path(root)
        self.service, self.reader = _reuse()._services(root)
        self.plan = None
        scope = dict(self.reader._read_private.__globals__)
        scope['PRIVATE_RUN_ROOT'] = Path(registration.PRIVATE_STATE_ROOT)
        for name in ('_private_directory', '_read_private'):
            function = getattr(self.reader, name)
            scope[name] = FunctionType(function.__code__, scope, name, function.__defaults__, function.__closure__)
        self._read_input = scope['_read_private']

    def __getattr__(self, name):
        return getattr(self.service, name)

    def preflight(self, sha):
        _require(socket.gethostname() == 'amax' and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
        plan, prepared = registration.private_preflight(self.root, sha)
        _require(_encode(registration.load_registration(_queue().REMOTE, sha, private=False)) == _encode(plan))
        cases.validate_preparation(prepared, plan['registered_inputs'])
        self.plan = self.service.plan = plan
        return plan, prepared

    def begin(self, plan, sha, slots):
        service = self.service
        service.private = Path(plan['private_run_root'])
        service.fd = self.reader._private_directory(service.private)
        ticks = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        with _exclusive(service.fd, 'study-intent.json') as stream:
            service._write(stream, dict(id=ID, plan_sha256=sha, sources=slots, automatic_retry=False,
                pid=os.getpid(), start_ticks=ticks,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                private_input_sha256=plan['private_input_sha256']))
        service.journal = _exclusive(service.fd, 'study-journal.jsonl')
        os.fsync(service.fd)

    def preflight_failure(self, report):
        return _reuse()._Runtime.preflight_failure(self, report)

    def binding(self, job, service):
        return _reuse()._Runtime.binding(self, job, service)

    def release(self, job, known_pids):
        return _reuse()._Runtime.release(self, job, known_pids)

    def check_input(self, sha, expected):
        fresh = registration.load_registration(self.root, sha, private=False)
        _require(_encode(fresh) == _encode(self.plan) and fresh['private_input'] == registration.PRIVATE_INPUT)
        raw = self._read_input(fresh['private_input'], maximum=64_000_000)
        _require(_hash(raw) == fresh['private_input_sha256'])
        prepared = _decode(raw)
        cases.validate_preparation(prepared, fresh['registered_inputs'])
        _require(_encode(prepared) == _encode(expected)
                 and _hash(self._read_input(fresh['private_input'], maximum=64_000_000)) == fresh['private_input_sha256'])
        return prepared

    def model(self, plan, service, *, max_tokens, source_index, stage, deadline):
        _require(type(source_index) is int and stage in {'writer', 'review'}
                 and max_tokens == (128 if stage == 'writer' else 256))
        return NativeModel(f'http://127.0.0.1:{registration.PORT}/v1', plan['model_id'],
            expected_backend=service['backend'], max_tokens=max_tokens, context_tokens=16384,
            temperature=0.7, timeout=120, deadline=deadline, _clock=self.now)


def run_writer(plan_sha, root=ROOT, _runtime=None):
    _require(registration._sha(plan_sha))
    runtime = _runtime or _Runtime(root)
    started = runtime.now()
    report = dict(id=ID, plan_sha256=plan_sha, status='unknown', automatic_retry=False,
        case_public=None, source_indices=None, case_result_sha256=None, pending_bundle_sha256=None,
        GPU_release=dict(verified=False, GPU_seconds=None, cuda_seconds=None), GPU_release_sha256=None,
        preflight_wall_seconds=None, parent_wall_seconds=None, whole_cost_known=False,
        canfreeze=False, accepted=False, content_certified=False, training_authorized=False,
        training_jobs=0, utility_or_dev_or_test_calls=0)
    begun = enqueued = False
    heartbeat = job = case_result = None
    known_pids, service, stage_frames, candidates, writer_status = [], None, {}, {}, {}
    try:
        plan, prepared = runtime.preflight(plan_sha)
        cases.validate_preparation(prepared, plan['registered_inputs'])
        report['preflight_wall_seconds'] = runtime.now() - started
        report['source_indices'] = [ctx['lineage']['source_index'] for ctx in prepared['contexts']]
        runtime.begin(plan, plan_sha, report['source_indices'])
        begun = True
        runtime.record(dict(event='inputs_bound', private_input_sha256=plan['private_input_sha256'],
                            registered_inputs=plan['registered_inputs']))
        runtime.monitor(empty_port=True)
        job = _queue().make_job(0, runtime.owner(), plan_sha)
        admission_started = runtime.now()
        runtime.record(dict(event='GPU_enqueue_intent', job=job))
        enqueued = True
        state = runtime.queue('enqueue', job)
        heartbeat = runtime.heartbeat(job)
        while service is None:
            heartbeat.check()
            runtime.monitor()
            _reuse()._source_core()._queue_state(state, runtime)
            _require(state.get('released') is False)
            known_pids = sorted(set(known_pids) | set(state.get('model_pids', [])))
            info = state.get('job')
            if type(info) is dict and info.get('status') == 'running':
                _require(runtime.epoch() - info['started_epoch'] < 900)
            _require(runtime.now() - admission_started < plan['admission_wait_seconds'])
            service = runtime.ready(job, state)
            if type(info) is dict and info.get('status') == 'running':
                _require(runtime.epoch() - info['started_epoch'] < 900)
            if service is None:
                runtime.wait(5)
                state = runtime.queue('pulse', job)
        started_epoch = state['job']['started_epoch']
        client_deadline = min(runtime.now() + 6000,
                              runtime.now() + 7200 - (runtime.epoch() - started_epoch) - 180)
        runtime.record(dict(event='GPU_ready', job=job, service=service, client_deadline=client_deadline))

        def guard():
            heartbeat.check()
            _require(runtime.now() < client_deadline and 7200 - (runtime.epoch() - started_epoch) > 180)
            fresh = runtime.binding(job, service)
            _reuse()._source_core()._queue_state(fresh, runtime)
            _require(fresh.get('released') is False and fresh.get('job', {}).get('status') == 'running')

        contexts = {ctx['lineage']['source_index']: ctx for ctx in prepared['contexts']}
        def record(event):
            runtime.record(dict(event='case_event', payload=event))
            if event['event'] == 'stage_start':
                key = event['source_index'], event['stage']
                _require(key not in stage_frames)
                stage_frames[key] = copy.deepcopy(event)
            elif event['event'] == 'stage_event' and event['stage'] == 'writer':
                payload, index = event['payload'], event['source_index']
                if payload['event'] == 'generation_result':
                    try:
                        candidates[index] = experience.parse_candidate(payload['response']['text'])
                    except Exception:
                        pass
                elif payload['event'] == 'writer_stop':
                    writer_status[index] = payload['status']

        def model_factory(*, max_tokens, source_index, stage, deadline):
            guard()
            fresh = runtime.check_input(plan_sha, prepared)
            _require(_encode(fresh) == _encode(prepared))
            frame = stage_frames[source_index, stage]
            _require(_encode(frame['deadline']) == _encode(deadline) and deadline <= client_deadline
                     and 0 < deadline - runtime.now() <= 240 and frame['max_tokens'] == max_tokens
                     and frame['seed'] == (17 if stage == 'writer' else 117) + source_index)
            if stage == 'writer':
                messages = experience.build_writer_messages(contexts[source_index])
            else:
                _require(writer_status[source_index] == 'candidate_unvalidated')
                messages = experience.build_review_messages(contexts[source_index], candidates[source_index])
            digest = _message_payload(messages)[1]
            _require(digest == _message_payload(frame['messages'])[1])
            guard()
            model = runtime.model(plan, service, max_tokens=max_tokens, source_index=source_index, stage=stage, deadline=deadline)
            return _reuse()._BoundModel(model, digest, guard)

        case_result = cases.run_cases(prepared, plan['registered_inputs'], model_factory=model_factory,
                                     record=record, deadline=client_deadline, _clock=runtime.now)
        cases.verify_public_projection(case_result, plan['registered_inputs'])
        report['case_public'] = case_result['public']
        report['case_result_sha256'] = runtime.write('writer-case-result.json', case_result)['sha256']
        report['pending_bundle_sha256'] = runtime.write('pending-bundle.json', case_result['private']['pending_bundle_rows'])['sha256']
        report['status'] = 'complete' if case_result['public']['whole_cost_known'] else 'unknown'
    except BaseException:
        report['status'] = 'unknown'
    finally:
        if heartbeat is not None:
            try:
                heartbeat.stop()
            except BaseException:
                report['status'] = 'unknown'
        if enqueued:
            release = dict(verified=False, GPU_seconds=None, cuda_seconds=None)
            try:
                release = runtime.release(job, known_pids)
                _require(release['verified'] is True and release.get('runtime_overrun') is False
                         and cases._finite(release.get('GPU_seconds')))
            except BaseException:
                report['status'] = 'unknown'
            try:
                report['GPU_release_sha256'] = runtime.write('GPU-release.json', release)['sha256']
            except BaseException:
                report['status'] = 'unknown'
            valid_cost = cases._finite(release.get('GPU_seconds'))
            report['GPU_release'] = dict(verified=release.get('verified') is True and valid_cost,
                GPU_seconds=release.get('GPU_seconds') if valid_cost else None, cuda_seconds=None)
        report['parent_wall_seconds'] = runtime.now() - started
        report['whole_cost_known'] = (report['status'] == 'complete' and report['GPU_release']['verified'] is True
            and report['case_public'] is not None and report['case_public']['whole_cost_known'] is True)
        if begun:
            try:
                if case_result is not None and report['case_result_sha256'] is None:
                    report['case_result_sha256'] = runtime.write('writer-case-result.json', case_result)['sha256']
                runtime.record(dict(event='study_stop', report=report))
                runtime.write('study-result.json', report)
            except BaseException:
                report.update(status='unknown', whole_cost_known=False)
        else:
            try:
                runtime.preflight_failure(report)
            except BaseException:
                pass
        runtime.close()
    return report


def launch_writer(plan_sha, *, root=ROOT):
    _require(socket.gethostname() == 'amax' and os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    runtime = _Runtime(root)
    plan = registration.load_registration(root, plan_sha, private=False)
    descriptor = runtime.reader._private_directory(Path(plan['private_run_root']))
    try:
        unit = 'wll-native-source-codeact-writer-' + runtime.owner() + '.service'
        log = Path(plan['private_run_root']) / 'writer-parent.log'
        with _exclusive(descriptor, log.name):
            pass
        command = unit_command(unit, ['--run', '--plan-sha256', plan_sha], root=root, log_path=log)
        with _exclusive(descriptor, 'parent-launch-intent.json') as stream:
            runtime.service._write(stream, dict(unit=unit, argv=command, plan_sha256=plan_sha))
        answer = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=10, check=False)
        _require(answer.returncode == 0)
        spec = dict(kind='pilot', unit=unit, command=command)
        runtime.service.units[unit] = spec
        state = runtime.unit_status(spec)
        _require(state['verified'] is True)
        return dict(unit=unit, plan_sha256=plan_sha, status='launch_returned_not_completion', main_pid=state['main_pid'])
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', action='store_true')
    group.add_argument('--launch', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.launch:
            print(json.dumps(launch_writer(args.plan_sha256), sort_keys=True))
            return 0
        def interrupted(*_):
            raise KeyboardInterrupt
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        report = run_writer(args.plan_sha256)
        print(json.dumps(dict(id=ID, plan_sha256=args.plan_sha256, status=report['status'],
            whole_cost_known=report['whole_cost_known'], pending_count=(report.get('case_public') or {}).get('pending_count'))))
        return 0 if report['status'] == 'complete' else 1
    except BaseException:
        sys.stderr.write(_ERROR + '\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
