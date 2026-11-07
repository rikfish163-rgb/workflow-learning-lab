"""One registered, private B3 writer call; no HTTP, world execution or training."""

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path

from scripts.completion_lesson import SYSTEM, parse_candidate
from scripts.completion_trace import PrivateTrace, _exception_type, _parent_fd, _plain_value
from scripts.control_plane_diagnostics import validate_directory
from scripts.run_observed_lesson import request_digest
from scripts.run_qwen38_b3_dev import SERVICE_SHA, static_backend
from scripts.serve_qwen38 import validate_request
from scripts.serve_qwen38_b3 import MODEL_IDS, loaded_engine
from scripts.serve_qwen38_b3 import preflight as service_preflight
from wll.protocol import fingerprint
from wll.storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-completion-lesson-002'
PLAN = 'artifacts/qwen38-completion-lesson-plan-002.json'
OUTPUT = 'artifacts/' + ID
TOKENIZER_FILES = {'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                   'merges.txt', 'chat_template.jinja'}
REQUIRED_SOURCES = {
    'scripts/run_completion_lesson.py', 'scripts/queue_completion_lesson.py',
    'scripts/completion_lesson.py', 'scripts/completion_trace.py',
    'scripts/control_plane_diagnostics.py', 'scripts/run_observed_lesson.py',
    'scripts/run_qwen38_b3_dev.py', 'scripts/serve_qwen38.py',
    'scripts/serve_qwen38_b3.py', 'wll/protocol.py', 'wll/storage.py',
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular_file_required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


def _load_registration(root, sha):
    from scripts.queue_completion_lesson import load_registration

    return load_registration(root, sha)


def _validate_worker_entry(root, sha):
    from scripts.queue_completion_lesson import validate_worker_entry

    return validate_worker_entry(root, sha)


def _strict_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate_json_key')
            value[key] = item
        return value

    def invalid(_):
        raise ValueError('nonfinite_json')

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def _engine_json(value):
    """Match the HTTP JSON boundary for trusted in-process engine metadata."""
    return _plain_value(_strict_json(json.dumps(value, allow_nan=False)))


def _read_request(plan):
    directory, path = _parent_fd(plan['private_input'])
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1,
                'private_input_permissions')
        require(0 < info.st_size <= 8 * 1024 * 1024, 'private_input_size')
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            raw = stream.read(8 * 1024 * 1024 + 1)
        require(hashlib.sha256(raw).hexdigest() == plan['private_input_sha256'],
                'private_input_hash')
        require(raw.endswith(b'\n') and len(raw.splitlines()) == 1, 'one_private_request_required')
        row = _strict_json(raw)
        require(type(row) is dict and set(row) == {'index', 'event', 'request'}
                and type(row['index']) is int and row['index'] == 0
                and row['event'] == 'writer_request', 'private_request_record')
        request = _plain_value(row['request'])
        require(type(request) is dict and set(request) == {
            'model', 'messages', 'seed', 'max_tokens', 'temperature'}, 'writer_request_fields')
        validate_request(request, MODEL_IDS['B3'])
        require(request['model'] == plan['model_id'] and type(request['seed']) is int
                and request['seed'] == 5017 and type(request['max_tokens']) is int
                and request['max_tokens'] == 1024
                and type(request['temperature']) in (int, float)
                and request['temperature'] == 0.0, 'writer_request_settings')
        require(len(request['messages']) == 2
                and request['messages'][0] == {'role': 'system', 'content': SYSTEM}
                and request['messages'][1]['role'] == 'user', 'writer_prompt_changed')
        require(request_digest(request) == plan['request_sha256'], 'writer_request_hash')
        return request
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def preflight(sha, *, root=ROOT):
    require(_sha(sha), 'plan_sha256_required')
    root = Path(root)
    plan = _load_registration(root, sha)
    require(plan.get('id') == ID and plan.get('status') == 'registered'
            and REQUIRED_SOURCES <= plan['source_sha256'].keys(), 'writer_source_binding')
    fixed = {'source_scenario': 'cf6abd2', 'model_id': MODEL_IDS['B3'], 'seed': 5017,
             'max_tokens': 1024, 'maximum_calls': 1, 'gpu_seconds': 1800}
    require(all(type(plan.get(k)) is type(v) and plan[k] == v for k, v in fixed.items())
            and type(plan.get('temperature')) in (int, float) and plan['temperature'] == 0.0
            and plan.get('automatic_retry') is False and plan.get('training_authorized') is False,
            'writer_plan_settings')
    require(plan.get('service_plan_sha256') == SERVICE_SHA
            and _sha(plan.get('gpu_policy_sha256'))
            and all(_sha(plan.get(k)) for k in ('private_input_sha256', 'request_sha256'))
            and type(plan.get('source_trace_sha256')) is list
            and len(plan['source_trace_sha256']) == 2
            and len(set(plan['source_trace_sha256'])) == 2
            and all(_sha(value) for value in plan['source_trace_sha256']), 'writer_provenance')
    context = plan.get('CPU_context')
    require(type(context) is dict and type(context.get('prompt_tokens')) is int
            and 0 < context['prompt_tokens'] <= 15360
            and type(context.get('max_context')) is int and context['max_context'] == 16384
            and type(context.get('tokenizer_files_sha256')) is dict
            and set(context['tokenizer_files_sha256']) == TOKENIZER_FILES
            and all(_sha(value) for value in context['tokenizer_files_sha256'].values()),
            'writer_context_proof')
    backend = _plain_value(plan.get('expected_backend_static'))
    require(type(backend) is dict and backend.get('arm') == 'B3'
            and backend.get('service_plan_sha256') == SERVICE_SHA, 'writer_backend_binding')
    directory, private_output = _parent_fd(plan['private_output'])
    os.close(directory)
    require(private_output != Path(plan['private_input'])
            and not os.path.lexists(private_output) and not os.path.lexists(root / OUTPUT),
            'writer_output_exists')
    request = _read_request(plan)
    service_preflight(SERVICE_SHA, MODEL_IDS['B3'], arm='B3')
    return plan, request


def _prompt_tokens(engine, request):
    rendered = engine.tokenizer.apply_chat_template(request['messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = engine.tokenizer(rendered, add_special_tokens=False)['input_ids']
    require(type(ids) is list and ids and all(type(value) is int for value in ids),
            'invalid_tokenizer_result')
    return len(ids)


def _worker_identity():
    raw = Path('/proc/self/stat').read_text()
    pid = int(raw.split(' (', 1)[0])
    fields = raw.rsplit(')', 1)[1].split()
    group, ticks = int(fields[2]), int(fields[19])
    require(pid == os.getpid() and group == os.getpgrp() and ticks > 0, 'worker_identity_changed')
    return {'pid': pid, 'process_group': group, 'start_ticks': ticks}


def run(sha, *, check_only=False, root=ROOT):
    root = Path(root)
    plan, request = preflight(sha, root=root)
    if check_only:
        return {'id': ID, 'status': 'passed_preflight', 'plan_sha256': sha,
                'request_sha256': plan['request_sha256'], 'generation_calls': 0,
                'training_authorized': False, 'uses_gpu': False}
    _validate_worker_entry(root, sha)
    public, private = root / OUTPUT, Path(plan['private_output'])
    public.mkdir(exist_ok=False)
    directory, _ = _parent_fd(private)
    try:
        os.mkdir(private.name, mode=0o700, dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(directory)
    validate_directory(private)
    started = time.monotonic()
    report = {'id': ID, 'plan_sha256': sha, 'status': 'running', 'candidate_status': 'not_attempted',
        'worker_identity': _worker_identity(),
        'phase': 'preparing', 'request_sha256': plan['request_sha256'],
        'private_input_sha256': plan['private_input_sha256'], 'source_scenario': plan['source_scenario'],
        'source_trace_sha256': list(plan['source_trace_sha256']), 'model_id': plan['model_id'],
        'service_plan_sha256': SERVICE_SHA, 'generation_calls': 0, 'generation_returns': 0,
        'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0},
        'unknown_additional_response_cost': False, 'trace_closed': False,
        'admission_checked': False, 'model_loaded': False, 'accepted_experiences': 0,
        'content_certified': False, 'training_authorized': False, 'gpu_release_verified': False}
    trace = None

    def save():
        atomic_json(public / 'worker-report.json', report)

    def admitted(_gpu):
        _validate_worker_entry(root, sha)
        report['admission_checked'] = True

    try:
        save()
        trace = PrivateTrace(private / 'worker-trace.jsonl')
        trace.record('writer_request', request=request)
        report['phase'] = 'loading'
        with loaded_engine(SERVICE_SHA, MODEL_IDS['B3'], arm='B3', admitted=admitted) as engine:
            report['model_loaded'] = True
            backend = _engine_json(engine.backend)
            require(fingerprint(static_backend(backend)) == fingerprint(plan['expected_backend_static'])
                    and engine.model_id == MODEL_IDS['B3'] and type(engine.calls) is int
                    and engine.calls == 0 and engine.failed is False, 'loaded_backend_changed')
            report['backend_sha256'] = fingerprint(backend)
            require(_prompt_tokens(engine, request) == plan['CPU_context']['prompt_tokens'],
                    'actual_prompt_tokens_changed')
            report['phase'] = 'generation'
            trace.record('generation_attempt', request_sha256=plan['request_sha256'])
            report.update(generation_calls=1, candidate_status='unknown',
                          unknown_additional_response_cost=True)
            save()  # Durable intent before the only actual generation call.
            response = engine.complete(copy.deepcopy(request))
            report['generation_returns'] = 1
            if type(response) is dict and type(response.get('usage')) is dict:
                for key in report['returned_response_cost']:
                    value = response['usage'].get(key)
                    if type(value) is int and value >= 0:
                        report['returned_response_cost'][key] = value
            response = _engine_json(response)
            trace.record('generation_returned', response=response)
            report['response_sha256'] = fingerprint(response)
            usage = response.get('usage') if type(response) is dict else None
            require(type(usage) is dict
                    and all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens'))
                    and usage['prompt_tokens'] == plan['CPU_context']['prompt_tokens']
                    and 0 < usage['completion_tokens'] <= 1024
                    and response.get('model') == MODEL_IDS['B3']
                    and response.get('backend') == backend
                    and _engine_json(engine.backend) == backend
                    and type(response.get('request_id')) is int and response['request_id'] == 1
                    and type(engine.calls) is int and engine.calls == 1 and engine.failed is False,
                    'returned_generation_metadata_changed')
            choices = response.get('choices')
            require(type(choices) is list and len(choices) == 1
                    and type(choices[0]) is dict and type(choices[0].get('message')) is dict
                    and choices[0]['message'].get('role') == 'assistant'
                    and type(choices[0]['message'].get('content')) is str, 'invalid_response_envelope')
            report['phase'] = 'parsing'
            report['unknown_additional_response_cost'] = False
            parsed = None
            if response.get('format_error') is None and choices[0].get('finish_reason') == 'stop':
                try:
                    parsed = parse_candidate(choices[0]['message']['content'],
                        source_scenario=plan['source_scenario'], source_hashes=plan['source_trace_sha256'])
                except (ValueError, RecursionError):
                    pass
            report['candidate_status'] = parsed['status'] if parsed else 'invalid_schema'
            if parsed is not None:
                with PrivateTrace(private / 'candidate.jsonl') as candidate_trace:
                    candidate_trace.record('writer_candidate', candidate=parsed)
                report['candidate_sha256'] = digest(private / 'candidate.jsonl')
            report['phase'] = 'engine_exit'
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='unknown', error_type=_exception_type(exc), reason_code='writer_operation_failed')
        if report['generation_calls']:
            report['unknown_additional_response_cost'] = True
    finally:
        if trace is not None:
            try:
                trace.close()
                report['trace_closed'] = True
                report['private_trace_sha256'] = digest(private / 'worker-trace.jsonl')
            except BaseException as exc:
                report.update(status='unknown', error_type=_exception_type(exc),
                              reason_code='private_trace_close_failed')
                report['unknown_additional_response_cost'] = bool(report['generation_calls'])
        report['wall_seconds'] = time.monotonic() - started
        try:
            save()
        except BaseException as exc:
            report.update(status='unknown', error_type=_exception_type(exc),
                          reason_code='public_report_failed', report_persistence_failed=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        report = run(args.plan_sha256, check_only=args.check)
    except BaseException as exc:
        report = {'id': ID, 'status': 'unknown', 'error_type': _exception_type(exc),
                  'reason_code': 'worker_preflight_or_output_failed', 'training_authorized': False}
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report['status'] in {'complete', 'passed_preflight'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
