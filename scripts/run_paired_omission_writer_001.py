"""Run one registered writer and at most one reviewer over two real recoveries.

The input is a typed projection of fixed, authenticated public evidence. The
code and lease helpers below are reused at their verified source digests.
Importing this entry point creates no plan, request, world, or GPU job.
"""

import argparse
import copy
import hashlib
import importlib.util
import os
import shutil
import socket
from functools import cache
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from scripts import codeact_within_writer_registration_001 as writer_registration
from scripts import native_experience_content_001 as content
from scripts import native_experience_writer_001 as writer
from scripts import native_source_registration as public
from scripts import paired_omission_writer_evidence_001 as evidence
from scripts.coverage_contract_gpu_queue import _public_service
from scripts.native_appworld_worker_client import _encode
from scripts.queue_recovery_evidence_lesson import verify_public_frozen_plan

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'scripts.run_paired_omission_writer_001'
ID = 'qwen38-paired-omission-writer-001'
PLAN = f'artifacts/{ID}-plan.json'
PORT = 18996
PRIVATE_STATE_ROOT = public.PRIVATE_STATE_ROOT
PRIVATE_RUN_ROOT = PRIVATE_STATE_ROOT + '/paired-omission-writer-001'
TMPDIR = public.TMPDIR
PYTHON = public.PYTHON
MODEL_ID = writer_registration.MODEL_ID
public_read, _public_hash = public.public_read, public._public_hash
SOURCE_SCENARIOS = ('b7a9ee9', '2a163ab')
SOURCE_PLAN = 'artifacts/paired-omission-evidence-plan-001.json'
SOURCE_PLAN_SHA = 'ae10ca458161d62c14e97623623418942d1448161f53845be0fffe60395405f5'
SOURCE_GATE = 'paired_verified_write_omission_v1'
ORIGINAL_PLANS = {
    'artifacts/qwen38-write-partial-plan-001.json':
        '3ef0eab880dc3695af260761e879e00b63213c09b5d0e189c0c2694193abdcc6',
    'artifacts/qwen38-named-write-partial-plan-001.json':
        '6941a421f6c1b785c4e68cefa018935772dd1fc6774081d50a6169c865e6d110',
}
ERROR = 'paired_omission_writer_001_invalid'
BASE = {
    'scripts/codeact_within_writer_gpu_queue_001.py':
        'd856646b4698a08c19b999b8c90546472327667f5d01143da2e9599fbc4dc9c1',
    'scripts/run_codeact_within_writer_001.py':
        '59ef0419f482e6518215b7d8793a5a73013b94e3f85278070dade310a37720a3',
    'scripts/run_codeact_within_writer_cases_001.py':
        '859e79b26499e28d16d200468c76be902c17ac271332116de8685c479bcc8f01',
    'scripts/native_experience_content_001.py':
        'd08b48932b5e82368b6674501b918881bfa710e2e189e1d798ad2abe8ac56334',
}
REQUIRED = frozenset({
    *BASE, 'scripts/run_paired_omission_writer_001.py',
    'scripts/paired_omission_writer_evidence_001.py',
})


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def _sha(value):
    return public._sha(value)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepared_input(root):
    typed = evidence.prepare_evidence({
        name: public.public_read(root, 'artifacts/' + name) for name in evidence.PROOF_SHA256
    })
    return dict(contexts=[dict(lineage=dict(source_index=0, source_scenario='paired_omission'),
                               evidence=typed)])


def input_binding(prepared):
    return dict(preparation_manifest_sha256=_digest(_encode(prepared)),
                origin_ids=list(SOURCE_SCENARIOS),
                selection=dict(typed_recovery_cases=2, writer_corpora=1))


def _inputs(prepared, registered):
    _require(type(prepared) is dict and set(prepared) == {'contexts'}
             and type(prepared['contexts']) is list and len(prepared['contexts']) == 1)
    ctx = prepared['contexts'][0]
    _require(type(ctx) is dict and set(ctx) == {'lineage', 'evidence'}
             and _encode(ctx['lineage']) == _encode(dict(source_index=0, source_scenario='paired_omission'))
             and _encode(registered) == _encode(input_binding(prepared)))
    evidence.validate_evidence(ctx['evidence'])
    return copy.deepcopy(prepared), copy.deepcopy(registered)


def fixed_metadata():
    return dict(id=ID, status='registered', condition='typed_two_recovery_cases',
                model_arm='B3', model_id=MODEL_ID, port=PORT, python=PYTHON,
                private_run_root=PRIVATE_RUN_ROOT, tmp_dir=TMPDIR,
                service_plan=writer_registration.SERVICE_PLAN,
                service_plan_sha256=writer_registration.SERVICE_SHA,
                gpu_seconds=7200, total_gpu_seconds=7200, client_seconds=6000,
                model_start_seconds=900, cleanup_seconds=180, admission_wait_seconds=86400,
                automatic_retry=False, writer_authorized=True, utility_authorized=False,
                training_authorized=False, dev_or_test_authorized=False,
                writer=dict(corpora=1, case_count=2, max_writer_calls=1, max_review_calls=1,
                            max_input_tokens=15360, writer_tokens=128, review_tokens=256,
                            writer_seed=17, review_seed=117, temperature=0.7,
                            per_call_seconds=240, request_seconds=120))


def check_predecessor(root, plan):
    inputs = source_bindings()
    _require(all(plan['input_sha256'].get(name) == sha
                 and _digest(public_read(root, name)) == sha for name, sha in inputs.items()))
    prepared = prepared_input(root)
    digest = _digest(_encode(prepared))
    _require(_encode(plan['registered_inputs']) == _encode(input_binding(prepared))
             and digest == plan['private_input_sha256']
             and _encode(plan.get('source')) == _encode(dict(
                 plan_sha256=SOURCE_PLAN_SHA, result_sha256=digest, source_gate_kind=SOURCE_GATE)))
    return True


def source_bindings():
    return {**{'artifacts/' + name: sha for name, sha in evidence.PROOF_SHA256.items()},
            **ORIGINAL_PLANS, SOURCE_PLAN: SOURCE_PLAN_SHA}


def validate_metadata(root, plan):
    _require(type(plan) is dict and all(key in plan and _encode(plan[key]) == _encode(value)
                                       for key, value in fixed_metadata().items())
             and type(plan.get('source_sha256')) is dict and REQUIRED <= plan['source_sha256'].keys()
             and all(plan['source_sha256'].get(name) == sha for name, sha in BASE.items())
             and type(plan.get('input_sha256')) is dict
             and type(plan.get('runtime_sha256')) is dict and bool(plan['runtime_sha256'])
             and all(type(name) is str and Path(name).is_absolute() and '..' not in Path(name).parts
                     and _sha(sha) for name, sha in plan['runtime_sha256'].items()))
    _public_service(root, plan, plan['source_sha256'], plan['input_sha256'])
    check_predecessor(root, plan)
    return plan


def load_registration(root, sha, *, private=False):
    _require(private is False and _sha(sha))
    return validate_metadata(root, verify_public_frozen_plan(root, Path(root) / PLAN, sha))


def private_preflight(root, sha):
    _require(socket.gethostname() == 'amax' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
             and os.environ.get('TMPDIR') == TMPDIR)
    plan = load_registration(root, sha)
    for name in (PRIVATE_STATE_ROOT, PRIVATE_RUN_ROOT, TMPDIR):
        public._private_directory(Path(name))
        _require(shutil.disk_usage(name).free >= 5 * 1024**3)
    for name, expected in plan['runtime_sha256'].items():
        _require(public._absolute_hash(name) == expected)
    prepared = prepared_input(root)
    _inputs(prepared, plan['registered_inputs'])
    return plan, prepared


def _load(relative, label, changes=(), inject=None):
    raw = public.public_read(ROOT, relative)
    _require(_digest(raw) == BASE[relative])
    source = raw.decode('utf-8')
    for old, new in changes:
        _require(source.count(old) == 1)
        source = source.replace(old, new)
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location('scripts._paired_omission_' + label, path)
    _require(spec is not None and spec.loader is not None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(inject or {})
    # Only hash-verified, human-authored repository code is compiled.
    exec(compile(source, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


@cache
def _queue():
    return _load('scripts/codeact_within_writer_gpu_queue_001.py', 'queue', (
        ("'qwen38-codeact-within-writer-001'", repr(ID)),
        ("'artifacts/qwen38-codeact-within-writer-plan-001.json'", repr(PLAN)),
        ('18993, 1', f'{PORT}, 1'),
        ("'codeact-within-writer-operations-001.lock'", "'paired-omission-writer-operations-001.lock'"),
        ("'scripts.codeact_within_writer_gpu_queue_001'", repr(MODULE)),
        ("'scripts.codeact_within_writer_registration_001'", repr(MODULE)),
    ))


def _typed(ctx):
    return evidence.validate_evidence(ctx['evidence'])


def _write_once(ctx, model, *, seed, record):
    return writer.write_once(evidence.build_writer_messages(_typed(ctx)), model,
                             seed=seed, record=record)


@cache
def _review_core():
    module = _load('scripts/native_experience_content_001.py', 'review')

    def typed_history(messages):
        _require(len(messages) == 2 and messages[0] == {
            'role': 'system', 'content': evidence.WRITER_SYSTEM})
        payload = content._decode(messages[1]['content'])
        _require(messages[1]['role'] == 'user' and set(payload) == {'history'})
        return evidence.validate_evidence(payload['history'])

    module.WRITER_SYSTEM = evidence.WRITER_SYSTEM
    module._REVIEW_SYSTEM = evidence.REVIEW_SYSTEM
    module._history, module.parse_review = typed_history, evidence.parse_review
    return module


def _review_once(ctx, candidate, model, *, seed, record):
    messages = evidence.build_review_messages(_typed(ctx), candidate)
    return _review_core().review_once(messages, model, seed=seed, record=record)


@cache
def _cases():
    exp = SimpleNamespace(
        build_writer_messages=lambda ctx: evidence.build_writer_messages(_typed(ctx)),
        build_review_messages=lambda ctx, candidate: evidence.build_review_messages(_typed(ctx), candidate),
        parse_candidate=writer.parse_candidate, _sha=_sha,
        write_once=_write_once, review_once=_review_once,
        screen=lambda ctx, candidate: evidence.screen_candidate(_typed(ctx), candidate),
    )
    module = _load('scripts/run_codeact_within_writer_cases_001.py', 'cases')
    module.experience, module._inputs = exp, _inputs
    module.ID = 'paired-omission-writer-cases-001'
    return module


@cache
def _runner():
    module = _load('scripts/run_codeact_within_writer_001.py', 'runner', (
        ("'scripts.run_codeact_within_writer_001'", repr(MODULE)),
        ("'wll-native-source-codeact-writer-'", "'wll-native-source-paired-omission-writer-'"),
    ))
    module.registration = import_module(MODULE)
    module.cases, module.experience = _cases(), _cases().experience
    module.ID, module.PLAN, module._queue = ID, PLAN, _queue

    class Runtime(module._Runtime):
        def check_input(self, sha, expected):
            fresh = load_registration(self.root, sha)
            _require(_encode(fresh) == _encode(self.plan))
            prepared = prepared_input(self.root)
            _require(_encode(prepared) == _encode(expected)
                     and _digest(_encode(prepared)) == fresh['private_input_sha256'])
            return prepared

    module._Runtime = Runtime
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--run', action='store_true')
    operation.add_argument('--launch', action='store_true')
    operation.add_argument('--serve', action='store_true')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--owner')
    args = parser.parse_args(argv)
    if args.serve:
        _queue().serve(_queue().make_job(args.batch, args.owner, args.plan_sha256))
        return 0
    _require(args.batch is None and args.owner is None)
    return _runner().main(['--plan-sha256', args.plan_sha256,
                           '--run' if args.run else '--launch'])


if __name__ == '__main__':
    raise SystemExit(main())
