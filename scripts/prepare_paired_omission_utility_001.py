"""Authenticate writer 003 and freeze its one reviewed paired-omission lesson.

This CPU-only preparation performs no model, world, queue mutation, utility or
training operation. It runs locally on amax and retains all content privately.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from scripts import native_experience_content_001 as content
from scripts import native_source_episode as episode
from scripts import run_paired_omission_writer_003 as writer
from scripts.native_appworld_worker_client import _decode, _encode, _exclusive, _same
from wll.protocol import fingerprint

ROOT = Path(__file__).resolve().parents[1]
BASE = 'scripts/prepare_codeact_within_utility_001.py'
BASE_SHA256 = 'fc3bdb15ad7e72453e2c03493952f5f4175c25c62e8667c3f5c18cc38e674027'
QUALIFICATION = 'artifacts/paired-omission-utility-qualification-001.json'
QUALIFICATION_ID = 'paired-omission-utility-qualification-001'
MANIFEST = 'review-manifest.json'
MANIFEST_ID = 'paired-omission-review-manifest-003'
SOURCE_SCENARIOS = ['2a163ab', 'b7a9ee9']
BUNDLE_PREFIX = 'paired-omission-experience-001:'
ERROR = 'paired_omission_utility_preparation_invalid'
MAX_BYTES = 64_000_000


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _save_new(path, value):
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with _exclusive(directory, path.name) as stream:
            raw = _encode(value) + b'\n'
            remaining = memoryview(raw)
            while remaining:
                written = stream.write(remaining)
                _require(type(written) is int and written > 0)
                remaining = remaining[written:]
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
        return _hash(raw)
    finally:
        os.close(directory)


def _render_rows(rows):
    _require(type(rows) is list and len(rows) == 1)
    row = rows[0]
    fields = {'source_index', 'source_scenario', 'candidate', 'insertion', 'candidate_sha256',
              'insertion_sha256', 'review_result_sha256', 'status', 'accepted', 'content_certified',
              'training_authorized'}
    _require(type(row) is dict and set(row) == fields and type(row['source_index']) is int
             and row['source_index'] == 0 and row['source_scenario'] == 'paired_omission'
             and row['status'] == 'plausible_pending_utility'
             and row['accepted'] is row['content_certified'] is row['training_authorized'] is False
             and writer._sha(row['review_result_sha256']))
    text = content.insertion(row['candidate'])
    _require(row['insertion'] == text and row['insertion_sha256'] == _hash(text.encode('utf-8'))
             and row['candidate_sha256'] == _hash(content._json(row['candidate']).encode('utf-8')))
    return text


def _reviewed_row(case, rows):
    _render_rows(rows)
    receipts = case['private']['receipts']
    _require(type(receipts) is list and len(receipts) == 1)
    review = receipts[0]['review']['result']
    _require(review['status'] == 'plausible_pending_utility'
        and _same(review['verdict'], dict(verdict='plausible_pending_utility',
                  reason_codes=['visible_conditional_support'], evidence_indices=[0, 1])))
    return copy.deepcopy(rows[0])


def _validators(*, raw=None):
    """Isolate selected pinned human-authored validators; no model code is run."""
    raw = writer.public_read(ROOT, BASE) if raw is None else raw
    _require(type(raw) is bytes and _hash(raw) == BASE_SHA256)
    tree = ast.parse(raw.decode('utf-8'))
    names = {'_require', '_hash', '_finite', '_Replay', '_verify_stage', '_verify_cases',
             '_verify_release', '_verify_launch_command', '_verify_journal', '_terminal', '_queue_evidence'}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    _require({node.name for node in nodes} == names and len(nodes) == len(names))
    constants = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {'_TERMINAL_CODE', '_QUEUE_CODE'}):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    _require(set(constants) == {'_TERMINAL_CODE', '_QUEUE_CODE'})

    class LocalOnly(ast.NodeTransformer):
        current = None
        calls = 0
        patterns = 0

        def visit_FunctionDef(self, node):
            previous, self.current = self.current, node.name
            self.generic_visit(node)
            self.current = previous
            return node

        def visit_Constant(self, node):
            if self.current == '_verify_journal' and node.value == r'wll-native-source-codeact-writer-[0-9a-f]{32}\.service':
                self.patterns += 1
                node.value = r'wll-native-source-paired-omission-writer-003-[0-9a-f]{32}\.service'
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if (self.current in {'_terminal', '_queue_evidence'} and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess'
                    and node.func.attr == 'run'):
                expression = ("['/usr/bin/python3', '-c', _TERMINAL_CODE, *args]" if self.current == '_terminal'
                              else "['/usr/bin/python3', '-c', _QUEUE_CODE, job['id']]")
                _require(len(node.args) == 1)
                node.args[0] = ast.parse(expression, mode='eval').body
                self.calls += 1
            return node

    transform = LocalOnly()
    isolated = ast.fix_missing_locations(transform.visit(ast.Module(body=nodes, type_ignores=[])))
    _require(transform.calls == 2 and transform.patterns == 1)
    cases, controller, queue = writer._cases(), writer._runner(), writer._queue()
    scope = dict(__name__='scripts._paired_omission_review_validators', ERROR=ERROR, ROOT=ROOT,
        SERVER_ROOT=ROOT, writer=writer, controller=controller, queue=queue, cases=cases,
        experience=cases.experience, utility=SimpleNamespace(render_bundle=_render_rows), episode=episode,
        copy=copy, hashlib=hashlib, math=math, re=re, shlex=shlex, subprocess=subprocess,
        Path=Path, FunctionType=FunctionType, fingerprint=fingerprint, _decode=_decode, _encode=_encode, _same=_same,
        **constants)
    # The AST is selected only after checking the complete fixed source digest.
    exec(compile(isolated, str(ROOT / BASE), 'exec', dont_inherit=True), scope)
    return SimpleNamespace(**scope)


def _reader():
    scope = dict(episode.__dict__)
    scope['PRIVATE_RUN_ROOT'] = Path(writer.PRIVATE_STATE_ROOT)
    for name in ('_private_directory', '_read_private'):
        fn = getattr(episode, name)
        scope[name] = FunctionType(fn.__code__, scope, name, fn.__defaults__, fn.__closure__)
    return scope['_read_private']


def _environment(root):
    _require(Path(root) == ROOT and ROOT == writer.ROOT and socket.gethostname() == 'amax'
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('TMPDIR') == writer.TMPDIR
             and writer.ID == 'qwen38-paired-omission-writer-003'
             and writer.PLAN == 'artifacts/qwen38-paired-omission-writer-003-plan.json'
             and writer.PRIVATE_RUN_ROOT == writer.PRIVATE_STATE_ROOT + '/paired-omission-writer-003')


def _stable_queue_proof(validators, job, release, fence, service):
    value = validators._queue_evidence(job, release, fence, service)
    keys = {'result_sha256', 'cancellation_sha256', 'service_ready_sha256', 'job_sha256', 'queue_archived'}
    _require(type(value) is dict and set(value) == keys and type(value['queue_archived']) is bool
             and all(writer._sha(value[key]) for key in keys - {'queue_archived'}))
    # A later valid utility enqueue archives this job; that does not change its evidence.
    return {key: value[key] for key in sorted(keys - {'queue_archived'})}


def _reconstruct(root, plan_sha, result_sha):
    _environment(root)
    _require(writer._sha(plan_sha) and writer._sha(result_sha))
    validators = _validators()
    plan = writer.load_registration(root, plan_sha, private=False)
    _require(_same(writer.load_registration(writer._queue().REMOTE, plan_sha, private=False), plan))
    reader, snapshots = _reader(), {}

    def read(name, expected=None):
        path = Path(writer.PRIVATE_RUN_ROOT) / name
        raw = reader(path, maximum=MAX_BYTES)
        sha = _hash(raw)
        _require(expected is None or sha == expected)
        snapshots[str(path)] = sha
        return raw

    report = _decode(read('study-result.json', result_sha))
    _require(type(report) is dict and report.get('status') == 'complete'
        and report.get('whole_cost_known') is True and report.get('plan_sha256') == plan_sha
        and all(writer._sha(report.get(key)) for key in
                ('case_result_sha256', 'pending_bundle_sha256', 'GPU_release_sha256')))
    launch = _decode(read('parent-launch-intent.json'))
    intent = _decode(read('study-intent.json'))
    # Check primitive values before passing them to the local process inspector.
    _require(type(launch) is dict and type(launch.get('unit')) is str
        and re.fullmatch(r'wll-native-source-paired-omission-writer-003-[0-9a-f]{32}\.service', launch['unit'])
        and type(intent) is dict and type(intent.get('pid')) is int and intent['pid'] > 0
        and type(intent.get('start_ticks')) is str and intent['start_ticks'].isdigit()
        and type(intent.get('boot_id')) is str and re.fullmatch('[0-9a-f-]{36}', intent['boot_id']))
    terminal = validators._terminal(launch, intent)
    journal_raw = read('study-journal.jsonl')
    _require(journal_raw and journal_raw.endswith(b'\n'))
    journal = [_decode(line) for line in journal_raw.splitlines()]
    case = _decode(read('writer-case-result.json', report['case_result_sha256']))
    pending = _decode(read('pending-bundle.json', report['pending_bundle_sha256']))
    release = _decode(read('GPU-release.json', report['GPU_release_sha256']))
    fresh_plan, prepared = writer.private_preflight(root, plan_sha)
    _require(_same(fresh_plan, plan) and _same(prepared, case['private']['source_private'])
        and _hash(_encode(prepared)) == plan['private_input_sha256']
        and _same(plan['source'], dict(plan_sha256=writer.SOURCE_PLAN_SHA,
             result_sha256=plan['private_input_sha256'], source_gate_kind=writer.SOURCE_GATE))
        and plan['registered_inputs']['origin_ids'] == ['b7a9ee9', '2a163ab'])
    rows, job, fence, service = validators._verify_journal(
        plan, plan_sha, intent, launch, journal, report, case, pending, release)
    row = _reviewed_row(case, rows)
    queue_proof = _stable_queue_proof(validators, job, release, fence, service)
    text = _render_rows(rows)
    bundle_id = BUNDLE_PREFIX + _hash(text.encode('utf-8'))
    for name, sha in snapshots.items():
        _require(_hash(reader(Path(name), maximum=MAX_BYTES)) == sha)
    rechecked_plan, rechecked_prepared = writer.private_preflight(root, plan_sha)
    _require(_same(rechecked_plan, plan) and _same(rechecked_prepared, prepared)
        and _same(writer.load_registration(writer._queue().REMOTE, plan_sha, private=False), plan)
        and _same(validators._terminal(launch, intent), terminal)
        and _same(_stable_queue_proof(validators, job, release, fence, service), queue_proof)
        and _hash(writer.public_read(root, BASE)) == BASE_SHA256)
    manifest = dict(id=MANIFEST_ID, status='plausible_pending_utility',
        source_plan_sha256=plan['source']['plan_sha256'], source_result_sha256=plan['source']['result_sha256'],
        source_gate_kind=plan['source']['source_gate_kind'], writer_plan_sha256=plan_sha,
        writer_result_sha256=result_sha, source_scenario_ids=list(SOURCE_SCENARIOS), experience_bundle_id=bundle_id,
        candidate_sha256=row['candidate_sha256'], insertion_sha256=row['insertion_sha256'],
        review_result_sha256=row['review_result_sha256'], private_files_sha256=snapshots,
        terminal=terminal, queue_release=queue_proof, validator_source_sha256=BASE_SHA256)
    return manifest, text


def _bundle(manifest, manifest_sha, text, manifest_path):
    _require(type(manifest) is dict and manifest['status'] == 'plausible_pending_utility'
        and manifest['source_plan_sha256'] == writer.SOURCE_PLAN_SHA
        and manifest['source_gate_kind'] == writer.SOURCE_GATE
        and manifest['source_scenario_ids'] == SOURCE_SCENARIOS and writer._sha(manifest_sha)
        and type(text) is str and bool(text.strip()) and len(text.encode('utf-8')) <= 65536
        and manifest['experience_bundle_id'] == BUNDLE_PREFIX + _hash(text.encode('utf-8'))
        and type(manifest['private_files_sha256']) is dict and type(manifest_path) is str)
    files = copy.deepcopy(manifest['private_files_sha256'])
    _require(manifest_path not in files)
    files[manifest_path] = manifest_sha
    return dict(private=True, status=manifest['status'], source_plan_sha256=manifest['source_plan_sha256'],
        source_result_sha256=manifest['source_result_sha256'], writer_plan_sha256=manifest['writer_plan_sha256'],
        writer_result_sha256=manifest['writer_result_sha256'], review_manifest_sha256=manifest_sha,
        experience_bundle_id=manifest['experience_bundle_id'], source_scenario_ids=list(SOURCE_SCENARIOS),
        insertion_text=text, private_files_sha256=files, source_gate_kind=manifest['source_gate_kind'])


def load_reviewed_bundle(root, writer_plan_sha, writer_result_sha, review_manifest_sha):
    """Reconstruct all evidence and return the utility loader's exact private schema."""
    try:
        _require(writer._sha(review_manifest_sha))
        manifest, text = _reconstruct(root, writer_plan_sha, writer_result_sha)
        path = Path(writer.PRIVATE_RUN_ROOT) / MANIFEST
        reader = _reader()
        raw = reader(path, maximum=MAX_BYTES)
        _require(_hash(raw) == review_manifest_sha and _same(_decode(raw), manifest)
                 and _hash(reader(path, maximum=MAX_BYTES)) == review_manifest_sha)
        return _bundle(manifest, review_manifest_sha, text, str(path))
    except Exception:
        raise ValueError(ERROR) from None


def prepare(writer_plan_sha, writer_result_sha):
    """Write one exclusive private review manifest and its public qualification."""
    try:
        manifest, text = _reconstruct(ROOT, writer_plan_sha, writer_result_sha)
        path, proof_path = Path(writer.PRIVATE_RUN_ROOT) / MANIFEST, ROOT / QUALIFICATION
        _require(not path.exists() and not path.is_symlink()
                 and not proof_path.exists() and not proof_path.is_symlink())
        manifest_sha = _save_new(path, manifest)
        bundle = _bundle(manifest, manifest_sha, text, str(path))
        proof = {key: bundle[key] for key in ('status', 'source_plan_sha256', 'source_result_sha256',
            'writer_plan_sha256', 'writer_result_sha256', 'review_manifest_sha256', 'experience_bundle_id',
            'source_scenario_ids', 'source_gate_kind')}
        proof['id'] = QUALIFICATION_ID
        _save_new(proof_path, proof)
        return proof
    except Exception:
        raise ValueError(ERROR) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--writer-plan-sha256', required=True)
    parser.add_argument('--writer-result-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        proof = prepare(args.writer_plan_sha256, args.writer_result_sha256)
        print(json.dumps({key: proof[key] for key in ('status', 'writer_plan_sha256',
                                                     'writer_result_sha256', 'review_manifest_sha256')}))
        return 0
    except Exception:
        sys.stderr.write(ERROR + '\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
