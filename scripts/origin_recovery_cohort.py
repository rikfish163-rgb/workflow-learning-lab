"""Fixed ID-only selection and read-only verification of the old study record.

The historical conclusion depends on the archived code's durable-start invariant
and retained records. This module does not load task text or inspect trajectories,
authenticate physical release, register a study, or start any resources.
"""

import hashlib
import json
import os
import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION_SHA256 = 'bd29dd438d7444777832f73be4f8c79f71d9056accf9a2833a525ea47cba4bd5'
OLD_DESIGN_SHA256 = '5563a02031871d1ac34a50e3b2c685485b3d1e4e45babc269021c814ac35e46e'
OLD_PLAN_SHA256 = '6c3e33cc37b5af475b7e74e626e4b7dc2a2dc9a12f3e4b7eff527bf62e547873'
OLD_BATCH_SOURCE_SHA256 = 'dd44979b7f151b8be270e0134c6f6ddacd6f8d44cda8e2ccbdaa0a82f9922576'
PREPARED_SHA256 = '7ebc9cebe3422b1750bed40854c246b177095b68d2a8895a522a9c64a6d01a3f'
ARCHIVED_BATCH = Path('/home/hetaisheng/.local/share/workflow-learning-lab/frozen-studies/'
                      'qwen38-origin-recovery-001/root/scripts/run_origin_recovery_batch.py')
PROOF = 'artifacts/origin-unattempted-selection-001.json'
DESIGN = 'artifacts/qwen38-origin-recovery-design-001.json'
PLAN = 'artifacts/qwen38-origin-recovery-plan-001.json'
TERMINAL = 'artifacts/origin-recovery-terminal-audit-CPU-001.json'
BOOTSTRAP = 'artifacts/teacher-bootstrap-plan-v1.json'
COVERAGE = 'artifacts/teacher-coverage-plan-v1.json'
STUDY = 'artifacts/qwen38-origin-recovery-001'
STATUS, REPORT = f'{STUDY}/status.json', f'{STUDY}/report.json'
STAGE = f'{STUDY}/stages/source-0'
DECISION, DIAGNOSTIC = f'{STAGE}/decision.json', f'{STAGE}/diagnostic.json'
MANIFEST, EVENTS = f'{STAGE}/run/manifest.json', f'{STAGE}/run/events.jsonl'
HISTORY_PATHS = frozenset((DESIGN, PLAN, TERMINAL, BOOTSTRAP, COVERAGE, STATUS, REPORT,
                           DECISION, DIAGNOSTIC, MANIFEST, EVENTS))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _open_below(root, relative, *, directory=False):
    """Open a fixed relative path without following any subdirectory symlink."""
    path = Path(relative)
    _require(not path.is_absolute() and all(p not in ('', '.', '..') for p in path.parts),
             'invalid relative evidence path')
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        current = os.open(Path(root).resolve(strict=True), flags | os.O_DIRECTORY)
        try:
            for part in path.parts[:-1]:
                following = os.open(part, flags | os.O_DIRECTORY, dir_fd=current)
                os.close(current)
                current = following
            return os.open(path.parts[-1], flags | (os.O_DIRECTORY if directory else 0),
                           dir_fd=current)
        finally:
            os.close(current)
    except (OSError, ValueError) as exc:
        raise ValueError('missing or unsafe historical input') from exc


def _read_regular(fd):
    with os.fdopen(fd, 'rb') as stream:
        _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'input is not a regular file')
        return stream.read()


def _read(root, path, expected):
    raw = _read_regular(_open_below(root, path))
    _require(hashlib.sha256(raw).hexdigest() == expected, 'historical input SHA-256 mismatch')
    return raw


def _object(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value

    try:
        result = json.loads(raw, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid historical JSON') from exc
    _require(type(result) is dict, 'historical record must be an object')
    return result


def _selection(root):
    proof = _object(_read(root, PROOF, SELECTION_SHA256))
    _require(type(proof.get('schema_version')) is int and proof['schema_version'] == 1
             and proof.get('old_study_id') == 'qwen38-origin-recovery-001'
             and proof.get('old_design_sha256') == OLD_DESIGN_SHA256
             and proof.get('old_plan_sha256') == OLD_PLAN_SHA256
             and proof.get('old_batch_source_sha256') == OLD_BATCH_SOURCE_SHA256
             and proof.get('prepared_manifest_sha256') == PREPARED_SHA256,
             'selection binding differs')
    hashes = proof.get('historical_file_sha256')
    _require(type(hashes) is dict and set(hashes) == HISTORY_PATHS
             and hashes[DESIGN] == OLD_DESIGN_SHA256 and hashes[PLAN] == OLD_PLAN_SHA256
             and all(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value)
                     for value in hashes.values()), 'historical input bindings differ')
    design = _object(_read(root, DESIGN, OLD_DESIGN_SHA256))
    selection = design.get('selection')
    ids = selection.get('task_ids') if type(selection) is dict else None
    _require(design.get('id') == 'qwen38-origin-recovery-001' and type(ids) is list
             and len(ids) == 12 and all(type(x) is str and re.fullmatch(r'[0-9a-f]{7}_2', x)
                                        for x in ids)
             and len(set(ids)) == 12 and proof.get('original_ids') == ids
             and proof.get('excluded_unknown_id') == ids[0]
             and proof.get('selected_ids') == ids[1:], 'fixed cohort differs')
    return proof, ids


def cohort_ids():
    """Return the fixed eleven IDs after verifying the selection and old design."""
    _, ids = _selection(ROOT)
    return list(ids[1:])


def select_prepared(tasks):
    """Select only after the caller's old full-twelve metadata loader succeeds.

    This validates identity, order and train/scenario fields without accessing
    instructions. It does not itself authenticate a prepared metadata file.
    """
    _, ids = _selection(ROOT)
    _require(type(tasks) in (list, tuple) and len(tasks) == 12, 'full twelve tasks required')
    for task, task_id in zip(tasks, ids, strict=True):
        _require(getattr(task, 'task_id', None) == task_id
                 and type(getattr(task, 'task_id', None)) is str
                 and getattr(task, 'scenario_id', None) == task_id.rsplit('_', 1)[0]
                 and getattr(task, 'split', None) == 'train', 'task identity/order differs')
    return list(tasks[1:])


def _inventory(root, path, expected):
    fd = _open_below(root, path, directory=True)
    try:
        with os.scandir(fd) as entries:
            found = list(entries)
        _require({entry.name for entry in found} == {expected}
                 and all(entry.is_dir(follow_symlinks=False) for entry in found),
                 'unexpected or unsafe stage/child inventory')
    finally:
        os.close(fd)


def _verify_events(raw, decision, manifest, excluded):
    _require(raw.endswith(b'\n'), 'truncated root journal')
    events = [_object(line) for line in raw.splitlines()]
    names = ['batch_started', 'idle_started', 'idle_returned', 'attempt_started',
             'container_intent', 'world_ready']
    _require(len(events) == len(names), 'unexpected root journal event count')
    for index, (event, name) in enumerate(zip(events, names, strict=True)):
        _require(type(event.get('event_index')) is int and event['event_index'] == index
                 and event.get('event') == name, 'root journal sequence differs')
        if index:
            _require(event.get('task_id') == excluded and event.get('kind') == 'source',
                     'unexpected attempt slot')
    _require(events[0].get('decision') == decision
             and events[0].get('client_identity') == manifest.get('client_identity')
             and events[3].get('directory') == f'attempts/00-source-{excluded}',
             'root journal binding differs')


def verify_history(root=ROOT):
    """Rehash and cross-check the bounded historical record, without task content."""
    proof, ids = _selection(root)
    raw = {name: _read(root, name, sha) for name, sha in proof['historical_file_sha256'].items()}
    records = {name: _object(value) for name, value in raw.items() if name != EVENTS}
    plan, design = records[PLAN], records[DESIGN]
    _require(plan.get('id') == 'qwen38-origin-recovery-001'
             and plan.get('design_sha256') == OLD_DESIGN_SHA256
             and plan.get('automatic_retry') is False
             and plan.get('selection') == design.get('selection')
             and type(plan.get('source_sha256')) is dict
             and plan['source_sha256'].get('scripts/run_origin_recovery_batch.py')
             == OLD_BATCH_SOURCE_SHA256, 'old plan/source binding differs')
    try:
        fd = os.open(ARCHIVED_BATCH, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError('missing archived batch source') from exc
    _require(hashlib.sha256(_read_regular(fd)).hexdigest() == OLD_BATCH_SOURCE_SHA256,
             'archived batch source differs')

    decision = {'action': 'launch_source_batch', 'reason': '', 'batch_index': 0,
                'attempts': [{'kind': 'source', 'task_id': task} for task in ids[:6]]}
    _require(records[DECISION] == decision and type(records[DECISION].get('batch_index')) is int,
             'original source decision differs')
    manifest, diagnostic = records[MANIFEST], records[DIAGNOSTIC]
    slot = [{'task_id': ids[0], 'kind': 'source'}]
    _require(manifest.get('status') == 'running' and manifest.get('decision') == decision
             and manifest.get('automatic_retry') is False
             and manifest.get('prepared_manifest_sha256') == PREPARED_SHA256
             and diagnostic.get('valid') is False
             and diagnostic.get('attempted_slots_verified') is True
             and diagnostic.get('attempted_slots') == slot
             and diagnostic.get('run_files_sha256') == {
                 'events.jsonl': proof['historical_file_sha256'][EVENTS],
                 'manifest.json': proof['historical_file_sha256'][MANIFEST]},
             'manifest/diagnostic binding differs')
    _verify_events(raw[EVENTS], decision, manifest, ids[0])
    _inventory(root, f'{STUDY}/stages', 'source-0')
    _inventory(root, f'{STAGE}/run/attempts', f'00-source-{ids[0]}')

    status, report, terminal = records[STATUS], records[REPORT], records[TERMINAL]
    _require(status.get('phase') == status.get('summary_phase') == 'stopped_unknown'
             and status.get('plan_sha256') == OLD_PLAN_SHA256
             and status.get('automatic_retry') is False and status.get('current_stage') == 'source-0'
             and status.get('current_decision') == decision
             and report.get('phase') == 'stopped_unknown' and report.get('automatic_retry') is False
             and report.get('training_authorized') is False
             and report.get('unaccepted_attempted_slots') == slot, 'old terminal status differs')
    stages = report.get('stages')
    _require(type(stages) is list and len(stages) == 1 and type(stages[0]) is dict
             and stages[0].get('stage') == 'source-0' and stages[0].get('accepted') is False
             and stages[0].get('decision') == decision, 'old report stage differs')
    summary = report.get('summary')
    classes = {'help': 0, 'harm': 0, 'tie_success': 0, 'tie_failure': 0,
               'unknown': 1, 'ineligible': 0, 'unattempted': 11}
    pairs = [{'task_id': item, 'pair_class': 'unknown' if i == 0 else 'unattempted'}
             for i, item in enumerate(ids)]
    _require(type(summary) is dict and type(summary.get('fixed_source_denominator')) is int
             and summary['fixed_source_denominator'] == 12 and summary.get('pairs') == pairs
             and summary.get('pair_classes') == classes
             and all(type(v) is int for v in summary['pair_classes'].values()),
             'old outcome accounting differs')
    counts = {'source_denominator': 12, 'study_unknown': 1, 'study_unattempted': 11,
              'source_batch_scheduled': 6, 'source_batch_started': 1,
              'source_batch_unattempted': 5, 'accepted_sources': 0}
    _require(terminal.get('plan_sha256') == OLD_PLAN_SHA256 and terminal.get('no_retry') is True
             and all(type(terminal.get(k)) is int and terminal[k] == v for k, v in counts.items()),
             'independent terminal audit differs')

    teachers = [records[BOOTSTRAP], records[COVERAGE]]
    train, calibration = [], []
    for record, count in zip(teachers, (4, 8), strict=True):
        original, variant3 = record.get('teacher_task_ids'), record.get('calibration_task_ids')
        _require(type(original) is list and type(variant3) is list and len(original) == count
                 and all(type(x) is str for x in original)
                 and variant3 == [x[:-1] + '3' for x in original],
                 'historical teacher-plan IDs differ')
        train.extend(original)
        calibration.extend(variant3)
    _require(len(set(train)) == 12 and set(train) == set(ids)
             and len(set(calibration)) == 12, 'historical exposure coverage differs')
    return {'verified': True, 'old_study_id': 'qwen38-origin-recovery-001',
            'old_plan_sha256': OLD_PLAN_SHA256, 'selection_sha256': SELECTION_SHA256,
            'source_denominator': 12, 'unknown_id': ids[0], 'unattempted_ids': list(ids[1:]),
            'root_events_count': 6, 'historical_resources_verified': False,
            'scope': 'Retained study records only; no actor outside the recorded study or release is certified.'}
