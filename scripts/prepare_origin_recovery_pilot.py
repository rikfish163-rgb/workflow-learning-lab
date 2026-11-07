#!/usr/bin/env python3
"""Read frozen train IDs only and print a draft recovery-pilot preparation.

This does not load task descriptions, start an environment, register a study,
or select IDs from results. The command accepts no task-selection arguments.
"""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = 'artifacts/partitions-v1/manifest.json'
OFFICIAL_TRAIN = Path(
    '/media/hetaisheng/044A81D94A81C83E/wll-runtime/data/datasets/train.txt'
)
SOURCE_PATHS = {
    'manifest': ROOT / MANIFEST_PATH,
    'origin': ROOT / 'artifacts/partitions-v1/origin.txt',
    'utility': ROOT / 'artifacts/partitions-v1/utility.txt',
    'distill': ROOT / 'artifacts/partitions-v1/distill.txt',
    'official_train': OFFICIAL_TRAIN,
    'source_plan': ROOT / 'artifacts/qwen38-source-capability-plan-001.json',
    'teacher_bootstrap': ROOT / 'artifacts/teacher-bootstrap-plan-v1.json',
    'teacher_coverage': ROOT / 'artifacts/teacher-coverage-plan-v1.json',
}
FROZEN_SHA256 = {
    'manifest': '3732b02da71c4031ba3a831c727d1956bb43872552841db275a86900e0dcf4a6',
    'origin': 'c8d69221c9ba3967b08fedd1e7c6dff31e0c3817101c24638c3ffeb1dc062c46',
    'utility': 'd9b24521f14ad512f36328f9561789db2424bc0471a215045a889fc866ce6580',
    'distill': '04934c1e3a3295c6613b84e91ddbc3af9acbd2bbfc953bb59a59531526c4c068',
    'official_train': '93d9fe71e7a2e3b7529803d4a20b604f4ebf5ae806f321081140238068189d37',
    'source_plan': '07e1059d459d6da2b74943d246975ab6da543b1fb16b35f3915f1669428a2e55',
    'teacher_bootstrap': '15e9d0f6a406543b110e504b2a06004feb0aff8d42bcf8c8edc1bff9e18efea5',
    'teacher_coverage': '8c53e976ce36e68f6b3e358b379728d8a0a5c83e0e32127710b65e2e7f100b88',
}
TASK_COUNTS = {'origin': 36, 'utility': 27, 'distill': 27}
SCENARIO_COUNTS = {'origin': 12, 'utility': 9, 'distill': 9}
TASK_ID = re.compile(r'[0-9a-f]{7}_[123]\Z')


def _read_frozen(path, expected):
    """Read one regular, non-symlink file; reject changed bytes."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f'not a regular frozen input: {path}')
            raw = stream.read()
    except OSError as exc:
        raise ValueError(f'missing or unsafe frozen input: {path}') from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'frozen input hash mismatch: {path}')
    return raw


def _json_object(raw, name):
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid JSON input: {name}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {name}')
    return value


def _ids(value, name):
    if not isinstance(value, list) or any(
        not isinstance(task, str) or TASK_ID.fullmatch(task) is None for task in value
    ):
        raise ValueError(f'invalid train task IDs: {name}')
    if len(value) != len(set(value)):
        raise ValueError(f'duplicate train task ID: {name}')
    return value


def _lines(raw, name):
    try:
        return _ids(raw.decode('utf-8').splitlines(), name)
    except UnicodeDecodeError as exc:
        raise ValueError(f'invalid UTF-8 input: {name}') from exc


def _prepare_sources(raw, frozen_sha256):
    """Validate already-read bytes against trusted hashes and their bindings.

    The command supplies module-level frozen hashes. Separate in-memory hashes
    are accepted here only to make synthetic test fixtures possible.
    """
    required = set(FROZEN_SHA256)
    if set(raw) != required or set(frozen_sha256) != required:
        raise ValueError('missing or unexpected frozen inputs')
    actual = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
    if actual != frozen_sha256:
        raise ValueError('frozen input hash mismatch')

    manifest = _json_object(raw['manifest'], 'manifest')
    plan = _json_object(raw['source_plan'], 'source plan')
    bootstrap = _json_object(raw['teacher_bootstrap'], 'teacher bootstrap plan')
    coverage = _json_object(raw['teacher_coverage'], 'teacher coverage plan')
    train = _lines(raw['official_train'], 'official train')
    if manifest.get('source_sha256') != actual['official_train']:
        raise ValueError('manifest is not bound to official train')
    if (plan.get('partition') != MANIFEST_PATH
            or plan.get('partition_sha256') != actual['manifest']
            or not isinstance(plan.get('input_sha256'), dict)
            or plan['input_sha256'].get(MANIFEST_PATH) != actual['manifest']):
        raise ValueError('source plan partition binding mismatch')
    for teacher in (bootstrap, coverage):
        if (teacher.get('partition_manifest') != MANIFEST_PATH
                or teacher.get('partition_sha256') != actual['manifest']):
            raise ValueError('teacher plan partition binding mismatch')

    partitions = manifest.get('task_ids')
    if not isinstance(partitions, dict) or set(partitions) != set(TASK_COUNTS):
        raise ValueError('wrong train partition set')
    scenarios = {}
    all_tasks = []
    for name, expected_count in TASK_COUNTS.items():
        tasks = _ids(partitions[name], name)
        if len(tasks) != expected_count or tasks != _lines(raw[name], name):
            raise ValueError(f'partition count or order mismatch: {name}')
        by_scenario = {}
        for task in tasks:
            scenario, variant = task.rsplit('_', 1)
            by_scenario.setdefault(scenario, set()).add(variant)
        if (len(by_scenario) != SCENARIO_COUNTS[name]
                or any(variants != {'1', '2', '3'} for variants in by_scenario.values())):
            raise ValueError(f'partition scenario or variant mismatch: {name}')
        scenarios[name] = set(by_scenario)
        all_tasks.extend(tasks)
    if any(scenarios[a] & scenarios[b] for a, b in (
        ('origin', 'utility'), ('origin', 'distill'), ('utility', 'distill')
    )):
        raise ValueError('scenario leakage across train partitions')
    if len(train) != 90 or len(set(all_tasks)) != 90 or set(all_tasks) != set(train):
        raise ValueError('train partitions do not cover the official 90 train IDs')

    selected = [task for task in partitions['origin'] if task.endswith('_2')]
    source_ids = _ids(plan.get('task_ids'), 'source plan')
    if (len(selected) != 12 or len(source_ids) != 12
            or any(not task.endswith('_1') for task in source_ids)
            or selected != [task[:-1] + '2' for task in source_ids]):
        raise ValueError('origin selection does not match frozen source order')
    first_teacher = _ids(bootstrap.get('teacher_task_ids'), 'teacher bootstrap')
    second_teacher = _ids(coverage.get('teacher_task_ids'), 'teacher coverage')
    if (len(first_teacher) != 4 or len(second_teacher) != 8
            or set(first_teacher) & set(second_teacher)
            or set(first_teacher) | set(second_teacher) != set(selected)):
        raise ValueError('historical teacher-plan exposure mismatch')

    return {
        'status': 'draft',
        'task_metadata_loaded': False,
        'study_registered': False,
        'ready_for_experiment': False,
        'selection_rule': 'origin manifest order; second variant of each scenario',
        'selected_ids': selected,
        'official_train_task_count': len(train),
        'partition_task_counts': dict(TASK_COUNTS),
        'partition_scenario_counts': dict(SCENARIO_COUNTS),
        'historical_teacher_exposure': {
            'bootstrap_ids': first_teacher,
            'coverage_ids': second_teacher,
            'interpretation': 'IDs listed in historical teacher plans; not unseen by the project',
        },
        'source_sha256': actual,
    }


def prepare():
    """Read only the frozen inputs and return the validated draft manifest."""
    raw = {key: _read_frozen(path, FROZEN_SHA256[key]) for key, path in SOURCE_PATHS.items()}
    plan = _json_object(raw['source_plan'], 'source plan')
    if plan.get('data_dir') != str(OFFICIAL_TRAIN.parent.parent):
        raise ValueError('official train location does not match source plan')
    return _prepare_sources(raw, FROZEN_SHA256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args(argv)
    try:
        result = prepare()
    except ValueError as exc:
        parser.exit(2, f'preparation failed: {exc}\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
