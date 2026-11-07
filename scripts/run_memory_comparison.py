"""Three fresh arms per task with a single shared control and bounded stopping."""
import json
import re
from dataclasses import asdict
from pathlib import Path

from scripts.report_memory_branch import audit_branch
from scripts.run_foundation_pilot import digest, require
from scripts.run_memory_branch import ARMS, protocol, run_branch
from wll.protocol import fingerprint
from wll.storage import atomic_json, require_space


def validate_inputs(tasks, candidates, task_offset):
    require(type(task_offset) is int and task_offset >= 0 and tasks
        and len({t.task_id for t in tasks}) == len(tasks)
        and all(re.fullmatch('[A-Za-z0-9_-]+', t.task_id) for t in tasks), 'invalid task schedule')
    require(set(candidates) == set(ARMS[1:])
        and len({c.experience_id for c in candidates.values()}) == 2, 'two distinct candidate identities required')


def branch_schedule(tasks, task_offset):
    return [(task.task_id, arm) for index, task in enumerate(tasks)
        for arm in ARMS[(index + task_offset) % 3:] + ARMS[:(index + task_offset) % 3]]


def audit_batch(directory, tasks, candidates, expected, *, task_offset, allow_running=False):
    directory = Path(directory)
    validate_inputs(tasks, candidates, task_offset)
    require(set(expected) == {t.task_id for t in tasks}
        and all(set(arms) == set(ARMS) for arms in expected.values()), 'expected arm set differs')
    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest['tasks'] == [asdict(t) for t in tasks]
        and manifest['candidates'] == {a: asdict(c) for a, c in candidates.items()}
        and manifest['task_offset'] == task_offset and manifest['expected_sha256'] == fingerprint(expected)
        and manifest['runner_sha256'] == digest(__file__)
        and manifest['automatic_training'] is False and manifest['maximum_attempts_per_branch'] == 1
        and json.loads((directory / 'expected.json').read_text()) == expected, 'batch protocol differs')
    require(manifest['status'] in ({'running', 'complete', 'incomplete'} if allow_running else {'complete', 'incomplete'}),
            'batch has no terminal status')
    schedule = branch_schedule(tasks, task_offset)
    by_id = {t.task_id: t for t in tasks}
    trials = directory / 'trials'
    present = {(task.name, arm.name) for task in trials.iterdir() if task.is_dir()
        for arm in task.iterdir() if arm.is_dir()} if trials.exists() else set()
    task_dirs = {p.name for p in trials.iterdir() if p.is_dir()} if trials.exists() else set()
    require(task_dirs <= by_id.keys() and present == set(schedule[:len(present)]), 'attempts not an ordered schedule prefix')
    reports = []
    for task_id, arm in schedule:
        if (task_id, arm) not in present:
            continue
        require(all(r['audit']['valid_for_continuation'] for r in reports), 'continued after an unknown branch')
        audit = audit_branch(trials / task_id / arm, by_id[task_id], candidates.get(arm), expected[task_id][arm])
        reports.append({'task_id': task_id, 'arm': arm, 'audit': audit})
    known = sum(r['audit']['valid_for_continuation'] for r in reports)
    complete = known == len(schedule)
    if not allow_running:
        require(manifest['status'] == ('complete' if complete else 'incomplete'), 'batch terminal status differs')
    return {'status': 'complete' if complete else 'incomplete', 'scheduled_branches': len(schedule),
        'attempted_branches': len(reports), 'known_branches': known, 'unknown_branches': len(reports) - known,
        'unattempted_branches': len(schedule) - len(reports), 'valid_for_aggregation': complete,
        'successes_by_arm': {a: sum(r['audit']['outcome'] is True for r in reports if r['arm'] == a) for a in ARMS},
        'returned_response_cost': {k: sum(r['audit']['returned_response_cost'][k] for r in reports)
            for k in ('prompt_tokens', 'completion_tokens')},
        'observed_step_cost': {k: sum(r['audit']['new_step_cost'][k] for r in reports)
            for k in ('steps', 'prompt_tokens', 'completion_tokens')},
        'generation_counters': {k: sum(r['audit']['generation_counters'][k] for r in reports) for k in
            ('generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage')},
        'reported_context_rejections': sum(r['audit']['reported_context_rejections'] for r in reports),
        'unknown_additional_response_cost': any(r['audit']['unknown_additional_response_cost'] for r in reports),
        'trials': reports, 'accepted_experiences': 0, 'automatic_training': False,
        'expected_protocol_sha256': fingerprint(expected),
        'scope': 'Raw three-arm batch only. Shared control counted once; no full-study selection, '
            'service-event accounting, physical cleanup or held-out efficacy certified.'}


def run_comparison(tasks, candidates, world_factory, model, *, output, task_offset, max_steps, seconds,
                   seed, environment_id, expected_backend, prompt_version, before_branch=None):
    tasks = list(tasks)
    validate_inputs(tasks, candidates, task_offset)
    args = dict(max_steps=max_steps, seconds=seconds, seed=seed, environment_id=environment_id,
                expected_backend=expected_backend, prompt_version=prompt_version)
    expected = {t.task_id: {a: protocol(t, candidates.get(a), model, arm=a, **args,
        backend_idle_check_enabled=before_branch is not None) for a in ARMS} for t in tasks}
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=True)
    require(not any((output / name).exists() for name in ('manifest.json', 'expected.json', 'trials', 'report.json')),
            'comparison already attempted')
    manifest = {'status': 'running', 'tasks': [asdict(t) for t in tasks],
        'candidates': {a: asdict(c) for a, c in candidates.items()}, 'task_offset': task_offset,
        'expected_sha256': fingerprint(expected), 'runner_sha256': digest(__file__),
        'maximum_attempts_per_branch': 1, 'automatic_training': False}
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'expected.json', expected)
    by_id = {t.task_id: t for t in tasks}
    try:
        for task_id, arm in branch_schedule(tasks, task_offset):
            task = by_id[task_id]
            result = run_branch(task, candidates.get(arm), lambda t, arm=arm: world_factory(t, arm), model,
                output=output / 'trials' / task_id / arm, arm=arm, **args,
                before_rollout=(lambda task=task, arm=arm: before_branch(task, arm)) if before_branch is not None else None)
            atomic_json(output / 'report.json', audit_batch(output, tasks, candidates, expected,
                task_offset=task_offset, allow_running=True))
            if result['status'] != 'complete':
                break
    finally:
        report = audit_batch(output, tasks, candidates, expected, task_offset=task_offset, allow_running=True)
        manifest['status'] = report['status']
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'report.json', report)
    return report
