"""CPU comparison of bare versus printed, literal documentation queries.

This retrospective mechanism probe never generates actions or scores tasks.
Modified code executes only in the existing constrained AppWorld container.
"""
import argparse
import ast
import collections
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.run_foundation_pilot import (
    ROOT,
    audit_run,
    digest,
    expected_protocol,
    read,
    require,
    validate_health,
    verify_frozen_plan,
)
from scripts.run_timed_tasks import CleanupTrackedWorld, RolloutDeadline
from wll.appworld import AppWorldEnvironment
from wll.cli import read_episodes
from wll.protocol import fingerprint
from wll.runner import ReplayMismatch
from wll.storage import atomic_json


def printable_documentation_call(action):
    try:
        tree = ast.parse(action)
    except SyntaxError as exc:
        raise ValueError('not a single literal documentation query') from exc
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr),
            'only a single bare expression is permitted')
    call = tree.body[0].value
    require(isinstance(call, ast.Call)
            and ast.unparse(call.func) == 'apis.api_docs.show_api_descriptions'
            and not call.args and len(call.keywords) == 1, 'not the allowed documentation query')
    keyword = call.keywords[0]
    require(keyword.arg == 'app_name' and isinstance(keyword.value, ast.Constant)
            and type(keyword.value.value) is str and 0 < len(keyword.value.value) <= 64,
            'only a literal bounded app name is permitted')
    return ast.unparse(ast.Call(func=ast.Name(id='print', ctx=ast.Load()), args=[call], keywords=[]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--batches', type=int, nargs='+', choices=(0, 1, 2), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only client required')
    require(args.batches == sorted(set(args.batches)), 'ordered unique batches required')
    plan = verify_frozen_plan(ROOT, Path(args.plan), args.plan_sha256)
    frozen = [json.loads(line) for line in
              (ROOT/'artifacts/foundation-inputs-001/tasks.jsonl').read_text().splitlines()]
    sources, files = [], {}
    for index in args.batches:
        batch = ROOT/'artifacts'/plan['id']/f'batch-{index}'
        health = read(batch/'health.json')
        validate_health(health, plan, index)
        report = audit_run(batch/'run', expected_protocol(plan, frozen[index*4:index*4+4], health))
        require(report == read(batch/'report.json') and report['valid_for_comparison'],
                'completed audited source batch required')
        require(read(batch/'release.json')['released'] is True, 'source GPU release missing')
        sources.extend(read_episodes(batch/'run/episodes.jsonl'))
        for name in ('report.json', 'release.json', 'run/episodes.jsonl'):
            files[str((batch/name).relative_to(ROOT))] = digest(batch/name)
    selected, seen = [], set()
    for source in sources:
        counts = collections.Counter(s.action for s in source.steps)
        for index, step in enumerate(source.steps):
            if counts[step.action] < 10:
                continue
            try:
                printed = printable_documentation_call(step.action)
            except ValueError:
                continue
            key = ast.dump(ast.parse(step.action), include_attributes=False)
            if key not in seen:
                selected.append((source, index, printed, counts[step.action]))
                seen.add(key)
    require(selected, 'no repeated literal documentation query found')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'purpose': 'retrospective CPU visibility probe,not recovery or model evaluation',
        'uses_gpu': False, 'llm_calls': 0, 'official_scoring_calls': 0,
        'plan_sha256': args.plan_sha256, 'script_sha256': digest(__file__), 'batches': args.batches,
        'source_files_sha256': files,
        'selection': '>=10 identical documentation queries within one completed source;first source per unique query',
        'cases': [{'task_id': e.task.task_id, 'source_hash': fingerprint(e.to_dict()),
                   'prefix_length': i, 'source_repetitions': n, 'action_hash': fingerprint(e.steps[i].action),
                   'printed_action_hash': fingerprint(printed)} for e, i, printed, n in selected]}
    atomic_json(output/'manifest.json', manifest)
    records, resources = [], []
    summary = {'phase': 'running', 'cases': len(selected), 'replayed_observations': 0,
               'llm_calls': 0, 'gpu_jobs': 0, 'official_scoring_calls': 0}
    started = time.monotonic()
    try:
        for number, (source, prefix_length, printed, _) in enumerate(selected):
            for condition in ('bare', 'printed'):
                record = {'case': number, 'condition': condition, 'task_id': source.task.task_id,
                          'status': 'unknown', 'replayed_observations': 0, 'cleanup_verified': False}
                world = None
                deadline = RolloutDeadline(120)
                try:
                    actual = AppWorldEnvironment(source.task.task_id, plan['data_dir'], plan['max_steps'],
                                                 plan['image'], plan['deps_dir'])
                    world = CleanupTrackedWorld(actual, deadline)
                    identifier = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}',
                        actual.worker.name], text=True, timeout=15).strip()
                    resources.append({'id': identifier, 'name': actual.worker.name})
                    atomic_json(output/'containers.json', resources)
                    with deadline:
                        for old in source.steps[:prefix_length]:
                            observed = world.execute(old.action)
                            record['replayed_observations'] += 1
                            summary['replayed_observations'] += 1
                            if observed != old.observation:
                                raise ReplayMismatch('source prefix observation changed')
                        action = source.steps[prefix_length].action if condition == 'bare' else printed
                        observed = world.execute(action)
                        if condition == 'bare':
                            require(observed == source.steps[prefix_length].observation,
                                    'bare query observation changed')
                    record.update(status='complete', observation_chars=len(observed),
                        observation_hash=fingerprint(observed),
                        generic_success_only=observed == 'Execution successful.',
                        execution_failed=observed.startswith('Execution failed'))
                except BaseException as exc:
                    record['error_type'] = type(exc).__name__
                    raise
                finally:
                    try:
                        if world is not None:
                            world.close()
                            record['cleanup_verified'] = world.closed
                    finally:
                        records.append(record)
                        atomic_json(output/'records.json', records)
        summary.update(phase='complete', visibility_changed_cases=sum(
            a['generic_success_only'] and not b['generic_success_only'] and not b['execution_failed']
            and b['observation_chars'] > a['observation_chars']
            for a, b in zip(records[::2], records[1::2], strict=True)))
    except BaseException as exc:
        summary.update(phase='failed', error_type=type(exc).__name__)
        raise
    finally:
        absent = {r['id']: not subprocess.check_output(['docker', 'ps', '-aq', '--filter',
                  'id='+r['id']], text=True, timeout=15).strip() for r in resources}
        summary.update(wall_seconds=time.monotonic()-started, recorded_container_absence=absent)
        atomic_json(output/'summary.json', summary)
        require(all(absent.values()), 'recorded container remains')
    print(json.dumps({k: v for k, v in summary.items() if k != 'recorded_container_absence'}))


if __name__ == '__main__':
    main()
