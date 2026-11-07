"""CPU-only replay readiness; no new actions, lessons, or model selection.

Audit the first visible execution error in every supplied, completed train source,
regardless of final success. A prefix score is terminal and is never fed to a
policy. Any later recovery must create another fresh environment and replay again.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from scripts.run_foundation_pilot import (
    ROOT,
    audit_run,
    digest,
    environment_id,
    expected_protocol,
    observation_world,
    read,
    require,
    validate_health,
    verify_frozen_plan,
)
from scripts.run_timed_tasks import CleanupTrackedWorld, RolloutDeadline
from wll.appworld import AppWorldEnvironment
from wll.cli import read_episodes
from wll.protocol import fingerprint
from wll.runner import ReplayMismatch, run_episode
from wll.storage import atomic_json


def audit_prefixes(episodes, world_factory, *, output, max_seconds=120, source_evidence=None,
                   expected_environment_id=None):
    episodes = list(episodes)
    require(episodes and len({e.task.task_id for e in episodes}) == len(episodes),
            'nonempty unique source tasks required')
    require(all(e.task.split == 'train' and e.branch == 'first' and e.replayed_steps == 0
                and e.experience_id is None and type(e.success) is bool for e in episodes),
            'only completed first-attempt train sources are permitted')
    if expected_environment_id is not None:
        require(isinstance(expected_environment_id, str) and expected_environment_id
                and all(e.metadata.get('environment_id') == expected_environment_id for e in episodes),
                'source and replay environment identities differ')
    RolloutDeadline(max_seconds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'purpose': 'CPU replay readiness, not new recovery or model evaluation',
        'script_sha256': digest(__file__), 'source_hashes': [fingerprint(e.to_dict()) for e in episodes],
        'task_ids': [e.task.task_id for e in episodes], 'selection': 'first_visible_execution_error',
        'prefix_deadline_seconds': max_seconds, 'uses_gpu': False,
        'source_evidence': source_evidence,
        'environment_id': expected_environment_id,
        'deadline_scope': 'replay and terminal boolean scoring; startup/cleanup separately bounded'}
    atomic_json(output/'manifest.json', manifest)
    selected = sum(any(s.observation.startswith('Execution failed') for s in e.steps)
                   for e in episodes)
    stats = {'status': 'running', 'source_tasks': len(episodes), 'selected_prefixes': selected,
        'skipped_without_visible_error': len(episodes)-selected, 'attempted_prefixes': 0,
        'verified_prefixes': 0, 'verified_failed_prefixes': 0, 'unknown_prefixes': selected,
        'replay_observations': 0, 'model_generation_calls': 0,
        'source_prompt_tokens': sum(s.prompt_tokens for e in episodes for s in e.steps),
        'source_completion_tokens': sum(e.generated_tokens for e in episodes),
        'cost_note': 'Source tokens are historical; do not add them to parent study costs. '
                     'Replay cost includes returned mismatch observations. No new model tokens.'}
    started = time.monotonic()
    atomic_json(output/'summary.json', stats)
    try:
        with (output/'prefixes.jsonl').open('x') as stream:
            for source in episodes:
                error = next((i for i, s in enumerate(source.steps)
                              if s.observation.startswith('Execution failed')), None)
                if error is None:
                    continue
                prefix = source.steps[:error+1]
                record = {'task_id': source.task.task_id, 'scenario_id': source.task.scenario_id,
                    'source_hash': fingerprint(source.to_dict()),
                    'source_class': ('successful_trace_with_visible_error' if source.success
                                     else 'failed_trace_with_visible_error'),
                    'source_success': source.success, 'prefix_length': len(prefix),
                    'prefix_sha256': fingerprint([asdict(s) for s in prefix]),
                    'source_prompt_tokens': sum(s.prompt_tokens for s in source.steps),
                    'source_completion_tokens': source.generated_tokens,
                    'status': 'unknown', 'prefix_success': None, 'replay_observations': 0,
                    'cleanup_verified': False, 'started_epoch': time.time()}
                stats['attempted_prefixes'] += 1
                deadline = RolloutDeadline(max_seconds)
                world = None

                class NoGeneration:
                    model_id = source.model_id

                    def generate(self, *args, **kwargs):
                        raise AssertionError('CPU prefix audit cannot generate')

                def observe(index, step, replayed, record=record):
                    require(replayed is True, 'unexpected newly generated action')
                    record['replay_observations'] += 1
                    stats['replay_observations'] += 1

                try:
                    world = CleanupTrackedWorld(world_factory(source.task), deadline)
                    with deadline:
                        result = run_episode(source.task, lambda world=world: world, NoGeneration(),
                            max_steps=len(prefix), seed=source.seed, prefix=prefix,
                            branch='prefix_readiness', on_step=observe,
                            prompt_version=source.metadata.get('prompt_version', 'appworld-v1'))
                    require(world.closed, 'prefix environment did not close')
                    record.update(status='verified', prefix_success=result.success)
                    stats['verified_prefixes'] += 1
                    stats['verified_failed_prefixes'] += int(not result.success)
                    stats['unknown_prefixes'] -= 1
                except BaseException as exc:
                    record.update(status='replay_mismatch' if isinstance(exc, ReplayMismatch)
                                  else 'unknown', error_type=type(exc).__name__)
                    raise
                finally:
                    try:
                        if world is not None:
                            world.close()
                            record['cleanup_verified'] = world.closed
                    finally:
                        record['finished_epoch'] = time.time()
                        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                        stream.flush()
                        atomic_json(output/'summary.json', stats)
        stats['status'] = 'complete'
        return stats
    except BaseException as exc:
        stats.update(status='interrupted', error_type=type(exc).__name__)
        raise
    finally:
        stats['wall_seconds'] = time.monotonic()-started
        atomic_json(output/'summary.json', stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--batch', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only execution required')
    plan_path = Path(args.plan).resolve(strict=True)
    plan = verify_frozen_plan(ROOT, plan_path, args.plan_sha256)
    identity = environment_id(plan)
    batch = ROOT/'artifacts'/plan['id']/f'batch-{args.batch}'
    health = read(batch/'health.json')
    validate_health(health, plan, args.batch)
    frozen = [json.loads(line) for line in
              (ROOT/'artifacts/foundation-inputs-001/tasks.jsonl').read_text().splitlines()]
    expected = expected_protocol(plan, frozen[args.batch*4:args.batch*4+4], health)
    report = audit_run(batch/'run', expected)
    require(report == read(batch/'report.json') and report['valid_for_comparison'],
            'source batch must be fully scored and journal-audited')
    release, cleanup = read(batch/'release.json'), read(batch/'local-cleanup.json')
    require(release['released'] is True and not release['model_pids']
            and release['job']['status'] == 'finished' and cleanup['recorded_containers_absent'],
            'source batch release is not recorded')
    episodes = read_episodes(batch/'run/episodes.jsonl')
    output = Path(args.output).resolve()
    require(not output.exists(), 'prefix output already exists; cannot repeat or overwrite')
    files = [batch/'run'/name for name in report['input_sha256']]
    files.extend(batch/name for name in ('report.json', 'health.json', 'release.json',
                                        'local-cleanup.json'))
    evidence = {'plan_sha256': args.plan_sha256, 'batch': args.batch,
        'environment_id': identity,
        'source_files_sha256': {str(p.relative_to(ROOT)): digest(p) for p in files},
        'source_study_may_be_incomplete': True, 'does_not_enable_recovery_or_training': True,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    resources = []
    feedback = None

    def factory(task):
        nonlocal feedback
        world = AppWorldEnvironment(task.task_id, plan['data_dir'], plan['max_steps'],
                                    plan['image'], plan['deps_dir'])
        try:
            identifier = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}',
                world.worker.name], text=True, timeout=15).strip()
            resources.append({'task_id': task.task_id, 'name': world.worker.name, 'id': identifier})
            atomic_json(output/'containers.json', resources)
            if feedback is None:
                feedback = (output/'feedback-events.jsonl').open('x')
            return observation_world(world, plan, task.task_id, feedback)
        except BaseException:
            world.close()
            raise

    try:
        result = audit_prefixes(episodes, factory, output=output, source_evidence=evidence,
                                expected_environment_id=identity)
    finally:
        if feedback is not None:
            feedback.close()
        if output.exists():
            absent = {}
            for record in resources:
                remaining = subprocess.check_output(['docker', 'ps', '-aq', '--filter',
                    'id='+record['id']], text=True, timeout=15)
                absent[record['id']] = not remaining.strip()
            atomic_json(output/'cleanup.json', {'recorded_container_absence': absent,
                'all_recorded_containers_absent': all(absent.values()),
                'unrecorded_startup_failures_not_certified': True})
            require(all(absent.values()), 'a recorded prefix container remains')
    print(json.dumps(result))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128+signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    main()
