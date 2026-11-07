"""Versioned full-dev runner with an explicitly verified AppWorld seed.

Train calls delegate to the frozen runner. Dev uses its deadline, cost and
cleanup helpers, with a local loop because the original has no hook between
world construction and generation or for per-attempt seed provenance.
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

from scripts import run_timed_tasks as legacy
from scripts.run_timed_tasks import (
    BackendMismatch,
    CheckedModel,
    CleanupTrackedWorld,
    RolloutDeadline,
    RolloutTimedOut,
    wait_for_idle,
)
from wll.appworld import AppWorldEnvironment, load_tasks
from wll.model import LocalModel
from wll.prompts import DEFAULT_VERSION, SYSTEMS
from wll.protocol import fingerprint
from wll.runner import run_episode
from wll.storage import atomic_json, require_space

ROOT = Path(__file__).resolve().parent.parent
PREPARATION = ROOT / 'artifacts/qwen38-b3-dev-evaluation-preparation-001.json'
PREPARATION_SHA256 = 'e748590baba716ac3fcc3fd2eaa9d6fdfed915833c15891b9a045d8100ca33e4'
OFFICIAL_DEV_SHA256 = '9fa976589300ea8905708257144d801d1604b06d85fb0181e381df8a3ba85001'
DEV_PROMPT = 'appworld-v3'


class SeedMismatch(ValueError):
    """The requested environment seed was not independently confirmed."""


def _bound_bytes(path, expected, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'{description} must be a regular file')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'{description} hash changed')
    return raw


def validate_dev_ids(ids, data_dir):
    """Verify both frozen preparation and the full official split before use."""
    prep = json.loads(_bound_bytes(PREPARATION, PREPARATION_SHA256, 'dev preparation'))
    expected = prep['dataset']['dev_ids_official_order']
    if (not isinstance(expected, list) or len(expected) != 57
            or any(not isinstance(item, str) or not item for item in expected)
            or len(set(expected)) != 57
            or len({item.rsplit('_', 1)[0] for item in expected}) != 19
            or prep['dataset']['metadata_file_sha256']['official_dev'] != OFFICIAL_DEV_SHA256):
        raise ValueError('invalid bound dev preparation')
    official = _bound_bytes(Path(data_dir) / 'datasets/dev.txt', OFFICIAL_DEV_SHA256,
                            'official dev split').decode('utf-8').splitlines()
    if official != expected or list(ids) != expected:
        raise ValueError('dev task IDs must match the complete official order')
    return expected


def _validate_seed(value):
    if (not isinstance(value, dict) or set(value) != {'requested', 'effective', 'verified'}
            or type(value['requested']) is not int or value['requested'] != 100
            or type(value['effective']) is not int or value['effective'] != 100
            or value['verified'] is not True):
        raise SeedMismatch('environment seed not verified')
    return {'requested': 100, 'effective': 100, 'verified': True}


def _validate_dev_recipe(*, environment_seed, max_steps, rollout_seconds, seed, model,
                         prompt_version):
    if type(environment_seed) is not int or environment_seed != 100:
        raise ValueError('dev requires explicit integer environment seed 100')
    if (type(seed) is not int or seed != 17 or type(max_steps) is not int or max_steps != 40
            or isinstance(rollout_seconds, bool) or rollout_seconds != 600
            or getattr(model, 'max_tokens', None) != 1024
            or getattr(model, 'temperature', None) != .7 or prompt_version != DEV_PROMPT):
        raise ValueError('different fixed dev rollout recipe')


def run_tasks(tasks, world_factory, model, *, output, max_steps, rollout_seconds, seed,
              expected_backend, prompt_version=DEFAULT_VERSION, environment_id=None,
              before_task=None, split='train', environment_seed=None, data_dir=None):
    """Run legacy training unchanged or the complete, seeded official dev split."""
    if split == 'train':
        if environment_seed is not None:
            raise ValueError('legacy train does not accept an environment seed')
        return legacy.run_tasks(tasks, world_factory, model, output=output, max_steps=max_steps,
            rollout_seconds=rollout_seconds, seed=seed, expected_backend=expected_backend,
            prompt_version=prompt_version, environment_id=environment_id, before_task=before_task)
    if split != 'dev' or data_dir is None:
        raise ValueError('dev requires its official data directory')
    _validate_dev_recipe(environment_seed=environment_seed, max_steps=max_steps,
        rollout_seconds=rollout_seconds, seed=seed, model=model, prompt_version=prompt_version)
    tasks = list(tasks)
    ids = [getattr(task, 'task_id', None) for task in tasks]
    validate_dev_ids(ids, data_dir)
    if any(getattr(task, 'split', None) != 'dev'
           or getattr(task, 'scenario_id', None) != task.task_id.rsplit('_', 1)[0]
           for task in tasks):
        raise ValueError('dev task split or scenario differs')
    if not isinstance(expected_backend, dict) or not expected_backend:
        raise ValueError('frozen backend is required')
    RolloutDeadline(rollout_seconds)
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    unknown_seed = {'requested': 100, 'effective': None, 'verified': False}
    manifest = {'tasks': [asdict(t) for t in tasks], 'model_id': model.model_id,
        'max_tokens': getattr(model, 'max_tokens', None),
        'temperature': getattr(model, 'temperature', None),
        'seed': seed, 'max_steps': max_steps, 'rollout_seconds': rollout_seconds,
        'prompt_version': prompt_version, 'environment_id': environment_id,
        'expected_backend': expected_backend, 'split': 'dev',
        'preparation_sha256': PREPARATION_SHA256, 'official_dev_sha256': OFFICIAL_DEV_SHA256,
        'environment_seed': dict(unknown_seed),
        'environment_seed_scope': 'verified for all scheduled tasks; see attempts for individual evidence',
        'backend_idle_check_enabled': before_task is not None,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wll_code_fingerprint': fingerprint({p.name: p.read_text() for p in
                                           (ROOT / 'wll').glob('*.py')}),
        'deadline_scope': 'rollout only; environment startup and cleanup separately bounded',
        'status': 'running'}
    atomic_json(output / 'manifest.json', manifest)
    counters = {'generation_calls': 0, 'generation_calls_without_response': 0,
                'generation_calls_with_unusable_usage': 0,
                'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0},
                'observed_step_cost': {'steps': 0, 'prompt_tokens': 0, 'completion_tokens': 0}}
    records = []
    started = time.monotonic()

    def summarize(status):
        completed = sum(r['status'] == 'complete' for r in records)
        successes = sum(r['success'] is True for r in records)
        unknown = len(tasks) - completed
        return {'status': status, 'scheduled': len(tasks), 'attempted': len(records),
            'completed': completed, 'successes': successes, 'unknown_outcomes': unknown,
            'unattempted': len(tasks)-len(records), 'valid_for_comparison': unknown == 0,
            'success_rate_completed_only': successes/completed if completed else None,
            'success_rate_all_scheduled': successes/len(tasks) if unknown == 0 else None,
            'success_bounds_all_scheduled': [successes/len(tasks), (successes+unknown)/len(tasks)],
            **counters, 'wall_seconds': time.monotonic()-started,
            'cost_note': 'Returned-response and observed-step costs overlap; do not add them. '
                         'Calls without a usable token count may incur unknown additional generation.'}

    terminal = 'interrupted'
    try:
        with (output / 'episodes.jsonl').open('w') as episodes, \
                (output / 'steps.jsonl').open('w') as journal, \
                (output / 'responses.jsonl').open('w') as responses, \
                (output / 'attempts.jsonl').open('w') as attempts:
            for task in tasks:
                deadline = RolloutDeadline(rollout_seconds)
                world = None
                abort = False
                record = {'task_id': task.task_id, 'status': 'infrastructure_error',
                          'success': None, 'cleanup_verified': None, 'started_epoch': time.time(),
                          'environment_seed': dict(unknown_seed)}

                def observe(index, step, replayed, task_id=task.task_id):
                    assert not replayed
                    journal.write(json.dumps({'task_id': task_id, 'index': index,
                        'replayed': False, 'step': asdict(step)}, ensure_ascii=False)+'\n')
                    journal.flush()
                    counters['observed_step_cost']['steps'] += 1
                    for key in ('prompt_tokens', 'completion_tokens'):
                        counters['observed_step_cost'][key] += getattr(step, key)

                def on_response(value, task_id=task.task_id):
                    responses.write(json.dumps({'task_id': task_id, **value})+'\n')
                    responses.flush()

                checked = CheckedModel(model, expected_backend, counters, on_response)
                try:
                    waiting = time.monotonic()
                    try:
                        if before_task is not None:
                            before_task(task)
                    finally:
                        record['backend_wait_seconds'] = time.monotonic()-waiting
                    startup = time.monotonic()
                    try:
                        world = CleanupTrackedWorld(world_factory(task), deadline)
                        try:
                            record['environment_seed'] = _validate_seed(world.environment_seed_provenance)
                        except Exception as exc:
                            raise SeedMismatch('environment seed not verified') from exc
                    finally:
                        record['startup_seconds'] = time.monotonic()-startup
                    with deadline:
                        episode = run_episode(task, lambda world=world: world, checked,
                            max_steps=max_steps, seed=seed, on_step=observe, prompt_version=prompt_version)
                    episode.metadata.update(environment_id=environment_id,
                        rollout_deadline_seconds=rollout_seconds,
                        environment_seed=dict(record['environment_seed']))
                    episodes.write(json.dumps(episode.to_dict(), ensure_ascii=False)+'\n')
                    episodes.flush()
                    record.update(status='complete', success=episode.success,
                                  episode_hash=fingerprint(episode.to_dict()))
                except RolloutTimedOut as exc:
                    record.update(status='timed_out', error_type=type(exc).__name__)
                except Exception as exc:
                    record['error_type'] = type(exc).__name__
                    abort = world is None or isinstance(exc, (BackendMismatch, SeedMismatch))
                except BaseException as exc:
                    record.update(status='interrupted', error_type=type(exc).__name__)
                    raise
                finally:
                    if world is not None:
                        try:
                            world.close()
                        except Exception as exc:
                            record['cleanup_error_type'] = type(exc).__name__
                            abort = True
                        record['cleanup_verified'] = world.closed
                    record['finished_epoch'] = time.time()
                    records.append(record)
                    attempts.write(json.dumps(record)+'\n')
                    attempts.flush()
                    atomic_json(output / 'summary.json', summarize('running'))
                if abort:
                    manifest['stop_reason'] = 'startup_backend_or_cleanup_failure'
                    break
        terminal = 'complete' if all(r['status'] == 'complete' for r in records) and (
            len(records) == len(tasks)) else 'incomplete'
    finally:
        if len(records) == len(tasks) and all(r['environment_seed']['verified'] is True for r in records):
            manifest['environment_seed'] = {'requested': 100, 'effective': 100, 'verified': True}
        manifest['status'] = terminal
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'summary.json', summarize(terminal))
    return summarize(terminal)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'dev'], default='train')
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--deps-dir', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--task-ids-file', required=True)
    parser.add_argument('--backend-file', required=True, help='Verified server health JSON with backend')
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--rollout-seconds', type=float, required=True)
    parser.add_argument('--server-idle-seconds', type=float, default=180)
    parser.add_argument('--prompt-version', choices=list(SYSTEMS))
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('this CPU evaluation client requires CUDA_VISIBLE_DEVICES empty')
    prompt_version = args.prompt_version or (DEV_PROMPT if args.split == 'dev' else DEFAULT_VERSION)
    ids = Path(args.task_ids_file).read_text().splitlines()
    if args.split == 'dev':
        # Check all selection and fixed settings before any Docker or model access.
        class Settings:
            max_tokens = args.max_tokens
            temperature = args.temperature

        _validate_dev_recipe(environment_seed=args.environment_seed, max_steps=args.steps,
            rollout_seconds=args.rollout_seconds, seed=args.seed, model=Settings(),
            prompt_version=prompt_version)
        validate_dev_ids(ids, args.data_dir)
    elif args.environment_seed is not None:
        raise ValueError('legacy train does not accept an environment seed')
    health = json.loads(Path(args.backend_file).read_text())
    if health['model'] != args.model:
        raise ValueError('health record model differs')
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                     args.image], text=True, timeout=15).strip()
    loaded = load_tasks(args.data_dir, args.split, image=image, deps_dir=args.deps_dir,
                        include_task_ids=ids)
    if args.split == 'dev':
        if [task.task_id for task in loaded] != ids:
            raise ValueError('loaded dev task order differs')
        from wll.appworld_v2 import AppWorldEnvironmentV2

        def factory(task):
            return AppWorldEnvironmentV2(task.task_id, args.data_dir, args.steps,
                image, args.deps_dir, environment_seed=args.environment_seed)

        ordered = loaded
    else:
        by_id = {task.task_id: task for task in loaded}
        ordered = [by_id[task_id] for task_id in ids]
        def factory(task):
            return AppWorldEnvironment(task.task_id, args.data_dir, args.steps, image, args.deps_dir)
    model = LocalModel(args.endpoint, args.model, max_tokens=args.max_tokens,
                       temperature=args.temperature)
    result = run_tasks(ordered, factory, model, output=args.output, max_steps=args.steps,
        rollout_seconds=args.rollout_seconds, seed=args.seed, expected_backend=health['backend'],
        prompt_version=prompt_version, environment_id=image,
        before_task=lambda task: wait_for_idle(args.endpoint, args.model, health['backend'],
                                              seconds=args.server_idle_seconds),
        split=args.split, environment_seed=args.environment_seed, data_dir=args.data_dir)
    print(json.dumps(result), flush=True)
    if not result['valid_for_comparison']:
        raise SystemExit(1)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
