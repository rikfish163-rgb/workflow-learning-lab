"""Sequential task deadlines with preserved unknown outcomes and verified cleanup.

This is a separate entrypoint; historical batch controllers remain unchanged.
The rollout deadline starts after environment startup and stops before cleanup.
AppWorld startup/RPC and container cleanup retain their own existing timeouts.
"""
import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from wll.appworld import AppWorldEnvironment, load_tasks
from wll.model import LocalModel
from wll.prompts import DEFAULT_VERSION, SYSTEMS
from wll.protocol import fingerprint
from wll.runner import run_episode
from wll.storage import atomic_json, require_space


class RolloutTimedOut(TimeoutError):
    pass


class BackendMismatch(RuntimeError):
    pass


class RolloutDeadline:
    def __init__(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('rollout deadline must be finite and positive')
        self.seconds = seconds
        self.active = False
        self.used = False

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('rollout deadlines require the main thread')
        if self.used or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError('cannot reuse a deadline or replace an existing timer')
        self.used = True
        self.previous = signal.getsignal(signal.SIGALRM)

        def expired(signum, frame):
            raise RolloutTimedOut('per-task rollout wall-time budget exhausted')

        signal.signal(signal.SIGALRM, expired)
        self.active = True
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def cancel(self):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.active = False

    def __exit__(self, *args):
        self.cancel()
        signal.signal(signal.SIGALRM, self.previous)


class CleanupTrackedWorld:
    def __init__(self, world, deadline):
        self.world, self.deadline = world, deadline
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.world, name)

    def close(self):
        # Cleanup has its own bounded Docker timeouts and must not be cut off by
        # a rollout alarm after the task outcome was already computed.
        self.deadline.cancel()
        if not self.closed:
            self.world.close()
            self.closed = True


def wait_for_idle(endpoint, model_id, expected_backend, *, seconds=180, poll_seconds=1):
    """Drain an in-flight generation before creating the next task environment.

    This does not cancel the previous request or recover its unknown token cost.
    Like rollout deadlines, the total health wait requires the main thread.
    """
    parsed = urlsplit(endpoint)
    if parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('health endpoint must use loopback HTTP')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('health endpoint cannot contain credentials, queries or fragments')
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError('health poll interval must be finite and positive')

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError('health redirects are forbidden')

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    url = parsed._replace(path='/health', query='', fragment='').geturl()
    try:
        with RolloutDeadline(seconds):
            until = time.monotonic() + seconds
            while True:
                remaining = until - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('model server remained busy')
                with opener.open(url, timeout=min(10, remaining)) as response:
                    raw = response.read(65_537)
                if len(raw) > 65_536:
                    raise ValueError('oversized health response')
                health = json.loads(raw)
                if not isinstance(health, dict) or health.get('ready') is not True or (
                        health.get('model') != model_id or type(health.get('busy')) is not bool):
                    raise ValueError('health response lacks matching ready model and explicit busy state')
                if health.get('backend') != expected_backend:
                    raise BackendMismatch('health backend differs from frozen health record')
                if not health['busy']:
                    return
                time.sleep(min(poll_seconds, max(0, until-time.monotonic())))
    except RolloutTimedOut as exc:
        raise TimeoutError('model server idle wait exhausted') from exc


class CheckedModel:
    def __init__(self, model, expected_backend, counters, on_response):
        self.model = model
        self.expected_backend = expected_backend
        self.counters = counters
        self.on_response = on_response

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, messages, seed):
        self.counters['generation_calls'] += 1
        try:
            response = self.model.generate(messages, seed)
        except BaseException:
            self.counters['generation_calls_without_response'] += 1
            raise
        if any(type(response.get(key)) is not int or response[key] < 0
               for key in ('prompt_tokens', 'completion_tokens')):
            self.counters['generation_calls_with_unusable_usage'] += 1
            raise ValueError('invalid returned token usage')
        for key in ('prompt_tokens', 'completion_tokens'):
            self.counters['returned_response_cost'][key] += response[key]
        matched = response.get('backend') == self.expected_backend
        self.on_response({'seed': seed, 'backend_matches': matched,
                          'backend_sha256': fingerprint(response.get('backend')),
                          'prompt_tokens': response['prompt_tokens'],
                          'completion_tokens': response['completion_tokens']})
        if not matched:
            raise BackendMismatch('inference backend differs from frozen health record')
        return response


def run_tasks(tasks, world_factory, model, *, output, max_steps, rollout_seconds, seed,
              expected_backend, prompt_version=DEFAULT_VERSION, environment_id=None,
              before_task=None):
    tasks = list(tasks)
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError('tasks must be nonempty and unique')
    if max_steps < 1 or not isinstance(expected_backend, dict) or not expected_backend:
        raise ValueError('positive step budget and frozen backend are required')
    RolloutDeadline(rollout_seconds)
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'tasks': [asdict(t) for t in tasks], 'model_id': model.model_id,
        'max_tokens': getattr(model, 'max_tokens', None),
        'temperature': getattr(model, 'temperature', None),
        'seed': seed, 'max_steps': max_steps, 'rollout_seconds': rollout_seconds,
        'prompt_version': prompt_version, 'environment_id': environment_id,
        'expected_backend': expected_backend,
        'backend_idle_check_enabled': before_task is not None,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wll_code_fingerprint': fingerprint({p.name: p.read_text() for p in
                                           (Path(__file__).resolve().parent.parent / 'wll').glob('*.py')}),
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
        unknown = len(tasks)-completed
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
                          'success': None, 'cleanup_verified': None, 'started_epoch': time.time()}

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
                    world = CleanupTrackedWorld(world_factory(task), deadline)
                    record['startup_seconds'] = time.monotonic()-startup
                    with deadline:
                        episode = run_episode(task, lambda world=world: world, checked,
                            max_steps=max_steps, seed=seed, on_step=observe, prompt_version=prompt_version)
                    episode.metadata.update(environment_id=environment_id,
                                             rollout_deadline_seconds=rollout_seconds)
                    episodes.write(json.dumps(episode.to_dict(), ensure_ascii=False)+'\n')
                    episodes.flush()
                    record.update(status='complete', success=episode.success,
                                  episode_hash=fingerprint(episode.to_dict()))
                except RolloutTimedOut as exc:
                    record.update(status='timed_out', error_type=type(exc).__name__)
                except Exception as exc:
                    record['error_type'] = type(exc).__name__
                    abort = world is None or isinstance(exc, BackendMismatch)
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
        manifest['status'] = terminal
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'summary.json', summarize(terminal))
    return summarize(terminal)


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--prompt-version', choices=list(SYSTEMS), default=DEFAULT_VERSION)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('this CPU evaluation client requires CUDA_VISIBLE_DEVICES empty')
    health = json.loads(Path(args.backend_file).read_text())
    if health['model'] != args.model:
        raise ValueError('health record model differs')
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                     args.image], text=True, timeout=15).strip()
    ids = Path(args.task_ids_file).read_text().splitlines()
    tasks = load_tasks(args.data_dir, 'train', image=image, deps_dir=args.deps_dir,
                       include_task_ids=ids)
    by_id = {t.task_id: t for t in tasks}
    ordered = [by_id[task_id] for task_id in ids]
    model = LocalModel(args.endpoint, args.model, max_tokens=args.max_tokens,
                       temperature=args.temperature)
    result = run_tasks(ordered, lambda task: AppWorldEnvironment(task.task_id, args.data_dir,
        args.steps, image, args.deps_dir), model, output=args.output, max_steps=args.steps,
        rollout_seconds=args.rollout_seconds, seed=args.seed, expected_backend=health['backend'],
        prompt_version=args.prompt_version, environment_id=image,
        before_task=lambda task: wait_for_idle(args.endpoint, args.model, health['backend'],
                                              seconds=args.server_idle_seconds))
    print(json.dumps(result), flush=True)
    if not result['valid_for_comparison']:
        raise SystemExit(1)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
