# Frozen predecessor: scripts/run_memory_branch.py. Only observed-lesson arm names/imports differ.
"""One bounded fresh branch using an unchanged optional experience projection."""
import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from scripts.run_foundation_pilot import digest, require
from scripts.run_timed_tasks import CheckedModel, CleanupTrackedWorld, RolloutDeadline
from wll.experience import Experience
from wll.prompts import SYSTEMS, initial_messages
from wll.protocol import fingerprint
from wll.runner import run_episode
from wll.storage import atomic_json, require_space

ARMS = ['no_memory', 'login_identifier', 'token_binding']


def initial_actor_messages(task, experience, prompt_version):
    messages = initial_messages(task, prompt_version)
    if experience is not None:
        messages.append({'role': 'user', 'content':
            'A previous training experience may apply; check its conditions:\n' + experience.render()})
    return messages


def protocol(task, experience, model, *, arm, max_steps, seconds, seed, environment_id,
             expected_backend, prompt_version, backend_idle_check_enabled):
    require(arm in ARMS and ((arm == 'no_memory' and experience is None) or
        (arm != 'no_memory' and isinstance(experience, Experience))), 'arm and experience differ')
    require(task.split == 'train' and task.scenario_id == task.task_id.rsplit('_', 1)[0]
        and (experience is None or (experience.source_scenarios and
            task.scenario_id not in experience.source_scenarios)), 'fresh independent train task required')
    require(type(max_steps) is int and 0 < max_steps <= 40 and type(seed) is int and seed >= 0
        and type(model.max_tokens) is int and 0 < model.max_tokens <= 1024
        and type(model.temperature) in (int, float) and math.isfinite(model.temperature) and model.temperature >= 0
        and prompt_version in SYSTEMS and isinstance(environment_id, str) and environment_id
        and isinstance(expected_backend, dict) and expected_backend, 'invalid branch budget, prompt or backend')
    RolloutDeadline(seconds)
    messages = initial_actor_messages(task, experience, prompt_version)
    return {'task': asdict(task), 'candidate': asdict(experience) if experience else None,
        'arm': arm, 'model_id': model.model_id, 'max_steps': max_steps, 'max_tokens': model.max_tokens,
        'temperature': model.temperature, 'seed': seed, 'rollout_seconds': seconds,
        'prompt_version': prompt_version, 'environment_id': environment_id, 'expected_backend': expected_backend,
        'initial_messages_sha256': fingerprint(messages), 'rendered_experience_sha256': fingerprint(experience.render() if experience else None),
        'backend_idle_check_enabled': backend_idle_check_enabled, 'script_sha256': digest(__file__),
        'wll_code_fingerprint': fingerprint({p.name: p.read_text() for p in
            (Path(__file__).resolve().parent.parent / 'wll').glob('*.py')})}


def run_branch(task, experience, world_factory, model, *, output, arm, max_steps, seconds, seed,
               environment_id, expected_backend, prompt_version, before_rollout=None):
    expected = protocol(task, experience, model, arm=arm, max_steps=max_steps, seconds=seconds, seed=seed,
        environment_id=environment_id, expected_backend=expected_backend, prompt_version=prompt_version,
        backend_idle_check_enabled=before_rollout is not None)
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**expected, 'status': 'running', 'no_automatic_retry': True, 'candidate_drafting': False,
        'automatic_training': False, 'maximum_attempts': 1,
        'deadline_scope': 'generation/scoring; startup, cleanup and backend wait separately bounded'}
    atomic_json(output / 'manifest.json', manifest)
    counters = {'generation_calls': 0, 'generation_calls_without_response': 0,
        'generation_calls_with_unusable_usage': 0, 'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0}}
    stats = {'status': 'running', 'outcome': None, 'counters': counters, 'replayed_observations': 0,
        'new_step_cost': {'steps': 0, 'prompt_tokens': 0, 'completion_tokens': 0},
        'execution_attempts': 0, 'cleanup_verified': False, 'accepted_experiences': 0}
    atomic_json(output / 'summary.json', stats)
    world, started = None, time.monotonic()
    streams = {name: (output / (name + '.jsonl')).open('x') for name in
               ('steps', 'requests', 'raw-responses', 'request-errors', 'meters', 'executions', 'episodes')}

    def write(name, row):
        streams[name].write(json.dumps(row, ensure_ascii=False) + '\n')
        streams[name].flush()

    class JournalModel:
        def __getattr__(self, name):
            return getattr(model, name)

        def generate(self, messages, seed):
            index = counters['generation_calls'] - 1
            if index == 0:
                require(fingerprint(messages) == expected['initial_messages_sha256'], 'initial actor prompt differs')
            body = {'model': model.model_id, 'messages': copy.deepcopy(messages), 'seed': seed,
                    'max_tokens': model.max_tokens, 'temperature': model.temperature}
            write('requests', {'index': index, 'body': body,
                'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()})
            try:
                response = model.generate(messages, seed)
                write('raw-responses', {'index': index, 'response': response})
                return response
            except BaseException as exc:
                write('request-errors', {'index': index, 'error_type': type(exc).__name__})
                raise

    class JournalWorld:
        def __init__(self, raw):
            self.raw = raw

        def __getattr__(self, name):
            return getattr(self.raw, name)

        def execute(self, action):
            index = stats['execution_attempts']
            stats['execution_attempts'] += 1
            before = time.monotonic()
            row = {'index': index, 'action': action, 'status': 'unknown'}
            try:
                observation = self.raw.execute(action)
                row.update(status='returned', observation=observation)
                return observation
            except BaseException as exc:
                row['error_type'] = type(exc).__name__
                raise
            finally:
                row['execute_seconds'] = time.monotonic() - before
                write('executions', row)

    def observe(index, step, replayed):
        require(replayed is False, 'utility branch must be a fresh first attempt')
        write('steps', {'index': index, 'replayed': False, 'step': asdict(step)})
        stats['new_step_cost']['steps'] += 1
        for key in ('prompt_tokens', 'completion_tokens'):
            stats['new_step_cost'][key] += getattr(step, key)

    checked = CheckedModel(JournalModel(), expected_backend, counters,
        lambda row: write('meters', {'index': counters['generation_calls'] - 1, **row}))
    try:
        if before_rollout is not None:
            before_rollout()
        deadline = RolloutDeadline(seconds)
        world = CleanupTrackedWorld(JournalWorld(world_factory(task)), deadline)
        with deadline:
            require(world.completed() is False, 'fresh environment is already completed')
            episode = run_episode(task, lambda: world, checked, max_steps=max_steps, seed=seed,
                prefix=(), branch='first', on_step=observe,
                experience=experience.render() if experience else None,
                experience_id=experience.experience_id if experience else None,
                prompt_version=prompt_version)
        require(world.closed and type(episode.success) is bool, 'unknown terminal state or cleanup')
        episode.metadata.update(environment_id=environment_id, arm=arm, rollout_seconds=seconds)
        write('episodes', episode.to_dict())
        stats.update(status='complete', outcome=episode.success, cleanup_verified=True)
    except Exception as exc:
        stats.update(status='incomplete', error_type=type(exc).__name__)
    except BaseException as exc:
        stats.update(status='interrupted', error_type=type(exc).__name__)
        raise
    finally:
        try:
            if world is not None:
                world.close()
                stats['cleanup_verified'] = world.closed
        except BaseException as exc:
            stats.update(status='incomplete', cleanup_error_type=type(exc).__name__)
            raise
        finally:
            for stream in streams.values():
                stream.close()
            stats['wall_seconds'] = time.monotonic() - started
            manifest['status'] = stats['status']
            atomic_json(output / 'manifest.json', manifest)
            atomic_json(output / 'summary.json', stats)
    return stats
