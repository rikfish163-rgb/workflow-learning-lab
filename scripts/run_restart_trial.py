"""One blind fresh restart after a known failure without an execution-error pivot.

This is extra sampling, not first-attempt success or a within-state repair.
No queue, GPU admission, candidate acceptance, or training is performed here.
"""
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from scripts.run_recovery_trial import (
    CheckedDraftModel,
    DraftFormatError,
    inference_signature,
    require,
)
from scripts.run_timed_tasks import CheckedModel, CleanupTrackedWorld, RolloutDeadline
from wll.experience import ExperienceDraftError, draft_experience
from wll.prompts import SYSTEMS
from wll.protocol import fingerprint
from wll.runner import run_episode
from wll.storage import atomic_json, require_space


def run_restart_trial(source, world_factory, model, *, output, max_steps, rollout_seconds,
                      draft_seconds, restart_seed, draft_seed, expected_backend,
                      environment_id, before_phase=None):
    require(source.task.split == 'train' and source.success is False and source.branch == 'first'
            and source.replayed_steps == 0 and source.experience_id is None,
            'requires a known natural failed first train attempt')
    require(not any(s.observation.startswith('Execution failed') for s in source.steps),
            'explicit execution errors belong to the separately registered prefix protocol')
    require(isinstance(expected_backend, dict) and expected_backend
            and isinstance(source.metadata.get('backend'), dict)
            and inference_signature(source.metadata['backend']) == inference_signature(expected_backend),
            'source and restart inference configurations differ')
    prompt_version = source.metadata.get('prompt_version')
    require(type(max_steps) is int and max_steps > 0 and prompt_version in SYSTEMS
            and source.metadata.get('max_steps') == max_steps
            and source.metadata.get('max_tokens') == model.max_tokens
            and source.metadata.get('temperature') == model.temperature
            and source.metadata.get('environment_id') == environment_id,
            'source and restart budgets, prompt or environment differ')
    require(all(type(s) is int and s >= 0 for s in (restart_seed, draft_seed))
            and restart_seed != source.seed, 'restart requires a different preregistered seed')
    RolloutDeadline(rollout_seconds)
    RolloutDeadline(draft_seconds)
    source_hash = fingerprint(source.to_dict())
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'sampling_mode': 'fresh_restart_v1',
        'classification': 'One extra blind fresh attempt after a natural train failure; not first-attempt performance or a within-state repair.',
        'source_hash': source_hash, 'task_id': source.task.task_id, 'source_seed': source.seed,
        'source_termination': source.termination, 'model_id': model.model_id,
        'max_steps': max_steps, 'max_tokens': model.max_tokens, 'temperature': model.temperature,
        'restart_seed': restart_seed, 'draft_seed': draft_seed, 'rollout_seconds': rollout_seconds,
        'draft_seconds': draft_seconds, 'prompt_version': prompt_version,
        'environment_id': environment_id, 'expected_backend': expected_backend,
        'source_backend': source.metadata['backend'], 'replay_prefix_length': 0,
        'maximum_restart_attempts': 1, 'maximum_draft_attempts': 1, 'no_automatic_retry': True,
        'failed_source_visible_to_restart_actor': False,
        'backend_idle_check_enabled': before_phase is not None,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wll_code_fingerprint': fingerprint({p.name: p.read_text() for p in
            (Path(__file__).resolve().parent.parent / 'wll').glob('*.py')}),
        'deadline_scope': 'rollout/scoring; startup, cleanup and backend waits separately bounded'}
    atomic_json(output / 'manifest.json', manifest)

    def meter():
        return {'generation_calls': 0, 'generation_calls_without_response': 0,
            'generation_calls_with_unusable_usage': 0,
            'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0}}

    stats = {'status': 'running', 'restart_success': None, 'candidate_status': 'not_attempted',
        'replayed_observations': 0, 'new_step_cost': {'steps': 0, 'prompt_tokens': 0, 'completion_tokens': 0},
        'phases': {'restart': meter(), 'draft': meter()}, 'cleanup_verified': False,
        'source_cost': {'prompt_tokens': sum(s.prompt_tokens for s in source.steps),
                        'completion_tokens': source.generated_tokens},
        'cost_note': 'Source costs are historical. Restart and draft response costs are disjoint; '
                     'step and response views overlap. Unreturned/unusable responses retain unknown '
                     'additional cost unless a separate server audit proves a pre-generation rejection.'}
    started = time.monotonic()
    world, phase = None, 'restart'
    atomic_json(output / 'summary.json', stats)
    try:
        with (output / 'steps.jsonl').open('x') as steps, \
                (output / 'responses.jsonl').open('x') as responses, \
                (output / 'episodes.jsonl').open('x') as episodes, \
                (output / 'experiences.jsonl').open('x') as experiences:
            def journal(stream, value):
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
                stream.flush()

            def observe(index, step, replayed):
                require(replayed is False, 'a fresh restart cannot replay source actions')
                journal(steps, {'index': index, 'replayed': False, 'step': asdict(step)})
                stats['new_step_cost']['steps'] += 1
                for key in ('prompt_tokens', 'completion_tokens'):
                    stats['new_step_cost'][key] += getattr(step, key)

            def checked(kind, cls=CheckedModel):
                return cls(model, expected_backend, stats['phases'][kind],
                    lambda value: journal(responses, {'phase': kind, **value}))

            if before_phase is not None:
                before_phase(phase)
            deadline = RolloutDeadline(rollout_seconds)
            world = CleanupTrackedWorld(world_factory(source.task), deadline)
            with deadline:
                require(world.completed() is False, 'fresh restart environment is already completed')
                restart = run_episode(source.task, lambda: world, checked('restart'),
                    max_steps=max_steps, seed=restart_seed, prefix=(), branch='recovery',
                    on_step=observe, prompt_version=prompt_version)
            require(world.closed, 'restart environment remains open')
            stats['cleanup_verified'] = True
            restart.metadata.update(source_hash=source_hash, source_seed=source.seed,
                environment_id=environment_id, sampling_mode='fresh_restart_v1',
                rollout_deadline_seconds=rollout_seconds, failed_source_visible_to_actor=False)
            journal(episodes, restart.to_dict())
            stats['restart_success'] = restart.success
            if not restart.success:
                stats.update(status='complete', candidate_status='not_eligible')
                return stats
            phase = 'draft'
            stats['candidate_status'] = 'unknown'
            atomic_json(output / 'summary.json', stats)
            if before_phase is not None:
                before_phase(phase)
            try:
                with RolloutDeadline(draft_seconds):
                    candidate, cost = draft_experience(source, restart,
                        checked('draft', CheckedDraftModel), seed=draft_seed)
            except (ExperienceDraftError, DraftFormatError) as exc:
                stats['candidate_status'] = ('invalid_schema' if isinstance(exc, ExperienceDraftError)
                                             else 'invalid_format')
            else:
                cost['sampling_mode'] = 'fresh_restart_v1'
                journal(experiences, {'experience': asdict(candidate), 'cost': cost,
                                     'status': 'candidate_unvalidated'})
                stats['candidate_status'] = 'unvalidated'
            stats['status'] = 'complete'
    except Exception as exc:
        stats.update(status='incomplete', error_phase=phase, error_type=type(exc).__name__)
    except BaseException as exc:
        stats.update(status='interrupted', error_phase=phase, error_type=type(exc).__name__)
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
            stats['wall_seconds'] = time.monotonic() - started
            manifest['status'] = stats['status']
            atomic_json(output / 'manifest.json', manifest)
            atomic_json(output / 'summary.json', stats)
    return stats
