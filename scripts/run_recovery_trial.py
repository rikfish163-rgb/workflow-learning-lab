"""One bounded train recovery and, only after success, one unvalidated lesson.

This component does not enqueue GPU work or open the foundation capability gate.
A separately registered controller must supply audited sources and verified model
health. Historical foundation and recovery entrypoints remain unchanged.
"""
import copy
import hashlib
import time
from dataclasses import asdict
from pathlib import Path

from scripts.run_timed_tasks import CheckedModel, CleanupTrackedWorld, RolloutDeadline
from wll.experience import ExperienceDraftError, draft_experience
from wll.prompts import SYSTEMS
from wll.protocol import fingerprint
from wll.runner import run_episode
from wll.storage import atomic_json, require_space


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inference_signature(backend):
    """Permit an authorized device switch, without dropping other backend checks."""
    value = copy.deepcopy(backend)
    if 'gpu_policy' in value:
        value['gpu_policy'].pop('physical_gpu_index', None)
        value['gpu_policy'].pop('gpu_uuid', None)
    return fingerprint(value)


class DraftFormatError(ValueError):
    pass


class CheckedDraftModel(CheckedModel):
    def generate(self, messages, seed):
        response = super().generate(messages, seed)
        if response.get('format_error'):
            raise DraftFormatError('model reported an unusable draft response')
        return response


def run_recovery_trial(source, prefix_proof, world_factory, model, *, output, max_steps,
                       rollout_seconds, draft_seconds, recovery_seed, draft_seed,
                       expected_backend, environment_id, before_phase=None):
    require(source.task.split == 'train' and source.success is False
            and source.branch == 'first' and source.replayed_steps == 0
            and source.experience_id is None, 'requires a natural failed first train attempt')
    source_hash = fingerprint(source.to_dict())
    length = next((i+1 for i, s in enumerate(source.steps)
                   if s.observation.startswith('Execution failed')), None)
    require(type(max_steps) is int and length is not None and 0 < length < max_steps,
            'first visible error must leave continuation budget')
    prefix = source.steps[:length]
    require(prefix_proof.get('source_hash') == source_hash
            and prefix_proof.get('task_id') == source.task.task_id
            and prefix_proof.get('prefix_length') == length
            and prefix_proof.get('prefix_sha256') == fingerprint([asdict(s) for s in prefix])
            and prefix_proof.get('source_class') == 'failed_trace_with_visible_error'
            and prefix_proof.get('status') == 'verified'
            and prefix_proof.get('prefix_success') is False
            and prefix_proof.get('cleanup_verified') is True, 'unverified or changed failed prefix')
    require(isinstance(expected_backend, dict) and expected_backend
            and isinstance(source.metadata.get('backend'), dict)
            and inference_signature(source.metadata['backend']) == inference_signature(expected_backend),
            'source and recovery inference configurations differ')
    prompt_version = source.metadata.get('prompt_version')
    require(prompt_version in SYSTEMS and source.metadata.get('max_steps') == max_steps
            and source.metadata.get('max_tokens') == model.max_tokens
            and source.metadata.get('temperature') == model.temperature
            and source.metadata.get('environment_id') == environment_id,
            'source and recovery budgets, prompt or environment differ')
    require(all(type(s) is int and s >= 0 for s in (recovery_seed, draft_seed)), 'invalid seeds')
    RolloutDeadline(rollout_seconds)
    RolloutDeadline(draft_seconds)
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'source_hash': source_hash, 'prefix_proof': prefix_proof,
        'task_id': source.task.task_id, 'model_id': model.model_id, 'max_steps': max_steps,
        'max_tokens': model.max_tokens, 'temperature': model.temperature,
        'recovery_seed': recovery_seed, 'draft_seed': draft_seed,
        'rollout_seconds': rollout_seconds, 'draft_seconds': draft_seconds,
        'prompt_version': prompt_version, 'environment_id': environment_id,
        'expected_backend': expected_backend, 'source_backend': source.metadata['backend'],
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wll_code_fingerprint': fingerprint({p.name: p.read_text() for p in
            (Path(__file__).resolve().parent.parent/'wll').glob('*.py')}),
        'classification': 'new recovery from a natural failed train source; not first-attempt success',
        'step_budget_includes_replay': True, 'maximum_recovery_attempts': 1,
        'maximum_draft_attempts': 1, 'no_automatic_retry': True,
        'deadline_scope': 'replay/generation/scoring; startup/cleanup and backend waits separately bounded'}
    atomic_json(output/'manifest.json', manifest)

    def meter():
        return {'generation_calls': 0, 'generation_calls_without_response': 0,
                'generation_calls_with_unusable_usage': 0,
                'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0}}

    stats = {'status': 'running', 'recovery_success': None, 'candidate_status': 'not_attempted',
        'replayed_observations': 0,
        'new_step_cost': {'steps': 0, 'prompt_tokens': 0, 'completion_tokens': 0},
        'phases': {'recovery': meter(), 'draft': meter()}, 'cleanup_verified': False,
        'source_cost': {'prompt_tokens': sum(s.prompt_tokens for s in source.steps),
                        'completion_tokens': source.generated_tokens},
        'cost_note': 'Source tokens are historical. Replay token/elapsed fields in step records '
                     'belong to source actions. Only new steps count toward new_step_cost. '
                     'Phase response costs are disjoint; step and response views overlap. '
                     'Unreturned or unusable responses have unknown additional token cost.'}
    started = time.monotonic()
    world, phase = None, 'recovery'
    atomic_json(output/'summary.json', stats)
    try:
        import json

        with (output/'steps.jsonl').open('x') as steps, \
                (output/'responses.jsonl').open('x') as responses, \
                (output/'episodes.jsonl').open('x') as episodes, \
                (output/'experiences.jsonl').open('x') as experiences:
            def journal(stream, value):
                stream.write(json.dumps(value, ensure_ascii=False)+'\n')
                stream.flush()

            def observe(index, step, replayed):
                journal(steps, {'index': index, 'replayed': replayed, 'step': asdict(step)})
                if replayed:
                    stats['replayed_observations'] += 1
                else:
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
                recovery = run_episode(source.task, lambda: world, checked('recovery'),
                    max_steps=max_steps, seed=recovery_seed, prefix=prefix, branch='recovery',
                    on_step=observe, prompt_version=prompt_version)
            require(world.closed, 'recovery environment remains open')
            stats['cleanup_verified'] = True
            recovery.metadata.update(source_hash=source_hash, environment_id=environment_id,
                rollout_deadline_seconds=rollout_seconds,
                pivot_rule='first_visible_error_verified_failed_prefix_v1',
                prefix_proof_hash=fingerprint(prefix_proof))
            journal(episodes, recovery.to_dict())
            stats['recovery_success'] = recovery.success
            if not recovery.success:
                stats.update(status='complete', candidate_status='not_eligible')
                return stats
            phase = 'draft'
            stats['candidate_status'] = 'unknown'
            atomic_json(output/'summary.json', stats)
            if before_phase is not None:
                before_phase(phase)
            try:
                with RolloutDeadline(draft_seconds):
                    candidate, cost = draft_experience(source, recovery,
                        checked('draft', CheckedDraftModel), seed=draft_seed)
            except (ExperienceDraftError, DraftFormatError) as exc:
                stats['candidate_status'] = ('invalid_schema' if isinstance(exc, ExperienceDraftError)
                                             else 'invalid_format')
            else:
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
            stats['wall_seconds'] = time.monotonic()-started
            manifest['status'] = stats['status']
            atomic_json(output/'manifest.json', manifest)
            atomic_json(output/'summary.json', stats)
    return stats
