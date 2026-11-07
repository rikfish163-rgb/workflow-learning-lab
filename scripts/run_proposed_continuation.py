"""One bounded branch from a verified origin prefix, with an optional unverified hint."""
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from scripts.run_foundation_pilot import digest, require
from scripts.run_recovery_trial import inference_signature
from scripts.run_timed_tasks import CheckedModel, CleanupTrackedWorld, RolloutDeadline
from scripts.semantic_repair_proposal import project_failure, validate_proposal
from wll.prompts import SYSTEMS, initial_messages
from wll.protocol import fingerprint
from wll.runner import run_episode
from wll.storage import atomic_json, require_space


def proposal_hint(proposal):
    value = proposal['proposal']
    return ('Unverified repair proposal derived from a prior failed attempt on this training task. '
        'It may be wrong. Check it against the current task, visible state and API documentation before acting. '
        'It is not accepted experience.\n' + json.dumps({k: value[k] for k in
            ('hypothesis', 'repair_plan', 'uncertainty')}, ensure_ascii=False))


def initial_continuation_messages(source, proposal, condition):
    require(condition in ('no_hint', 'proposal_hint'), 'unknown continuation condition')
    messages = initial_messages(source.task, source.metadata['prompt_version'])
    for step in source.steps[:proposal['proposal']['prefix_length']]:
        messages += [{'role': 'assistant', 'content': step.raw_response or step.action},
                     {'role': 'user', 'content': step.observation}]
    if condition == 'proposal_hint':
        messages.append({'role': 'user', 'content':
            'A previous training experience may apply; check its conditions:\n' + proposal_hint(proposal)})
    return messages


def protocol(source, proposal, proof, model, *, condition, max_steps, seconds, seed,
             environment_id, expected_backend, origin_scenarios, backend_idle_check_enabled):
    projection = project_failure(source, origin_scenarios, max_chars=32768)
    require(proposal == validate_proposal(proposal['proposal'], projection)
        and proposal['proposal']['status'] == 'repair', 'proposal differs from visible source')
    length = proposal['proposal']['prefix_length']
    require(proof.get('audit') == 'passed' and proof.get('eligible_for_continuation') is True
        and proof.get('cleanup_verified') is True and proof.get('prefix_success') is False
        and proof.get('prefix_completed') is False and proof.get('LLM_calls') == proof.get('new_actions') == 0
        and proof['source_hash'] == fingerprint(source.to_dict()) and proof['proposal_sha256'] == fingerprint(proposal)
        and proof['prefix_length'] == length and proof['environment_id'] == environment_id
        and proof['prefix_sha256'] == fingerprint([asdict(s) for s in source.steps[:length]]),
        'verified unsuccessful prefix required')
    require(type(max_steps) is int and length < max_steps <= 40
        and type(seed) is int and seed >= 0 and seed != source.seed
        and source.metadata.get('prompt_version') in SYSTEMS, 'invalid continuation budget, seed or prompt')
    for key, value in {'max_steps': max_steps, 'max_tokens': model.max_tokens,
                       'temperature': model.temperature, 'environment_id': environment_id}.items():
        require(source.metadata.get(key) == value, 'source setting differs: ' + key)
    require(isinstance(expected_backend, dict) and expected_backend
        and inference_signature(source.metadata['backend']) == inference_signature(expected_backend),
        'source and continuation backend differ')
    RolloutDeadline(seconds)
    messages = initial_continuation_messages(source, proposal, condition)
    return {'source_hash': fingerprint(source.to_dict()), 'proposal_sha256': fingerprint(proposal),
        'prefix_proof_sha256': fingerprint(proof), 'prefix_length': length, 'condition': condition,
        'origin_scenarios': sorted(origin_scenarios), 'model_id': model.model_id, 'max_steps': max_steps,
        'max_tokens': model.max_tokens, 'temperature': model.temperature, 'seed': seed,
        'rollout_seconds': seconds, 'prompt_version': source.metadata['prompt_version'],
        'environment_id': environment_id, 'expected_backend': expected_backend,
        'initial_messages_sha256': fingerprint(messages),
        'hint_sha256': fingerprint(proposal_hint(proposal) if condition == 'proposal_hint' else None),
        'backend_idle_check_enabled': backend_idle_check_enabled, 'script_sha256': digest(__file__),
        'wll_code_fingerprint': fingerprint({p.name: p.read_text() for p in
            (Path(__file__).resolve().parent.parent / 'wll').glob('*.py')})}


def run_continuation(source, proposal, proof, world_factory, model, *, output, condition,
                     max_steps, seconds, seed, environment_id, expected_backend, origin_scenarios,
                     before_rollout=None):
    expected = protocol(source, proposal, proof, model, condition=condition, max_steps=max_steps,
        seconds=seconds, seed=seed, environment_id=environment_id, expected_backend=expected_backend,
        origin_scenarios=origin_scenarios, backend_idle_check_enabled=before_rollout is not None)
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**expected, 'status': 'running', 'no_automatic_retry': True, 'candidate_drafting': False,
        'posthoc_exploratory_only': True, 'maximum_attempts': 1,
        'deadline_scope': 'replay/generation/scoring; startup, cleanup and backend wait separately bounded'}
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
        write('steps', {'index': index, 'replayed': replayed, 'step': asdict(step)})
        if replayed:
            stats['replayed_observations'] += 1
        else:
            stats['new_step_cost']['steps'] += 1
            for key in ('prompt_tokens', 'completion_tokens'):
                stats['new_step_cost'][key] += getattr(step, key)

    checked = CheckedModel(JournalModel(), expected_backend, counters,
        lambda row: write('meters', {'index': counters['generation_calls'] - 1, **row}))
    try:
        if before_rollout is not None:
            before_rollout()
        deadline = RolloutDeadline(seconds)
        world = CleanupTrackedWorld(JournalWorld(world_factory(source.task)), deadline)
        with deadline:
            episode = run_episode(source.task, lambda: world, checked, max_steps=max_steps, seed=seed,
                prefix=source.steps[:expected['prefix_length']], branch='recovery', on_step=observe,
                experience=proposal_hint(proposal) if condition == 'proposal_hint' else None,
                prompt_version=expected['prompt_version'])
        require(world.closed and type(episode.success) is bool, 'unknown terminal state or cleanup')
        episode.metadata.update(source_hash=expected['source_hash'], proposal_sha256=expected['proposal_sha256'],
            prefix_proof_sha256=expected['prefix_proof_sha256'], environment_id=environment_id,
            condition=condition, posthoc_exploratory_only=True, rollout_seconds=seconds)
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
