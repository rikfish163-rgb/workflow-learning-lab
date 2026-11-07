"""One bounded critic call, preserving raw outputs and invalid/unknown outcomes."""
import hashlib
import json
import time
from pathlib import Path

from scripts.run_foundation_pilot import digest
from scripts.run_recovery_trial import inference_signature
from scripts.run_timed_tasks import CheckedModel, RolloutDeadline
from scripts.semantic_repair_compact import (
    critic_messages,
    project_failure,
    require,
    validate_proposal,
)
from wll.protocol import fingerprint
from wll.storage import atomic_json, require_space


def classify_response(response, projection):
    if response.get('format_error') is not None or not isinstance(response.get('text'), str) or len(response['text']) > 65536:
        return 'invalid_response_format', None

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('non-finite JSON constant')

    try:
        value = json.loads(response['text'], object_pairs_hook=unique, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return 'invalid_json', None
    try:
        validated = validate_proposal(value, projection)
    except ValueError:
        return 'invalid_schema', None
    return value['status'], validated


def run_critic(source, origin_scenarios, projection, model, *, output, seed, seconds,
               expected_backend, max_projection_chars, before_call=None):
    require(projection == project_failure(source, origin_scenarios, max_chars=max_projection_chars),
            'critic projection differs from visible source')
    require(type(seed) is int and seed >= 0 and isinstance(expected_backend, dict) and expected_backend
            and isinstance(source.metadata.get('backend'), dict)
            and inference_signature(source.metadata['backend']) == inference_signature(expected_backend),
            'source and critic backend differ or invalid seed')
    require(type(model.max_tokens) is int and 0 < model.max_tokens <= 1024, 'invalid critic output budget')
    RolloutDeadline(seconds)
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    request = {'model': model.model_id, 'messages': critic_messages(projection), 'seed': seed,
               'max_tokens': model.max_tokens, 'temperature': model.temperature}
    manifest = {'status': 'running', 'source_hash': fingerprint(source.to_dict()),
        'projection_sha256': fingerprint(projection), 'model_id': model.model_id,
        'max_tokens': model.max_tokens, 'temperature': model.temperature, 'seed': seed, 'seconds': seconds,
        'expected_backend': expected_backend, 'origin_scenarios': sorted(origin_scenarios),
        'max_projection_chars': max_projection_chars, 'backend_idle_check_enabled': before_call is not None,
        'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
        'script_sha256': digest(__file__),
        'proposal_source_sha256': digest(Path(__file__).with_name('semantic_repair_compact.py')),
        'maximum_calls': 1, 'no_automatic_retry': True, 'automatic_execution': False,
        'deadline_scope': 'one generation and parsing; prior backend availability separately bounded'}
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'projection.json', projection)
    atomic_json(output / 'request.json', request)
    counters = {'generation_calls': 0, 'generation_calls_without_response': 0,
        'generation_calls_with_unusable_usage': 0,
        'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0}}
    stats = {'status': 'running', 'proposal_status': 'not_attempted', 'counters': counters,
        'accepted_experience': False, 'automatic_execution': False}
    atomic_json(output / 'summary.json', stats)
    started = time.monotonic()

    class RawJournal:
        def generate(self, messages, seed):
            response = model.generate(messages, seed)
            atomic_json(output / 'raw-response.json', response)
            return response

    checked = CheckedModel(RawJournal(), expected_backend, counters,
                           lambda row: atomic_json(output / 'response-meter.json', row))
    try:
        if before_call is not None:
            before_call()
        stats['proposal_status'] = 'unknown'
        with RolloutDeadline(seconds):
            response = checked.generate(request['messages'], seed=seed)
            require(response['prompt_tokens'] > 0 and 0 < response['completion_tokens'] <= model.max_tokens,
                    'returned usage outside critic budget')
            status, proposal = classify_response(response, projection)
            if proposal is not None:
                atomic_json(output / 'proposal.json', proposal)
            stats.update(status='complete', proposal_status=status)
    except Exception as exc:
        stats.update(status='incomplete', error_type=type(exc).__name__)
    except BaseException as exc:
        stats.update(status='interrupted', error_type=type(exc).__name__)
        raise
    finally:
        stats['wall_seconds'] = time.monotonic() - started
        manifest['status'] = stats['status']
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'summary.json', stats)
    return stats
