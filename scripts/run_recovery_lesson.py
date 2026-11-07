"""One bounded, journaled call to the unchanged experience writer, without execution."""
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from scripts.prepare_recovery_lesson_inputs import lesson_messages
from scripts.run_foundation_pilot import digest, require
from scripts.run_recovery_trial import inference_signature
from scripts.run_timed_tasks import CheckedModel, RolloutDeadline
from wll.experience import ExperienceDraftError, draft_experience
from wll.protocol import fingerprint
from wll.storage import atomic_json, require_space


def classify_lesson(response, source, recovery, *, condition, origin_scenarios, model_id, seed):
    if response.get('format_error') is not None or not isinstance(response.get('text'), str) or len(response['text']) > 65536:
        return 'invalid_response_format', None
    raw = response['text'].strip()
    if raw.startswith('```json\n') and raw.endswith('\n```'):
        raw = raw[8:-4]

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError('non-finite JSON constant')

    try:
        json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return 'invalid_json', None
    expected_messages = lesson_messages(source, recovery, condition=condition, origin_scenarios=origin_scenarios)

    class RecordedResponse:
        def generate(self, messages, seed):
            require(messages == expected_messages, 'writer prompt changed during reconstruction')
            return response

    recorded = RecordedResponse()
    recorded.model_id = model_id
    try:
        experience, cost = draft_experience(source, recovery, recorded, seed=seed)
    except ExperienceDraftError:
        return 'invalid_schema', None
    return 'candidate_unvalidated', {'experience': asdict(experience), 'cost': cost,
        'status': 'candidate_unvalidated', 'condition': condition, 'posthoc_exploratory_only': True,
        'origin_hint_benefit_demonstrated': False, 'accepted': False}


def protocol(source, recovery, model, *, condition, origin_scenarios, seed, seconds,
             expected_backend, backend_idle_check_enabled):
    messages = lesson_messages(source, recovery, condition=condition, origin_scenarios=origin_scenarios)
    require(type(seed) is int and seed >= 0 and isinstance(expected_backend, dict) and expected_backend
        and all(isinstance(e.metadata.get('backend'), dict) and
            inference_signature(e.metadata['backend']) == inference_signature(expected_backend)
            for e in (source, recovery)), 'source and writer backend differ or invalid seed')
    require(type(model.max_tokens) is int and 0 < model.max_tokens <= 1024, 'invalid writer output budget')
    RolloutDeadline(seconds)
    root = Path(__file__).resolve().parent.parent
    return {'source_hash': fingerprint(source.to_dict()), 'recovery_hash': fingerprint(recovery.to_dict()),
        'condition': condition, 'origin_scenarios': sorted(origin_scenarios), 'model_id': model.model_id,
        'max_tokens': model.max_tokens, 'temperature': model.temperature, 'seed': seed, 'seconds': seconds,
        'expected_backend': expected_backend, 'backend_idle_check_enabled': backend_idle_check_enabled,
        'messages_sha256': fingerprint(messages), 'script_sha256': digest(__file__),
        'writer_sha256': digest(root / 'wll/experience.py'),
        'preparation_sha256': digest(root / 'scripts/prepare_recovery_lesson_inputs.py')}


def run_lesson(source, recovery, model, *, output, condition, origin_scenarios, seed, seconds,
               expected_backend, before_call=None):
    expected = protocol(source, recovery, model, condition=condition, origin_scenarios=origin_scenarios,
        seed=seed, seconds=seconds, expected_backend=expected_backend, backend_idle_check_enabled=before_call is not None)
    request = {'model': model.model_id, 'messages': lesson_messages(source, recovery,
        condition=condition, origin_scenarios=origin_scenarios), 'seed': seed,
        'max_tokens': model.max_tokens, 'temperature': model.temperature}
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**expected, 'status': 'running', 'maximum_calls': 1, 'no_automatic_retry': True,
        'automatic_training': False, 'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
        'deadline_scope': 'one generation and parsing; prior backend availability separately bounded'}
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'request.json', request)
    counters = {'generation_calls': 0, 'generation_calls_without_response': 0,
        'generation_calls_with_unusable_usage': 0,
        'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0}}
    stats = {'status': 'running', 'candidate_status': 'not_attempted', 'counters': counters,
             'accepted_experiences': 0, 'automatic_training': False}
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
        stats['candidate_status'] = 'unknown'
        with RolloutDeadline(seconds):
            response = checked.generate(request['messages'], seed=seed)
            require(response['prompt_tokens'] > 0 and 0 < response['completion_tokens'] <= model.max_tokens,
                    'returned usage outside writer budget')
            status, candidate = classify_lesson(response, source, recovery, condition=condition,
                origin_scenarios=origin_scenarios, model_id=model.model_id, seed=seed)
            if candidate is not None:
                atomic_json(output / 'candidate.json', candidate)
            stats.update(status='complete', candidate_status=status)
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
