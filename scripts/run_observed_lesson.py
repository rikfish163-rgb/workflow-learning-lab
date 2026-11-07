"""One costed writer call over an already verified historical self-correction.

This component does not enqueue GPU work, replay actions or train. Its caller
must separately bind the prepared source/proof, generation settings and service.
The original recovery-writer scripts remain frozen for the active experiment.
"""
import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from scripts.prepare_observed_lesson_inputs import observed_messages
from scripts.report_timed_tasks import finite_time
from scripts.run_foundation_pilot import digest, require
from scripts.run_recovery_trial import inference_signature
from scripts.run_timed_tasks import CheckedModel, RolloutDeadline
from wll.experience import ExperienceDraftError, draft_experience
from wll.protocol import fingerprint
from wll.storage import atomic_json, require_space


def request_digest(request):
    """Match the frozen ChatEngine generation-event serialization, including Unicode."""
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def protocol(source, proof, model, *, origin_scenarios, seed, seconds,
             expected_backend, backend_idle_check_enabled):
    prefix, messages = observed_messages(source, proof, origin_scenarios=origin_scenarios)
    require(type(seed) is int and seed >= 0 and isinstance(expected_backend, dict)
        and expected_backend and isinstance(source.metadata.get('backend'), dict)
        and inference_signature(source.metadata['backend']) == inference_signature(expected_backend),
        'source and writer inference settings differ or invalid seed')
    require(isinstance(model.model_id, str) and model.model_id
        and type(model.max_tokens) is int and 0 < model.max_tokens <= 1024
        and type(model.temperature) in (int, float) and math.isfinite(model.temperature)
        and 0 <= model.temperature <= 2, 'invalid writer generation settings')
    RolloutDeadline(seconds)
    root = Path(__file__).resolve().parent.parent
    return {'condition': 'observed_self_correction', 'source_hash': fingerprint(source.to_dict()),
        'proof_hash': fingerprint(proof), 'derived_prefix_hash': fingerprint(prefix.to_dict()),
        'origin_scenarios': sorted(origin_scenarios), 'model_id': model.model_id,
        'max_tokens': model.max_tokens, 'temperature': model.temperature, 'seed': seed,
        'seconds': seconds, 'expected_backend': copy.deepcopy(expected_backend),
        'backend_idle_check_enabled': backend_idle_check_enabled,
        'messages_sha256': fingerprint(messages), 'script_sha256': digest(__file__),
        'writer_sha256': digest(root / 'wll/experience.py'),
        'preparation_sha256': digest(root / 'scripts/prepare_observed_lesson_inputs.py'),
        'new_recovery_successes': 0, 'new_environment_executions': 0,
        'content_certified': False}


def classify(response, prefix, source, messages, *, model_id, seed):
    if (response.get('format_error') is not None or not isinstance(response.get('text'), str)
            or len(response['text']) > 65536):
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

    expected_seed = seed

    class Recorded:
        def generate(self, actual, seed):
            require(actual == messages and seed == expected_seed, 'reconstructed writer request differs')
            return response

    recorded = Recorded()
    recorded.model_id = model_id
    try:
        experience, cost = draft_experience(prefix, source, recorded, seed=seed)
    except ExperienceDraftError:
        return 'invalid_schema', None
    require(cost['evidence_kind'] == 'observed_self_correction', 'wrong recovery provenance')
    return 'candidate_unvalidated', {'experience': asdict(experience), 'cost': cost,
        'status': 'candidate_unvalidated', 'observed_self_correction': True,
        'posthoc_exploratory_only': True, 'content_certified': False, 'accepted': False}


def run_observed_lesson(source, proof, model, *, output, origin_scenarios, seed, seconds,
                        expected_backend, before_call=None):
    expected = protocol(source, proof, model, origin_scenarios=origin_scenarios, seed=seed,
        seconds=seconds, expected_backend=expected_backend, backend_idle_check_enabled=before_call is not None)
    prefix, messages = observed_messages(source, proof, origin_scenarios=origin_scenarios)
    request = {'model': model.model_id, 'messages': messages, 'seed': seed,
        'max_tokens': model.max_tokens, 'temperature': model.temperature}
    output = Path(output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**expected, 'status': 'running', 'maximum_calls': 1, 'no_automatic_retry': True,
        'automatic_training': False, 'request_sha256': request_digest(request)}
    counters = {'generation_calls': 0, 'generation_calls_without_response': 0,
        'generation_calls_with_unusable_usage': 0,
        'returned_response_cost': {'prompt_tokens': 0, 'completion_tokens': 0}}
    summary = {'status': 'running', 'candidate_status': 'not_attempted', 'counters': counters,
        'accepted_experiences': 0, 'automatic_training': False, 'new_environment_executions': 0,
        'new_recovery_successes': 0, 'content_certified': False}
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'request.json', request)
    atomic_json(output / 'summary.json', summary)
    started = time.monotonic()

    class Journal:
        def generate(self, messages, seed):
            response = model.generate(messages, seed)
            atomic_json(output / 'raw-response.json', response)
            # Keep the exact raw value. The existing cost meter must classify a
            # non-object return as unusable usage, never as a zero-cost success.
            return response if isinstance(response, dict) else {}

    checked = CheckedModel(Journal(), expected['expected_backend'], counters,
        lambda row: atomic_json(output / 'response-meter.json', row))
    try:
        if before_call is not None:
            before_call()
        summary['candidate_status'] = 'unknown'
        with RolloutDeadline(seconds):
            response = checked.generate(messages, seed=seed)
            require(response['prompt_tokens'] > 0 and 0 < response['completion_tokens'] <= model.max_tokens,
                'returned usage outside writer budget')
            status, candidate = classify(response, prefix, source, messages, model_id=model.model_id, seed=seed)
            if candidate is not None:
                atomic_json(output / 'candidate.json', candidate)
            summary.update(status='complete', candidate_status=status)
    except Exception as exc:
        summary.update(status='incomplete', error_type=type(exc).__name__)
    except BaseException as exc:
        summary.update(status='interrupted', error_type=type(exc).__name__)
        raise
    finally:
        summary['wall_seconds'] = time.monotonic() - started
        manifest['status'] = summary['status']
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'summary.json', summary)
    return summary


def audit_observed_lesson(directory, source, proof, expected):
    directory, hashes = Path(directory), {}

    def read(name):
        raw = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    model = SimpleNamespace(**{k: expected[k] for k in ('model_id', 'max_tokens', 'temperature')})
    args = {k: expected[k] for k in ('origin_scenarios', 'seed', 'seconds',
        'expected_backend', 'backend_idle_check_enabled')}
    require(expected == protocol(source, proof, model, **args), 'external observed-source protocol differs')
    prefix, messages = observed_messages(source, proof, origin_scenarios=expected['origin_scenarios'])
    manifest, summary = read('manifest.json'), read('summary.json')
    require(all(manifest.get(k) == v for k, v in expected.items())
        and manifest['status'] == summary['status']
        and summary['status'] in {'complete', 'incomplete', 'interrupted'}, 'writer manifest/status differs')
    require(manifest['maximum_calls'] == 1 and manifest['no_automatic_retry'] is True
        and manifest['automatic_training'] is summary['automatic_training'] is False
        and summary['accepted_experiences'] == summary['new_recovery_successes']
        == summary['new_environment_executions'] == 0 and summary['content_certified'] is False,
        'writer authority or historical source classification differs')
    request = {'model': expected['model_id'], 'messages': messages, 'seed': expected['seed'],
        'max_tokens': expected['max_tokens'], 'temperature': expected['temperature']}
    require(read('request.json') == request and manifest['request_sha256'] == request_digest(request),
        'writer request differs')
    counters = summary['counters']
    keys = ('generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage')
    require(all(type(counters[k]) is int and 0 <= counters[k] <= 1 for k in keys), 'invalid writer counters')
    raw_path, meter_path, candidate_path = [directory / n for n in
        ('raw-response.json', 'response-meter.json', 'candidate.json')]
    cost, returned, classification, candidate = {'prompt_tokens': 0, 'completion_tokens': 0}, 0, None, None
    complete = summary['status'] == 'complete'
    if raw_path.exists():
        raw = read('raw-response.json')
        require(counters['generation_calls'] == 1
            and counters['generation_calls_without_response'] == 0, 'raw response counters differ')
        usable = isinstance(raw, dict) and all(type(raw.get(k)) is int and raw[k] >= 0 for k in cost)
        if usable:
            returned, cost = 1, {k: raw[k] for k in cost}
            matched = raw.get('backend') == expected['expected_backend']
            require(read('response-meter.json') == {'seed': expected['seed'], 'backend_matches': matched,
                'backend_sha256': fingerprint(raw.get('backend')), **cost}
                and counters['generation_calls_with_unusable_usage'] == 0, 'response meter differs')
            if matched:
                classification, candidate = classify(raw, prefix, source, messages,
                    model_id=expected['model_id'], seed=expected['seed'])
            if complete:
                require(matched and cost['prompt_tokens'] > 0
                    and 0 < cost['completion_tokens'] <= expected['max_tokens'], 'invalid complete response')
        else:
            require(counters['generation_calls_with_unusable_usage'] == 1 and not meter_path.exists()
                and not candidate_path.exists() and not complete, 'unusable usage misreported')
    else:
        require(not meter_path.exists() and not candidate_path.exists()
            and counters['generation_calls_without_response'] == counters['generation_calls']
            and counters['generation_calls_with_unusable_usage'] == 0, 'missing response not accounted')
    require(counters['generation_calls'] == returned + counters['generation_calls_without_response']
        + counters['generation_calls_with_unusable_usage'] and counters['returned_response_cost'] == cost,
        'writer cost differs from raw response')
    if candidate_path.exists():
        require(candidate is not None and read('candidate.json') == candidate, 'candidate differs from raw response')
    if complete:
        require(returned == 1 and classification is not None and summary['candidate_status'] == classification
            and candidate_path.exists() is (candidate is not None), 'writer classification differs')
    else:
        require(summary['candidate_status'] in {'not_attempted', 'unknown', classification}, 'unknown misclassified')
    finite_time(summary['wall_seconds'])
    return {'audit': 'passed', 'status': summary['status'], 'candidate_status': summary['candidate_status'],
        'condition': 'observed_self_correction', 'valid_for_continuation': complete,
        'unvalidated_candidates': int(complete and candidate is not None),
        'generation_counters': {k: counters[k] for k in keys}, 'returned_response_cost': cost,
        'unknown_additional_response_cost': bool(counters['generation_calls_without_response']
            or counters['generation_calls_with_unusable_usage']), 'accepted_experiences': 0,
        'automatic_training': False, 'new_recovery_successes': 0, 'new_environment_executions': 0,
        'content_certified': False, 'request_sha256': request_digest(request),
        'expected_protocol_sha256': fingerprint(expected), 'reported_wall_seconds': summary['wall_seconds'],
        'input_sha256': hashes, 'scope': 'One writer request over an observed historical self-correction. '
            'Syntax, provenance and reported cost only; not new recovery, semantic certification, '
            'independent utility, service-event accounting or GPU release evidence.'}
