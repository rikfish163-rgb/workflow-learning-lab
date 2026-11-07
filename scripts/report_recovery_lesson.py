"""Reconstruct a writer request, response, candidate and cost from externally bound traces."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.prepare_recovery_lesson_inputs import lesson_messages
from scripts.report_timed_tasks import finite_time
from scripts.run_foundation_pilot import require
from scripts.run_recovery_lesson import classify_lesson, protocol
from wll.protocol import fingerprint


def audit_lesson(directory, source, recovery, expected):
    directory, hashes = Path(directory), {}

    def read(name):
        raw = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    args = {k: expected[k] for k in ('condition', 'origin_scenarios', 'seed', 'seconds', 'expected_backend',
                                     'backend_idle_check_enabled')}
    model = SimpleNamespace(**{k: expected[k] for k in ('model_id', 'max_tokens', 'temperature')})
    require(expected == protocol(source, recovery, model, **args), 'external writer protocol differs')
    manifest, summary = read('manifest.json'), read('summary.json')
    require(all(manifest.get(k) == v for k, v in expected.items())
        and manifest['status'] == summary['status']
        and summary['status'] in {'complete', 'incomplete', 'interrupted'}, 'writer manifest/status differs')
    require(manifest['maximum_calls'] == 1 and manifest['no_automatic_retry'] is True
        and manifest['automatic_training'] is summary['automatic_training'] is False
        and summary['accepted_experiences'] == 0, 'writer authority or attempt limit differs')
    request = {'model': expected['model_id'], 'messages': lesson_messages(source, recovery,
        condition=expected['condition'], origin_scenarios=expected['origin_scenarios']),
        'seed': expected['seed'], 'max_tokens': expected['max_tokens'], 'temperature': expected['temperature']}
    require(read('request.json') == request and manifest['request_sha256'] ==
        hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(), 'writer request differs')
    counters = summary['counters']
    keys = ('generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage')
    require(all(type(counters[k]) is int and 0 <= counters[k] <= 1 for k in keys), 'invalid writer counters')
    raw_path, meter_path, candidate_path = [directory / name for name in
        ('raw-response.json', 'response-meter.json', 'candidate.json')]
    cost = {'prompt_tokens': 0, 'completion_tokens': 0}
    classification, candidate, returned = None, None, 0
    if raw_path.exists():
        raw = read('raw-response.json')
        require(isinstance(raw, dict) and counters['generation_calls'] == 1
            and counters['generation_calls_without_response'] == 0, 'raw writer response counters differ')
        usable = all(type(raw.get(k)) is int and raw[k] >= 0 for k in cost)
        if usable:
            returned, cost = 1, {k: raw[k] for k in cost}
            matched = raw.get('backend') == expected['expected_backend']
            require(read('response-meter.json') == {'seed': expected['seed'], 'backend_matches': matched,
                'backend_sha256': fingerprint(raw.get('backend')), **cost}
                and counters['generation_calls_with_unusable_usage'] == 0, 'writer response meter differs')
            if matched:
                classification, candidate = classify_lesson(raw, source, recovery, condition=expected['condition'],
                    origin_scenarios=expected['origin_scenarios'], model_id=expected['model_id'], seed=expected['seed'])
            if summary['status'] == 'complete':
                require(matched and cost['prompt_tokens'] > 0
                    and 0 < cost['completion_tokens'] <= expected['max_tokens'], 'invalid complete writer response')
        else:
            require(counters['generation_calls_with_unusable_usage'] == 1 and not meter_path.exists()
                and not candidate_path.exists() and summary['status'] != 'complete', 'unusable usage misreported')
    else:
        require(not meter_path.exists() and not candidate_path.exists() and
            counters['generation_calls_without_response'] == counters['generation_calls']
            and counters['generation_calls_with_unusable_usage'] == 0, 'missing response not accounted')
    require(counters['generation_calls'] == returned + counters['generation_calls_without_response']
        + counters['generation_calls_with_unusable_usage'] and counters['returned_response_cost'] == cost,
        'writer cost differs from raw response')
    if candidate_path.exists():
        require(candidate is not None and read('candidate.json') == candidate, 'candidate differs from raw response')
    complete = summary['status'] == 'complete'
    if complete:
        require(returned == 1 and classification is not None and summary['candidate_status'] == classification
            and candidate_path.exists() is (candidate is not None), 'writer classification differs')
    else:
        require(summary['candidate_status'] in {'not_attempted', 'unknown', classification}, 'unknown result misclassified')
    finite_time(summary['wall_seconds'])
    return {'audit': 'passed', 'status': summary['status'], 'candidate_status': summary['candidate_status'],
        'condition': expected['condition'], 'valid_for_continuation': complete,
        'unvalidated_candidates': int(complete and candidate is not None),
        'generation_counters': {k: counters[k] for k in keys}, 'returned_response_cost': cost,
        'unknown_additional_response_cost': bool(counters['generation_calls_without_response']
            or counters['generation_calls_with_unusable_usage']), 'accepted_experiences': 0, 'automatic_training': False,
        'request_sha256': manifest['request_sha256'], 'expected_protocol_sha256': fingerprint(expected),
        'reported_wall_seconds': summary['wall_seconds'], 'input_sha256': hashes,
        'scope': 'Candidate syntax, provenance and reported cost only. No independent utility, semantic '
            'correctness, service-event accounting or GPU release certified here.'}
