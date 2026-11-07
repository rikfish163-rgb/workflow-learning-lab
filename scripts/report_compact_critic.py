"""Reconstruct critic outcomes and costs from raw files and external expectations."""
import hashlib
import json
from pathlib import Path

from scripts.report_timed_tasks import finite_time
from scripts.run_compact_critic import classify_response
from scripts.run_recovery_trial import inference_signature
from scripts.semantic_repair_compact import critic_messages, project_failure, require
from wll.protocol import fingerprint

REQUIRED = {'source_hash', 'model_id', 'seed', 'max_tokens', 'temperature', 'seconds',
    'expected_backend', 'origin_scenarios', 'max_projection_chars', 'backend_idle_check_enabled',
    'script_sha256', 'proposal_source_sha256'}


def audit_critic(directory, source, expected):
    directory = Path(directory)
    hashes = {}

    def read(name):
        raw = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    require(isinstance(expected, dict) and REQUIRED <= expected.keys(), 'complete expected protocol required')
    manifest, summary = read('manifest.json'), read('summary.json')
    require(all(manifest.get(k) == v for k, v in expected.items()), 'critic manifest differs')
    require(manifest['status'] == summary['status'] and summary['status'] in {'complete', 'incomplete', 'interrupted'},
            'critic terminal status differs')
    require(manifest['maximum_calls'] == 1 and manifest['no_automatic_retry'] is True
            and manifest['automatic_execution'] is False and summary['automatic_execution'] is False
            and summary['accepted_experience'] is False, 'critic authority or attempt limit differs')
    require(expected['source_hash'] == fingerprint(source.to_dict()) and
            inference_signature(source.metadata['backend']) == inference_signature(expected['expected_backend']),
            'source provenance or backend differs')
    projection = project_failure(source, set(expected['origin_scenarios']), max_chars=expected['max_projection_chars'])
    require(read('projection.json') == projection and manifest['projection_sha256'] == fingerprint(projection),
            'projection differs from source')
    request = {'model': expected['model_id'], 'messages': critic_messages(projection), 'seed': expected['seed'],
               'max_tokens': expected['max_tokens'], 'temperature': expected['temperature']}
    require(read('request.json') == request and manifest['request_sha256'] ==
            hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(), 'request differs')
    counters = summary['counters']
    keys = ('generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage')
    require(all(type(counters[k]) is int and 0 <= counters[k] <= 1 for k in keys), 'invalid generation counters')
    raw_path, meter_path, proposal_path = [directory / n for n in
                                         ('raw-response.json', 'response-meter.json', 'proposal.json')]
    cost = {'prompt_tokens': 0, 'completion_tokens': 0}
    classification, validated, returned = None, None, 0
    if raw_path.exists():
        raw = read('raw-response.json')
        require(isinstance(raw, dict) and counters['generation_calls'] == 1
                and counters['generation_calls_without_response'] == 0, 'raw response and counters differ')
        usable = all(type(raw.get(k)) is int and raw[k] >= 0 for k in cost)
        if usable:
            returned = 1
            cost = {k: raw[k] for k in cost}
            matched = raw.get('backend') == expected['expected_backend']
            meter = read('response-meter.json')
            require(meter == {'seed': expected['seed'], 'backend_matches': matched,
                'backend_sha256': fingerprint(raw.get('backend')), **cost}, 'response meter differs')
            require(counters['generation_calls_with_unusable_usage'] == 0, 'usable response counted unusable')
            if matched:
                classification, validated = classify_response(raw, projection)
            if summary['status'] == 'complete':
                require(matched and cost['prompt_tokens'] > 0
                        and 0 < cost['completion_tokens'] <= expected['max_tokens'], 'complete critic has invalid response')
        else:
            require(counters['generation_calls_with_unusable_usage'] == 1 and not meter_path.exists()
                    and not proposal_path.exists() and summary['status'] != 'complete', 'unusable response misreported')
    else:
        require(not meter_path.exists() and not proposal_path.exists() and
                counters['generation_calls_without_response'] == counters['generation_calls']
                and counters['generation_calls_with_unusable_usage'] == 0, 'missing response not accounted')
    require(counters['generation_calls'] == returned + counters['generation_calls_without_response']
            + counters['generation_calls_with_unusable_usage'] and counters['returned_response_cost'] == cost,
            'critic cost differs from raw response')
    if proposal_path.exists():
        require(validated is not None and read('proposal.json') == validated, 'validated proposal differs')
    complete = summary['status'] == 'complete'
    if complete:
        require(returned == 1 and classification is not None and summary['proposal_status'] == classification
                and proposal_path.exists() is (validated is not None), 'critic classification differs')
    else:
        require(summary['proposal_status'] in {'not_attempted', 'unknown', classification},
                'incomplete critic outcome misclassified')
    finite_time(summary['wall_seconds'])
    return {'audit': 'passed', 'status': summary['status'], 'proposal_status': summary['proposal_status'],
        'valid_for_continuation': complete, 'eligible_for_prefix_audit': complete and classification == 'repair',
        'generation_counters': {k: counters[k] for k in keys}, 'returned_response_cost': cost,
        'unknown_additional_response_cost': bool(counters['generation_calls_without_response']
                                                 or counters['generation_calls_with_unusable_usage']),
        'accepted_experience': False, 'automatic_execution': False,
        'request_sha256': manifest['request_sha256'], 'expected_protocol_sha256': fingerprint(expected),
        'reported_wall_seconds': summary['wall_seconds'], 'input_sha256': hashes,
        'scope': 'Raw JSON, citation fidelity and reported usage only. No semantic/cause proof, '
                 'prefix replay, task outcome, server-event accounting or GPU release certified here.'}
