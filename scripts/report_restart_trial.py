"""Reconstruct a blind restart and optional candidate from immutable raw journals."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from scripts.report_timed_tasks import count, finite_time, require
from scripts.run_recovery_trial import inference_signature
from wll.protocol import fingerprint

REQUIRED = {'source_hash', 'source_seed', 'source_termination', 'task_id', 'model_id',
    'max_steps', 'max_tokens', 'temperature', 'restart_seed', 'draft_seed', 'rollout_seconds',
    'draft_seconds', 'prompt_version', 'environment_id', 'expected_backend', 'source_backend',
    'script_sha256', 'wll_code_fingerprint', 'backend_idle_check_enabled'}


def audit_trial(directory, source, expected):
    directory, hashes = Path(directory), {}

    def read(name, lines=False):
        raw = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines()] if lines else json.loads(raw)

    require(isinstance(expected, dict) and REQUIRED <= expected.keys(), 'complete external restart protocol required')
    manifest, summary = read('manifest.json'), read('summary.json')
    require(all(manifest.get(k) == v for k, v in expected.items()), 'restart manifest differs from protocol')
    require(manifest['status'] == summary['status'] and summary['status'] in
            {'complete', 'incomplete', 'interrupted'}, 'restart is not terminal')
    require(source.success is False and source.task.split == 'train' and source.branch == 'first'
            and source.replayed_steps == 0 and source.experience_id is None
            and not any(s.observation.startswith('Execution failed') for s in source.steps)
            and fingerprint(source.to_dict()) == expected['source_hash']
            and source.task.task_id == expected['task_id'] and source.seed == expected['source_seed']
            and source.termination == expected['source_termination'], 'source provenance differs')
    require(source.metadata['backend'] == expected['source_backend']
            and inference_signature(expected['source_backend']) == inference_signature(expected['expected_backend']),
            'source and restart backend differ')
    for key in ('max_steps', 'max_tokens', 'temperature', 'prompt_version', 'environment_id'):
        require(source.metadata[key] == expected[key], 'source setting differs: ' + key)
    require(manifest['sampling_mode'] == 'fresh_restart_v1' and manifest['replay_prefix_length'] == 0
            and manifest['maximum_restart_attempts'] == manifest['maximum_draft_attempts'] == 1
            and manifest['failed_source_visible_to_restart_actor'] is False
            and manifest['no_automatic_retry'] is True
            and expected['restart_seed'] != expected['source_seed'], 'restart exposure or attempt rule differs')
    steps, responses = read('steps.jsonl', True), read('responses.jsonl', True)
    episodes, candidates = read('episodes.jsonl', True), read('experiences.jsonl', True)
    require(len(steps) <= expected['max_steps'] and len(episodes) <= 1 and len(candidates) <= 1,
            'restart exceeds attempt or step budget')
    new_cost = {'steps': len(steps), 'prompt_tokens': 0, 'completion_tokens': 0}
    for index, row in enumerate(steps):
        require(type(row['index']) is int and row['index'] == index and row['replayed'] is False,
                'step order or fresh-restart classification differs')
        finite_time(row['step']['elapsed_seconds'])
        for key in ('prompt_tokens', 'completion_tokens'):
            new_cost[key] += count(row['step'][key])
    phase_rows = {'restart': [], 'draft': []}
    phase_cost = {p: {'prompt_tokens': 0, 'completion_tokens': 0} for p in phase_rows}
    for row in responses:
        phase = row['phase']
        require(phase in phase_rows and (phase != 'restart' or not phase_rows['draft']), 'response phase order differs')
        group = phase_rows[phase]
        seed = expected['draft_seed'] if phase == 'draft' else expected['restart_seed'] + len(group)
        require(type(row['seed']) is int and row['seed'] == seed, 'response seed differs')
        require(type(row['backend_matches']) is bool and row['backend_matches'] ==
                (row['backend_sha256'] == fingerprint(expected['expected_backend']))
                and (not group or group[-1]['backend_matches']), 'response backend evidence differs')
        for key in ('prompt_tokens', 'completion_tokens'):
            phase_cost[phase][key] += count(row[key])
        group.append(row)
    require(len(steps) <= len(phase_rows['restart']), 'action without a response')
    for row, response in zip(steps, phase_rows['restart'], strict=False):
        require(response['backend_matches'] and all(row['step'][k] == response[k]
                for k in ('prompt_tokens', 'completion_tokens')), 'step and response cost differ')
    counters = {}
    for phase, rows in phase_rows.items():
        meter = summary['phases'][phase]
        keys = ('generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage')
        counters[phase] = {k: count(meter[k]) for k in keys}
        require(meter['generation_calls'] == len(rows) + meter[keys[1]] + meter[keys[2]]
                and meter['returned_response_cost'] == phase_cost[phase]
                and meter['generation_calls'] <= (1 if phase == 'draft' else expected['max_steps']),
                'phase accounting or attempt limit differs')
    outcome = episodes[0]['success'] if episodes else None
    require(outcome is None or type(outcome) is bool, 'restart outcome must be boolean or unknown')
    if episodes:
        e = episodes[0]
        require(e['task'] == asdict(source.task) and e['model_id'] == expected['model_id']
                and e['seed'] == expected['restart_seed'] and e['branch'] == 'recovery'
                and e['experience_id'] is None and e['replayed_steps'] == 0
                and e['steps'] == [r['step'] for r in steps], 'episode and restart journal differ')
        for key in ('max_steps', 'max_tokens', 'temperature', 'prompt_version', 'environment_id',
                    'source_hash', 'source_seed'):
            require(e['metadata'].get(key) == expected[key], 'episode setting differs: ' + key)
        require(e['metadata']['sampling_mode'] == 'fresh_restart_v1'
                and e['metadata']['failed_source_visible_to_actor'] is False
                and e['metadata']['rollout_deadline_seconds'] == expected['rollout_seconds'],
                'episode mode, exposure or deadline differs')
        if steps:
            require(e['metadata'].get('backend') == expected['expected_backend'], 'episode backend differs')
        finite_time(e['elapsed_seconds'])
    candidate_status = summary['candidate_status']
    if outcome is not True:
        require(not candidates and counters['draft']['generation_calls'] == 0
                and candidate_status == ('not_attempted' if outcome is None else 'not_eligible'),
                'failed or unverified restart cannot draft')
    else:
        require(candidate_status in {'unvalidated', 'invalid_schema', 'invalid_format', 'unknown'}, 'invalid draft status')
    require(bool(candidates) == (candidate_status == 'unvalidated'), 'candidate artifact/status differs')
    if candidates:
        record, e = candidates[0], episodes[0]
        candidate, cost = record['experience'], record['cost']
        fields = {k: candidate[k] for k in ('failure_mode', 'applies_when', 'avoid_when', 'rule')}
        trace_hashes = [expected['source_hash'], fingerprint(e)]
        require(set(candidate) == set(fields) | {'experience_id', 'source_scenarios', 'source_episode_hashes'}
                and all(isinstance(v, str) and 1 <= len(v) <= 1500 for v in fields.values())
                and candidate['source_episode_hashes'] == trace_hashes
                and candidate['source_scenarios'] == [source.task.scenario_id]
                and candidate['experience_id'] == fingerprint([fields, trace_hashes])[:16]
                and record['status'] == 'candidate_unvalidated', 'candidate provenance differs')
        require(len(phase_rows['draft']) == 1 and phase_rows['draft'][0]['backend_matches']
                and all(cost[k] == phase_cost['draft'][k] for k in ('prompt_tokens', 'completion_tokens'))
                and cost['model_id'] == expected['model_id'] and cost['evidence_kind'] == 'new_recovery'
                and cost['sampling_mode'] == 'fresh_restart_v1'
                and cost['trace_projection'] == 'trigger_tail_recovery_head_tail_v1', 'candidate cost or mode differs')
    require(summary['restart_success'] is outcome and summary['replayed_observations'] == 0
            and summary['new_step_cost'] == new_cost and summary['source_cost'] == {
                'prompt_tokens': sum(s.prompt_tokens for s in source.steps),
                'completion_tokens': source.generated_tokens}, 'summary differs from raw evidence')
    if summary['status'] == 'complete':
        require(type(outcome) is bool and summary['cleanup_verified'] is True and candidate_status != 'unknown',
                'complete restart has unknown outcome or cleanup')
    finite_time(summary['wall_seconds'])
    return {'audit': 'passed', 'sampling_mode': 'fresh_restart_v1', 'status': summary['status'],
        'restart_success': outcome, 'candidate_status': candidate_status, 'candidate_count': len(candidates),
        'valid_for_continuation': summary['status'] == 'complete' and summary['cleanup_verified'] is True,
        'replayed_observations': 0, 'new_step_cost': new_cost, 'phase_response_cost': phase_cost,
        'generation_counters_from_summary': counters, 'reported_wall_seconds': summary['wall_seconds'],
        'source_cost_historical': summary['source_cost'], 'expected_protocol_sha256': fingerprint(expected),
        'input_sha256': hashes,
        'limits': 'Extra blind sampling, not within-state repair or first-attempt efficacy. Candidates remain '
                  'unvalidated. Unknown response costs and wall time rely on summary snapshots. '
                  'This does not independently certify container destruction or server request payloads.'}
