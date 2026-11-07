"""Recompute one bounded recovery from journals and an external expected protocol."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from scripts.report_timed_tasks import count, finite_time, require
from wll.protocol import fingerprint

REQUIRED = {'source_hash', 'prefix_proof', 'task_id', 'model_id', 'max_steps', 'max_tokens',
            'temperature', 'recovery_seed', 'draft_seed', 'rollout_seconds', 'draft_seconds',
            'prompt_version', 'environment_id', 'expected_backend', 'source_backend',
            'script_sha256', 'wll_code_fingerprint'}


def audit_trial(directory, source, expected):
    directory = Path(directory)
    hashes = {}

    def read(name, *, lines=False):
        raw = (directory/name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines()] if lines else json.loads(raw)

    require(isinstance(expected, dict) and REQUIRED <= expected.keys(),
            'complete external recovery protocol required')
    manifest, summary = read('manifest.json'), read('summary.json')
    require(all(k in manifest and manifest[k] == v for k, v in expected.items()),
            'recovery manifest differs from expected protocol')
    require(manifest['status'] == summary['status'] and summary['status'] in
            {'complete', 'incomplete', 'interrupted'}, 'recovery is not terminal')
    require(source.success is False and source.task.split == 'train' and source.branch == 'first'
            and source.replayed_steps == 0 and source.experience_id is None
            and fingerprint(source.to_dict()) == expected['source_hash'], 'source provenance differs')
    proof = expected['prefix_proof']
    length = next((i+1 for i, s in enumerate(source.steps)
                   if s.observation.startswith('Execution failed')), None)
    require(length is not None and 0 < length < expected['max_steps']
            and proof['prefix_length'] == length and proof['prefix_success'] is False
            and proof['status'] == 'verified' and proof['cleanup_verified'] is True
            and proof['source_hash'] == expected['source_hash']
            and proof['task_id'] == expected['task_id'] == source.task.task_id
            and proof['prefix_sha256'] == fingerprint([asdict(s) for s in source.steps[:length]]),
            'first-error prefix proof differs')
    require(manifest['step_budget_includes_replay'] is True
            and manifest['maximum_recovery_attempts'] == manifest['maximum_draft_attempts'] == 1
            and manifest['no_automatic_retry'] is True, 'recovery retry or budget rule differs')
    steps, responses = read('steps.jsonl', lines=True), read('responses.jsonl', lines=True)
    episodes, candidates = read('episodes.jsonl', lines=True), read('experiences.jsonl', lines=True)
    require(len(steps) <= expected['max_steps'] and len(episodes) <= 1 and len(candidates) <= 1,
            'trial exceeds its attempt or step budget')
    replayed, new_steps = [], []
    new_cost = {'steps': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
    for index, row in enumerate(steps):
        require(type(row['index']) is int and row['index'] == index
                and row['replayed'] is (index < length), 'step order or replay classification differs')
        step = row['step']
        finite_time(step['elapsed_seconds'])
        for key in ('prompt_tokens', 'completion_tokens'):
            count(step[key])
        if row['replayed']:
            old = asdict(source.steps[index])
            if step != old:
                require(summary['status'] != 'complete' and summary.get('error_type') == 'ReplayMismatch'
                        and index == len(steps)-1 and not responses and not episodes
                        and step['observation'] != old['observation']
                        and all(step[k] == old[k] for k in old
                                if k not in ('observation', 'elapsed_seconds')),
                        'replayed source differs outside a recorded final mismatch')
            replayed.append(step)
        else:
            new_steps.append(step)
            new_cost['steps'] += 1
            for key in ('prompt_tokens', 'completion_tokens'):
                new_cost[key] += step[key]
    phase_rows = {'recovery': [], 'draft': []}
    phase_cost = {phase: {'prompt_tokens': 0, 'completion_tokens': 0} for phase in phase_rows}
    counters = {}
    for row in responses:
        phase = row['phase']
        require(phase in phase_rows and (phase != 'recovery' or not phase_rows['draft']),
                'response phases out of order')
        group = phase_rows[phase]
        seed = expected['draft_seed'] if phase == 'draft' else expected['recovery_seed']+length+len(group)
        require(type(row['seed']) is int and row['seed'] == seed, 'response seed differs')
        require(type(row['backend_matches']) is bool and row['backend_matches'] ==
                (row['backend_sha256'] == fingerprint(expected['expected_backend']))
                and (not group or group[-1]['backend_matches']), 'response backend evidence differs')
        for key in ('prompt_tokens', 'completion_tokens'):
            phase_cost[phase][key] += count(row[key])
        group.append(row)
    require(len(new_steps) <= len(phase_rows['recovery']), 'new action without a response')
    for step, response in zip(new_steps, phase_rows['recovery'], strict=False):
        require(response['backend_matches'] and all(step[k] == response[k]
                for k in ('prompt_tokens', 'completion_tokens')), 'new step and response costs differ')
    for phase, rows in phase_rows.items():
        meter = summary['phases'][phase]
        keys = ('generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage')
        counters[phase] = {k: count(meter[k]) for k in keys}
        require(meter['generation_calls'] == len(rows)+meter[keys[1]]+meter[keys[2]]
                and meter['returned_response_cost'] == phase_cost[phase], 'phase accounting differs')
        require(meter['generation_calls'] <= (1 if phase == 'draft' else expected['max_steps']-length),
                'generation attempt limit exceeded')
    outcome = episodes[0]['success'] if episodes else None
    require(outcome is None or type(outcome) is bool, 'recovery outcome must be boolean or unknown')
    if episodes:
        e = episodes[0]
        require(e['task'] == asdict(source.task) and e['model_id'] == expected['model_id']
                and e['seed'] == expected['recovery_seed'] and e['branch'] == 'recovery'
                and e['experience_id'] is None and e['replayed_steps'] == length
                and e['steps'] == [r['step'] for r in steps] and len(replayed) == length,
                'scored episode and recovery journal differ')
        for key in ('max_steps', 'max_tokens', 'temperature', 'prompt_version', 'environment_id', 'source_hash'):
            require(e['metadata'].get(key) == expected[key], 'episode setting differs: '+key)
        require(e['metadata']['prefix_proof_hash'] == fingerprint(proof)
                and e['metadata']['rollout_deadline_seconds'] == expected['rollout_seconds'],
                'episode prefix or deadline differs')
        if new_steps:
            require(e['metadata'].get('backend') == expected['expected_backend'], 'episode backend differs')
        finite_time(e['elapsed_seconds'])
    candidate_status = summary['candidate_status']
    if outcome is not True:
        require(not candidates and counters['draft']['generation_calls'] == 0
                and candidate_status == ('not_attempted' if outcome is None else 'not_eligible'),
                'unverified or failed recovery cannot draft')
    else:
        require(candidate_status in {'unvalidated', 'invalid_schema', 'invalid_format', 'unknown'},
                'unexpected draft status')
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
                and cost['model_id'] == expected['model_id']
                and cost['evidence_kind'] == 'new_recovery', 'candidate cost or evidence kind differs')
    require(summary['recovery_success'] is outcome and summary['replayed_observations'] == len(replayed)
            and summary['new_step_cost'] == new_cost and summary['source_cost'] == {
                'prompt_tokens': sum(s.prompt_tokens for s in source.steps),
                'completion_tokens': source.generated_tokens}, 'summary differs from raw evidence')
    if summary['status'] == 'complete':
        require(type(outcome) is bool and summary['cleanup_verified'] is True
                and candidate_status != 'unknown', 'complete trial has unknown outcome or cleanup')
    finite_time(summary['wall_seconds'])
    return {'audit': 'passed', 'status': summary['status'], 'recovery_success': outcome,
        'candidate_status': candidate_status, 'candidate_count': len(candidates),
        'valid_for_continuation': summary['status'] == 'complete' and summary['cleanup_verified'] is True,
        'replayed_observations': len(replayed), 'new_step_cost': new_cost,
        'phase_response_cost': phase_cost, 'generation_counters_from_summary': counters,
        'reported_wall_seconds': summary['wall_seconds'], 'source_cost_historical': summary['source_cost'],
        'expected_protocol_sha256': fingerprint(expected), 'input_sha256': hashes,
        'limits': 'Candidates remain unvalidated. Phase response costs are disjoint; step costs overlap. '
                  'Unreturned usage, error classifications and wall time rely on summary snapshots. '
                  'Unknown cost is not zero; this does not independently certify container destruction.'}
