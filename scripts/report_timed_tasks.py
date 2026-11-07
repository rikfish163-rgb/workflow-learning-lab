"""Recompute timed-run results from journals against an external expected protocol."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from wll.protocol import fingerprint
from wll.storage import atomic_json

REQUIRED = {'tasks', 'model_id', 'max_tokens', 'temperature', 'seed', 'max_steps',
            'rollout_seconds', 'prompt_version', 'environment_id', 'expected_backend',
            'script_sha256', 'wll_code_fingerprint', 'backend_idle_check_enabled', 'deadline_scope'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    require(type(value) is int and value >= 0, 'invalid nonnegative count')
    return value


def finite_time(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'invalid timing value')


def audit_run(directory, expected):
    directory = Path(directory)
    hashes = {}

    def read(name, lines=False):
        raw = (directory/name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)

    manifest, summary = read('manifest.json'), read('summary.json')
    require(isinstance(expected, dict) and REQUIRED <= expected.keys(),
            'complete externally expected protocol required')
    require(all(k in manifest and manifest[k] == v for k, v in expected.items()),
            'manifest differs from expected protocol')
    require(manifest['status'] in {'complete', 'incomplete', 'interrupted'}, 'run is not terminal')
    require(summary['status'] == manifest['status'], 'terminal status differs')
    tasks = manifest['tasks']
    ids = [t['task_id'] for t in tasks]
    require(ids and len(set(ids)) == len(ids), 'empty or duplicate planned tasks')
    require(all(t['split'] == 'train' and t['scenario_id'] == t['task_id'].rsplit('_', 1)[0]
                for t in tasks), 'timed study requires consistent train task provenance')
    by_id = {t['task_id']: t for t in tasks}
    budget = count(manifest['max_steps'])
    require(budget > 0, 'empty step budget')
    require(isinstance(manifest['expected_backend'], dict) and manifest['expected_backend'],
            'backend must be explicit')
    expected_backend_sha = fingerprint(manifest['expected_backend'])
    attempts, episodes = read('attempts.jsonl', True), read('episodes.jsonl', True)
    steps, responses = read('steps.jsonl', True), read('responses.jsonl', True)
    attempted_ids = [r['task_id'] for r in attempts]
    require(attempted_ids == ids[:len(attempts)] and len(attempts) <= len(ids),
            'attempts are not a unique ordered prefix')
    for attempt in attempts:
        require(attempt['status'] in {'complete', 'timed_out', 'infrastructure_error', 'interrupted'},
                'unknown attempt status')
        if attempt['status'] == 'complete':
            require(type(attempt['success']) is bool and attempt['cleanup_verified'] is True,
                    'completed outcome or cleanup missing')
        else:
            require(attempt['success'] is None, 'incomplete outcome must remain unknown')
        require(attempt['cleanup_verified'] in (None, True), 'cleanup was not verified')
        for key in ('started_epoch', 'finished_epoch'):
            finite_time(attempt[key])
        require(attempt['finished_epoch'] >= attempt['started_epoch'], 'attempt time reversed')
    groups = {task_id: {'steps': [], 'responses': []} for task_id in attempted_ids}
    step_cost = {'steps': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
    response_cost = {'prompt_tokens': 0, 'completion_tokens': 0}
    for kind, rows in [('steps', steps), ('responses', responses)]:
        order = []
        for row in rows:
            task_id = row['task_id']
            require(task_id in groups, 'journal contains an unattempted task')
            order.append(attempted_ids.index(task_id))
            group = groups[task_id][kind]
            require(len(group) < budget, 'journal exceeds step budget')
            if kind == 'steps':
                require(type(row['index']) is int and row['index'] == len(group)
                        and row['replayed'] is False, 'invalid step index or replay')
                value = row['step']
                finite_time(value['elapsed_seconds'])
                step_cost['steps'] += 1
                for key in ('prompt_tokens', 'completion_tokens'):
                    step_cost[key] += count(value[key])
            else:
                require(type(row['seed']) is int and row['seed'] == manifest['seed'] + len(group),
                        'response seed/order differs')
                require(type(row['backend_matches']) is bool and row['backend_matches'] == (
                    row['backend_sha256'] == expected_backend_sha), 'response backend claim differs')
                require(not group or group[-1]['backend_matches'], 'responses continue after backend drift')
                for key in ('prompt_tokens', 'completion_tokens'):
                    response_cost[key] += count(row[key])
            group.append(row)
        require(order == sorted(order), 'journal task order differs')
    for group in groups.values():
        require(len(group['steps']) <= len(group['responses']), 'step without returned response')
        for step, response in zip(group['steps'], group['responses'], strict=False):
            require(response['backend_matches'] and all(step['step'][k] == response[k]
                for k in ('prompt_tokens', 'completion_tokens')), 'action and response disagree')
    completed = [r for r in attempts if r['status'] == 'complete']
    require([e['task']['task_id'] for e in episodes] == [r['task_id'] for r in completed],
            'episodes do not match complete attempts')
    for episode, attempt in zip(episodes, completed, strict=True):
        task_id = attempt['task_id']
        require(episode['task'] == by_id[task_id] and episode['model_id'] == manifest['model_id']
                and episode['seed'] == manifest['seed'] and type(episode['success']) is bool
                and episode['success'] == attempt['success'] and episode['branch'] == 'first'
                and episode['experience_id'] is None and episode['replayed_steps'] == 0,
                'episode violates task/model/first-attempt protocol')
        require(fingerprint(episode) == attempt['episode_hash'], 'episode hash changed')
        require(episode['steps'] == [r['step'] for r in groups[task_id]['steps']],
                'episode and step journal differ')
        metadata = episode['metadata']
        for key in ('max_steps', 'max_tokens', 'temperature', 'prompt_version', 'environment_id'):
            require(metadata.get(key) == manifest[key], 'episode setting differs: '+key)
        require(metadata.get('rollout_deadline_seconds') == manifest['rollout_seconds'],
                'episode deadline differs')
        if episode['steps']:
            require(metadata.get('backend') == manifest['expected_backend'], 'episode backend differs')
        finite_time(episode['elapsed_seconds'])
    n, successes = len(ids), sum(e['success'] for e in episodes)
    unknown = n-len(episodes)
    computed = {'scheduled': n, 'attempted': len(attempts), 'completed': len(episodes),
        'successes': successes, 'unknown_outcomes': unknown, 'unattempted': n-len(attempts),
        'valid_for_comparison': unknown == 0,
        'success_rate_completed_only': successes/len(episodes) if episodes else None,
        'success_rate_all_scheduled': successes/n if not unknown else None,
        'success_bounds_all_scheduled': [successes/n, (successes+unknown)/n],
        'observed_step_cost': step_cost, 'returned_response_cost': response_cost}
    require(all(summary.get(k) == value for k, value in computed.items()),
            'summary disagrees with raw journals')
    require(manifest['status'] != 'complete' or unknown == 0, 'complete status has unknown outcomes')
    failed_calls = count(summary['generation_calls_without_response'])
    unusable_calls = count(summary['generation_calls_with_unusable_usage'])
    require(count(summary['generation_calls']) == len(responses)+failed_calls+unusable_calls,
            'generation counters inconsistent with response journal')
    finite_time(summary['wall_seconds'])
    return {'audit': 'passed', 'status': manifest['status'], **computed,
        'successful_scenarios': sorted({e['task']['scenario_id'] for e in episodes if e['success']}),
        'generation_counters_from_summary': {k: summary[k] for k in (
            'generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage')},
        'reported_wall_seconds': summary['wall_seconds'],
        'limits': 'The two cost views overlap. Missing-response counters and total wall time are '
                  'summary snapshots, not independently journaled request starts. Unknown generation '
                  'cost is not zero. This audit does not certify model quality or container destruction.',
        'expected_protocol_sha256': fingerprint(expected), 'input_sha256': hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--expected', required=True, help='Protocol derived from the frozen experiment plan')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = audit_run(args.run, json.loads(Path(args.expected).read_text()))
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
