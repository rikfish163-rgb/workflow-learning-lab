"""Independent raw prompt, response, replay, execution and outcome reconstruction."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from scripts.report_timed_tasks import count, finite_time, require
from scripts.run_memory_branch import initial_actor_messages, protocol
from wll.protocol import fingerprint
from wll.runner import extract_code


def audit_branch(directory, task, experience, expected):
    directory, hashes = Path(directory), {}

    def read(name, lines=False):
        raw = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines()] if lines else json.loads(raw)

    computed = protocol(task, experience, SimpleNamespace(model_id=expected['model_id'],
        max_tokens=expected['max_tokens'], temperature=expected['temperature']), arm=expected['arm'],
        max_steps=expected['max_steps'], seconds=expected['rollout_seconds'], seed=expected['seed'],
        environment_id=expected['environment_id'], expected_backend=expected['expected_backend'],
        prompt_version=expected['prompt_version'], backend_idle_check_enabled=expected['backend_idle_check_enabled'])
    require(computed == expected, 'external memory branch protocol differs')
    manifest, summary = read('manifest.json'), read('summary.json')
    require(all(manifest.get(k) == v for k, v in expected.items()) and manifest['no_automatic_retry'] is True
        and manifest['candidate_drafting'] is False and manifest['automatic_training'] is False
        and manifest['maximum_attempts'] == 1, 'memory branch manifest differs')
    require(manifest['status'] == summary['status'] and summary['status'] in {'complete', 'incomplete', 'interrupted'},
            'memory branch status is not terminal')
    data = {name: read(name + '.jsonl', True) for name in
            ('steps', 'requests', 'raw-responses', 'request-errors', 'meters', 'executions', 'episodes')}
    steps, requests, executions, episodes = [data[n] for n in ('steps', 'requests', 'executions', 'episodes')]
    length = replayed = verified = 0
    require(len(steps) <= expected['max_steps'] and len(episodes) <= 1, 'step or attempt budget exceeded')
    new = []
    for index, row in enumerate(steps):
        require(type(row['index']) is int and row['index'] == index and row['replayed'] is False,
                'fresh branch step order or replay label differs')
        finite_time(row['step']['elapsed_seconds'])
        new.append(row['step'])
    require(len(requests) <= expected['max_steps'] and len(new) <= len(requests) <= len(new) + 1,
            'generation/step count differs')

    def indexed(name):
        rows = data[name]
        ids = [r['index'] for r in rows]
        require(all(type(i) is int and 0 <= i < len(requests) for i in ids)
                and ids == sorted(set(ids)), 'duplicate or invalid response index')
        return {r['index']: r for r in rows}

    raw, errors, meters = indexed('raw-responses'), indexed('request-errors'), indexed('meters')
    require(not (raw.keys() & errors.keys()) and raw.keys() | errors.keys() == set(range(len(requests))),
            'each attempted request requires a raw response or error')
    cost = {'prompt_tokens': 0, 'completion_tokens': 0}
    usable, unusable, returned_requests = 0, 0, []
    messages = initial_actor_messages(task, experience, expected['prompt_version'])
    actor_actions = []
    for index, request in enumerate(requests):
        seed = expected['seed'] + length + index
        body = {'model': expected['model_id'], 'messages': messages, 'seed': seed,
                'max_tokens': expected['max_tokens'], 'temperature': expected['temperature']}
        require(request['index'] == index and request['body'] == body and request['request_sha256'] ==
                hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(), 'actor request differs')
        if index in raw:
            response = raw[index]['response']
            valid_usage = all(type(response.get(k)) is int and response[k] >= 0 for k in cost)
            if valid_usage:
                usable += 1
                matched = response.get('backend') == expected['expected_backend']
                require(meters.get(index) == {'index': index, 'seed': seed, 'backend_matches': matched,
                    'backend_sha256': fingerprint(response.get('backend')),
                    **{k: response[k] for k in cost}}, 'raw response and meter differ')
                for key in cost:
                    cost[key] += response[key]
                if matched:
                    require(0 < response['prompt_tokens'] and 0 < response['completion_tokens'] <= expected['max_tokens'],
                            'response usage outside budget')
                    action = None if response.get('format_error') else extract_code(response['text'])
                    if action is not None:
                        actor_actions.append(action)
                    returned_requests.append({'task_id': task.task_id, 'phase': 'followup',
                        'mode': expected['arm'], 'seed': seed, 'step_index': length + index,
                        'request_sha256': request['request_sha256'], 'action': action,
                        **{k: response[k] for k in cost}})
            else:
                unusable += 1
                require(index not in meters, 'unusable usage has a meter')
            if index < len(new):
                step = new[index]
                require(valid_usage and response.get('backend') == expected['expected_backend']
                    and step['raw_response'] == response['text']
                    and all(step[k] == response[k] for k in cost), 'step differs from response')
                expected_action = '' if response.get('format_error') else extract_code(response['text'])
                require(step['action'] == expected_action, 'executed action differs from response')
                if response.get('format_error'):
                    require(step['observation'] == 'Generation failed: ' + response['format_error'], 'format failure differs')
        else:
            require(index >= len(new) and index not in meters, 'missing response generated a step')
        if index < len(new):
            messages = messages + [{'role': 'assistant', 'content': new[index]['raw_response']},
                                   {'role': 'user', 'content': new[index]['observation']}]
    require(len(meters) == usable, 'extra response meter')
    counters = summary['counters']
    require(count(counters['generation_calls']) == len(requests)
        and count(counters['generation_calls_without_response']) == len(errors)
        and count(counters['generation_calls_with_unusable_usage']) == unusable
        and counters['returned_response_cost'] == cost, 'generation accounting differs')
    expected_actions = actor_actions
    require(len(executions) <= len(expected_actions) and summary['execution_attempts'] == len(executions),
            'execution accounting differs')
    for index, row in enumerate(executions):
        require(row['index'] == index and row['action'] == expected_actions[index]
            and row['status'] in {'returned', 'unknown'}, 'executed action differs')
        require(all(r['status'] == 'returned' for r in executions[:index]), 'continued after unknown execution')
        finite_time(row['execute_seconds'])
    actual_steps = [row['step'] for i, row in enumerate(steps) if not raw[i]['response'].get('format_error')]
    require(len(actual_steps) <= len(executions) <= len(actual_steps) + 1, 'step/execution journal count differs')
    for step, row in zip(actual_steps, executions, strict=False):
        require(row['status'] == 'returned' and step['action'] == row['action']
                and step['observation'] == row['observation'], 'step/execution observations differ')
    outcome = episodes[0]['success'] if episodes else None
    require(outcome is None or type(outcome) is bool, 'nonboolean outcome')
    if episodes:
        e = episodes[0]
        require(e['task'] == asdict(task) and e['model_id'] == expected['model_id']
            and e['seed'] == expected['seed'] and e['branch'] == 'first' and e['experience_id'] == (experience.experience_id if experience else None)
            and e['replayed_steps'] == length and e['steps'] == [r['step'] for r in steps], 'episode differs')
        for key in ('max_steps', 'max_tokens', 'temperature', 'prompt_version', 'environment_id', 'arm', 'rollout_seconds'):
            require(e['metadata'].get(key) == expected[key], 'episode setting differs: ' + key)
        if new:
            require(e['metadata']['backend'] == expected['expected_backend'], 'episode backend differs')
        finite_time(e['elapsed_seconds'])
    new_cost = {'steps': len(new), **{k: sum(count(s[k]) for s in new) for k in cost}}
    require(summary['outcome'] is outcome and summary['replayed_observations'] == replayed
        and summary['new_step_cost'] == new_cost and summary['accepted_experiences'] == 0, 'summary differs')
    complete = summary['status'] == 'complete'
    context = sum(r['error_type'] == 'ContextBudgetExceeded' for r in errors.values())
    if complete:
        require(type(outcome) is bool and summary['cleanup_verified'] is True and verified == length
            and not unusable and usable == len(returned_requests) and len(errors) == context
            and context == int(episodes[0]['termination'] == 'context_budget')
            and len(executions) == len(actual_steps), 'complete branch has unknown costs or cleanup')
    finite_time(summary['wall_seconds'])
    return {'audit': 'passed', 'status': summary['status'], 'arm': expected['arm'], 'outcome': outcome,
        'valid_for_continuation': complete, 'replayed_observations': replayed, 'verified_replay_observations': verified,
        'new_step_cost': new_cost, 'returned_response_cost': cost,
        'generation_counters': {k: counters[k] for k in counters if k != 'returned_response_cost'},
        'reported_context_rejections': context,
        'unknown_additional_response_cost': bool(len(errors) - context or unusable),
        'unknown_execution_result': any(r['status'] == 'unknown' for r in executions),
        'execute_seconds': sum(r['execute_seconds'] for r in executions),
        'reported_wall_seconds': summary['wall_seconds'], 'requests': returned_requests,
        'accepted_experiences': 0, 'automatic_training': False,
        'expected_protocol_sha256': fingerprint(expected), 'input_sha256': hashes,
        'scope': 'Raw fresh-branch reconstruction, not independent scoring reproduction or utility acceptance. '
                 'Context rejection and GPU cost still require server event reconciliation; container absence external.'}
