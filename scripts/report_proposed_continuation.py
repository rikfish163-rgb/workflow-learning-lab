"""Independent raw prompt, response, replay, execution and outcome reconstruction."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from scripts.report_timed_tasks import count, finite_time, require
from scripts.run_proposed_continuation import initial_continuation_messages, protocol
from wll.protocol import fingerprint
from wll.runner import extract_code


def audit_trial(directory, source, proposal, proof, expected):
    directory, hashes = Path(directory), {}

    def read(name, lines=False):
        raw = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines()] if lines else json.loads(raw)

    computed = protocol(source, proposal, proof, SimpleNamespace(model_id=expected['model_id'],
        max_tokens=expected['max_tokens'], temperature=expected['temperature']), condition=expected['condition'],
        max_steps=expected['max_steps'], seconds=expected['rollout_seconds'], seed=expected['seed'],
        environment_id=expected['environment_id'], expected_backend=expected['expected_backend'],
        origin_scenarios=set(expected['origin_scenarios']), backend_idle_check_enabled=expected['backend_idle_check_enabled'])
    require(computed == expected, 'external continuation protocol differs')
    manifest, summary = read('manifest.json'), read('summary.json')
    require(all(manifest.get(k) == v for k, v in expected.items()) and manifest['no_automatic_retry'] is True
        and manifest['candidate_drafting'] is False and manifest['posthoc_exploratory_only'] is True
        and manifest['maximum_attempts'] == 1, 'continuation manifest differs')
    require(manifest['status'] == summary['status'] and summary['status'] in {'complete', 'incomplete', 'interrupted'},
            'continuation status is not terminal')
    data = {name: read(name + '.jsonl', True) for name in
            ('steps', 'requests', 'raw-responses', 'request-errors', 'meters', 'executions', 'episodes')}
    steps, requests, executions, episodes = [data[n] for n in ('steps', 'requests', 'executions', 'episodes')]
    length = expected['prefix_length']
    require(len(steps) <= expected['max_steps'] and len(episodes) <= 1, 'step or attempt budget exceeded')
    replayed, verified, new = 0, 0, []
    for index, row in enumerate(steps):
        require(type(row['index']) is int and row['index'] == index and row['replayed'] is (index < length),
                'step order or replay label differs')
        step = row['step']
        finite_time(step['elapsed_seconds'])
        if index < length:
            old = asdict(source.steps[index])
            require(all(step[k] == v for k, v in old.items() if k not in ('observation', 'elapsed_seconds')),
                    'replayed source action or historical cost changed')
            require(verified == index, 'continued after replay mismatch')
            verified += int(step['observation'] == old['observation'])
            replayed += 1
        else:
            require(verified == length, 'new action without a verified prefix')
            new.append(step)
    require(not requests or verified == length, 'generation before exact prefix replay')
    require(len(requests) <= expected['max_steps'] - length and len(new) <= len(requests) <= len(new) + 1,
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
    messages = initial_continuation_messages(source, proposal, expected['condition'])
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
                    returned_requests.append({'task_id': source.task.task_id, 'phase': 'followup',
                        'mode': expected['condition'], 'seed': seed, 'step_index': length + index,
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
    expected_actions = [s.action for s in source.steps[:length]] + actor_actions
    require(len(executions) <= len(expected_actions) and summary['execution_attempts'] == len(executions),
            'execution accounting differs')
    for index, row in enumerate(executions):
        require(row['index'] == index and row['action'] == expected_actions[index]
            and row['status'] in {'returned', 'unknown'}, 'executed action differs')
        require(all(r['status'] == 'returned' for r in executions[:index]), 'continued after unknown execution')
        finite_time(row['execute_seconds'])
    actual_steps = [r['step'] for r in steps if r['replayed'] or not r['step']['observation'].startswith('Generation failed:')]
    require(len(actual_steps) <= len(executions) <= len(actual_steps) + 1, 'step/execution journal count differs')
    for step, row in zip(actual_steps, executions, strict=False):
        require(row['status'] == 'returned' and step['action'] == row['action']
                and step['observation'] == row['observation'], 'step/execution observations differ')
    outcome = episodes[0]['success'] if episodes else None
    require(outcome is None or type(outcome) is bool, 'nonboolean outcome')
    if episodes:
        e = episodes[0]
        require(e['task'] == asdict(source.task) and e['model_id'] == expected['model_id']
            and e['seed'] == expected['seed'] and e['branch'] == 'recovery' and e['experience_id'] is None
            and e['replayed_steps'] == length and e['steps'] == [r['step'] for r in steps], 'episode differs')
        for key in ('max_steps', 'max_tokens', 'temperature', 'prompt_version', 'environment_id', 'source_hash',
                    'proposal_sha256', 'prefix_proof_sha256', 'condition', 'rollout_seconds'):
            require(e['metadata'].get(key) == expected[key], 'episode setting differs: ' + key)
        require(e['metadata']['posthoc_exploratory_only'] is True, 'exploratory label missing')
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
            and len(executions) == len(actual_steps), 'complete branch has unknown costs, replay or cleanup')
    finite_time(summary['wall_seconds'])
    return {'audit': 'passed', 'status': summary['status'], 'condition': expected['condition'], 'outcome': outcome,
        'valid_for_continuation': complete, 'replayed_observations': replayed, 'verified_replay_observations': verified,
        'new_step_cost': new_cost, 'returned_response_cost': cost,
        'generation_counters': {k: counters[k] for k in counters if k != 'returned_response_cost'},
        'reported_context_rejections': context,
        'unknown_additional_response_cost': bool(len(errors) - context or unusable),
        'unknown_execution_result': any(r['status'] == 'unknown' for r in executions),
        'execute_seconds': sum(r['execute_seconds'] for r in executions),
        'reported_wall_seconds': summary['wall_seconds'], 'requests': returned_requests,
        'accepted_experiences': 0, 'posthoc_exploratory_only': True,
        'expected_protocol_sha256': fingerprint(expected), 'input_sha256': hashes,
        'scope': 'Raw branch reconstruction, not independent scoring reproduction or held-out efficacy. '
                 'Context rejection and GPU cost still require server event reconciliation; container absence external.'}
