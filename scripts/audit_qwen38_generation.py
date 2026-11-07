"""Read-only reconstruction of complete source requests against real server events."""
import argparse
import ast
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.report_timed_tasks import audit_run
from scripts.run_qwen38_capability import (
    ID,
    PLAN,
    ROOT,
    TASKS,
    digest,
    expected_protocol,
    load_registration,
    read,
    require,
    validate_health,
)
from wll.prompts import initial_messages
from wll.protocol import Task
from wll.storage import atomic_json


def parse_events(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith('{"event":')]


def audit_events(episodes, events, plan, model_id, counters):
    requests, rejected = [], 0
    for episode in episodes:
        require(episode['model_id'] == model_id and episode['seed'] == plan['seed']
                and episode['branch'] == 'first' and episode['replayed_steps'] == 0
                and episode['experience_id'] is None, 'different first-attempt protocol')
        messages = initial_messages(Task(**episode['task']), plan['prompt_version'])
        for index, step in enumerate(episode['steps']):
            body = {'model': model_id, 'messages': messages, 'seed': plan['seed'] + index,
                    'max_tokens': plan['max_tokens'], 'temperature': plan['temperature']}
            request_sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            requests.append({'task_id': episode['task']['task_id'], 'step_index': index,
                             'request_sha256': request_sha, 'step': step})
            messages.extend([{'role': 'assistant', 'content': step['raw_response']},
                             {'role': 'user', 'content': step['observation']}])
        rejected += episode['termination'] == 'context_budget'
    require(len(events) == 2 * len(requests), 'missing, extra or failed generation events')
    reasons, syntax, rows = Counter(), Counter(), []
    max_reserved = max_allocated = retries = 0
    total_seconds = 0.0
    for index, request in enumerate(requests):
        start, done = events[index * 2:index * 2 + 2]
        require(start['event'] == 'generation_started' and done['event'] == 'generation_completed'
                and start['request_id'] == done['request_id'] == index + 1
                and start['request_sha256'] == request['request_sha256'], 'request chain differs from server')
        step, mem = request['step'], done['memory']
        require(start['prompt_tokens'] == done['prompt_tokens'] == step['prompt_tokens']
                and done['completion_tokens'] == step['completion_tokens']
                and start['max_new'] == plan['max_tokens'], 'usage or output budget differs')
        require(type(step['prompt_tokens']) is int and step['prompt_tokens'] > 0
                and type(step['completion_tokens']) is int and 0 < step['completion_tokens'] <= plan['max_tokens']
                and step['prompt_tokens'] + plan['max_tokens'] <= plan['max_context'], 'invalid token budget')
        reason = done['finish_reason']
        require(reason in ('stop', 'length') and (reason != 'length' or step['completion_tokens'] == plan['max_tokens']),
                'unknown or inconsistent stopping reason')
        require(0 <= mem['allocated_bytes.all.peak'] <= mem['reserved_bytes.all.peak'] <= 20 * 1024**3
                and mem['num_ooms'] == 0 and mem['num_alloc_retries'] >= retries, 'invalid memory evidence')
        require(math.isfinite(done['generation_seconds']) and done['generation_seconds'] >= 0, 'invalid time')
        reasons[reason] += 1
        syntax_error = False
        try:
            ast.parse(step['action'])  # Parse only; never execute model-produced code on the host.
        except SyntaxError:
            syntax[reason] += 1
            syntax_error = True
        max_reserved = max(max_reserved, mem['reserved_bytes.all.peak'])
        max_allocated = max(max_allocated, mem['allocated_bytes.all.peak'])
        retries = mem['num_alloc_retries']
        total_seconds += done['generation_seconds']
        rows.append({k: v for k, v in request.items() if k != 'step'} | {
            'request_id': index + 1, 'finish_reason': reason, 'syntax_error': syntax_error,
            'prompt_tokens': step['prompt_tokens'], 'completion_tokens': step['completion_tokens']})
    require(counters == {'generation_calls': len(requests) + rejected,
                         'generation_calls_without_response': rejected,
                         'generation_calls_with_unusable_usage': 0}, 'unaccounted or unknown generation cost')
    return {'generated_requests': len(requests), 'context_rejections': rejected,
            'returned_token_cost': {key: sum(row[key] for row in rows)
                                    for key in ('prompt_tokens', 'completion_tokens')},
            'finish_reasons': dict(reasons), 'syntax_errors_by_finish_reason': dict(syntax),
            'unknown_generated_response_cost': False, 'generation_seconds': total_seconds,
            'peak_reserved_bytes': max_reserved, 'peak_allocated_bytes': max_allocated,
            'allocator_retries': retries, 'requests': rows,
            'scope': 'Complete source batch only. Context rejections produced no server generation event; their CPU tokenization cost is not measured separately. Generation seconds overlap job wall. Syntax/stop counts are descriptive, not a causal attribution.'}


def audit_stage(plan, stage, index):
    health = read(stage / 'health.json')
    validate_health(health, plan, index)
    tasks = [json.loads(x) for x in (ROOT / TASKS).read_text().splitlines()][index * 4:index * 4 + 4]
    report = audit_run(stage / 'run', expected_protocol(plan, tasks, health))
    require(report['valid_for_comparison'] and report['completed'] == 4,
            'partial or unknown batch; do not infer missing requests')
    require(report == read(stage / 'report.json'), 'saved workflow report differs')
    release = read(stage / 'release.json')
    require(release['released'] is True and release['owned'] is True, 'resource release unverified')
    remote = stage / 'remote-artifacts'
    manifest = read(remote / 'manifest.json')
    require(not manifest['missing'], 'missing remote artifacts')
    for name, sha in manifest['sha256'].items():
        require(Path(name).name == name and digest(remote / name) == sha, 'changed remote evidence')
    job = read(remote / f'gpu-idle-job-{ID}-b{index}.json')
    require(job['status'] == 'finished' and job == release['job'], 'GPU job not terminal or differs')
    episodes = [json.loads(x) for x in (stage / 'run/episodes.jsonl').read_text().splitlines()]
    log = remote / f'gpu-idle-job-{ID}-b{index}.log'
    ledger = audit_events(episodes, parse_events(log.read_text()), plan, health['model'],
                          report['generation_counters_from_summary'])
    require(ledger['returned_token_cost'] == report['returned_response_cost'], 'aggregate token cost differs')
    return {'audit': 'passed', 'script_sha256': digest(Path(__file__)), 'batch': index,
            'report_sha256': digest(stage / 'report.json'), 'server_log_sha256': digest(log),
            'job_wall_seconds': job['finished_epoch'] - job['started_epoch'],
            'completed': 4, 'successes': report['successes'], **ledger}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--batch', required=True, type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    plan, _ = load_registration(ROOT, ROOT / PLAN, args.plan_sha256)
    stage = ROOT / 'artifacts' / ID / f'batch-{args.batch}'
    output = stage / 'generation-audit.json'
    require(not output.exists(), 'audit already exists; preserve original')
    result = audit_stage(plan, stage, args.batch)
    result['plan_sha256'] = args.plan_sha256
    atomic_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('requests', 'scope')}))


if __name__ == '__main__':
    main()
