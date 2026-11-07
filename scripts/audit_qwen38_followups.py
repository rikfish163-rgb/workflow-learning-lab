"""Read-only reconstruction of restart, continuation and draft server requests."""
import argparse
import ast
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.audit_qwen38_generation import parse_events
from scripts.run_qwen38_recovery import (
    ID,
    PLAN,
    ROOT,
    audit_batch,
    digest,
    load_registration,
    load_sources,
    read,
    require,
    trial_protocol,
    validate_health,
)
from wll.cli import read_episodes
from wll.experience import draft_experience
from wll.prompts import initial_messages
from wll.storage import atomic_json


def draft_messages(source, episode, seed):
    class Captured(Exception):
        pass

    messages = []

    class Capture:
        def generate(self, value, *, seed):
            messages.extend(value)
            raise Captured()

    try:
        draft_experience(source, episode, Capture(), seed=seed)
    except Captured:
        pass
    require(len(messages) == 2, 'draft request was not reconstructed')
    return messages


def trial_requests(source, episode, responses, summary, expected, phase):
    require(phase in {'restart', 'recovery'} and episode.model_id == expected['model_id']
            and episode.seed == expected[phase + '_seed'] and episode.branch == 'recovery'
            and episode.experience_id is None, 'followup protocol differs')
    require(phase != 'restart' or episode.replayed_steps == 0, 'blind restart replayed history')
    requests = []

    def append(messages, seed, row, kind, step_index=None, action=None):
        body = {'model': expected['model_id'], 'messages': messages, 'seed': seed,
                'max_tokens': expected['max_tokens'], 'temperature': expected['temperature']}
        requests.append({'task_id': source.task.task_id, 'phase': kind, 'mode': phase,
            'seed': seed, 'step_index': step_index, 'action': action,
            'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
            **{k: row[k] for k in ('prompt_tokens', 'completion_tokens')}})

    messages = initial_messages(source.task, expected['prompt_version'])
    actor = [r for r in responses if r['phase'] == phase]
    drafts = [r for r in responses if r['phase'] == 'draft']
    require(len(actor) == len(episode.steps) - episode.replayed_steps and len(drafts) <= 1
            and responses == actor + drafts, 'unaccounted or reordered returned responses')
    for index, step in enumerate(episode.steps):
        if index >= episode.replayed_steps:
            row = actor[index - episode.replayed_steps]
            require(row['seed'] == episode.seed + index and all(row[k] == getattr(step, k)
                    for k in ('prompt_tokens', 'completion_tokens')), 'actor seed or usage differs')
            append(messages, episode.seed + index, row, 'followup', index, step.action)
        messages.extend([{'role': 'assistant', 'content': step.raw_response},
                         {'role': 'user', 'content': step.observation}])
    if drafts:
        require(episode.success is True and drafts[0]['seed'] == expected['draft_seed'],
                'draft requires successful followup and registered seed')
        append(draft_messages(source, episode, expected['draft_seed']), expected['draft_seed'], drafts[0], 'draft')
    rejected = int(episode.termination == 'context_budget')
    for name, rows, missing in [(phase, actor, rejected), ('draft', drafts, 0)]:
        meter = summary['phases'][name]
        require(meter['generation_calls'] == len(rows) + missing
                and meter['generation_calls_without_response'] == missing
                and meter['generation_calls_with_unusable_usage'] == 0, 'unaccounted generation cost')
    return requests, rejected


def audit_events(requests, events, plan, rejected):
    require(len(events) == 2 * len(requests), 'missing, extra or failed generation events')
    reasons, syntax, rows = Counter(), Counter(), []
    costs = {p: {'prompt_tokens': 0, 'completion_tokens': 0} for p in ('followup', 'draft')}
    peak_reserved = peak_allocated = retries = 0
    seconds = 0.0
    for index, request in enumerate(requests, 1):
        start, done = events[(index - 1) * 2:index * 2]
        require(start['event'] == 'generation_started' and done['event'] == 'generation_completed'
                and start['request_id'] == done['request_id'] == index
                and start['request_sha256'] == request['request_sha256'], 'request chain differs')
        require(start['prompt_tokens'] == done['prompt_tokens'] == request['prompt_tokens']
                and done['completion_tokens'] == request['completion_tokens']
                and start['max_new'] == plan['max_tokens'], 'server usage or budget differs')
        require(type(request['prompt_tokens']) is int and request['prompt_tokens'] > 0
                and type(request['completion_tokens']) is int
                and 0 < request['completion_tokens'] <= plan['max_tokens']
                and request['prompt_tokens'] + plan['max_tokens'] <= plan['max_context'], 'invalid token budget')
        reason, mem = done['finish_reason'], done['memory']
        require(reason in ('stop', 'length') and
                (reason != 'length' or request['completion_tokens'] == plan['max_tokens']), 'invalid stopping reason')
        require(0 <= mem['allocated_bytes.all.peak'] <= mem['reserved_bytes.all.peak'] <= 20 * 1024**3
                and mem['num_ooms'] == 0 and mem['num_alloc_retries'] >= retries, 'invalid memory evidence')
        require(math.isfinite(done['generation_seconds']) and done['generation_seconds'] >= 0, 'invalid timing')
        reasons[reason] += 1
        syntax_error = None
        if request['action'] is not None:
            syntax_error = False
            try:
                ast.parse(request['action'])  # Parse only; no model code executes on the host.
            except SyntaxError:
                syntax[reason] += 1
                syntax_error = True
        for key in ('prompt_tokens', 'completion_tokens'):
            costs[request['phase']][key] += request[key]
        peak_reserved = max(peak_reserved, mem['reserved_bytes.all.peak'])
        peak_allocated = max(peak_allocated, mem['allocated_bytes.all.peak'])
        retries = mem['num_alloc_retries']
        seconds += done['generation_seconds']
        rows.append({k: v for k, v in request.items() if k != 'action'} |
                    {'request_id': index, 'finish_reason': reason, 'action_syntax_error': syntax_error})
    return {'generated_requests': len(rows), 'context_rejections': rejected,
        'returned_token_cost': {k: sum(p[k] for p in costs.values()) for k in ('prompt_tokens', 'completion_tokens')},
        'phase_response_cost': costs, 'finish_reasons': dict(reasons), 'action_syntax_errors_by_finish_reason': dict(syntax),
        'unknown_generated_response_cost': False, 'generation_seconds': seconds,
        'peak_reserved_bytes': peak_reserved, 'peak_allocated_bytes': peak_allocated,
        'allocator_retries': retries, 'requests': rows}


def audit_stage(root, plan, parent, inputs, stage):
    health = read(stage / 'health.json')
    validate_health(health, plan)
    entries = inputs['selected']
    sources = load_sources(root, entries)
    expected = {s.task.task_id: trial_protocol(s, r, plan, parent, health)
                for s, r in zip(sources, entries, strict=True)}
    require(read(stage / 'expected-trials.json') == expected, 'external expected protocols differ')
    report = audit_batch(stage, sources, entries, expected)
    require(report['valid_for_continuation'] and report == read(stage / 'report.json'),
            'complete raw-audited followups required')
    release, cleanup = read(stage / 'release.json'), read(stage / 'local-cleanup.json')
    require(release['owned'] and release['released'] and cleanup['local_client_exited']
            and cleanup['recorded_containers_absent'], 'resource release unconfirmed')
    remote = stage / 'remote-artifacts'
    manifest = read(remote / 'manifest.json')
    require(not manifest['missing'], 'remote evidence missing')
    for name, sha in manifest['sha256'].items():
        require(Path(name).name == name and digest(remote / name) == sha, 'remote evidence changed')
    job = read(remote / f'gpu-idle-job-{ID}-b0.json')
    require(job['status'] == 'finished' and job == release['job'], 'GPU job differs or remains active')
    requests, rejected = [], 0
    for source, entry in zip(sources, entries, strict=True):
        folder = stage / 'trials' / source.task.task_id
        episodes = read_episodes(folder / 'episodes.jsonl')
        require(len(episodes) == 1, 'missing scored followup')
        responses = [json.loads(s) for s in (folder / 'responses.jsonl').read_text().splitlines()]
        records, missing = trial_requests(source, episodes[0], responses, read(folder / 'summary.json'),
            expected[source.task.task_id], 'restart' if entry['mode'] == 'fresh_restart' else 'recovery')
        requests.extend(records)
        rejected += missing
    log = remote / f'gpu-idle-job-{ID}-b0.log'
    ledger = audit_events(requests, parse_events(log.read_text()), plan, rejected)
    require(ledger['phase_response_cost'] == report['phase_response_cost'], 'aggregate cost differs')
    return {'audit': 'passed', 'script_sha256': digest(__file__), 'report_sha256': digest(stage / 'report.json'),
        'server_log_sha256': digest(log), 'job_wall_seconds': job['finished_epoch'] - job['started_epoch'],
        **ledger, 'scope': 'Read-only complete-batch reconstruction. Prefix history is not regenerated; '
            'blind restarts exclude old failure history. Draft prompts use frozen bounded projection. '
            'Generation seconds overlap job wall; step costs overlap phase costs. Pre-generation context '
            'rejection has no GPU generation event; CPU tokenization is not timed separately. '
            'Candidate utility remains untested; AST syntax checks never execute model code.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    plan, parent, inputs = load_registration(ROOT, ROOT / PLAN, args.plan_sha256)
    stage = ROOT / 'artifacts' / ID / 'batch-0'
    output = stage / 'generation-audit.json'
    require(not output.exists(), 'generation audit already exists')
    result = audit_stage(ROOT, plan, parent, inputs, stage) | {'plan_sha256': args.plan_sha256}
    atomic_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('requests', 'scope')}))


if __name__ == '__main__':
    main()
