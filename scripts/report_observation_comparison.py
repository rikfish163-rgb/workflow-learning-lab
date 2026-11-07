"""Paired known-origin engineering report; manual feedback is not learned experience."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.documentation_feedback import FEEDBACK
from scripts.probe_observation_visibility import printable_documentation_call
from scripts.report_timed_tasks import audit_run
from scripts.run_foundation_pilot import (
    OBSERVATION_ID,
    combine_batches,
    digest,
    expected_protocol,
    read,
    require,
    validate_health,
    validate_observation_comparison,
    verify_frozen_plan,
)
from wll.storage import atomic_json


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def paired_outcomes(baseline, current):
    require(baseline and baseline.keys() == current.keys(), 'paired task identities differ')
    require(all(type(v) is bool for table in (baseline, current) for v in table.values()),
            'all paired outcomes must be known booleans')
    helped = sorted(k for k in baseline if current[k] and not baseline[k])
    harmed = sorted(k for k in baseline if baseline[k] and not current[k])
    return {'baseline': baseline, 'feedback': current, 'helped': helped, 'harmed': harmed,
            'net_successes': len(helped)-len(harmed)}


def audit_feedback(stage, *, enabled):
    raw, augmented = 'Execution successful.', 'Execution successful.\n'+FEEDBACK
    expected, queries, silent = [], Counter(), 0

    def sha(value):
        return hashlib.sha256(value.encode()).hexdigest()

    for row in lines(stage/'run/steps.jsonl'):
        action, observation = row['step']['action'], row['step']['observation']
        try:
            printable_documentation_call(action)
        except ValueError:
            require(observation != augmented, 'feedback attached to an ineligible action')
            continue
        queries[(row['task_id'], action)] += 1
        silent += observation == raw
        if enabled:
            require(observation != raw, 'eligible silent query lacks registered feedback')
        if observation == augmented:
            require(enabled, 'baseline contains unregistered feedback')
            expected.append({'task_id': row['task_id'], 'index': row['index'],
                'action_sha256': sha(action), 'raw_observation_sha256': sha(raw),
                'raw_observation_chars': len(raw), 'augmented_observation_sha256': sha(augmented),
                'augmented_observation_chars': len(augmented)})
    if enabled:
        require(lines(stage/'feedback-events.jsonl') == expected, 'feedback journal differs from steps')
    return {'feedback_events': len(expected), 'bare_documentation_queries': sum(queries.values()),
            'silent_bare_queries': silent,
            'repeated_at_least10_task_ids': sorted({key[0] for key, n in queries.items() if n >= 10})}


def audit_study(root, plan, plan_sha, *, feedback):
    tasks = lines(root/'artifacts/foundation-inputs-001/tasks.jsonl')
    reports, releases, observations, outcomes, costs = [], [], [], {}, []
    for index in range(3):
        stage = root/'artifacts'/plan['id']/f'batch-{index}'
        health = read(stage/'health.json')
        validate_health(health, plan, index)
        expected = expected_protocol(plan, tasks[4*index:4*index+4], health)
        require(read(stage/'expected.json') == expected, 'saved protocol differs from frozen plan')
        require(read(stage/'job.json')['plan_sha256'] == plan_sha, 'job plan differs')
        reports.append(audit_run(stage/'run', expected))
        release = read(stage/'release.json')
        job = release['job']
        require(release['owned'] and release['released'] and not release['model_pids']
                and job['status'] == 'finished' and job['job_id'] == f"{plan['id']}-b{index}",
                'service termination evidence differs')
        require(job['ticket']['idle_seconds'] >= 600 and job['ticket']['max_gap_seconds'] <= 35,
                'idle admission evidence differs')
        releases.append(release)
        costs.append(job['finished_epoch']-job['started_epoch'])
        observations.append(audit_feedback(stage, enabled=feedback))
        for episode in lines(stage/'run/episodes.jsonl'):
            task_id = episode['task']['task_id']
            require(task_id not in outcomes, 'duplicate episode')
            outcomes[task_id] = episode['success']
    result = combine_batches(reports, releases)
    require(list(outcomes) == plan['task_ids'], 'complete frozen task order required')
    return {'result': result, 'outcomes': outcomes, 'observation_diagnostics': observations,
            'gpu_service_wall_seconds_by_batch': costs, 'gpu_service_wall_seconds': sum(costs)}


def compare(root, plan_path, plan_sha):
    plan = verify_frozen_plan(root, plan_path, plan_sha)
    require(plan['id'] == OBSERVATION_ID, 'registered observation comparison required')
    validate_observation_comparison(root, plan)
    historical = read(root/'artifacts/foundation-complete-integration-002.json')
    for name, expected in historical['files_sha256'].items():
        require(digest(root/name) == expected, 'historical result changed: '+name)
    baseline = read(root/plan['baseline_plan'])
    old = audit_study(root, baseline, digest(root/plan['baseline_plan']), feedback=False)
    new = audit_study(root, plan, plan_sha, feedback=True)
    paired = paired_outcomes(old['outcomes'], new['outcomes'])
    return {'scope': 'Known-origin adaptive engineering comparison of a manual scaffold. '
                    'Not a learning, unseen-scene, RSI, or matched-hardware timing claim.',
            'plan_sha256': plan_sha, 'baseline': old, 'feedback': new, 'paired': paired,
            'accounting': 'Token cost views overlap. Service wall includes model loading/idle. '
                          'Prior failed studies and CPU preparation remain separate retained costs.',
            'decision': 'eligible_for_separately_registered_recovery_study'
                        if new['result']['advance_to_recovery_study'] else 'stop_configuration',
            'automatic_recovery_or_training': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = Path(args.output)
    require(not output.exists(), 'report already exists')
    result = compare(root, Path(args.plan), args.plan_sha256)
    atomic_json(output, result)
    print(json.dumps({'decision': result['decision'], 'paired': result['paired']}))


if __name__ == '__main__':
    main()
