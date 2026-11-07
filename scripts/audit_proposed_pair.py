"""Reconcile both continuation conditions with the frozen service event ledger."""
import argparse

from scripts.audit_qwen38_followups import audit_events
from scripts.audit_qwen38_generation import parse_events
from scripts.run_qwen38_proposed_pair import (
    CONDITIONS,
    ID,
    PLAN,
    ROOT,
    audit_batch,
    digest,
    expected_protocol,
    load_registration,
    load_sources,
    read,
    require,
    validate_health,
)
from wll.storage import atomic_json


def audit_generation(stage, sources, entries, expected, plan, events):
    report = audit_batch(stage, sources, entries, expected)
    require(report['valid_for_continuation'] and report == read(stage / 'report.json'), 'complete raw-audited pair required')
    requests = [row for condition in CONDITIONS for row in report['trials'][condition]['requests']]
    rejected = sum(r['reported_context_rejections'] for r in report['trials'].values())
    ledger = audit_events(requests, events, plan, rejected)
    require(ledger['returned_token_cost'] == report['returned_response_cost'], 'pair aggregate cost differs')
    ledger['phase_response_cost'] = {'continuation': ledger['returned_token_cost'].copy()}
    for row in ledger['requests']:
        row['phase'] = 'continuation'
    return ledger


def audit_stage(root, plan, inputs, stage):
    health = read(stage / 'health.json')
    validate_health(health, plan)
    entries = inputs['ordered_sources']
    sources = load_sources(root, entries)
    expected = {c: expected_protocol(sources[0], entries[0], plan, health, c) for c in CONDITIONS}
    require(read(stage / 'expected.json') == expected, 'pair protocol differs')
    release, cleanup = read(stage / 'release.json'), read(stage / 'local-cleanup.json')
    require(release['owned'] and release['released'] and cleanup['local_client_exited']
            and cleanup['recorded_containers_absent'], 'resource release unconfirmed')
    remote = stage / 'remote-artifacts'
    manifest = read(remote / 'manifest.json')
    require(not manifest['missing'], 'service evidence missing')
    for name, sha in manifest['sha256'].items():
        require('/' not in name and name not in {'.', '..'} and digest(remote / name) == sha, 'service evidence differs')
    job = read(remote / f'gpu-idle-job-{ID}-b0.json')
    require(job['status'] == 'finished' and job == release['job'], 'GPU job differs or active')
    log = remote / f'gpu-idle-job-{ID}-b0.log'
    ledger = audit_generation(stage, sources, entries, expected, plan, parse_events(log.read_text()))
    return {'audit': 'passed', 'script_sha256': digest(__file__), 'report_sha256': digest(stage / 'report.json'),
        'server_log_sha256': digest(log), 'job_wall_seconds': job['finished_epoch'] - job['started_epoch'], **ledger,
        'scope': 'One exploratory origin pair. Generation time overlaps service wall time. Prefix replay, '
                 'historical critic costs and prior source generation are separate; no accepted experience or learning.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    plan, _, inputs = load_registration(ROOT, ROOT / PLAN, args.plan_sha256)
    stage = ROOT / 'artifacts' / ID / 'batch-0'
    output = stage / 'generation-audit.json'
    require(not output.exists(), 'pair generation audit already exists')
    atomic_json(output, audit_stage(ROOT, plan, inputs, stage) | {'plan_sha256': args.plan_sha256})


if __name__ == '__main__':
    main()
