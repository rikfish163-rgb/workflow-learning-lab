"""Verify and project the terminal paired-omission utility result on amax.

This read-only CPU reporter emits pair indices and fixed aggregate fields only.
It never launches a study, scorer, model, queue operation, or GPU query. Scores
are those recorded by the controller; this reporter does not re-score tasks.
"""
import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from types import FunctionType

from scripts import native_experience_utility_001 as estimator
from scripts import native_source_episode as episode
from scripts import native_source_registration as public
from scripts import paired_omission_utility_gpu_queue_001 as queue
from scripts import paired_omission_utility_registration_001 as registration
from scripts import run_paired_omission_utility_001 as runner
from scripts.native_appworld_controller import _duration_seconds
from scripts.native_appworld_worker_client import _decode, _same
from wll.protocol import fingerprint

ROOT = Path(__file__).resolve().parents[1]
ERROR = 'paired_omission_utility_report_invalid'
MAX_BYTES = 64_000_000
_COUNTERS = ('generations', 'prompt_tokens', 'completion_tokens', 'action_attempts',
             'api_invocations', 'upstream_requests')
_STATISTICS = ('n_pairs', 'scenario_count', 'control_success', 'treatment_success', 'delta',
               'helped', 'harmed', 'ties', 'net_helped', 'observed_harm_rate',
               'cluster_bootstrap_ci95', 'exact_cluster_ci95', 'bootstrap_seed',
               'bootstrap_degenerate', 'exact_resample_count', 'exact_degenerate')
_CHECKS = ('at_least_8_scenarios', 'bootstrap_lower_positive', 'exact_cluster_lower_positive',
           'observed_harm_at_most_5_percent')


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def _sha(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _plan(plan):
    pairs = [[task, task[:-2], 17] for task in registration.UTILITY_TASK_IDS]
    batches = [list(range(i, min(i + 2, 27))) for i in range(0, 27, 2)]
    orders = [['control', 'experience'] if i % 2 == 0 else ['experience', 'control'] for i in range(27)]
    _require(type(plan) is dict and plan.get('id') == registration.ID and plan.get('status') == 'registered'
        and plan.get('private_run_root') == registration.PRIVATE_RUN_ROOT
        and _same(plan.get('expected_pairs'), pairs) and _same(plan.get('batches'), batches)
        and _same(plan.get('branch_orders'), orders) and type(plan.get('source')) is dict
        and _same(plan['source'].get('source_scenario_ids'), list(registration.SOURCE_SCENARIOS))
        and type(plan['source'].get('experience_bundle_id')) is str
        and re.fullmatch(re.escape(registration.BUNDLE_PREFIX) + '[a-f0-9]{64}',
                         plan['source']['experience_bundle_id']) is not None)
    estimator._identity(plan['identity'])
    return pairs


def _indexed(rows, *, maximum):
    _require(type(rows) is list and len(rows) <= maximum)
    result = {}
    for row in rows:
        _require(type(row) is dict and type(row.get('pair_index')) is int
                 and 0 <= row['pair_index'] < 27 and row.get('arm') in ('control', 'experience'))
        key = row['pair_index'], row['arm']
        _require(key not in result)
        result[key] = row
    return result


def _known_episode(cost):
    if not (cost.get('unknown_cost') is False and cost.get('counts_complete') is True
            and cost.get('generation_accounting_valid') is True and cost.get('replay') is None
            and _finite(cost.get('parent_wall_seconds'))
            and all(type(cost.get(key)) is int and cost[key] >= 0 for key in _COUNTERS)
            and type(cost.get('generation_results')) is int and cost['generation_results'] >= 0
            and type(cost.get('generation_attempts_after_cancellations')) is int
            and cost['generation_attempts_after_cancellations'] == cost['generations']
            and type(cost.get('unreturned_generation_intents_upper_bound')) is int
            and cost['unreturned_generation_intents_upper_bound'] == 0
            and type(cost.get('journal_returned_prompt_tokens_lower_bound')) is int
            and cost['journal_returned_prompt_tokens_lower_bound'] == cost['prompt_tokens']
            and type(cost.get('journal_returned_completion_tokens_lower_bound')) is int
            and cost['journal_returned_completion_tokens_lower_bound'] == cost['completion_tokens']):
        return False
    return cost['generation_results'] == cost['generations']


def _known_scorer(cost):
    return cost.get('attempted') is True and cost.get('unknown_cost') is False and _finite(cost.get('wall_seconds'))


def _audit_costs(costs):
    audits = costs.get('source_verifications')
    if type(audits) is not list or len(audits) != 55:
        return None
    stages = [row.get('stage') if type(row) is dict else None for row in audits]
    if stages != ['preflight'] + ['hash'] * 54:
        return None
    if not all(row.get('returned') is True and type(row.get('child_returncode')) is int
               and row['child_returncode'] == 0 and _finite(row.get('wall_seconds'))
               and _finite(row.get('timeout_seconds')) and row['wall_seconds'] <= row['timeout_seconds']
               for row in audits):
        return None
    return sum(row['wall_seconds'] for row in audits)


def _journal_jobs(journal, sha):
    _require(type(journal) is list and len(journal) <= 100000
             and all(type(row) is dict and type(row.get('event')) is str for row in journal))
    enqueued, ready, fences = {}, {}, {}
    for row in journal:
        event = row['event']
        if event in ('GPU_enqueue_intent', 'GPU_ready'):
            job = row.get('job')
            _require(type(job) is dict and type(job.get('batch_index')) is int
                and type(job.get('owner')) is str and _same(job, queue.make_job(job['batch_index'], job['owner'], sha)))
            index = job['batch_index']
            target = enqueued if event == 'GPU_enqueue_intent' else ready
            _require(index not in target and (event != 'GPU_enqueue_intent' or index == len(enqueued))
                and (event != 'GPU_ready' or (index in enqueued and fingerprint(job) not in fences
                                               and type(row.get('service')) is dict)))
            target[index] = row
        elif event == 'release_fence_started':
            digest = row.get('job_sha256')
            _require(_sha(digest) and digest not in fences
                     and digest in {fingerprint(item['job']) for item in enqueued.values()})
            fences[digest] = row
    _require(len(enqueued) <= 14 and set(ready) <= set(enqueued)
             and all(_same(ready[i]['job'], enqueued[i]['job']) for i in ready))
    return enqueued, ready, fences


def _recorded_release(item, job, fence, boot_id=None):
    try:
        release = item['release']
        state, info = release['state'], release['state']['job']
        return (item.get('unknown_cost') is False and release.get('verified') is True
            and release.get('runtime_overrun') is False and release.get('cuda_seconds') is None
            and _finite(release.get('GPU_seconds')) and release['GPU_seconds'] <= 7200
            and _finite(release.get('release_wall_seconds')) and release['release_wall_seconds'] <= 90
            and release.get('service_wall_seconds') == release['GPU_seconds']
            and type(state) is dict and state.get('owned') is True and state.get('admission_cancelled') is True
            and state.get('monitor_live') is True and state.get('released') is True and state.get('model_pids') == []
            and _finite(state.get('epoch')) and type(info) is dict and info.get('status') == 'finished'
            and info.get('job_id') == job['id'] and _sha(info.get('queue_sha256'))
            and _finite(info.get('started_epoch')) and _finite(info.get('finished_epoch'))
            and 0 <= info['started_epoch'] <= info['finished_epoch'] <= state['epoch']
            and release['GPU_seconds'] == info['finished_epoch'] - info['started_epoch']
            and type(fence) is dict and fence.get('job_sha256') == fingerprint(job)
            and type(fence.get('monitor_pid')) is int and fence['monitor_pid'] > 0
            and type(fence.get('monitor_start_ticks')) is str and fence['monitor_start_ticks'].isdigit()
            and type(fence.get('boot_id')) is str and re.fullmatch('[0-9a-f-]{36}', fence['boot_id']) is not None
            and (boot_id is None or fence['boot_id'] == boot_id)
            and _finite(fence.get('cancel_epoch')) and fence['cancel_epoch'] <= state['epoch']
            and type(state.get('watch')) is dict and state['watch'].get('pid') == fence['monitor_pid']
            and _finite(state['watch'].get('checked_epoch'))
            and 0 <= state['epoch'] - state['watch']['checked_epoch'] < 35
            and _finite(item.get('admission_to_release_seconds')) and _finite(item.get('admission_wait_seconds')))
    except (KeyError, TypeError, ValueError):
        return False


def _executions(rows, matrix, pairs):
    executions = _indexed(rows, maximum=54)
    for index, (task, _scenario, seed) in enumerate(pairs):
        metadata = []
        for arm in ('control', 'experience'):
            entry = matrix.get(((task, seed), arm))
            if entry is None or entry[1] != 'known':
                continue
            row = executions.get((index, arm))
            _require(type(row) is dict and _sha(row.get('metadata_sha256'))
                and type(row.get('receipt_binding')) is dict and _sha(row['receipt_binding'].get('sha256'))
                and type(row.get('model_reported_complete')) is bool
                and row.get('actor_status') in ('model_reported_complete', 'incomplete')
                and row['model_reported_complete'] is (row['actor_status'] == 'model_reported_complete'))
            metadata.append(row['metadata_sha256'])
        if len(metadata) == 2:
            _require(metadata[0] == metadata[1])
    return executions


def _project(plan, plan_sha, report, terminal, journal, *, release_check, bundle_verified=True, boot_id=None):
    pairs = _plan(plan)
    _require(_sha(plan_sha) and type(report) is dict and report.get('id') == registration.ID
        and report.get('plan_sha256') == plan_sha and report.get('automatic_retry') is False
        and type(report.get('planned_pairs')) is int and report['planned_pairs'] == 27
        and type(report.get('planned_summaries')) is int and report['planned_summaries'] == 54
        and all(type(report.get(key)) is int and report[key] == 0
                for key in ('writer_calls', 'training_jobs', 'dev_or_test_calls'))
        and terminal in ('success', 'failed', 'running', 'unknown')
        and type(bundle_verified) is bool and callable(release_check)
        and (boot_id is None or (type(boot_id) is str and re.fullmatch('[0-9a-f-]{36}', boot_id))))
    summaries = report.get('summaries')
    identity, bundle = plan['identity'], plan['source']['experience_bundle_id']
    matrix = estimator._matrix(pairs, summaries, identity, bundle)
    computed = estimator.evaluate_utility(pairs, summaries, identity=identity,
        source_scenario_ids=plan['source']['source_scenario_ids'], experience_bundle_id=bundle)
    _executions(report.get('executions'), matrix, pairs)
    projected_pairs = []
    for index, (task, _scenario, seed) in enumerate(pairs):
        item = {'index': index}
        for arm in ('control', 'experience'):
            entry = matrix.get(((task, seed), arm))
            status = 'missing' if entry is None else entry[1]
            item[arm] = {'status': status, 'official_success': entry[0]['success'] if status == 'known' else None}
        projected_pairs.append(item)
    costs = report.get('costs')
    _require(type(costs) is dict)
    episodes, scorers = _indexed(costs.get('episodes'), maximum=54), _indexed(costs.get('scorers'), maximum=54)
    enqueued, ready, fences = _journal_jobs(journal, plan_sha)
    jobs = costs.get('GPU_jobs')
    _require(type(jobs) is list and len(jobs) <= 14)
    by_index, recorded, durable = {}, 0, 0
    for item in jobs:
        _require(type(item) is dict and type(item.get('job')) is dict)
        job = item['job']
        index, owner = job.get('batch_index'), job.get('owner')
        _require(type(index) is int and index not in by_index and index in enqueued
            and type(owner) is str and _same(job, queue.make_job(index, owner, plan_sha))
            and _same(job, enqueued[index]['job']))
        by_index[index] = item
        fence, ready_row = fences.get(fingerprint(job)), ready.get(index)
        if _recorded_release(item, job, fence, boot_id):
            recorded += 1
            try:
                if ready_row is not None and release_check(job, item['release'], fence, ready_row['service']) is True:
                    durable += 1
            except Exception:
                pass
    complete_costs = (len(episodes) == len(scorers) == 54 and len(by_index) == len(enqueued) == 14
        and set(by_index) == set(range(14)) and recorded == durable == 14
        and all(_known_episode(value) for value in episodes.values())
        and all(_known_scorer(value) for value in scorers.values()))
    audit_wall = _audit_costs(costs)
    complete_costs = (complete_costs and audit_wall is not None and _finite(costs.get('parent_wall_seconds'))
                      and _finite(costs.get('utility_validation_wall_seconds')))
    saved = report.get('utility')
    globally_known = (report.get('status') == 'complete' and terminal == 'success' and bundle_verified
        and computed['status'] == 'complete' and complete_costs and type(saved) is dict
        and saved.get('global_execution_and_costs_verified') is True and saved.get('status') == 'complete')
    statistics = checks = passed = None
    if globally_known:
        saved_computed = {key: value for key, value in saved.items() if key != 'global_execution_and_costs_verified'}
        _require(_same(saved_computed, computed))
        statistics = {key: computed['statistics'][key] for key in _STATISTICS}
        checks = {key: computed['checks'][key] for key in _CHECKS}
        passed = computed['passed']
    cost_output = dict(known=bool(globally_known), GPU_seconds=None, episode_wall_seconds=None,
        scorer_wall_seconds=None, source_verification_wall_seconds=None, admission_wait_seconds=None,
        utility_validation_wall_seconds=None, parent_wall_seconds=None,
        **{key: None for key in _COUNTERS})
    if globally_known:
        cost_output.update(GPU_seconds=sum(item['release']['GPU_seconds'] for item in jobs),
            episode_wall_seconds=sum(item['parent_wall_seconds'] for item in episodes.values()),
            scorer_wall_seconds=sum(item['wall_seconds'] for item in scorers.values()),
            source_verification_wall_seconds=audit_wall,
            admission_wait_seconds=sum(item['admission_wait_seconds'] for item in jobs),
            utility_validation_wall_seconds=costs['utility_validation_wall_seconds'],
            parent_wall_seconds=costs['parent_wall_seconds'],
            **{key: sum(item[key] for item in episodes.values()) for key in _COUNTERS})
    lifecycle = ('complete' if report.get('status') == 'complete' else
                 'stopped' if report.get('status') in {'stopped_admission_timeout', 'stopped_model_ready_budget',
                                                       'stopped_GPU_cap'} else 'unknown')
    return dict(id='paired-omission-utility-public-report-001', plan_sha256=plan_sha,
        status='complete' if globally_known else 'not_evaluable', recorded_lifecycle_status=lifecycle,
        parent_terminal=terminal, reviewed_bundle_bound=bundle_verified, planned_pairs=27,
        planned_scenarios=9, complete_pairs=computed['complete_pairs'], pairs=projected_pairs,
        statistics=statistics, checks=checks, passed=passed, costs=cost_output,
        leases=dict(planned=14, enqueue_records=len(enqueued), recorded_release_verified=recorded,
                    durable_release_records_matched=durable, release_unverified=len(enqueued) - durable),
        scope='exposed_train_first_attempts',
        evidence_scope='Controller-recorded official scores; scorer receipts are not independently re-scored. '
                       'Durable release records are checked; no fresh GPU occupancy query is made.',
        interval_scope='Scenario-cluster resampling is conditional on these observed train scenarios; '
                       'it does not establish unseen transfer or a population harm bound.')


def _reader():
    scope = dict(episode.__dict__)
    scope['PRIVATE_RUN_ROOT'] = Path(registration.PRIVATE_STATE_ROOT)
    for name in ('_private_directory', '_read_private'):
        function = getattr(episode, name)
        scope[name] = FunctionType(function.__code__, scope, name,
                                   function.__defaults__, function.__closure__)
    return scope['_read_private']


def _verify_launch(launch, intent, plan_sha):
    _require(type(launch) is dict and set(launch) == {'unit', 'plan_sha256', 'argv'}
        and launch['plan_sha256'] == plan_sha and type(launch['unit']) is str
        and re.fullmatch(r'wll-native-source-experience-utility-[a-f0-9]{32}\.service', launch['unit'])
        and type(intent) is dict and set(intent) == {'plan_sha256', 'summaries', 'automatic_retry',
            'reviewed_bundle_sha256', 'pid', 'boot_id'} and intent['plan_sha256'] == plan_sha
        and intent['automatic_retry'] is False and _sha(intent['reviewed_bundle_sha256'])
        and type(intent['pid']) is int and intent['pid'] > 0
        and type(intent['boot_id']) is str and re.fullmatch('[0-9a-f-]{36}', intent['boot_id']))
    expected = runner._get_module().unit_command('pilot', launch['unit'], ['--run', '--plan-sha256', plan_sha],
        root=ROOT, log_path=Path(registration.PRIVATE_RUN_ROOT) / 'utility-parent.log')
    actual = launch['argv']
    _require(type(actual) is list and all(type(value) is str for value in actual)
             and actual.count(registration.PYTHON) == 1)
    prefix, expected_end, actual_end = expected.index('--') + 3, expected.index(registration.PYTHON), actual.index(registration.PYTHON)
    _require(actual[:prefix] == expected[:prefix] and actual[actual_end:] == expected[expected_end:])
    def environment(values):
        _require(all('=' in value for value in values))
        pairs = [value.split('=', 1) for value in values]
        keys = [pair[0] for pair in pairs]
        _require(keys == sorted(set(keys)))
        return dict(pairs)
    wanted, observed = environment(expected[prefix:expected_end]), environment(actual[prefix:actual_end])
    inherited = {'PATH', 'HOME', 'USER', 'LOGNAME', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS'}
    _require(set(observed) - inherited == set(wanted) - inherited
             and all(observed[key] == value for key, value in wanted.items() if key not in inherited))


def _terminal(launch, intent):
    """Inspect only the recorded local unit; uncertain or absent state stays unknown."""
    try:
        unit = launch['unit']
        names = ('MainPID', 'ExecMainPID', 'ActiveState', 'SubState', 'LoadState', 'Result',
                 'ExecMainStatus', 'ExecMainCode', 'Description', 'MemoryMax', 'CPUQuotaPerSecUSec',
                 'TasksMax', 'NoNewPrivileges', 'KillMode', 'RestrictAddressFamilies', 'RuntimeMaxUSec',
                 'RemainAfterExit', 'ControlGroup')
        runtime = '/run/user/' + str(os.getuid())
        environment = dict(os.environ, XDG_RUNTIME_DIR=runtime,
                           DBUS_SESSION_BUS_ADDRESS='unix:path=' + runtime + '/bus')
        process = subprocess.run(['/usr/bin/systemctl', '--user', 'show', unit,
            *['--property=' + name for name in names]], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=5, check=True, env=environment)
        _require(len(process.stdout) <= 32768)
        lines = process.stdout.decode('utf-8').splitlines()
        _require(all('=' in line for line in lines) and len({line.split('=', 1)[0] for line in lines}) == len(lines))
        values = dict(line.split('=', 1) for line in lines)
        _require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == intent['boot_id']
            and values['Description'] == 'wll-native-source:' + unit and values['LoadState'] == 'loaded'
            and values['MemoryMax'] == str(6 * 1024**3) and _duration_seconds(values['CPUQuotaPerSecUSec']) == 4
            and values['TasksMax'] == '256' and values['NoNewPrivileges'] == 'yes' and values['KillMode'] == 'control-group'
            and values['RemainAfterExit'] == 'yes'
            and set(values['RestrictAddressFamilies'].split()) == {'AF_UNIX', 'AF_INET', 'AF_INET6'}
            and _duration_seconds(values['RuntimeMaxUSec']) == runner._get_module().PARENT_SECONDS)
        if values['MainPID'] == str(intent['pid']) and values['ActiveState'] == 'active' and values['SubState'] == 'running':
            return 'running'
        _require(values['MainPID'] == '0' and values['ExecMainPID'] == str(intent['pid']))
        group = values['ControlGroup']
        _require(not group or (group.startswith('/') and '..' not in Path(group).parts))
        if group:
            procs = Path('/sys/fs/cgroup') / group.lstrip('/') / 'cgroup.procs'
            try:
                _require(not procs.read_text().strip())
            except FileNotFoundError:
                pass
        if (values['ActiveState'] == 'active' and values['SubState'] == 'exited'
                and values['Result'] == 'success' and values['ExecMainCode'] == '1' and values['ExecMainStatus'] == '0'):
            return 'success'
        if values['ActiveState'] == 'failed' and values['Result'] != 'success':
            return 'failed'
    except Exception:
        pass
    return 'unknown'


def _verify_backend(plan, backend):
    from scripts import native_source_registration_006
    scope = dict(episode._backend.__globals__)
    scope['registration'] = native_source_registration_006
    FunctionType(episode._backend.__code__, scope, '_backend')(plan, backend)


def _durable_checker(snapshots, plan):
    def read(name):
        raw = public.public_read(queue.REMOTE, 'artifacts/' + name)
        snapshots[name] = _hash(raw)
        return _decode(raw), raw
    def check(job, release, fence, service):
        info, _ = read('gpu-idle-job-' + job['id'] + '.json')
        cancel, _ = read('completion-admission-cancelled-' + job['id'] + '.json')
        ready, _ = read('completion-service-ready-' + job['id'] + '.json')
        current, current_raw = read('gpu-idle-queue.json')
        archived = current.get('id') != job['id']
        if archived:
            current, current_raw = read('gpu-idle-queue-archived-' + job['id'] + '.json')
        _require(_same(current, job) and _same(info, release['state']['job'])
            and (archived or _hash(current_raw) == info['queue_sha256'])
            and type(cancel) is dict and cancel.get('reason') == 'controller_stop'
            and cancel.get('job_sha256') == fingerprint(job) and _finite(cancel.get('epoch'))
            and cancel['epoch'] <= fence['cancel_epoch'])
        identity = service.get('identity')
        _verify_backend(plan, service['backend'])
        _require(type(identity) in (list, tuple) and len(identity) == 5
            and type(identity[0]) is int and identity[0] > 0
            and type(identity[1]) is str and identity[1].isdigit()
            and type(identity[2]) is int and identity[2] > 0
            and type(identity[3]) is str and identity[3].isdigit() and _sha(identity[4])
            and info.get('process_group') == identity[2] and info.get('start_ticks') == identity[3]
            and info.get('gpu_index') == service['backend']['gpu_policy']['physical_gpu_index']
            and type(ready) is dict and set(ready) == {'job_sha256', 'ready_epoch', 'ready_monotonic',
                                                       'backend_sha256', 'pid'}
            and ready['job_sha256'] == fingerprint(job) and ready['backend_sha256'] == fingerprint(service['backend'])
            and ready['pid'] == identity[0] and fingerprint(ready) == identity[4]
            and _finite(ready['ready_epoch']) and info['started_epoch'] <= ready['ready_epoch'] <= info['finished_epoch']
            and _finite(ready['ready_monotonic']) and ready['ready_monotonic'] > 0)
        heartbeat = queue.REMOTE / 'artifacts' / ('gpu-idle-heartbeat-' + job['id'])
        try:
            os.lstat(heartbeat)
        except FileNotFoundError:
            return True
        return False
    return check


def _report(plan_sha, result_sha):
    _require(_sha(plan_sha) and _sha(result_sha) and socket.gethostname() == 'amax'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and ROOT == registration.ROOT)
    plan = registration.load_registration(ROOT, plan_sha, private=False)
    _plan(plan)
    reader, snapshots = _reader(), {}
    private = Path(registration.PRIVATE_RUN_ROOT)
    def read(name, expected=None):
        path = private / name
        raw = reader(path, maximum=MAX_BYTES)
        digest = _hash(raw)
        _require(expected is None or digest == expected)
        snapshots[path] = digest
        return raw
    report_value = _decode(read('study-result.json', result_sha))
    launch, intent = _decode(read('parent-launch-intent.json')), _decode(read('study-intent.json'))
    _verify_launch(launch, intent, plan_sha)
    _require(_same(intent['summaries'], runner._get_module()._summaries(plan)))
    journal_raw = read('study-journal.jsonl')
    _require(journal_raw and journal_raw.endswith(b'\n'))
    journal = [_decode(line) for line in journal_raw.splitlines()]
    _require(journal and type(journal[-1]) is dict and journal[-1].get('event') == 'study_stop'
        and sum(type(row) is dict and row.get('event') == 'study_stop' for row in journal) == 1
        and _same(journal[-1].get('report'), report_value))
    bundle_bound = False
    try:
        bundle_raw = read('reviewed-bundle.json', intent['reviewed_bundle_sha256'])
        bundle = _decode(bundle_raw)
        runner._get_module()._validate_bundle(plan, bundle)
        bundle_bound = True
    except Exception:
        pass
    terminal = _terminal(launch, intent)
    durable_snapshots = {}
    result = _project(plan, plan_sha, report_value, terminal, journal,
                      release_check=_durable_checker(durable_snapshots, plan), bundle_verified=bundle_bound,
                      boot_id=intent['boot_id'])
    for path, digest in snapshots.items():
        _require(_hash(reader(path, maximum=MAX_BYTES)) == digest)
    for name, digest in durable_snapshots.items():
        _require(_hash(public.public_read(queue.REMOTE, 'artifacts/' + name)) == digest)
    _require(_same(registration.load_registration(ROOT, plan_sha, private=False), plan)
             and _terminal(launch, intent) == terminal)
    result['result_sha256'] = result_sha
    return result


def report(plan_sha, result_sha):
    try:
        return _report(plan_sha, result_sha)
    except Exception:
        raise ValueError(ERROR) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--result-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(report(args.plan_sha256, args.result_sha256), sort_keys=True, allow_nan=False))
        return 0
    except Exception:
        sys.stderr.write(ERROR + '\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
