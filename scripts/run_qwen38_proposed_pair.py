"""One registered no-hint/hinted continuation pair, on one idle-qualified GPU."""
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from scripts.critic_compatibility import explore_legacy_format
from scripts.report_proposed_continuation import audit_trial
from scripts.run_foundation_pilot import (
    digest,
    observation_world,
    process_matches,
    read,
    require,
    stop_client,
    verify_cpu_environment,
    verify_frozen_plan,
)
from scripts.run_proposed_continuation import (
    initial_continuation_messages,
    proposal_hint,
    protocol,
    run_continuation,
)
from scripts.run_queued_decoder_completion import SSH, remote_python
from scripts.run_qwen38_capability import (
    ENV_PLAN,
    PYTHON,
    REMOTE,
    ROOT,
    SERVICE_SHA,
    runtime_state,
    validate_live_service,
)
from scripts.run_qwen38_compact_critic import load_registration as load_parent_registration
from scripts.run_qwen38_compact_critic import load_sources
from scripts.run_qwen38_recovery import collect_remote_evidence
from scripts.run_timed_tasks import RolloutDeadline, wait_for_idle
from wll.appworld import AppWorldEnvironment
from wll.model import LocalModel
from wll.storage import atomic_json, require_space

ID = 'qwen38-proposed-recovery-pair-001'
PLAN = 'artifacts/critic-proposed-recovery-plan-001.json'
DESIGN = 'artifacts/critic-proposed-recovery-design-001.json'
PORT = 18979
PARTITION = 'artifacts/partitions-v1/manifest.json'
PARENT_SHA = 'c9ea6efeddc6da60098c89d0f6bf002852bb845ef467d37c4116cfc55fd518a3'
PREFIX_PLAN = 'artifacts/critic-proposed-prefix-plan-001.json'
PREFIX_SHA = 'f4a9a8c780cd78bf3cdfeab21c7bc7ed5927696295630ef201701e84bb039126'
CONDITIONS = ['no_hint', 'proposal_hint']


def validate_study(root, plan):
    root = Path(root)
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['design'] == DESIGN
        and plan['design_sha256'] == digest(root / DESIGN), 'registered unchanged pair design required')
    require(all(plan.get(k) == v for k, v in read(root / DESIGN).items()
        if k not in {'state', 'missing_before_registration'}), 'pair configuration differs')
    require(plan['parent_critic_plan_sha256'] == PARENT_SHA and plan['conditions'] == CONDITIONS,
            'parent or condition order differs')
    parent, baseline, _ = load_parent_registration(root, root / plan['parent_critic_plan'], PARENT_SHA)
    require(digest(root / PREFIX_PLAN) == PREFIX_SHA, 'prefix plan changed')
    prefix_plan = read(root / PREFIX_PLAN)
    for name, sha in prefix_plan['input_sha256'].items():
        require(digest(root / name) == sha, 'prefix preparation changed')
    for name, sha in [('inputs', 'inputs_sha256'), ('prefix_integration', 'prefix_integration_sha256'),
                      ('CPU_context_preflight', 'CPU_context_preflight_sha256')]:
        require(digest(root / plan[name]) == plan[sha], 'pair inputs changed')
    inputs, whole, context = [read(root / plan[k]) for k in ('inputs', 'prefix_integration', 'CPU_context_preflight')]
    require(inputs['state'] == 'prepared_not_registered' and whole['phase'] == 'complete'
        and whole['strict_compact_repair_count'] == 0 and whole['posthoc_exploratory_only'] is True
        and all(whole['recorded_container_absence'].values()) and context['phase'] == 'complete'
        and context['uses_gpu'] is False and context['model_loaded'] is False
        and context['manifest_sha256'] == plan['inputs_sha256'], 'prefix or context evidence differs')
    for name, sha in whole['input_sha256'].items():
        require(digest(root / name) == sha, 'prefix integration changed')
    require(plan['expected_backend_static'] == parent['expected_backend_static']
        and plan['gpu_policy'] == parent['gpu_policy'] and plan['wll_code_fingerprint'] == parent['wll_code_fingerprint']
        and plan['service_plan_sha256'] == SERVICE_SHA and plan['port'] == PORT, 'backend or runtime differs')
    require(all(plan[k] == baseline[k] for k in ('data_dir', 'deps_dir', 'image', 'environment_id', 'tmp_dir')),
            'execution environment differs')
    old_selection = read(root / prefix_plan['inputs'])
    eligible = [r for r in old_selection['selected'] if whole['reports'][r['task_id']]['eligible_for_continuation']]
    entries = inputs['ordered_sources']
    require(len(entries) == 1 and [r['task_id'] for r in entries] == plan['task_ids']
        == [r['task_id'] for r in eligible], 'paired source selection differs')
    for source, row, original in zip(load_sources(root, entries), entries, eligible, strict=True):
        require(all(row[k] == original[k] for k in ('source_file', 'source_hash', 'canonical_proposal'))
            and row['prefix_proof'] == whole['reports'][source.task.task_id], 'source or prefix proof differs')
        compatibility = explore_legacy_format(read(root / original['raw_response']), read(root / original['projection']))
        require(compatibility == old_selection['all_source_reports'][source.task.task_id]
            and compatibility['legacy_schema_compatible'] and compatibility['posthoc_exploratory_only'],
            'retrospective compatibility record differs')
        require(digest(root / row['hint_file']) == row['hint_sha256']
            and read(root / row['hint_file']) == {'text': proposal_hint(row['canonical_proposal'])}, 'hint changed')
        require([r['condition'] for r in row['conditions']] == CONDITIONS, 'prepared condition order differs')
        for condition, ctx in zip(row['conditions'], context['inputs'], strict=True):
            messages = initial_continuation_messages(source, row['canonical_proposal'], condition['condition'])
            require(digest(root / condition['messages']) == condition['messages_sha256'] == ctx['messages_sha256']
                and read(root / condition['messages']) == messages and ctx['task_id'] == source.task.task_id
                and ctx['condition'] == condition['condition'] and ctx['max_new_tokens'] == plan['max_tokens']
                and 0 < ctx['prompt_tokens'] and ctx['total_max_tokens'] == ctx['prompt_tokens'] + plan['max_tokens']
                and ctx['total_max_tokens'] <= plan['max_context'], 'initial prompt or token preflight differs')
    return baseline, inputs


def load_registration(root, path, sha):
    root = Path(root)
    plan = verify_frozen_plan(root, path, sha)
    baseline, inputs = validate_study(root, plan)
    parent = read(root / plan['parent_critic_plan'])
    required = set(parent['source_sha256']) | {'scripts/run_qwen38_proposed_pair.py',
        'scripts/run_proposed_continuation.py', 'scripts/report_proposed_continuation.py',
        'scripts/audit_proposed_prefix.py', 'scripts/critic_compatibility.py'}
    require(required <= plan['source_sha256'].keys() and all(plan['source_sha256'].get(n) == s
        for n, s in parent['source_sha256'].items()), 'pair source bindings incomplete')
    for mapping in (parent['input_sha256'], read(root / PREFIX_PLAN)['input_sha256'],
                    read(root / plan['prefix_integration'])['input_sha256']):
        require(all(plan['input_sha256'].get(n) == s for n, s in mapping.items()), 'parent/prefix evidence unbound')
    required_inputs = {DESIGN, PREFIX_PLAN, plan['parent_critic_plan'], plan['inputs'], plan['prefix_integration'],
        plan['CPU_context_preflight'], 'artifacts/proposed-pair-cpu-verification-001.json'}
    for row in inputs['ordered_sources']:
        required_inputs.update((row['source_file'], row['hint_file']))
        required_inputs.update(r['messages'] for r in row['conditions'])
    require(required_inputs <= plan['input_sha256'].keys(), 'pair input bindings incomplete')
    return plan, baseline, inputs


def make_job(owner, plan_sha):
    require(re.fullmatch('[a-f0-9]{32}', owner) and re.fullmatch('[a-f0-9]{64}', plan_sha), 'invalid owner or plan hash')
    return {'id': ID + '-b0', 'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 2700,
        'argv': [PYTHON, '-B', '-m', 'scripts.serve_qwen38', '--plan-sha256', SERVICE_SHA,
                 '--model-id', ID + '-b0', '--port', str(PORT)]}


def expected_protocol(source, row, plan, health, condition):
    origin = {s.rsplit('_', 1)[0] for s in read(ROOT / PARTITION)['task_ids']['origin']}
    model = SimpleNamespace(model_id=health['model'], max_tokens=plan['max_tokens'], temperature=plan['temperature'])
    result = protocol(source, row['canonical_proposal'], row['prefix_proof'], model,
        condition=condition, max_steps=plan['max_steps_including_replay'], seconds=plan['rollout_seconds'],
        seed=plan['seed'], environment_id=plan['environment_id'], expected_backend=health['backend'],
        origin_scenarios=origin, backend_idle_check_enabled=True)
    require(result['script_sha256'] == plan['source_sha256']['scripts/run_proposed_continuation.py']
        and result['wll_code_fingerprint'] == plan['wll_code_fingerprint'], 'worker implementation differs')
    return result


def audit_batch(stage, sources, entries, expected):
    require(len(sources) == len(entries) == 1 and list(expected) == CONDITIONS, 'one ordered pair required')
    folder = stage / 'trials'
    present = {p.name for p in folder.iterdir() if p.is_dir()} if folder.exists() else set()
    require(present == set(CONDITIONS[:len(present)]), 'conditions are not an ordered prefix')
    reports = {}
    for condition in CONDITIONS:
        if condition not in present:
            continue
        require(all(r['valid_for_continuation'] for r in reports.values()), 'continued after unknown branch')
        reports[condition] = audit_trial(folder / condition, sources[0], entries[0]['canonical_proposal'],
                                        entries[0]['prefix_proof'], expected[condition])
    complete = len(reports) == 2 and all(r['valid_for_continuation'] for r in reports.values())
    outcomes = {condition: reports.get(condition, {}).get('outcome') for condition in CONDITIONS}
    return {'status': 'complete' if complete else 'incomplete', 'valid_for_continuation': complete,
        'scheduled_branches': 2, 'attempted_branches': len(reports), 'unattempted_branches': 2 - len(reports),
        'known_outcomes': sum(type(r['outcome']) is bool for r in reports.values()),
        'unknown_outcomes': sum(r['outcome'] is None for r in reports.values()), 'outcomes': outcomes,
        'hint_helped': complete and outcomes == {'no_hint': False, 'proposal_hint': True},
        'hint_harmed': complete and outcomes == {'no_hint': True, 'proposal_hint': False},
        'new_step_cost': {k: sum(r['new_step_cost'][k] for r in reports.values())
                          for k in ('steps', 'prompt_tokens', 'completion_tokens')},
        'returned_response_cost': {k: sum(r['returned_response_cost'][k] for r in reports.values())
                                  for k in ('prompt_tokens', 'completion_tokens')},
        'replayed_observations': sum(r['replayed_observations'] for r in reports.values()),
        'unknown_additional_response_cost': any(r['unknown_additional_response_cost'] for r in reports.values()),
        'trials': reports, 'accepted_experiences': 0, 'candidate_drafting': False,
        'posthoc_exploratory_only': True,
        'scope': 'One retrospectively selected origin case, same prefix and sampling budget; no held-out '
                 'efficacy or learning claim. Critic/input preparation costs are historical and must be reported separately.'}


def worker(root, plan, parent, inputs, stage):
    health = read(stage / 'health.json')
    validate_health(health, plan)
    entries = inputs['ordered_sources']
    sources = load_sources(root, entries)
    require(len(sources) == 1 and [s.task.task_id for s in sources] == plan['task_ids']
        and plan['conditions'] == CONDITIONS, 'worker source/order differs')
    source, row = sources[0], entries[0]
    expected = {c: expected_protocol(source, row, plan, health, c) for c in CONDITIONS}
    require(read(stage / 'expected.json') == expected, 'worker expectations differ')
    endpoint = f"http://127.0.0.1:{plan['port']}/v1"
    model = LocalModel(endpoint, health['model'], max_tokens=plan['max_tokens'], temperature=plan['temperature'])
    resources = []
    try:
        with (stage / 'feedback-events.jsonl').open('x') as feedback:
            for condition in CONDITIONS:
                def factory(task, condition=condition):
                    raw = AppWorldEnvironment(task.task_id, plan['data_dir'], plan['max_steps_including_replay'],
                                              plan['image'], plan['deps_dir'])
                    try:
                        identifier = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}',
                            raw.worker.name], text=True, timeout=15).strip()
                        require(identifier not in {r['id'] for r in resources}, 'branches must start fresh')
                        resources.append({'condition': condition, 'task_id': task.task_id,
                                          'id': identifier, 'name': raw.worker.name})
                        atomic_json(stage / 'containers.json', resources)
                        return observation_world(raw, read(Path(root) / ENV_PLAN), task.task_id, feedback)
                    except BaseException:
                        raw.close()
                        raise

                result = run_continuation(source, row['canonical_proposal'], row['prefix_proof'], factory, model,
                    output=stage / 'trials' / condition, condition=condition,
                    max_steps=plan['max_steps_including_replay'], seconds=plan['rollout_seconds'],
                    seed=plan['seed'], environment_id=plan['environment_id'], expected_backend=health['backend'],
                    origin_scenarios=set(expected[condition]['origin_scenarios']),
                    before_rollout=lambda: wait_for_idle(endpoint, health['model'], health['backend'],
                                                        seconds=plan['server_idle_seconds']))
                atomic_json(stage / 'report.json', audit_batch(stage, sources, entries, expected))
                if result['status'] != 'complete' or not result['cleanup_verified']:
                    return 1
    finally:
        atomic_json(stage / 'report.json', audit_batch(stage, sources, entries, expected))
    return 0


def verify_remote(plan_sha):
    return remote_python('''import json,sys,subprocess,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));p=json.load(sys.stdin)
from scripts.run_qwen38_proposed_pair import load_registration,PLAN,PYTHON,ID,SERVICE_SHA
plan,parent,inputs=load_registration(root,root/PLAN,p['plan_sha'])
r=subprocess.run([PYTHON,'-B','-m','scripts.serve_qwen38','--check','--model-id',ID+'-b0','--plan-sha256',SERVICE_SHA],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),capture_output=True,text=True,timeout=15,check=True)
assert json.loads(r.stdout)['preflight']=='passed'
for name in [f'gpu-idle-job-{ID}-b0.json',f'gpu-idle-ticket-{ID}-b0.json',f'gpu-idle-ticket-{ID}-b0.consumed.json',f'gpu-idle-heartbeat-{ID}-b0']:
 assert not (root/'artifacts'/name).exists(),name
print(json.dumps({'preflight':'passed','uses_gpu':False,'sources':len(inputs['ordered_sources']),'source_bindings':len(plan['source_sha256'])}))
''', {'plan_sha': plan_sha})


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'invalid queue operation')
    require(job == make_job(job.get('owner', ''), job.get('plan_sha256', '')), 'queue command or identity changed')
    root = Path(root)
    art = root / 'artifacts'
    queue, heartbeat = art / 'gpu-idle-queue.json', art / f"gpu-idle-heartbeat-{job['id']}"
    result = art / f"gpu-idle-job-{job['id']}.json"
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = read(queue) if queue.exists() else None
        if operation == 'enqueue':
            markers = [result, heartbeat, art / f"gpu-idle-ticket-{job['id']}.json",
                       art / f"gpu-idle-ticket-{job['id']}.consumed.json"]
            if any(p.exists() for p in markers):
                raise FileExistsError('continuation pair already attempted or owned')
            load_registration(root, root / PLAN, job['plan_sha256'])
            if current:
                require(re.fullmatch('[a-z0-9_-]+', current['id']), 'invalid prior queue identity')
                old = art / f"gpu-idle-job-{current['id']}.json"
                require(old.exists() and read(old)['status'] == 'finished'
                        and not (art / f"gpu-idle-heartbeat-{current['id']}").exists(), 'prior queue active or owned')
                archive = art / f"gpu-idle-queue-archived-{current['id']}.json"
                if archive.exists():
                    require(archive.read_bytes() == queue.read_bytes(), 'queue archive differs')
                else:
                    archive.write_bytes(queue.read_bytes())
            heartbeat.touch(exist_ok=False)
            try:
                atomic_json(queue, job)
            except BaseException:
                heartbeat.unlink(missing_ok=True)
                raise
            current = job
        owned = current == job
        if owned and operation == 'pulse' and (not result.exists() or read(result)['status'] != 'finished'):
            heartbeat.touch()
        if owned and operation == 'stop':
            heartbeat.unlink(missing_ok=True)
            if result.exists():
                info = read(result)
                require(info['queue_sha256'] == digest(queue), 'running queue identity changed')
                if process_matches(info, job['argv']):
                    os.killpg(info['process_group'], signal.SIGTERM)
    return {'owned': owned, **runtime_state(job, art, known_pids)}


def remote_queue(operation, job, known_pids=()):
    return remote_python('''import json,sys
sys.path.insert(0,'/home/hts/workflow-learning-lab')
from scripts.run_qwen38_proposed_pair import queue_operation
d=json.load(sys.stdin)
print(json.dumps(queue_operation(d['operation'],d['job'],known_pids=d['known_pids'])))
''', {'operation': operation, 'job': job, 'known_pids': list(known_pids)})


def validate_health(health, plan):
    require(health.get('model') == ID + '-b0' and health.get('ready') is True
            and health.get('busy') is False and health.get('max_context') == 16384
            and health.get('max_output_tokens') == 1024, 'unexpected service health or context')
    backend = health['backend']
    require({k: v for k, v in backend.items() if k != 'gpu_policy'} == plan['expected_backend_static'],
            'inference backend differs')
    gpu, policy = backend['gpu_policy'], plan['gpu_policy']
    require(gpu.get('hostname') == policy['hostname'] and type(gpu.get('physical_gpu_index')) is int
            and str(gpu['physical_gpu_index']) in policy['allowed_gpus']
            and gpu.get('gpu_uuid') == policy['allowed_gpus'][str(gpu['physical_gpu_index'])]
            and gpu.get('visible_device_count') == 1 and gpu.get('policy_sha256') == plan['gpu_policy_sha256'],
            'unauthorized service GPU')


def run_batch(root, plan, parent, inputs, plan_path, plan_sha):
    require(load_registration(root, plan_path, plan_sha) == (plan, parent, inputs), 'in-memory configuration changed')
    remote_check = verify_remote(plan_sha)
    stage = root / 'artifacts' / ID / 'batch-0'
    stage.mkdir()
    atomic_json(stage / 'freeze-check.json', {'plan_sha256': plan_sha, 'checked_epoch': time.time(),
                                             'local_files_match': True, 'remote': remote_check})
    entries = inputs['ordered_sources']
    sources = load_sources(root, entries)
    verify_cpu_environment(parent, [asdict(s.task) for s in sources], stage / 'cpu-environment-preflight.json')
    job = make_job(uuid.uuid4().hex, plan_sha)
    atomic_json(stage / 'job.json', job)
    state, known, stopped = {}, set(), threading.Event()
    thread = tunnel = client = None
    report = release = None

    def heartbeat():
        while not stopped.is_set():
            try:
                snapshot = remote_queue('pulse', job, known)
                known.update(snapshot.get('model_pids', []))
                state.update(snapshot, local_observed_monotonic=time.monotonic())
                atomic_json(stage / 'remote-state.json', snapshot)
            except Exception as exc:
                state['observation_error'] = type(exc).__name__
            stopped.wait(15)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        state.update(remote_queue('enqueue', job), local_observed_monotonic=time.monotonic())
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        with (stage / 'tunnel.log').open('x') as log:
            tunnel = subprocess.Popen([*SSH[:-1], '-N', '-o', 'ExitOnForwardFailure=yes',
                '-L', f'{PORT}:127.0.0.1:{PORT}', '4090'], stdout=log, stderr=subprocess.STDOUT)
        loading = None
        while True:
            require(tunnel.poll() is None, 'SSH tunnel exited')
            require(state.get('owned') is True and state.get('monitor_live') is True,
                    'queue ownership or monitor unconfirmed')
            require(time.monotonic() - state['local_observed_monotonic'] <= 90, 'remote status stale')
            job_state = state.get('job') or {}
            require(job_state.get('status') != 'finished', 'server exited before health verification')
            if job_state.get('status') == 'running':
                loading = loading or time.monotonic()
                require(time.monotonic() - loading <= plan['model_start_seconds'], 'model startup timed out')
                try:
                    with RolloutDeadline(6), opener.open(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
                        raw = response.read(65_537)
                    require(len(raw) <= 65_536, 'oversized health response')
                    health = json.loads(raw)
                    break
                except OSError:
                    pass
            time.sleep(2)
        validate_health(health, plan)
        confirmed = remote_queue('status', job, known)
        validate_live_service(confirmed, health)
        known.update(confirmed['model_pids'])
        atomic_json(stage / 'activation.json', confirmed)
        atomic_json(stage / 'health.json', health)
        expected = {c: expected_protocol(sources[0], entries[0], plan, health, c) for c in CONDITIONS}
        atomic_json(stage / 'expected.json', expected)
        with (stage / 'client.log').open('x') as log:
            client = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'scripts.run_qwen38_proposed_pair',
                '--plan', str(plan_path), '--plan-sha256', plan_sha, '--worker', '--stage', str(stage)],
                cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root), TMPDIR=plan['tmp_dir']),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = client.wait(timeout=plan['client_seconds'])
        try:
            report = audit_batch(stage, sources, entries, expected)
            atomic_json(stage / 'report.json', report)
        except Exception as exc:
            atomic_json(stage / 'report-failure.json', {'client_exit_code': code,
                'error_type': type(exc).__name__, 'success_rate': None,
                'reason': 'Raw continuation artifacts incomplete or inconsistent; no outcome imputed'})
            raise
        require(code == 0 and report['valid_for_continuation'], 'continuation pair incomplete; no automatic retry')
    finally:
        try:
            stop_client(client, stage)
        finally:
            stopped.set()
            if thread is not None:
                thread.join(timeout=25)
            try:
                release = remote_queue('stop', job, known)
                deadline = time.monotonic() + 90
                while release['owned'] and not release['released'] and time.monotonic() < deadline:
                    time.sleep(5)
                    release = remote_queue('status', job, known)
                atomic_json(stage / 'release.json', release)
                require(release['owned'] and release['released'], 'model release unconfirmed')
                collect_remote_evidence(job, known, stage)
            finally:
                if tunnel is not None and tunnel.poll() is None:
                    tunnel.terminate()
                    tunnel.wait(timeout=10)
    return report, release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=PLAN)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--stage')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only controller and worker required')
    path = Path(args.plan).resolve(strict=True)
    plan, parent, inputs = load_registration(ROOT, path, args.plan_sha256)
    if args.worker:
        require(args.stage is not None, 'worker stage required')
        raise SystemExit(worker(ROOT, plan, parent, inputs, Path(args.stage)))
    with (ROOT / 'artifacts/foundation-pilot-001.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = ROOT / 'artifacts' / ID
        require(not output.exists(), 'paired study has an attempt; no automatic restart')
        require_space(ROOT / 'artifacts')
        verify_remote(args.plan_sha256)
        if args.check:
            print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'batches': 1, 'sources': 1, 'branches': 2}))
            return
        output.mkdir()
        state = {'phase': 'running', 'plan_sha256': args.plan_sha256, 'pid': os.getpid(),
            'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
            'started_epoch': time.time(), 'completed_batches': 0, 'current_batch': 0}
        atomic_json(output / 'status.json', state)
        try:
            result, release = run_batch(ROOT, plan, parent, inputs, path, args.plan_sha256)
            require(result['valid_for_continuation'] and release['released'], 'complete released batch required')
            atomic_json(output / 'report.json', result)
            state.update(phase='complete', completed_batches=1, result=result)
        except BaseException as exc:
            state.update(phase='failed_or_interrupted', error_type=type(exc).__name__, error=str(exc)[:1000])
            raise
        finally:
            state['finished_epoch'] = time.time()
            atomic_json(output / 'status.json', state)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
