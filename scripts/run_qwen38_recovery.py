"""One registered serial batch of blind restarts and verified-prefix continuations."""
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

from scripts.prepare_recovery_inputs import select_sources
from scripts.report_recovery_trial import audit_trial as audit_prefix
from scripts.report_restart_trial import audit_trial as audit_restart
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
from scripts.run_qwen38_capability import load_registration as load_source_registration
from scripts.run_recovery_pilot import expected_trial as prefix_protocol
from scripts.run_recovery_pilot import load_sources
from scripts.run_recovery_trial import run_recovery_trial
from scripts.run_restart_trial import run_restart_trial
from scripts.run_timed_tasks import RolloutDeadline, wait_for_idle
from wll.appworld import AppWorldEnvironment
from wll.gpu_idle import validate_ticket
from wll.model import LocalModel
from wll.storage import atomic_json, require_space

ID = 'qwen38-27b-recovery-001'
PORT = 18979
PLAN = 'artifacts/qwen38-recovery-plan-001.json'
DESIGN = 'artifacts/qwen38-recovery-design-001.json'
SOURCE_PLAN = 'artifacts/qwen38-source-capability-plan-001.json'
SOURCE_SHA = '07e1059d459d6da2b74943d246975ab6da543b1fb16b35f3915f1669428a2e55'
INPUTS = 'artifacts/qwen38-recovery-inputs-001.json'
INPUTS_SHA = '16a40fe2a3d715bc2687ea86bf2ff59ec83a74b1778071fb82e801568b5811b6'
SOURCE_AUDIT = 'artifacts/qwen38-source-complete-integration-001.json'
SOURCE_AUDIT_SHA = 'eb3e1306abc7a0ab5fa90c4af6ba8266948e115e9fca04353035fc48c51f1caf'


def validate_study(root, plan):
    root = Path(root)
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['design'] == DESIGN
            and plan['design_sha256'] == digest(root / DESIGN), 'registered unchanged recovery design required')
    design = read(root / DESIGN)
    require(all(plan.get(k) == v for k, v in design.items() if k not in
                {'state', 'missing_before_registration'}), 'registered study configuration changed')
    require(plan['source_plan'] == SOURCE_PLAN and plan['source_plan_sha256'] == SOURCE_SHA
            and plan['source_inputs'] == INPUTS and plan['source_inputs_sha256'] == INPUTS_SHA
            and digest(root / INPUTS) == INPUTS_SHA and plan['source_integration'] == SOURCE_AUDIT
            and plan['source_integration_sha256'] == SOURCE_AUDIT_SHA
            and digest(root / SOURCE_AUDIT) == SOURCE_AUDIT_SHA, 'source identity changed')
    parent, _ = load_source_registration(root, root / SOURCE_PLAN, SOURCE_SHA)
    inputs, whole = read(root / INPUTS), read(root / SOURCE_AUDIT)
    state = read(root / 'artifacts' / parent['id'] / 'status.json')
    require(whole['result']['completed'] == 12 and whole['result']['advance_to_recovery_study'] is True
            and whole['result'] == state['result'] and state['phase'] == 'complete'
            and state['plan_sha256'] == SOURCE_SHA and whole['native_controller_and_forwarder_released']
            and all(whole['recorded_container_absence'].values()), 'complete released source gate required')
    require(inputs['status'] == 'prepared_not_registered' and inputs['source_count'] == 12
            and inputs['selected_count'] == 3 and inputs['source_plan_sha256'] == SOURCE_SHA
            and inputs['source_integration_sha256'] == SOURCE_AUDIT_SHA, 'source preparation differs')
    for name, sha in inputs['input_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and digest(root / name) == sha, 'prepared source file changed: ' + name)
    entries = inputs['selected']
    ids = [r['task_id'] for r in entries]
    failed = [r['task_id'] for r in whole['outcomes'] if r['success'] is False]
    require(ids == failed == plan['task_ids'] and plan['batches'] == [ids]
            and plan['trial_modes'] == {r['task_id']: r['mode'] for r in entries}
            and plan['environment_id'] == inputs['environment_id'] == parent['environment_id']
            and plan['expected_backend_static'] == parent['expected_backend_static']
            and plan['gpu_policy'] == parent['gpu_policy']
            and plan['service_plan_sha256'] == SERVICE_SHA, 'source selection, environment or backend differs')
    for source, record in zip(load_sources(root, entries), entries, strict=True):
        require(source.success is False and source.task.split == 'train' and source.branch == 'first'
                and source.replayed_steps == 0 and source.experience_id is None
                and source.seed == parent['seed'], 'only completed natural first failures are permitted')
        error = any(s.observation.startswith('Execution failed') for s in source.steps)
        if record['mode'] == 'fresh_restart':
            require(not error and record['prefix_proof'] is None and record['remaining_steps'] == 40
                    and record['failed_source_visible_to_actor'] is False, 'invalid blind restart source')
        elif record['mode'] == 'prefix_continuation':
            selected = select_sources([source], [record['prefix_proof']], max_steps=40)['selected']
            require(len(selected) == 1 and all(record[k] == v for k, v in selected[0].items()),
                    'unverified or changed continuation prefix')
        else:
            raise ValueError('unknown followup mode')
    return parent, inputs


def load_registration(root, path, sha):
    root = Path(root)
    plan = verify_frozen_plan(root, path, sha)
    parent, inputs = validate_study(root, plan)
    required = set(parent['source_sha256']) | {'scripts/run_qwen38_recovery.py',
        'scripts/run_recovery_pilot.py', 'scripts/run_recovery_trial.py', 'scripts/report_recovery_trial.py',
        'scripts/run_restart_trial.py', 'scripts/report_restart_trial.py', 'scripts/prepare_recovery_inputs.py',
        'scripts/audit_recovery_prefixes.py', 'scripts/report_observation_comparison.py'}
    require(required <= plan['source_sha256'].keys(), 'recovery source bindings incomplete')
    require(all(plan['source_sha256'].get(k) == v for k, v in parent['source_sha256'].items()),
            'frozen source implementation changed')
    required_inputs = {DESIGN, SOURCE_PLAN, INPUTS, SOURCE_AUDIT, 'artifacts/restart-baseline-cpu-verification-001.json'}
    require(required_inputs <= plan['input_sha256'].keys(), 'recovery input bindings incomplete')
    for bindings in (parent['input_sha256'], inputs['input_sha256']):
        require(all(plan['input_sha256'].get(k) == v for k, v in bindings.items()), 'parent evidence binding differs')
    return plan, parent, inputs


def make_job(owner, plan_sha):
    require(re.fullmatch('[a-f0-9]{32}', owner) and re.fullmatch('[a-f0-9]{64}', plan_sha),
            'invalid queue owner or plan hash')
    job_id = ID + '-b0'
    return {'id': job_id, 'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 5400,
        'argv': [PYTHON, '-B', '-m', 'scripts.serve_qwen38', '--plan-sha256', SERVICE_SHA,
                 '--model-id', job_id, '--port', str(PORT)]}


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
                raise FileExistsError('followup batch already attempted or owned')
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
from scripts.run_qwen38_recovery import queue_operation
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


def trial_protocol(source, record, plan, parent, health):
    if record['mode'] == 'prefix_continuation':
        return prefix_protocol(source, record, plan, parent, health)
    require(record['mode'] == 'fresh_restart', 'unknown trial mode')
    return {'source_hash': record['source_hash'], 'source_seed': source.seed,
        'source_termination': source.termination, 'task_id': source.task.task_id,
        'model_id': health['model'], 'max_steps': plan['restart_max_steps'],
        'max_tokens': plan['max_tokens'], 'temperature': plan['temperature'],
        'restart_seed': plan['restart_seed'], 'draft_seed': plan['draft_seed'],
        'rollout_seconds': plan['restart_seconds'], 'draft_seconds': plan['draft_seconds'],
        'prompt_version': parent['prompt_version'], 'environment_id': plan['environment_id'],
        'expected_backend': health['backend'], 'source_backend': source.metadata['backend'],
        'script_sha256': plan['source_sha256']['scripts/run_restart_trial.py'],
        'wll_code_fingerprint': plan['wll_code_fingerprint'], 'backend_idle_check_enabled': True}


def audit_batch(stage, sources, entries, expected):
    trial_root = stage / 'trials'
    ids = [s.task.task_id for s in sources]
    require(ids == [r['task_id'] for r in entries] and len(ids) == len(set(ids)), 'trial source order differs')
    present = {p.name for p in trial_root.iterdir() if p.is_dir()} if trial_root.exists() else set()
    require(present == set(ids[:len(present)]), 'trial directories are not an ordered source prefix')
    reports, outcomes = {}, {}
    by_mode = {mode: {'scheduled': sum(e['mode'] == mode for e in entries), 'known': 0, 'successes': 0}
               for mode in ('fresh_restart', 'prefix_continuation')}
    phase_cost = {phase: {'prompt_tokens': 0, 'completion_tokens': 0} for phase in ('followup', 'draft')}
    for source, entry in zip(sources, entries, strict=True):
        if source.task.task_id not in present:
            continue
        require(all(r['valid_for_continuation'] for r in reports.values()), 'continued after an incomplete trial')
        mode, task_id = entry['mode'], source.task.task_id
        require(mode in by_mode, 'unknown trial mode')
        prefix = mode == 'prefix_continuation'
        report = (audit_prefix if prefix else audit_restart)(trial_root / task_id, source, expected[task_id])
        outcome = report['recovery_success' if prefix else 'restart_success']
        reports[task_id] = {'mode': mode, **report}
        outcomes[task_id] = outcome
        by_mode[mode]['known'] += type(outcome) is bool
        by_mode[mode]['successes'] += outcome is True
        for phase, key in [('followup', 'recovery' if prefix else 'restart'), ('draft', 'draft')]:
            for token in ('prompt_tokens', 'completion_tokens'):
                phase_cost[phase][token] += report['phase_response_cost'][key][token]
    known = sum(type(v) is bool for v in outcomes.values())
    complete = len(reports) == len(ids) and all(r['valid_for_continuation'] for r in reports.values())
    return {'status': 'complete' if complete else 'incomplete', 'scheduled_sources': len(ids),
        'attempted_sources': len(reports), 'unattempted_sources': len(ids) - len(reports),
        'known_followups': known, 'unknown_followups': len(reports) - known, 'by_mode': by_mode,
        'unvalidated_candidates': sum(r['candidate_count'] for r in reports.values()),
        'valid_for_continuation': complete, 'trials': reports,
        'replayed_observations': sum(r['replayed_observations'] for r in reports.values()),
        'new_step_cost': {k: sum(r['new_step_cost'][k] for r in reports.values())
                          for k in ('steps', 'prompt_tokens', 'completion_tokens')},
        'phase_response_cost': phase_cost, 'automatic_utility_or_training': False,
        'scope': 'Extra sampling; modes have different state and additional budgets. Not a causal mode '
                 'comparison or first-attempt performance. Candidates remain unvalidated. Phase costs '
                 'are disjoint, step costs overlap, source/replay token fields are historical.'}


def worker(root, plan, parent, inputs, stage):
    health = read(stage / 'health.json')
    validate_health(health, plan)
    entries = inputs['selected']
    require([r['task_id'] for r in entries] == plan['task_ids'], 'worker source order differs')
    sources = load_sources(root, entries)
    expected = {s.task.task_id: trial_protocol(s, r, plan, parent, health)
                for s, r in zip(sources, entries, strict=True)}
    require(read(stage / 'expected-trials.json') == expected, 'worker expected protocols differ')
    endpoint = f"http://127.0.0.1:{plan['port']}/v1"
    model = LocalModel(endpoint, health['model'], max_tokens=plan['max_tokens'], temperature=plan['temperature'])
    env = read(root / ENV_PLAN)
    with (stage / 'containers.jsonl').open('x') as resources, (stage / 'feedback-events.jsonl').open('x') as feedback:
        def factory(task):
            world = AppWorldEnvironment(task.task_id, parent['data_dir'], plan['max_steps_including_replay'],
                                        parent['image'], parent['deps_dir'])
            try:
                identifier = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}',
                    world.worker.name], text=True, timeout=15).strip()
                resources.write(json.dumps({'task_id': task.task_id, 'name': world.worker.name, 'id': identifier}) + '\n')
                resources.flush()
                return observation_world(world, env, task.task_id, feedback)
            except BaseException:
                world.close()
                raise

        try:
            for source, entry in zip(sources, entries, strict=True):
                common = {'output': stage / 'trials' / source.task.task_id,
                    'draft_seconds': plan['draft_seconds'], 'draft_seed': plan['draft_seed'],
                    'expected_backend': health['backend'], 'environment_id': plan['environment_id'],
                    'before_phase': lambda phase: wait_for_idle(endpoint, health['model'], health['backend'],
                                                                seconds=plan['server_idle_seconds'])}
                if entry['mode'] == 'prefix_continuation':
                    result = run_recovery_trial(source, entry['prefix_proof'], factory, model,
                        max_steps=plan['max_steps_including_replay'], rollout_seconds=plan['recovery_seconds'],
                        recovery_seed=plan['recovery_seed'], **common)
                else:
                    require(entry['mode'] == 'fresh_restart', 'unknown worker trial mode')
                    result = run_restart_trial(source, factory, model, max_steps=plan['restart_max_steps'],
                        rollout_seconds=plan['restart_seconds'], restart_seed=plan['restart_seed'], **common)
                atomic_json(stage / 'report.json', audit_batch(stage, sources, entries, expected))
                if result['status'] != 'complete' or result['cleanup_verified'] is not True:
                    return 1
        finally:
            atomic_json(stage / 'report.json', audit_batch(stage, sources, entries, expected))
    return 0


def verify_remote(plan_sha):
    return remote_python('''import json,sys,subprocess,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));p=json.load(sys.stdin)
from scripts.run_qwen38_recovery import load_registration,PLAN,PYTHON,ID,SERVICE_SHA
plan,parent,inputs=load_registration(root,root/PLAN,p['plan_sha'])
r=subprocess.run([PYTHON,'-B','-m','scripts.serve_qwen38','--check','--model-id',ID+'-b0','--plan-sha256',SERVICE_SHA],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),capture_output=True,text=True,timeout=15,check=True)
assert json.loads(r.stdout)['preflight']=='passed'
for name in [f'gpu-idle-job-{ID}-b0.json',f'gpu-idle-ticket-{ID}-b0.json',f'gpu-idle-ticket-{ID}-b0.consumed.json',f'gpu-idle-heartbeat-{ID}-b0']:
 assert not (root/'artifacts'/name).exists(),name
print(json.dumps({'preflight':'passed','uses_gpu':False,'sources':len(inputs['selected']),'source_bindings':len(plan['source_sha256'])}))
''', {'plan_sha': plan_sha})


def collect_remote_evidence(job, known, stage):
    payload = remote_python('''import json,sys,hashlib
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab/artifacts');p=json.load(sys.stdin);id=p['job']['id']
names=[f'gpu-idle-job-{id}.json',f'gpu-idle-job-{id}.log',f'gpu-idle-ticket-{id}.consumed.json']
names += [f'gpu9-admission-{pid}.json' for pid in p['known']]
files={};missing=[]
for name in names:
 path=root/name
 if not path.exists():missing.append(name);continue
 data=path.read_bytes();assert len(data)<=16000000
 files[name]={'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()}
print(json.dumps({'files':files,'missing':missing}))
''', {'job': job, 'known': sorted(known)})
    folder = stage / 'remote-artifacts'
    folder.mkdir()
    for name, item in payload['files'].items():
        require(Path(name).name == name, 'invalid remote artifact path')
        with (folder / name).open('x') as stream:
            stream.write(item['text'])
        require(digest(folder / name) == item['sha256'], 'remote evidence changed during transfer')
    atomic_json(folder / 'manifest.json', {'sha256': {n: r['sha256'] for n, r in payload['files'].items()},
                                         'missing': payload['missing']})
    if (stage / 'report.json').exists() and read(stage / 'report.json')['valid_for_continuation']:
        require(not payload['missing'] and len(known) == 1, 'complete batch lacks admission evidence')
        info = read(folder / f"gpu-idle-job-{job['id']}.json")
        admission = read(folder / f'gpu9-admission-{next(iter(known))}.json')
        ticket = read(folder / f"gpu-idle-ticket-{job['id']}.consumed.json")
        policy = read(ROOT / 'config/gpu-policy.json')
        validate_ticket(ticket, policy, info['gpu_index'], digest(ROOT / 'config/gpu-policy.json'),
                        ticket['boot_id'], ticket['qualified_at_monotonic'])
        require(ticket == info['ticket'] == admission['ticket'] and ticket['job_id'] == job['id']
                and admission['gpu_uuid'] == read(stage / 'health.json')['backend']['gpu_policy']['gpu_uuid'],
                'admission differs from live service')


def run_batch(root, plan, parent, inputs, plan_path, plan_sha):
    require(load_registration(root, plan_path, plan_sha) == (plan, parent, inputs), 'in-memory configuration changed')
    remote_check = verify_remote(plan_sha)
    stage = root / 'artifacts' / ID / 'batch-0'
    stage.mkdir()
    atomic_json(stage / 'freeze-check.json', {'plan_sha256': plan_sha, 'checked_epoch': time.time(),
                                             'local_files_match': True, 'remote': remote_check})
    entries = inputs['selected']
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
        expected = {s.task.task_id: trial_protocol(s, r, plan, parent, health)
                    for s, r in zip(sources, entries, strict=True)}
        atomic_json(stage / 'expected-trials.json', expected)
        with (stage / 'client.log').open('x') as log:
            client = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'scripts.run_qwen38_recovery',
                '--plan', str(plan_path), '--plan-sha256', plan_sha, '--worker', '--stage', str(stage)],
                cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root), TMPDIR=plan['tmp_dir']),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = client.wait(timeout=plan['client_seconds_per_batch'])
        try:
            report = audit_batch(stage, sources, entries, expected)
            atomic_json(stage / 'report.json', report)
        except Exception as exc:
            atomic_json(stage / 'report-failure.json', {'client_exit_code': code,
                'error_type': type(exc).__name__, 'success_rate': None,
                'reason': 'Raw followup artifacts incomplete or inconsistent; no outcome imputed'})
            raise
        require(code == 0 and report['valid_for_continuation'], 'followup batch incomplete; no automatic retry')
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
        require(not output.exists(), 'followup study has an attempt; no automatic restart')
        require_space(ROOT / 'artifacts')
        actual_image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                                plan['image']], text=True, timeout=15).strip()
        require(actual_image == plan['image'], 'execution image changed')
        verify_remote(args.plan_sha256)
        if args.check:
            print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'batches': 1, 'sources': 3}))
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
