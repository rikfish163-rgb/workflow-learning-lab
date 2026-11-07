"""Separate27B source capability controller; reuse frozen task and cleanup primitives."""
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

from scripts.report_timed_tasks import audit_run
from scripts.run_foundation_pilot import (
    combine_batches,
    digest,
    environment_id,
    observation_world,
    process_matches,
    read,
    require,
    require_rollout_artifacts,
    stop_client,
    verify_cpu_environment,
    verify_frozen_plan,
)
from scripts.run_queued_decoder_completion import SSH, remote_python
from scripts.run_timed_tasks import RolloutDeadline, run_tasks, wait_for_idle
from wll.appworld import AppWorldEnvironment, load_tasks
from wll.gpu_idle import validate_ticket
from wll.model import LocalModel
from wll.storage import atomic_json, require_space

ROOT = Path(__file__).resolve().parent.parent
REMOTE = Path('/home/hts/workflow-learning-lab')
PYTHON = '/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python'
ID = 'qwen38-27b-source-capability-001'
MODULE = 'scripts.serve_qwen38'
PORT = 18979
PLAN = 'artifacts/qwen38-source-capability-plan-001.json'
DESIGN = 'artifacts/qwen38-source-capability-design-001.json'
ENV_PLAN = 'artifacts/qwen35-9b-observation-plan-001.json'
SERVICE_PLAN = 'artifacts/qwen38-service-plan-001.json'
SERVICE_SHA = '0f92c4a1558655ebbae6ce5ace64990425596e5172942794316eae0ff1ae7fe8'
SERVICE_AUDIT = 'artifacts/qwen38-service-integration-001.json'
SERVICE_RESULT = 'artifacts/qwen38-27b-service-smoke-001.json'
TASKS = 'artifacts/foundation-inputs-001/tasks.jsonl'
PARTITION = 'artifacts/partitions-v1/manifest.json'


def validate_study(root, plan):
    root = Path(root)
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['design'] == DESIGN
            and plan['design_sha256'] == digest(root / DESIGN), 'unregistered or changed design')
    design = read(root / DESIGN)
    fields = {'task_ids', 'data_dir', 'deps_dir', 'tmp_dir', 'image', 'prompt_version', 'seed',
              'max_tokens', 'temperature', 'max_steps', 'rollout_seconds', 'model_start_seconds',
              'client_seconds', 'gpu_seconds', 'server_idle_seconds', 'environment_id',
              'wll_code_fingerprint', 'observation_feedback', 'max_context', 'port',
              'service_plan', 'service_plan_sha256', 'service_integration', 'service_integration_sha256',
              'batches', 'task_selection', 'partition', 'partition_sha256', 'tasks_sha256',
              'expected_backend_static', 'success_gate', 'stop_rule', 'context_budget_rule',
              'automatic_recovery_or_training'}
    require(all(plan.get(k) == design[k] for k in fields), 'registered study configuration changed')
    require(plan['service_plan'] == SERVICE_PLAN and plan['service_plan_sha256'] == SERVICE_SHA
            and digest(root / SERVICE_PLAN) == SERVICE_SHA and plan['max_context'] == 16384,
            'unmeasured service or context')
    audit, result = read(root / SERVICE_AUDIT), read(root / SERVICE_RESULT)
    require(audit['completion_gate'] is True and audit['model_group_and_controller_released'] is True
            and audit['plan_sha256'] == SERVICE_SHA and audit['result_sha256'] == digest(root / SERVICE_RESULT)
            and result['phase'] == 'complete' and len(result['cases']) == 3,
            'incomplete or changed service check')
    require(plan['expected_backend_static'] == {k: v for k, v in result['backend'].items() if k != 'gpu_policy'},
            'different measured serving backend')
    env = read(root / ENV_PLAN)
    require(environment_id(env) == plan['environment_id'] and env['image'] == plan['image']
            and env['observation_feedback'] == plan['observation_feedback'], 'environment recipe differs')
    require(plan['gpu_policy'] == read(root / 'config/gpu-policy.json')
            and plan['gpu_policy_sha256'] == digest(root / 'config/gpu-policy.json'), 'GPU policy changed')
    tasks = [json.loads(line) for line in (root / TASKS).read_text().splitlines()]
    origin = read(root / PARTITION)['task_ids']['origin']
    first = sorted(min(t for t in origin if t.rsplit('_', 1)[0] == s)
                   for s in {t.rsplit('_', 1)[0] for t in origin})
    require(plan['task_ids'] == [t['task_id'] for t in tasks] == first and len(first) == 12
            and digest(root / TASKS) == plan['tasks_sha256'], 'source selection changed')
    return tasks


def load_registration(root, path, sha):
    root = Path(root)
    plan = verify_frozen_plan(root, path, sha)
    required = {'scripts/run_qwen38_capability.py', 'scripts/run_foundation_pilot.py',
                'scripts/run_timed_tasks.py', 'scripts/report_timed_tasks.py',
                'scripts/documentation_feedback.py', 'scripts/probe_observation_visibility.py',
                'scripts/run_queued_decoder_completion.py', 'config/gpu-policy.json'}
    required.update(str(p.relative_to(root)) for p in (root / 'wll').glob('*.py'))
    require(required <= plan['source_sha256'].keys(), 'incomplete source bindings')
    service = read(root / SERVICE_PLAN)
    require(all(plan['source_sha256'].get(k) == v for k, v in service['source_sha256'].items()),
            'service sources not frozen')
    require({DESIGN, ENV_PLAN, SERVICE_PLAN, SERVICE_AUDIT, SERVICE_RESULT, TASKS, PARTITION}
            <= plan['input_sha256'].keys(), 'incomplete input bindings')
    return plan, validate_study(root, plan)


def make_job(index, owner, plan_sha):
    require(type(index) is int and index in (0, 1, 2) and re.fullmatch('[a-f0-9]{32}', owner)
            and re.fullmatch('[a-f0-9]{64}', plan_sha), 'invalid batch, owner or plan hash')
    job_id = f'{ID}-b{index}'
    return {'id': job_id, 'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 5400,
            'argv': [PYTHON, '-B', '-m', MODULE, '--plan-sha256', SERVICE_SHA,
                     '--model-id', job_id, '--port', str(PORT)]}


def model_processes(job):
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / 'cmdline').read_bytes().split(b'\0')[:-1]
            if args != [x.encode() for x in job['argv']]:
                continue
            env = dict(x.split(b'=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
            found.append({'pid': int(proc.name), 'uid': proc.stat().st_uid,
                          'start_ticks': (proc / 'stat').read_text().split(')', 1)[1].split()[19],
                          'process_group': os.getpgid(int(proc.name)), 'cwd': str((proc / 'cwd').resolve()),
                          'cuda_visible_devices': env.get(b'CUDA_VISIBLE_DEVICES', b'').decode()})
        except (OSError, IndexError, UnicodeError):
            continue
    return sorted(found, key=lambda x: x['pid'])


def runtime_state(job, art, known_pids=()):
    info_path, watch_path = art / f"gpu-idle-job-{job['id']}.json", art / 'gpu-idle-watch-status.json'
    info, watch = read(info_path) if info_path.exists() else None, read(watch_path) if watch_path.exists() else None
    live, monitor_live = model_processes(job), False
    monitor_path = art / 'gpu-idle-watch-process.json'
    if watch and monitor_path.exists():
        monitor = read(monitor_path)
        try:
            proc = Path('/proc') / str(monitor['pid'])
            monitor_live = (watch['pid'] == monitor['pid'] and proc.stat().st_uid == os.getuid()
                and 0 <= time.time() - watch['checked_epoch'] < 35
                and (proc / 'stat').read_text().split()[21] == monitor['start_ticks'])
        except (OSError, IndexError):
            pass
    output = {'job': info, 'watch': watch, 'epoch': time.time(), 'model_processes': live,
              'model_pids': [x['pid'] for x in live], 'monitor_live': monitor_live, 'released': False}
    if info and info['status'] == 'finished' and not live and not process_matches(info, job['argv']):
        try:
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'], text=True, timeout=15)
            occupied = {int(line.strip()) for line in raw.splitlines() if line.strip()}
            output['released'] = not occupied.intersection([*known_pids, info['process_group']])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            output['gpu_release_query_error'] = type(exc).__name__
    return output


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'invalid queue operation')
    match = re.fullmatch(re.escape(ID) + r'-b([012])', job['id'])
    require(match is not None, 'different queued study')
    require(job == make_job(int(match[1]), job['owner'], job['plan_sha256']), 'changed queued command')
    art = Path(root) / 'artifacts'
    queue, heartbeat = art / 'gpu-idle-queue.json', art / f"gpu-idle-heartbeat-{job['id']}"
    result = art / f"gpu-idle-job-{job['id']}.json"
    with (art / 'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = read(queue) if queue.exists() else None
        if operation == 'enqueue':
            markers = [result, heartbeat, art / f"gpu-idle-ticket-{job['id']}.json",
                       art / f"gpu-idle-ticket-{job['id']}.consumed.json"]
            if any(p.exists() for p in markers):
                raise FileExistsError('batch already attempted or owned')
            if current:
                require(re.fullmatch('[a-z0-9_-]+', current['id']), 'invalid previous queue ID')
                old = art / f"gpu-idle-job-{current['id']}.json"
                require(old.exists() and read(old)['status'] == 'finished'
                        and not (art / f"gpu-idle-heartbeat-{current['id']}").exists(), 'previous queue active or owned')
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
from scripts.run_qwen38_capability import queue_operation
d=json.load(sys.stdin)
print(json.dumps(queue_operation(d['operation'],d['job'],known_pids=d['known_pids'])))
''', {'operation': operation, 'job': job, 'known_pids': list(known_pids)})


def validate_health(health, plan, index):
    require(health.get('model') == f'{ID}-b{index}' and health.get('ready') is True
            and health.get('busy') is False and health.get('max_context') == 16384
            and health.get('max_output_tokens') == 1024, 'unexpected model health or context')
    backend = health['backend']
    require({k: v for k, v in backend.items() if k != 'gpu_policy'} == plan['expected_backend_static'],
            'serving backend changed')
    gpu, policy = backend['gpu_policy'], plan['gpu_policy']
    require(gpu['hostname'] == policy['hostname'] and type(gpu['physical_gpu_index']) is int
            and str(gpu['physical_gpu_index']) in policy['allowed_gpus']
            and gpu['gpu_uuid'] == policy['allowed_gpus'][str(gpu['physical_gpu_index'])]
            and gpu['visible_device_count'] == 1 and gpu['policy_sha256'] == plan['gpu_policy_sha256'],
            'unauthorized GPU backend')


def validate_live_service(snapshot, health):
    processes = snapshot['model_processes']
    require(snapshot['owned'] and snapshot['job']['status'] == 'running' and len(processes) == 1
            and snapshot['model_pids'] == [processes[0]['pid']], 'live service identity missing')
    proc = processes[0]
    require(proc['uid'] == 1031 and proc['cwd'] == str(REMOTE)
            and proc['process_group'] == snapshot['job']['process_group']
            and proc['cuda_visible_devices'] == health['backend']['gpu_policy']['gpu_uuid'],
            'live service directory, owner or CUDA visibility differs')


def expected_protocol(plan, tasks, health):
    return {'tasks': tasks, 'model_id': health['model'], 'max_tokens': plan['max_tokens'],
            'temperature': plan['temperature'], 'seed': plan['seed'], 'max_steps': plan['max_steps'],
            'rollout_seconds': plan['rollout_seconds'], 'prompt_version': plan['prompt_version'],
            'environment_id': plan['environment_id'], 'expected_backend': health['backend'],
            'script_sha256': plan['source_sha256']['scripts/run_timed_tasks.py'],
            'wll_code_fingerprint': plan['wll_code_fingerprint'], 'backend_idle_check_enabled': True,
            'deadline_scope': 'rollout only; environment startup and cleanup separately bounded'}


def verify_remote(plan, first_batch):
    return remote_python('''import json,sys,subprocess,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));p=json.load(sys.stdin)
from scripts.run_qwen38_capability import load_registration,PLAN,PYTHON,ID,SERVICE_SHA
plan,tasks=load_registration(root,root/PLAN,p['plan_sha'])
result=subprocess.run([PYTHON,'-B','-m','scripts.serve_qwen38','--check','--model-id',ID+'-b0',
 '--plan-sha256',SERVICE_SHA],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),
 capture_output=True,text=True,timeout=15,check=True)
assert json.loads(result.stdout)['preflight']=='passed'
for i in range(p['first_batch'],3):
 for name in [f'gpu-idle-job-{ID}-b{i}.json',f'gpu-idle-ticket-{ID}-b{i}.json',f'gpu-idle-ticket-{ID}-b{i}.consumed.json',f'gpu-idle-heartbeat-{ID}-b{i}']:
  assert not (root/'artifacts'/name).exists(),name
print(json.dumps({'preflight':'passed','uses_gpu':False,'tasks':len(tasks),'source_bindings':len(plan['source_sha256'])}))
''', {'plan_sha': digest(ROOT / PLAN), 'first_batch': first_batch})


def worker(root, plan, index, stage):
    health = read(stage / 'health.json')
    validate_health(health, plan, index)
    ids = plan['task_ids'][index * 4:index * 4 + 4]
    tasks = load_tasks(plan['data_dir'], 'train', image=plan['image'], deps_dir=plan['deps_dir'], include_task_ids=ids)
    by_id = {t.task_id: t for t in tasks}
    tasks = [by_id[t] for t in ids]
    frozen = [json.loads(line) for line in (root / TASKS).read_text().splitlines()]
    require([asdict(t) for t in tasks] == frozen[index * 4:index * 4 + 4], 'live task descriptions changed')
    env = read(root / ENV_PLAN)
    require(environment_id(env) == plan['environment_id'], 'different observation wrapper')
    with (stage / 'containers.jsonl').open('x') as resources, (stage / 'feedback-events.jsonl').open('x') as feedback:
        def factory(task):
            world = AppWorldEnvironment(task.task_id, plan['data_dir'], plan['max_steps'], plan['image'], plan['deps_dir'])
            try:
                container = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}',
                                                     world.worker.name], text=True, timeout=15).strip()
                resources.write(json.dumps({'name': world.worker.name, 'id': container}) + '\n')
                resources.flush()
                return observation_world(world, env, task.task_id, feedback)
            except BaseException:
                world.close()
                raise

        endpoint = f"http://127.0.0.1:{plan['port']}/v1"
        result = run_tasks(tasks, factory, LocalModel(endpoint, health['model'], max_tokens=plan['max_tokens'],
            temperature=plan['temperature']), output=stage / 'run', max_steps=plan['max_steps'],
            rollout_seconds=plan['rollout_seconds'], seed=plan['seed'], expected_backend=health['backend'],
            prompt_version=plan['prompt_version'], environment_id=plan['environment_id'],
            before_task=lambda task: wait_for_idle(endpoint, health['model'], health['backend'],
                                                    seconds=plan['server_idle_seconds']))
    return 0 if result['valid_for_comparison'] else 1


def collect_remote_evidence(job, known, stage):
    payload = remote_python('''import json,sys,hashlib
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab/artifacts');p=json.load(sys.stdin);job=p['job'];id=job['id']
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
        require(Path(name).name == name, 'invalid remote artifact name')
        with (folder / name).open('x') as f:
            f.write(item['text'])
        require(digest(folder / name) == item['sha256'], 'remote artifact transfer changed')
    atomic_json(folder / 'manifest.json', {'sha256': {n: i['sha256'] for n, i in payload['files'].items()},
                                         'missing': payload['missing']})
    if (stage / 'report.json').exists() and read(stage / 'report.json')['valid_for_comparison']:
        require(not payload['missing'] and len(known) == 1, 'complete batch lacks admission evidence')
        info = read(folder / f"gpu-idle-job-{job['id']}.json")
        admission = read(folder / f'gpu9-admission-{next(iter(known))}.json')
        ticket = read(folder / f"gpu-idle-ticket-{job['id']}.consumed.json")
        policy = read(ROOT / 'config/gpu-policy.json')
        validate_ticket(ticket, policy, info['gpu_index'], digest(ROOT / 'config/gpu-policy.json'),
                        ticket['boot_id'], ticket['qualified_at_monotonic'])
        require(ticket == info['ticket'] == admission['ticket'] and ticket['job_id'] == job['id']
                and admission['gpu_uuid'] == read(stage / 'health.json')['backend']['gpu_policy']['gpu_uuid'],
                'batch admission differs from live service')


def run_batch(plan, tasks, index, root, plan_path, *, plan_sha):
    verify_frozen_plan(root, plan_path, plan_sha, expected_plan=plan)
    remote_check = verify_remote(plan, index)
    stage = root / 'artifacts' / ID / f'batch-{index}'
    stage.mkdir()
    atomic_json(stage / 'freeze-check.json', {'plan_sha256': plan_sha, 'checked_epoch': time.time(),
                                             'local_files_match': True, 'remote': remote_check})
    verify_cpu_environment(plan, tasks[index * 4:index * 4 + 4], stage / 'cpu-environment-preflight.json')
    job = make_job(index, uuid.uuid4().hex, plan_sha)
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
        def redirect_request(self, *args, **kwargs):
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
            require(state.get('owned') is True, 'queue ownership changed')
            require(state.get('monitor_live') is True, 'monitor is not confirmed live')
            require(time.monotonic() - state['local_observed_monotonic'] <= 90, 'remote status stale')
            remote_job = state.get('job') or {}
            require(remote_job.get('status') != 'finished', 'server exited before health verification')
            if remote_job.get('status') == 'running':
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
        validate_health(health, plan, index)
        confirmed = remote_queue('status', job, known)
        validate_live_service(confirmed, health)
        known.update(confirmed['model_pids'])
        atomic_json(stage / 'activation.json', confirmed)
        atomic_json(stage / 'health.json', health)
        atomic_json(stage / 'expected.json', expected_protocol(plan, tasks[index * 4:index * 4 + 4], health))
        with (stage / 'client.log').open('x') as log:
            client = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.run_qwen38_capability',
                '--plan', str(plan_path), '--plan-sha256', plan_sha, '--worker', '--batch', str(index),
                '--stage', str(stage)], cwd=root,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root), TMPDIR=plan['tmp_dir']),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = client.wait(timeout=plan['client_seconds'])
        require_rollout_artifacts(stage, code)
        report = audit_run(stage / 'run', read(stage / 'expected.json'))
        atomic_json(stage / 'report.json', report)
        require(code == 0 and report['valid_for_comparison'], 'batch incomplete; no automatic retry')
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
    parser.add_argument('--batch', type=int, choices=(0, 1, 2))
    parser.add_argument('--stage')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only controller and worker required')
    path = Path(args.plan).resolve(strict=True)
    plan, tasks = load_registration(ROOT, path, args.plan_sha256)
    if args.worker:
        require(args.batch is not None and args.stage is not None, 'worker batch and stage required')
        raise SystemExit(worker(ROOT, plan, args.batch, Path(args.stage)))
    with (ROOT / 'artifacts/foundation-pilot-001.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = ROOT / 'artifacts' / ID
        require(not output.exists(), 'study has an attempt; no automatic restart')
        require_space(ROOT / 'artifacts')
        actual_image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                                plan['image']], text=True, timeout=15).strip()
        require(actual_image == plan['image'], 'execution image changed')
        verify_remote(plan, 0)
        if args.check:
            print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'batches': 3, 'tasks': len(tasks)}))
            return
        output.mkdir()
        state = {'phase': 'running', 'plan_sha256': args.plan_sha256, 'pid': os.getpid(),
                 'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
                 'started_epoch': time.time(), 'completed_batches': 0}
        atomic_json(output / 'status.json', state)
        try:
            reports, releases = [], []
            for index in range(3):
                state.update(current_batch=index, checked_epoch=time.time())
                atomic_json(output / 'status.json', state)
                report, release = run_batch(plan, tasks, index, ROOT, path, plan_sha=args.plan_sha256)
                reports.append(report)
                releases.append(release)
                state['completed_batches'] = len(reports)
            result = combine_batches(reports, releases)
            atomic_json(output / 'report.json', result)
            state.update(phase='complete', result=result)
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
