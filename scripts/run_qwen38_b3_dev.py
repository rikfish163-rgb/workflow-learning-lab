"""Fixed full-dev paired study: one sequential GPU lease and one CPU client.

The GPU monitor enforces admission and a 7200s cap. A separate CPU watchdog
enforces the client deadline and removes only durable, identity-checked owned
containers. Incomplete batches stop the study without retry or resampling.
"""
import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from scripts.owned_appworld_v3 import inspect_owned_container, remove_owned_container
from scripts.probe_chunked_prefill import digest, require
from scripts.run_foundation_pilot import process_matches, read, verify_frozen_plan
from scripts.run_queued_decoder_completion import SSH
from scripts.run_qwen38_capability import runtime_state, validate_live_service
from scripts.run_timed_tasks import RolloutDeadline
from scripts.run_timed_tasks_v3 import (
    SEED_PROTOCOL,
    environment_identity,
    feedback_recipe,
    load_prepared_tasks,
)
from scripts.serve_qwen38_b3 import MODEL_IDS
from wll.storage import atomic_json, require_space

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/home/hts/workflow-learning-lab')
PYTHON = '/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python'
ID = 'qwen38-b3-dev-001'
PLAN = 'artifacts/qwen38-b3-dev-plan-001.json'
DESIGN = 'artifacts/qwen38-b3-dev-design-001.json'
TASKS = 'artifacts/b3-dev-inputs-001/tasks.json'
SERVICE_SHA = 'd7f56521d101394c8559ae0604680dc75d1bd4e346a96032c1c1c94ec676bad5'
GATE = 'artifacts/qwen38-b3-inference-gate-integration-001.json'
PORT = 18979


def load_registration(root, sha):
    root = Path(root)
    plan = verify_frozen_plan(root, root/PLAN, sha)
    design = read(root/DESIGN)
    require(plan['id'] == ID and plan['status'] == 'registered'
            and plan['design_sha256'] == digest(root/DESIGN)
            and all(plan[k] == v for k, v in design.items() if k != 'status'), 'design changed')
    require(plan['gpu_seconds'] == 7200 and plan['model_start_seconds'] == 900
            and plan['client_seconds'] == 5760 and plan['cleanup_seconds'] == 300
            and plan['automatic_retry'] is plan['continue_after_incomplete_batch'] is False,
            'registered lifecycle changed')
    ids = plan['task_ids']
    require(len(ids) == len(set(ids)) == 57 and plan['schedule'] == [
        {'arm': arm, 'batch_index': i, 'task_ids': ids[i*6:i*6+6]}
        for i in range(10) for arm in ('B0', 'B3')], 'fixed schedule changed')
    gate = read(root/GATE)
    require(gate['phase'] == 'complete' and gate['all_resources_released'] is True
            and gate['plan_sha256'] == SERVICE_SHA and gate['GPU_attempts'] == 2,
            'matched inference gate not complete')
    require(plan['gpu_policy'] == read(root/'config/gpu-policy.json')
            and plan['gpu_policy_sha256'] == digest(root/'config/gpu-policy.json')
            and plan['gpu_policy']['idle_seconds'] == 180, 'resource policy changed')
    require(plan['tasks_sha256'] == plan['input_sha256'][TASKS]
            and plan['protocol']['environment_id'] == environment_identity(plan['protocol']['environment_image_id']),
            'metadata/environment binding changed')
    return plan


def make_job(index, arm, owner, plan_sha):
    require(type(index) is int and 0 <= index < 10 and arm in MODEL_IDS
            and re.fullmatch('[a-f0-9]{32}', owner) and re.fullmatch('[a-f0-9]{64}', plan_sha),
            'invalid scheduled job')
    return {'id': f'{ID}-{arm.lower()}-{index}', 'arm': arm, 'batch_index': index,
        'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 7200,
        'argv': [PYTHON, '-B', '-m', 'scripts.serve_qwen38_b3', '--plan-sha256', SERVICE_SHA,
                 '--arm', arm, '--model-id', MODEL_IDS[arm], '--port', str(PORT)]}


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'unknown operation')
    require(job == make_job(job['batch_index'], job['arm'], job['owner'], job['plan_sha256']),
            'queued command changed')
    art = Path(root)/'artifacts'
    queue = art/'gpu-idle-queue.json'
    heartbeat = art/f"gpu-idle-heartbeat-{job['id']}"
    result = art/f"gpu-idle-job-{job['id']}.json"
    with (art/'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = read(queue) if queue.exists() else None
        if operation == 'enqueue':
            load_registration(root, job['plan_sha256'])
            markers = [heartbeat, result, art/f"gpu-idle-ticket-{job['id']}.json",
                       art/f"gpu-idle-ticket-{job['id']}.consumed.json"]
            if any(p.exists() for p in markers):
                raise FileExistsError('registered batch already attempted or owned')
            if current:
                require(re.fullmatch('[a-z0-9_-]+', current['id']), 'invalid previous queue')
                previous = runtime_state(current, art)
                require(previous['released'] and not (art/f"gpu-idle-heartbeat-{current['id']}").exists(),
                        'previous job not confirmed released')
                archive = art/f"gpu-idle-queue-archived-{current['id']}.json"
                if archive.exists():
                    require(archive.read_bytes() == queue.read_bytes(), 'archive changed')
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
                require(info['queue_sha256'] == digest(queue), 'different running queue')
                if process_matches(info, job['argv']):
                    os.killpg(info['process_group'], signal.SIGTERM)
    return {'owned': owned, **runtime_state(job, art, known_pids)}


def remote_call(source, payload=None):
    ssh = [v.replace('ConnectTimeout=8', 'ConnectTimeout=30') for v in SSH]
    result = subprocess.run([*ssh, 'CUDA_VISIBLE_DEVICES= python3 -B -c '+shlex.quote(source)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout)


def remote_queue(operation, job, known=()):
    return remote_call('''import json,sys,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));os.chdir(root)
from scripts.run_qwen38_b3_dev import queue_operation
p=json.load(sys.stdin)
print(json.dumps(queue_operation(p['operation'],p['job'],known_pids=p['known'])))
''', {'operation': operation, 'job': job, 'known': sorted(known)})


def static_backend(backend):
    return {k: v for k, v in backend.items() if k not in {'gpu_policy', 'loading_peak_reserved_bytes'}}


def validate_health(health, plan, arm):
    require(health.get('model') == MODEL_IDS[arm] and health.get('ready') is True
            and health.get('busy') is False and health.get('max_context') == 16384
            and health.get('max_output_tokens') == 1024, 'service health differs')
    backend = health['backend']
    require(static_backend(backend) == plan['expected_backend_static'][arm]
            and type(backend.get('loading_peak_reserved_bytes')) is int
            and 0 < backend['loading_peak_reserved_bytes'] <= 20*1024**3,
            'actual service recipe/adapter/loading differs')
    gpu, policy = backend['gpu_policy'], plan['gpu_policy']
    require(gpu['hostname'] == policy['hostname'] and type(gpu['physical_gpu_index']) is int
            and str(gpu['physical_gpu_index']) in policy['allowed_gpus']
            and gpu['gpu_uuid'] == policy['allowed_gpus'][str(gpu['physical_gpu_index'])]
            and gpu['visible_device_count'] == 1 and gpu['policy_sha256'] == plan['gpu_policy_sha256'],
            'unauthorized GPU')


def identity(pid):
    stat = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(stat[19])}


def live(who):
    try:
        path = Path('/proc')/str(who['pid'])
        stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return (path.stat().st_uid == os.getuid() and stat[0] not in {'Z', 'X'}
                and int(stat[19]) == who['start_ticks'])
    except (OSError, ValueError, IndexError):
        return False


def group_members(pid):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if int(stat[2]) == pid and int(stat[3]) == pid and stat[0] != 'Z':
                require(path.stat().st_uid == os.getuid(), 'foreign process in owned session')
                members.append(identity(int(path.name)))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return members


def stop_owned_client(process, who):
    require(process.pid == who['pid'], 'different client process')
    if process.poll() is None:
        require(live(who) and os.getpgid(process.pid) == process.pid
                and os.getsid(process.pid) == process.pid, 'client identity changed')
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            require(live(who), 'client PID reused')
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    # Docker CLI can outlive its client. Only this session/group's same-UID
    # members are signalled, with each PID/start-time checked immediately.
    for child in group_members(process.pid):
        if live(child):
            os.kill(child['pid'], signal.SIGKILL)
    deadline = time.monotonic()+10
    while group_members(process.pid) and time.monotonic() < deadline:
        time.sleep(.1)
    require(not live(who) and not group_members(process.pid), 'client/session remains alive')
    return {**who, 'verified_exited': True}


def cleanup_containers(stage, who):
    path = stage/'run/events.jsonl'
    intents, ready = [], set()
    if path.exists():
        for line in path.read_bytes().splitlines(keepends=True):
            require(line.endswith(b'\n'), 'truncated container intent journal')
            event = json.loads(line)
            if event.get('event') == 'container_intent':
                intent = event['container_intent']
                require(intent['owner_pid'] == who['pid'] and intent['owner_start_ticks'] == who['start_ticks'],
                        'container intent belongs to a different client')
                intents.append((event['task_id'], intent))
            elif event.get('event') == 'world_ready':
                ready.add(event['task_id'])
    removed = []
    for task_id, intent in intents:
        container_id = remove_owned_container(intent)
        removed.append(container_id)
        require(inspect_owned_container(intent) is None, 'owned container remains')
        # An interrupted Docker creation request may still be in the daemon.
        # Absence alone is not final proof when startup was never observed.
        require(container_id is not None or task_id in ready, 'container creation completion unknown')
    return {'verified': True, 'intents': [value for _, value in intents], 'removed_ids': removed,
            'scope': 'Client/session exited before exact label/image/ID cleanup; final name absence checked.'}


def client_command(plan, arm, index, stage):
    return [sys.executable, '-u', '-B', '-m', 'scripts.run_timed_tasks_v3', '--arm', arm,
        '--batch-index', str(index), '--task-manifest', str(ROOT/TASKS),
        '--task-manifest-sha256', plan['tasks_sha256'], '--data-dir', plan['data_dir'],
        '--deps-dir', plan['deps_dir'], '--image', plan['protocol']['environment_image_id'],
        '--backend-file', str(stage/'health.json'), '--endpoint', f'http://127.0.0.1:{PORT}/v1',
        '--model', MODEL_IDS[arm], '--output', str(stage/'run')]


def recover_client_after_guard_exit(plan, arm, index, stage, guard):
    """Parent cleanup when a dead watchdog could not run its own finally."""
    require(guard.poll() is not None, 'watchdog still owns cleanup')
    path = stage/'watchdog.json'
    record = read(path) if path.exists() else {'phase': 'watchdog_died_before_state'}
    if record.get('client_release', {}).get('verified_exited') is True:
        return
    expected = [arg.encode() for arg in client_command(plan, arm, index, stage)]

    def exact_client(pid):
        proc = Path('/proc')/str(pid)
        return (proc.stat().st_uid == os.getuid() and str((proc/'cwd').resolve()) == str(ROOT)
                and (proc/'cmdline').read_bytes().split(b'\0')[:-1] == expected
                and os.getsid(pid) == os.getpgid(pid) == pid)

    who = record.get('client_identity')
    if who is None:
        # Covers the short Popen-to-durable-client-record window. The exact
        # frozen command includes this unique stage; never match by name alone.
        candidates = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if exact_client(int(proc.name)):
                    candidates.append(identity(int(proc.name)))
            except (OSError, IndexError):
                pass
        require(len(candidates) == 1, 'client creation/identity remains unconfirmed')
        who = candidates[0]
        record['client_identity'] = who
    require(type(who) is dict and set(who) == {'pid', 'start_ticks'}
            and all(type(v) is int and v > 0 for v in who.values()), 'invalid durable client identity')
    if Path(f"/proc/{who['pid']}").exists():
        require(identity(who['pid']) == who, 'durable client PID reused')
    if live(who):
        require(exact_client(who['pid']), 'orphan client command, directory or session differs')
        os.killpg(who['pid'], signal.SIGTERM)
        deadline = time.monotonic()+15
        while live(who) and time.monotonic() < deadline:
            time.sleep(.1)
        if live(who):
            require(exact_client(who['pid']), 'orphan identity changed before kill')
            os.killpg(who['pid'], signal.SIGKILL)
    for child in group_members(who['pid']):
        if live(child):
            os.kill(child['pid'], signal.SIGKILL)
    deadline = time.monotonic()+10
    while (live(who) or group_members(who['pid'])) and time.monotonic() < deadline:
        time.sleep(.1)
    require(not live(who) and not group_members(who['pid']), 'orphan client/session remains')
    if path.exists():
        with (stage/'watchdog-before-controller-cleanup.json').open('xb') as stream:
            stream.write(path.read_bytes())
    record.update(client_release={**who, 'verified_exited': True}, release_checked_by_controller=True,
                  controller_cleanup_epoch=time.time(), controller_identity=identity(os.getpid()))
    try:
        record['container_release'] = cleanup_containers(stage, who)
    finally:
        atomic_json(path, record)


def watchdog(plan, arm, index, stage, parent):
    require(live(parent), 'controller absent before client startup')
    state = {'phase': 'starting', 'watchdog': identity(os.getpid()), 'parent': parent,
             'started_epoch': time.time(), 'client_seconds': plan['client_seconds']}
    atomic_json(stage/'watchdog.json', state)
    client = who = None
    try:
        with (stage/'client.log').open('x') as log:
            client = subprocess.Popen(client_command(plan, arm, index, stage), cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                         TMPDIR=plan['tmp_dir']), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            who = identity(client.pid)
            state.update(phase='running', client_identity=who)
            atomic_json(stage/'watchdog.json', state)
            deadline = time.monotonic()+plan['client_seconds']
            while client.poll() is None:
                require(live(parent), 'controller exited')
                require(time.monotonic() < deadline, 'CPU client hard deadline exceeded')
                time.sleep(1)
            state.update(phase='finished', client_exit_code=client.returncode)
    except BaseException as exc:
        state.update(phase='failed', error_type=type(exc).__name__, error=str(exc)[:1000])
        raise
    finally:
        try:
            if client is not None and who is not None:
                state['client_release'] = stop_owned_client(client, who)
                state['container_release'] = cleanup_containers(stage, who)
        finally:
            state['finished_epoch'] = time.time()
            atomic_json(stage/'watchdog.json', state)


def expected_manifest(plan, tasks, health, arm, index):
    from scripts.run_timed_tasks_v2 import OFFICIAL_DEV_SHA256, PREPARATION_SHA256

    return {'arm': arm, 'batch_index': index, 'slice': [index*6, min(index*6+6, 57)],
        'tasks': tasks, 'batch_task_ids': plan['task_ids'][index*6:index*6+6],
        'model_id': MODEL_IDS[arm], 'expected_backend': health['backend'],
        'image': plan['protocol']['environment_image_id'], 'environment_id': plan['protocol']['environment_id'],
        'data_dir': str(Path(plan['data_dir']).resolve()), 'observation_feedback': feedback_recipe(),
        'seed_protocol': SEED_PROTOCOL, 'prepared_manifest_sha256': plan['tasks_sha256'],
        'max_steps': 40, 'rollout_seconds': 600, 'seed': 17, 'max_tokens': 1024, 'temperature': .7,
        'prompt_version': 'appworld-v3', 'official_dev_sha256': OFFICIAL_DEV_SHA256,
        'preparation_sha256': PREPARATION_SHA256,
        'script_sha256': plan['source_sha256']['scripts/run_timed_tasks_v3.py'],
        'deadline_scope': 'rollout only; parent owns hard watchdog and independent release verification'}


def get_health():
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with RolloutDeadline(6), opener.open(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
        raw = response.read(262145)
    require(len(raw) <= 262144, 'oversized health response')
    return json.loads(raw)


def collect_remote(job, known, stage):
    payload = remote_call('''import json,sys,hashlib
from pathlib import Path
p=json.load(sys.stdin);root=Path('/home/hts/workflow-learning-lab/artifacts');id=p['job']['id']
names=['gpu-idle-job-'+id+'.json','gpu-idle-job-'+id+'.log','gpu-idle-ticket-'+id+'.consumed.json']
names.extend('gpu9-admission-'+str(pid)+'.json' for pid in p['known'])
files={};missing=[]
for name in names:
 path=root/name
 if not path.exists():missing.append(name);continue
 raw=path.read_bytes();assert len(raw)<=16000000
 files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
print(json.dumps({'files':files,'missing':missing}))
''', {'job': job, 'known': sorted(known)})
    folder = stage/'remote-artifacts'
    folder.mkdir()
    for name, value in payload['files'].items():
        require(Path(name).name == name, 'invalid evidence name')
        with (folder/name).open('x') as stream:
            stream.write(value['text'])
        require(digest(folder/name) == value['sha256'], 'transfer hash differs')
    atomic_json(folder/'manifest.json', {'sha256': {n: v['sha256'] for n, v in payload['files'].items()},
                                        'missing': payload['missing']})
    return payload['missing']


def verify_admission(stage, job, known, health):
    from wll.gpu_idle import validate_ticket

    require(len(known) == 1, 'ambiguous model process')
    raw = stage/'remote-artifacts'
    info = read(raw/f"gpu-idle-job-{job['id']}.json")
    ticket = read(raw/f"gpu-idle-ticket-{job['id']}.consumed.json")
    admission = read(raw/f'gpu9-admission-{next(iter(known))}.json')
    policy = read(ROOT/'config/gpu-policy.json')
    validate_ticket(ticket, policy, info['gpu_index'], digest(ROOT/'config/gpu-policy.json'),
                    ticket['boot_id'], ticket['qualified_at_monotonic'])
    require(ticket == info['ticket'] == admission['ticket'] and ticket['job_id'] == job['id']
            and admission['gpu_uuid'] == health['backend']['gpu_policy']['gpu_uuid']
            and admission['mode'] == 'continuous_idle_only' and admission['required_idle_seconds'] == 180
            and admission['allocator_limit_mib'] == 20480
            and info['queue_sha256'] == digest(stage/'job.json'), 'actual admission differs')
    return {'verified': True, 'job_seconds': info['finished_epoch']-info['started_epoch'],
            'gpu_index': info['gpu_index'], 'idle_seconds': ticket['idle_seconds'],
            'job_exit_code': info['exit_code']}


def audit_terminated_batch(stage):
    from scripts.audit_b3_dev_batch import audit_batch

    guard = read(stage/'watchdog.json')
    require(guard.get('client_release', {}).get('verified_exited') is True,
            'cannot audit a still-running client')
    if not (stage/'run/manifest.json').exists():
        return None
    report = audit_batch(stage/'run', expected_manifest=read(stage/'expected.json'),
                         terminated_identity=guard['client_release'])
    atomic_json(stage/'report.json', report)
    return report


def run_batch(plan, tasks, plan_sha, arm, index, output):
    load_registration(ROOT, plan_sha)
    for name, sha in plan['native_runtime_sha256'].items():
        require(digest(name) == sha, 'native AppWorld runtime or split changed')
    stage = output/f'{arm.lower()}-{index}'
    stage.mkdir()
    job = make_job(index, arm, uuid.uuid4().hex, plan_sha)
    atomic_json(stage/'job.json', job)
    state, known, stopped = {}, set(), threading.Event()
    thread = tunnel = guard = health = None
    guard_identity = None
    report = None
    costs = {'started_epoch': time.time(), 'phase': 'waiting_for_idle_GPU'}
    atomic_json(stage/'lifecycle.json', costs)

    def heartbeat():
        while not stopped.is_set():
            try:
                snapshot = remote_queue('pulse', job, known)
                known.update(snapshot.get('model_pids', []))
                state.update(snapshot, local_monotonic=time.monotonic())
                atomic_json(stage/'remote-state.json', snapshot)
            except Exception as exc:
                state['observation_error'] = type(exc).__name__
            stopped.wait(15)

    try:
        snapshot = remote_queue('enqueue', job)
        require(snapshot['owned'] and snapshot['monitor_live'], 'monitor or queue unavailable')
        state.update(snapshot, local_monotonic=time.monotonic())
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        with (stage/'tunnel.log').open('x') as log:
            tunnel = subprocess.Popen([*SSH[:-1], '-N', '-o', 'ExitOnForwardFailure=yes',
                '-L', f'127.0.0.1:{PORT}:127.0.0.1:{PORT}', '4090'],
                stdout=log, stderr=subprocess.STDOUT)
        loading = None
        while health is None:
            require(tunnel.poll() is None and state.get('owned') and state.get('monitor_live'),
                    'tunnel, monitor or queue ownership lost')
            require(time.monotonic()-state['local_monotonic'] < 90, 'remote observation stale')
            info = state.get('job') or {}
            require(info.get('status') != 'finished', 'GPU job finished before readiness')
            if info.get('status') == 'running':
                loading = loading or time.monotonic()
                require(time.monotonic()-loading <= plan['model_start_seconds'], 'model readiness timed out')
                try:
                    health = get_health()
                except OSError:
                    pass
            if health is None:
                time.sleep(2)
        validate_health(health, plan, arm)
        actual = remote_queue('status', job, known)
        validate_live_service(actual, health)
        known.update(actual['model_pids'])
        atomic_json(stage/'activation.json', actual)
        atomic_json(stage/'health.json', health)
        atomic_json(stage/'expected.json', expected_manifest(plan, tasks, health, arm, index))
        costs.update(phase='running', model_ready_epoch=time.time())
        atomic_json(stage/'lifecycle.json', costs)
        parent = identity(os.getpid())
        with (stage/'watchdog.log').open('x') as log:
            guard = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'scripts.run_qwen38_b3_dev',
                '--plan-sha256', plan_sha, '--watchdog', '--arm', arm, '--batch', str(index),
                '--parent-pid', str(parent['pid']), '--parent-ticks', str(parent['start_ticks'])],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                                  TMPDIR=plan['tmp_dir']),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            guard_identity = identity(guard.pid)
            atomic_json(stage/'watchdog-launch.json', {'identity': guard_identity, 'parent': parent,
                                                       'started_epoch': time.time()})
            deadline = time.monotonic()+plan['client_seconds']+plan['cleanup_seconds']
            while guard.poll() is None:
                require(time.monotonic() < deadline, 'watchdog did not finish in lifecycle budget')
                require(tunnel.poll() is None and state.get('owned') and state.get('monitor_live'),
                        'tunnel, monitor or ownership lost during rollout')
                require(time.monotonic()-state['local_monotonic'] < 90, 'remote observation stale')
                require((state.get('job') or {}).get('status') == 'running', 'GPU service exited during rollout')
                time.sleep(2)
        guard_result = read(stage/'watchdog.json')
        report = audit_terminated_batch(stage)
        require(guard.returncode == 0, 'client watchdog failed')
        require(guard_result['client_release']['verified_exited']
                and guard_result['container_release']['verified'], 'CPU resources not released')
        require(report is not None, 'client produced no run manifest')
        final_health = get_health()
        validate_health(final_health, plan, arm)
        require(final_health['backend'] == health['backend'], 'service changed during batch')
        atomic_json(stage/'final-health.json', final_health)
        costs['phase'] = 'completed' if report['valid_for_comparison'] else 'incomplete'
        require(guard_result['client_exit_code'] == 0 and report['valid_for_comparison'],
                'incomplete batch; preserve unknowns and stop without retry')
    except BaseException as exc:
        costs.update(phase='failed_or_interrupted', error_type=type(exc).__name__, error=str(exc)[:1000])
        raise
    finally:
        stopped.set()
        if thread is not None:
            thread.join(timeout=65)
        try:
            release = remote_queue('stop', job, known)
            deadline = time.monotonic()+120
            while release['owned'] and not release['released'] and time.monotonic() < deadline:
                time.sleep(5)
                release = remote_queue('status', job, known)
            atomic_json(stage/'release.json', release)
            require(release['owned'] and release['released'], 'GPU release unconfirmed')
            missing = collect_remote(job, known, stage)
            if health is not None:
                require(not missing, 'missing terminal remote evidence')
                costs['admission_and_GPU_cost'] = verify_admission(stage, job, known, health)
        finally:
            try:
                if guard is not None and guard.poll() is None:
                    require(live(guard_identity), 'watchdog PID identity changed')
                    guard.terminate()
                    guard.wait(timeout=plan['cleanup_seconds'])
                if guard is not None and guard.poll() is not None:
                    recover_client_after_guard_exit(plan, arm, index, stage, guard)
                if (stage/'watchdog.json').exists():
                    result = read(stage/'watchdog.json')
                    if report is None and result.get('client_release', {}).get('verified_exited'):
                        try:
                            report = audit_terminated_batch(stage)
                        except Exception as exc:
                            costs['audit_error'] = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
                    require(result.get('client_release', {}).get('verified_exited') is True
                            and result.get('container_release', {}).get('verified') is True,
                            'CPU client or container release not verified')
            finally:
                if tunnel is not None and tunnel.poll() is None:
                    tunnel.terminate()
                    tunnel.wait(timeout=10)
                costs['finished_epoch'] = time.time()
                atomic_json(stage/'lifecycle.json', costs)
    # Reached only after a complete diagnostic audit, final health, admission,
    # and independent CPU/GPU release checks all succeeded.
    atomic_json(stage/'accepted.json', {'accepted': True, 'checked_epoch': time.time(),
        'sha256': {name: digest(stage/name) for name in ('report.json', 'health.json',
            'final-health.json', 'activation.json', 'watchdog.json', 'release.json',
            'expected.json', 'lifecycle.json', 'remote-artifacts/manifest.json')}})
    return report


def verify_remote(plan_sha):
    return remote_call('''import json,sys,os,subprocess
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));os.chdir(root)
from scripts.run_qwen38_b3_dev import load_registration,make_job,runtime_state,PYTHON,SERVICE_SHA,MODEL_IDS
p=json.load(sys.stdin);plan=load_registration(root,p['sha'])
for item in plan['schedule']:
 job=make_job(item['batch_index'],item['arm'],'0'*32,p['sha'])
 for prefix,suffix in [('gpu-idle-job-','.json'),('gpu-idle-heartbeat-',''),
  ('gpu-idle-ticket-','.json'),('gpu-idle-ticket-','.consumed.json')]:
  assert not (root/'artifacts'/(prefix+job['id']+suffix)).exists()
state=runtime_state(job,root/'artifacts');assert state['monitor_live']
for arm in ['B0','B3']:
 r=subprocess.run([PYTHON,'-B','-m','scripts.serve_qwen38_b3','--check','--plan-sha256',SERVICE_SHA,
  '--arm',arm,'--model-id',MODEL_IDS[arm]],capture_output=True,text=True,check=True,timeout=20)
 assert json.loads(r.stdout)['preflight']=='passed'
print(json.dumps({'preflight':'passed','uses_gpu':False,'schedule_jobs':20,'monitor_live':True}))
''', {'sha': plan_sha})


def aggregate_study(plan, output):
    from scripts.report_b3_dev_pairs import analyze

    arms = {arm: {'protocol': plan['protocol'], 'outcomes': []} for arm in MODEL_IDS}
    batches, gpu_seconds, missing_cost = [], 0.0, []
    for item in plan['schedule']:
        stage = output/f"{item['arm'].lower()}-{item['batch_index']}"
        default = 'unknown' if stage.exists() else 'unattempted'
        outcomes = [{'task_id': task_id, 'status': default, 'success': None, 'generation_seed': 17,
                     'environment_seed': {'requested': 100, 'effective': None, 'verified': False}}
                    for task_id in item['task_ids']]
        entry = {**item, 'stage_created': stage.exists(), 'resources_released': False}
        if (stage/'report.json').exists():
            report = read(stage/'report.json')
            require(report['arm'] == item['arm'] and report['batch_index'] == item['batch_index']
                    and [r['task_id'] for r in report['normalized_outcomes']] == item['task_ids'],
                    'batch report identity changed')
            entry.update(costs=report['costs'], report_sha256=digest(stage/'report.json'))
        if (stage/'release.json').exists() and (stage/'watchdog.json').exists():
            gpu, cpu = read(stage/'release.json'), read(stage/'watchdog.json')
            entry['resources_released'] = (gpu.get('owned') is gpu.get('released') is True
                and cpu.get('client_release', {}).get('verified_exited') is True
                and cpu.get('container_release', {}).get('verified') is True)
        if (stage/'accepted.json').exists():
            accepted = read(stage/'accepted.json')
            names = {'report.json', 'health.json', 'final-health.json', 'activation.json',
                     'watchdog.json', 'release.json', 'expected.json', 'lifecycle.json',
                     'remote-artifacts/manifest.json'}
            require(accepted.get('accepted') is True and set(accepted['sha256']) == names
                    and all(digest(stage/name) == sha for name, sha in accepted['sha256'].items())
                    and entry['resources_released'], 'batch acceptance evidence changed')
            accepted_report = read(stage/'report.json')
            run_files = accepted_report['run_files_sha256']
            expected_files = {'manifest.json', 'summary.json', 'events.jsonl', 'attempts.jsonl',
                              'episodes.jsonl', 'steps.jsonl', 'responses.jsonl', 'feedback-events.jsonl'}
            require(set(run_files) == expected_files and all(
                not (stage/'run'/name).is_symlink() and digest(stage/'run'/name) == sha
                for name, sha in run_files.items()), 'accepted run evidence changed')
            outcomes = accepted_report['normalized_outcomes']
            entry['accepted_for_comparison'] = True
        else:
            entry['accepted_for_comparison'] = False
        cost = None
        if (stage/'lifecycle.json').exists():
            life = read(stage/'lifecycle.json')
            entry['lifecycle'] = life
            cost = life.get('admission_and_GPU_cost')
            if cost is not None:
                gpu_seconds += cost['job_seconds']
        if stage.exists() and cost is None:
            missing_cost.append({'arm': item['arm'], 'batch_index': item['batch_index']})
        arms[item['arm']]['outcomes'].extend(outcomes)
        batches.append(entry)
    analysis = analyze(plan['task_ids'], arms)
    result = {'analysis': analysis, 'normalized_arms': arms, 'batches': batches,
              'verified_GPU_job_seconds': gpu_seconds,
              'GPU_job_cost_complete': not missing_cost, 'GPU_job_cost_unknown_batches': missing_cost,
              'GPU_job_cost_is_lower_bound': bool(missing_cost),
              'all_resources_released': all(b['resources_released'] for b in batches if b['stage_created']),
              'created_batches': sum(b['stage_created'] for b in batches),
              'accepted_batches': sum(b['accepted_for_comparison'] for b in batches),
              'scope': 'Development comparison; only complete paired outcomes allow a primary estimate. '
                       'GPU wall and overlapping task/HTTP/token costs must not be added. '
                       'No efficacy or final-test claim follows from harness completion.'}
    atomic_json(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--arm', choices=['B0', 'B3'])
    parser.add_argument('--batch', type=int)
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--parent-ticks', type=int)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'controller/watchdog must remain CPU-only')
    os.chdir(ROOT)
    plan = load_registration(ROOT, args.plan_sha256)
    output = ROOT/'artifacts'/ID
    if args.watchdog:
        require(args.arm in MODEL_IDS and type(args.batch) is int and 0 <= args.batch < 10,
                'invalid watchdog slice')
        watchdog(plan, args.arm, args.batch, output/f'{args.arm.lower()}-{args.batch}',
                 {'pid': args.parent_pid, 'start_ticks': args.parent_ticks})
        return
    with (ROOT/'artifacts/foundation-pilot-001.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not output.exists(), 'study already has an attempt; no restart')
        require_space(ROOT/'artifacts')
        tasks, metadata = load_prepared_tasks(ROOT/TASKS, expected_sha256=plan['tasks_sha256'],
            data_dir=plan['data_dir'], deps_dir=plan['deps_dir'], image=plan['protocol']['environment_image_id'])
        require([t.task_id for t in tasks] == plan['task_ids'], 'metadata coverage changed')
        for name, sha in plan['native_runtime_sha256'].items():
            require(digest(name) == sha, 'native AppWorld runtime or split changed')
        image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
            plan['protocol']['environment_image_id']], text=True, timeout=15).strip()
        require(image == plan['protocol']['environment_image_id'], 'CPU image changed')
        preflight = verify_remote(args.plan_sha256)
        if args.check:
            print(json.dumps(preflight))
            return
        output.mkdir()
        state = {'phase': 'running', 'identity': identity(os.getpid()), 'plan_sha256': args.plan_sha256,
                 'started_epoch': time.time(), 'completed_jobs': [], 'schedule': plan['schedule'],
                 'automatic_retry': False, 'uses_GPU': False}
        atomic_json(output/'status.json', state)
        try:
            for item in plan['schedule']:
                state.update(current=item, checked_epoch=time.time())
                atomic_json(output/'status.json', state)
                run_batch(plan, metadata['tasks'], args.plan_sha256, item['arm'], item['batch_index'], output)
                state['completed_jobs'].append(item)
            state['phase'] = 'complete'
        except BaseException as exc:
            state.update(phase='failed_or_interrupted', error_type=type(exc).__name__, error=str(exc)[:1000])
            raise
        finally:
            state['finished_epoch'] = time.time()
            atomic_json(output/'status.json', state)
            try:
                aggregate_study(plan, output)
            except Exception as exc:
                state['aggregation_error'] = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
                atomic_json(output/'status.json', state)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
