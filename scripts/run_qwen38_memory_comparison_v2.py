"""Registered fresh three-arm utility study, serialized on one idle-qualified GPU."""
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
from scripts.run_memory_branch import ARMS, initial_actor_messages, protocol
from scripts.run_memory_comparison import audit_batch, branch_schedule, run_comparison
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
from scripts.run_qwen38_recovery import collect_remote_evidence
from scripts.run_qwen38_recovery_lessons import load_registration as load_writer_registration
from scripts.run_timed_tasks import RolloutDeadline, wait_for_idle
from wll.appworld import AppWorldEnvironment
from wll.experience import Experience
from wll.model import LocalModel
from wll.protocol import Task, fingerprint
from wll.storage import atomic_json, require_space

ID = 'qwen38-recovery-memory-comparison-002'
PLAN = 'artifacts/qwen38-recovery-memory-comparison-plan-002.json'
DESIGN = 'artifacts/recovery-memory-comparison-design-002.json'
CONTEXT = 'artifacts/recovery-memory-comparison-context-preflight-001.json'
PREPARATION = 'artifacts/recovery-memory-comparison-preparation-001.json'
PARTITION = 'artifacts/partitions-v1/manifest.json'
WRITER_SHA = 'eddc6c1a8d7be567ee737723010b3e1e78ca7c55b0fa32f65783fcd9f903e6aa'
PORT = 18979
NEW_SOURCES = ['scripts/run_memory_branch.py', 'scripts/report_memory_branch.py',
    'scripts/run_memory_comparison.py', 'scripts/report_memory_comparison.py',
    'scripts/run_qwen38_memory_comparison_v2.py', 'scripts/audit_memory_comparison_v2.py']


def validate_study(root, plan):
    root = Path(root)
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['design'] == DESIGN
        and plan['design_sha256'] == digest(root / DESIGN), 'registered unchanged three-arm design required')
    design = read(root / DESIGN)
    require(all(plan.get(k) == v for k, v in design.items() if k not in {'state', 'missing_before_registration'}),
        'three-arm design settings differ')
    require(plan['writer_plan_sha256'] == WRITER_SHA, 'writer registration differs')
    writer, parent, _ = load_writer_registration(root, root / plan['writer_plan'], WRITER_SHA)
    for key in ('expected_backend_static', 'gpu_policy', 'gpu_policy_sha256', 'wll_code_fingerprint'):
        require(plan[key] == writer[key], 'writer runtime differs: ' + key)
    for key in ('data_dir', 'deps_dir', 'image', 'environment_id', 'tmp_dir'):
        require(plan[key] == parent[key], 'environment differs: ' + key)
    require(plan['arms'] == ARMS and plan['port'] == PORT and plan['model_start_seconds'] == 600
        and plan['server_idle_seconds'] == 180 and plan['client_seconds_per_batch'] == 6000
        and plan['gpu_seconds_per_batch'] == 6900
        and plan['monitor_max_runtime_seconds'] == 7200, 'three-arm lifecycle differs')
    require([len(b) for b in plan['batches']] == [2, 2, 2, 2, 1]
        and 6 * (plan['rollout_seconds'] + 120 + plan['server_idle_seconds'] + 60) <= plan['client_seconds_per_batch']
        and plan['client_seconds_per_batch'] + plan['model_start_seconds'] + 90 <= plan['gpu_seconds_per_batch']
        <= plan['monitor_max_runtime_seconds'], 'batch cannot fit existing monitor cap')
    require(plan['preparation'] == PREPARATION and plan['preparation_sha256'] == digest(root / PREPARATION)
        and plan['CPU_context_preflight'] == CONTEXT and plan['CPU_context_preflight_sha256'] == digest(root / CONTEXT)
        and digest(root / plan['inputs']) == plan['inputs_sha256'], 'preparation evidence changed')
    prep, inputs, context = read(root / PREPARATION), read(root / plan['inputs']), read(root / CONTEXT)
    require(prep['phase'] == 'complete' and prep['uses_gpu'] is False and prep['LLM_calls'] == 0
        and prep['scheduled_branches'] == 27 and prep['CPU_descriptor_container_absent'], 'preparation incomplete')
    complete = read(root / inputs['writer_complete'])
    require(digest(root / inputs['writer_complete']) == inputs['writer_complete_sha256']
        and complete['phase'] == 'complete' and complete['result']['unvalidated_candidates'] == 2
        and complete['accepted_experiences'] == 0 and complete['native_controller_and_forwarder_released'],
        'audited writer completion required')
    for mapping in (prep['source_sha256'], prep['input_sha256'], complete['input_sha256'], inputs['source_sha256']):
        for name, sha in mapping.items():
            require(digest(root / name) == sha, 'bound preparation or writer evidence differs: ' + name)
    require(inputs['state'] == 'prepared_not_registered' and inputs['writer_plan_sha256'] == WRITER_SHA
        and inputs['partition_sha256'] == digest(root / PARTITION)
        and context['phase'] == 'complete' and context['uses_gpu'] is context['model_loaded'] is False
        and context['LLM_calls'] == 0 and context['manifest_sha256'] == plan['inputs_sha256'], 'input provenance differs')
    tasks = [Task(**json.loads(line)) for line in (root / inputs['tasks']).read_text().splitlines()]
    require(digest(root / inputs['tasks']) == inputs['tasks_sha256']
        and [t.task_id for t in tasks] == plan['task_ids'] == inputs['task_ids']
        and len(tasks) == 9 and [t.task_id for t in tasks] == sum(plan['batches'], []), 'task schedule differs')
    utility = set(read(root / PARTITION)['task_ids']['utility'])
    require(all(t.task_id in utility and t.split == 'train' for t in tasks), 'non-utility task')
    review = read(root / inputs['content_review'])
    require(digest(root / inputs['content_review']) == inputs['content_review_sha256'], 'content review changed')
    candidates = {}
    for arm, value in inputs['candidates'].items():
        candidate = Experience(**value['candidate'])
        record = next(r for r in review['conditions'] if r['condition'] == arm.removesuffix('_memory'))
        require(read(root / record['candidate_file'])['experience'] == value['candidate']
            and digest(root / record['candidate_file']) == record['candidate_file_sha256']
            and record['raw_candidate_modified'] is record['accepted_experience'] is record['eligible_for_training'] is False,
            'candidate changed or promoted')
        require((root / value['rendered_file']).read_text() == candidate.render()
            and digest(root / value['rendered_file']) == value['rendered_sha256'] == record['rendered_sha256']
            and record['failure_mode_in_actor_prompt'] is False
            and all(t.scenario_id not in candidate.source_scenarios for t in tasks), 'actor projection or origin differs')
        candidates[arm] = candidate
    require(set(candidates) == set(ARMS[1:]) and next(r for r in review['conditions'] if r['condition'] == 'proposal_hint')
        ['full_output_writer_instruction_conformance'] is False, 'two unchanged candidates and recorded violation required')
    schedule, by_id = branch_schedule(tasks, 0), {t.task_id: t for t in tasks}
    require(len(inputs['ordered_requests']) == len(context['inputs']) == 27, 'prompt set incomplete')
    for (task_id, arm), row, ctx in zip(schedule, inputs['ordered_requests'], context['inputs'], strict=True):
        messages = initial_actor_messages(by_id[task_id], candidates.get(arm), plan['prompt_version'])
        require(row['task_id'] == ctx['task_id'] == task_id and row['arm'] == ctx['arm'] == arm
            and read(root / row['messages']) == messages and fingerprint(messages) == row['message_fingerprint']
            and digest(root / row['messages']) == row['messages_sha256'] == ctx['messages_sha256']
            and ctx['max_new_tokens'] == plan['max_tokens'] and ctx['prompt_tokens'] > 0
            and ctx['total_max_tokens'] == ctx['prompt_tokens'] + plan['max_tokens']
            and ctx['total_max_tokens'] <= plan['max_context'], 'actor prompt or token budget differs')
    return inputs, tasks, candidates


def load_registration(root, path, sha):
    root = Path(root)
    plan = verify_frozen_plan(root, path, sha)
    inputs, tasks, candidates = validate_study(root, plan)
    writer = read(root / plan['writer_plan'])
    require(set(writer['source_sha256']) | set(NEW_SOURCES) <= plan['source_sha256'].keys()
        and all(plan['source_sha256'].get(n) == s for n, s in writer['source_sha256'].items()), 'source bindings incomplete')
    required = {DESIGN, CONTEXT, PREPARATION, plan['inputs'], plan['writer_plan'],
        'artifacts/recovery-memory-comparison-cpu-verification-002.json'}
    require(required <= plan['input_sha256'].keys(), 'input bindings incomplete')
    for mapping in (writer['input_sha256'], read(root / PREPARATION)['input_sha256'],
                    read(root / inputs['writer_complete'])['input_sha256']):
        require(all(plan['input_sha256'].get(n) == s for n, s in mapping.items()), 'transitive bindings differ')
    return plan, inputs, tasks, candidates


def task_offset(plan, index):
    require(type(index) is int and 0 <= index < len(plan['batches']), 'invalid batch index')
    return sum(len(batch) for batch in plan['batches'][:index])


def expected_protocols(tasks, candidates, plan, health):
    model = SimpleNamespace(model_id=health['model'], max_tokens=plan['max_tokens'], temperature=plan['temperature'])
    expected = {t.task_id: {a: protocol(t, candidates.get(a), model, arm=a, max_steps=plan['max_steps'],
        seconds=plan['rollout_seconds'], seed=plan['seed'], environment_id=plan['environment_id'],
        expected_backend=health['backend'], prompt_version=plan['prompt_version'], backend_idle_check_enabled=True)
        for a in ARMS} for t in tasks}
    require(all(v['script_sha256'] == plan['source_sha256']['scripts/run_memory_branch.py']
        and v['wll_code_fingerprint'] == plan['wll_code_fingerprint'] for arms in expected.values() for v in arms.values()),
        'branch implementation differs')
    return expected


def worker(root, plan, inputs, tasks, candidates, index, stage):
    health = read(stage / 'health.json')
    validate_health(health, plan, index)
    subset = [t for t in tasks if t.task_id in plan['batches'][index]]
    expected = expected_protocols(subset, candidates, plan, health)
    require(read(stage / 'expected.json') == expected, 'worker protocol differs')
    endpoint = f"http://127.0.0.1:{plan['port']}/v1"
    model = LocalModel(endpoint, health['model'], max_tokens=plan['max_tokens'], temperature=plan['temperature'])
    resources = []
    with (stage / 'feedback-events.jsonl').open('x') as feedback:
        def factory(task, arm):
            raw = AppWorldEnvironment(task.task_id, plan['data_dir'], plan['max_steps'], plan['image'], plan['deps_dir'])
            try:
                identifier = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}',
                    raw.worker.name], text=True, timeout=15).strip()
                require(identifier and identifier not in {r['id'] for r in resources}, 'branches must start fresh')
                resources.append({'arm': arm, 'task_id': task.task_id, 'id': identifier, 'name': raw.worker.name})
                atomic_json(stage / 'containers.json', resources)
                return observation_world(raw, read(Path(root) / ENV_PLAN), task.task_id, feedback)
            except BaseException:
                raw.close()
                raise

        try:
            result = run_comparison(subset, candidates, factory, model, output=stage / 'run', task_offset=task_offset(plan, index),
                max_steps=plan['max_steps'], seconds=plan['rollout_seconds'], seed=plan['seed'],
                environment_id=plan['environment_id'], expected_backend=health['backend'], prompt_version=plan['prompt_version'],
                before_branch=lambda task, arm: wait_for_idle(endpoint, health['model'], health['backend'],
                    seconds=plan['server_idle_seconds']))
        finally:
            if (stage / 'run/manifest.json').exists():
                atomic_json(stage / 'report.json', audit_batch(stage / 'run', subset, candidates, expected, task_offset=task_offset(plan, index)))
    return 0 if result['valid_for_aggregation'] else 1


def make_job(index, owner, plan_sha):
    require(type(index) is int and index in range(5) and re.fullmatch('[a-f0-9]{32}', owner)
        and re.fullmatch('[a-f0-9]{64}', plan_sha), 'invalid batch, owner or plan hash')
    job_id = f'{ID}-b{index}'
    return {'id': job_id, 'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 6900,
        'argv': [PYTHON, '-B', '-m', 'scripts.serve_qwen38', '--plan-sha256', SERVICE_SHA,
                 '--model-id', job_id, '--port', str(PORT)]}


def verify_remote(plan_sha, index):
    return remote_python('''import json,sys,subprocess,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));p=json.load(sys.stdin)
from scripts.run_qwen38_memory_comparison_v2 import load_registration,PLAN,PYTHON,ID,SERVICE_SHA
plan,inputs,tasks,candidates=load_registration(root,root/PLAN,p['plan_sha'])
r=subprocess.run([PYTHON,'-B','-m','scripts.serve_qwen38','--check','--model-id',ID+'-b'+str(p['index']),'--plan-sha256',SERVICE_SHA],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),capture_output=True,text=True,timeout=15,check=True)
assert json.loads(r.stdout)['preflight']=='passed'
job_id=ID+'-b'+str(p['index'])
for name in [f'gpu-idle-job-{job_id}.json',f'gpu-idle-ticket-{job_id}.json',f'gpu-idle-ticket-{job_id}.consumed.json',f'gpu-idle-heartbeat-{job_id}']:
 assert not (root/'artifacts'/name).exists(),name
print(json.dumps({'preflight':'passed','uses_gpu':False,'sources':len(tasks),'source_bindings':len(plan['source_sha256'])}))
''', {'plan_sha': plan_sha, 'index': index})


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'invalid queue operation')
    require(job.get('id') in [f'{ID}-b{i}' for i in range(5)] and
        job == make_job(int(job['id'][-1]), job.get('owner', ''), job.get('plan_sha256', '')), 'queue command or identity changed')
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
                raise FileExistsError('memory batch already attempted or owned')
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
from scripts.run_qwen38_memory_comparison_v2 import queue_operation
d=json.load(sys.stdin)
print(json.dumps(queue_operation(d['operation'],d['job'],known_pids=d['known_pids'])))
''', {'operation': operation, 'job': job, 'known_pids': list(known_pids)})


def validate_health(health, plan, index):
    require(health.get('model') == f'{ID}-b{index}' and health.get('ready') is True
            and health.get('busy') is False and health.get('max_context') == 16384
            and health.get('max_output_tokens') == 1024, 'unexpected service health or context')
    backend = health['backend']
    require({k: v for k, v in backend.items() if k != 'gpu_policy'} == plan['expected_backend_static'],
            'inference backend differs')
    gpu, policy = backend['gpu_policy'], plan['gpu_policy']
    require(gpu.get('hostname') == policy['hostname'] and type(gpu.get('physical_gpu_index')) is int
            and str(gpu['physical_gpu_index']) in policy['allowed_gpus']
            and gpu.get('gpu_uuid') == policy['allowed_gpus'][str(gpu['physical_gpu_index'])]
            and type(gpu.get('visible_device_count')) is int and gpu['visible_device_count'] == 1 and gpu.get('policy_sha256') == plan['gpu_policy_sha256'],
            'unauthorized service GPU')


def run_batch(root, plan, inputs, tasks, candidates, index, plan_path, plan_sha):
    require(load_registration(root, plan_path, plan_sha) == (plan, inputs, tasks, candidates), 'in-memory configuration changed')
    remote_check = verify_remote(plan_sha, index)
    stage = root / 'artifacts' / ID / f'batch-{index}'
    stage.mkdir()
    atomic_json(stage / 'freeze-check.json', {'plan_sha256': plan_sha, 'checked_epoch': time.time(),
                                             'local_files_match': True, 'remote': remote_check})
    subset = [t for t in tasks if t.task_id in plan['batches'][index]]
    verify_cpu_environment(plan, [asdict(t) for t in subset], stage / 'cpu-environment-preflight.json')
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
        validate_health(health, plan, index)
        confirmed = remote_queue('status', job, known)
        validate_live_service(confirmed, health)
        known.update(confirmed['model_pids'])
        atomic_json(stage / 'activation.json', confirmed)
        atomic_json(stage / 'health.json', health)
        expected = expected_protocols(subset, candidates, plan, health)
        atomic_json(stage / 'expected.json', expected)
        with (stage / 'client.log').open('x') as log:
            client = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'scripts.run_qwen38_memory_comparison_v2',
                '--plan', str(plan_path), '--plan-sha256', plan_sha, '--worker', '--batch', str(index), '--stage', str(stage)],
                cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root), TMPDIR=plan['tmp_dir']),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = client.wait(timeout=plan['client_seconds_per_batch'])
        try:
            report = audit_batch(stage / 'run', subset, candidates, expected, task_offset=task_offset(plan, index))
            atomic_json(stage / 'report.json', report)
        except Exception as exc:
            atomic_json(stage / 'report-failure.json', {'client_exit_code': code,
                'error_type': type(exc).__name__, 'success_rate': None,
                'reason': 'Raw memory artifacts incomplete or inconsistent; no outcome imputed'})
            raise
        require(code == 0 and report['valid_for_aggregation'], 'memory batch incomplete; no automatic retry')
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
    parser.add_argument('--batch', type=int, choices=tuple(range(5)))
    parser.add_argument('--stage')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only controller and worker required')
    path = Path(args.plan).resolve(strict=True)
    plan, inputs, tasks, candidates = load_registration(ROOT, path, args.plan_sha256)
    if args.worker:
        require(args.stage is not None and args.batch is not None, 'worker stage and batch required')
        raise SystemExit(worker(ROOT, plan, inputs, tasks, candidates, args.batch, Path(args.stage)))
    from scripts.audit_memory_comparison_v2 import audit_stage, combine_study
    with (ROOT / 'artifacts/foundation-pilot-001.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = ROOT / 'artifacts' / ID
        require(not output.exists(), 'memory study has an attempt; no automatic restart')
        require_space(ROOT / 'artifacts')
        actual_image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
            plan['image']], text=True, timeout=15).strip()
        require(actual_image == plan['image'], 'Docker image differs')
        for index in range(5):
            verify_remote(args.plan_sha256, index)
        if args.check:
            print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'batches': 5, 'branches': 27}))
            return
        output.mkdir()
        state = {'phase': 'running', 'plan_sha256': args.plan_sha256, 'pid': os.getpid(),
            'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
            'started_epoch': time.time(), 'completed_batches': 0, 'current_batch': 0}
        atomic_json(output / 'status.json', state)
        try:
            for index in range(5):
                state['current_batch'] = index
                atomic_json(output / 'status.json', state)
                result, release = run_batch(ROOT, plan, inputs, tasks, candidates, index, path, args.plan_sha256)
                require(result['valid_for_aggregation'] and release['released'], 'complete released batch required')
                stage = output / f'batch-{index}'
                ledger = audit_stage(ROOT, plan, inputs, tasks, candidates, index, stage)
                atomic_json(stage / 'generation-audit.json', ledger)
                state['completed_batches'] = index + 1
                atomic_json(output / 'status.json', state)
            report = combine_study(ROOT, plan, inputs, tasks, candidates, output)
            atomic_json(output / 'report.json', report)
            state.update(phase='complete', result=report)
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
