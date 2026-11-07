"""Continue only original remaining9 tasks after an audited pre-actor interruption."""
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

import scripts.run_qwen38_distill_baseline_002 as original
from scripts.audit_qwen38_generation import parse_events
from scripts.report_timed_tasks import audit_run
from scripts.run_foundation_pilot import (
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
from scripts.run_qwen38_capability import (
    collect_remote_evidence as parent_collect_evidence,
)
from scripts.run_qwen38_capability import (
    expected_protocol,
    runtime_state,
    validate_live_service,
)
from scripts.run_timed_tasks import RolloutDeadline, run_tasks, wait_for_idle
from scripts.serve_qwen38_idle180 import validate_changes as validate_service_changes
from wll.appworld import AppWorldEnvironment, load_tasks
from wll.model import LocalModel
from wll.prompts import initial_messages
from wll.protocol import Task
from wll.storage import atomic_json, require_space

ROOT = Path(__file__).resolve().parent.parent
REMOTE = Path('/home/hts/workflow-learning-lab')
PYTHON = '/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python'
ID = 'qwen38-distill-baseline-003'
MODULE = 'scripts.serve_qwen38_idle180'
PORT = 18979
PLAN = 'artifacts/qwen38-distill-baseline-plan-003.json'
DESIGN = 'artifacts/qwen38-distill-baseline-design-003.json'
ENV_PLAN = 'artifacts/qwen35-9b-observation-plan-001.json'
SERVICE_PLAN = 'artifacts/qwen38-service-idle180-plan-001.json'
SERVICE_SHA = '040a6a4476d1b4c6b4ef1d78d2e76fe1e600783b47c121d7ef7cf7a1ab668889'
SERVICE_AUDIT = 'artifacts/qwen38-service-integration-001.json'
SERVICE_RESULT = 'artifacts/qwen38-27b-service-smoke-001.json'
TASKS = 'artifacts/distill-baseline-inputs-001/tasks.jsonl'
PARTITION = 'artifacts/partitions-v1/manifest.json'
ORIGINAL_PLAN = 'artifacts/qwen38-distill-baseline-plan-002.json'
ORIGINAL_SHA = '7d9e697adf0ea1ac298e4c7f3b478f68c87680cdaac6af07a8c9d7f71cb71cbf'
INCIDENT = 'artifacts/distill-baseline-interruption-002/audit.json'
INCIDENT_SHA = '706c0b1e20d9cf4fb18387d23ae826b083b2bc215fba94547fbccc477232683c'
CPU_PROOF = 'artifacts/distill-resume-003-CPU-verification-001.json'


def validate_resume_evidence(evidence, ids):
    require(evidence['status'] == 'audited_interruption_before_actor'
        and evidence['completed_batches'] == 3 and evidence['completed_tasks'] == 18
        and evidence['interrupted_batch'] == 3
        and evidence['interrupted_server_generation_events'] == evidence['interrupted_task_outcomes'] == 0
        and evidence['native_controller_absent'] is True and evidence['native_forwarder_port_free'] is True
        and evidence['no_task_outcome_imputed'] is True
        and evidence['remote_release']['released'] is True and not evidence['remote_release']['model_pids']
        and evidence['remote_release']['job']['status'] == 'finished'
        and len(ids) == 27 and evidence['remaining_original_task_ids'] == ids[18:],
        'only original remaining9 after verified pre-actor interruption may resume')


def validate_resume(root, plan, tasks):
    parent, parent_tasks = original.load_registration(root, root / ORIGINAL_PLAN, ORIGINAL_SHA)
    require(tasks == parent_tasks and plan['resume_parent_plan'] == ORIGINAL_PLAN
        and plan['resume_parent_plan_sha256'] == ORIGINAL_SHA
        and plan['retained_batch_indices'] == [0, 1, 2] and plan['remaining_batch_indices'] == [3, 4]
        and plan['interruption_evidence'] == INCIDENT and plan['interruption_sha256'] == INCIDENT_SHA
        and digest(root / INCIDENT) == INCIDENT_SHA, 'different retained study or resume schedule')
    historical_design = read(root / original.DESIGN)
    for key in historical_design.keys() - {'id', 'state', 'created_epoch', 'operational_repair'}:
        require(plan[key] == parent[key], 'scientific collection configuration changed: ' + key)
    evidence = read(root / INCIDENT)
    validate_resume_evidence(evidence, plan['task_ids'])
    require(evidence['parent_plan_sha256'] == ORIGINAL_SHA, 'interrupted parent differs')
    for name, sha in {**evidence['input_sha256'], **evidence['remote_artifact_sha256']}.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
            and (root / name).resolve().is_relative_to(root.resolve()) and digest(root / name) == sha,
            'retained evidence changed: ' + name)
    previous = root / 'artifacts' / original.ID / 'batch-3'
    for name in ['containers.jsonl', 'feedback-events.jsonl', 'run/attempts.jsonl',
                 'run/episodes.jsonl', 'run/responses.jsonl', 'run/steps.jsonl']:
        require((previous / name).read_bytes() == b'', 'interrupted batch contains actor activity')
    log = root / Path(INCIDENT).parent / 'remote-artifacts' / f'gpu-idle-job-{original.ID}-b3.log'
    require(not parse_events(log.read_text()), 'interrupted server generated; no silent retry')
    require(plan['prior_infrastructure_cost_seconds'] == parent['prior_infrastructure_cost_seconds']
        + evidence['interrupted_GPU_job_seconds'], 'infrastructure launch costs omitted or duplicated')
    retained_reports(root, plan)


def retained_reports(root, plan):
    parent = read(root / ORIGINAL_PLAN)
    tasks = [json.loads(line) for line in (root / TASKS).read_text().splitlines()]
    reports, releases = [], []
    for index in (0, 1, 2):
        stage = root / 'artifacts' / original.ID / f'batch-{index}'
        health = read(stage / 'health.json')
        original.validate_health(health, parent, index)
        expected = original.expected_protocol(parent, original.batch_rows(tasks, index), health)
        require(expected == read(stage / 'expected.json'), 'retained protocol differs')
        report = audit_run(stage / 'run', expected)
        release = read(stage / 'release.json')
        require(report == read(stage / 'report.json') and report['valid_for_comparison']
            and report['completed'] == 6 and report['unknown_outcomes'] == 0
            and release['released'] is True, 'retained batch incomplete')
        reports.append(report)
        releases.append(release)
    return reports, releases


def validate_study(root, plan):
    root = Path(root)
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['design'] == DESIGN
        and plan['design_sha256'] == digest(root / DESIGN), 'registered unchanged distill design required')
    design = read(root / DESIGN)
    require(all(plan.get(k) == v for k, v in design.items() if k != 'state'), 'distill configuration changed')
    require(plan['collection_kind'] == 'B3_no_memory_first_attempt'
        and plan['automatic_retry'] is plan['automatic_recovery'] is plan['automatic_training'] is False,
        'first-attempt collection scope differs')
    require(plan['service_plan'] == SERVICE_PLAN and digest(root / SERVICE_PLAN) == SERVICE_SHA
        and plan['service_plan_sha256'] == SERVICE_SHA, 'unregistered inference service')
    service = read(root / SERVICE_PLAN)
    validate_service_changes(root, service)
    historical = read(root / SERVICE_RESULT)['backend']
    expected = {k: v for k, v in historical.items() if k != 'gpu_policy'}
    expected.update(server_sha256=digest(root / 'scripts/serve_qwen38_idle180.py'),
        service_plan_sha256=SERVICE_SHA, source_sha256=service['source_sha256'])
    require(plan['expected_backend_static'] == expected, 'inference recipe changed')
    audit, result = read(root / SERVICE_AUDIT), read(root / SERVICE_RESULT)
    require(audit['completion_gate'] and audit['model_group_and_controller_released']
        and audit['result_sha256'] == digest(root / SERVICE_RESULT)
        and result['phase'] == 'complete' and len(result['cases']) == 3, 'prior service evidence incomplete')
    env = read(root / ENV_PLAN)
    require(environment_id(env) == plan['environment_id'] and env['image'] == plan['image']
        and env['observation_feedback'] == plan['observation_feedback'], 'environment recipe differs')
    require(plan['gpu_policy'] == read(root / 'config/gpu-policy.json')
        and plan['gpu_policy_sha256'] == digest(root / 'config/gpu-policy.json')
        and plan['gpu_policy']['idle_seconds'] == 180, 'GPU policy changed')
    require(plan['client_seconds'] == 6000 and plan['gpu_seconds'] == 6900
        and plan['model_start_seconds'] == 600 and plan['server_idle_seconds'] == 180
        and plan['monitor_max_runtime_seconds'] == 7200
        and 6*(plan['rollout_seconds'] + 120 + plan['server_idle_seconds'] + 60) <= plan['client_seconds']
        and plan['client_seconds'] + plan['model_start_seconds'] + 90 <= plan['gpu_seconds'],
        'batch exceeds lifecycle budget')
    require(digest(root / PARTITION) == plan['partition_sha256']
        == '3732b02da71c4031ba3a831c727d1956bb43872552841db275a86900e0dcf4a6'
        and plan['inputs_sha256'] == digest(root / plan['inputs'])
        == 'fb7851b812bc5ad15f3a528b96b5eaf5ea127658bb2b90d07f22e3d66a3d8c91'
        and plan['prompts_sha256'] == digest(root / plan['prompts'])
        == 'ea432e2cb199657dbf87fa333855f465470c8c1a39fb1cae40f64b4a913e755c',
        'original partition or CPU preparation changed')
    tasks = [json.loads(line) for line in (root / TASKS).read_text().splitlines()]
    inputs, partition = read(root / plan['inputs']), read(root / PARTITION)
    ids = [t['task_id'] for t in tasks]
    require(ids == plan['task_ids'] == inputs['task_ids'] == partition['task_ids']['distill']
        and len(ids) == len(set(ids)) == 27
        and digest(root / TASKS) == plan['tasks_sha256'] == inputs['tasks_sha256']
        and plan['batches'] == [ids[i:i+6] for i in range(0, 27, 6)], 'original distill schedule differs')
    scenarios = {t.rsplit('_', 1)[0] for t in ids}
    other = {t.rsplit('_', 1)[0] for role in ['origin', 'utility'] for t in partition['task_ids'][role]}
    require(len(scenarios) == 9 and scenarios.isdisjoint(other), 'scenario partition overlap')
    prompts = read(root / plan['prompts'])
    require(len(prompts['rows']) == 27, 'initial prompt coverage incomplete')
    for task, row in zip(tasks, prompts['rows'], strict=True):
        require(task['task_id'] == row['task_id'] and task['split'] == 'train'
            and initial_messages(Task(**task), plan['prompt_version']) == read(root / row['messages'])
            and digest(root / row['messages']) == row['messages_sha256']
            and row['fits'] and row['max_new_tokens_for_check'] == plan['max_tokens']
            and row['max_context_for_check'] == plan['max_context']
            and 0 < row['input_tokens'] <= plan['max_context'] - plan['max_tokens'],
            'initial prompt or budget differs')
    validate_resume(root, plan, tasks)
    return tasks


def load_registration(root, path, sha):
    root = Path(root)
    plan = verify_frozen_plan(root, path, sha)
    required = {'scripts/run_qwen38_distill_baseline_003.py', 'scripts/run_qwen38_capability.py',
                'scripts/serve_qwen38_idle180.py', 'scripts/run_foundation_pilot.py',
                'scripts/run_timed_tasks.py', 'scripts/report_timed_tasks.py',
                'scripts/documentation_feedback.py', 'scripts/probe_observation_visibility.py',
                'scripts/run_queued_decoder_completion.py', 'config/gpu-policy.json'}
    required.update({'scripts/run_qwen38_distill_baseline_002.py', 'scripts/audit_qwen38_generation.py'})
    required.update(str(p.relative_to(root)) for p in (root / 'wll').glob('*.py'))
    require(required <= plan['source_sha256'].keys(), 'incomplete source bindings')
    service = read(root / SERVICE_PLAN)
    require(all(plan['source_sha256'].get(k) == v for k, v in service['source_sha256'].items()),
            'service sources not frozen')
    require({ORIGINAL_PLAN, INCIDENT, CPU_PROOF, DESIGN, ENV_PLAN, SERVICE_PLAN, SERVICE_AUDIT, SERVICE_RESULT, TASKS, PARTITION,
        plan['inputs'], plan['prompts'], 'artifacts/distill-baseline-controller-CPU-verification-002.json'}
            <= plan['input_sha256'].keys(), 'incomplete input bindings')
    return plan, validate_study(root, plan)


def make_job(index, owner, plan_sha):
    require(type(index) is int and index in (3, 4) and re.fullmatch('[a-f0-9]{32}', owner)
            and re.fullmatch('[a-f0-9]{64}', plan_sha), 'invalid batch, owner or plan hash')
    job_id = f'{ID}-b{index}'
    return {'id': job_id, 'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 6900,
            'argv': [PYTHON, '-B', '-m', MODULE, '--plan-sha256', SERVICE_SHA,
                     '--model-id', job_id, '--port', str(PORT)]}



def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'invalid queue operation')
    match = re.fullmatch(re.escape(ID) + r'-b([34])', job['id'])
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
from scripts.run_qwen38_distill_baseline_003 import queue_operation
d=json.load(sys.stdin)
print(json.dumps(queue_operation(d['operation'],d['job'],known_pids=d['known_pids'])))
''', {'operation': operation, 'job': job, 'known_pids': list(known_pids)})


def validate_health(health, plan, index):
    require(type(index) is int and index in (3, 4) and health.get('model') == f'{ID}-b{index}' and health.get('ready') is True
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



def require_monitor_ready(snapshot):
    watch = snapshot.get('watch') or {}
    require(snapshot.get('monitor_live') is True and not snapshot.get('model_processes')
        and watch.get('required_idle_seconds') == 180
        and watch.get('phase') in {'monitoring', 'monitoring_job_finished'},
        'current180s monitor must be confirmed live before enqueue')


def verify_remote(plan, first_batch):
    return remote_python('''import json,sys,subprocess,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));p=json.load(sys.stdin)
from scripts.run_qwen38_distill_baseline_003 import load_registration,PLAN,PYTHON,ID,SERVICE_SHA,require_monitor_ready,runtime_state,make_job
plan,tasks=load_registration(root,root/PLAN,p['plan_sha'])
require_monitor_ready(runtime_state(make_job(p['first_batch'],'0'*32,p['plan_sha']),root/'artifacts'))
result=subprocess.run([PYTHON,'-B','-m','scripts.serve_qwen38_idle180','--check','--model-id',ID+'-b3',
 '--plan-sha256',SERVICE_SHA],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),
 capture_output=True,text=True,timeout=15,check=True)
assert json.loads(result.stdout)['preflight']=='passed'
for i in range(p['first_batch'],5):
 for name in [f'gpu-idle-job-{ID}-b{i}.json',f'gpu-idle-ticket-{ID}-b{i}.json',f'gpu-idle-ticket-{ID}-b{i}.consumed.json',f'gpu-idle-heartbeat-{ID}-b{i}']:
  assert not (root/'artifacts'/name).exists(),name
print(json.dumps({'preflight':'passed','uses_gpu':False,'tasks':len(tasks),'source_bindings':len(plan['source_sha256'])}))
''', {'plan_sha': digest(ROOT / PLAN), 'first_batch': first_batch})


def worker(root, plan, index, stage):
    require(type(index) is int and index in (3, 4), 'only original remaining batches may execute')
    health = read(stage / 'health.json')
    validate_health(health, plan, index)
    ids = plan['batches'][index]
    tasks = load_tasks(plan['data_dir'], 'train', image=plan['image'], deps_dir=plan['deps_dir'], include_task_ids=ids)
    by_id = {t.task_id: t for t in tasks}
    tasks = [by_id[t] for t in ids]
    frozen = [json.loads(line) for line in (root / TASKS).read_text().splitlines()]
    require([asdict(t) for t in tasks] == batch_rows(frozen, index), 'live task descriptions changed')
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
    parent_collect_evidence(job, known, stage)
    if (stage / 'report.json').exists() and read(stage / 'report.json')['valid_for_comparison']:
        admission = read(stage / 'remote-artifacts' / f'gpu9-admission-{next(iter(known))}.json')
        require(admission['mode'] == 'continuous_idle_only' and admission['required_idle_seconds'] == 180
            and admission['allocator_limit_mib'] == 20480, 'different admission or allocator cap')


def batch_rows(tasks, index):
    require(type(index) is int and 0 <= index < 5 and len(tasks) == 27, 'original27-task schedule required')
    return tasks[index*6:index*6+6]


def combine_batches(reports, releases):
    require(len(reports) == len(releases) == 5, 'five complete batches required')
    for report, release, size in zip(reports, releases, [6, 6, 6, 6, 3], strict=True):
        require(report['audit'] == 'passed' and report['scheduled'] == report['completed'] == size
            and report['valid_for_comparison'] is True and report['unknown_outcomes'] == 0
            and type(report['successes']) is int and 0 <= report['successes'] <= size
            and release['released'] is True, 'incomplete collection or release')
    scenarios = [s for r in reports for s in r['successful_scenarios']]
    require(len(scenarios) == len(set(scenarios)), 'scenario repeated between fixed batches')
    successes = sum(r['successes'] for r in reports)
    return {'completed': 27, 'successes': successes, 'success_rate': successes/27,
        'scenario_count': 9, 'successful_scenarios': sorted(scenarios), 'unknown_outcomes': 0,
        'observed_step_cost': {k: sum(r['observed_step_cost'][k] for r in reports)
            for k in ['steps', 'prompt_tokens', 'completion_tokens']},
        'returned_response_cost': {k: sum(r['returned_response_cost'][k] for r in reports)
            for k in ['prompt_tokens', 'completion_tokens']},
        'generation_counters_from_summary': {k: sum(r['generation_counters_from_summary'][k] for r in reports)
            for k in ['generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage']},
        'automatic_training_started': False, 'accepted_experiences': 0,
        'scope': 'Originaltrain/distill first-attempt collection only; success-only SFT curation requires separateaudit. No heldoutscore or methodgain. Costviews overlap; absent response usage remains unknown.'}


def run_batch(plan, tasks, index, root, plan_path, *, plan_sha):
    verify_frozen_plan(root, plan_path, plan_sha, expected_plan=plan)
    remote_check = verify_remote(plan, index)
    stage = root / 'artifacts' / ID / f'batch-{index}'
    stage.mkdir()
    atomic_json(stage / 'freeze-check.json', {'plan_sha256': plan_sha, 'checked_epoch': time.time(),
                                             'local_files_match': True, 'remote': remote_check})
    verify_cpu_environment(plan, batch_rows(tasks, index), stage / 'cpu-environment-preflight.json')
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
        atomic_json(stage / 'expected.json', expected_protocol(plan, batch_rows(tasks, index), health))
        with (stage / 'client.log').open('x') as log:
            client = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.run_qwen38_distill_baseline_003',
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
    parser.add_argument('--batch', type=int, choices=(3, 4))
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
        verify_remote(plan, 3)
        if args.check:
            print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'batches': 2, 'tasks': 9, 'retained_tasks': 18, 'total_tasks': len(tasks)}))
            return
        output.mkdir()
        state = {'phase': 'running', 'plan_sha256': args.plan_sha256, 'pid': os.getpid(),
                 'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
                 'started_epoch': time.time(), 'completed_batches': 3, 'retained_batches': 3, 'new_completed_batches': 0}
        atomic_json(output / 'status.json', state)
        try:
            reports, releases = retained_reports(ROOT, plan)
            for index in (3, 4):
                state.update(current_batch=index, checked_epoch=time.time())
                atomic_json(output / 'status.json', state)
                report, release = run_batch(plan, tasks, index, ROOT, path, plan_sha=args.plan_sha256)
                reports.append(report)
                releases.append(release)
                state['completed_batches'] = len(reports)
                state['new_completed_batches'] = len(reports) - 3
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
