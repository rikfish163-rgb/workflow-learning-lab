"""Registered natural-failure recovery, with serial idle-qualified GPU batches."""
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

from scripts.prepare_recovery_inputs import prepare_inputs
from scripts.report_recovery_trial import audit_trial
from scripts.run_foundation_pilot import (
    MODEL_PATH,
    NF4_MEMORY_SHA,
    OBSERVATION_ID,
    PYTHON,
    REMOTE,
    ROOT,
    digest,
    environment_id,
    observation_world,
    process_matches,
    read,
    require,
    runtime_state,
    server_job,
    stop_client,
    verify_cpu_environment,
    verify_frozen_plan,
)
from scripts.run_queued_decoder_completion import SSH, remote_python
from scripts.run_recovery_trial import run_recovery_trial
from scripts.run_timed_tasks import RolloutDeadline, wait_for_idle
from wll.appworld import AppWorldEnvironment
from wll.cli import read_episodes
from wll.model import LocalModel
from wll.protocol import fingerprint
from wll.storage import atomic_json, require_space

ID = 'qwen35-9b-recovery-001'
PORT = 19019
SOURCE_PLAN = 'artifacts/qwen35-9b-observation-plan-001.json'
SOURCE_SHA = '1cddc7428726450573b228f1ca4c869d9a8ef42fdd3028a63881e0f62f22a8fa'
INPUTS = 'artifacts/observation-recovery-inputs-001.json'
INPUTS_SHA = '461c1aa7ddc87dc7ea1b875899812114fafa9aef769269fca937f4d4aa61e00e'
PLAN = 'artifacts/qwen35-9b-recovery-plan-001.json'


def make_job(index, owner, plan_sha):
    require(type(index) is int and index in (0, 1, 2), 'invalid recovery batch')
    require(re.fullmatch('[a-f0-9]{32}', owner) and re.fullmatch('[a-f0-9]{64}', plan_sha),
            'invalid owner or plan SHA256')
    job = server_job({'id': OBSERVATION_ID, 'model_path': MODEL_PATH, 'port': PORT,
                      'max_context': 32768, 'gpu_seconds': 5400}, index, owner, plan_sha)
    job['id'] = f'{ID}-b{index}'
    job['argv'][job['argv'].index('--model-id')+1] = job['id']
    return job


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'invalid queue operation')
    match = re.fullmatch(ID+r'-b([012])', job.get('id', ''))
    require(match is not None, 'unregistered recovery job identity')
    require(job == make_job(int(match[1]), job.get('owner', ''), job.get('plan_sha256', '')),
            'recovery queue command differs from fixed configuration')
    art = Path(root)/'artifacts'
    queue, heartbeat = art/'gpu-idle-queue.json', art/f"gpu-idle-heartbeat-{job['id']}"
    result_path = art/f"gpu-idle-job-{job['id']}.json"
    with (art/'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = read(queue) if queue.exists() else None
        if operation == 'enqueue':
            if result_path.exists() or heartbeat.exists():
                raise FileExistsError('recovery batch already attempted or has a heartbeat')
            if current:
                require(re.fullmatch('[a-z0-9_-]+', current['id']), 'invalid previous queue identity')
                old_result = art/f"gpu-idle-job-{current['id']}.json"
                require(old_result.exists() and read(old_result)['status'] == 'finished'
                        and not (art/f"gpu-idle-heartbeat-{current['id']}").exists(),
                        'cannot replace an active or unconfirmed queue')
                archive = art/f"gpu-idle-queue-archived-{current['id']}.json"
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
        if owned and operation == 'pulse':
            if not result_path.exists() or read(result_path)['status'] != 'finished':
                heartbeat.touch()
        if owned and operation == 'stop':
            heartbeat.unlink(missing_ok=True)
            if result_path.exists():
                info = read(result_path)
                require(info['queue_sha256'] == digest(queue), 'running job differs from owned queue')
                if process_matches(info, job['argv']):
                    os.killpg(info['process_group'], signal.SIGTERM)
    return {'owned': owned, **runtime_state(job, art, known_pids)}


def remote_queue(operation, job, known_pids=()):
    return remote_python('''import json,sys
sys.path.insert(0,'/home/hts/workflow-learning-lab')
from scripts.run_recovery_pilot import queue_operation
d=json.load(sys.stdin)
print(json.dumps(queue_operation(d['operation'],d['job'],known_pids=d['known_pids'])))
''', {'operation': operation, 'job': job, 'known_pids': list(known_pids)})


def validate_health(health, parent, index):
    require(health.get('model') == f'{ID}-b{index}' and health.get('ready') is True
            and health.get('busy') is False, 'unexpected recovery model health')
    backend = health['backend']
    require({k: v for k, v in backend.items() if k != 'gpu_policy'} ==
            parent['expected_backend_static'], 'recovery inference configuration changed')
    policy = backend['gpu_policy']
    allowed = parent['gpu_policy']['allowed_gpus']
    require(policy.get('hostname') == parent['gpu_policy']['hostname']
            and type(policy.get('physical_gpu_index')) is int
            and str(policy['physical_gpu_index']) in allowed
            and policy.get('gpu_uuid') == allowed[str(policy['physical_gpu_index'])]
            and policy.get('visible_device_count') == 1
            and policy.get('policy_sha256') == parent['gpu_policy_sha256'], 'unauthorized recovery GPU')


def load_registration(root, plan_path, plan_sha):
    plan = verify_frozen_plan(root, plan_path, plan_sha)
    require(plan['id'] == ID and plan['state'] == 'registered', 'registered recovery plan required')
    require(plan['source_plan'] == SOURCE_PLAN and plan['source_plan_sha256'] == SOURCE_SHA
            and plan['source_inputs'] == INPUTS and plan['source_inputs_sha256'] == INPUTS_SHA,
            'registered source study or prepared inputs differ')
    parent = verify_frozen_plan(root, root/SOURCE_PLAN, SOURCE_SHA)
    inputs = read(root/INPUTS)
    require(digest(root/INPUTS) == INPUTS_SHA and inputs == prepare_inputs(root, root/SOURCE_PLAN,
            SOURCE_SHA, root/'artifacts/observation-prefix-readiness-001'), 'recovery inputs changed')
    fixed = {'max_steps_including_replay': 40, 'recovery_seconds': 600, 'draft_seconds': 120,
        'recovery_seed': 1017, 'draft_seed': 2017, 'max_tokens': 1024, 'temperature': .7,
        'max_context': 32768, 'gpu_seconds_per_batch': 5400, 'client_seconds_per_batch': 4500,
        'model_start_seconds': 600, 'server_idle_seconds': 180,
        'max_recovery_attempts_per_source': 1, 'max_draft_attempts_per_success': 1, 'port': PORT}
    require(all(plan.get(k) == v for k, v in fixed.items()), 'recovery scientific settings changed')
    ids = [r['task_id'] for r in inputs['selected']]
    require(len(ids) == 8 and plan['task_ids'] == ids
            and plan['batches'] == [ids[i:i+3] for i in range(0, 8, 3)]
            and plan['environment_id'] == environment_id(parent)
            and plan['gpu_policy'] == parent['gpu_policy'], 'source selection or environment changed')
    required_sources = set(parent['source_sha256']) | {
        'scripts/run_recovery_pilot.py', 'scripts/report_recovery_trial.py',
        'scripts/run_recovery_trial.py', 'scripts/prepare_recovery_inputs.py',
        'scripts/audit_recovery_prefixes.py', 'scripts/report_observation_comparison.py'}
    require(required_sources <= plan['source_sha256'].keys(), 'recovery source registration incomplete')
    required_inputs = {SOURCE_PLAN, INPUTS, 'artifacts/qwen35-9b-recovery-design-001.json',
                       'artifacts/observation-complete-integration-001.json'} | set(inputs['input_sha256'])
    require(required_inputs <= plan['input_sha256'].keys(), 'recovery input registration incomplete')
    design = read(root/'artifacts/qwen35-9b-recovery-design-001.json')
    require(all(plan.get(k) == design[k] for k in design
                if k not in {'state', 'missing_before_registration'}), 'registered design changed')
    return plan, parent, inputs


def load_sources(root, entries):
    sources = []
    for record in entries:
        name = record['source_episodes']
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'non-project source')
        matches = [e for e in read_episodes(root/name) if e.task.task_id == record['task_id']]
        require(len(matches) == 1 and fingerprint(matches[0].to_dict()) == record['source_hash'],
                'source episode missing or changed')
        sources.append(matches[0])
    return sources


def expected_trial(source, record, plan, parent, health):
    return {'source_hash': record['source_hash'], 'prefix_proof': record['prefix_proof'],
        'task_id': source.task.task_id, 'model_id': health['model'],
        'max_steps': plan['max_steps_including_replay'], 'max_tokens': plan['max_tokens'],
        'temperature': plan['temperature'], 'recovery_seed': plan['recovery_seed'],
        'draft_seed': plan['draft_seed'], 'rollout_seconds': plan['recovery_seconds'],
        'draft_seconds': plan['draft_seconds'], 'prompt_version': parent['prompt_version'],
        'environment_id': plan['environment_id'], 'expected_backend': health['backend'],
        'source_backend': source.metadata['backend'],
        'script_sha256': plan['source_sha256']['scripts/run_recovery_trial.py'],
        'wll_code_fingerprint': plan['wll_code_fingerprint']}


def audit_batch(stage, sources, expected):
    trial_root = stage/'trials'
    present = {p.name for p in trial_root.iterdir() if p.is_dir()} if trial_root.exists() else set()
    ids = [s.task.task_id for s in sources]
    require(present == set(ids[:len(present)]), 'recovery trials are not an ordered source prefix')
    reports = [audit_trial(trial_root/s.task.task_id, s, expected[s.task.task_id])
               for s in sources if s.task.task_id in present]
    require(all(r['valid_for_continuation'] for r in reports[:-1]),
            'recovery continued after an incomplete trial')
    known = sum(type(r['recovery_success']) is bool for r in reports)
    complete = len(reports) == len(sources) and all(r['valid_for_continuation'] for r in reports)
    return {'status': 'complete' if complete else 'incomplete', 'scheduled_sources': len(sources),
        'attempted_sources': len(reports), 'unattempted_sources': len(sources)-len(reports),
        'known_recoveries': known, 'unknown_recoveries': len(sources)-known,
        'successful_recoveries': sum(r['recovery_success'] is True for r in reports),
        'unvalidated_candidates': sum(r['candidate_count'] for r in reports),
        'valid_for_continuation': complete, 'trials': dict(zip(ids, reports, strict=False)),
        'replayed_observations': sum(r['replayed_observations'] for r in reports),
        'new_step_cost': {k: sum(r['new_step_cost'][k] for r in reports)
                          for k in ('steps', 'prompt_tokens', 'completion_tokens')},
        'phase_response_cost': {phase: {k: sum(r['phase_response_cost'][phase][k] for r in reports)
            for k in ('prompt_tokens', 'completion_tokens')} for phase in ('recovery', 'draft')},
        'scope': 'New bounded recovery, not first-attempt success. Candidates remain unvalidated. '
                 'Source/replay token fields are historical. Step and response costs overlap.'}


def worker(root, plan, parent, inputs, index, stage):
    health = read(stage/'health.json')
    validate_health(health, parent, index)
    ids = plan['batches'][index]
    entries = [r for r in inputs['selected'] if r['task_id'] in ids]
    require([r['task_id'] for r in entries] == ids, 'worker batch source order differs')
    sources = load_sources(root, entries)
    expected = {s.task.task_id: expected_trial(s, r, plan, parent, health)
                for s, r in zip(sources, entries, strict=True)}
    require(read(stage/'expected-trials.json') == expected, 'worker expected protocol differs')
    endpoint = f'http://127.0.0.1:{plan["port"]}/v1'
    model = LocalModel(endpoint, health['model'], max_tokens=plan['max_tokens'],
                       temperature=plan['temperature'])
    with (stage/'containers.jsonl').open('x') as resources, (stage/'feedback-events.jsonl').open('x') as feedback:
        def factory(task):
            world = AppWorldEnvironment(task.task_id, parent['data_dir'], plan['max_steps_including_replay'],
                                        parent['image'], parent['deps_dir'])
            try:
                container = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}',
                    world.worker.name], text=True, timeout=15).strip()
                resources.write(json.dumps({'name': world.worker.name, 'id': container,
                                            'task_id': task.task_id})+'\n')
                resources.flush()
                return observation_world(world, parent, task.task_id, feedback)
            except BaseException:
                world.close()
                raise

        try:
            for source, record in zip(sources, entries, strict=True):
                result = run_recovery_trial(source, record['prefix_proof'], factory, model,
                    output=stage/'trials'/source.task.task_id,
                    max_steps=plan['max_steps_including_replay'], rollout_seconds=plan['recovery_seconds'],
                    draft_seconds=plan['draft_seconds'], recovery_seed=plan['recovery_seed'],
                    draft_seed=plan['draft_seed'], expected_backend=health['backend'],
                    environment_id=plan['environment_id'], before_phase=lambda phase:
                        wait_for_idle(endpoint, health['model'], health['backend'],
                                      seconds=plan['server_idle_seconds']))
                report = audit_batch(stage, sources, expected)
                atomic_json(stage/'report.json', report)
                if result['status'] != 'complete' or result['cleanup_verified'] is not True:
                    return 1
        finally:
            atomic_json(stage/'report.json', audit_batch(stage, sources, expected))
    return 0


def verify_remote(plan, parent, first_batch):
    return remote_python('''import hashlib,json,sys,subprocess,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');p=json.load(sys.stdin)
for name,expected in p['sources'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
result=subprocess.run([p['python'],'-B','artifacts/probe_qwen35_nf4_001.py',
 '--check','--plan-sha256',p['memory_sha']],cwd=root,env=dict(os.environ,
 CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),capture_output=True,text=True,timeout=15,check=True)
assert json.loads(result.stdout)['preflight']=='passed'
model=Path(p['model_path'])
files={n:hashlib.sha256((model/n).read_bytes()).hexdigest() for n in
 ['config.json','tokenizer_config.json','generation_config.json','weight-source.json'] if (model/n).is_file()}
assert files==p['backend_files']
for i in range(p['first_batch'],3):
 assert not (root/'artifacts'/f"gpu-idle-job-{p['id']}-b{i}.json").exists()
print(json.dumps({'preflight':'passed'}))
''', {'sources': plan['source_sha256'], 'python': PYTHON, 'memory_sha': NF4_MEMORY_SHA,
       'model_path': MODEL_PATH, 'backend_files': parent['expected_backend_static']['files'],
       'first_batch': first_batch, 'id': ID})


def run_batch(root, plan, parent, inputs, index, plan_path, plan_sha):
    checked = load_registration(root, plan_path, plan_sha)
    require(checked == (plan, parent, inputs), 'in-memory recovery configuration changed')
    remote_check = verify_remote(plan, parent, index)
    stage = root/'artifacts'/ID/f'batch-{index}'
    stage.mkdir()
    atomic_json(stage/'freeze-check.json', {'plan_sha256': plan_sha, 'checked_epoch': time.time(),
                                          'local_files_match': True, 'remote': remote_check})
    entries = [r for r in inputs['selected'] if r['task_id'] in plan['batches'][index]]
    sources = load_sources(root, entries)
    verify_cpu_environment(parent, [asdict(s.task) for s in sources], stage/'cpu-environment-preflight.json')
    job = make_job(index, uuid.uuid4().hex, plan_sha)
    atomic_json(stage/'job.json', job)
    state, known = {}, set()
    stopped = threading.Event()
    thread = tunnel = client = None
    report = release = None

    def heartbeat():
        while not stopped.is_set():
            try:
                snapshot = remote_queue('pulse', job, known)
                known.update(snapshot.get('model_pids', []))
                state.update(snapshot, local_observed_monotonic=time.monotonic())
                atomic_json(stage/'remote-state.json', snapshot)
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
        with (stage/'tunnel.log').open('x') as log:
            tunnel = subprocess.Popen([*SSH[:-1], '-N', '-o', 'ExitOnForwardFailure=yes',
                '-L', f'{PORT}:127.0.0.1:{PORT}', '4090'], stdout=log, stderr=subprocess.STDOUT)
        loading = None
        while True:
            require(tunnel.poll() is None, 'SSH tunnel exited')
            require(state.get('owned') is True and state.get('monitor_live') is True,
                    'queue ownership or monitor identity unconfirmed')
            require(time.monotonic()-state['local_observed_monotonic'] <= 90, 'remote status stale')
            remote_job = state.get('job') or {}
            require(remote_job.get('status') != 'finished', 'server exited before health verification')
            if remote_job.get('status') == 'running':
                loading = loading or time.monotonic()
                require(time.monotonic()-loading <= plan['model_start_seconds'], 'model startup timed out')
                try:
                    with RolloutDeadline(6), opener.open(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
                        raw = response.read(65_537)
                    require(len(raw) <= 65_536, 'oversized health response')
                    health = json.loads(raw)
                    break
                except OSError:
                    pass
            time.sleep(2)
        validate_health(health, parent, index)
        confirmed = remote_queue('status', job, known)
        require(confirmed['owned'] and (confirmed.get('job') or {}).get('status') == 'running'
                and confirmed.get('model_pids'), 'live recovery service identity missing')
        known.update(confirmed['model_pids'])
        atomic_json(stage/'health.json', health)
        expected = {s.task.task_id: expected_trial(s, r, plan, parent, health)
                    for s, r in zip(sources, entries, strict=True)}
        atomic_json(stage/'expected-trials.json', expected)
        with (stage/'client.log').open('x') as log:
            client = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.run_recovery_pilot',
                '--plan', str(plan_path), '--plan-sha256', plan_sha, '--worker', '--batch', str(index),
                '--stage', str(stage)], cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                PYTHONPATH=str(root), TMPDIR=parent['tmp_dir']), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            code = client.wait(timeout=plan['client_seconds_per_batch'])
        try:
            report = audit_batch(stage, sources, expected)
            atomic_json(stage/'report.json', report)
        except Exception as exc:
            atomic_json(stage/'report-failure.json', {'client_exit_code': code,
                'error_type': type(exc).__name__, 'success_rate': None,
                'reason': 'Incomplete or inconsistent raw recovery artifacts; no results imputed'})
            raise
        require(code == 0 and report['valid_for_continuation'], 'recovery batch incomplete; no automatic retry')
    finally:
        try:
            stop_client(client, stage)
        finally:
            stopped.set()
            if thread is not None:
                thread.join(timeout=25)
            try:
                release = remote_queue('stop', job, known)
                deadline = time.monotonic()+90
                while release['owned'] and not release['released'] and time.monotonic() < deadline:
                    time.sleep(5)
                    release = remote_queue('status', job, known)
                atomic_json(stage/'release.json', release)
                require(release['owned'] and release['released'], 'recovery service release unconfirmed')
            finally:
                if tunnel is not None and tunnel.poll() is None:
                    tunnel.terminate()
                    tunnel.wait(timeout=10)
    return report, release


def combine_batches(reports, releases):
    require(len(reports) == len(releases) == 3
            and [r['scheduled_sources'] for r in reports] == [3, 3, 2]
            and all(r['valid_for_continuation'] for r in reports)
            and all(r['released'] is True for r in releases), 'complete released recovery batches required')
    return {'scheduled_sources': 8, 'completed_sources': 8,
        'successful_recoveries': sum(r['successful_recoveries'] for r in reports),
        'unvalidated_candidates': sum(r['unvalidated_candidates'] for r in reports),
        'replayed_observations': sum(r['replayed_observations'] for r in reports),
        'new_step_cost': {k: sum(r['new_step_cost'][k] for r in reports)
                          for k in ('steps', 'prompt_tokens', 'completion_tokens')},
        'phase_response_cost': {phase: {k: sum(r['phase_response_cost'][phase][k] for r in reports)
            for k in ('prompt_tokens', 'completion_tokens')} for phase in ('recovery', 'draft')},
        'automatic_utility_or_training': False,
        'scope': 'One extra recovery per natural failed source; not first-attempt performance. '
                 'Any generated candidates still require separately registered independent utility.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=PLAN)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--batch', type=int, choices=(0, 1, 2))
    parser.add_argument('--stage')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'recovery controller/worker must be CPU-only')
    plan_path = Path(args.plan).resolve(strict=True)
    plan_sha = args.plan_sha256 if args.worker else digest(plan_path)
    plan, parent, inputs = load_registration(ROOT, plan_path, plan_sha)
    if args.worker:
        require(args.batch is not None and args.stage is not None, 'worker batch and stage required')
        stage = Path(args.stage).resolve()
        require(stage == ROOT/'artifacts'/ID/f'batch-{args.batch}', 'worker stage differs from registration')
        raise SystemExit(worker(ROOT, plan, parent, inputs, args.batch, stage))
    with (ROOT/'artifacts/foundation-pilot-001.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = ROOT/'artifacts'/ID
        require(not output.exists(), 'recovery controller already has an attempt')
        require_space(output)
        image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                        parent['image']], text=True, timeout=15).strip()
        require(image == parent['image'], 'recovery execution image changed')
        verify_remote(plan, parent, 0)
        if args.check:
            print(json.dumps({'preflight': 'passed', 'sources': 8, 'batches': 3, 'uses_gpu': False}))
            return
        output.mkdir()
        state = {'phase': 'running', 'pid': os.getpid(), 'plan_sha256': plan_sha,
                 'started_epoch': time.time(), 'completed_batches': 0}
        atomic_json(output/'status.json', state)
        try:
            reports, releases = [], []
            for index in range(3):
                state.update(current_batch=index, checked_epoch=time.time())
                atomic_json(output/'status.json', state)
                report, release = run_batch(ROOT, plan, parent, inputs, index, plan_path, plan_sha)
                reports.append(report)
                releases.append(release)
                state['completed_batches'] = len(reports)
            result = combine_batches(reports, releases)
            atomic_json(output/'report.json', result)
            state.update(phase='complete', result=result)
        except BaseException as exc:
            state.update(phase='failed_or_interrupted', error_type=type(exc).__name__, error=str(exc)[:1000])
            raise
        finally:
            state['finished_epoch'] = time.time()
            atomic_json(output/'status.json', state)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
