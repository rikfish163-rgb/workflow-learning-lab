"""One registered batch of three critic requests on one idle-qualified service."""
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
from collections import Counter
from pathlib import Path

from scripts.report_compact_critic import audit_critic
from scripts.run_compact_critic import run_critic
from scripts.run_foundation_pilot import (
    digest,
    process_matches,
    read,
    require,
    stop_client,
    verify_frozen_plan,
)
from scripts.run_queued_decoder_completion import SSH, remote_python
from scripts.run_qwen38_capability import (
    PYTHON,
    REMOTE,
    ROOT,
    SERVICE_SHA,
    runtime_state,
    validate_live_service,
)
from scripts.run_qwen38_recovery import collect_remote_evidence
from scripts.run_qwen38_recovery import load_registration as load_parent_registration
from scripts.run_timed_tasks import RolloutDeadline, wait_for_idle
from scripts.semantic_repair_compact import critic_messages, project_failure
from wll.cli import read_episodes
from wll.model import LocalModel
from wll.protocol import fingerprint
from wll.storage import atomic_json, require_space

ID = 'qwen38-27b-semantic-critic-002'
PORT = 18979
PLAN = 'artifacts/qwen38-semantic-critic-plan-002.json'
DESIGN = 'artifacts/qwen38-semantic-critic-design-002.json'
PARENT_SHA = '1c167eb405b9210ca9c54e8188e310a60cce6fe8c2191e657433b8ba0cc37de4'
COMPLETE = 'artifacts/qwen38-recovery-complete-integration-001.json'
PARTITION = 'artifacts/partitions-v1/manifest.json'


def load_sources(root, entries):
    sources = []
    for row in entries:
        name = row['source_file']
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'non-project source')
        episodes = read_episodes(Path(root) / name)
        require(len(episodes) == 1 and episodes[0].task.task_id == row['task_id']
                and fingerprint(episodes[0].to_dict()) == row['source_hash'], 'source episode differs')
        sources.extend(episodes)
    return sources


def validate_study(root, plan):
    root = Path(root)
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['design'] == DESIGN
            and plan['design_sha256'] == digest(root / DESIGN), 'unchanged registered critic design required')
    design = read(root / DESIGN)
    require(all(plan.get(k) == v for k, v in design.items() if k not in {'state', 'missing_before_registration'}),
            'registered critic settings differ')
    previous = verify_frozen_plan(root, root / plan['predecessor_critic_plan'], plan['predecessor_critic_plan_sha256'])
    prior = read(root / plan['predecessor_complete'])
    require(previous['id'] == 'qwen38-27b-semantic-critic-001'
            and plan['predecessor_critic_plan_sha256'] == '5a3b1c1b5636d8eea1a60be9a93dcd9f5167b7dd519c31ba6fb133a90065d402'
            and digest(root / plan['predecessor_complete']) == plan['predecessor_complete_sha256']
            and prior['phase'] == 'complete' and prior['result']['known_outputs'] == 3
            and prior['result']['eligible_for_prefix_audit'] == 0
            and prior['native_controller_and_forwarder_released'], 'previous negative batch differs')
    for name, sha in prior['input_sha256'].items():
        require(digest(root / name) == sha, 'previous critic evidence changed')
    require(plan['parent_plan_sha256'] == PARENT_SHA and plan['recovery_integration'] == COMPLETE
            and plan['recovery_integration_sha256'] == digest(root / COMPLETE), 'parent or completion evidence differs')
    parent, _, _ = load_parent_registration(root, root / plan['parent_plan'], PARENT_SHA)
    whole, state = read(root / COMPLETE), read(root / 'artifacts' / parent['id'] / 'status.json')
    require(whole['phase'] == state['phase'] == 'complete' and state['result'] == whole['result']
            and state['plan_sha256'] == PARENT_SHA and whole['result']['known_followups'] == 3
            and whole['result']['unvalidated_candidates'] == 0 and whole['native_controller_and_forwarder_released']
            and all(whole['recorded_container_absence'].values()), 'complete released parent required')
    require(plan['expected_backend_static'] == parent['expected_backend_static']
            and plan['gpu_policy'] == parent['gpu_policy'] and plan['tmp_dir'] == parent['tmp_dir']
            and plan['wll_code_fingerprint'] == parent['wll_code_fingerprint'], 'backend, policy or runtime differs')
    for name, sha in whole['input_sha256'].items():
        require(digest(root / name) == sha, 'completed recovery evidence changed: ' + name)
    require(digest(root / plan['inputs']) == plan['inputs_sha256']
            and digest(root / plan['CPU_context_preflight']) == plan['CPU_context_preflight_sha256'],
            'critic preparation changed')
    inputs, context = read(root / plan['inputs']), read(root / plan['CPU_context_preflight'])
    require(inputs['state'] == 'prepared_not_registered' and context['phase'] == 'complete'
            and context['uses_gpu'] is False and context['model_loaded'] is False
            and context['manifest_sha256'] == plan['inputs_sha256']
            and context['source_sha256'] == inputs['source_sha256'] == plan['proposal_source_sha256']
            and context['max_context'] == plan['max_context'], 'critic preflight differs')
    for name, sha in inputs['source_sha256'].items():
        require(digest(root / name) == sha, 'critic projection implementation changed')
    require(digest(root / PARTITION) == inputs['partition_sha256'], 'origin partition changed')
    origin = {x.rsplit('_', 1)[0] for x in read(root / PARTITION)['task_ids']['origin']}
    entries = inputs['ordered_sources']
    require([r['task_id'] for r in entries] == plan['task_ids'] == parent['task_ids']
            == [r['task_id'] for r in context['inputs']], 'critic source order differs')
    for source, row, ctx in zip(load_sources(root, entries), entries, context['inputs'], strict=True):
        require(row['source_file'] == f"artifacts/{parent['id']}/batch-0/trials/{source.task.task_id}/episodes.jsonl"
                and digest(root / row['source_file']) == row['source_file_sha256'], 'not the registered failed followup')
        for kind in ('projection', 'messages'):
            require(not Path(row[kind]).is_absolute() and '..' not in Path(row[kind]).parts
                    and digest(root / row[kind]) == row[kind + '_sha256'], 'prepared critic input changed')
        projection = project_failure(source, origin, max_chars=plan['projection_max_chars'])
        require(read(root / row['projection']) == projection
                and read(root / row['messages']) == critic_messages(projection)
                and ctx['messages_sha256'] == row['messages_sha256']
                and ctx['max_new_tokens'] == plan['max_tokens']
                and 0 < ctx['prompt_tokens'] and ctx['total_max_tokens'] == ctx['prompt_tokens'] + plan['max_tokens']
                and ctx['total_max_tokens'] <= plan['max_context'], 'critic projection or token budget differs')
    return parent, inputs


def load_registration(root, path, sha):
    root = Path(root)
    plan = verify_frozen_plan(root, path, sha)
    parent, inputs = validate_study(root, plan)
    required = set(parent['source_sha256']) | {'scripts/run_qwen38_compact_critic.py',
        'scripts/run_compact_critic.py', 'scripts/report_compact_critic.py', 'scripts/semantic_repair_compact.py'}
    require(required <= plan['source_sha256'].keys() and all(plan['source_sha256'].get(n) == s
            for n, s in parent['source_sha256'].items()), 'critic source bindings incomplete or changed')
    required_inputs = {DESIGN, plan['parent_plan'], COMPLETE, plan['inputs'], plan['CPU_context_preflight'],
                       'artifacts/semantic-critic-cpu-verification-002.json', plan['predecessor_critic_plan'],
                       plan['predecessor_complete']}
    for row in inputs['ordered_sources']:
        required_inputs.update(row[k] for k in ('source_file', 'projection', 'messages'))
    require(required_inputs <= plan['input_sha256'].keys(), 'critic input bindings incomplete')
    for mapping in (parent['input_sha256'], read(root / COMPLETE)['input_sha256']):
        require(all(plan['input_sha256'].get(n) == s for n, s in mapping.items()), 'parent evidence binding differs')
    return plan, parent, inputs


def make_job(owner, plan_sha):
    require(re.fullmatch('[a-f0-9]{32}', owner) and re.fullmatch('[a-f0-9]{64}', plan_sha), 'invalid owner or plan hash')
    return {'id': ID + '-b0', 'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 1800,
        'argv': [PYTHON, '-B', '-m', 'scripts.serve_qwen38', '--plan-sha256', SERVICE_SHA,
                 '--model-id', ID + '-b0', '--port', str(PORT)]}


def expected_protocol(source, plan, health):
    origin = sorted({x.rsplit('_', 1)[0] for x in read(ROOT / PARTITION)['task_ids']['origin']})
    return {'source_hash': fingerprint(source.to_dict()), 'model_id': health['model'],
        'max_tokens': plan['max_tokens'], 'temperature': plan['temperature'], 'seed': plan['critic_seed'],
        'seconds': plan['critic_seconds'], 'expected_backend': health['backend'], 'origin_scenarios': origin,
        'max_projection_chars': plan['projection_max_chars'], 'backend_idle_check_enabled': True,
        'script_sha256': plan['source_sha256']['scripts/run_compact_critic.py'],
        'proposal_source_sha256': plan['source_sha256']['scripts/semantic_repair_compact.py']}


def audit_batch(stage, sources, expected):
    directory = stage / 'requests'
    present = {p.name for p in directory.iterdir() if p.is_dir()} if directory.exists() else set()
    ids = [s.task.task_id for s in sources]
    require(len(ids) == len(set(ids)) and present == set(ids[:len(present)]), 'critic attempts not an ordered prefix')
    reports = {}
    for source in sources:
        if source.task.task_id not in present:
            continue
        require(all(r['valid_for_continuation'] for r in reports.values()), 'continued after incomplete critic')
        reports[source.task.task_id] = audit_critic(directory / source.task.task_id, source, expected[source.task.task_id])
    known = sum(r['valid_for_continuation'] for r in reports.values())
    complete = known == len(ids)
    return {'status': 'complete' if complete else 'incomplete', 'scheduled_sources': len(ids),
        'attempted_sources': len(reports), 'unattempted_sources': len(ids) - len(reports),
        'known_outputs': known, 'unknown_outputs': len(reports) - known, 'valid_for_continuation': complete,
        'output_status_counts': dict(Counter(r['proposal_status'] for r in reports.values())),
        'eligible_for_prefix_audit': sum(r['eligible_for_prefix_audit'] for r in reports.values()),
        'generation_counters': {k: sum(r['generation_counters'][k] for r in reports.values()) for k in
            ('generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage')},
        'returned_response_cost': {k: sum(r['returned_response_cost'][k] for r in reports.values())
                                  for k in ('prompt_tokens', 'completion_tokens')},
        'unknown_additional_response_cost': any(r['unknown_additional_response_cost'] for r in reports.values()),
        'trials': reports, 'accepted_experiences': 0, 'automatic_execution': False,
        'scope': 'Critic output collection only; citations do not prove causal correctness. No prefix replay, '
                 'autonomous recovery, candidate utility or parameter training is implied.'}


def worker(root, plan, parent, inputs, stage):
    health = read(stage / 'health.json')
    validate_health(health, plan)
    entries = inputs['ordered_sources']
    require([r['task_id'] for r in entries] == plan['task_ids'], 'worker source order differs')
    sources = load_sources(root, entries)
    expected = {s.task.task_id: expected_protocol(s, plan, health) for s in sources}
    require(read(stage / 'expected.json') == expected, 'worker expected protocols differ')
    endpoint = f"http://127.0.0.1:{plan['port']}/v1"
    model = LocalModel(endpoint, health['model'], max_tokens=plan['max_tokens'], temperature=plan['temperature'])
    try:
        for source, row in zip(sources, entries, strict=True):
            result = run_critic(source, set(expected[source.task.task_id]['origin_scenarios']),
                read(Path(root) / row['projection']), model, output=stage / 'requests' / source.task.task_id,
                seed=plan['critic_seed'], seconds=plan['critic_seconds'], expected_backend=health['backend'],
                max_projection_chars=plan['projection_max_chars'],
                before_call=lambda: wait_for_idle(endpoint, health['model'], health['backend'],
                                                    seconds=plan['server_idle_seconds']))
            atomic_json(stage / 'report.json', audit_batch(stage, sources, expected))
            if result['status'] != 'complete':
                return 1
    finally:
        atomic_json(stage / 'report.json', audit_batch(stage, sources, expected))
    return 0


def verify_remote(plan_sha):
    return remote_python('''import json,sys,subprocess,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));p=json.load(sys.stdin)
from scripts.run_qwen38_compact_critic import load_registration,PLAN,PYTHON,ID,SERVICE_SHA
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
                raise FileExistsError('critic batch already attempted or owned')
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
from scripts.run_qwen38_compact_critic import queue_operation
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
        expected = {s.task.task_id: expected_protocol(s, plan, health) for s in sources}
        atomic_json(stage / 'expected.json', expected)
        with (stage / 'client.log').open('x') as log:
            client = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'scripts.run_qwen38_compact_critic',
                '--plan', str(plan_path), '--plan-sha256', plan_sha, '--worker', '--stage', str(stage)],
                cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root), TMPDIR=plan['tmp_dir']),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = client.wait(timeout=plan['client_seconds'])
        try:
            report = audit_batch(stage, sources, expected)
            atomic_json(stage / 'report.json', report)
        except Exception as exc:
            atomic_json(stage / 'report-failure.json', {'client_exit_code': code,
                'error_type': type(exc).__name__, 'success_rate': None,
                'reason': 'Raw critic artifacts incomplete or inconsistent; no outcome imputed'})
            raise
        require(code == 0 and report['valid_for_continuation'], 'critic batch incomplete; no automatic retry')
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
        require(not output.exists(), 'critic study has an attempt; no automatic restart')
        require_space(ROOT / 'artifacts')
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
