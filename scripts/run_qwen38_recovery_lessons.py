"""One registered pair of standalone lesson calls on an idle-qualified GPU.

The frozen predecessor controller is retained unchanged; this version binds the
new writer, audited recovery inputs, and candidate-only authority.
"""
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
from types import SimpleNamespace

from scripts.prepare_recovery_lesson_inputs import lesson_messages
from scripts.report_recovery_lesson import audit_lesson
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
from scripts.run_qwen38_proposed_pair import load_registration as load_parent_registration
from scripts.run_qwen38_recovery import collect_remote_evidence
from scripts.run_recovery_lesson import protocol, run_lesson
from scripts.run_timed_tasks import RolloutDeadline, wait_for_idle
from wll.cli import read_episodes
from wll.model import LocalModel
from wll.protocol import fingerprint
from wll.storage import atomic_json, require_space

ID = 'qwen38-recovery-lessons-001'
PORT = 18979
PLAN = 'artifacts/qwen38-recovery-lessons-plan-001.json'
DESIGN = 'artifacts/recovery-lesson-design-001.json'
CONTEXT = 'artifacts/recovery-lesson-context-preflight-001.json'
COMPLETE = 'artifacts/proposed-pair-complete-integration-001.json'
PARENT_SHA = '0520a92e25ea994e38ae1fbe58f80ad3c93487e2cd6f7ef07c0daa05084c033a'
PARTITION = 'artifacts/partitions-v1/manifest.json'
CONDITIONS = ['no_hint', 'proposal_hint']
NEW_SOURCES = ['scripts/run_recovery_lesson.py', 'scripts/report_recovery_lesson.py',
    'scripts/run_qwen38_recovery_lessons.py', 'scripts/prepare_recovery_lesson_inputs.py']


def load_sources(root, entries):
    pairs = []
    for row in entries:
        for key in ('source_file', 'recovery_file'):
            require(not Path(row[key]).is_absolute() and '..' not in Path(row[key]).parts, 'non-project source')
        source, = read_episodes(Path(root) / row['source_file'])
        recovery, = read_episodes(Path(root) / row['recovery_file'])
        require(source.task == recovery.task and source.task.task_id == row['task_id']
            and fingerprint(source.to_dict()) == row['source_hash']
            and fingerprint(recovery.to_dict()) == row['recovery_hash']
            and digest(Path(root) / row['recovery_file']) == row['recovery_file_sha256'], 'source pair differs')
        pairs.append((source, recovery, row['condition']))
    return pairs


def validate_study(root, plan):
    root = Path(root)
    require(plan['id'] == ID and plan['state'] == 'registered' and plan['design'] == DESIGN
        and plan['design_sha256'] == digest(root / DESIGN), 'registered unchanged writer design required')
    design = read(root / DESIGN)
    require(all(plan.get(k) == v for k, v in design.items() if k not in {'state', 'missing_before_registration'}),
        'writer design settings differ')
    require(plan['parent_plan_sha256'] == PARENT_SHA and plan['complete_integration'] == COMPLETE
        and plan['complete_integration_sha256'] == digest(root / COMPLETE), 'parent pair evidence differs')
    parent, _, _ = load_parent_registration(root, root / plan['parent_plan'], PARENT_SHA)
    complete = read(root / COMPLETE)
    require(complete['phase'] == 'complete' and complete['result']['outcomes'] == dict.fromkeys(CONDITIONS, True)
        and complete['result']['hint_helped'] is False and complete['result']['hint_harmed'] is False
        and complete['result']['accepted_experiences'] == 0
        and complete['native_controller_and_forwarder_released'], 'audited successful pair required')
    for name, sha in complete['input_sha256'].items():
        require(digest(root / name) == sha, 'completed pair evidence changed')
    for key in ('expected_backend_static', 'gpu_policy', 'gpu_policy_sha256', 'tmp_dir', 'wll_code_fingerprint'):
        require(plan[key] == parent[key], 'parent runtime or backend differs: ' + key)
    require(plan['conditions'] == CONDITIONS and plan['port'] == PORT and plan['model_start_seconds'] == 600
        and plan['server_idle_seconds'] == 180 and plan['client_seconds'] == 600, 'writer lifecycle budget differs')
    require(plan['CPU_context_preflight'] == CONTEXT and plan['CPU_context_preflight_sha256'] == digest(root / CONTEXT)
        and digest(root / plan['inputs']) == plan['inputs_sha256'], 'writer input preparation changed')
    inputs, context = read(root / plan['inputs']), read(root / CONTEXT)
    require(inputs['state'] == 'prepared_not_registered' and context['phase'] == 'complete'
        and context['uses_gpu'] is context['model_loaded'] is False and context['LLM_calls'] == 0
        and context['manifest_sha256'] == plan['inputs_sha256']
        and inputs['parent_plan_sha256'] == PARENT_SHA and inputs['complete_integration_sha256'] == digest(root / COMPLETE)
        and inputs['partition_sha256'] == digest(root / PARTITION), 'writer context evidence differs')
    for name, sha in inputs['source_sha256'].items():
        require(digest(root / name) == sha, 'prepared writer implementation changed')
    entries = inputs['ordered_sources']
    require([r['condition'] for r in entries] == CONDITIONS == [r['condition'] for r in context['inputs']],
        'writer order differs')
    origin = {x.rsplit('_', 1)[0] for x in read(root / PARTITION)['task_ids']['origin']}
    pairs = load_sources(root, entries)
    require(pairs[0][0] == pairs[1][0], 'writer source failure differs between conditions')
    for (source, recovery, condition), row, ctx in zip(pairs, entries, context['inputs'], strict=True):
        require(row['recovery_file'] == f"artifacts/{parent['id']}/batch-0/trials/{condition}/episodes.jsonl"
            and recovery.replayed_steps == row['replayed_steps'], 'not the registered recovery')
        path = row['messages']
        require(not Path(path).is_absolute() and '..' not in Path(path).parts
            and digest(root / path) == row['messages_sha256'] == ctx['messages_sha256'], 'writer messages changed')
        messages = lesson_messages(source, recovery, condition=condition, origin_scenarios=origin)
        require(read(root / path) == messages and fingerprint(messages) == row['message_fingerprint']
            and row['draft_seed'] == plan['draft_seed'] and ctx['max_new_tokens'] == plan['max_tokens']
            and 0 < ctx['prompt_tokens'] and ctx['total_max_tokens'] == ctx['prompt_tokens'] + plan['max_tokens']
            and ctx['total_max_tokens'] <= plan['max_context'], 'writer prompt or token budget differs')
    return parent, inputs


def load_registration(root, path, sha):
    root = Path(root)
    plan = verify_frozen_plan(root, path, sha)
    parent, inputs = validate_study(root, plan)
    require(set(parent['source_sha256']) | set(NEW_SOURCES) <= plan['source_sha256'].keys()
        and all(plan['source_sha256'].get(n) == s for n, s in parent['source_sha256'].items()),
        'writer source bindings incomplete')
    required = {DESIGN, CONTEXT, COMPLETE, plan['inputs'], plan['parent_plan'],
        'artifacts/recovery-lesson-inputs-verification-001.json', 'artifacts/recovery-lessons-cpu-verification-001.json'}
    for row in inputs['ordered_sources']:
        required.update(row[k] for k in ('source_file', 'recovery_file', 'messages'))
    require(required <= plan['input_sha256'].keys(), 'writer input bindings incomplete')
    for mapping in (parent['input_sha256'], read(root / COMPLETE)['input_sha256']):
        require(all(plan['input_sha256'].get(n) == s for n, s in mapping.items()), 'parent bindings differ')
    return plan, parent, inputs


def expected_protocol(pair, plan, health):
    source, recovery, condition = pair
    origin = {x.rsplit('_', 1)[0] for x in read(ROOT / PARTITION)['task_ids']['origin']}
    model = SimpleNamespace(model_id=health['model'], max_tokens=plan['max_tokens'], temperature=plan['temperature'])
    expected = protocol(source, recovery, model, condition=condition, origin_scenarios=origin,
        seed=plan['draft_seed'], seconds=plan['draft_seconds'], expected_backend=health['backend'],
        backend_idle_check_enabled=True)
    require(expected['script_sha256'] == plan['source_sha256']['scripts/run_recovery_lesson.py']
        and expected['preparation_sha256'] == plan['source_sha256']['scripts/prepare_recovery_lesson_inputs.py'],
        'writer implementation differs from registration')
    return expected


def audit_batch(stage, sources, expected):
    directory = stage / 'requests'
    present = {p.name for p in directory.iterdir() if p.is_dir()} if directory.exists() else set()
    ids = [p[2] for p in sources]
    require(ids == CONDITIONS and present == set(ids[:len(present)]), 'writer attempts not an ordered prefix')
    reports = {}
    for source, recovery, condition in sources:
        if condition not in present:
            continue
        require(all(r['valid_for_continuation'] for r in reports.values()), 'continued after incomplete writer')
        reports[condition] = audit_lesson(directory / condition, source, recovery, expected[condition])
    known = sum(r['valid_for_continuation'] for r in reports.values())
    complete = known == len(ids)
    return {'status': 'complete' if complete else 'incomplete', 'scheduled_sources': len(ids),
        'attempted_sources': len(reports), 'unattempted_sources': len(ids) - len(reports),
        'known_outputs': known, 'unknown_outputs': len(reports) - known, 'valid_for_continuation': complete,
        'output_status_counts': dict(Counter(r['candidate_status'] for r in reports.values())),
        'unvalidated_candidates': sum(r['unvalidated_candidates'] for r in reports.values()),
        'generation_counters': {k: sum(r['generation_counters'][k] for r in reports.values()) for k in
            ('generation_calls', 'generation_calls_without_response', 'generation_calls_with_unusable_usage')},
        'returned_response_cost': {k: sum(r['returned_response_cost'][k] for r in reports.values())
            for k in ('prompt_tokens', 'completion_tokens')},
        'unknown_additional_response_cost': any(r['unknown_additional_response_cost'] for r in reports.values()),
        'trials': reports, 'accepted_experiences': 0, 'automatic_training': False,
        'scope': 'Two correlated candidates from one post-hoc origin. No demonstrated hint success-rate '
            'benefit, independent utility or parameter learning.'}


def worker(root, plan, parent, inputs, stage):
    health = read(stage / 'health.json')
    validate_health(health, plan)
    sources = load_sources(root, inputs['ordered_sources'])
    expected = {p[2]: expected_protocol(p, plan, health) for p in sources}
    require(read(stage / 'expected.json') == expected, 'worker writer protocol differs')
    endpoint = f"http://127.0.0.1:{plan['port']}/v1"
    model = LocalModel(endpoint, health['model'], max_tokens=plan['max_tokens'], temperature=plan['temperature'])
    try:
        for source, recovery, condition in sources:
            result = run_lesson(source, recovery, model, output=stage / 'requests' / condition,
                condition=condition, origin_scenarios=set(expected[condition]['origin_scenarios']),
                seed=plan['draft_seed'], seconds=plan['draft_seconds'], expected_backend=health['backend'],
                before_call=lambda: wait_for_idle(endpoint, health['model'], health['backend'],
                    seconds=plan['server_idle_seconds']))
            atomic_json(stage / 'report.json', audit_batch(stage, sources, expected))
            if result['status'] != 'complete':
                return 1
    finally:
        atomic_json(stage / 'report.json', audit_batch(stage, sources, expected))
    return 0



def make_job(owner, plan_sha):
    require(re.fullmatch('[a-f0-9]{32}', owner) and re.fullmatch('[a-f0-9]{64}', plan_sha), 'invalid owner or plan hash')
    return {'id': ID + '-b0', 'owner': owner, 'plan_sha256': plan_sha, 'max_runtime_seconds': 1800,
        'argv': [PYTHON, '-B', '-m', 'scripts.serve_qwen38', '--plan-sha256', SERVICE_SHA,
                 '--model-id', ID + '-b0', '--port', str(PORT)]}


def verify_remote(plan_sha):
    return remote_python('''import json,sys,subprocess,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));p=json.load(sys.stdin)
from scripts.run_qwen38_recovery_lessons import load_registration,PLAN,PYTHON,ID,SERVICE_SHA
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
                raise FileExistsError('writer batch already attempted or owned')
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
from scripts.run_qwen38_recovery_lessons import queue_operation
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
        expected = {s[2]: expected_protocol(s, plan, health) for s in sources}
        atomic_json(stage / 'expected.json', expected)
        with (stage / 'client.log').open('x') as log:
            client = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'scripts.run_qwen38_recovery_lessons',
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
                'reason': 'Raw writer artifacts incomplete or inconsistent; no outcome imputed'})
            raise
        require(code == 0 and report['valid_for_continuation'], 'writer batch incomplete; no automatic retry')
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
        require(not output.exists(), 'writer study has an attempt; no automatic restart')
        require_space(ROOT / 'artifacts')
        verify_remote(args.plan_sha256)
        if args.check:
            print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'batches': 1, 'sources': 2}))
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
