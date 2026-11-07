"""Three sequential idle-qualified batches, activated only by a registered memory-backed plan."""
import argparse
import fcntl
import hashlib
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
from scripts.run_queued_decoder_completion import SSH, remote_python
from scripts.run_timed_tasks import RolloutDeadline, run_tasks, wait_for_idle
from scripts.serve_transformers_presence import quantization_metadata
from wll.appworld import AppWorldEnvironment, AppWorldWorker, load_tasks
from wll.model import LocalModel
from wll.protocol import fingerprint
from wll.storage import atomic_json, require_space

ROOT = Path(__file__).resolve().parent.parent
REMOTE = Path('/home/hts/workflow-learning-lab')
ID = 'qwen35-9b-foundation-002'
OBSERVATION_ID = 'qwen35-9b-observation-001'
FEEDBACK_SOURCES = {
    'scripts/documentation_feedback.py': 'e7eb24a8c87a5304815514203c483cfd4c2fe5bd32b0f82e5cd07b9de032b5b2',
    'scripts/probe_observation_visibility.py': 'd9bd7d258eabdc006a9aa470eff2cc0d06bf14e8437c4b1c74d3e6287129e301',
}
MODEL_PATH = '/home/data4/hts/workflow-learning-lab/models/qwen35-9b-c2022362'
PYTHON = '/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python'
MODULE = 'scripts.serve_transformers_presence'
PLAN = 'artifacts/qwen35-9b-foundation-plan-002.json'
NF4_MEMORY_SHA = '7e6e85e948618a4979671061709ce39ae3ac2f6c80302a3500d164394e8c5865'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def study_id(plan):
    require(plan.get('id') in (ID, OBSERVATION_ID), 'unregistered study identity')
    return plan['id']


def environment_id(plan):
    if study_id(plan) == ID:
        require('observation_feedback' not in plan
                and plan.get('environment_id', plan['image']) == plan['image'],
                'baseline observation configuration changed')
        return plan['image']
    feedback = {'kind': 'silent_documentation_print_hint_v1', 'source_sha256': FEEDBACK_SOURCES}
    require(plan.get('observation_feedback') == feedback, 'unregistered observation feedback')
    require(all(plan['source_sha256'].get(k) == v for k, v in FEEDBACK_SOURCES.items()),
            'feedback sources are not frozen')
    identity = fingerprint({'image': plan['image'], 'observation_feedback': feedback})
    require(plan.get('environment_id') == identity, 'observation environment identity changed')
    return identity


def observation_world(world, plan, task_id, journal):
    environment_id(plan)
    if study_id(plan) == ID:
        return world
    # Lazy import: the CPU selector also uses this module's read-only audit helpers.
    from scripts.documentation_feedback import DocumentationFeedbackWorld

    class AuditedFeedbackWorld(DocumentationFeedbackWorld):
        action_count = 0

        def execute(self, action):
            index = self.action_count
            self.action_count += 1
            before = self.feedback_events
            observation = super().execute(action)
            if self.feedback_events != before:
                raw = 'Execution successful.'
                journal.write(json.dumps({'task_id': task_id, 'index': index,
                    'action_sha256': hashlib.sha256(action.encode()).hexdigest(),
                    'raw_observation_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                    'raw_observation_chars': len(raw),
                    'augmented_observation_sha256': hashlib.sha256(observation.encode()).hexdigest(),
                    'augmented_observation_chars': len(observation)})+'\n')
                journal.flush()
            return observation

    return AuditedFeedbackWorld(world)


def validate_observation_comparison(root, plan):
    if study_id(plan) == ID:
        return
    references = {
        'artifacts/qwen35-9b-foundation-plan-002.json':
            'f8f598cdca2600a36656b9fc20b45eb1b556e9e1f0378caa2d992962290376ce',
        'artifacts/observation-feedback-design-001.json':
            '0555e15d2678598b93ba416eeebd12ae355150ed68927a36ff251ef2adf9ee26',
        'artifacts/foundation-complete-integration-002.json':
            'fc9efdae074fe4b4a9a7d20130c71a67a3e409bb3a164eeea9d154a039071c29',
        'artifacts/observation-feedback-integration-001.json':
            'b792c829c85ff816e2f0f6ab1e0752a1a85cddab680e97e59de4131c4d4b6f52',
    }
    require(plan.get('baseline_plan') == PLAN, 'original baseline required')
    require(all(plan['input_sha256'].get(k) == v and digest(root/k) == v
                for k, v in references.items()), 'comparison evidence changed or unbound')
    baseline = read(root/PLAN)
    design = read(root/'artifacts/observation-feedback-design-001.json')
    fields = set(design['scientific_fields_to_preserve']) | {
        'port', 'data_dir', 'deps_dir', 'tmp_dir', 'memory', 'success_gate', 'gpu_policy',
        'task_selection', 'stop_rule', 'wll_code_fingerprint'}
    require(all(plan.get(k) == baseline[k] for k in fields), 'comparison science changed')
    require(all(plan['source_sha256'].get(k) == v for k, v in baseline['source_sha256'].items()
                if k != 'scripts/run_foundation_pilot.py'), 'shared baseline source changed')


def verify_frozen_plan(root, plan_path, plan_sha, *, expected_plan=None):
    require(isinstance(plan_sha, str) and re.fullmatch('[a-f0-9]{64}', plan_sha),
            'original plan SHA256 required')
    raw = Path(plan_path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == plan_sha, 'frozen plan changed')
    plan = json.loads(raw)
    require(expected_plan is None or plan == expected_plan, 'in-memory plan changed')
    for entries in (plan['source_sha256'], plan['input_sha256']):
        for name, expected in entries.items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'non-project input')
            require(digest(root/name) == expected, 'frozen file changed: '+name)
    require(plan['wll_code_fingerprint'] == fingerprint({p.name: p.read_text() for p in
                                                       (root/'wll').glob('*.py')}), 'core code changed')
    return plan


def validate_memory(plan, probe, job, release):
    require(job['status'] == 'finished' and release.get('own_processes_released') is True,
            'memory job and its release must be verified')
    require(probe['plan_sha256'] == plan['memory']['plan_sha256'], 'memory protocol changed')
    require(plan['memory']['plan_sha256'] == NF4_MEMORY_SHA, 'NF4 memory evidence required')
    actual = probe.get('quantization_actual')
    require(isinstance(actual, dict) and actual.get('linear4bit_layers') == 248
            and actual.get('parameter_devices') == ['cuda:0'], 'unmeasured NF4 placement or layers')
    recipe = actual.get('quantization_config', {})
    require(recipe.get('load_in_4bit') is True and recipe.get('bnb_4bit_quant_type') == 'nf4'
            and recipe.get('bnb_4bit_compute_dtype') == 'bfloat16'
            and recipe.get('bnb_4bit_use_double_quant') is True, 'NF4 recipe changed')
    require(plan['expected_backend_static'].get('quantization') == quantization_metadata(actual),
            'task loader differs from the measured memory configuration')
    complete = probe['phase'] == 'complete' and job['exit_code'] == 0
    partial = (plan['memory'].get('allow_partial_oom') is True and probe['phase'] == 'failed'
               and probe.get('error_type') == 'OutOfMemoryError' and job['exit_code'] not in (None, 0))
    require(complete or partial, 'unaccepted memory-probe termination')
    cases = [c for c in probe['cases'] if c['input_tokens'] == plan['max_context']]
    require(len(cases) == 1 and cases[0]['produced_tokens'] == cases[0]['output_tokens'] == 32
            and 0 < cases[0]['peak_reserved_bytes'] <= 20*1024**3,
            'requested context lacks a completed bounded memory measurement')


def validate_health(health, plan, index):
    require(health.get('model') == f'{study_id(plan)}-b{index}' and health.get('ready') is True
            and health.get('busy') is False, 'unexpected model health')
    backend = health['backend']
    require({k: v for k, v in backend.items() if k != 'gpu_policy'} ==
            plan['expected_backend_static'], 'inference configuration changed')
    policy = backend['gpu_policy']
    allowed = plan['gpu_policy']['allowed_gpus']
    require(policy.get('hostname') == plan['gpu_policy']['hostname']
            and type(policy.get('physical_gpu_index')) is int
            and str(policy['physical_gpu_index']) in allowed
            and policy.get('gpu_uuid') == allowed[str(policy['physical_gpu_index'])]
            and policy.get('visible_device_count') == 1
            and policy.get('policy_sha256') == plan['gpu_policy_sha256'], 'unauthorized device backend')


def server_job(plan, index, owner, plan_sha):
    require(index in (0, 1, 2) and re.fullmatch('[a-f0-9]{32}', owner), 'invalid batch or owner')
    job_id = f'{study_id(plan)}-b{index}'
    argv = [PYTHON, '-B', '-m', MODULE, '--model-path', plan['model_path'], '--model-id', job_id,
            '--port', str(plan['port']), '--loader', 'multimodal', '--top-p', '0.8',
            '--top-k', '20', '--min-p', '0', '--presence-penalty', '1.5',
            '--max-context', str(plan['max_context']), '--max-output-tokens', '1024',
            '--quantization', 'nf4', '--nf4-plan-sha256', NF4_MEMORY_SHA]
    return {'id': job_id, 'owner': owner, 'argv': argv, 'max_runtime_seconds': plan['gpu_seconds'],
            'plan_sha256': plan_sha}


def expected_protocol(plan, tasks, health):
    return {'tasks': tasks, 'model_id': health['model'], 'max_tokens': plan['max_tokens'],
        'temperature': plan['temperature'], 'seed': plan['seed'], 'max_steps': plan['max_steps'],
        'rollout_seconds': plan['rollout_seconds'], 'prompt_version': plan['prompt_version'],
        'environment_id': environment_id(plan), 'expected_backend': health['backend'],
        'script_sha256': plan['source_sha256']['scripts/run_timed_tasks.py'],
        'wll_code_fingerprint': plan['wll_code_fingerprint'], 'backend_idle_check_enabled': True,
        'deadline_scope': 'rollout only; environment startup and cleanup separately bounded'}


def process_matches(info, argv):
    pid = info.get('process_group')
    proc = Path('/proc')/str(pid)
    try:
        command = (proc/'cmdline').read_bytes().split(b'\0')[:-1]
        return (pid and proc.stat().st_uid == os.getuid()
                and (proc/'stat').read_text().split()[21] == info.get('start_ticks')
                and os.getpgid(pid) == pid
                and command[-len(argv):] == [arg.encode() for arg in argv])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def model_pids(job):
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')[:-1]
            if (proc.stat().st_uid == os.getuid() and MODULE.encode() in args
                    and b'--model-id' in args and args[args.index(b'--model-id')+1] == job['id'].encode()):
                found.append(int(proc.name))
        except (OSError, IndexError):
            continue
    return sorted(found)


def runtime_state(job, art, known_pids=()):
    info_path = art/f"gpu-idle-job-{job['id']}.json"
    info = read(info_path) if info_path.exists() else None
    watch = read(art/'gpu-idle-watch-status.json') if (art/'gpu-idle-watch-status.json').exists() else None
    live = model_pids(job)
    monitor_live = False
    monitor_path = art/'gpu-idle-watch-process.json'
    if watch and monitor_path.exists():
        monitor = read(monitor_path)
        proc = Path('/proc')/str(monitor['pid'])/'stat'
        try:
            monitor_live = (watch['pid'] == monitor['pid'] and 0 <= time.time()-watch['checked_epoch'] < 35
                            and proc.read_text().split()[21] == monitor['start_ticks'])
        except (OSError, IndexError):
            pass
    result = {'job': info, 'watch': watch, 'epoch': time.time(), 'model_pids': live,
              'monitor_live': monitor_live, 'released': False}
    if info and info['status'] == 'finished' and not live and not process_matches(info, job['argv']):
        try:
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True, timeout=15)
            occupied = {int(line.strip()) for line in raw.splitlines() if line.strip()}
            result['released'] = not occupied.intersection(known_pids)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result['gpu_release_query_error'] = type(exc).__name__
    return result


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    require(operation in {'enqueue', 'pulse', 'stop', 'status'}, 'invalid queue operation')
    require(re.fullmatch('(?:'+ID+'|'+OBSERVATION_ID+r')-b[012]', job['id'])
            and re.fullmatch('[a-f0-9]{32}', job['owner']),
            'invalid job identity')
    require(job['max_runtime_seconds'] == 5400 and job['argv'][:4] == [PYTHON, '-B', '-m', MODULE],
            'unexpected queued command')
    require(job['argv'].count('--model-id') == 1
            and job['argv'].index('--model-id')+1 < len(job['argv'])
            and job['argv'][job['argv'].index('--model-id')+1] == job['id'],
            'queued model identity differs')
    art = Path(root)/'artifacts'
    queue, heartbeat = art/'gpu-idle-queue.json', art/f"gpu-idle-heartbeat-{job['id']}"
    result_path = art/f"gpu-idle-job-{job['id']}.json"
    with (art/'gpu-idle-queue-edit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = read(queue) if queue.exists() else None
        if operation == 'enqueue':
            if result_path.exists() or heartbeat.exists():
                raise FileExistsError('this batch was already attempted or has a heartbeat')
            if current:
                old_result = art/f"gpu-idle-job-{current['id']}.json"
                require(old_result.exists() and read(old_result)['status'] == 'finished'
                        and not (art/f"gpu-idle-heartbeat-{current['id']}").exists(),
                        'cannot replace an active or unconfirmed queue')
                archive = art/f"gpu-idle-queue-archived-{current['id']}.json"
                if archive.exists():
                    require(archive.read_bytes() == queue.read_bytes(), 'queue archive differs')
                else:
                    archive.write_bytes(queue.read_bytes())
            # Owner is embedded in the exact queue persisted by the monitor.
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
from scripts.run_foundation_pilot import queue_operation
d=json.load(sys.stdin)
print(json.dumps(queue_operation(d['operation'],d['job'],known_pids=d['known_pids'])))
''', {'operation': operation, 'job': job, 'known_pids': list(known_pids)})


def combine_batches(reports, releases):
    require(len(reports) == len(releases) == 3, 'three batches and releases required')
    require(all(r['scheduled'] == r['completed'] == 4 and r['valid_for_comparison'] is True
                for r in reports), 'all twelve outcomes must be known')
    require(all(r['released'] is True for r in releases), 'a model release is unverified')
    scenarios = [s for r in reports for s in r['successful_scenarios']]
    require(len(scenarios) == len(set(scenarios)), 'scenario repeated across batches')
    successes = sum(r['successes'] for r in reports)
    require(successes == len(scenarios), 'success/scenario count differs')
    return {'completed': 12, 'successes': successes, 'success_rate': successes/12,
        'successful_scenarios': sorted(scenarios), 'advance_to_recovery_study': successes >= 3,
        'observed_step_cost': {k: sum(r['observed_step_cost'][k] for r in reports)
                               for k in ('steps', 'prompt_tokens', 'completion_tokens')},
        'returned_response_cost': {k: sum(r['returned_response_cost'][k] for r in reports)
                                   for k in ('prompt_tokens', 'completion_tokens')},
        'scope': 'Origin engineering feasibility, not unseen-scene efficacy or RSI. Cost views overlap.'}


def preflight(root, plan_path):
    plan_sha = digest(plan_path)
    plan = verify_frozen_plan(root, plan_path, plan_sha)
    require(plan.get('state') == 'registered', 'registered study plan required')
    environment_id(plan)
    validate_observation_comparison(root, plan)
    fixed = {'model_path': MODEL_PATH, 'port': 19009, 'gpu_seconds': 5400,
        'client_seconds': 4500, 'model_start_seconds': 600, 'rollout_seconds': 600,
        'server_idle_seconds': 180, 'seed': 17, 'max_steps': 40, 'max_tokens': 1024,
        'temperature': .7, 'prompt_version': 'appworld-v3'}
    require(all(plan.get(k) == v for k, v in fixed.items()) and plan['max_context'] in (16384, 32768),
            'unregistered runtime settings')
    required_sources = {'scripts/run_foundation_pilot.py', 'scripts/run_timed_tasks.py',
        'scripts/report_timed_tasks.py', 'scripts/serve_transformers_presence.py',
        'scripts/generated_presence.py', 'scripts/gpu_idle_watch.py', 'config/gpu-policy.json',
        'scripts/nf4_loader.py', 'artifacts/probe_qwen35_nf4_001.py'}
    required_sources.update(str(p.relative_to(root)) for p in (root/'wll').glob('*.py'))
    require(required_sources <= plan['source_sha256'].keys(), 'source registration is incomplete')
    required_inputs = {'artifacts/foundation-inputs-001/tasks.jsonl', 'artifacts/partitions-v1/manifest.json',
                       'artifacts/qwen35-9b-nf4-feasibility-plan-001.json',
                       'artifacts/qwen35-nf4-runtime-audit-003.json',
                       *[plan['memory'][k] for k in ('probe', 'job', 'release_audit')]}
    require(required_inputs <= plan['input_sha256'].keys(), 'input registration is incomplete')
    require(plan['gpu_policy'] == read(root/'config/gpu-policy.json')
            and plan['gpu_policy_sha256'] == digest(root/'config/gpu-policy.json'), 'GPU policy changed')
    tasks = [json.loads(line) for line in (root/'artifacts/foundation-inputs-001/tasks.jsonl').read_text().splitlines()]
    ids = [t['task_id'] for t in tasks]
    partition = read(root/'artifacts/partitions-v1/manifest.json')['task_ids']['origin']
    first = sorted(min(t for t in partition if t.rsplit('_', 1)[0] == s)
                   for s in {t.rsplit('_', 1)[0] for t in partition})
    require(ids == first and len(ids) == 12 and plan['task_ids'] == ids, 'task selection changed')
    validate_memory(plan, *[read(root/plan['memory'][k]) for k in ('probe', 'job', 'release_audit')])
    require(not (root/'artifacts'/study_id(plan)).exists(), 'controller already has an attempt')
    require_space(root/'artifacts')
    actual_image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                           plan['image']], text=True, timeout=15).strip()
    require(actual_image == plan['image'], 'execution image changed')
    verify_remote(plan, 0)
    return plan, tasks, plan_sha


def verify_remote(plan, first_batch):
    require(first_batch in (0, 1, 2), 'invalid pending batch')
    return remote_python('''import hashlib,json,sys,subprocess,os
from pathlib import Path
root=Path('/home/hts/workflow-learning-lab');sys.path.insert(0,str(root));p=json.load(sys.stdin)
result=subprocess.run([p['python'],'-B','artifacts/probe_qwen35_nf4_001.py',
    '--check','--plan-sha256',p['memory_sha']],cwd=root,env=dict(os.environ,
    CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),capture_output=True,text=True,timeout=15,check=True)
assert json.loads(result.stdout)['preflight']=='passed'
for name,expected in p['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
model=Path(p['model_path'])
files={n:hashlib.sha256((model/n).read_bytes()).hexdigest() for n in ['config.json','tokenizer_config.json','generation_config.json','weight-source.json'] if (model/n).is_file()}
assert files==p['backend_files']
for i in range(p['first_batch'],3):assert not (root/'artifacts'/f"gpu-idle-job-{p['study_id']}-b{i}.json").exists()
print(json.dumps({'preflight':'passed'}))
''', {'sources': plan['source_sha256'], 'model_path': plan['model_path'],
       'backend_files': plan['expected_backend_static']['files'], 'first_batch': first_batch,
       'python': PYTHON, 'memory_sha': NF4_MEMORY_SHA, 'study_id': study_id(plan)})


def verify_cpu_environment(plan, tasks, output):
    """Check the actual descriptor container before occupying a GPU; no task actions."""
    started = time.time()
    world = AppWorldWorker(plan['data_dir'], plan['image'], plan['deps_dir'])
    state = {'phase': 'checking', 'started_epoch': started, 'uses_gpu': False,
             'container_name': world.name, 'docker_cli_pid': world.proc.pid,
             'task_ids': [t['task_id'] for t in tasks], 'model_generation_calls': 0}
    atomic_json(output, state)
    try:
        official = world.call('list', split='train')
        require(tasks and set(state['task_ids']) <= set(official), 'tasks absent from official train')
        live = [{'task_id': t['task_id'], 'scenario_id': t['task_id'].rsplit('_', 1)[0],
                 'split': 'train', **world.call('describe', task_id=t['task_id'])} for t in tasks]
        require(live == tasks, 'live task descriptions changed before GPU queue')
        state.update(phase='complete', task_descriptions_match=True)
    except BaseException as exc:
        state.update(phase='failed', error_type=type(exc).__name__, error=str(exc)[:1000])
        raise
    finally:
        try:
            world.close()
            remaining = subprocess.check_output(['docker', 'ps', '-aq', '--filter',
                'name=^/'+world.name+'$'], text=True, timeout=15)
            state['container_absent'] = not remaining.strip()
            require(state['container_absent'], 'CPU preflight container remains')
        except BaseException as exc:
            state.update(phase='failed', cleanup_error=type(exc).__name__)
            raise
        finally:
            state.update(finished_epoch=time.time(), seconds=time.time()-started)
            atomic_json(output, state)
    return state


def require_rollout_artifacts(stage, exit_code):
    if not (stage/'run/manifest.json').exists():
        log = stage/'client.log'
        atomic_json(stage/'startup-failure.json', {'client_exit_code': exit_code,
            'completed_tasks': 0, 'unknown_tasks': 4, 'success_rate': None,
            'client_log_sha256': digest(log) if log.exists() else None,
            'reason': 'worker exited before task rollout; retain original client traceback'})
        raise ValueError('worker exited before task rollout; inspect client.log and startup-failure.json')


def worker(root, plan, index, stage):
    identity = environment_id(plan)
    health = read(stage/'health.json')
    validate_health(health, plan, index)
    ids = plan['task_ids'][index*4:index*4+4]
    tasks = load_tasks(plan['data_dir'], 'train', image=plan['image'], deps_dir=plan['deps_dir'],
                       include_task_ids=ids)
    by_id = {t.task_id: t for t in tasks}
    tasks = [by_id[t] for t in ids]
    frozen = [json.loads(line) for line in (root/'artifacts/foundation-inputs-001/tasks.jsonl').read_text().splitlines()]
    require([asdict(t) for t in tasks] == frozen[index*4:index*4+4], 'live task descriptions changed')
    with (stage/'containers.jsonl').open('x') as resources, (stage/'feedback-events.jsonl').open('x') as feedback:
        def factory(task):
            world = AppWorldEnvironment(task.task_id, plan['data_dir'], plan['max_steps'],
                                        plan['image'], plan['deps_dir'])
            try:
                container = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}',
                    world.worker.name], text=True, timeout=15).strip()
                resources.write(json.dumps({'name': world.worker.name, 'id': container})+'\n')
                resources.flush()
                return observation_world(world, plan, task.task_id, feedback)
            except BaseException:
                world.close()
                raise

        endpoint = f"http://127.0.0.1:{plan['port']}/v1"
        result = run_tasks(tasks, factory, LocalModel(endpoint, health['model'],
            max_tokens=plan['max_tokens'], temperature=plan['temperature']), output=stage/'run',
            max_steps=plan['max_steps'], rollout_seconds=plan['rollout_seconds'], seed=plan['seed'],
            expected_backend=health['backend'], prompt_version=plan['prompt_version'],
            environment_id=identity, before_task=lambda task: wait_for_idle(endpoint,
                health['model'], health['backend'], seconds=plan['server_idle_seconds']))
    return 0 if result['valid_for_comparison'] else 1


def stop_client(client, stage):
    captured = set()
    # Capture Docker CLI names before a forced process-group shutdown loses them.
    if client is not None and client.poll() is None:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if proc.stat().st_uid != os.getuid() or os.getpgid(int(proc.name)) != client.pid:
                    continue
                args = (proc/'cmdline').read_bytes().decode().split('\0')
                if Path(args[0]).name == 'docker' and args[1] == 'run' and '--name' in args:
                    name = args[args.index('--name')+1]
                    if re.fullmatch(r'wll-[a-f0-9]{16}', name):
                        captured.add(name)
            except (OSError, IndexError, UnicodeError):
                continue
        try:
            os.killpg(client.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            client.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(client.pid, signal.SIGKILL)
            client.wait(timeout=10)
    resources = stage/'containers.jsonl'
    if resources.exists():
        for line in resources.read_text().splitlines():
            item = json.loads(line)
            require(re.fullmatch('[a-f0-9]{64}', item['id']), 'invalid recorded container ID')
            captured.add(item['id'])
    for identifier in sorted(captured):
        subprocess.run(['docker', 'rm', '-f', identifier], capture_output=True, timeout=15, check=False)
        remaining = subprocess.check_output(['docker', 'ps', '-aq', '--filter',
            'id='+identifier if len(identifier) == 64 else 'name=^/'+identifier+'$'],
            text=True, timeout=15)
        require(not remaining.strip(), 'owned Docker container remains')
    result = {'local_client_exited': client is None or client.poll() is not None,
              'recorded_containers_absent': True, 'checked_identifiers': sorted(captured)}
    atomic_json(stage/'local-cleanup.json', result)
    return result


def run_batch(plan, tasks, index, root, plan_path, *, plan_sha):
    verify_frozen_plan(root, plan_path, plan_sha, expected_plan=plan)
    remote_check = verify_remote(plan, index)
    stage = root/'artifacts'/study_id(plan)/f'batch-{index}'
    stage.mkdir()
    atomic_json(stage/'freeze-check.json', {'plan_sha256': plan_sha, 'checked_epoch': time.time(),
                'local_files_match': True, 'remote': remote_check})
    verify_cpu_environment(plan, tasks[index*4:index*4+4], stage/'cpu-environment-preflight.json')
    job = server_job(plan, index, uuid.uuid4().hex, plan_sha)
    atomic_json(stage/'job.json', job)
    state, known = {}, set()
    stopped = threading.Event()
    thread = tunnel = client = None
    report = None

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
    release = None
    try:
        state.update(remote_queue('enqueue', job), local_observed_monotonic=time.monotonic())
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        with (stage/'tunnel.log').open('x') as log:
            port = plan['port']
            tunnel = subprocess.Popen([*SSH[:-1], '-N', '-o', 'ExitOnForwardFailure=yes',
                '-L', f'{port}:127.0.0.1:{port}', '4090'], stdout=log, stderr=subprocess.STDOUT)
        loading = None
        while True:
            require(tunnel.poll() is None, 'SSH tunnel exited')
            require(state.get('owned') is True, 'queue ownership changed')
            require(state.get('monitor_live') is True, 'GPU monitor is not confirmed live')
            require(time.monotonic()-state['local_observed_monotonic'] <= 90, 'remote status stale')
            remote_job = state.get('job') or {}
            require(remote_job.get('status') != 'finished', 'server exited before health verification')
            if remote_job.get('status') == 'running':
                loading = loading or time.monotonic()
                require(time.monotonic()-loading <= plan['model_start_seconds'], 'model startup timed out')
                try:
                    with RolloutDeadline(6), opener.open(f'http://127.0.0.1:{port}/health', timeout=5) as response:
                        raw = response.read(65_537)
                    require(len(raw) <= 65_536, 'oversized health response')
                    health = json.loads(raw)
                    break
                except OSError:
                    pass
            time.sleep(2)
        validate_health(health, plan, index)
        confirmed = remote_queue('status', job, known)
        require(confirmed['owned'] and (confirmed.get('job') or {}).get('status') == 'running'
                and confirmed.get('model_pids'), 'live service process identity missing')
        known.update(confirmed['model_pids'])
        atomic_json(stage/'health.json', health)
        expected = expected_protocol(plan, tasks[index*4:index*4+4], health)
        atomic_json(stage/'expected.json', expected)
        with (stage/'client.log').open('x') as log:
            client = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.run_foundation_pilot',
                '--plan', str(plan_path), '--plan-sha256', plan_sha, '--worker',
                '--batch', str(index), '--stage', str(stage)],
                cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root),
                                  TMPDIR=plan['tmp_dir']), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            code = client.wait(timeout=plan['client_seconds'])
        require_rollout_artifacts(stage, code)
        report = audit_run(stage/'run', expected)
        atomic_json(stage/'report.json', report)
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
                deadline = time.monotonic()+90
                while release['owned'] and not release['released'] and time.monotonic() < deadline:
                    time.sleep(5)
                    release = remote_queue('status', job, known)
                atomic_json(stage/'release.json', release)
                require(release['owned'] and release['released'], 'model release unconfirmed; stop pipeline')
            finally:
                if tunnel is not None and tunnel.poll() is None:
                    tunnel.terminate()
                    tunnel.wait(timeout=10)
    return report, release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=PLAN)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--batch', type=int, choices=(0, 1, 2))
    parser.add_argument('--stage')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'controller/client must be CPU only')
    plan_path = Path(args.plan).resolve(strict=True)
    if args.worker:
        require(args.batch is not None and args.stage is not None, 'worker batch/stage required')
        plan = verify_frozen_plan(ROOT, plan_path, args.plan_sha256)
        raise SystemExit(worker(ROOT, plan, args.batch, Path(args.stage)))
    with (ROOT/'artifacts/foundation-pilot-001.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, tasks, plan_sha = preflight(ROOT, plan_path)
        if args.check:
            print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'batches': 3}))
            return
        output = ROOT/'artifacts'/study_id(plan)
        output.mkdir()
        state = {'phase': 'running', 'plan_sha256': plan_sha, 'pid': os.getpid(),
                 'started_epoch': time.time(), 'completed_batches': 0}
        atomic_json(output/'status.json', state)
        try:
            reports, releases = [], []
            for index in range(3):
                state.update(current_batch=index, checked_epoch=time.time())
                atomic_json(output/'status.json', state)
                report, release = run_batch(plan, tasks, index, ROOT, plan_path, plan_sha=plan_sha)
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
