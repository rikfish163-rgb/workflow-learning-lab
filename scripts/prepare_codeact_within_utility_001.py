"""Seal a completed CodeAct writer's entire reviewed bank for later utility.

This terminal-only CPU preparation never starts a writer, model, queue, world,
training or utility study. All writer histories, responses and rows stay private.
Both the writer plan and its terminal result must be named by their exact hashes.
"""
import argparse
import copy
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import FunctionType

from scripts import codeact_within_experience_001 as experience
from scripts import codeact_within_utility_registration_001 as utility
from scripts import codeact_within_writer_gpu_queue_001 as queue
from scripts import codeact_within_writer_registration_001 as writer
from scripts import native_source_episode as episode
from scripts import run_codeact_within_writer_001 as controller
from scripts import run_codeact_within_writer_cases_001 as cases
from scripts.native_appworld_worker_client import _decode, _encode, _same
from scripts.prepare_codeact_within_writer_001 import _save_new
from scripts.remote_runtime_storage import REMOTE_TMP_DIR, verify_storage
from wll.protocol import fingerprint

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path('/home/hetaisheng/.local/share/workflow-learning-lab')
SERVER_ROOT = Path('/home/data4/hts/workflow-learning-lab/runtime-remote-001/workspace')
SOURCE = 'scripts/prepare_codeact_within_utility_001.py'
ERROR = 'codeact_within_utility_preparation_invalid'
MAX_BYTES = 64_000_000


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _reader():
    scope = dict(episode.__dict__)
    scope['PRIVATE_RUN_ROOT'] = PRIVATE
    for name in ('_private_directory', '_read_private'):
        function = getattr(episode, name)
        scope[name] = FunctionType(function.__code__, scope, name,
                                   function.__defaults__, function.__closure__)
    return scope['_read_private']


def prepare_source_export(report_sha256):
    # Import only when preparing: the fault adapter loads its pinned human code.
    from scripts.codeact_fault_source_export_001 import prepare_source_export as export
    return export(report_sha256)


class _Replay:
    """Serve only previously recorded return values; no service is contacted."""
    def __init__(self, stage):
        self.stage, self.max_tokens, self.position = stage, stage['max_tokens'], 0
        self.operations = stage['operations']
        _require(type(self.operations) is list and 2 <= len(self.operations) <= 3)

    def _next(self, kind, messages=None, seed=None):
        _require(self.position < len(self.operations))
        op = self.operations[self.position]
        self.position += 1
        _require(type(op) is dict and set(op) == {'operation', 'returned', 'result', 'error'}
                 and op['operation'] == kind and op['returned'] is True and op['error'] is False)
        if messages is not None:
            _require(_same(messages, self.stage['messages']))
        if seed is not None:
            _require(type(seed) is int and seed == self.stage['seed'])
        return copy.deepcopy(op['result'])

    def check_health(self):
        return self._next('health')

    def count_input_tokens(self, messages):
        return self._next('count', messages)

    def generate(self, messages, seed):
        return self._next('generate', messages, seed)


def _verify_stage(context, stage, name, deadline, candidate=None, backend=None):
    index = context['lineage']['source_index']
    _require(type(stage) is dict and set(stage) == {
        'stage', 'max_tokens', 'seed', 'deadline', 'messages', 'result', 'operations',
        'events', 'status', 'reason', 'error', 'wall_seconds'}
        and stage['stage'] == name and type(stage['seed']) is int
        and stage['seed'] == (17 if name == 'writer' else 117) + index
        and type(stage['max_tokens']) is int and stage['max_tokens'] == (128 if name == 'writer' else 256)
        and _finite(stage['deadline']) and 0 < stage['deadline'] <= deadline
        and _finite(stage['wall_seconds']) and stage['wall_seconds'] <= 240
        and stage['error'] is None and cases._cost(stage)['known'] is True)
    expected_messages = (experience.build_writer_messages(context) if name == 'writer'
                         else experience.build_review_messages(context, candidate))
    _require(_same(stage['messages'], expected_messages))
    model, emitted = _Replay(stage), []
    result = (experience.write_once(context, model, seed=stage['seed'], record=emitted.append)
              if name == 'writer' else experience.review_once(context, candidate, model,
                                                            seed=stage['seed'], record=emitted.append))
    _require(model.position == len(model.operations) and _same(result, stage['result'])
             and _same(emitted, stage['events']) and stage['status'] == result['status']
             and stage['reason'] == result['reason'])
    if backend is not None:
        health = model.operations[0]['result']
        _require(type(health) is dict and health.get('model') == writer.MODEL_ID
                 and health.get('ready') is True and health.get('busy') is False
                 and type(health.get('max_context')) is int and health['max_context'] == 16384
                 and type(health.get('max_output_tokens')) is int
                 and stage['max_tokens'] <= health['max_output_tokens'] <= 1024
                 and _same(health.get('backend'), backend))
        if len(model.operations) == 3:
            response = model.operations[2]['result']
            raw = response.get('raw_reply') if type(response) is dict else None
            _require(type(raw) is dict and raw.get('model') == writer.MODEL_ID
                and raw.get('format_error') is None and response.get('format_error') is None
                and _same(raw.get('backend'), backend) and _same(response.get('backend'), backend)
                and type(raw.get('usage')) is dict and set(raw['usage']) == {'prompt_tokens', 'completion_tokens'}
                and all(type(raw['usage'][key]) is int for key in ('prompt_tokens', 'completion_tokens'))
                and type(response.get('prompt_tokens')) is int
                and response['prompt_tokens'] == model.operations[1]['result'] == raw['usage']['prompt_tokens']
                and type(response.get('completion_tokens')) is int
                and 1 <= response['completion_tokens'] <= stage['max_tokens']
                and raw['usage']['completion_tokens'] == response['completion_tokens']
                and type(raw.get('choices')) is list and len(raw['choices']) == 1
                and type(raw['choices'][0]) is dict
                and raw['choices'][0].get('finish_reason') in {'stop', 'length', 'unknown'}
                and type(response.get('text')) is str
                and raw['choices'][0].get('message') == {'role': 'assistant', 'content': response['text']})
    start = dict(event='stage_start', source_index=index, stage=name, seed=stage['seed'],
                 max_tokens=stage['max_tokens'], deadline=stage['deadline'], messages=stage['messages'])
    return [start] + [dict(event='stage_event', source_index=index, stage=name, payload=event)
                      for event in emitted]


def _verify_cases(result, prepared, registered, journal_events, deadline, backend=None):
    _require(_finite(deadline) and deadline > 0)
    _require(cases.verify_public_projection(result, registered) is True)
    private, public = result['private'], result['public']
    _require(_same(private['source_private'], prepared) and private['completed'] is True
             and private['journal_failed'] is False and public['status'] == 'complete'
             and public['whole_cost_known'] is True and public['canfreeze'] is False
             and public['accepted'] is public['training_authorized'] is False
             and _finite(private['wall_seconds']))
    expected = [dict(event='cases_start', preparation_manifest_sha256=registered['preparation_manifest_sha256'])]
    for item, context in zip(private['receipts'], prepared['contexts'], strict=True):
        index = context['lineage']['source_index']
        _require(type(item) is dict and set(item) == {'source_index', 'context', 'status', 'reason',
                                                       'writer', 'screen', 'review'}
                 and type(item['source_index']) is int and item['source_index'] == index
                 and _same(item['context'], context))
        expected.append(dict(event='case_start', source_index=index, context=context))
        expected.extend(_verify_stage(context, item['writer'], 'writer', deadline, backend=backend))
        result_writer = item['writer']['result']
        if result_writer['status'] == 'abstained':
            _require(item['screen'] is item['review'] is None and item['status'] == 'abstained'
                     and item['reason'] == result_writer['reason'])
        else:
            _require(result_writer['status'] == 'candidate_unvalidated')
            candidate = result_writer['candidate']
            screen = experience.screen(context, candidate)
            _require(_same(item['screen'], screen))
            if screen['status'] == 'rejected':
                _require(item['review'] is None and item['status'] == 'screen_rejected'
                         and item['reason'] == 'deterministic_screen_rejected')
            else:
                _require(screen['status'] == 'passed')
                expected.extend(_verify_stage(context, item['review'], 'review', deadline, candidate, backend))
                review = item['review']['result']
                _require(item['status'] == review['status'] and item['reason'] == review['reason'])
        expected.append(dict(event='case_stop', source_index=index, receipt=item))
    expected.append(dict(event='cases_stop', completed=True))
    _require(type(private['events']) is list
             and all(type(row) is dict and set(row) == {'payload', 'recorded'} and row['recorded'] is True
                     for row in private['events'])
             and _same(expected, [row['payload'] for row in private['events']])
             and _same(expected, journal_events))
    rows = private['pending_bundle_rows']
    _require(type(rows) is list and 1 <= len(rows) <= 12 and public['pending_count'] == len(rows))
    utility.render_bundle(rows)
    return copy.deepcopy(rows)


def _verify_release(release, public, job):
    _require(type(release) is dict and set(release) == {'verified', 'GPU_seconds', 'service_wall_seconds',
        'cuda_seconds', 'runtime_overrun', 'state', 'release_wall_seconds'}
        and release['verified'] is True and release['runtime_overrun'] is False
        and release['cuda_seconds'] is None and _finite(release['GPU_seconds'])
        and release['GPU_seconds'] <= 7200 and _finite(release['release_wall_seconds'])
        and release['release_wall_seconds'] <= 180
        and _same(public, dict(verified=True, GPU_seconds=release['GPU_seconds'], cuda_seconds=None)))
    state = release['state']
    _require(type(state) is dict and state.get('owned') is True and state.get('admission_cancelled') is True
             and state.get('monitor_live') is True and state.get('released') is True
             and state.get('model_pids') == [] and _finite(state.get('epoch')))
    info = state.get('job')
    _require(type(info) is dict and info.get('status') == 'finished' and info.get('job_id') == job['id']
             and writer._sha(info.get('queue_sha256')) and _finite(info.get('started_epoch'))
             and _finite(info.get('finished_epoch'))
             and 0 <= info['started_epoch'] <= info['finished_epoch'] <= state['epoch']
             and release['GPU_seconds'] == info['finished_epoch'] - info['started_epoch']
             and release['service_wall_seconds'] == release['GPU_seconds'])
    return True


def _verify_launch_command(actual, expected):
    # The launcher inherits only a small allowlist of host/session variables;
    # their desktop values cannot be substituted for the server's values.
    _require(type(actual) is list and all(type(value) is str for value in actual))
    marker = expected.index(writer.PYTHON)
    _require(actual.count(writer.PYTHON) == 1)
    actual_marker = actual.index(writer.PYTHON)
    prefix = expected.index('--') + 3  # -- /usr/bin/env -i
    _require(actual[:prefix] == expected[:prefix] and actual[actual_marker:] == expected[marker:])
    def env(values):
        _require(all('=' in value for value in values))
        pairs = [value.split('=', 1) for value in values]
        keys = [pair[0] for pair in pairs]
        _require(keys == sorted(set(keys)))
        return dict(pairs)
    wanted, seen = env(expected[prefix:marker]), env(actual[prefix:actual_marker])
    inherited = {'PATH', 'HOME', 'USER', 'LOGNAME', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS'}
    _require(set(seen) - inherited == set(wanted) - inherited
             and all(seen[key] == value for key, value in wanted.items() if key not in inherited))


def _verify_journal(plan, sha, intent, launch, journal, report, result, pending, release):
    _require(type(intent) is dict and set(intent) == {'id', 'plan_sha256', 'sources', 'automatic_retry',
        'pid', 'start_ticks', 'boot_id', 'private_input_sha256'} and intent['id'] == writer.ID
        and intent['plan_sha256'] == sha and intent['automatic_retry'] is False
        and type(intent['pid']) is int and intent['pid'] > 0
        and type(intent['start_ticks']) is str and intent['start_ticks'].isdigit()
        and type(intent['boot_id']) is str and re.fullmatch('[0-9a-f-]{36}', intent['boot_id'])
        and intent['private_input_sha256'] == plan['private_input_sha256'])
    _require(type(launch) is dict and set(launch) == {'unit', 'argv', 'plan_sha256'}
        and launch['plan_sha256'] == sha and type(launch['unit']) is str
        and re.fullmatch('wll-native-source-codeact-writer-[0-9a-f]{32}\\.service', launch['unit']))
    expected_command = controller.unit_command(launch['unit'], ['--run', '--plan-sha256', sha],
        root=SERVER_ROOT, log_path=Path(writer.PRIVATE_RUN_ROOT) / 'writer-parent.log')
    _verify_launch_command(launch['argv'], expected_command)
    fields = {'id', 'plan_sha256', 'status', 'automatic_retry', 'case_public', 'source_indices',
        'case_result_sha256', 'pending_bundle_sha256', 'GPU_release', 'GPU_release_sha256',
        'preflight_wall_seconds', 'parent_wall_seconds', 'whole_cost_known', 'canfreeze',
        'accepted', 'content_certified', 'training_authorized', 'training_jobs', 'utility_or_dev_or_test_calls'}
    _require(type(report) is dict and set(report) == fields and report['id'] == writer.ID
        and report['plan_sha256'] == sha and report['status'] == 'complete'
        and report['automatic_retry'] is False and report['whole_cost_known'] is True
        and report['canfreeze'] is report['accepted'] is report['content_certified']
            is report['training_authorized'] is False
        and type(report['training_jobs']) is int and report['training_jobs'] == 0
        and type(report['utility_or_dev_or_test_calls']) is int and report['utility_or_dev_or_test_calls'] == 0
        and _finite(report['preflight_wall_seconds']) and _finite(report['parent_wall_seconds'])
        and report['preflight_wall_seconds'] <= report['parent_wall_seconds'] <= controller.PARENT_SECONDS
        and _same(report['case_public'], result['public'])
        and _same(report['source_indices'], intent['sources']))
    _require(type(journal) is list and journal and all(type(row) is dict and type(row.get('event')) is str
        and row['event'] in {'inputs_bound', 'GPU_enqueue_intent', 'GPU_ready', 'GPU_heartbeat',
                             'case_event', 'release_fence_started', 'study_stop'} for row in journal))
    def one(name):
        found = [(i, row) for i, row in enumerate(journal) if row['event'] == name]
        _require(len(found) == 1)
        return found[0]
    bound_i, bound = one('inputs_bound')
    enqueue_i, enqueue = one('GPU_enqueue_intent')
    ready_i, ready = one('GPU_ready')
    fence_i, fence = one('release_fence_started')
    stop_i, stop = one('study_stop')
    _require(bound_i == 0 and stop_i == len(journal) - 1
        and bound_i < enqueue_i < ready_i < fence_i < stop_i
        and _same(bound, dict(event='inputs_bound', private_input_sha256=plan['private_input_sha256'],
                              registered_inputs=plan['registered_inputs']))
        and set(enqueue) == {'event', 'job'} and set(ready) == {'event', 'job', 'service', 'client_deadline'}
        and _same(enqueue['job'], ready['job']) and _same(stop, dict(event='study_stop', report=report)))
    job = enqueue['job']
    queue._validate_job(job)
    _require(job['plan_sha256'] == sha and set(fence) == {'event', 'job_sha256', 'monitor_pid',
        'monitor_start_ticks', 'boot_id', 'cancel_epoch'} and fence['job_sha256'] == fingerprint(job)
        and type(fence['monitor_pid']) is int and fence['monitor_pid'] > 0
        and type(fence['monitor_start_ticks']) is str and fence['monitor_start_ticks'].isdigit()
        and fence['boot_id'] == intent['boot_id'] and _finite(fence['cancel_epoch']))
    service = ready['service']
    _require(type(service) is dict and set(service) == {'backend', 'identity'}
             and type(service['identity']) in (list, tuple) and len(service['identity']) == 5)
    # Use the exact backend policy verifier bound by the writer registration.
    scope = dict(episode._backend.__globals__)
    from scripts import native_source_registration_006
    scope['registration'] = native_source_registration_006
    check_backend = FunctionType(episode._backend.__code__, scope, '_backend')
    check_backend(plan, service['backend'])
    _require(_verify_release(release, report['GPU_release'], job) is True
        and release['state']['epoch'] >= fence['cancel_epoch'])
    identity, info, watch = service['identity'], release['state']['job'], release['state'].get('watch')
    _require(type(identity[0]) is int and identity[0] > 0
             and type(identity[1]) is str and identity[1].isdigit()
             and type(identity[2]) is int and identity[2] > 0
             and type(identity[3]) is str and identity[3].isdigit() and writer._sha(identity[4])
             and info.get('process_group') == identity[2] and info.get('start_ticks') == identity[3]
             and info.get('gpu_index') == service['backend']['gpu_policy']['physical_gpu_index']
             and type(watch) is dict and watch.get('pid') == fence['monitor_pid']
             and _finite(watch.get('checked_epoch'))
             and 0 <= release['state']['epoch'] - watch['checked_epoch'] < 35)
    indexed = [(i, row) for i, row in enumerate(journal) if row['event'] == 'case_event']
    _require(bool(indexed) and all(ready_i < i < fence_i and set(row) == {'event', 'payload'}
                                  for i, row in indexed)
        and all(enqueue_i < i < fence_i for i, row in enumerate(journal) if row['event'] == 'GPU_heartbeat'))
    prepared = result['private']['source_private']
    _require(_same(intent['sources'], [ctx['lineage']['source_index'] for ctx in prepared['contexts']]))
    rows = _verify_cases(result, prepared, plan['registered_inputs'],
                         [row['payload'] for _, row in indexed], ready['client_deadline'], service['backend'])
    _require(_same(rows, pending))
    return rows, job, fence, service


# Runs only fixed read-only inspection on 4090. A missing or garbage-collected
# transient unit is intentionally unknown, not evidence of a successful exit.
_TERMINAL_CODE = r'''
import json, os, pathlib, socket, subprocess, sys
unit, pid, ticks, boot = sys.argv[1:]
props = ('MainPID','ExecMainPID','ActiveState','SubState','LoadState','Result',
         'ExecMainStatus','ExecMainCode','Description','MemoryMax','CPUQuotaPerSecUSec',
         'TasksMax','NoNewPrivileges','KillMode','RestrictAddressFamilies','RuntimeMaxUSec','ControlGroup')
env = dict(os.environ, XDG_RUNTIME_DIR='/run/user/'+str(os.getuid()),
           DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/'+str(os.getuid())+'/bus')
res = subprocess.run(['/usr/bin/systemctl','--user','show',unit]+['--property='+x for x in props],
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, timeout=5, check=True)
if len(res.stdout)>32768: raise ValueError('oversize')
values = dict(line.split('=',1) for line in res.stdout.decode().splitlines() if '=' in line)
proc = pathlib.Path('/proc')/pid/'stat'
try: current_ticks = proc.read_text().rsplit(')',1)[1].split()[19]
except FileNotFoundError: current_ticks = None
group = values.get('ControlGroup','')
if group and (not group.startswith('/') or '..' in pathlib.Path(group).parts): raise ValueError('group')
cg = pathlib.Path('/sys/fs/cgroup')/group.lstrip('/')/'cgroup.procs' if group else None
try: populated = bool(cg.read_text().strip()) if cg else False
except FileNotFoundError: populated = False
print(json.dumps({'hostname':socket.gethostname(), 'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  'same_process_alive':current_ticks == ticks, 'cgroup_populated':populated,'properties':values}))
'''


def _terminal(launch, intent):
    args = [launch['unit'], str(intent['pid']), intent['start_ticks'], intent['boot_id']]
    command = ' '.join(shlex.quote(part) for part in ['/usr/bin/python3', '-c', _TERMINAL_CODE, *args])
    process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '4090', command],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15, check=True)
    _require(len(process.stdout) <= 32768)
    value = _decode(process.stdout)
    props = value['properties']
    from scripts.run_native_source_pilot import _duration_seconds
    _require(value['hostname'] == 'amax' and value['boot_id'] == intent['boot_id']
        and value['same_process_alive'] is False and value['cgroup_populated'] is False
        and props['LoadState'] == 'loaded' and props['ActiveState'] == 'active' and props['SubState'] == 'exited'
        and props['MainPID'] == '0' and props['ExecMainPID'] == str(intent['pid'])
        and props['Result'] == 'success' and props['ExecMainStatus'] == '0' and props['ExecMainCode'] == '1'
        and props['Description'] == 'wll-native-source:' + launch['unit']
        and props['MemoryMax'] == str(6 * 1024**3) and _duration_seconds(props['CPUQuotaPerSecUSec']) == 4
        and props['TasksMax'] == '256' and props['NoNewPrivileges'] == 'yes' and props['KillMode'] == 'control-group'
        and set(props['RestrictAddressFamilies'].split()) == {'AF_UNIX', 'AF_INET', 'AF_INET6'}
        and _duration_seconds(props['RuntimeMaxUSec']) == controller.PARENT_SECONDS)
    return dict(unit=launch['unit'], pid=intent['pid'], start_ticks=intent['start_ticks'],
                boot_id=intent['boot_id'], terminal_success=True)


_QUEUE_CODE = r'''
import hashlib, json, os, pathlib, stat, sys
root=pathlib.Path('/home/hts/workflow-learning-lab/artifacts')
job_id=sys.argv[1]
def pairs(items):
    value={}
    for key,item in items:
        if key in value: raise ValueError('duplicate')
        value[key]=item
    return value
def invalid(_): raise ValueError('nonfinite')
def read(name):
    fd=os.open(root/name, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>1000000: raise ValueError('file')
        raw=b''
        while len(raw)<=1000000:
            chunk=os.read(fd,65536)
            if not chunk: break
            raw+=chunk
        after=os.fstat(fd)
        current=os.stat(root/name, follow_symlinks=False)
        def stamp(value): return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        if len(raw)!=before.st_size or stamp(before)!=stamp(after) or stamp(after)!=stamp(current): raise ValueError('changed')
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid), hashlib.sha256(raw).hexdigest()
    finally: os.close(fd)
result,rsha=read('gpu-idle-job-'+job_id+'.json')
cancel,csha=read('completion-admission-cancelled-'+job_id+'.json')
ready,ready_sha=read('completion-service-ready-'+job_id+'.json')
current,qsha=read('gpu-idle-queue.json')
if current.get('id')!=job_id:
    current,qsha=read('gpu-idle-queue-archived-'+job_id+'.json')
    archived=True
else: archived=False
try: os.lstat(root/('gpu-idle-heartbeat-'+job_id)); heartbeat=True
except FileNotFoundError: heartbeat=False
print(json.dumps({'result':result,'result_sha256':rsha,'cancel':cancel,'cancel_sha256':csha,
                  'ready':ready,'ready_sha256':ready_sha,'queue':current,'queue_sha256':qsha,
                  'archived':archived,'heartbeat':heartbeat}))
'''


def _queue_evidence(job, release, fence, service):
    queue._validate_job(job)
    command = ' '.join(shlex.quote(part) for part in ['/usr/bin/python3', '-c', _QUEUE_CODE, job['id']])
    process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '4090', command],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15, check=True)
    _require(len(process.stdout) <= 4_000_000)
    value = _decode(process.stdout)
    _require(type(value) is dict and set(value) == {'result', 'result_sha256', 'cancel', 'cancel_sha256',
        'ready', 'ready_sha256', 'queue', 'queue_sha256', 'archived', 'heartbeat'} and value['heartbeat'] is False
        and type(value['archived']) is bool and _same(value['queue'], job)
        and _same(value['result'], release['state']['job'])
        and all(writer._sha(value[key]) for key in ('result_sha256', 'cancel_sha256', 'queue_sha256', 'ready_sha256'))
        and (value['archived'] is True or value['queue_sha256'] == value['result']['queue_sha256']))
    cancel = value['cancel']
    _require(type(cancel) is dict and cancel.get('reason') == 'controller_stop'
             and cancel.get('job_sha256') == fingerprint(job) and _finite(cancel.get('epoch'))
             and cancel['epoch'] <= fence['cancel_epoch'])
    ready, info = value['ready'], value['result']
    _require(type(ready) is dict and set(ready) == {'job_sha256', 'ready_epoch', 'ready_monotonic',
        'backend_sha256', 'pid'} and ready['job_sha256'] == fingerprint(job)
        and ready['backend_sha256'] == fingerprint(service['backend'])
        and type(ready['pid']) is int and ready['pid'] == service['identity'][0]
        and fingerprint(ready) == service['identity'][4] and _finite(ready['ready_epoch'])
        and info['started_epoch'] <= ready['ready_epoch'] <= info['finished_epoch']
        and _finite(ready['ready_monotonic']) and ready['ready_monotonic'] > 0)
    return dict(result_sha256=value['result_sha256'], cancellation_sha256=value['cancel_sha256'],
                service_ready_sha256=value['ready_sha256'], job_sha256=_hash(_encode(job)),
                queue_archived=value['archived'])


def prepare(plan_sha256, result_sha256):
    """Authenticate immutable evidence before making either exclusive output."""
    try:
        return _prepare(plan_sha256, result_sha256)
    except Exception:
        raise ValueError(ERROR) from None


def _prepare(plan_sha256, result_sha256):
    _require(writer._sha(plan_sha256) and writer._sha(result_sha256))
    storage = verify_storage(ROOT, REMOTE_TMP_DIR)
    sealer_sha = _hash(writer.public_read(ROOT, SOURCE))
    plan = writer.load_registration(ROOT, plan_sha256, private=False)
    _require(plan['private_run_root'] == writer.PRIVATE_RUN_ROOT and plan['private_input'] == writer.PRIVATE_INPUT)
    reader, snapshots = _reader(), []
    def read(path, expected=None):
        raw = reader(path, maximum=MAX_BYTES)
        digest = _hash(raw)
        _require(expected is None or digest == expected)
        snapshots.append((path, digest))
        return raw
    run = PRIVATE / 'codeact-within-writer-001'
    report_raw = read(run / 'study-result.json', result_sha256)
    report = _decode(report_raw)
    # Refuse incomplete/unknown reports before opening any raw case payload.
    _require(type(report) is dict and report.get('status') == 'complete'
             and report.get('whole_cost_known') is True and report.get('plan_sha256') == plan_sha256
             and all(writer._sha(report.get(key)) for key in
                     ('case_result_sha256', 'pending_bundle_sha256', 'GPU_release_sha256')))
    intent_raw = read(run / 'study-intent.json')
    launch_raw = read(run / 'parent-launch-intent.json')
    intent, launch = _decode(intent_raw), _decode(launch_raw)
    # Validate the launch name and primitive PID fields before any remote invocation.
    _require(type(launch) is dict and type(launch.get('unit')) is str
        and re.fullmatch('wll-native-source-codeact-writer-[0-9a-f]{32}\\.service', launch['unit'])
        and type(intent) is dict and type(intent.get('pid')) is int and intent['pid'] > 0
        and type(intent.get('start_ticks')) is str and intent['start_ticks'].isdigit()
        and type(intent.get('boot_id')) is str and re.fullmatch('[0-9a-f-]{36}', intent['boot_id']))
    terminal = _terminal(launch, intent)
    journal_raw = read(run / 'study-journal.jsonl')
    _require(journal_raw and journal_raw.endswith(b'\n'))
    journal = [_decode(line) for line in journal_raw.splitlines()]
    case = _decode(read(run / 'writer-case-result.json', report['case_result_sha256']))
    pending = _decode(read(run / 'pending-bundle.json', report['pending_bundle_sha256']))
    release = _decode(read(run / 'GPU-release.json', report['GPU_release_sha256']))
    prepared = _decode(read(PRIVATE / 'codeact-within-writer-preparation-001/source-private.json',
                            plan['private_input_sha256']))
    rows, job, fence, service = _verify_journal(plan, plan_sha256, intent, launch, journal, report, case, pending, release)
    queue_proof = _queue_evidence(job, release, fence, service)
    _require(_same(prepared, case['private']['source_private']))
    exported = prepare_source_export(plan['source_report_sha256'])
    _require(_same(exported['private'], prepared))
    exported_public = exported['public']
    auth = exported_public['source_authentication']
    _require(exported_public['id'] == 'codeact-fault-source-export-001'
        and exported_public['status'] == 'authenticated_writer_preparation_only'
        and auth['study_id'] == writer.SOURCE_ID and auth['plan'] == writer.SOURCE_PLAN
        and auth['plan_sha256'] == writer.SOURCE_PLAN_SHA == utility.SOURCE_PLAN_SHA and auth['qualified'] is True
        and auth['report_sha256'] == plan['source_report_sha256']
        and auth['all_costs_known'] is True and type(auth['episode_count']) is int and auth['episode_count'] == 12
        and auth['origin_ids'] == list(utility.SOURCE_SCENARIOS))
    for path, digest in snapshots:
        _require(_hash(reader(path, maximum=MAX_BYTES)) == digest)
    _require(_hash(writer.public_read(ROOT, SOURCE)) == sealer_sha)
    _require(_same(writer.load_registration(ROOT, plan_sha256, private=False), plan))
    _require(_same(_terminal(launch, intent), terminal))
    _require(_same(_queue_evidence(job, release, fence, service), queue_proof))
    storage = verify_storage(ROOT, REMOTE_TMP_DIR)
    text = utility.render_bundle(rows)
    text_sha = _hash(text.encode('utf-8'))
    writer_auth = dict(plan_sha256=plan_sha256, result_sha256=result_sha256,
        parent_launch_sha256=_hash(launch_raw), study_intent_sha256=_hash(intent_raw),
        journal_sha256=_hash(journal_raw), case_result_sha256=report['case_result_sha256'],
        pending_bundle_sha256=report['pending_bundle_sha256'], release_sha256=report['GPU_release_sha256'],
        terminal=terminal, queue_release=queue_proof, whole_cost_known=True)
    bank = dict(status='plausible_pending_utility', accepted=False, training_authorized=False,
        insertion_text=text, source_scenario_ids=list(utility.SOURCE_SCENARIOS), rows=rows,
        source_authentication=copy.deepcopy(auth), writer_authentication=writer_auth)
    directory = PRIVATE / 'codeact-within-utility-preparation-001'
    proof_path = ROOT / utility.PREPARATION_PROOF
    _require(not directory.exists() and not directory.is_symlink()
             and not proof_path.exists() and not proof_path.is_symlink())
    directory.mkdir(mode=0o700)
    bank_sha = _save_new(directory / 'bank.json', bank)
    proof = dict(id='codeact-within-utility-inputs-CPU-001',
        status='authenticated_frozen_bank_before_utility', source_all_twelve_authenticated=True,
        writer_all_costs_known=True, writer_release_verified=True, writer_plan_sha256=plan_sha256,
        writer_result_sha256=result_sha256, sealer_source_sha256=sealer_sha,
        experience_input_sha256=bank_sha, experience_sha256=text_sha,
        experience_bundle_id='codeact-experience-001:' + text_sha, pending_count=len(rows),
        utility_calls=0, training_authorized=False, storage=storage)
    _save_new(proof_path, proof)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--result-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        proof = prepare(args.plan_sha256, args.result_sha256)
        print(json.dumps({key: proof[key] for key in ('status', 'pending_count', 'writer_plan_sha256',
                                                     'experience_input_sha256', 'experience_sha256')}))
        return 0
    except Exception:
        sys.stderr.write(ERROR + '\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
