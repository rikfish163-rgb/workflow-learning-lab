"""One-shot private train metadata preparation; no world, actor or evaluation.

The frozen ID-only selector determines all twelve tasks before this entrypoint
opens a CPU-only, read-only metadata worker. A failed attempt is retained. This
does not register or launch the recovery experiment.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from artifacts import prepare_b3_dev_metadata_001 as base
from scripts import owned_appworld_v3 as owned
from scripts import prepare_origin_recovery_pilot as selection
from scripts import run_timed_tasks_v3 as runner
from wll.protocol import Task, fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'artifacts/origin-recovery-inputs-001'
IMAGE = 'sha256:2f5a87274ea53b165a7f1ec656d2f6534aa0fbdd6479f4a77efbd89e39ef7cb2'
DATA = selection.OFFICIAL_TRAIN.parent.parent
DEPS = DATA.parent/'deps'
BUILD = 'artifacts/appworld-v2-image-build-001.json'
DESIGN = 'artifacts/qwen38-origin-recovery-design-001.json'
BINDINGS = {
    DESIGN: '5563a02031871d1ac34a50e3b2c685485b3d1e4e45babc269021c814ac35e46e',
    'scripts/prepare_origin_recovery_pilot.py': '77dda941fbb30f211abcf44b50bf85d569e5992df7487e22fc3b2e05065b712d',
    'artifacts/prepare_b3_dev_metadata_001.py': 'c64390b203722a402428c8d2c291374f1515dd2bf842e5e5804091b1439268be',
    'scripts/owned_appworld_v3.py': 'ff45079593b2ef91f5e17cf19f21e8d3835e5268e9f49769fb850c3de1bd6335',
    'scripts/run_timed_tasks_v3.py': '01e8d3c51553b316e04793d1552c03c854ba1f8fe74df471ecfd2d432e6598d2',
    BUILD: 'e3dffe634df548fb34fcc3bd4c7558c657b029ae7a587902c8d60cd36b7aea17',
    str(DATA/'version.txt'): '911fc0c48cb0c70601db5775a9bef1b740dc4cc9f9b46389b9f0563fe7eb94d7',
    str(DEPS/'appworld/environment.py'): 'f988f1e6f257041e73682725522662674d4d65e8a1673787614f54b89fe2baeb',
    str(DEPS/'appworld/task.py'): 'f999c02316639eae33316ad5bac52c58c217a80ee553f38b3084eb5ea18c5d7e',
}


def cpu_only():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA_VISIBLE_DEVICES must be explicitly empty')


def preflight():
    cpu_only()
    prepared = selection.prepare()
    for name, digest in BINDINGS.items():
        selection._read_frozen(ROOT/name, digest)
    design = json.loads((ROOT/DESIGN).read_text())
    if (design['selection']['task_ids'] != prepared['selected_ids']
            or design['protocol']['image'] != IMAGE
            or design['protocol']['environment_id'] != runner.environment_identity(IMAGE)
            or design['status'] != 'design_frozen_before_new_train_metadata_preparation'):
        raise ValueError('frozen origin metadata design differs')
    build = json.loads((ROOT/BUILD).read_text())
    if build['returncode'] != 0 or build['GPU_work'] is not False or build['image_id'] != IMAGE:
        raise ValueError('seeded image build acceptance differs')
    bindings = {**BINDINGS, str(Path(__file__).relative_to(ROOT)): base.sha256(__file__)}
    for key, path in selection.SOURCE_PATHS.items():
        bindings[str(path)] = selection.FROZEN_SHA256[key]
    for key, digest in build.get('source_sha256', {}).items():
        selection._read_frozen(ROOT/key, digest)
        bindings[key] = digest
    layout = {'data_dir': str(DATA.resolve(strict=True)), 'deps_dir': str(DEPS.resolve(strict=True)),
              'protocol': {'environment_image_id': IMAGE}}
    return prepared, layout, bindings, build


def official_ids():
    return selection._read_frozen(selection.OFFICIAL_TRAIN,
        selection.FROZEN_SHA256['official_train']).decode().splitlines()


def verify_image(image, build):
    result = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True,
                            text=True, timeout=10, check=True)
    info = json.loads(result.stdout)[0]
    if info['Id'] != image or any(info['Config'][key] != value
                                  for key, value in build['image_config'].items()):
        raise ValueError('local image differs from frozen CPU worker')


def metadata_call(worker, op, events, selected_ids, **kwargs):
    if not ((op == 'list' and kwargs == {'split': 'train'}) or (op == 'describe'
            and set(kwargs) == {'task_id'} and type(kwargs['task_id']) is str
            and kwargs['task_id'] in selected_ids)):
        raise ValueError('only list(train) and fixed twelve descriptions are authorized')
    base.append(events, {'event': 'metadata_call_started', 'op': op, **kwargs, 'epoch': time.time()})
    started = time.monotonic()
    result = worker.call(op, **kwargs)
    base.append(events, {'event': 'metadata_call_returned', 'op': op, **kwargs,
                        'seconds': time.monotonic()-started, 'epoch': time.time()})
    return result


def task_record(task_id, info):
    task = base.task_record(task_id, info)
    return Task(task.task_id, task.scenario_id, 'train', task.instruction, task.supervisor)


def validate_manifest(value, prepared, layout, bindings):
    image = layout['protocol']['environment_image_id']
    expected = {'split': 'train', 'image': image, 'data_dir': layout['data_dir'],
        'deps_dir': layout['deps_dir'], 'load_ground_truth': False,
        'selection_sha256': fingerprint(prepared),
        'official_train_sha256': prepared['source_sha256']['official_train'],
        'seed_protocol': runner.SEED_PROTOCOL, 'environment_seed': 100,
        'observation_feedback': runner.feedback_recipe(),
        'environment_id': runner.environment_identity(image), 'source_and_evidence_sha256': bindings}
    if (type(value) is not dict or any(value.get(k) != v for k, v in expected.items())
            or type(value.get('environment_seed')) is not int
            or value.get('load_ground_truth') is not False):
        raise ValueError('metadata provenance differs')
    rows = value.get('tasks')
    if type(rows) is not list or len(rows) != 12:
        raise ValueError('exact twelve train task records required')
    tasks = []
    for task_id, row in zip(prepared['selected_ids'], rows, strict=True):
        if (type(row) is not dict or set(row) != {'task_id', 'scenario_id', 'split',
                'instruction', 'supervisor'} or row['task_id'] != task_id
                or row['scenario_id'] != task_id.rsplit('_', 1)[0] or row['split'] != 'train'):
            raise ValueError('metadata identity/order/split differs')
        tasks.append(task_record(task_id, {k: row[k] for k in ('instruction', 'supervisor')}))
    return tasks


def load_prepared(path, expected_sha256):
    prepared, layout, bindings, _ = preflight()
    path = Path(path)
    value = base.bound(path, expected_sha256)
    report = json.loads((path.parent/'preparation.json').read_text())
    if (report.get('status') != 'passed' or report.get('tasks_sha256') != expected_sha256
            or report.get('cleanup', {}).get('verified') is not True):
        raise ValueError('metadata preparation was not successfully released')
    return validate_manifest(value, prepared, layout, bindings), value


def prepare(*, output=OUTPUT, worker_factory=None):
    cpu_only()
    prepared, layout, bindings, build = preflight()
    image = layout['protocol']['environment_image_id']
    output = Path(output)
    base.reserve(output)
    started = time.monotonic()
    report = {'status': 'started', 'started_epoch': time.time(), 'errors': [], 'container_intents': [],
        'selection_sha256': fingerprint(prepared), 'source_and_evidence_sha256': bindings,
        'task_count': 0, 'list_calls': 0, 'describe_calls': 0, 'actor_calls': 0,
        'execute_calls': 0, 'evaluate_calls': 0, 'world_start_calls': 0, 'GPU_jobs_added': 0,
        'ground_truth_loaded': False, 'study_registered': False, 'automatic_retry': False}
    base.checkpoint(output/'preparation.json', report)
    worker, creation_completed = None, False
    try:
        verify_image(image, build)
        with (output/'events.jsonl').open('x') as events:
            os.chmod(events.name, 0o600)

            def intent_callback(intent):
                runner._intent(intent, runner.client_identity(), image)
                if report['container_intents']:
                    raise ValueError('duplicate metadata container intent')
                base.append(events, {'event': 'container_intent', 'intent': intent, 'epoch': time.time()})
                report['container_intents'].append(intent)
                base.checkpoint(output/'preparation.json', report)

            factory = worker_factory or owned.OwnedAppWorldWorkerV3
            worker = factory(layout['data_dir'], image, layout['deps_dir'],
                             on_container_intent=intent_callback)
            report['list_calls'] += 1
            ids = metadata_call(worker, 'list', events, prepared['selected_ids'], split='train')
            creation_completed = True
            if type(ids) is not list or ids != official_ids():
                raise ValueError('container train IDs/order differs')
            report['container'] = base.inspect_isolation(report['container_intents'][0], layout)
            tasks = []
            with (output/'described-tasks.jsonl').open('x') as partial:
                os.chmod(partial.name, 0o600)
                for task_id in prepared['selected_ids']:
                    report['describe_calls'] += 1
                    info = metadata_call(worker, 'describe', events, prepared['selected_ids'], task_id=task_id)
                    task = task_record(task_id, info)
                    base.append(partial, asdict(task))
                    tasks.append(task)
                    report['task_count'] = len(tasks)
            value = {'tasks': [asdict(t) for t in tasks], 'split': 'train', 'image': image,
                'data_dir': layout['data_dir'], 'deps_dir': layout['deps_dir'], 'load_ground_truth': False,
                'selection_sha256': fingerprint(prepared),
                'official_train_sha256': prepared['source_sha256']['official_train'],
                'seed_protocol': runner.SEED_PROTOCOL, 'environment_seed': 100,
                'observation_feedback': runner.feedback_recipe(), 'environment_id': runner.environment_identity(image),
                'source_and_evidence_sha256': bindings,
                'provenance': {'worker': 'OwnedAppWorldWorkerV3', 'allowed_RPC': ['list(train)', 'describe(fixed12)'],
                    'ground_truth_loaded': False, 'world_created': False, 'study_registered': False,
                    'seed_used_for_metadata_loading': None, 'seed_100_required_for_later_rollouts': True}}
            validate_manifest(value, prepared, layout, bindings)
            base.write_new(output/'tasks.json', value)
            report.update(status='metadata_prepared', tasks_sha256=base.sha256(output/'tasks.json'),
                          tasks_bytes=(output/'tasks.json').stat().st_size)
    except BaseException as exc:
        report['errors'].append({'phase': 'preparation', 'type': type(exc).__name__})
    finally:
        cleanup_started = time.monotonic()
        if worker is not None:
            try:
                worker.close()
            except BaseException as exc:
                report['errors'].append({'phase': 'worker_close', 'type': type(exc).__name__})
        cleanup = {'creation_completed': creation_completed, 'containers': []}
        for intent in report['container_intents']:
            try:
                removed = owned.remove_owned_container(intent)
                absent = owned.inspect_owned_container(intent) is None
                cleanup['containers'].append({'intent': intent, 'extra_removed_id': removed,
                                              'exact_name_absent': absent})
                if not absent:
                    raise ValueError('owned container remains')
            except BaseException as exc:
                report['errors'].append({'phase': 'owned_cleanup', 'type': type(exc).__name__})
        cleanup['creator_process_exited'] = worker is not None and worker.proc.poll() is not None
        cleanup['seconds'] = time.monotonic()-cleanup_started
        cleanup['verified'] = (creation_completed and cleanup['creator_process_exited']
            and len(cleanup['containers']) == 1 and cleanup['containers'][0]['exact_name_absent']
            and not report['errors'])
        report.update(cleanup=cleanup, finished_epoch=time.time(), elapsed_seconds=time.monotonic()-started)
        report['status'] = 'passed' if report['status'] == 'metadata_prepared' and cleanup['verified'] else 'failed'
        report['permissions'] = {'requested_directory': '0700', 'requested_private_file': '0600',
            'actual_directory': oct(output.stat().st_mode & 0o777),
            'actual_tasks_file': oct((output/'tasks.json').stat().st_mode & 0o777)
                if (output/'tasks.json').exists() else None}
        base.checkpoint(output/'preparation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        prepared, _, _, _ = preflight()
        print(json.dumps({'status': 'CPU_preflight_passed', 'tasks': len(prepared['selected_ids']),
                          'metadata_loaded': False, 'uses_GPU': False}))
        return 0
    report = prepare()
    print(json.dumps({k: report[k] for k in ('status', 'task_count', 'list_calls', 'describe_calls', 'errors')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
