"""Independent source-006 episode over authenticated 005 phase machinery.

Only the new registration/queue namespace and diagnostic worker/controller are
bound. The 004 actor and model, replay, guard, budgets and scorer separation
are unchanged. Import starts no task, model, worker or queue operation.
"""

import argparse
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

from scripts.native_source_registration import public_read

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/native_source_episode_005.py'
BASE_SHA256 = '0daba5434fea3202241de0030fed60c5229f50a0be2e07768df7ae6dec20c612'
ID = 'qwen38-native-source-pilot-006'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-006.json'
_ERROR = 'native_source_episode_006_source_invalid'
_CORE = None


def _isolated(*, raw=None, _modules=None):
    raw = public_read(ROOT, 'scripts/native_source_episode_005.py') if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError(_ERROR)
    modules = _modules if _modules is not None else {
        'registration': importlib.import_module('scripts.native_source_registration_006'),
        'queue': importlib.import_module('scripts.native_source_gpu_queue_006'),
        'actor': importlib.import_module('scripts.native_appworld_actor_004'),
        'controller': importlib.import_module('scripts.native_appworld_controller_006'),
        'model': importlib.import_module('scripts.native_appworld_model_client_004'),
    }
    if (type(modules) is not dict or set(modules) != {'registration', 'queue', 'actor', 'controller', 'model'}
            or any(modules[name].ID != ID or modules[name].PLAN != PLAN for name in ('registration', 'queue'))
            or not callable(modules['controller'].run_task) or not callable(modules['model'].NativeModel)):
        raise ValueError('native_source_episode_006_binding_invalid')
    spec = importlib.util.spec_from_file_location('scripts._native_source_episode_006_base', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR)
    base = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), 'exec', dont_inherit=True), base.__dict__)
    base.ID, base.PLAN = ID, PLAN
    core = base._isolated(_modules=modules)
    core.NativeWorkerClient = importlib.import_module('scripts.native_appworld_worker_client_006').NativeWorkerClient
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def run_episode(config_path, expected_config_sha256, *, root=None, _load_plan=None,
                _model_factory=None, _run_task=None, _queue_status=None):
    core = _get_core()
    return core.run_episode(config_path, expected_config_sha256,
        root=core.WORKSPACE if root is None else root, _load_plan=_load_plan,
        _model_factory=_model_factory, _run_task=_run_task, _queue_status=_queue_status)


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_get_core(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.parse_args(argv)
    try:
        return _get_core().main(argv)
    except Exception:
        sys.stderr.write('native_source_episode_failed\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
