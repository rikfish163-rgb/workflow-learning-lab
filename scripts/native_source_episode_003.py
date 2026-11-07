"""V3 private episode using only the V3 actor, controller and model client."""

import argparse
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/native_source_episode_002.py'
BASE_SHA256 = '700a7b9d083f946860ae6f422d912862c2407723379e4ac0fd36e17fb14a082f'
ID = 'qwen38-native-source-pilot-003'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-003.json'
_CORE = None


def _isolated(*, raw=None, _modules=None):
    from scripts.native_source_registration import public_read

    raw = public_read(ROOT, 'scripts/native_source_episode_002.py') if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('native_source_episode_003_source_invalid')
    modules = _modules if _modules is not None else {
        'registration': importlib.import_module('scripts.native_source_registration_003'),
        'queue': importlib.import_module('scripts.native_source_gpu_queue_003'),
        'actor': importlib.import_module('scripts.native_appworld_actor_003'),
        'controller': importlib.import_module('scripts.native_appworld_controller_003'),
        'model': importlib.import_module('scripts.native_appworld_model_client_003'),
    }
    if (type(modules) is not dict or set(modules) != {'registration', 'queue', 'actor', 'controller', 'model'}
            or any(modules[name].ID != ID or modules[name].PLAN != PLAN for name in ('registration', 'queue'))
            or not callable(modules['controller'].run_task) or not callable(modules['model'].NativeModel)):
        raise ValueError('native_source_episode_003_binding_invalid')
    source = raw.decode('utf-8')
    for old, new in (('qwen38-native-source-pilot-002', ID),
                     ('artifacts/qwen38-native-source-pilot-plan-002.json', PLAN)):
        if source.count(old) != 1:
            raise ValueError('native_source_episode_003_binding_invalid')
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._native_source_episode_003_base', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError('native_source_episode_003_source_invalid')
    base = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASE_SOURCE), 'exec', dont_inherit=True), base.__dict__)
    legacy = public_read(ROOT, 'scripts/native_source_episode.py')
    core = base._isolated(raw=legacy, _modules={'registration': modules['registration'], 'queue': modules['queue']})
    core.actor, core.run_task, core.NativeModel = modules['actor'], modules['controller'].run_task, modules['model'].NativeModel
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
