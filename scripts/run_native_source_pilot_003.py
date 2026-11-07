"""V3 coordinator over authenticated V2 lifecycle and release-fence code."""

import argparse
import hashlib
import importlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/run_native_source_pilot_002.py'
BASE_SHA256 = '5d7819334cbdd6a1034aa2d09e71a029df372c3ddf24a63f75b865b12d74ca29'
ID = 'qwen38-native-source-pilot-003'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-003.json'
RELEASE_SECONDS = 90
SHUTDOWN_RESERVE_SECONDS = 150
_CORE = None


def _isolated(*, raw=None, _modules=None):
    if raw is None:
        from scripts.native_source_registration import public_read

        raw = public_read(ROOT, 'scripts/run_native_source_pilot_002.py')
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('native_source_pilot_003_source_invalid')
    modules = _modules if _modules is not None else {
        'registration': importlib.import_module('scripts.native_source_registration_003'),
        'episode': importlib.import_module('scripts.native_source_episode_003'),
        'queue': importlib.import_module('scripts.native_source_gpu_queue_003'),
        'actor': importlib.import_module('scripts.native_appworld_actor_003'),
    }
    if (type(modules) is not dict or set(modules) != {'registration', 'episode', 'queue', 'actor'}
            or any(modules[name].ID != ID or modules[name].PLAN != PLAN
                   for name in ('registration', 'episode', 'queue'))
            or not callable(modules['queue'].fenced_release)):
        raise ValueError('native_source_pilot_003_binding_invalid')
    source = raw.decode('utf-8')
    for old, new in (
        ('qwen38-native-source-pilot-002', ID),
        ('artifacts/qwen38-native-source-pilot-plan-002.json', PLAN),
        ('"_002"', '"_003"'),
        ('scripts.native_source_episode_002', 'scripts.native_source_episode_003'),
        ('scripts.run_native_source_pilot_002', 'scripts.run_native_source_pilot_003'),
        ('scripts._native_source_pilot_002_core', 'scripts._native_source_pilot_003_core'),
    ):
        if source.count(old) != 1:
            raise ValueError('native_source_pilot_003_binding_invalid')
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._native_source_pilot_003_base', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError('native_source_pilot_003_source_invalid')
    base = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASE_SOURCE), 'exec', dont_inherit=True), base.__dict__)
    core = base._isolated(_modules={name: modules[name] for name in ('registration', 'episode', 'queue')})
    core.actor = modules['actor']
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def run_pilot(plan_sha, root=None, _runtime=None):
    core = _get_core()
    return core.run_pilot(plan_sha, root=core.WORKSPACE if root is None else root, _runtime=_runtime)


def launch_pilot(plan_sha, *, root=None):
    core = _get_core()
    return core.launch_pilot(plan_sha, root=core.WORKSPACE if root is None else root)


def unit_command(*args, **kwargs):
    return _get_core().unit_command(*args, **kwargs)


def initial_slots():
    return _get_core().initial_slots()


def __getattr__(name):
    if name in {'_Runtime', '_Heartbeat', '_queue_state', 'NativeSourcePilotError'}:
        return getattr(_get_core(), name)
    raise AttributeError(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan-sha256', required=True)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument('--run', action='store_true')
    choices.add_argument('--launch', action='store_true')
    parser.parse_args(argv)
    return _get_core().main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
