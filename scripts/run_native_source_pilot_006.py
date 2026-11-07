"""Independent 006 schedule over authenticated 005 resource and gate logic.

No predecessor outcome is resumed or treated as a new source observation.
The complete twelve-source roster, paired controls, cost accounting and stop
rules remain in the isolated core. Import/help starts no experimental work.
"""

import argparse
import hashlib
import importlib
import importlib.util
from pathlib import Path

from scripts.native_source_registration import public_read

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/run_native_source_pilot_005.py'
BASE_SHA256 = '353182da5bf05b2aa5aaeb6def303388d61d82c9086ae8e621baa88edeb195aa'
ID = 'qwen38-native-source-pilot-006'
PLAN = 'artifacts/qwen38-native-source-pilot-plan-006.json'
PARENT_SECONDS = 12 * (86400 + 7200) + 600
RELEASE_SECONDS = 90
SHUTDOWN_RESERVE_SECONDS = 150
_ERROR = 'native_source_pilot_006_source_invalid'
_CORE = None


def _isolated(*, raw=None, _modules=None):
    raw = public_read(ROOT, 'scripts/run_native_source_pilot_005.py') if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError(_ERROR)
    source = raw.decode('utf-8')
    for old, new, count in (
        ('qwen38-native-source-pilot-005', ID, 1),
        ('artifacts/qwen38-native-source-pilot-plan-005.json', PLAN, 1),
        ('scripts.native_source_episode_005', 'scripts.native_source_episode_006', 3),
        ('scripts.run_native_source_pilot_005', 'scripts.run_native_source_pilot_006', 2),
        ('scripts.native_source_registration_005', 'scripts.native_source_registration_006', 1),
        ('scripts.native_source_gpu_queue_005', 'scripts.native_source_gpu_queue_006', 1),
        ('scripts._native_source_pilot_005_core', 'scripts._native_source_pilot_006_core', 1),
        ('scripts._native_source_pilot_005_base', 'scripts._native_source_pilot_006_base', 1),
        ('"_005"', '"_006"', 1),
        ('native_source_pilot_005_source_invalid', _ERROR, 1),
    ):
        if source.count(old) != count:
            raise ValueError(_ERROR)
        source = source.replace(old, new)
    modules = _modules if _modules is not None else {
        'registration': importlib.import_module('scripts.native_source_registration_006'),
        'episode': importlib.import_module('scripts.native_source_episode_006'),
        'queue': importlib.import_module('scripts.native_source_gpu_queue_006'),
        'actor': importlib.import_module('scripts.native_appworld_actor_004'),
    }
    spec = importlib.util.spec_from_file_location('scripts._native_source_pilot_006_parent', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR)
    base = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASE_SOURCE), 'exec', dont_inherit=True), base.__dict__)
    return base._isolated(_modules=modules)


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
