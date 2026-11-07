"""Use the full setup remainder for first metadata; retain other phase limits.

The original 004 lifecycle/actor is authenticated and isolated. Default RPC60
and explicit actor/replay overrides keep their historical phase/worker bounds;
save and close still use the original min(default60, remaining) calculation.
This new setup contract does not establish the original source004 root cause.
Import starts no task, worker, model, scorer or accelerator operation.
"""

import ast
import hashlib
import importlib.util
from pathlib import Path

from scripts.native_source_registration import public_read

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/native_appworld_controller_004.py'
BASE_SHA256 = '5dda22d17a0a74c870564fc9eed853c5ed2041f526eb72fbb187cc77d1dce156'
LEGACY_SOURCE = ROOT / 'scripts/native_appworld_controller.py'
LEGACY_SHA256 = '6b29dfeb4c62ae4641cbf4cbaf07687f8eac45ba899a5fc938f73278736eed45'
_ERROR = 'native_controller_006_source_invalid'
_CORE = None


def _isolated(*, raw=None, raw_legacy=None):
    raw = public_read(ROOT, 'scripts/native_appworld_controller_004.py') if raw is None else raw
    raw_legacy = public_read(ROOT, 'scripts/native_appworld_controller.py') if raw_legacy is None else raw_legacy
    if (type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256
            or type(raw_legacy) is not bytes or hashlib.sha256(raw_legacy).hexdigest() != LEGACY_SHA256):
        raise ValueError(_ERROR)
    spec = importlib.util.spec_from_file_location('scripts._native_controller_006_parent', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR)
    parent = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), 'exec', dont_inherit=True), parent.__dict__)
    core = parent._isolated(raw=raw_legacy)
    tree = ast.parse(raw_legacy)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_task']
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError(_ERROR)
    function = ast.get_source_segment(raw_legacy.decode('utf-8'), functions[0])
    before = '            metadata = worker.request({"op": "metadata"}, timeout=remaining(setup_deadline))'
    after = ('            metadata_timeout = setup_deadline - _clock()\n'
             '            if not math.isfinite(metadata_timeout) or metadata_timeout <= 0:\n'
             '                raise NativeControllerError("native_controller_deadline")\n'
             '            metadata = worker.request({"op": "metadata"}, timeout=metadata_timeout)')
    if type(function) is not str or function.count(before) != 1:
        raise ValueError(_ERROR)
    exec(compile(function.replace(before, after, 1), str(LEGACY_SOURCE), 'exec', dont_inherit=True), core.__dict__)
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def run_task(*args, **kwargs):
    return _get_core().run_task(*args, **kwargs)


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_get_core(), name)
