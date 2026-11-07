"""V2 episode with the unchanged, authenticated V1 actor boundary and limits.

Registration, queue operations and private paths bind exclusively to V2. The
historical native-source-001 experiment prefix is retained; root and owner are
distinct. Import and help perform no task, queue, model or registration action.
"""

import argparse
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_source_episode.py"
BASE_SHA256 = "6b034e2b0c53050832c43384b1e3c4bc5853f5c8e940bbe52c348a0d3ce36e1b"
ID = "qwen38-native-source-pilot-002"
PLAN = "artifacts/qwen38-native-source-pilot-plan-002.json"
_ERROR = "native_source_episode_002_source_invalid"
_CORE = None


def _isolated(*, raw=None, _modules=None):
    if raw is None:
        from scripts.native_source_registration_002 import authenticated_source

        raw = authenticated_source(BASE_SOURCE.name, BASE_SHA256)
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError(_ERROR)
    modules = _modules if _modules is not None else {
        "registration": importlib.import_module("scripts.native_source_registration_002"),
        "queue": importlib.import_module("scripts.native_source_gpu_queue_002")}
    if (type(modules) is not dict or set(modules) != {"registration", "queue"}
            or any(module.ID != ID or module.PLAN != PLAN for module in modules.values())):
        raise ValueError(_ERROR)
    source = raw.decode("utf-8")
    for original, replacement in (
        ("from scripts import native_source_gpu_queue as queue", "queue = _BOUND_QUEUE"),
        ("from scripts import native_source_registration as registration", "registration = _BOUND_REGISTRATION")):
        if source.count(original) != 1:
            raise ValueError(_ERROR)
        source = source.replace(original, replacement)
    spec = importlib.util.spec_from_file_location("scripts._native_source_episode_002_core", BASE_SOURCE)
    core = importlib.util.module_from_spec(spec)
    core.__dict__.update(_BOUND_QUEUE=modules["queue"], _BOUND_REGISTRATION=modules["registration"])
    exec(compile(source, str(BASE_SOURCE), "exec", dont_inherit=True), core.__dict__)
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
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_get_core(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.parse_args(argv)
    try:
        return _get_core().main(argv)
    except Exception:
        sys.stderr.write("native_source_episode_failed\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
