"""Independent V2 coordinator over authenticated, unchanged V1 source bytes.

V2 binds its own registration, episode, serialized queue and cancellation fence.
The original experimental schedule, model, seeds, gates and budgets are reused.
Import/help do not load a registration, start a unit, touch a queue or run a model.
Initiating exception details are written only through private runtime writers.
"""

import argparse
import hashlib
import importlib
import importlib.util
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_native_source_pilot.py"
SOURCE_SHA256 = "c9cac394a244e8764d3bbd7143d3b44e0db00e20cc5f6016d84504d89bbee7f6"
ID = "qwen38-native-source-pilot-002"
PLAN = "artifacts/qwen38-native-source-pilot-plan-002.json"
RELEASE_SECONDS = 90
SHUTDOWN_RESERVE_SECONDS = 150
_ERROR = "native_source_pilot_002_source_invalid"
_CORE = None


def _replace_once(source, original, replacement):
    if source.count(original) != 1:
        raise ValueError(_ERROR)
    return source.replace(original, replacement)


def _capture_initiating_exception(runtime, error):
    """Keep exception bodies out of stdout, public reports and return values."""
    try:
        value = {"event": "initiating_exception", "error_type": type(error).__name__,
                 "traceback": "".join(traceback.format_exception(error))}
        try:
            runtime.record(value)
        except BaseException:
            runtime.write("initiating-exception.json", value)
    except BaseException:
        # A failed private writer cannot authorize a retry or replace the
        # original conservative unknown outcome with a fabricated diagnosis.
        pass


def _isolated(*, raw=None, _modules=None):
    if raw is None:
        from scripts.native_source_registration import public_read

        raw = public_read(ROOT, "scripts/run_native_source_pilot.py")
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError(_ERROR)
    modules = _modules if _modules is not None else {
        name: importlib.import_module("scripts.native_source_" + suffix + "_002")
        for name, suffix in (("registration", "registration"), ("episode", "episode"), ("queue", "gpu_queue"))
    }
    if (type(modules) is not dict or set(modules) != {"registration", "episode", "queue"}
            or modules["registration"].ID != ID or modules["registration"].PLAN != PLAN
            or modules["queue"].ID != ID or modules["queue"].PLAN != PLAN
            or not callable(modules["queue"].fenced_release)):
        raise ValueError(_ERROR)
    source = raw.decode("utf-8")
    for original, replacement in (
        ("from scripts import native_source_episode as episode", "episode = _BOUND_EPISODE"),
        ("from scripts import native_source_gpu_queue as queue", "queue = _BOUND_QUEUE"),
        ("from scripts import native_source_registration as registration", "registration = _BOUND_REGISTRATION"),
        ('"episode": "scripts.native_source_episode"', '"episode": "scripts.native_source_episode_002"'),
        ('"pilot": "scripts.run_native_source_pilot"', '"pilot": "scripts.run_native_source_pilot_002"'),
        ("    except BaseException:\n        if not begun:",
         "    except BaseException as initiating_error:\n"
         "        _capture_initiating_exception(runtime, initiating_error)\n        if not begun:"),
        ('    except _BudgetStop as error:\n        report["status"] = "stopped_" + str(error)',
         '    except _BudgetStop as error:\n        _capture_initiating_exception(runtime, error)\n'
         '        report["status"] = "stopped_" + str(error)'),
        ("                    runtime.monitor()\n                    _queue_state(state, runtime)",
         "                    runtime.monitor()\n                    state = runtime.queue(\"status\", job)\n"
         "                    _queue_state(state, runtime)"),
        ('                    runtime.record({"event": "GPU_released", "release": release})\n'
         '                if heartbeat_failure is not None:',
         '                    runtime.record({"event": "GPU_released", "release": release})\n'
         '                    if release.get("runtime_overrun") is True:\n'
         '                        raise _BudgetStop("GPU_runtime_overrun")\n'
         '                if heartbeat_failure is not None:'),
    ):
        source = _replace_once(source, original, replacement)
    spec = importlib.util.spec_from_file_location("scripts._native_source_pilot_002_core", SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR)
    core = importlib.util.module_from_spec(spec)
    core.__dict__.update(_BOUND_EPISODE=modules["episode"], _BOUND_QUEUE=modules["queue"],
                         _BOUND_REGISTRATION=modules["registration"],
                         _capture_initiating_exception=_capture_initiating_exception)
    # Only exact authenticated repository bytes plus the listed trusted edits
    # are compiled; no policy-generated Python reaches this operation.
    exec(compile(source, str(SOURCE), "exec", dont_inherit=True), core.__dict__)
    core.RELEASE_SECONDS = RELEASE_SECONDS
    core.SHUTDOWN_RESERVE_SECONDS = SHUTDOWN_RESERVE_SECONDS
    core._release = modules["queue"].fenced_release
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
    if name in {"_Runtime", "_Heartbeat", "_queue_state", "NativeSourcePilotError"}:
        return getattr(_get_core(), name)
    raise AttributeError(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-sha256", required=True)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--run", action="store_true")
    choices.add_argument("--launch", action="store_true")
    # Parse help before initializing any of the bound V2 modules.
    parser.parse_args(argv)
    return _get_core().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
