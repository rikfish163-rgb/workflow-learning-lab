"""CPU-only future actor behavior for authenticated, typed count budget expiry.

The frozen actor's messages, actions, seeds, costs and journal ordering remain
unchanged. Only a trusted CPU-count deadline exception may produce a normal
wall-budget stop. Import never starts a task, counter, model or GPU.
"""

import hashlib
import importlib
import importlib.util
import math
import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_appworld_actor.py"
BASE_SHA256 = "d9ae9bcd2809bcaf10cd8c73baee5417e54eb3ffc4a7f7978a25dacbf1dc9e9c"
_ERROR = "native_actor_003_source_invalid"
_CORE = None


def authenticated_source(name, digest):
    """Authenticate one fixed actor/controller source without symlink traversal."""
    if name not in {"native_appworld_actor.py", "native_appworld_controller.py"}:
        raise ValueError(_ERROR)
    from scripts.native_appworld_worker_client import _open_directory

    directory = _open_directory(ROOT / "scripts")
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1_000_000:
                raise ValueError(_ERROR)
            raw = file.read(1_000_001)
    finally:
        os.close(directory)
    if len(raw) > 1_000_000 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(_ERROR)
    return raw


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count_failure(error, model, reserved_end, now, write, counter):
    audit = None
    try:
        audit = counter.validate_count_audit(getattr(error, "count_audit", None))
    except Exception:
        pass
    cause, cause_types = error.__cause__, []
    while cause is not None and len(cause_types) < 8:
        cause_types.append(type(cause).__name__)
        cause = cause.__cause__
    details = {"error_type": type(error).__name__, "message": "CPU token count failed.",
               "count_audit": audit, "cause_types": cause_types}
    error_deadline, model_deadline = getattr(error, "deadline", None), getattr(model, "deadline", None)
    phase = getattr(error, "phase", None)
    trusted = (type(error) is counter.ActorCountDeadlineExpired
               and _finite(error_deadline) and _finite(model_deadline)
               and error_deadline == model_deadline and error_deadline <= reserved_end
               and phase in {"pre_count", "counter_budget_timeout", "validated_count_returned_late"}
               and type(audit) is dict and audit.get("actor_deadline") == error_deadline)
    if trusted and now() >= error_deadline:
        write({"event": "token_count_budget_stop", **details,
               "message": "actor_count_deadline_expired", "phase": phase})
        return True
    write({"event": "token_count_failure", **details})
    return False


def _isolated(*, raw=None, _counter_module=None):
    raw = authenticated_source(BASE_SOURCE.name, BASE_SHA256) if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError(_ERROR)
    counter = _counter_module or importlib.import_module("scripts.native_verification_token_counter_003")
    previous = importlib.import_module("scripts.native_appworld_actor")
    original = ('            except Exception as error:\n'
                '                exception_event("token_count_failure", error)\n'
                '                return finish("unknown", "token_count_failure")')
    replacement = ('            except Exception as error:\n'
                   '                if _count_failure_is_budget(error, model, deadline, now, write):\n'
                   '                    return finish("incomplete", "wall_seconds")\n'
                   '                return finish("unknown", "token_count_failure")')
    source = raw.decode("utf-8")
    if source.count(original) != 1:
        raise ValueError(_ERROR)
    spec = importlib.util.spec_from_file_location("scripts._native_appworld_actor_003_core", BASE_SOURCE)
    core = importlib.util.module_from_spec(spec)
    exec(compile(source.replace(original, replacement), str(BASE_SOURCE), "exec", dont_inherit=True), core.__dict__)
    core.ActorLimits = previous.ActorLimits
    core._count_failure_is_budget = lambda error, model, end, now, write: _count_failure(
        error, model, end, now, write, counter)
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def run_actor(*args, **kwargs):
    return _get_core().run_actor(*args, **kwargs)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_get_core(), name)
