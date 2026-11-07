"""Unchanged frozen controller lifecycle with the future CPU Actor3 binding.

Only the actor delegate changes. Worker ownership, replay, setup, cleanup,
save/close and unknown-result behavior reuse authenticated frozen source.
Import never starts a task, counter, model, scorer or GPU.
"""

import hashlib
import importlib.util
from pathlib import Path

from scripts import native_appworld_actor_003 as actor

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_appworld_controller.py"
BASE_SHA256 = "6b29dfeb4c62ae4641cbf4cbaf07687f8eac45ba899a5fc938f73278736eed45"
_ERROR = "native_controller_003_source_invalid"
_CORE = None


def _isolated(*, raw=None):
    raw = actor.authenticated_source(BASE_SOURCE.name, BASE_SHA256) if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError(_ERROR)
    spec = importlib.util.spec_from_file_location("scripts._native_appworld_controller_003_core", BASE_SOURCE)
    core = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), "exec", dont_inherit=True), core.__dict__)
    core.ActorLimits, core.run_actor = actor.ActorLimits, actor.run_actor
    core._validate_review_hint = actor._validate_review_hint
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def run_task(*args, **kwargs):
    return _get_core().run_task(*args, **kwargs)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_get_core(), name)
