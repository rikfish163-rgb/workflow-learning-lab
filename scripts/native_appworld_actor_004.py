"""Future CPU actor with bounded generation admission and durable cancellation.

Frozen actor semantics and typed count handling are loaded in isolated globals.
This module does not launch a model, task, worker, queue or accelerator.
"""

import hashlib
import importlib
import importlib.util
import math
import os
import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_appworld_actor.py"
BASE_SHA256 = "d9ae9bcd2809bcaf10cd8c73baee5417e54eb3ffc4a7f7978a25dacbf1dc9e9c"
ACTOR3_SOURCE = ROOT / "scripts/native_appworld_actor_003.py"
ACTOR3_SHA256 = "e2a532d454ad7c2becdc211e841663f5d7f2eeea753dfea522b5d69894af8c62"
REQUEST_WINDOW_SECONDS = 120.0
JOURNAL_RESERVE_SECONDS = 5.0
_ERROR = "native_actor_004_source_invalid"
_CORE = None


def _authenticated_source(name, expected):
    if name not in {BASE_SOURCE.name, ACTOR3_SOURCE.name}:
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
    if len(raw) > 1_000_000 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(_ERROR)
    return raw


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _same_model_deadline(model, expected):
    try:
        current = model.deadline
        return _finite(current) and current > 0 and current == expected
    except Exception:
        return False


def _valid_refusal(client, error, *, deadline, messages_sha256, seed, intent_monotonic):
    """Validate the client proof without exposing an exception or request body."""
    try:
        if type(error) is not client.NativeGenerationNotStarted:
            return None
        audit = client.validate_generation_not_started(
            error, deadline=deadline, messages_sha256=messages_sha256, seed=seed)
        if (type(audit) is not dict or set(audit) != {
                "source_revision", "deadline", "observed_monotonic", "request_window_seconds",
                "request_sha256", "messages_sha256", "seed", "transport_started"}
                or type(audit["source_revision"]) is not int or audit["source_revision"] != 4
                or not _finite(audit["deadline"]) or audit["deadline"] != deadline
                or not _finite(audit["observed_monotonic"])
                or audit["observed_monotonic"] < intent_monotonic
                or not _finite(audit["request_window_seconds"])
                or audit["request_window_seconds"] != REQUEST_WINDOW_SECONDS
                or deadline - audit["observed_monotonic"] >= REQUEST_WINDOW_SECONDS
                or type(audit["seed"]) is not int or audit["seed"] != seed
                or audit["messages_sha256"] != messages_sha256
                or type(audit["request_sha256"]) is not str
                or re.fullmatch(r"[a-f0-9]{64}", audit["request_sha256"]) is None
                or audit["transport_started"] is not False):
            return None
        return audit
    except Exception:
        return None


def _cancel_event(*, source, reason, seed, prompt_tokens, messages_sha256, request_sha256,
                  intent_monotonic, observed_monotonic, validated_monotonic,
                  model_deadline, actor_deadline, cost_counter_decremented):
    return {"event": "generation_not_started", "proof_source": source, "reason": reason,
            "seed": seed, "prompt_tokens": prompt_tokens, "messages_sha256": messages_sha256,
            "request_sha256": request_sha256, "intent_monotonic": intent_monotonic,
            "observed_monotonic": observed_monotonic, "validated_monotonic": validated_monotonic,
            "model_deadline": model_deadline, "actor_deadline": actor_deadline,
            "request_window_seconds": REQUEST_WINDOW_SECONDS,
            "journal_reserve_seconds": JOURNAL_RESERVE_SECONDS, "transport_started": False,
            "cost_counter_decremented": cost_counter_decremented}


def _isolated(*, raw=None, raw003=None, _counter_module=None, _client_module=None):
    raw = _authenticated_source(BASE_SOURCE.name, BASE_SHA256) if raw is None else raw
    raw003 = _authenticated_source(ACTOR3_SOURCE.name, ACTOR3_SHA256) if raw003 is None else raw003
    if (type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256
            or type(raw003) is not bytes or hashlib.sha256(raw003).hexdigest() != ACTOR3_SHA256):
        raise ValueError(_ERROR)
    spec = importlib.util.spec_from_file_location("scripts._native_appworld_actor_004_actor3", ACTOR3_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR)
    actor3 = importlib.util.module_from_spec(spec)
    exec(compile(raw003, str(ACTOR3_SOURCE), "exec", dont_inherit=True), actor3.__dict__)
    core = actor3._isolated(raw=raw, _counter_module=_counter_module)
    client = _client_module or importlib.import_module("scripts.native_appworld_model_client_004")
    original_actor = importlib.import_module("scripts.native_appworld_actor")

    count_old = ('            except Exception as error:\n'
                 '                exception_event("token_count_failure", error)\n'
                 '                return finish("unknown", "token_count_failure")')
    count_new = ('            except Exception as error:\n'
                 '                if _count_failure_is_budget(error, model, deadline, now, write):\n'
                 '                    return finish("incomplete", "wall_seconds")\n'
                 '                return finish("unknown", "token_count_failure")')
    intent_old = ('            generation_seed = seed + cost["generations"]\n'
                  '            write({"event": "generation_intent", "messages": messages, "seed": generation_seed,\n'
                  '                   "prompt_tokens": prompt_tokens, "max_new_tokens": limits.max_new_tokens})\n'
                  '            if now() >= deadline:\n'
                  '                return finish("incomplete", "wall_seconds")')
    intent_new = '''            try:
                model_deadline = getattr(model, "deadline", None)
            except Exception:
                model_deadline = None
            if not _actor4_finite(model_deadline) or not 0 < model_deadline <= deadline:
                write({"event": "request_window_failure", "reason": "invalid_model_deadline"})
                return finish("unknown", "invalid_model_deadline")
            admission_time = now()
            if admission_time < 0:
                raise _ClockError
            if model_deadline - admission_time < _actor4_window + _actor4_reserve:
                write({"event": "request_window_budget_stop", "observed_monotonic": admission_time,
                       "model_deadline": model_deadline, "actor_deadline": deadline,
                       "request_window_seconds": _actor4_window, "journal_reserve_seconds": _actor4_reserve})
                return finish("incomplete", "request_window_seconds")
            try:
                messages_sha256 = _actor4_client.generation_messages_sha256(messages)
            except Exception:
                write({"event": "request_window_failure", "reason": "message_hash_failure"})
                return finish("unknown", "message_hash_failure")
            intent_monotonic = now()
            if model_deadline - intent_monotonic < _actor4_window + _actor4_reserve:
                write({"event": "request_window_budget_stop", "observed_monotonic": intent_monotonic,
                       "model_deadline": model_deadline, "actor_deadline": deadline,
                       "request_window_seconds": _actor4_window, "journal_reserve_seconds": _actor4_reserve})
                return finish("incomplete", "request_window_seconds")
            generation_seed = seed + cost["generations"]
            write({"event": "generation_intent", "messages": messages, "seed": generation_seed,
                   "prompt_tokens": prompt_tokens, "max_new_tokens": limits.max_new_tokens})
            after_intent = now()
            if after_intent >= deadline:
                phase = "generation"
                write(_actor4_cancel_event(source="actor_before_model_call", reason="wall_seconds",
                      seed=generation_seed, prompt_tokens=prompt_tokens, messages_sha256=messages_sha256,
                      request_sha256=None, intent_monotonic=intent_monotonic,
                      observed_monotonic=after_intent, validated_monotonic=after_intent,
                      model_deadline=model_deadline, actor_deadline=deadline, cost_counter_decremented=False))
                phase = None
                return finish("incomplete", "wall_seconds")'''
    failure_old = ('            except Exception as error:\n'
                   '                unknown_cost = True\n'
                   '                exception_event("generation_failure", error)\n'
                   '                phase = None\n'
                   '                return finish("unknown", "generation_failure")')
    failure_new = '''            except Exception as error:
                audit = _actor4_valid_refusal(_actor4_client, error, deadline=model_deadline,
                         messages_sha256=messages_sha256, seed=generation_seed,
                         intent_monotonic=intent_monotonic)
                if audit is not None:
                    validated_monotonic = now()
                    if (audit["observed_monotonic"] <= validated_monotonic
                            and _actor4_same_model_deadline(model, model_deadline)):
                        write(_actor4_cancel_event(source="client_before_transport", reason="request_window_seconds",
                              seed=generation_seed, prompt_tokens=prompt_tokens, messages_sha256=messages_sha256,
                              request_sha256=audit["request_sha256"], intent_monotonic=intent_monotonic,
                              observed_monotonic=audit["observed_monotonic"], validated_monotonic=validated_monotonic,
                              model_deadline=model_deadline, actor_deadline=deadline, cost_counter_decremented=True))
                        cost["generations"] -= 1
                        phase = None
                        return finish("incomplete", "request_window_seconds")
                unknown_cost = True
                write({"event": "generation_failure", "error_type": type(error).__name__,
                       "message": "Model generation failed."})
                phase = None
                return finish("unknown", "generation_failure")'''
    exception_old = ('    def exception_event(event, error):\n'
                     '        try:\n'
                     '            message = str(error)\n'
                     '        except Exception:\n'
                     '            message = "Exception message unavailable."\n'
                     '        write({"event": event, "error_type": type(error).__name__, "message": message})')
    exception_new = ('    def exception_event(event, error):\n'
                     '        write({"event": event, "error_type": type(error).__name__,\n'
                     '               "message": "Operation failed."})')
    source = raw.decode("utf-8")
    for old, new in ((count_old, count_new), (intent_old, intent_new), (failure_old, failure_new),
                     (exception_old, exception_new)):
        if source.count(old) != 1:
            raise ValueError(_ERROR)
        source = source.replace(old, new)
    exec(compile(source, str(BASE_SOURCE), "exec", dont_inherit=True), core.__dict__)
    core.ActorLimits = original_actor.ActorLimits
    core._actor4_client = client
    core._actor4_finite = _finite
    core._actor4_same_model_deadline = _same_model_deadline
    core._actor4_window = REQUEST_WINDOW_SECONDS
    core._actor4_reserve = JOURNAL_RESERVE_SECONDS
    core._actor4_valid_refusal = _valid_refusal
    core._actor4_cancel_event = _cancel_event
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
