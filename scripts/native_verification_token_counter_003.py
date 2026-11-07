"""CPU counting with explicit provenance for a planned actor deadline.

Only the frozen CPU_CODE is sent to a direct child. All exceptions and audits
belong to trusted private diagnostics; their strings contain no request bodies.
A child failure remains unknown unless it proves the selected actor deadline
expired, or a fully validated successful count returned after that deadline.
"""

import hashlib
import json
import math
import os
import re
import socket
import stat
import subprocess
import time
from pathlib import Path

from scripts import verification_token_counter as protocol

CPU_CODE = protocol.CPU_CODE
MODEL_PATH = protocol.MODEL_PATH
PYTHON_PATH = protocol.PYTHON_PATH
TIMEOUT_SECONDS = protocol.TIMEOUT_SECONDS
MAX_RESPONSE_BYTES = protocol.MAX_RESPONSE_BYTES
TOKENIZER_SHA256 = dict(protocol.TOKENIZER_SHA256)
TokenCountError = protocol.TokenCountError
TMPDIR = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state/runtime-tmp"
PROTOCOL_SOURCE_SHA256 = "22f50fe45d6e437cf8632ae782c927418cf37fcf2c26dca71c39a85d6ea372c8"
_PINNED_CODE = CPU_CODE
_PINNED_TOKENIZER = dict(TOKENIZER_SHA256)
_PINNED_MODEL = MODEL_PATH
_PINNED_PYTHON = PYTHON_PATH
_PHASES = frozenset({"pre_count", "counter_budget_timeout", "validated_count_returned_late"})
COUNT_AUDIT_FIELDS = frozenset({
    "started_monotonic", "finished_monotonic", "actor_deadline", "timeout_seconds",
    "timeout_limit_seconds", "deadline_limited", "child_started", "child_returncode",
    "child_stdout_bytes", "child_stdout_sha256", "child_stderr_bytes", "child_stderr_sha256",
    "messages_sha256", "proof_validated",
})
_TIMES = {"started_monotonic", "finished_monotonic", "actor_deadline", "timeout_seconds", "timeout_limit_seconds"}
_HASHES = {"child_stdout_sha256", "child_stderr_sha256", "messages_sha256"}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_count_audit(audit):
    """Return an independent, fixed-shape metadata copy or a fixed error."""
    try:
        if type(audit) is not dict or set(audit) != COUNT_AUDIT_FIELDS:
            raise ValueError
        result = dict(audit)
        for key in _TIMES:
            if result[key] is not None:
                if not _number(result[key]) or result[key] < 0:
                    raise ValueError
                result[key] = float(result[key])
        for key in ("deadline_limited", "proof_validated"):
            if type(result[key]) is not bool:
                raise ValueError
        if result["child_started"] is not None and type(result["child_started"]) is not bool:
            raise ValueError
        for key in ("child_stdout_bytes", "child_stderr_bytes"):
            if result[key] is not None and (type(result[key]) is not int or not 0 <= result[key] < 2**63):
                raise ValueError
        code = result["child_returncode"]
        if code is not None and (type(code) is not int or not -(2**31) <= code < 2**31):
            raise ValueError
        for key in _HASHES:
            if result[key] is not None and (type(result[key]) is not str
                                           or re.fullmatch(r"[a-f0-9]{64}", result[key]) is None):
                raise ValueError
        if (result["started_monotonic"] is not None and result["finished_monotonic"] is not None
                and result["finished_monotonic"] < result["started_monotonic"]):
            raise ValueError
        if result["timeout_limit_seconds"] is not None and not 0 < result["timeout_limit_seconds"] <= 55:
            raise ValueError
        if result["timeout_seconds"] is not None and not 0 < result["timeout_seconds"] <= 55:
            raise ValueError
        return result
    except Exception:
        raise ValueError("count_audit_invalid") from None


class ActorCountDeadlineExpired(TimeoutError):
    """Trusted typed signal; never contains child output or request content."""

    def __init__(self, *, deadline, phase, count_audit):
        if not _number(deadline) or deadline <= 0 or type(phase) is not str or phase not in _PHASES:
            raise ValueError("actor_count_deadline_invalid")
        audit = validate_count_audit(count_audit)
        if audit["actor_deadline"] != float(deadline):
            raise ValueError("actor_count_deadline_invalid")
        self.deadline, self.phase, self.count_audit = float(deadline), phase, audit
        super().__init__("actor_count_deadline_expired")


def _new_audit(deadline, timeout_limit):
    return {"started_monotonic": None, "finished_monotonic": None,
            "actor_deadline": float(deadline) if _number(deadline) and deadline > 0 else None,
            "timeout_seconds": None,
            "timeout_limit_seconds": float(timeout_limit) if _number(timeout_limit) and 0 < timeout_limit <= 55 else None,
            "deadline_limited": False, "child_started": False, "child_returncode": None,
            "child_stdout_bytes": None, "child_stdout_sha256": None,
            "child_stderr_bytes": None, "child_stderr_sha256": None,
            "messages_sha256": None, "proof_validated": False}


def _clock():
    value = time.monotonic()
    if not _number(value) or value < 0:
        raise ValueError("counter_clock_invalid")
    return float(value)


def _finish(audit):
    value = _clock()
    if audit["started_monotonic"] is not None and value < audit["started_monotonic"]:
        raise ValueError("counter_clock_invalid")
    audit["finished_monotonic"] = value
    return value


def _failure(code, audit, cause):
    error = TokenCountError(code)
    error.count_audit = validate_count_audit(audit)
    raise error from cause


def _source_digest(path):
    """Hash a regular source file without symlink traversal or byte changes."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("counter_source_binding")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    descriptor = None
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 2_000_000:
            raise ValueError("counter_source_binding")
        digest, total = hashlib.sha256(), 0
        while block := os.read(descriptor, 65536):
            total += len(block)
            if total > 2_000_000:
                raise ValueError("counter_source_binding")
            digest.update(block)
        after = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        if total != before.st_size or any(getattr(before, key) != getattr(after, key)
                                         or getattr(after, key) != getattr(named, key) for key in fields):
            raise ValueError("counter_source_binding")
        return digest.hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _verify_protocol_source():
    if (_source_digest(protocol.__file__) != PROTOCOL_SOURCE_SHA256
            or CPU_CODE != protocol.CPU_CODE or CPU_CODE != _PINNED_CODE
            or TOKENIZER_SHA256 != protocol.TOKENIZER_SHA256 or TOKENIZER_SHA256 != _PINNED_TOKENIZER
            or MODEL_PATH != protocol.MODEL_PATH or MODEL_PATH != _PINNED_MODEL
            or PYTHON_PATH != protocol.PYTHON_PATH or PYTHON_PATH != _PINNED_PYTHON):
        raise ValueError("counter_source_binding")


def _bytes_audit(audit, stdout, stderr):
    for name, value in (("stdout", stdout), ("stderr", stderr)):
        if type(value) is bytes:
            audit["child_" + name + "_bytes"] = len(value)
            audit["child_" + name + "_sha256"] = hashlib.sha256(value).hexdigest()


def _reject(_value):
    raise ValueError("counter_nonfinite_JSON")


def _valid_proof(proof, digest):
    return (type(proof) is dict and set(proof) == protocol._REPLY_FIELDS
            and type(proof["status"]) is str and proof["status"] == "measured_CPU_tokenization"
            and type(proof["messages_sha256"]) is str and proof["messages_sha256"] == digest
            and type(proof["model_path"]) is str and proof["model_path"] == MODEL_PATH
            and type(proof["tokenizer_files_sha256"]) is dict and proof["tokenizer_files_sha256"] == TOKENIZER_SHA256
            and type(proof["CUDA_VISIBLE_DEVICES"]) is str and proof["CUDA_VISIBLE_DEVICES"] == ""
            and proof["cuda_initialized"] is False and proof["model_weights_loaded"] is False
            and type(proof["GPU_jobs"]) is int and proof["GPU_jobs"] == 0
            and type(proof["model_calls"]) is int and proof["model_calls"] == 0
            and type(proof["input_tokens"]) is int and proof["input_tokens"] > 0)


def count_input_tokens(messages, *, deadline, timeout_seconds=55):
    """Count one private chat under an absolute actor deadline and a <=55s cap."""
    audit = _new_audit(deadline, timeout_seconds)
    if audit["actor_deadline"] is None or audit["timeout_limit_seconds"] is None:
        _failure("token_count_invalid_timeout", audit, ValueError("counter_limits_invalid"))
    deadline = float(deadline)
    try:
        audit["started_monotonic"] = _clock()
    except Exception as error:
        _failure("token_count_invalid_clock", audit, error)
    try:
        _verify_protocol_source()
    except Exception as error:
        _failure("token_count_source_binding_failed", audit, error)
    try:
        if socket.gethostname() != "amax":
            raise ValueError("counter_host_invalid")
    except Exception as error:
        _failure("token_count_wrong_host", audit, error)
    try:
        payload, digest = protocol._message_payload(messages)
        audit["messages_sha256"] = digest
    except Exception as error:
        _failure("token_count_invalid_messages", audit, error)
    try:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", HF_HOME="/dev/null",
                   HF_HUB_CACHE="/dev/null", XDG_CACHE_HOME="/dev/null", TOKENIZERS_PARALLELISM="false", TMPDIR=TMPDIR)
        _verify_protocol_source()
    except Exception as error:
        _failure("token_count_source_binding_failed", audit, error)
    try:
        before_spawn = _finish(audit)
    except Exception as error:
        _failure("token_count_invalid_clock", audit, error)
    remaining = deadline - before_spawn
    if remaining <= 0:
        audit["deadline_limited"] = True
        raise ActorCountDeadlineExpired(deadline=deadline, phase="pre_count", count_audit=audit)
    audit["deadline_limited"] = remaining <= float(timeout_seconds)
    audit["timeout_seconds"] = min(remaining, float(timeout_seconds))
    try:
        audit["child_started"] = None
        result = subprocess.run([PYTHON_PATH, "-I", "-B", "-c", CPU_CODE], input=payload,
                                capture_output=True, timeout=audit["timeout_seconds"], check=False, env=env)
    except subprocess.TimeoutExpired as error:
        audit["child_started"] = True
        _bytes_audit(audit, error.output, error.stderr)
        try:
            finished = _finish(audit)
        except Exception as clock_error:
            _failure("token_count_invalid_clock", audit, clock_error)
        try:
            _verify_protocol_source()
            if protocol._message_payload(messages)[1] != digest:
                raise ValueError("counter_request_binding_changed")
        except Exception as binding_error:
            _failure("token_count_source_binding_failed", audit, binding_error)
        if audit["deadline_limited"] and finished >= deadline:
            raise ActorCountDeadlineExpired(deadline=deadline, phase="counter_budget_timeout", count_audit=audit) from error
        _failure("token_count_transport_failed", audit, error)
    except Exception as error:
        try:
            _finish(audit)
        except Exception:
            pass
        _failure("token_count_transport_failed", audit, error)
    audit["child_started"] = True
    try:
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
        audit["child_returncode"] = code if type(code) is int else None
        _bytes_audit(audit, stdout, stderr)
        _finish(audit)
        if type(code) is not int or code != 0 or type(stdout) is not bytes or len(stdout) > MAX_RESPONSE_BYTES:
            raise ValueError("counter_child_transport_invalid")
    except Exception as error:
        _failure("token_count_transport_failed", audit, error)
    try:
        proof = json.loads(stdout.decode("utf-8"), object_pairs_hook=protocol._unique_pairs, parse_constant=_reject)
        if not _valid_proof(proof, digest) or protocol._message_payload(messages)[1] != digest:
            raise ValueError("counter_proof_invalid")
    except Exception as error:
        _failure("token_count_invalid_proof", audit, error)
    try:
        _verify_protocol_source()
    except Exception as error:
        _failure("token_count_source_binding_failed", audit, error)
    audit["proof_validated"] = True
    try:
        finished = _finish(audit)
    except Exception as error:
        _failure("token_count_invalid_clock", audit, error)
    if finished >= deadline:
        raise ActorCountDeadlineExpired(deadline=deadline, phase="validated_count_returned_late", count_audit=audit)
    return proof["input_tokens"]
