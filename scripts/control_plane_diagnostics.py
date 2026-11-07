"""Bounded, private-content-free diagnostics for a future control-plane caller.

This module performs no RPC, lease, retry, process or GPU action. A caller owns
the truth of the supplied phase/operation/reason and handles any write failure.
One object owns one new journal; it may be shared by threads, not reopened.
Pulse-response times describe local completed RPC responses only. They do not
establish remote heartbeat renewal, lease age, ownership or an active job.
"""
import json
import math
import os
import re
import stat
import subprocess
import threading
import time
from pathlib import Path

_PHASES = frozenset({"admission", "readiness", "client", "cleanup"})
_RPC_OPERATIONS = frozenset({"enqueue", "status", "pulse", "stop", "health"})
_OPERATIONS = _RPC_OPERATIONS | {"guard", "cleanup"}
_RPC_REASONS = frozenset({"rpc_started", "rpc_succeeded", "rpc_failed"})
_MESSAGES = {
    "rpc_started": None,
    "rpc_succeeded": None,
    "rpc_failed": None,
    "queue_ownership_or_monitor_lost": "Confirmed queue ownership or monitor lost.",
    "gpu_service_finished": "GPU service finished during client execution.",
    "ssh_tunnel_ended": "SSH tunnel ended.",
    "guard_failed": "Client or independent audit incomplete.",
    "cleanup_failed": None,
}
_KNOWN_ERRORS = (
    (subprocess.TimeoutExpired, "TimeoutExpired", "timeout"),
    (TimeoutError, "TimeoutError", "timeout"),
    (ConnectionResetError, "ConnectionResetError", "connection"),
    (ConnectionError, "ConnectionError", "connection"),
    (OSError, "OSError", "os"),
    (FileNotFoundError, "FileNotFoundError", "os"),
    (PermissionError, "PermissionError", "os"),
    (ValueError, "ValueError", "validation"),
    (TypeError, "TypeError", "validation"),
    (KeyError, "KeyError", "validation"),
    (RuntimeError, "RuntimeError", "runtime"),
    (UnicodeError, "UnicodeError", "validation"),
    (json.JSONDecodeError, "JSONDecodeError", "validation"),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_directory(path) -> Path:
    """Check an existing private directory without following any symlink.

    Parents may have system permissions; the final directory must be owned by
    the current UID with exact mode 0700. No permission or content is changed.
    This check returns a path, not a reservation against later path changes.
    """
    _require(type(path) is str or isinstance(path, Path), "directory path must be a string or Path")
    raw = str(path)
    directory = Path(raw)
    _require(directory.is_absolute() and directory.root == "/"
             and str(directory) == raw and ".." not in directory.parts and "\0" not in raw,
             "canonical absolute directory path required")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        for component in directory.parts[1:]:
            child_fd = os.open(component, flags, dir_fd=fd)
            previous_fd, fd = fd, child_fd
            os.close(previous_fd)
        info = os.fstat(fd)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise PermissionError("diagnostic directory requires current owner and exact mode 0700")
    finally:
        os.close(fd)
    return directory


def _time_value(value):
    _require(type(value) in (int, float), "invalid diagnostic time")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("invalid diagnostic time") from exc
    _require(math.isfinite(number) and number >= 0, "invalid diagnostic time")
    return number


def _identity(value):
    _require(type(value) is dict
             and set(value) == {"plan_sha256", "job_id", "pid", "start_ticks"},
             "complete diagnostic identity required")
    _require(type(value["plan_sha256"]) is str
             and re.fullmatch("[a-f0-9]{64}", value["plan_sha256"]) is not None,
             "invalid diagnostic plan identity")
    _require(type(value["job_id"]) is str
             and re.fullmatch("[a-z0-9][a-z0-9_-]{0,127}", value["job_id"]) is not None,
             "invalid diagnostic job identity")
    _require(type(value["pid"]) is int and 0 < value["pid"] < 2**31
             and type(value["start_ticks"]) is int and 0 < value["start_ticks"] < 2**63,
             "invalid diagnostic process identity")
    return dict(value)


def _error_fields(error):
    if error is None:
        return None, None, None
    _require(isinstance(error, BaseException), "diagnostic error must be an exception")
    if type(error) is subprocess.CalledProcessError:
        code = error.returncode
        _require(type(code) is int and -(2**31) <= code < 2**31 and code != 0,
                 "invalid subprocess return code")
        category = "subprocess_signal" if code < 0 else "subprocess_exit"
        return "CalledProcessError", category, code
    # Do not inspect an unknown class name, args, str/repr, command, output or stderr.
    for kind, name, category in _KNOWN_ERRORS:
        if type(error) is kind:
            return name, category, None
    return "OtherException", "other", None


class DiagnosticJournal:
    """Append fsynced JSON lines with explicit identity and bounded enum fields.

    The path must be absolute with an existing parent. Creation is exclusive,
    with verified effective mode 0600. A filesystem that cannot enforce this
    mode is rejected before any event is written. A write/fsync failure propagates
    and permanently disables append;
    partial or apparently complete unacknowledged bytes remain as evidence.
    """

    def __init__(self, path, *, identity, epoch_clock=time.time,
                 monotonic_clock=time.monotonic):
        self._identity = _identity(identity)
        _require(callable(epoch_clock) and callable(monotonic_clock),
                 "diagnostic clocks must be callable")
        path = Path(path)
        _require(path.is_absolute(), "absolute diagnostic path required")
        self._epoch_clock = epoch_clock
        self._monotonic_clock = monotonic_clock
        self._lock = threading.Lock()
        self._index = 0
        self._last_monotonic = None
        self._last_pulse_response = None
        self._failed = False
        self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT
                           | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(self._fd, 0o600)
            mode = os.fstat(self._fd).st_mode
            if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o600:
                raise PermissionError("diagnostic journal requires effective mode 0600")
            os.fsync(self._fd)
            parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except BaseException:
            fd, self._fd = self._fd, None
            os.close(fd)
            raise

    def record(self, *, phase, operation, reason_code, error=None,
               rpc_started_monotonic=None):
        """Capture local timing; rpc_succeeded does not certify a remote mutation.

        Clocks are sampled under the writer lock. RPC duration ends at this
        capture and can include caller handoff/lock delay. Pulse response times
        are not remote touch times; the caller must check owned/active/heartbeat
        independently and must never use this local interval as a lease age.
        """
        with self._lock:
            if self._fd is None:
                raise RuntimeError("diagnostic journal is closed")
            if self._failed:
                raise RuntimeError("diagnostic journal failed; append is disabled")
            _require(type(phase) is str and phase in _PHASES,
                     "unknown diagnostic phase")
            _require(type(operation) is str and operation in _OPERATIONS,
                     "unknown diagnostic operation")
            _require(type(reason_code) is str and reason_code in _MESSAGES,
                     "unknown diagnostic reason")
            is_rpc = operation in _RPC_OPERATIONS
            _require((reason_code in _RPC_REASONS) == is_rpc,
                     "diagnostic operation and reason differ")
            if operation == "guard":
                _require(reason_code != "cleanup_failed", "invalid guard reason")
            if operation == "cleanup":
                _require(reason_code == "cleanup_failed", "invalid cleanup reason")
            if reason_code in {"rpc_started", "rpc_succeeded"}:
                _require(error is None, "successful or started RPC cannot have an error")
            if reason_code == "rpc_failed":
                _require(error is not None, "failed RPC requires an exception")
            error_type, error_category, returncode = _error_fields(error)
            epoch = _time_value(self._epoch_clock())
            monotonic = _time_value(self._monotonic_clock())
            _require(self._last_monotonic is None or monotonic >= self._last_monotonic,
                     "diagnostic monotonic time moved backwards")
            duration = None
            if is_rpc:
                start = _time_value(rpc_started_monotonic)
                _require(start <= monotonic, "RPC start is after diagnostic event")
                duration = monotonic - start
            else:
                _require(rpc_started_monotonic is None, "non-RPC event has RPC timing")
            pulse_response = self._last_pulse_response
            if operation == "pulse" and reason_code == "rpc_succeeded":
                pulse_response = (epoch, monotonic)
            response_age = None if pulse_response is None else monotonic - pulse_response[1]
            event = {
                "schema_version": 1,
                "event_index": self._index,
                "identity": dict(self._identity),
                "phase": phase,
                "operation": operation,
                "reason_code": reason_code,
                "safe_message": _MESSAGES[reason_code],
                "exception_type": error_type,
                "error_category": error_category,
                "returncode": returncode,
                "epoch": epoch,
                "monotonic": monotonic,
                "rpc_duration_seconds": duration,
                "last_successful_pulse_response_epoch": None if pulse_response is None else pulse_response[0],
                "last_successful_pulse_response_monotonic": None if pulse_response is None else pulse_response[1],
                "seconds_since_last_successful_pulse_response": response_age,
            }
            raw = (json.dumps(event, ensure_ascii=True, sort_keys=True,
                              separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
            try:
                remaining = memoryview(raw)
                while remaining:
                    count = os.write(self._fd, remaining)
                    if count <= 0:
                        raise OSError("diagnostic write made no progress")
                    remaining = remaining[count:]
                os.fsync(self._fd)
            except BaseException:
                self._failed = True
                raise
            self._last_monotonic = monotonic
            self._last_pulse_response = pulse_response
            self._index += 1
            return event

    def close(self):
        with self._lock:
            if self._fd is not None:
                fd, self._fd = self._fd, None
                os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
