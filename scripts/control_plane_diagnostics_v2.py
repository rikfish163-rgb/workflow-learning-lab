"""Versioned journal retaining only safe numeric write-failure metadata.

The frozen journal still owns validation, event schema, fsync and fail-closed
semantics. This adapter serializes callers so the first write failure is
captured before a later caller observes the disabled journal. It stores neither
the exception object nor its text, arguments, path, traceback or class name.
"""

import threading
import time

from scripts.control_plane_diagnostics import DiagnosticJournal as _DiagnosticJournal
from scripts.control_plane_diagnostics import validate_directory

__all__ = ['DiagnosticJournal', 'safe_oserror_errno', 'validate_directory']

_SAFE_OS_ERRORS = (
    OSError, BlockingIOError, ChildProcessError, ConnectionError, BrokenPipeError,
    ConnectionAbortedError, ConnectionRefusedError, ConnectionResetError, FileExistsError,
    FileNotFoundError, InterruptedError, IsADirectoryError, NotADirectoryError,
    PermissionError, ProcessLookupError, TimeoutError,
)


def safe_oserror_errno(error):
    """Extract a bounded integer only from an exact builtin OS exception type.

    This is also usable by a caller handling a constructor failure, before a
    journal instance exists. It never invokes a custom exception attribute hook.
    """
    if not any(type(error) is kind for kind in _SAFE_OS_ERRORS):
        return None
    value = OSError.errno.__get__(error, OSError)
    return value if type(value) is int and 0 <= value < 2**31 else None


class DiagnosticJournal(_DiagnosticJournal):
    """Preserve the original API and add a thread-safe write_failure snapshot."""

    def __init__(self, path, *, identity, epoch_clock=time.time, monotonic_clock=time.monotonic):
        self._capture_lock = threading.Lock()
        self._write_failure_seen = False
        self._write_failure_errno = None
        super().__init__(path, identity=identity, epoch_clock=epoch_clock, monotonic_clock=monotonic_clock)

    @property
    def write_failure(self):
        """Return only whether a write/fsync failed and its known errno, if any."""
        with self._capture_lock:
            return {'failed': self._write_failure_seen, 'errno': self._write_failure_errno}

    def record(self, *, phase, operation, reason_code, error=None, rpc_started_monotonic=None):
        with self._capture_lock:
            try:
                return super().record(phase=phase, operation=operation, reason_code=reason_code,
                                      error=error, rpc_started_monotonic=rpc_started_monotonic)
            except BaseException as exc:
                if self._failed and not self._write_failure_seen:
                    self._write_failure_seen = True
                    self._write_failure_errno = safe_oserror_errno(exc)
                raise
