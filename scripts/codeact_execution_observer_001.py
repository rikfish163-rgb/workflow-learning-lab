"""Observe a trusted world's cell-result boundary without inspecting cell output.

The caller installs this observer in the worker, brackets each ``execute`` with
``begin``/``finish``, and restores it before releasing the world. This module
does not execute cells or import AppWorld, a model, or its optional dependencies.
"""

import inspect
import threading
from types import MethodType


class ObserverError(RuntimeError):
    """The hook cannot be installed, verified, or safely restored."""


_ERROR_TYPES = {kind: kind.__name__ for kind in (
    Exception, ArithmeticError, AssertionError, AttributeError, ImportError,
    IndexError, KeyError, LookupError, NameError, OverflowError, RuntimeError,
    SyntaxError, TypeError, UnboundLocalError, ValueError, ZeroDivisionError,
)}


def _timeout(error):
    # AppWorld's timeout is a different class from builtins.TimeoutError, and
    # may be captured by IPython in error_in_exec. A same-named user exception
    # is conservatively non-recoverable as well.
    try:
        return isinstance(error, TimeoutError) or any(
            base.__name__ == "TimeoutError" for base in type(error).__mro__)
    except BaseException:
        return False


def _classify(value):
    if value is None:
        return "timeout", None
    try:
        before, during, success = value.error_before_exec, value.error_in_exec, value.success
        if type(success) is not bool or (before is not None and during is not None):
            return "unknown", None
        error = before if before is not None else during
        if success != (error is None):
            return "unknown", None
        if success:
            return "success", None
        if not isinstance(error, Exception) or isinstance(error, BaseExceptionGroup):
            return "unknown", None
        if _timeout(error):
            return "timeout", "TimeoutError"
        return ("error_before_exec" if before is not None else "error_in_exec"), _ERROR_TYPES.get(type(error), "other")
    except BaseException:
        # A malformed or unusual result must not alter the real cell outcome.
        return "unknown", None


class ExecutionObserver:
    def __init__(self, world, *, expected_function=None):
        self._world = world
        self._expected_function = expected_function
        self._lock = threading.RLock()
        self._installed = self._active = self._poisoned = False
        self._inflight = self._calls = self._serial = 0
        self._status = ("unknown", None)
        self._original = self._original_func = self._hook = None

    def verify(self):
        with self._lock:
            try:
                return (self._installed
                        and vars(self._world).get("_shell_run_cell") is self._hook
                        and inspect.getattr_static(type(self._world), "_shell_run_cell") is self._original_func)
            except Exception:
                return False

    def install(self):
        with self._lock:
            if self._installed:
                raise ObserverError("observer_install_failed")
            self._hook = None
            try:
                if "_shell_run_cell" in vars(self._world):
                    raise ValueError
                original = self._world._shell_run_cell
                source = inspect.getattr_static(type(self._world), "_shell_run_cell")
                if (not isinstance(original, MethodType) or original.__self__ is not self._world
                        or original.__func__ is not source
                        or (self._expected_function is not None and source is not self._expected_function)):
                    raise ValueError

                def hook(_world, code):
                    return self._capture(code)

                self._original, self._original_func = original, source
                self._hook = MethodType(hook, self._world)
                self._world._shell_run_cell = self._hook
                self._installed = True
                if not self.verify():
                    raise ValueError
            except Exception:
                if self._hook is not None and vars(self._world).get("_shell_run_cell") is self._hook:
                    del self._world._shell_run_cell
                self._installed = False
                raise ObserverError("observer_install_failed") from None

    def begin(self):
        with self._lock:
            if not self.verify() or self._active or self._inflight or self._poisoned:
                raise ObserverError("observer_begin_failed")
            self._serial += 1
            self._calls, self._status, self._active = 0, ("unknown", None), True

    def _capture(self, code):
        with self._lock:
            active, serial = self._active, self._serial
            if active:
                self._calls += 1
            else:
                self._poisoned = True
            self._inflight += 1
        status = ("unknown", None)
        try:
            try:
                value = self._original(code)
            except BaseException as error:
                if _timeout(error):
                    status = ("timeout", "TimeoutError")
                raise
            status = _classify(value)
            return value
        finally:
            with self._lock:
                self._inflight -= 1
                if active and self._active and self._serial == serial and self._calls == 1:
                    self._status = status

    def finish(self):
        with self._lock:
            if not self._active:
                raise ObserverError("observer_finish_failed")
            self._active = False
            status = ("unknown", None)
            if self.verify() and not self._poisoned and not self._inflight:
                if self._calls == 0:
                    status = ("unobserved", None)
                elif self._calls == 1:
                    status = self._status
            return {"status": status[0], "error_type": status[1], "hook_calls": self._calls}

    def restore(self):
        with self._lock:
            if not self._installed or self._active or self._inflight:
                raise ObserverError("observer_restore_failed")
            intact = self.verify()
            if vars(self._world).get("_shell_run_cell") is self._hook:
                del self._world._shell_run_cell
            self._installed = False
            if not intact:
                raise ObserverError("observer_restore_failed")
