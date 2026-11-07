"""Private actor-visible transcripts for a caller-owned intervention lifecycle.

The caller must supply only actor-visible material, create the private parent
directory, and keep the raw world separately for terminal scoring and cleanup.
Trace and hash journals overlap; neither is an additional call or cost. This
module does not infer provenance from free text or enforce a study protocol.
"""

import json
import math
import os
import re
import stat
import threading
from pathlib import Path

_EVENT = re.compile(r'[a-z][a-z0-9_]{0,127}\Z')
_ERROR_TYPE = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,127}\Z')
_FORBIDDEN = frozenset({
    'ground_truth', 'groundtruth', 'evaluation', 'evaluator', 'evaluate', 'eval',
    'score', 'scores', 'scorer', 'assertion', 'assertions', 'evaluation_assertions',
    'test_assertions', 'verification_assertions', 'success',
})


class TraceWriteError(BaseException):
    """Fatal trace persistence/validation failure; the lifecycle owner must stop."""


def _forbidden_label(name):
    normalized = re.sub(r'[^a-z0-9]+', '_', name.casefold()).strip('_')
    return (normalized in _FORBIDDEN or normalized.startswith('ground_truth_')
            or normalized.startswith('evaluator_') or normalized.startswith('evaluation_')
            or normalized.startswith('eval_')
            or normalized.startswith('scoring_') or normalized.startswith('score_'))


def _plain_value(value, depth=0):
    if depth > 100:
        raise ValueError('trace nesting exceeds limit')
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return value
    if kind is float and math.isfinite(value):
        return value
    if kind is list:
        return [_plain_value(item, depth + 1) for item in value]
    if kind is dict:
        if any(type(key) is not str or _forbidden_label(key) for key in value):
            raise ValueError('invalid trace field')
        return {key: _plain_value(item, depth + 1) for key, item in value.items()}
    raise ValueError('non-JSON trace value')


def _parent_fd(path):
    raw = os.fspath(path)
    if type(raw) is not str:
        raise ValueError('absolute trace path required')
    parsed = Path(raw)
    if (not parsed.is_absolute() or str(parsed) != raw or '..' in parsed.parts
            or '\0' in raw or not parsed.name):
        raise ValueError('canonical absolute trace path required')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open('/', flags)
    try:
        for component in parsed.parent.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise PermissionError('trace parent must be owned and mode 0700')
        return fd, parsed
    except BaseException:
        os.close(fd)
        raise


class PrivateTrace:
    """New exclusive 0600 trace in an already-existing, private 0700 directory.

    The writer does not create or chmod the directory. Every line is fsynced;
    failed writes disable further writes and never authorize an operation retry.
    Errors never include the private record or underlying exception text.
    """

    def __init__(self, path):
        self._lock = threading.Lock()
        self._fd = None
        self._failed = False
        self._index = 0
        directory = None
        try:
            directory, self.path = _parent_fd(path)
            self._fd = os.open(self.path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                               | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
            os.fchmod(self._fd, 0o600)
            info = os.fstat(self._fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise PermissionError('trace file must be owned and mode 0600')
            os.fsync(self._fd)
            os.fsync(directory)
        except BaseException as exc:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
            if isinstance(exc, Exception):
                raise TraceWriteError('private trace creation failed') from None
            raise
        finally:
            if directory is not None:
                os.close(directory)

    def record(self, event, **fields):
        with self._lock:
            if self._fd is None or self._failed:
                raise TraceWriteError('private trace unavailable')
            try:
                if (type(event) is not str or _EVENT.fullmatch(event) is None
                        or _forbidden_label(event) or 'event' in fields or 'index' in fields):
                    raise ValueError('invalid trace event')
                safe = _plain_value(fields)
                record = {'index': self._index, 'event': event, **safe}
                raw = (json.dumps(record, ensure_ascii=True, allow_nan=False,
                                  separators=(',', ':')) + '\n').encode('ascii')
                remaining = memoryview(raw)
                while remaining:
                    count = os.write(self._fd, remaining)
                    if count <= 0:
                        raise OSError('private trace write made no progress')
                    remaining = remaining[count:]
                os.fsync(self._fd)
            except Exception:
                self._failed = True
                raise TraceWriteError('private trace write failed') from None
            except BaseException:
                self._failed = True
                raise
            self._index += 1

    def close(self):
        with self._lock:
            if self._fd is not None:
                fd, self._fd = self._fd, None
                try:
                    os.close(fd)
                except Exception:
                    self._failed = True
                    raise TraceWriteError('private trace close failed') from None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _exception_type(exc):
    name = type(exc).__name__
    return name if type(name) is str and _ERROR_TYPE.fullmatch(name) else 'OtherException'


def _invoke(trace, kind, operation, fields):
    trace.record(kind + '_attempt', **fields)
    try:
        result = operation()
    except TraceWriteError:
        raise
    except BaseException as exc:
        trace.record(kind + '_failed', error_type=_exception_type(exc))
        raise
    trace.record(kind + '_returned', result=result)
    return result


class TracedModel:
    def __init__(self, model, trace):
        self._model, self._trace = model, trace

    def generate(self, messages, seed):
        self._trace.record('generation_attempt', messages=messages, seed=seed)
        try:
            response = self._model.generate(messages, seed=seed)
        except TraceWriteError:
            raise
        except BaseException as exc:
            self._trace.record('generation_failed', error_type=_exception_type(exc))
            raise
        self._trace.record('generation_returned', response=response)
        return response


class TracedWorld:
    """Expose only actor-phase operations; scoring/cleanup remain with the owner."""

    def __init__(self, world, trace):
        self._world, self._trace = world, trace

    def execute(self, code):
        return _invoke(self._trace, 'execution', lambda: self._world.execute(code), {'code': code})

    def inspect(self, text):
        return _invoke(self._trace, 'inspection', lambda: self._world.inspect(text), {'text': text})

    def completed(self):
        return _invoke(self._trace, 'completed', self._world.completed, {})

    def api_count(self):
        return _invoke(self._trace, 'api_count', self._world.api_count, {})
