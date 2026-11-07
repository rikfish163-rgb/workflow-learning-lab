"""Content-free request diagnostics over the authenticated native worker client.

The original worker launch, runtime profile, reply checks, save, close and abort
remain inherited. Only the trusted parent's request function is instrumented in
isolated globals. Diagnostics are private and never contain request, response or
exception bodies. Import starts no worker, model, task or accelerator operation.
"""

import ast
import hashlib
import importlib
import importlib.util
import os
import selectors
import time
from pathlib import Path

from scripts.native_source_registration import public_read

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / 'scripts/native_appworld_worker_client.py'
BASE_SHA256 = '16eac615ae5d9f2e143bb86b43ade4435ad0114e3bb626c3a1c90533e7d978dc'
_ERROR = 'native_worker_client_006_source_invalid'

_base_bytes = public_read(ROOT, 'scripts/native_appworld_worker_client.py')
if hashlib.sha256(_base_bytes).hexdigest() != BASE_SHA256:
    raise ValueError(_ERROR)
original = importlib.import_module('scripts.native_appworld_worker_client')
NativeWorkerClientError = original.NativeWorkerClientError


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(_ERROR)
    return source.replace(before, after, 1)


def _received(audit, data):
    audit['received_byte_count'] += len(data)
    audit['digest'].update(data)


def _no_pending(client, audit, fail):
    try:
        value = os.read(client._process.stdout.fileno(), 1)
    except BlockingIOError:
        return
    except Exception:
        audit['code'] = 'pending_read_failed'
        raise
    _received(audit, value)
    audit['code'] = 'pending_data' if value else 'pending_eof'
    fail()


def _read_line(fd, deadline, audit, wait_fd, fail, limit):
    """Match the original bounded reader while retaining only bytes/hash metadata."""
    data = bytearray()
    while True:
        try:
            wait_fd(fd, selectors.EVENT_READ, deadline)
        except Exception:
            audit['code'] = 'read_deadline' if time.monotonic() >= deadline else 'read_wait_failed'
            raise
        try:
            chunk = os.read(fd, min(65536, limit + 1 - len(data)))
        except BlockingIOError:
            continue
        except Exception:
            audit['code'] = 'read_failed'
            raise
        if not chunk:
            audit['code'] = 'read_eof'
            fail()
        data.extend(chunk)
        _received(audit, chunk)
        if len(data) > limit:
            audit['code'] = 'read_frame_too_large'
            fail()
        if b'\n' in chunk:
            if data.find(b'\n') != len(data) - 1:
                audit['code'] = 'read_invalid_framing'
                fail()
            return bytes(data)


def _error_class(error):
    # A fixed vocabulary avoids copying arbitrary error class/message content.
    if isinstance(error, original.NativeWorkerClientError):
        return 'NativeWorkerClientError'
    for kind in (BrokenPipeError, TimeoutError, OSError, ValueError, TypeError, AttributeError):
        if isinstance(error, kind):
            return kind.__name__
    return 'Other'


def _failure(client, audit, error, started, deadline):
    """Capture before abort; a failed diagnostic write cannot make cost known."""
    try:
        now = time.monotonic()
        expired = now >= deadline
        stage = audit['stage']
        fallback = {
            'no_pending': 'pending_failed',
            'write': 'write_deadline' if expired else 'write_failed',
            'read': 'read_deadline' if expired else 'read_failed',
            'decode': 'decode_invalid',
            'validate': 'reply_invalid',
            'log': 'post_result_failed' if audit['journal_phase'] == 'after_result' else 'journal_failed',
        }
        event = {'event': 'request_failure_diagnostic', 'stage': stage,
                 'code': audit['code'] or fallback[stage], 'error_class': _error_class(error),
                 'elapsed_seconds': max(0.0, now - started), 'deadline_expired': expired,
                 'received_byte_count': audit['received_byte_count'],
                 'frame_sha256': audit['digest'].hexdigest(), 'journal_phase': audit['journal_phase']}
        client._last_failure_diagnostic = event
        try:
            client._record(event)
        except Exception:
            pass
    except Exception:
        # Diagnostics are best effort; the original request still fails closed.
        pass


def _isolated(*, raw=None):
    raw = public_read(ROOT, 'scripts/native_appworld_worker_client.py') if raw is None else raw
    if (type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256
            or Path(original.__file__).resolve() != BASE_SOURCE.resolve()):
        raise ValueError(_ERROR)
    source = raw.decode('utf-8')
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NativeWorkerClient']
    if len(classes) != 1:
        raise ValueError(_ERROR)
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == 'request']
    if len(methods) != 1 or methods[0].decorator_list:
        raise ValueError(_ERROR)
    request = ast.get_source_segment(source, methods[0])
    if type(request) is not str:
        raise ValueError(_ERROR)
    request = _replace_once(request, 'def request(', 'def _request_006(')
    request = _replace_once(
        request, '        try:\n            self._no_pending()\n'
        '            self._record({"event": "request_intent", "request": request})\n'
        '            _write_all(self._process.stdin.fileno(), data, deadline)\n'
        '            response = _decode(_read_line(self._process.stdout.fileno(), deadline))\n'
        '            self._validate_reply(response, op)\n'
        '            self._record({"event": "request_result", "reply": response})\n',
        '        started = time.monotonic()\n'
        '        audit = {"stage": "no_pending", "code": None, "journal_phase": None,\n'
        '                 "received_byte_count": 0, "digest": hashlib.sha256()}\n'
        '        try:\n'
        '            _audit_no_pending(self, audit, _fail)\n'
        '            audit.update(stage="log", journal_phase="intent")\n'
        '            self._record({"event": "request_intent", "request": request})\n'
        '            audit.update(stage="write", journal_phase=None)\n'
        '            _write_all(self._process.stdin.fileno(), data, deadline)\n'
        '            audit["stage"] = "read"\n'
        '            frame = _audit_read_line(self._process.stdout.fileno(), deadline, audit,\n'
        '                                     _wait_fd, _fail, MAX_RPC_BYTES)\n'
        '            audit["stage"] = "decode"\n'
        '            response = _decode(frame)\n'
        '            audit["stage"] = "validate"\n'
        '            self._validate_reply(response, op)\n'
        '            audit.update(stage="log", journal_phase="result")\n'
        '            self._record({"event": "request_result", "reply": response})\n'
        '            audit["journal_phase"] = "after_result"\n')
    request = _replace_once(
        request, '        except Exception:\n            self._failed = True\n            self.abort()\n            _fail()\n',
        '        except Exception as error:\n            self._failed = True\n'
        '            _audit_failure(self, audit, error, started, deadline)\n'
        '            self.abort()\n            _fail()\n')
    spec = importlib.util.spec_from_file_location('scripts._native_appworld_worker_client_006_core', BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError(_ERROR)
    core = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), 'exec', dont_inherit=True), core.__dict__)
    core.NativeWorkerClientError = original.NativeWorkerClientError
    core.__dict__.update(_audit_no_pending=_no_pending, _audit_read_line=_read_line, _audit_failure=_failure)
    exec(compile(request, str(BASE_SOURCE), 'exec', dont_inherit=True), core.__dict__)
    return core


_CORE = _isolated()


class NativeWorkerClient(original.NativeWorkerClient):
    """Original client plus content-free diagnostics for admitted RPC failures."""

    request = _CORE._request_006

    @property
    def last_failure_diagnostic(self):
        value = getattr(self, '_last_failure_diagnostic', None)
        return dict(value) if value is not None else None
