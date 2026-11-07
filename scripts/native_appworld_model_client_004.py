"""Future generation admission over the immutable 003 client and CPU counter.

The 120-second window is a policy choice, not measured model performance. Only
this trusted pretransport boundary can issue a no-request signal. Admission is
checked after serialization and immediately before the HTTP child/test opener;
it cannot make OS scheduling after that check atomic. The future actor owns its
separate journal reserve. No existing study is registered or reclassified here.
"""

import ast
import hashlib
import math
import re
import textwrap
from pathlib import Path

from scripts import native_appworld_model_client as original
from scripts import native_appworld_model_client_003 as previous
from scripts import native_verification_token_counter_003 as counter

NativeModelError = original.NativeModelError
SOURCE_REVISION = 4
REQUEST_WINDOW_SECONDS = 120.0
BASE_SOURCE_SHA256 = "cada4da0696a4f7b4e138edd54beb0bb62fd5db0e439a6b0586d14cb404e0344"
CLIENT_003_SHA256 = "fb40793026fa8d6e15dcaf9b6d37eef107c13599260fb12e6166e341946c8630"
COUNTER_003_SHA256 = "fa3ca418366d2e8d5cbe9b3e72117ee05113725ba5452f214d9d033ff6246c65"
GENERATION_AUDIT_FIELDS = (
    "source_revision", "deadline", "observed_monotonic", "request_window_seconds",
    "request_sha256", "messages_sha256", "seed", "transport_started",
)
_ISSUER = object()
_REQUEST = None


def _verify_sources():
    if (counter._source_digest(original.__file__) != BASE_SOURCE_SHA256
            or counter._source_digest(previous.__file__) != CLIENT_003_SHA256
            or counter._source_digest(counter.__file__) != COUNTER_003_SHA256):
        raise ValueError("native_model_004_source_binding")
    previous._verify_sources()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def generation_messages_sha256(messages):
    """Use the same exact message fingerprint as the authenticated CPU counter."""
    try:
        return original._message_payload(messages)[1]
    except Exception:
        raise NativeModelError("native_model_generation_admission_invalid") from None


def _generation_body(messages, seed, model_id, max_tokens, temperature):
    if (type(seed) is not int or not 0 <= seed < 2**31
            or type(model_id) is not str or not 1 <= len(model_id) <= 128
            or any(ord(c) <= 32 or ord(c) == 127 for c in model_id)
            or type(max_tokens) is not int or not 1 <= max_tokens <= original.MAX_OUTPUT
            or not _finite(temperature) or not 0 <= temperature <= 2):
        raise ValueError("generation_admission_binding")
    payload, digest = original._message_payload(messages)
    snapshot = original._json(payload)["messages"]
    return {"model": model_id, "messages": snapshot, "seed": seed,
            "max_tokens": max_tokens, "temperature": float(temperature)}, digest


def generation_request_sha256(messages, seed, *, model_id, max_tokens, temperature):
    """Fingerprint a complete generation request without retaining its body."""
    try:
        body, _ = _generation_body(messages, seed, model_id, max_tokens, temperature)
        return hashlib.sha256(original._encoded(body)).hexdigest()
    except Exception:
        raise NativeModelError("native_model_generation_admission_invalid") from None


def validate_generation_audit(audit):
    """Copy fixed-shape, content-free refusal metadata or raise a generic error."""
    try:
        if type(audit) is not dict or set(audit) != set(GENERATION_AUDIT_FIELDS):
            raise ValueError
        copied = dict(audit)
        if (type(copied["source_revision"]) is not int or copied["source_revision"] != SOURCE_REVISION
                or copied["transport_started"] is not False
                or type(copied["request_window_seconds"]) is not float
                or copied["request_window_seconds"] != REQUEST_WINDOW_SECONDS
                or type(copied["deadline"]) is not float or not _finite(copied["deadline"])
                or copied["deadline"] <= 0
                or type(copied["observed_monotonic"]) is not float
                or not _finite(copied["observed_monotonic"]) or copied["observed_monotonic"] < 0
                or copied["deadline"] - copied["observed_monotonic"] >= REQUEST_WINDOW_SECONDS
                or not _hash(copied["request_sha256"]) or not _hash(copied["messages_sha256"])
                or type(copied["seed"]) is not int or not 0 <= copied["seed"] < 2**31):
            raise ValueError
        return copied
    except Exception:
        raise NativeModelError("native_model_generation_admission_invalid") from None


class NativeGenerationNotStarted(NativeModelError):
    """Exact trusted signal: a valid request was refused before HTTP launch."""

    def __init__(self, *, generation_audit, _issuer=None):
        if _issuer is not _ISSUER:
            raise NativeModelError("native_model_generation_admission_invalid") from None
        self.generation_audit = validate_generation_audit(generation_audit)
        self._issued_binding = tuple(self.generation_audit[k] for k in GENERATION_AUDIT_FIELDS)
        self._issuer = _issuer
        super().__init__("native_model_generation_not_started")


def validate_generation_not_started(error, *, deadline, messages_sha256, seed):
    """Bind a genuine refusal to the future actor's pending generation.

    The actor must additionally compare observed_monotonic with its own current
    clock and pending-intent time, and bind the deadline to its reserved end.
    """
    try:
        _verify_sources()
        if type(error) is not NativeGenerationNotStarted or error._issuer is not _ISSUER:
            raise ValueError
        audit = validate_generation_audit(error.generation_audit)
        if (tuple(audit[k] for k in GENERATION_AUDIT_FIELDS) != error._issued_binding
                or not _finite(deadline) or deadline <= 0 or audit["deadline"] != float(deadline)
                or not _hash(messages_sha256) or audit["messages_sha256"] != messages_sha256
                or type(seed) is not int or not 0 <= seed < 2**31 or audit["seed"] != seed):
            raise ValueError
        return audit
    except Exception:
        raise NativeModelError("native_model_generation_admission_invalid") from None


def _admit_generation_transport(model, path, raw):
    try:
        _verify_sources()
        if path != "/v1/chat/completions":
            return
        body = original._json(raw)
        if type(body) is not dict or set(body) != {"model", "messages", "seed", "max_tokens", "temperature"}:
            raise ValueError("generation_admission_binding")
        expected, messages_digest = _generation_body(
            body["messages"], body["seed"], model.model_id, model.max_tokens, model.temperature)
        if body != expected or original._encoded(expected) != raw:
            raise ValueError("generation_admission_binding")
        if (not _finite(model.deadline) or model.deadline <= 0
                or not _finite(model.timeout) or not 0 < model.timeout <= REQUEST_WINDOW_SECONDS):
            raise ValueError("generation_admission_limits")
        request_digest = hashlib.sha256(raw).hexdigest()
        observed = previous._now(model._clock)
        audit = {"source_revision": SOURCE_REVISION, "deadline": float(model.deadline),
                 "observed_monotonic": observed, "request_window_seconds": REQUEST_WINDOW_SECONDS,
                 "request_sha256": request_digest,
                 "messages_sha256": messages_digest, "seed": body["seed"], "transport_started": False}
    except Exception:
        raise NativeModelError("native_model_generation_admission_invalid") from None
    if audit["deadline"] - observed < REQUEST_WINDOW_SECONDS:
        raise NativeGenerationNotStarted(generation_audit=audit, _issuer=_ISSUER)


def _isolated_request(*, raw=None):
    """Authenticate and patch only the original request method, in isolation."""
    try:
        if raw is None:
            from scripts.native_source_registration import public_read

            raw = public_read(Path(__file__).resolve().parents[1], "scripts/native_appworld_model_client.py")
        if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SOURCE_SHA256:
            raise ValueError("source")
        source = raw.decode("utf-8")
        tree = ast.parse(source)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeModel"]
        methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "_request"]
        if len(classes) != 1 or len(methods) != 1:
            raise ValueError("source")
        node = methods[0]
        method = textwrap.dedent("".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]))
        before = (
            '        operation_remaining()\n'
            '        raw = None if body is None else _encoded(body)\n'
            '        if raw is not None and len(raw) > 2_000_000:\n'
            '            raise ValueError("request_size")\n'
            '        timeout = operation_remaining()\n'
            '        if self._opener is not None:\n'
            '            request = urllib.request.Request(self._base + path, data=raw,\n'
            '                                            headers={"Content-Type": "application/json"})\n'
            '            with self._opener.open(request, timeout=timeout) as response:\n'
        )
        after = before.replace('        operation_remaining()\n', '', 1).replace(
            '        timeout = operation_remaining()\n',
            '        _admit_generation_transport(self, path, raw)\n'
            '        timeout = operation_remaining()\n', 1).replace(
                '            with self._opener.open(request, timeout=timeout) as response:\n',
                '            _admit_generation_transport(self, path, raw)\n'
                '            with self._opener.open(request, timeout=timeout) as response:\n', 1)
        launch = '            result = subprocess.run(\n'
        if method.count(before) != 1 or method.count(launch) != 1:
            raise ValueError("source")
        method = method.replace(before, after, 1).replace(
            launch, '            _admit_generation_transport(self, path, raw)\n' + launch, 1)
        names = ("math", "_encoded", "urllib", "MAX_RESPONSE_BYTES", "base64", "subprocess",
                 "sys", "_HTTP_CHILD", "TMPDIR", "_json")
        namespace = {name: getattr(original, name) for name in names}
        namespace.update(NativeModelError=NativeModelError,
                         _admit_generation_transport=_admit_generation_transport)
        exec(compile(method, str(original.__file__), "exec", dont_inherit=True), namespace)
        return namespace["_request"]
    except Exception:
        raise NativeModelError("native_model_004_source_binding_failed") from None


def _get_request():
    global _REQUEST
    if _REQUEST is None:
        _REQUEST = _isolated_request()
    return _REQUEST


class NativeModel(previous.NativeModel):
    def __init__(self, *args, **kwargs):
        try:
            _verify_sources()
        except Exception:
            raise NativeModelError("native_model_004_source_binding_failed") from None
        super().__init__(*args, **kwargs)
        self.timeout = min(float(self.timeout), REQUEST_WINDOW_SECONDS)

    def _request(self, path, body=None):
        return _get_request()(self, path, body)
