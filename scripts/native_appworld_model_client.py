"""Source-bound loopback model client for the remote native parent controller.

No model or CUDA is loaded here. Counts, requests and replies stay private.
Each generation consumes one exact CPU count; failed calls are never retried.
The owning controller must durably journal generation intent before calling it.
"""

import base64
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit

from scripts.native_verification_token_counter import TMPDIR, count_input_tokens
from scripts.verification_token_counter import _message_payload

MAX_RESPONSE_BYTES = 4_000_000
MAX_CONTEXT = 16384
MAX_OUTPUT = 1024

# This fixed stdlib-only child never receives Python from the policy. A separate
# process lets the parent enforce one wall deadline even if HTTP headers or a
# response body arrive slowly enough to evade a per-socket inactivity timeout.
_HTTP_CHILD = r'''
import base64,json,sys,urllib.request
from urllib.parse import urlsplit

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None

try:
    spec = json.loads(sys.stdin.buffer.read(2700001))
    if type(spec) is not dict or set(spec) != {"url", "body", "timeout"}:
        raise ValueError()
    parsed = urlsplit(spec["url"])
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in {"/health", "/v1/chat/completions"}
            or parsed.port is None or not 1 <= parsed.port <= 65535):
        raise ValueError()
    data = None if spec["body"] is None else base64.b64decode(spec["body"], validate=True)
    if data is not None and len(data) > 2000000:
        raise ValueError()
    request = urllib.request.Request(spec["url"], data=data, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=spec["timeout"]) as response:
        if type(response.status) is not int or response.status != 200:
            raise ValueError()
        raw = response.read(4000001)
        if len(raw) > 4000000:
            raise ValueError()
    sys.stdout.buffer.write(raw)
except Exception:
    sys.exit(1)
'''


class NativeModelError(RuntimeError):
    """A fixed content-free configuration, binding or transport failure."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _reject(_value):
    raise ValueError("nonfinite_number")


def _values(value, depth=0):
    if depth > 32:
        raise ValueError("JSON_depth")
    if type(value) is str:
        value.encode("utf-8")
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite_number")
    elif type(value) in (dict, list):
        if type(value) is dict:
            for key in value:
                if type(key) is not str:
                    raise ValueError("JSON_key")
                key.encode("utf-8")
            value = value.values()
        for child in value:
            _values(child, depth + 1)
    elif value is not None and type(value) not in (int, bool):
        raise ValueError("JSON_type")


def _json(raw):
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject)
    _values(value)
    return value


def _encoded(value):
    _values(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _base_url(endpoint):
    if type(endpoint) is not str or any(ord(c) <= 32 or ord(c) == 127 for c in endpoint):
        raise ValueError("endpoint")
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in {"", "/", "/v1", "/v1/"}
            or parsed.port is None or not 1 <= parsed.port <= 65535):
        raise ValueError("endpoint")
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}:{parsed.port}"


class NativeModel:
    def __init__(self, endpoint, model_id, *, expected_backend, max_tokens,
                 context_tokens, temperature, timeout, deadline,
                 _token_counter=None, _opener=None, _clock=time.monotonic):
        try:
            if socket.gethostname() != "amax" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise ValueError("host_or_CUDA")
            self._base = _base_url(endpoint)
            if (type(model_id) is not str or not 1 <= len(model_id) <= 128
                    or any(ord(c) <= 32 or ord(c) == 127 for c in model_id)
                    or type(expected_backend) is not dict or not expected_backend
                    or type(max_tokens) is not int or not 1 <= max_tokens <= MAX_OUTPUT
                    or type(context_tokens) is not int or not max_tokens <= context_tokens <= MAX_CONTEXT
                    or type(temperature) not in (int, float) or not math.isfinite(temperature)
                    or not 0 <= temperature <= 2
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                           for x in (timeout, deadline))):
                raise ValueError("configuration")
            self._expected_backend = _json(_encoded(expected_backend))
            self._backend_digest = _digest(self._expected_backend)
        except Exception:
            raise NativeModelError("native_model_invalid_configuration") from None
        self.model_id, self.max_tokens = model_id, max_tokens
        self.context_tokens, self.temperature = context_tokens, float(temperature)
        self.timeout, self.deadline = timeout, deadline
        self._counter, self._clock = _token_counter or count_input_tokens, _clock
        self._default_counter = _token_counter is None
        self._opener = _opener  # Test-only injection; production uses the bounded child.
        self._counted = None
        self._health_verified = False

    def _time_left(self):
        remaining = self.deadline - self._clock()
        if remaining <= 0:
            raise NativeModelError("native_model_deadline")
        return remaining

    def _request(self, path, body=None):
        try:
            started = self._clock()
            operation_deadline = min(self.deadline, started + self.timeout)

            def operation_remaining():
                remaining = operation_deadline - self._clock()
                if not math.isfinite(remaining) or remaining <= 0:
                    raise NativeModelError("native_model_deadline")
                return remaining

            operation_remaining()
            raw = None if body is None else _encoded(body)
            if raw is not None and len(raw) > 2_000_000:
                raise ValueError("request_size")
            timeout = operation_remaining()
            if self._opener is not None:
                request = urllib.request.Request(self._base + path, data=raw,
                                                headers={"Content-Type": "application/json"})
                with self._opener.open(request, timeout=timeout) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if type(response.status) is not int or response.status != 200 or len(raw) > MAX_RESPONSE_BYTES:
                        raise ValueError("response_status_or_size")
            else:
                payload = _encoded({"url": self._base + path,
                                    "body": None if raw is None else base64.b64encode(raw).decode("ascii"),
                                    "timeout": timeout})
                result = subprocess.run(
                    [sys.executable, "-I", "-B", "-c", _HTTP_CHILD], input=payload,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=operation_remaining(),
                    check=False, env={"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                                      "TMPDIR": TMPDIR, "LANG": "C.UTF-8"},
                )
                if result.returncode != 0 or len(result.stdout) > MAX_RESPONSE_BYTES:
                    raise ValueError("response_status_or_size")
                raw = result.stdout
            operation_remaining()
            reply = _json(raw)
            operation_remaining()
            return reply
        except NativeModelError:
            raise
        except Exception:
            raise NativeModelError("native_model_transport_or_JSON_failed") from None

    def check_health(self):
        self._health_verified = False
        reply = self._request("/health")
        try:
            if (type(reply) is not dict or reply.get("model") != self.model_id
                    or reply.get("ready") is not True or reply.get("busy") is not False
                    or type(reply.get("max_context")) is not int
                    or reply["max_context"] != self.context_tokens
                    or type(reply.get("max_output_tokens")) is not int
                    or not self.max_tokens <= reply["max_output_tokens"] <= MAX_OUTPUT
                    or _digest(reply.get("backend")) != self._backend_digest):
                raise ValueError("health_binding")
        except Exception:
            raise NativeModelError("native_model_health_binding_failed") from None
        self._health_verified = True
        return reply

    def count_input_tokens(self, messages):
        self._counted = None
        try:
            self._time_left()
            payload, digest = _message_payload(messages)
            snapshot = _json(payload)["messages"]
            if self._default_counter:
                count = self._counter(snapshot, timeout_seconds=self._time_left())
            else:
                count = self._counter(snapshot)
            self._time_left()
            _, after = _message_payload(snapshot)
            if type(count) is not int or count < 1 or after != digest:
                raise ValueError("count")
            self._counted = (digest, count)
            return count
        except NativeModelError:
            raise
        except Exception:
            raise NativeModelError("native_model_CPU_count_failed") from None

    def generate(self, messages, seed):
        counted, self._counted = self._counted, None
        try:
            payload, digest = _message_payload(messages)
            if (not self._health_verified or counted is None or counted[0] != digest
                    or type(seed) is not int or not 0 <= seed < 2**31
                    or counted[1] + self.max_tokens > self.context_tokens):
                raise ValueError("generation_binding")
            snapshot = _json(payload)["messages"]
        except Exception:
            raise NativeModelError("native_model_generation_binding_failed") from None
        reply = self._request("/v1/chat/completions", {
            "model": self.model_id, "messages": snapshot, "seed": seed,
            "max_tokens": self.max_tokens, "temperature": self.temperature})
        try:
            if (type(reply) is not dict or reply.get("model") != self.model_id
                    or reply.get("format_error") is not None
                    or _digest(reply.get("backend")) != self._backend_digest
                    or type(reply.get("choices")) is not list or len(reply["choices"]) != 1):
                raise ValueError("generation_reply")
            choice, usage = reply["choices"][0], reply["usage"]
            message = choice["message"]
            if (type(message) is not dict or message.get("role") != "assistant"
                    or type(message.get("content")) is not str
                    or choice.get("finish_reason") not in {"stop", "length", "unknown"}
                    or type(usage) is not dict or set(usage) != {"prompt_tokens", "completion_tokens"}
                    or type(usage["prompt_tokens"]) is not int or usage["prompt_tokens"] != counted[1]
                    or type(usage["completion_tokens"]) is not int
                    or not 1 <= usage["completion_tokens"] <= self.max_tokens):
                raise ValueError("usage_or_choice")
            return {"text": message["content"], "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"], "format_error": None,
                    "backend": reply["backend"], "raw_reply": reply}
        except Exception:
            raise NativeModelError("native_model_invalid_response") from None
