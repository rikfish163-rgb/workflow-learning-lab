"""Exact CPU token counts using a service's already-loaded tokenizer.

Only the new route is implemented here. Existing health/generation handlers and
the engine's generation counter remain owned by the frozen service. Counting
time is part of the current service/job wall time, not an additional GPU job.
"""

import itertools
import json
import math
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from scripts import serve_qwen38 as frozen
from wll.protocol import fingerprint

PATH = '/v1/token-count'
MAX_REQUEST_BYTES = 2_000_000
MAX_RESPONSE_BYTES = 32768
TIMEOUT_SECONDS = 30
TOKENIZER_SEMANTICS = 'chat_template_no_thinking_generation_prompt_cpu_no_special_tokens_v1'
_REPLY_FIELDS = {'model', 'messages_sha256', 'input_tokens', 'backend_sha256', 'tokenizer_semantics'}
_ERROR_STATUS = {'invalid_request_body': 400, 'invalid_messages': 400,
                 'wrong_model': 400, 'busy': 409, 'model_unavailable': 503}


class TokenCountError(RuntimeError):
    """Content-free local validation, transport or count-proof failure; no retry."""


class _Rejected(Exception):
    def __init__(self, code):
        self.code, self.status = code, _ERROR_STATUS[code]
        super().__init__(code)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError('nonfinite_json_number')


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        _nonfinite(value)
    return number


def _json(raw):
    return json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                      parse_constant=_nonfinite, parse_float=_float)


def _messages(body, model_id):
    if type(body) is not dict or set(body) != {'model', 'messages'}:
        raise _Rejected('invalid_request_body')
    if type(body['model']) is not str or body['model'] != model_id:
        raise _Rejected('wrong_model')
    messages = body['messages']
    if (type(messages) is not list or not 1 <= len(messages) <= 256
            or any(type(row) is not dict or set(row) != {'role', 'content'}
                   or type(row['role']) is not str or row['role'] not in {'system', 'user', 'assistant'}
                   or type(row['content']) is not str for row in messages)):
        raise _Rejected('invalid_messages')
    copied = [{'role': row['role'], 'content': row['content']} for row in messages]
    try:
        digest = fingerprint(copied)
    except (ValueError, TypeError, UnicodeError):
        raise _Rejected('invalid_messages') from None
    return copied, digest


def make_handler(engine):
    """Subclass the frozen handler; add only the fixed token-count POST route."""
    base = frozen.make_handler(engine)
    request_ids = itertools.count(1)

    class Handler(base):
        def do_POST(self):
            if self.path != PATH:
                return super().do_POST()
            started, request_id, locked = time.monotonic(), next(request_ids), False
            status, body = 500, {'error': 'token_count_failed'}

            def event(kind, **fields):
                writer = getattr(engine, 'event', None)
                if callable(writer):
                    writer(kind, count_request_id=request_id, **fields)

            try:
                self.connection.settimeout(TIMEOUT_SECONDS)
                lengths = self.headers.get_all('Content-Length', [])
                if (len(lengths) != 1 or re.fullmatch('[0-9]{1,10}', lengths[0]) is None
                        or not 0 < int(lengths[0]) <= MAX_REQUEST_BYTES
                        or self.headers.get('Transfer-Encoding')):
                    raise _Rejected('invalid_request_body')
                length = int(lengths[0])
                raw = self.rfile.read(length)
                try:
                    if len(raw) != length:
                        raise ValueError('incomplete_body')
                    request = _json(raw)
                except (ValueError, UnicodeError, RecursionError):
                    raise _Rejected('invalid_request_body') from None
                messages, messages_sha = _messages(request, engine.model_id)
                if not engine.lock.acquire(blocking=False):
                    raise _Rejected('busy')
                locked = True
                if engine.failed:
                    raise _Rejected('model_unavailable')
                backend_sha = fingerprint(engine.backend)
                event('token_count_started', messages_sha256=messages_sha,
                      backend_sha256=backend_sha, request_bytes=length)
                prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                if type(prompt) is not str:
                    raise ValueError('invalid_template_result')
                # Default list output is CPU-only and has the same tokenization
                # options as the frozen generation path before tensor placement.
                ids = engine.tokenizer(prompt, add_special_tokens=False)['input_ids']
                if type(ids) is not list or not ids or any(type(value) is not int or value < 0 for value in ids):
                    raise ValueError('invalid_tokenizer_result')
                if fingerprint(messages) != messages_sha or fingerprint(engine.backend) != backend_sha:
                    raise ValueError('count_binding_changed')
                body = {'model': engine.model_id, 'messages_sha256': messages_sha,
                    'input_tokens': len(ids), 'backend_sha256': backend_sha,
                    'tokenizer_semantics': TOKENIZER_SEMANTICS}
                event('token_count_completed', messages_sha256=messages_sha, backend_sha256=backend_sha,
                      input_tokens=len(ids), count_seconds=time.monotonic() - started)
                status = 200
            except _Rejected as exc:
                status, body = exc.status, {'error': exc.code}
                try:
                    event('token_count_rejected', reason=exc.code, count_seconds=time.monotonic() - started)
                except BaseException:
                    status, body = 500, {'error': 'token_count_failed'}
            except TimeoutError:
                status, body = 408, {'error': 'request_timeout'}
            except BaseException as exc:
                status, body = 500, {'error': 'token_count_failed'}
                name = type(exc).__name__
                try:
                    event('token_count_failed', error_type=name if re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,127}', name)
                          else 'OtherException', count_seconds=time.monotonic() - started)
                except BaseException:
                    pass
            finally:
                if locked:
                    engine.lock.release()
            self.respond(status, body)

    return Handler


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _url(endpoint):
    if type(endpoint) is not str or any(ord(char) <= 32 or ord(char) == 127 for char in endpoint):
        raise ValueError('invalid_token_count_endpoint')
    parsed = urlsplit(endpoint)
    if (parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {'', '/', '/v1', '/v1/'}
            or (parsed.port is not None and not 0 < parsed.port < 65536)):
        raise ValueError('invalid_token_count_endpoint')
    return parsed._replace(path=PATH, query='', fragment='').geturl()


def count_tokens(endpoint, messages, *, model_id, expected_backend):
    """Make one bounded loopback request and return its exact authenticated count.

    Counts above the actor input budget are returned for the caller's local
    stopping rule. A failed request is unknown; this client never retries it.
    """
    try:
        url = _url(endpoint)
        if type(model_id) is not str or not model_id or type(expected_backend) is not dict or not expected_backend:
            raise ValueError('invalid_count_binding')
        copied, messages_sha = _messages({'model': model_id, 'messages': messages}, model_id)
        backend_sha = fingerprint(expected_backend)
        raw = json.dumps({'model': model_id, 'messages': copied}, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError('oversized_count_request')
    except (ValueError, TypeError, UnicodeError, _Rejected):
        raise TokenCountError('token_count_invalid_request') from None
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        request = urllib.request.Request(url, data=raw, headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if response.status != 200 or len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError('invalid_count_response')
        reply = _json(raw)
        if (type(reply) is not dict or set(reply) != _REPLY_FIELDS or reply['model'] != model_id
                or reply['messages_sha256'] != messages_sha or reply['backend_sha256'] != backend_sha
                or reply['tokenizer_semantics'] != TOKENIZER_SEMANTICS
                or type(reply['input_tokens']) is not int or reply['input_tokens'] < 1):
            raise ValueError('invalid_count_proof')
        return reply['input_tokens']
    except urllib.error.HTTPError as exc:
        exc.close()
        raise TokenCountError('token_count_request_failed') from None
    except Exception:
        raise TokenCountError('token_count_request_failed') from None
