"""Bounded CodeAct HTTP client; no model or generated Python is executed here.

The owner binds host/service identity and durably stores every private record.
The exact counter must honor its supplied absolute deadline. Length/unknown
finishes retain their usage; the coordinator owns action/target admission.
"""
import base64
import math
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

from scripts import native_appworld_model_client as wire
from wll.model import ContextBudgetExceeded

MAX_INPUT, MAX_OUTPUT, MAX_CONTEXT = 15360, 1024, 16384
_METRICS = ('allocated_bytes', 'reserved_bytes', 'active_bytes', 'inactive_split_bytes')
_MEMORY_KEYS = {f'{m}.all.{p}' for m in _METRICS for p in ('current', 'peak')} | {
    'num_alloc_retries', 'num_ooms', 'num_device_alloc', 'num_device_free'}
_REPLY_KEYS = {'model', 'choices', 'usage', 'format_error', 'backend',
               'generation_seconds', 'request_id', 'memory'}


class CodeActModelError(RuntimeError):
    """Content-free client failure; an instance cannot retry after failure."""


def _require(condition):
    if not condition:
        raise ValueError('invalid')


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _copy(value):
    return wire._json(wire._encoded(value))


def _usage(reply):
    usage = reply.get('usage') if type(reply) is dict else None
    if (type(usage) is dict and set(usage) == {'prompt_tokens', 'completion_tokens'}
            and type(usage['prompt_tokens']) is int and 1 <= usage['prompt_tokens'] <= MAX_INPUT
            and type(usage['completion_tokens']) is int and 1 <= usage['completion_tokens'] <= MAX_OUTPUT):
        return dict(usage)
    return None


def _memory(value):
    if value is None:
        return
    _require(type(value) is dict and set(value) == _MEMORY_KEYS
             and all(type(v) is int and v >= 0 for v in value.values())
             and all(value[f'{m}.all.peak'] >= value[f'{m}.all.current'] for m in _METRICS))


class CodeActModel:
    def __init__(self, endpoint, model_id, *, expected_backend, count_tokens, record,
                 deadline, tmp_dir, timeout=180, temperature=.7,
                 _transport=None, _clock=time.monotonic):
        try:
            self._base = wire._base_url(endpoint)
            _require(type(model_id) is str and 1 <= len(model_id) <= 128
                     and not any(ord(c) <= 32 or ord(c) == 127 for c in model_id))
            _require(type(expected_backend) is dict and bool(expected_backend)
                     and all(callable(v) for v in (count_tokens, record, _clock))
                     and (_transport is None or callable(_transport)))
            _require(_finite(deadline) and deadline > 0 and _finite(timeout) and 0 < timeout <= 180
                     and _finite(temperature) and temperature <= 2)
            raw = os.fspath(tmp_dir)
            path = Path(raw)
            _require(type(raw) is str and path.is_absolute() and str(path) == raw
                     and path.resolve(strict=True) == path)
            info = path.lstat()
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
                     and stat.S_IMODE(info.st_mode) == 0o700)
            self.tmp_dir = raw
            self._backend = _copy(expected_backend)
            self._backend_digest = wire._digest(self._backend)
            self.model_id, self.max_tokens, self.context_tokens = model_id, MAX_OUTPUT, MAX_CONTEXT
            self.deadline, self.timeout, self.temperature = float(deadline), float(timeout), float(temperature)
            self._counter, self._record, self._transport = count_tokens, record, _transport
            self._clock, self._last_now, self._failed = _clock, 0.0, False
            self._last_request_id = 0
            self._left(self.deadline)
        except Exception:
            raise CodeActModelError('codeact_model_configuration') from None

    def _now(self):
        value = self._clock()
        _require(_finite(value) and value >= self._last_now)
        self._last_now = float(value)
        return self._last_now

    def _left(self, deadline):
        remaining = min(self.deadline, deadline) - self._now()
        if remaining <= 0:
            raise CodeActModelError('codeact_model_deadline')
        return remaining

    def _emit(self, event, **fields):
        try:
            self._record(_copy({'event': event, **fields}))
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            raise CodeActModelError('codeact_model_record') from None

    def generate(self, messages, seed):
        if self._failed:
            raise CodeActModelError('codeact_model_unavailable') from None
        stage, started, observed, context_stop = 'binding', False, None, False
        try:
            self._left(self.deadline)
            _require(type(seed) is int and 0 <= seed < 2**31)
            payload, digest = wire._message_payload(messages)
            snapshot = wire._json(payload)['messages']
            stage = 'count'
            self._emit('count_intent', messages=snapshot, messages_sha256=digest, seed=seed)
            self._left(self.deadline)
            count = self._counter(snapshot, deadline=self.deadline)
            _require(type(count) is int and count > 0
                     and wire._message_payload(snapshot)[1] == digest
                     and wire._message_payload(messages)[1] == digest)
            self._emit('count_returned', input_tokens=count, messages_sha256=digest)
            self._left(self.deadline)
            if count > MAX_INPUT or count + self.max_tokens > self.context_tokens:
                context_stop = True
                raise ContextBudgetExceeded('codeact_model_context_budget')
            body = {'model': self.model_id, 'messages': snapshot, 'seed': seed,
                    'max_tokens': self.max_tokens, 'temperature': self.temperature}
            raw_body = wire._encoded(body)
            _require(len(raw_body) <= 2_000_000)
            stage = 'intent'
            self._emit('generation_intent', request=body, request_sha256=wire._digest(body),
                       messages_sha256=digest, seed=seed)
            _require(wire._message_payload(messages)[1] == digest
                     and wire._message_payload(snapshot)[1] == digest)
            stage = 'transport'
            operation_end = min(self.deadline, self._now() + self.timeout)
            url = self._base + '/v1/chat/completions'
            child_input = wire._encoded({'url': url, 'body': base64.b64encode(raw_body).decode('ascii'),
                                        'timeout': self._left(operation_end)})
            remaining = self._left(operation_end)
            started = True
            if self._transport is not None:
                raw = self._transport(url, raw_body, timeout=remaining, tmp_dir=self.tmp_dir)
            else:
                child = subprocess.run(
                    [sys.executable, '-I', '-B', '-c', wire._HTTP_CHILD], input=child_input,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=remaining,
                    cwd=self.tmp_dir, check=False,
                    env={'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                         'TMPDIR': self.tmp_dir, 'LANG': 'C.UTF-8'})
                _require(type(child.returncode) is int and child.returncode == 0)
                raw = child.stdout
            _require(type(raw) is bytes and len(raw) <= wire.MAX_RESPONSE_BYTES)
            stage = 'reply'
            self._emit('generation_raw_reply', raw_reply_base64=base64.b64encode(raw).decode('ascii'))
            reply = wire._json(raw)
            observed = _usage(reply)
            self._left(operation_end)
            _require(type(reply) is dict and set(reply) == _REPLY_KEYS
                     and reply['model'] == self.model_id
                     and wire._digest(reply['backend']) == self._backend_digest
                     and (reply['format_error'] is None or type(reply['format_error']) is str)
                     and type(reply['choices']) is list and len(reply['choices']) == 1)
            choice = reply['choices'][0]
            _require(type(choice) is dict and set(choice) == {'message', 'finish_reason'})
            message = choice['message']
            _require(type(message) is dict and set(message) == {'role', 'content'}
                     and message['role'] == 'assistant' and type(message['content']) is str
                     and type(choice['finish_reason']) is str
                     and choice['finish_reason'] in {'stop', 'length', 'unknown'}
                     and observed is not None and observed['prompt_tokens'] == count
                     and _finite(reply['generation_seconds'])
                     and type(reply['request_id']) is int
                     and reply['request_id'] > self._last_request_id)
            _memory(reply['memory'])
            response = {'text': message['content'], **observed,
                        **{key: reply[key] for key in ('format_error', 'backend', 'generation_seconds',
                                                       'request_id', 'memory')},
                        'finish_reason': choice['finish_reason']}
            self._emit('generation_validated', response=response)
            self._left(operation_end)
            self._last_request_id = reply['request_id']
            return response
        except BaseException as error:
            self._failed = True
            try:
                self._emit('generation_failed', stage=stage, transport_started=started,
                           observed_usage=observed,
                           known_not_dispatched=context_stop and not started,
                           unknown_cost=not (context_stop and not started))
            except CodeActModelError:
                raise CodeActModelError('codeact_model_record') from None
            if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            if context_stop:
                raise ContextBudgetExceeded('codeact_model_context_budget') from None
            code = 'codeact_model_' + stage
            if (type(error) is CodeActModelError and len(error.args) == 1
                    and type(error.args[0]) is str
                    and error.args[0] in {'codeact_model_deadline', 'codeact_model_record'}):
                code = error.args[0]
            raise CodeActModelError(code) from None
