"""Bounded single-transport CPU token counting for named private inputs.

Only hashes, counts and tokenizer/resource metadata are returned. Input content
is sent on stdin and never included in the shell command, proof, or errors.
"""

import json
import os
import re
import selectors
import shlex
import subprocess
import time

from scripts import verification_token_counter as original
from wll.protocol import fingerprint

MODEL_PATH = original.MODEL_PATH
PYTHON_PATH = original.PYTHON_PATH
SSH = original.SSH
TOKENIZER_SHA256 = dict(original.TOKENIZER_SHA256)
TIMEOUT_SECONDS = original.TIMEOUT_SECONDS
MAX_REQUEST_BYTES = original.MAX_REQUEST_BYTES
MAX_RESPONSE_BYTES = original.MAX_RESPONSE_BYTES
MAX_ENTRIES = 128
TokenCountError = original.TokenCountError

CPU_CODE = '''import hashlib,json,os,sys
from pathlib import Path
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('cuda_visibility_invalid')
p = json.load(sys.stdin)
model = Path(p['model_path'])
def hashes():
    return {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
            for name in p['tokenizer_files_sha256']}
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
before = hashes()
if before != p['tokenizer_files_sha256']:
    raise RuntimeError('tokenizer_files_changed')
import torch
if torch.cuda.is_initialized():
    raise RuntimeError('cuda_initialized')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
results = {}
for group,kind in (('message_batches','messages'),('texts','text')):
    for name,item in p[group].items():
        value = item['content']
        sha = digest(value)
        if sha != item['content_sha256'] or name in results:
            raise RuntimeError('input_changed')
        text = (tokenizer.apply_chat_template(value, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False) if kind == 'messages' else value)
        if type(text) is not str:
            raise RuntimeError('tokenizer_text_invalid')
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        if (type(ids) is not list or not ids or any(type(token) is not int for token in ids)
                or digest(value) != sha):
            raise RuntimeError('tokenizer_result_invalid')
        results[name] = dict(kind=kind, content_sha256=sha, input_tokens=len(ids))
after = hashes()
if torch.cuda.is_initialized() or after != before:
    raise RuntimeError('tokenizer_runtime_changed')
print(json.dumps(dict(status='measured_CPU_batch_tokenization', results=results,
    model_path=str(model), tokenizer_files_sha256=before, tokenizer_files_sha256_after=after,
    CUDA_VISIBLE_DEVICES='', cuda_initialized=False, model_weights_loaded=False,
    GPU_jobs=0, model_calls=0)))
'''


def _invalid_input():
    raise TokenCountError('batch_token_count_invalid_input') from None


def _payload(message_batches, texts):
    if (type(message_batches) is not dict or type(texts) is not dict
            or not 1 <= len(message_batches) + len(texts) <= MAX_ENTRIES):
        _invalid_input()
    try:
        names = [*message_batches, *texts]
        if (any(type(name) is not str or not 1 <= len(name) <= 80
                or re.fullmatch('[A-Za-z0-9._:/-]+', name) is None for name in names)
                or len(set(names)) != len(names)):
            _invalid_input()
        request = {'model_path': MODEL_PATH, 'tokenizer_files_sha256': dict(TOKENIZER_SHA256),
                   'message_batches': {}, 'texts': {}}
        expected = {}
        total = 0
        for name, messages in message_batches.items():
            if type(messages) is not list or not 1 <= len(messages) <= 256:
                _invalid_input()
            copied = []
            for item in messages:
                if (type(item) is not dict or any(type(key) is not str for key in item)
                        or set(item) != {'role', 'content'}
                        or type(item['role']) is not str or item['role'] not in {'system', 'user', 'assistant'}
                        or type(item['content']) is not str or len(item['content']) > MAX_REQUEST_BYTES):
                    _invalid_input()
                total += len(item['content'].encode('utf-8'))
                if total > MAX_REQUEST_BYTES:
                    _invalid_input()
                copied.append({'role': item['role'], 'content': item['content']})
            sha = fingerprint(copied)
            request['message_batches'][name] = {'content': copied, 'content_sha256': sha}
            expected[name] = {'kind': 'messages', 'content_sha256': sha}
        for name, value in texts.items():
            if type(value) is not str or not 1 <= len(value) <= MAX_REQUEST_BYTES:
                _invalid_input()
            total += len(value.encode('utf-8'))
            if total > MAX_REQUEST_BYTES:
                _invalid_input()
            sha = fingerprint(value)
            request['texts'][name] = {'content': value, 'content_sha256': sha}
            expected[name] = {'kind': 'text', 'content_sha256': sha}
        raw = json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(raw) > MAX_REQUEST_BYTES:
            _invalid_input()
        return raw, expected
    except (ValueError, TypeError, UnicodeError, RecursionError, MemoryError, OverflowError, RuntimeError):
        _invalid_input()


def _transport(payload):
    """Send stdin and collect bounded stdout without buffering an unbounded reply."""
    command = ('CUDA_VISIBLE_DEVICES= TOKENIZERS_PARALLELISM=false '
               'timeout --signal=TERM --kill-after=1s 54s '
               + shlex.quote(PYTHON_PATH) + ' -I -B -c ' + shlex.quote(CPU_CODE))
    process = None
    selector = None
    try:
        deadline = time.monotonic() + TIMEOUT_SECONDS
        process = subprocess.Popen([*SSH, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, bufsize=0)
        if process.stdin is None or process.stdout is None:
            raise RuntimeError('missing_pipe')
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdin, selectors.EVENT_WRITE, 'input')
        selector.register(process.stdout, selectors.EVENT_READ, 'output')
        offset, size, chunks = 0, 0, []
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('deadline')
            events = selector.select(remaining)
            if not events:
                raise TimeoutError('deadline')
            for key, _ in events:
                if key.data == 'input':
                    count = os.write(key.fileobj.fileno(), payload[offset:offset + 65536])
                    if count <= 0:
                        raise RuntimeError('short_write')
                    offset += count
                    if offset == len(payload):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                else:
                    part = os.read(key.fileobj.fileno(), min(65536, MAX_RESPONSE_BYTES + 1 - size))
                    if not part:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    else:
                        size += len(part)
                        if size > MAX_RESPONSE_BYTES:
                            raise RuntimeError('response_too_large')
                        chunks.append(part)
        if offset != len(payload) or process.wait(timeout=max(0, deadline - time.monotonic())) != 0:
            raise RuntimeError('transport_failed')
        return b''.join(chunks)
    except Exception:
        raise TokenCountError('batch_token_count_transport_failed') from None
    finally:
        if selector is not None:
            selector.close()
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1)
            except (OSError, subprocess.SubprocessError):
                pass


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def _bad_constant(_value):
    raise ValueError('nonfinite_json')


def _validate_proof(raw, expected):
    try:
        if type(raw) is not bytes or len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError('invalid_reply')
        proof = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_bad_constant)
        fields = {'status', 'results', 'model_path', 'tokenizer_files_sha256',
                  'tokenizer_files_sha256_after', 'CUDA_VISIBLE_DEVICES', 'cuda_initialized',
                  'model_weights_loaded', 'GPU_jobs', 'model_calls'}
        if (type(proof) is not dict or set(proof) != fields
                or proof['status'] != 'measured_CPU_batch_tokenization' or proof['model_path'] != MODEL_PATH
                or proof['tokenizer_files_sha256'] != TOKENIZER_SHA256
                or proof['tokenizer_files_sha256_after'] != TOKENIZER_SHA256
                or proof['CUDA_VISIBLE_DEVICES'] != '' or proof['cuda_initialized'] is not False
                or proof['model_weights_loaded'] is not False
                or type(proof['GPU_jobs']) is not int or proof['GPU_jobs'] != 0
                or type(proof['model_calls']) is not int or proof['model_calls'] != 0
                or type(proof['results']) is not dict or set(proof['results']) != set(expected)):
            raise ValueError('invalid_reply')
        for name, binding in expected.items():
            row = proof['results'][name]
            if (type(row) is not dict or set(row) != {'kind', 'content_sha256', 'input_tokens'}
                    or row['kind'] != binding['kind'] or row['content_sha256'] != binding['content_sha256']
                    or type(row['input_tokens']) is not int or row['input_tokens'] < 1):
                raise ValueError('invalid_reply')
        return proof
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, MemoryError, OverflowError):
        raise TokenCountError('batch_token_count_invalid_proof') from None


def count_batch(message_batches, texts):
    """Measure at most 128 named chats/texts in one CPU-only tokenizer process."""
    payload, expected = _payload(message_batches, texts)
    return _validate_proof(_transport(payload), expected)
