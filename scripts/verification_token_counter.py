"""CPU-only exact chat-token counts using local, pinned tokenizer files."""

import hashlib
import json
import os
import subprocess
import sys

MODEL_PATH = os.environ.get('WLL_TOKENIZER_PATH', '')
TOKENIZER_SHA256 = {
    'tokenizer_config.json': 'b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27',
    'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
    'chat_template.jinja': 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041',
    'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
    'tokenizer.json': '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3',
}
TIMEOUT_SECONDS = 55
MAX_REQUEST_BYTES = 2_000_000
MAX_RESPONSE_BYTES = 32768
_REPLY_FIELDS = frozenset({
    'status', 'messages_sha256', 'input_tokens', 'model_path',
    'tokenizer_files_sha256', 'CUDA_VISIBLE_DEVICES', 'cuda_initialized',
    'model_weights_loaded', 'GPU_jobs', 'model_calls',
})

CPU_CODE = '''import hashlib,json,os,sys
from pathlib import Path
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('cuda_visibility_invalid')
p = json.load(sys.stdin)
model = Path(p['model_path'])
def hashes():
    return {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
            for name in p['tokenizer_files_sha256']}
actual = hashes()
if actual != p['tokenizer_files_sha256']:
    raise RuntimeError('tokenizer_files_changed')
import torch
if torch.cuda.is_initialized():
    raise RuntimeError('cuda_initialized')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                          trust_remote_code=False)
messages = p['messages']
sha = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
if sha != p['messages_sha256']:
    raise RuntimeError('messages_changed')
text = tokenizer.apply_chat_template(messages, tokenize=False,
          add_generation_prompt=True, enable_thinking=False)
count = len(tokenizer(text, add_special_tokens=False)['input_ids'])
if torch.cuda.is_initialized() or hashes() != actual:
    raise RuntimeError('tokenizer_runtime_changed')
print(json.dumps(dict(status='measured_CPU_tokenization', messages_sha256=sha,
    input_tokens=count, tokenizer_files_sha256=actual, model_path=str(model),
    CUDA_VISIBLE_DEVICES='', cuda_initialized=False, model_weights_loaded=False,
    GPU_jobs=0, model_calls=0)))
'''


class TokenCountError(RuntimeError):
    """A fixed, content-free failure from token-count preparation or transport."""


def _message_payload(messages):
    if (type(messages) is not list or not 1 <= len(messages) <= 256
            or any(type(item) is not dict or set(item) != {'role', 'content'}
                   or type(item['role']) is not str
                   or item['role'] not in {'system', 'user', 'assistant'}
                   or type(item['content']) is not str for item in messages)):
        raise TokenCountError('token_count_invalid_messages') from None
    try:
        # Copy while validating, then use exactly this immutable text snapshot on
        # the wire and for the fingerprint used by the rest of the project.
        copied = [{'role': item['role'], 'content': item['content']} for item in messages]
        digest = hashlib.sha256(json.dumps(copied, sort_keys=True,
                                          ensure_ascii=False).encode()).hexdigest()
        payload = {'messages': copied, 'messages_sha256': digest,
                   'model_path': MODEL_PATH, 'tokenizer_files_sha256': TOKENIZER_SHA256}
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError, UnicodeError):
        raise TokenCountError('token_count_invalid_messages') from None
    if len(encoded) > MAX_REQUEST_BYTES:
        raise TokenCountError('token_count_invalid_messages') from None
    return encoded, digest


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def count_input_tokens(messages):
    """Count exactly one validated chat input; fail closed without echoing content."""
    if not MODEL_PATH:
        raise TokenCountError('token_count_tokenizer_path_required') from None
    payload, digest = _message_payload(messages)
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': '',
        'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
    })
    try:
        result = subprocess.run([sys.executable, '-I', '-B', '-c', CPU_CODE],
                                input=payload, capture_output=True,
                                timeout=TIMEOUT_SECONDS, check=False, env=env)
    except Exception:
        raise TokenCountError('token_count_transport_failed') from None
    if (result.returncode != 0 or type(result.stdout) is not bytes
            or len(result.stdout) > MAX_RESPONSE_BYTES):
        raise TokenCountError('token_count_transport_failed') from None
    try:
        proof = json.loads(result.stdout, object_pairs_hook=_unique_pairs)
    except (ValueError, UnicodeError, RecursionError):
        raise TokenCountError('token_count_invalid_proof') from None
    if (type(proof) is not dict or set(proof) != _REPLY_FIELDS
            or proof['status'] != 'measured_CPU_tokenization'
            or proof['messages_sha256'] != digest or proof['model_path'] != MODEL_PATH
            or proof['tokenizer_files_sha256'] != TOKENIZER_SHA256
            or proof['CUDA_VISIBLE_DEVICES'] != '' or proof['cuda_initialized'] is not False
            or proof['model_weights_loaded'] is not False
            or type(proof['GPU_jobs']) is not int or proof['GPU_jobs'] != 0
            or type(proof['model_calls']) is not int or proof['model_calls'] != 0
            or type(proof['input_tokens']) is not int or proof['input_tokens'] < 1):
        raise TokenCountError('token_count_invalid_proof') from None
    return proof['input_tokens']


class TokenCountingModel:
    """Add exact CPU context counts to a trusted model delegate, without I/O at init."""

    def __init__(self, delegate):
        if not callable(getattr(delegate, 'generate', None)):
            raise ValueError('invalid_model_delegate') from None
        self._delegate = delegate

    def generate(self, messages, seed):
        return self._delegate.generate(messages, seed=seed)

    def count_input_tokens(self, messages):
        return count_input_tokens(messages)
