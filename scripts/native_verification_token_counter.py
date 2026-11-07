"""Future server-native CPU token counting using the frozen tokenizer protocol.

Only the trusted, fixed CPU_CODE runs in the child process. The response limit
matches the original counter's validation after capture; it is not a memory cap.
"""

import json
import math
import os
import socket
import subprocess
import time

from scripts.verification_token_counter import (
    _REPLY_FIELDS,
    CPU_CODE,
    MAX_RESPONSE_BYTES,
    MODEL_PATH,
    PYTHON_PATH,
    TIMEOUT_SECONDS,
    TOKENIZER_SHA256,
    TokenCountError,
    _message_payload,
)

TMPDIR = '/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state/runtime-tmp'


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def count_input_tokens(messages, *, timeout_seconds=None):
    """Count one validated chat on amax; optionally cap the whole preparation."""
    deadline = None
    if timeout_seconds is not None:
        try:
            if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                    or timeout_seconds <= 0):
                raise ValueError
            deadline = time.monotonic() + min(TIMEOUT_SECONDS, timeout_seconds)
        except (TypeError, ValueError, OverflowError):
            raise TokenCountError('token_count_invalid_timeout') from None
    try:
        hostname = socket.gethostname()
    except Exception:
        raise TokenCountError('token_count_wrong_host') from None
    if hostname != 'amax':
        raise TokenCountError('token_count_wrong_host') from None
    try:
        payload, digest = _message_payload(messages)
    except Exception:
        raise TokenCountError('token_count_invalid_messages') from None
    try:
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '',
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1',
            'DO_NOT_TRACK': '1',
            'HF_HOME': '/dev/null',
            'HF_HUB_CACHE': '/dev/null',
            'XDG_CACHE_HOME': '/dev/null',
            'TOKENIZERS_PARALLELISM': 'false',
            'TMPDIR': TMPDIR,
        })
        timeout = TIMEOUT_SECONDS if deadline is None else deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError
        result = subprocess.run([PYTHON_PATH, '-I', '-B', '-c', CPU_CODE],
                                input=payload, capture_output=True,
                                timeout=timeout, check=False, env=env)
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
            or type(proof['status']) is not str or proof['status'] != 'measured_CPU_tokenization'
            or type(proof['messages_sha256']) is not str or proof['messages_sha256'] != digest
            or type(proof['model_path']) is not str or proof['model_path'] != MODEL_PATH
            or type(proof['tokenizer_files_sha256']) is not dict
            or proof['tokenizer_files_sha256'] != TOKENIZER_SHA256
            or type(proof['CUDA_VISIBLE_DEVICES']) is not str or proof['CUDA_VISIBLE_DEVICES'] != ''
            or proof['cuda_initialized'] is not False
            or proof['model_weights_loaded'] is not False
            or type(proof['GPU_jobs']) is not int or proof['GPU_jobs'] != 0
            or type(proof['model_calls']) is not int or proof['model_calls'] != 0
            or type(proof['input_tokens']) is not int or proof['input_tokens'] < 1):
        raise TokenCountError('token_count_invalid_proof') from None
    return proof['input_tokens']
