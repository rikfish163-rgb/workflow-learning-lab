"""Separate16K workflow service using the measured27B NF4/chunked recipe."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.probe_allocator import allocator_details, configure_allocator, memory
from scripts.probe_chunked_prefill import CAP, ROOT, digest, require
from scripts.probe_qwen38 import MODEL, validate_case
from scripts.probe_qwen38 import preflight as memory_preflight
from scripts.serve_transformers_presence import generation_options, quantization_metadata

MAX_CONTEXT = 16384
MAX_OUTPUT = 1024
PORT = 18979
PLAN = 'artifacts/qwen38-service-plan-001.json'
MEMORY_SHA = 'b896c7311e52cde4a7594e95afa2e9657c2530a6787e896d32e7226b7d14b352'
PROBE = 'artifacts/qwen38-27b-nf4-feasibility-001.json'
INTEGRATION = 'artifacts/qwen38-probe-integration-001.json'
RECIPE = {'max_context': MAX_CONTEXT, 'max_output_tokens': MAX_OUTPUT,
          'prefill_chunk_size': 512, 'disable_compile': True, 'thinking': False,
          'top_p': .8, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 1.5,
          'repetition_penalty': 1.0, 'presence_scope': 'current_completion_tokens_only',
          'attention_implementation': 'sdpa', 'loader': 'multimodal',
          'allocator': 'backend:native,expandable_segments:True'}


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class ModelExecutionError(RuntimeError):
    pass


def validate_request(body, model_id):
    if not isinstance(body, dict) or not set(body) <= {'model', 'messages', 'max_tokens', 'temperature', 'seed'}:
        raise RequestError('unsupported_request_fields')
    if body.get('model') != model_id:
        raise RequestError('wrong_model')
    messages = body.get('messages')
    if (not isinstance(messages, list) or not 1 <= len(messages) <= 256
            or any(not isinstance(m, dict) or set(m) != {'role', 'content'}
                   or m['role'] not in ('system', 'user', 'assistant')
                   or not isinstance(m['content'], str) for m in messages)):
        raise RequestError('invalid_messages')
    max_new = body.get('max_tokens', 512)
    temperature = body.get('temperature', .7)
    seed = body.get('seed', 0)
    if (type(max_new) is not int or not 1 <= max_new <= MAX_OUTPUT
            or type(temperature) not in (int, float) or not math.isfinite(temperature)
            or not 0 <= temperature <= 2 or type(seed) is not int or not 0 <= seed < 2**31):
        raise RequestError('invalid_generation_parameters')
    return {'messages': messages, 'max_new': max_new, 'temperature': float(temperature), 'seed': seed}


def validate_context(prompt_tokens, max_new):
    if prompt_tokens < 1 or prompt_tokens + max_new > MAX_CONTEXT:
        raise RequestError('context_budget_exceeded')


def request_generation_options(*, temperature, max_new, pad_token_id, prompt_length):
    options = generation_options(temperature=temperature, max_new=max_new, pad_token_id=pad_token_id,
        top_p=.8, top_k=20, min_p=0.0, presence_penalty=1.5, prompt_length=prompt_length)
    options.update(prefill_chunk_size=512, disable_compile=True, use_cache=True, logits_to_keep=1,
                   output_logits=False, output_scores=False, return_dict_in_generate=False,
                   output_hidden_states=False, output_attentions=False, num_beams=1)
    return options


def finish_reason(tokens, budget, eos_ids):
    eos = [] if eos_ids is None else ([eos_ids] if isinstance(eos_ids, int) else eos_ids)
    if tokens and tokens[-1] in eos:
        return 'stop'
    return 'length' if len(tokens) == budget else 'unknown'


class ChatEngine:
    def __init__(self, model, tokenizer, model_id, backend, *, device='cuda:0'):
        require(device in ('cpu', 'cuda:0'), 'only an explicit CPU test or admitted logical GPU')
        self.model, self.tokenizer, self.model_id = model, tokenizer, model_id
        self.backend, self.device = backend, device
        self.lock = threading.Lock()
        self.failed = False
        self.calls = 0

    def event(self, event, **details):
        if self.device == 'cuda:0':
            print(json.dumps({'event': event, 'request_id': self.calls, **details}), flush=True)

    def complete(self, body):
        request = validate_request(body, self.model_id)
        if not self.lock.acquire(blocking=False):
            raise RequestError('busy', 409)
        inputs = outputs = tokens = None
        try:
            if self.failed:
                raise RequestError('model_unavailable', 503)
            prompt = self.tokenizer.apply_chat_template(request['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
            n = inputs.input_ids.shape[1]
            validate_context(n, request['max_new'])  # Still on CPU; reject before any CUDA allocation.
            import torch

            if self.device == 'cuda:0':
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            inputs = inputs.to(self.device)
            torch.manual_seed(request['seed'])
            options = request_generation_options(temperature=request['temperature'],
                max_new=request['max_new'], pad_token_id=self.tokenizer.eos_token_id, prompt_length=n)
            start = time.monotonic()
            self.calls += 1
            self.event('generation_started', prompt_tokens=n, max_new=request['max_new'],
                       request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **options)
            if self.device == 'cuda:0':
                torch.cuda.synchronize()
                require(torch.cuda.max_memory_reserved() <= CAP, 'allocator_limit_exceeded')
            tokens = outputs[0, n:]
            require(0 < len(tokens) <= request['max_new'], 'invalid_generation_length')
            values = tokens.tolist()
            text = self.tokenizer.decode(values, skip_special_tokens=True)
            reason = finish_reason(values, request['max_new'], self.model.generation_config.eos_token_id)
            elapsed = time.monotonic() - start
            telemetry = memory(torch) if self.device == 'cuda:0' else None
            self.event('generation_completed', prompt_tokens=n, completion_tokens=len(values),
                       finish_reason=reason, generation_seconds=elapsed, memory=telemetry)
            return {'model': self.model_id, 'choices': [{'message': {'role': 'assistant', 'content': text},
                                                       'finish_reason': reason}],
                    'format_error': None, 'backend': self.backend,
                    'usage': {'prompt_tokens': n, 'completion_tokens': len(values)},
                    'generation_seconds': elapsed, 'memory': telemetry, 'request_id': self.calls}
        except RequestError:
            raise
        except Exception as exc:
            self.failed = True
            self.event('generation_failed', error_type=type(exc).__name__)
            raise ModelExecutionError(type(exc).__name__) from exc
        finally:
            inputs = outputs = tokens = None
            if self.device == 'cuda:0':
                try:
                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    self.failed = True
                    self.event('cleanup_failed', error_type=type(exc).__name__)
            self.lock.release()


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(180)

        def log_message(self, *args):
            pass

        def respond(self, status, body):
            payload = json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Generation already drained; a disconnected client has unknown response cost.

        def do_GET(self):
            if self.path != '/health':
                self.respond(404, {'error': 'not_found'})
                return
            self.respond(200, {'model': engine.model_id, 'ready': not engine.failed,
                              'busy': engine.lock.locked(), 'max_context': MAX_CONTEXT,
                              'max_output_tokens': MAX_OUTPUT, 'backend': engine.backend})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.respond(404, {'error': 'not_found'})
                return
            try:
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 2_000_000 or self.headers.get('Transfer-Encoding'):
                        raise ValueError()
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError()
                    body = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise RequestError('invalid_request_body') from None
                self.respond(200, engine.complete(body))
            except RequestError as exc:
                self.respond(exc.status, {'error': str(exc)})
            except TimeoutError:
                self.respond(408, {'error': 'request_timeout'})
            except ModelExecutionError as exc:
                self.respond(500, {'error': str(exc)})
            finally:
                if engine.failed:
                    threading.Thread(target=self.server.shutdown, daemon=True).start()

    return Handler


def preflight(expected_sha):
    require(digest(ROOT / PLAN) == expected_sha, 'service registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    require(plan['status'] == 'registered' and plan['recipe'] == RECIPE and plan['port'] == PORT,
            'unregistered service configuration')
    require({'scripts/serve_qwen38.py', 'scripts/generated_presence.py',
             'scripts/serve_transformers_presence.py'} <= plan['source_sha256'].keys(),
            'service implementation binding missing')
    require({PROBE, INTEGRATION} <= plan['input_sha256'].keys(), 'actual memory evidence missing')
    for field in ('source_sha256', 'input_sha256'):
        for name, sha in plan[field].items():
            require(digest(ROOT / name) == sha, f'bound service file changed: {name}')
    memory_preflight(MEMORY_SHA)
    probe = json.loads((ROOT / PROBE).read_text())
    audit = json.loads((ROOT / INTEGRATION).read_text())
    validate_memory_evidence(probe, audit, digest(ROOT / PROBE))
    return plan


def validate_memory_evidence(probe, audit, probe_sha):
    require(probe['plan_sha256'] == audit['plan_sha256'] == MEMORY_SHA
            and audit['probe_sha256'] == probe_sha
            and audit['model_group_and_controller_released'] is True
            and audit['completion_gate'] is False and probe['phase'] == 'failed'
            and probe['error_type'] == 'OutOfMemoryError', 'unexpected memory evidence')
    require(len(probe['cases']) == 3, 'different memory coverage')
    validate_case(probe['cases'][-1], {'input_tokens': 16384, 'output_tokens': 1024})
    require(probe['quantization_actual']['linear4bit_layers'] == 496
            and probe['quantization_actual']['parameter_devices'] == ['cuda:0'], 'unmeasured placement')


@contextmanager
def loaded_engine(plan_sha, model_id, *, admitted=None):
    configure_allocator('expandable')
    plan = preflight(plan_sha)
    from wll.gpu_policy import acquire_gpu, verify_cuda

    with acquire_gpu() as gpu:
        if admitted is not None:
            admitted(gpu)
        import torch
        from transformers import AutoTokenizer

        from scripts.nf4_loader import inspect_nf4, load_nf4
        from scripts.text_chunked_prefill import install_text_chunked_prefill

        verify_cuda(torch)
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = load_nf4(MODEL, 'cuda:0')
        actual = inspect_nf4(model, 'cuda:0')
        require(actual['linear4bit_layers'] == 496, 'different quantization coverage')
        install_text_chunked_prefill(model)
        torch.cuda.synchronize()
        require(torch.cuda.max_memory_reserved() <= CAP, 'model_loading_exceeded_cap')
        allocator = allocator_details(torch, 'expandable', require_segments=True)
        backend = {'server_sha256': digest(Path(__file__)), 'service_plan_sha256': plan_sha,
                   'memory_plan_sha256': MEMORY_SHA, 'recipe': RECIPE, 'gpu_policy': gpu,
                   'quantization': quantization_metadata(actual),
                   'allocator_backend': allocator['backend'],
                   'expandable_segments': allocator['settings']['expandable_segments'],
                   'source_sha256': plan['source_sha256'],
                   'packages': {n: importlib.metadata.version(n) for n in
                                ('torch', 'transformers', 'bitsandbytes', 'accelerate', 'safetensors')},
                   'files': {n: digest(MODEL / n) for n in
                             ('config.json', 'tokenizer_config.json', 'generation_config.json', 'weight-source.json')}}
        yield ChatEngine(model, tokenizer, model_id, backend)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    require(re.fullmatch('[a-z0-9_-]{1,100}', args.model_id) is not None and args.port == PORT,
            'invalid model ID or unregistered port')
    if args.check:
        configure_allocator('expandable')
        preflight(args.plan_sha256)
        print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'recipe': RECIPE}))
        return
    with loaded_engine(args.plan_sha256, args.model_id) as engine:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(engine))
        print(json.dumps({'ready': True, 'model': args.model_id, 'port': args.port, 'pid': os.getpid(),
                          'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19]}), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
        if engine.failed:
            raise RuntimeError('model service stopped after execution failure')


if __name__ == '__main__':
    main()
