"""Same measured27B inference implementation with a separately frozen180s admission policy."""
import argparse
import importlib.metadata
import json
import os
import re
import sys
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts.probe_allocator import allocator_details, configure_allocator
from scripts.probe_chunked_prefill import CAP, PYTHON, ROOT, digest, require
from scripts.probe_qwen38 import MODEL, RUNTIME, WEIGHT_PROOF, verify_model_provenance
from scripts.serve_qwen38 import (
    INTEGRATION,
    MEMORY_SHA,
    PORT,
    PROBE,
    RECIPE,
    ChatEngine,
    make_handler,
    validate_memory_evidence,
)
from scripts.serve_transformers_presence import quantization_metadata

ID = 'qwen38-service-idle180-001'
PLAN = 'artifacts/qwen38-service-idle180-plan-001.json'
PREVIOUS = 'artifacts/qwen38-service-plan-001.json'
PREVIOUS_SHA = '0f92c4a1558655ebbae6ce5ace64990425596e5172942794316eae0ff1ae7fe8'
DOWNLOAD = 'artifacts/qwen38-27b-download-plan-001.json'
ADMISSION_SOURCES = {'scripts/gpu_idle_watch.py', 'wll/gpu_policy.py'}


def validate_changes(root, plan):
    root = Path(root)
    require(digest(root / PREVIOUS) == plan.get('previous_plan_sha256') == PREVIOUS_SHA,
        'previous service registration changed')
    previous = json.loads((root / PREVIOUS).read_text())
    require(plan['id'] == ID and plan['status'] == 'registered' and plan['recipe'] == RECIPE
        and plan['port'] == PORT
        and plan['admission_change'] == {'old_idle_seconds': 600, 'new_idle_seconds': 180},
        'unregistered service or admission change')
    require(set(previous['source_sha256']) | {'scripts/serve_qwen38_idle180.py'}
        <= plan['source_sha256'].keys(), 'missing service source binding')
    for name, original in previous['source_sha256'].items():
        require(plan['source_sha256'][name] == digest(root / name)
            and (name in ADMISSION_SOURCES or original == plan['source_sha256'][name]),
            'inference implementation changed: ' + name)
    policy = json.loads((root / 'config/gpu-policy.json').read_text())
    require(plan['gpu_policy_sha256'] == digest(root / 'config/gpu-policy.json')
        and policy['idle_seconds'] == 180 and set(policy['allowed_gpus']) == {'8', '9'},
        'different admission policy')


def preflight(expected_sha):
    require(digest(ROOT / PLAN) == expected_sha, 'service registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    validate_changes(ROOT, plan)
    require({PREVIOUS, RUNTIME, WEIGHT_PROOF, DOWNLOAD, PROBE, INTEGRATION,
        'config/gpu-policy.json'} <= plan['input_sha256'].keys(), 'missing service provenance')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in plan[field].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and digest(ROOT / name) == expected, 'bound service file changed: ' + name)
    audit = json.loads((ROOT / RUNTIME).read_text())
    require(sys.prefix == audit['runtime']['python_prefix'] == str(Path(PYTHON).parent.parent),
        'different Python environment')
    for name, expected in {**audit['runtime_sha256'], **plan['runtime_sha256']}.items():
        require(digest(name) == expected, 'service runtime changed: ' + name)
    for name, version in audit['runtime']['packages'].items():
        require(importlib.metadata.version(name) == version, 'different package: ' + name)
    verify_model_provenance(MODEL, json.loads((ROOT / DOWNLOAD).read_text()),
        json.loads((ROOT / WEIGHT_PROOF).read_text()))
    validate_memory_evidence(json.loads((ROOT / PROBE).read_text()),
        json.loads((ROOT / INTEGRATION).read_text()), digest(ROOT / PROBE))
    return plan


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
