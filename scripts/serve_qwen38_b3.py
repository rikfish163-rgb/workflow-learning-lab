"""Separately registered, matched B0/NF4 and B3/FP32-LoRA inference service.

Startup proves provenance and enforces admission/cap guards. The independently
registered CUDA gate and its release must pass before a benchmark controller
uses this service. A service registration alone is not memory/quality evidence.
"""
import argparse
import importlib.metadata
import json
import os
import re
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts.probe_allocator import allocator_details, configure_allocator
from scripts.probe_chunked_prefill import CAP, ROOT, digest, require
from scripts.probe_qwen38 import MODEL
from scripts.serve_qwen38 import PORT, RECIPE, ChatEngine, make_handler
from scripts.serve_qwen38_idle180 import preflight as parent_preflight
from scripts.serve_transformers_presence import quantization_metadata

ID = 'qwen38-b3-service-001'
PLAN = 'artifacts/qwen38-b3-service-plan-001.json'
PARENT_SERVICE_PLAN = 'artifacts/qwen38-service-idle180-plan-001.json'
PARENT_SERVICE_SHA = '040a6a4476d1b4c6b4ef1d78d2e76fe1e600783b47c121d7ef7cf7a1ab668889'
TRAINING_RESULT = 'artifacts/qwen38-b3-sft-001.json'
ADAPTER = 'artifacts/qwen38-b3-sft-001/adapter'
GATE_DESIGN = 'artifacts/qwen38-b3-inference-gate-design-001.json'
GATE_OUTPUT = 'artifacts/qwen38-b3-inference-gate-integration-001.json'
MODEL_IDS = {'B0': 'qwen38-b3-matched-b0-001', 'B3': 'qwen38-b3-matched-b3-001'}
REQUIRED_SOURCES = {
    'scripts/serve_qwen38_b3.py', 'scripts/serve_qwen38.py', 'scripts/serve_qwen38_idle180.py',
    'scripts/load_hybrid_adapter.py', 'scripts/save_hybrid_adapter.py',
    'scripts/qwen38_training.py', 'scripts/training_numerics.py',
    'scripts/nf4_loader.py', 'scripts/text_chunked_prefill.py',
    'scripts/probe_allocator.py', 'scripts/probe_chunked_prefill.py',
    'scripts/generated_presence.py', 'scripts/serve_transformers_presence.py',
    'wll/gpu_policy.py', 'wll/gpu_idle.py',
}


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _relative(name):
    require(type(name) is str and name and not Path(name).is_absolute()
        and '..' not in Path(name).parts and '.' not in Path(name).parts,
        'repository-relative registered path required')
    return Path(name)


def _file(root, name):
    path = root / _relative(name)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
        and all(not parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents),
        'registered file missing or symlinked: ' + name)
    return path


def _validate_export(root, plan):
    reference = plan.get('training_result', {})
    require(reference.get('path') == TRAINING_RESULT and _sha(reference.get('sha256'))
        and plan['input_sha256'].get(TRAINING_RESULT) == reference['sha256'],
        'completed training result binding required')
    state = json.loads(_file(root, TRAINING_RESULT).read_text())
    result, update = state.get('result', {}), state.get('training_result', {})
    saved = state.get('saved_adapter', {})
    require(state.get('phase') == 'complete' and state.get('adapter_saved') is True
        and state.get('baseline') == 'B3_success_only_SFT' and state.get('method_experiences_used') == 0
        and result.get('training_complete') is True and result.get('adapter_saved') is True
        and result.get('optimizer_steps') == update.get('optimizer_steps') == 86
        and result.get('examples') == 344 and result.get('supervised_tokens') == 11085
        and update.get('finite_gradients') is update.get('finite_parameters') is True
        and _sha(saved.get('source_adapter_sha256'))
        and update.get('final_adapter_sha256') == saved['source_adapter_sha256']
        and plan.get('saved_adapter') == saved, 'incomplete or different trained adapter')
    path = root / ADAPTER
    require(path.is_dir() and not path.is_symlink() and path.resolve() == path
        and saved.get('output_directory') == plan.get('adapter_directory') == str(path)
        and saved.get('status') == 'complete' and saved.get('complete') is saved.get('adapter_saved') is True
        and saved.get('format') == 'safetensors' and saved.get('tensor_count') == 608
        and saved.get('total_parameter_count') == 23760896
        and saved.get('adapter_tensor_bytes') == 95043584
        and saved.get('adapter_dtypes') == ['torch.float32']
        and _sha(saved.get('source_peft_state_sha256'))
        and saved['source_peft_state_sha256'] == saved.get('saved_peft_state_sha256'),
        'registered FP32 adapter export required')
    files = saved.get('files', {})
    required = {'adapter_config.json', 'adapter_model.safetensors'}
    require(required <= files.keys() <= required | {'README.md'}
        and {p.name for p in path.iterdir()} == set(files), 'adapter inventory differs')
    for name, expected in files.items():
        item = path / name
        require(item.is_file() and not item.is_symlink() and _sha(expected.get('sha256'))
            and type(expected.get('bytes')) is int and expected['bytes'] > 0
            and item.stat().st_size == expected['bytes'] and digest(item) == expected['sha256'],
            'adapter file differs: ' + name)
    config = json.loads((path / 'adapter_config.json').read_text())
    expected = {'base_model_name_or_path': str(MODEL), 'peft_type': 'LORA', 'task_type': 'CAUSAL_LM',
        'r': 8, 'lora_alpha': 16, 'lora_dropout': 0, 'bias': 'none', 'inference_mode': True}
    require(all(config.get(k) == v == saved.get('config', {}).get(k) for k, v in expected.items())
        and saved['config'].get('target_module_count') == 304, 'adapter configuration differs')


def validate_registration(root, plan, *, arm, model_id):
    """CPU-only provenance checks; never treats a future gate path as proof."""
    root = Path(root).resolve()
    require(arm in MODEL_IDS and model_id == MODEL_IDS[arm]
        and plan.get('id') == ID and plan.get('status') == 'registered'
        and plan.get('model_ids') == MODEL_IDS and plan.get('recipe') == RECIPE
        and plan.get('port') == PORT, 'unregistered service/arm/model/recipe')
    require(plan.get('parent_service_plan') == PARENT_SERVICE_PLAN
        and plan.get('parent_service_plan_sha256') == PARENT_SERVICE_SHA
        and digest(_file(root, PARENT_SERVICE_PLAN)) == PARENT_SERVICE_SHA,
        'parent service registration changed')
    parent = json.loads((root / PARENT_SERVICE_PLAN).read_text())
    sources, inputs = plan.get('source_sha256', {}), plan.get('input_sha256', {})
    gate = plan.get('inference_gate_design', {})
    require(REQUIRED_SOURCES | set(parent['source_sha256']) <= sources.keys()
        and all(sources[name] == value for name, value in parent['source_sha256'].items()),
        'missing or changed inherited service source binding')
    require(gate.get('path') == GATE_DESIGN and gate.get('expected_output') == GATE_OUTPUT
        and _sha(gate.get('sha256')) and inputs.get(GATE_DESIGN) == gate['sha256']
        and GATE_OUTPUT not in inputs
        and {PARENT_SERVICE_PLAN, TRAINING_RESULT, GATE_DESIGN, 'config/gpu-policy.json'} <= inputs.keys(),
        'gate design/provenance binding required; future gate proof cannot authorize startup')
    for mapping in (sources, inputs):
        for name, expected in mapping.items():
            require(_sha(expected) and digest(_file(root, name)) == expected,
                    'bound service file changed: ' + name)
    runtime = plan.get('runtime_sha256', {})
    require(all(runtime.get(name) == value for name, value in parent['runtime_sha256'].items()),
            'inherited runtime binding missing')
    for name, expected in runtime.items():
        require(Path(name).is_absolute() and _sha(expected) and digest(Path(name)) == expected,
                'service runtime changed: ' + name)
    _validate_export(root, plan)  # Both arms bind the same actual completed export.


def preflight(expected_sha, model_id, *, arm):
    require(_sha(expected_sha) and digest(_file(ROOT, PLAN)) == expected_sha,
            'service registration changed')
    plan = json.loads((ROOT / PLAN).read_text())
    validate_registration(ROOT, plan, arm=arm, model_id=model_id)
    # Includes policy, Python/package/runtime and complete base weight identity.
    # This path imports no Torch and neither acquires nor queries a GPU.
    parent_preflight(PARENT_SERVICE_SHA)
    return plan


def _validate_adapter_reload(report, saved):
    require(report.get('status') == 'verified' and report.get('tensor_count') == 608
        and report.get('source_adapter_sha256') == saved['source_adapter_sha256']
        and report.get('adapter_state_sha256') == saved['saved_peft_state_sha256']
        and report.get('base_parameter_storage_unchanged') is report.get('all_parameters_frozen') is True
        and report.get('active_adapters') == ['default'] and report.get('merged') is False
        and report.get('use_cache') is True and report.get('adapter_dtypes') == ['torch.float32'],
        'native FP32 frozen adapter reload differs')


@contextmanager
def loaded_engine(plan_sha, model_id, *, arm, admitted=None):
    configure_allocator('expandable')
    plan = preflight(plan_sha, model_id, arm=arm)
    from wll.gpu_policy import acquire_gpu, verify_cuda

    with acquire_gpu() as gpu:
        if admitted is not None:
            admitted(gpu)
        import torch
        from transformers import AutoTokenizer

        from scripts.load_hybrid_adapter import load_hybrid_adapter
        from scripts.nf4_loader import inspect_nf4, load_nf4
        from scripts.text_chunked_prefill import install_text_chunked_prefill

        verify_cuda(torch)
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = load_nf4(MODEL, 'cuda:0')
        actual = inspect_nf4(model, 'cuda:0')
        require(actual['linear4bit_layers'] == 496, 'different quantization coverage')
        install_text_chunked_prefill(model)  # PEFT.generate delegates to this base.
        model.requires_grad_(False)
        model.eval()
        model.config.use_cache = model.config.text_config.use_cache = True
        adapter = None
        if arm == 'B3':
            model, adapter = load_hybrid_adapter(model, plan['adapter_directory'],
                saved_record=plan['saved_adapter'], expected_base_model_name_or_path=str(MODEL))
            _validate_adapter_reload(adapter, plan['saved_adapter'])
        require(all(not p.requires_grad and str(p.device) == 'cuda:0' for p in model.parameters()),
                'all inference parameters must be frozen on logical GPU0')
        torch.cuda.synchronize()
        loading_peak = torch.cuda.max_memory_reserved()
        require(loading_peak <= CAP, 'model_loading_exceeded_cap')
        allocator = allocator_details(torch, 'expandable', require_segments=True)
        backend = {'server_sha256': digest(Path(__file__)), 'service_plan_sha256': plan_sha,
            'arm': arm, 'recipe': RECIPE, 'gpu_policy': gpu,
            'training_result': plan['training_result'], 'adapter': adapter,
            'saved_adapter': plan['saved_adapter'], 'quantization': quantization_metadata(actual),
            'loading_peak_reserved_bytes': loading_peak, 'allocator_backend': allocator['backend'],
            'expandable_segments': allocator['settings']['expandable_segments'],
            'source_sha256': plan['source_sha256'],
            'packages': {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'bitsandbytes', 'accelerate', 'safetensors', 'peft')},
            'files': {name: digest(MODEL / name) for name in
                ('config.json', 'tokenizer_config.json', 'generation_config.json', 'weight-source.json')}}
        yield ChatEngine(model, tokenizer, model_id, backend)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--arm', choices=list(MODEL_IDS), required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    require(args.model_id == MODEL_IDS[args.arm] and args.port == PORT,
            'invalid model ID or unregistered port')
    if args.check:
        configure_allocator('expandable')
        preflight(args.plan_sha256, args.model_id, arm=args.arm)
        print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'arm': args.arm, 'recipe': RECIPE}))
        return
    with loaded_engine(args.plan_sha256, args.model_id, arm=args.arm) as engine:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(engine))
        print(json.dumps({'ready': True, 'model': args.model_id, 'arm': args.arm, 'port': args.port,
            'pid': os.getpid(), 'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19]}),
            flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
        if engine.failed:
            raise RuntimeError('model service stopped after execution failure')


if __name__ == '__main__':
    main()
