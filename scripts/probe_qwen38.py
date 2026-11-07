"""Registered27B synthetic context/output probe; one attempt, no task scoring."""
import argparse
import gc
import importlib.metadata
import json
import os
import stat
import sys
import time
from pathlib import Path

from scripts.download_qwen38 import MODEL, REPO, REVISION
from scripts.probe_allocator import allocator_details, configure_allocator, counter_delta, memory
from scripts.probe_chunked_prefill import CAP, PYTHON, ROOT, digest, require
from wll.storage import atomic_json

ID = 'qwen38-27b-nf4-feasibility-001'
PLAN = 'artifacts/qwen38-27b-feasibility-plan-001.json'
DESIGN = 'artifacts/qwen38-27b-feasibility-design-001.json'
WEIGHT_PROOF = 'artifacts/qwen38-27b-weight-verification-001.json'
CASES = [{'input_tokens': n, 'output_tokens': t}
         for n, t in [(2048, 32), (8192, 1024), (16384, 1024), (31744, 1024)]]
RUNTIME = 'artifacts/qwen35-nf4-runtime-audit-003.json'


class FiniteScoreGuard:
    """Return scores unchanged and retain only integer counters, never tensors."""

    def __init__(self, input_tokens, output_tokens, vocab_size, progress=None):
        require(all(type(n) is int and n > 0 for n in (input_tokens, output_tokens, vocab_size)),
                'positive integer generation sizes required')
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.vocab_size = vocab_size
        self.checks = 0
        self.progress = progress

    def __call__(self, input_ids, scores):
        import torch

        require(self.checks < self.output_tokens
                and tuple(input_ids.shape) == (1, self.input_tokens + self.checks)
                and tuple(scores.shape) == (1, self.vocab_size), 'unexpected decode step or score shape')
        require(bool(torch.isfinite(scores).all()), 'nonfinite candidate scores')
        self.checks += 1
        if self.progress and (self.checks % 64 == 0 or self.checks == self.output_tokens):
            self.progress(self.checks)
        return scores


def generation_options(ids, mask, output_tokens, pad_token_id, guard, chunk_size=512):
    # Explicit kwargs override defaults carried by the model's generation_config,
    # including None values that would otherwise inherit its EOS/stop settings.
    return {'input_ids': ids, 'attention_mask': mask, 'do_sample': False,
            'max_new_tokens': output_tokens, 'min_new_tokens': output_tokens,
            'use_cache': True, 'num_beams': 1, 'num_return_sequences': 1,
            'eos_token_id': None, 'forced_eos_token_id': None, 'forced_bos_token_id': None,
            'pad_token_id': pad_token_id, 'stop_strings': None, 'max_time': None,
            'bad_words_ids': None, 'suppress_tokens': None, 'begin_suppress_tokens': None,
            'sequence_bias': None, 'no_repeat_ngram_size': 0, 'repetition_penalty': 1.0,
            'temperature': 1.0, 'top_p': 1.0, 'top_k': 0,
            'disable_compile': True, 'logits_to_keep': 1, 'prefill_chunk_size': chunk_size,
            'return_dict_in_generate': True, 'output_logits': False, 'output_scores': False,
            'output_attentions': False, 'output_hidden_states': False, 'logits_processor': [guard]}


def validate_case(result, expected):
    n = expected['output_tokens']
    require(all(result.get(k) == v for k, v in expected.items()), 'different registered case')
    require(result['produced_tokens'] == result['score_checks'] == n
            and result['all_scores_finite'] is True and len(result['output_token_ids']) == n,
            'incomplete or nonfinite generation')
    require(type(result['vocab_size']) is int and result['vocab_size'] > 0
            and all(type(i) is int and 0 <= i < result['vocab_size'] for i in result['output_token_ids']),
            'invalid generated token IDs')
    require(0 <= result['peak_allocated_bytes'] <= result['peak_reserved_bytes'] <= CAP,
            'invalid or excessive memory peak')
    require(result['generation_counter_delta']['num_ooms'] == 0, 'OOM during generation')


def assess_complete(cases):
    require(len(cases) == len(CASES), 'not all registered cases completed')
    for result, expected in zip(cases, CASES, strict=True):
        validate_case(result, expected)
    return {'generations': len(CASES), 'input_tokens': sum(x['input_tokens'] for x in CASES),
            'output_tokens': sum(x['output_tokens'] for x in CASES)}


def verify_model_provenance(model, download, proof):
    model = Path(model)
    require(proof['phase'] == 'complete' and proof['hf_repo'] == REPO
            and proof['hf_revision'] == REVISION and proof['model_directory'] == str(model)
            and proof['download_writer_exited'] is True, 'incomplete or different weight verification')
    verified = {x['name']: x for x in proof['files']}
    require(len(verified) == len(proof['files']) == len(download['files'])
            and set(verified) == {x['name'] for x in download['files']}, 'incomplete file coverage')
    for expected in download['files']:
        actual = verified[expected['name']]
        require(actual['sha256'] == expected['sha256'] and actual['size'] == expected['size'],
                'different verified digest or size')
        st = (model / expected['name']).lstat()
        require(stat.S_ISREG(st.st_mode) and actual['stat'] == {
            'size': st.st_size, 'inode': st.st_ino, 'mtime_ns': st.st_mtime_ns,
            'ctime_ns': st.st_ctime_ns}, 'model file changed after full verification')


def load_registration(root, expected_sha):
    root = Path(root)
    require(digest(root / PLAN) == expected_sha, 'plan digest changed')
    plan = json.loads((root / PLAN).read_text())
    require(plan['id'] == ID and plan['status'] == 'registered'
            and plan['design'] == DESIGN and plan['design_sha256'] == digest(root / DESIGN),
            'unregistered or changed design')
    required = {'scripts/probe_qwen38.py', 'scripts/run_qwen38_probe.py', 'scripts/nf4_loader.py',
                'scripts/probe_allocator.py', 'scripts/probe_chunked_prefill.py',
                'scripts/download_qwen38.py', 'scripts/resumable_ranges.py',
                'scripts/text_chunked_prefill.py', 'scripts/gpu_idle_watch.py',
                'wll/gpu_policy.py', 'wll/gpu_idle.py', 'wll/storage.py'}
    require(required <= plan['source_sha256'].keys(), 'missing source binding')
    require({WEIGHT_PROOF, RUNTIME, 'config/gpu-policy.json'} <= plan['input_sha256'].keys(),
            'missing provenance or runtime binding')
    for field in ('source_sha256', 'input_sha256'):
        for name, sha in plan[field].items():
            require(digest(root / name) == sha, f'bound file changed: {name}')
    design = json.loads((root / DESIGN).read_text())
    require([{k: c[k] for k in ('input_tokens', 'output_tokens')} for c in design['cases_in_order']] == CASES
            and design['allocator_cap_bytes'] == CAP and design['max_gpu_job_seconds'] == 1800
            and design['prefill_chunk_size'] == 512 and design['batch_size'] == 1
            and design['attempts'] == 1 and design['automatic_retry'] is False
            and design['automatic_lower_context_fallback'] is False
            and design['repo'] == REPO and design['revision'] == REVISION
            and design['model_directory'] == str(MODEL), 'different experiment configuration')
    return plan, design


def preflight(expected_sha):
    plan, design = load_registration(ROOT, expected_sha)
    audit = json.loads((ROOT / RUNTIME).read_text())
    require(sys.prefix == audit['runtime']['python_prefix'] == str(Path(PYTHON).parent.parent),
            'different Python environment')
    for name, sha in {**audit['runtime_sha256'], **plan['runtime_sha256']}.items():
        require(digest(Path(name)) == sha, f'runtime changed: {name}')
    for name, version in audit['runtime']['packages'].items():
        require(importlib.metadata.version(name) == version, f'different package: {name}')
    path = ROOT / design['download_plan']
    require(digest(path) == design['download_plan_sha256'], 'download registration changed')
    verify_model_provenance(MODEL, json.loads(path.read_text()),
                            json.loads((ROOT / WEIGHT_PROOF).read_text()))
    return design


def measure(model, ids, mask, case, pad_token_id, progress):
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = memory(torch)
    guard = FiniteScoreGuard(case['input_tokens'], case['output_tokens'],
                             model.config.text_config.vocab_size, progress)
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**generation_options(ids, mask, case['output_tokens'], pad_token_id, guard))
    torch.cuda.synchronize()
    elapsed, after = time.monotonic() - start, memory(torch)
    require(output.logits is None and output.scores is None, 'unexpected retained vocabulary tensors')
    result = {**case, 'produced_tokens': output.sequences.shape[1] - case['input_tokens'],
              'output_token_ids': output.sequences[0, case['input_tokens']:].tolist(),
              'score_checks': guard.checks, 'all_scores_finite': True, 'vocab_size': guard.vocab_size,
              'seconds': elapsed, 'before_generation': before, 'after_generation': after,
              'peak_allocated_bytes': after['allocated_bytes.all.peak'],
              'peak_reserved_bytes': after['reserved_bytes.all.peak'],
              'generation_counter_delta': counter_delta(before, after)}
    del output, guard
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result['after_cleanup'] = memory(torch)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    env = configure_allocator('expandable')
    design = preflight(args.plan_sha256)
    if args.check:
        print(json.dumps({'preflight': 'passed', 'uses_gpu': False, 'id': ID, 'cases': CASES}))
        return
    path = ROOT / f'artifacts/{ID}.json'
    if path.exists():
        raise FileExistsError('attempt exists; no automatic retry')
    from wll.gpu_policy import acquire_gpu, verify_cuda

    state = {'id': ID, 'phase': 'waiting_admission', 'pid': os.getpid(),
             'start_ticks': Path('/proc/self/stat').read_text().split(')', 1)[1].split()[19],
             'started_epoch': time.time(), 'plan_sha256': args.plan_sha256,
             'allocator_environment': env, 'attempted_measurements': 0, 'cases': [],
             'unattempted_cases': CASES, 'benchmark_evaluation': False, 'training': False}
    atomic_json(path, state)
    try:
        with acquire_gpu() as admission:
            import torch
            from transformers import AutoTokenizer

            from scripts.nf4_loader import inspect_nf4, load_nf4
            from scripts.text_chunked_prefill import install_text_chunked_prefill

            verify_cuda(torch)
            torch.set_num_threads(2)
            torch.manual_seed(17)
            state.update(phase='loading', gpu=admission)
            atomic_json(path, state)
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            model = load_nf4(MODEL, 'cuda:0')
            state['quantization_actual'] = inspect_nf4(model, 'cuda:0')
            require(state['quantization_actual']['linear4bit_layers'] == design['expected_linear4bit_layers'],
                    'unexpected NF4 layer coverage')
            install_text_chunked_prefill(model)
            torch.cuda.synchronize()
            state.update(phase='probing', load_seconds=time.monotonic() - start,
                         loading_memory=memory(torch),
                         actual_allocator=allocator_details(torch, 'expandable', require_segments=True))
            atomic_json(path, state)
            require(state['loading_memory']['reserved_bytes.all.peak'] <= CAP, 'loading exceeded cap')
            seed = tokenizer(design['input_seed_text'], add_special_tokens=False)['input_ids']
            require(bool(seed), 'empty input seed')
            for index, case in enumerate(CASES):
                length = case['input_tokens']
                values = (seed * (length // len(seed) + 1))[:length]
                ids = torch.tensor([values], dtype=torch.long, device='cuda:0')
                mask = torch.ones_like(ids)
                state.update(current_case=case, attempted_measurements=index + 1,
                             unattempted_cases=CASES[index + 1:], current_score_checks=0)
                atomic_json(path, state)

                def progress(count):
                    state.update(current_score_checks=count, checked_epoch=time.time())
                    atomic_json(path, state)

                result = measure(model, ids, mask, case, tokenizer.eos_token_id, progress)
                state['cases'].append(result)
                atomic_json(path, state)
                validate_case(result, case)
                del ids, mask
                gc.collect()
                torch.cuda.empty_cache()
            state.update(phase='complete', finished_epoch=time.time(),
                         aggregate=assess_complete(state['cases']), completion_gate_pending_release=True)
            atomic_json(path, state)
    except BaseException as exc:
        state.update(phase='failed', error_type=type(exc).__name__, error=str(exc)[:1200],
                     finished_epoch=time.time())
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            try:
                state['failure_memory'] = memory(torch)
            except Exception as secondary:
                state['failure_memory_error'] = type(secondary).__name__
        atomic_json(path, state)
        raise


if __name__ == '__main__':
    main()
