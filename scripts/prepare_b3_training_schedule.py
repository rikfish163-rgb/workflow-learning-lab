"""Prepare, but do not authorize or run, the audited B3 success-only SFT pilot."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import scripts.audit_distill_baseline_004 as audited
import scripts.run_qwen38_distill_baseline_004 as base
from scripts.encode_no_thinking_targets import encode_target
from scripts.serve_qwen38_idle180 import DOWNLOAD
from scripts.serve_qwen38_idle180 import preflight as service_preflight
from wll.cli import read_episodes
from wll.protocol import fingerprint

ROOT = Path(__file__).resolve().parents[1]
CORPUS = 'artifacts/qwen38-distill-baseline-corpus-004'
OUTPUT = 'artifacts/b3-training-schedule-preparation-001'
PLAN_SHA = '188e89219b9b59ddbba0e20b427f8aa5c23295bb156ebf1a5f1a73a2605fdfcb'
AUDIT_SHA = '6649fedbeb5c4d5a8e8e859b9ddbba26ded11d24fe8459e36e8cee0cbd5e79a9'
TOKEN_AUDIT_SHA = '2ce2fc2b0ffc83f66feb8420a6f40a9a0735f4ec2232c40eb713b450cf4df75e'
SFT_SHA = '27850327ef503f4945f395be349250aee34f6aa360087c31769faf445945b9f8'
EPISODES_SHA = '1c070a56f4ab488e30791f96fbfc3a789f3ec1c4169fc213377c757c557f52c7'
ENCODER_SHA = 'b56cb336ab991024e2d4ffaa77409cb1f061dc5c43d40fcedba4fc196f81a3f8'
SEED = 17
GROUP_SIZE = 4
MAX_TOKENS = 16384
EXPECTED_ROWS = 344
EXPECTED_TARGETS = 11085
EXPECTED_INPUT = 2028299


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def token_hash(tokens):
    require(isinstance(tokens, list) and all(type(token) is int for token in tokens), 'invalid token sequence')
    return digest_bytes(json.dumps(tokens, separators=(',', ':')).encode())


def validate_partition(tasks, partition):
    ids = [task['task_id'] for task in tasks]
    roles = partition['task_ids']
    require(ids and len(ids) == len(set(ids)) and ids == roles['distill']
        and all(task['split'] == 'train' and task['scenario_id'] == task['task_id'].rsplit('_', 1)[0]
            for task in tasks), 'original ordered train/distill partition required')
    own = {task['scenario_id'] for task in tasks}
    other = {name.rsplit('_', 1)[0] for role, values in roles.items() if role != 'distill' for name in values}
    require(own.isdisjoint(other), 'scenario leakage across partition roles')


def validate_row_scope(rows, episodes, tasks):
    allowed = {task['task_id']: task for task in tasks}
    by_id = {episode.task.task_id: episode for episode in episodes}
    require(len(by_id) == len(episodes) == len(tasks), 'unique complete original episodes required')
    seen = set()
    for row in rows:
        task_id, step = row['task_id'], row['step_index']
        require(task_id in allowed and task_id in by_id and type(step) is int and step >= 0,
            'unknown task or invalid step')
        episode, task = by_id[task_id], allowed[task_id]
        key = (task_id, step)
        require(key not in seen and step < len(episode.steps) and episode.success is True
            and row['scenario_id'] == task['scenario_id'] and row['split'] == task['split'] == 'train'
            and row['source_model'] == episode.model_id and row['source_experience_id'] is None
            and row['source_episode_hash'] == fingerprint(episode.to_dict())
            and row['target_role'] == 'assistant' and row['target_selection'] == 'exclude_observed_errors_v2',
            'duplicate, leaking or non-successful example')
        seen.add(key)
    return by_id


def derive_index(rows, raw_rows, samples, episodes, tokenizer, requests=None):
    require(rows and len(rows) == len(raw_rows) == len(samples), 'complete row and token coverage required')
    index, seen_pairs, seen_hashes = [], set(), set()
    for number, (row, raw, sample) in enumerate(zip(rows, raw_rows, samples, strict=True)):
        task_id, step = row['task_id'], row['step_index']
        key, row_sha = (task_id, step), digest_bytes(raw)
        require(key not in seen_pairs and row_sha not in seen_hashes and type(step) is int and step >= 0
            and sample['task_id'] == task_id and sample['step_index'] == step,
            'duplicate or mismatched corpus row')
        require(task_id in episodes and step < len(episodes[task_id].steps), 'unknown episode or step')
        encoded = encode_target(tokenizer, row['messages'], MAX_TOKENS)
        ids, labels, attention = encoded['input_ids'], encoded['labels'], encoded['attention_mask']
        total = len(ids)
        supervised = sum(value != -100 for value in labels)
        prompt = total - supervised
        require(len(labels) == len(attention) == total and attention == [1] * total
            and 0 < prompt < total <= MAX_TOKENS and labels[:prompt] == [-100] * prompt
            and labels[prompt:] == ids[prompt:] and sample['total_tokens'] == total
            and sample['supervised_tokens'] == supervised and sample['fits_8192'] is (total <= 8192)
            and sample['fits_16384'] is True and episodes[task_id].steps[step].prompt_tokens == prompt,
            'token audit or original serving prompt length differs')
        if requests is not None:
            request = requests.get(key)
            require(request is not None and request['prompt_tokens'] == prompt
                and request['completion_tokens'] == episodes[task_id].steps[step].completion_tokens,
                'original serving request or generation usage differs')
        index.append({'source_index': number, 'row_sha256': row_sha, 'input_ids_sha256': token_hash(ids),
            'labels_sha256': token_hash(labels), 'input_tokens': total, 'prompt_tokens': prompt,
            'supervised_tokens': supervised})
        seen_pairs.add(key)
        seen_hashes.add(row_sha)
    return index


def build_schedule(index):
    require(index and len(index) % GROUP_SIZE == 0, 'complete groups of four required')
    require([row['source_index'] for row in index] == list(range(len(index))), 'ordered source index required')
    order = list(range(len(index)))
    random.Random(SEED).shuffle(order)
    groups = []
    for start in range(0, len(order), GROUP_SIZE):
        members = order[start:start + GROUP_SIZE]
        groups.append({'update_index': len(groups), 'sample_indices': members,
            'input_tokens': sum(index[i]['input_tokens'] for i in members),
            'supervised_tokens': sum(index[i]['supervised_tokens'] for i in members)})
    return {'seed': SEED, 'shuffle': 'Python random.Random(seed).shuffle; MT19937',
        'microbatch_size': 1, 'gradient_accumulation_steps': GROUP_SIZE,
        'optimizer_updates': len(groups), 'order_sha256': token_hash(order), 'groups': groups}


def verify_preparation(expected_index, expected_schedule, saved_index, saved_schedule):
    require(saved_index == expected_index and saved_schedule == expected_schedule
        and expected_schedule == build_schedule(expected_index), 'schedule or sample index changed')


def quantiles(values):
    values = sorted(values)
    require(values and all(type(v) is int and v > 0 for v in values), 'positive token counts required')
    return {'minimum': values[0], **{f'p{percent}': values[math.ceil(percent * len(values) / 100) - 1]
        for percent in (50, 90, 95, 99)}, 'maximum': values[-1]}


def summarize(index):
    inputs = [row['input_tokens'] for row in index]
    targets = [row['supervised_tokens'] for row in index]
    return {'examples': len(index), 'input_tokens_one_epoch': sum(inputs),
        'supervised_tokens_one_epoch': sum(targets), 'input_token_quantiles': quantiles(inputs),
        'target_token_quantiles': quantiles(targets), 'quantile_definition': 'nearest rank: ceil(p*n)-1, zero based',
        'longest_input_source_indices': [i for i, value in enumerate(inputs) if value == max(inputs)],
        'maximum_target_source_indices': [i for i, value in enumerate(targets) if value == max(targets)],
        'samples_above_8192': sum(value > 8192 for value in inputs),
        'samples_above_16384': sum(value > 16384 for value in inputs)}


def verify_tokenizer_files(model, download, token_audit):
    files = audited.validate_tokenizer_files(model, download)
    require(files == token_audit['tokenizer_files_sha256']
        and Path(model).resolve(strict=True) == Path(token_audit['tokenizer_path']).resolve(strict=True),
        'tokenizer provenance differs from the serving-token audit')
    return files


def bound_corpus(root):
    root = Path(root).resolve(strict=True)
    folder = root / CORPUS
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve().is_relative_to(root),
        'registered private corpus directory required')
    expected = {'audit.json': AUDIT_SHA, 'token-audit.json': TOKEN_AUDIT_SHA,
                'sft.jsonl': SFT_SHA, 'episodes.jsonl': EPISODES_SHA}
    for name, sha in expected.items():
        path = folder / name
        require(not path.is_symlink() and path.is_file() and digest(path) == sha, 'frozen corpus changed: ' + name)
    require(digest(root / 'scripts/encode_no_thinking_targets.py') == ENCODER_SHA, 'encoder changed')
    proof, token = base.read(folder / 'audit.json'), base.read(folder / 'token-audit.json')
    require(proof['phase'] == 'complete' and proof['plan_sha256'] == PLAN_SHA
        and proof['known_actor_outcomes'] == 27 and proof['operational_attempt_records'] == 28
        and proof['all_resources_released'] is True and proof['unknown_generated_response_cost'] is False
        and proof['new_training_started'] is False and proof['accepted_experiences'] == 0
        and proof['result']['successes'] == 23 and proof['curation']['exported_targets'] == EXPECTED_ROWS
        and proof['auditor_sha256'] == digest(root / 'scripts/audit_distill_baseline_004.py')
        and proof['encoder_sha256'] == token['encoder_sha256'] == ENCODER_SHA
        and proof['sft_sha256'] == token['sft_sha256'] == SFT_SHA
        and proof['episodes_sha256'] == EPISODES_SHA and token['examples'] == EXPECTED_ROWS
        and token['all_training_serving_prefixes_identical'] is True,
        'complete audited collection and tokenizer evidence required')
    audited.validate_hashes(root, proof['source_sha256'])
    audited.validate_hashes(root, proof['input_sha256'])
    plan, tasks = base.load_registration(root, root / base.PLAN, PLAN_SHA)
    partition = base.read(root / plan['partition'])
    validate_partition(tasks, partition)
    raw_rows = (folder / 'sft.jsonl').read_bytes().splitlines()
    require(raw_rows and all(raw_rows), 'empty corpus or blank sample')
    rows = [json.loads(line) for line in raw_rows]
    episodes = read_episodes(folder / 'episodes.jsonl')
    curation, reconstructed = audited.curate_rows(episodes, tasks, plan)
    require(reconstructed == rows and curation == proof['curation'] and len(rows) == EXPECTED_ROWS,
        'private corpus differs from complete audited trajectories')
    episode_map = validate_row_scope(rows, episodes, tasks)
    requests = {}
    for segment in proof['segments']:
        for row in segment['generation']['requests']:
            key = (row['task_id'], row['step_index'])
            require(key not in requests, 'duplicate generation request')
            requests[key] = row
    require(len(requests) == proof['generated_requests'], 'generation request coverage differs')
    return plan, proof, token, rows, raw_rows, episode_map, requests


def prepare(root, tokenizer_path):
    root = Path(root).resolve(strict=True)
    source_sha = digest(__file__)
    plan, proof, token, rows, raw_rows, episodes, requests = bound_corpus(root)
    require(plan['input_sha256'][DOWNLOAD] == digest(root / DOWNLOAD), 'model download binding differs')
    files = verify_tokenizer_files(tokenizer_path, base.read(root / DOWNLOAD), token)
    # The recorded service preflight validates the original CPU runtime and model provenance.
    service_preflight(plan['service_plan_sha256'])
    from transformers import AutoTokenizer
    from transformers import __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    index = derive_index(rows, raw_rows, token['samples'], episodes, tokenizer, requests)
    schedule = build_schedule(index)
    summary = summarize(index)
    require(summary['examples'] == EXPECTED_ROWS and summary['input_tokens_one_epoch'] == EXPECTED_INPUT
        and summary['supervised_tokens_one_epoch'] == EXPECTED_TARGETS and schedule['optimizer_updates'] == 86
        and summary['samples_above_16384'] == 0, 'complete original token workload differs')
    require(digest(__file__) == source_sha and bound_corpus(root)[:3] == (plan, proof, token)
        and verify_tokenizer_files(tokenizer_path, base.read(root / DOWNLOAD), token) == files,
        'source, corpus or tokenizer changed during preparation')
    proposal = {'status': 'prepared_for_review_not_training_authorization',
        'objective': 'B3 success-only SFT baseline; not the recovery method',
        'proposal': {'epochs': 1, 'seed': SEED, 'microbatch_size': 1, 'gradient_accumulation_steps': GROUP_SIZE,
            'optimizer_updates': 86, 'learning_rate': 0.0001, 'optimizer': 'AdamW', 'weight_decay': 0.0,
            'betas': [0.9, 0.999], 'epsilon': 1e-8, 'learning_rate_schedule': 'constant',
            'foreach': False, 'gradient_clip_norm': 1.0, 'context_limit': MAX_TOKENS,
            'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': 0, 'lora_bias': 'none',
            'base': 'existing registered 27B NF4', 'packing': False, 'truncation': False,
            'loss_weighting_proposal': 'Sum supervised-token cross-entropy within each group and divide by the '
                'group supervised-token count before one optimizer update; clip after all four backwards.'},
        'summary': summary, 'corpus_audit_sha256': AUDIT_SHA, 'token_audit_sha256': TOKEN_AUDIT_SHA,
        'sft_sha256': SFT_SHA, 'episodes_sha256': EPISODES_SHA, 'collection_plan_sha256': PLAN_SHA,
        'encoder_sha256': ENCODER_SHA, 'materializer_sha256': source_sha,
        'tokenizer_files_sha256': files, 'python_version': sys.version.split()[0],
        'transformers_version': transformers_version,
        'scope': 'CPU-only schedule preparation; no model load, training, GPU admission, automatic truncation, '
            'heldout evaluation, method claim or authorization. Future training time and cost remain unknown.'}
    return index, schedule, proposal


def encoded_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def write_or_verify(target, index, schedule, proposal, verify):
    index_bytes, schedule_bytes = encoded_json(index), encoded_json(schedule)
    proposal = {**proposal, 'sample_index_sha256': digest_bytes(index_bytes),
                'update_groups_sha256': digest_bytes(schedule_bytes)}
    contents = {'sample-index.json': index_bytes, 'update-groups.json': schedule_bytes,
                'proposal.json': encoded_json(proposal)}
    if verify:
        require(target.is_dir() and not target.is_symlink()
            and {p.name for p in target.iterdir()} == set(contents), 'prepared file coverage differs')
        for name, expected in contents.items():
            path = target / name
            require(not path.is_symlink() and path.is_file() and path.read_bytes() == expected,
                'prepared schedule or metadata changed: ' + name)
        return
    require(not target.exists() and not target.is_symlink(), 'schedule output already exists')
    target.mkdir(mode=0o700)
    for name, content in contents.items():
        with (target / name).open('xb') as stream:
            stream.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-path', required=True)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only visibility required')
    target = (ROOT / args.output).resolve()
    require(target == (ROOT / OUTPUT).resolve() and not (ROOT / OUTPUT).is_symlink(),
        'registered preparation output path required')
    if not args.verify:
        require(not target.exists(), 'schedule output already exists')
    index, schedule, proposal = prepare(ROOT, args.tokenizer_path)
    write_or_verify(target, index, schedule, proposal, args.verify)
    print(json.dumps({'status': 'verified' if args.verify else 'prepared_for_review',
        'examples': len(index), 'optimizer_updates': schedule['optimizer_updates'], **proposal['summary']}))


if __name__ == '__main__':
    main()
