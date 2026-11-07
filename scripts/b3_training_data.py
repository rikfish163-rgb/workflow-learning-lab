"""Regenerate verified B3 microbatches without loading a model or using CUDA.

The returned token sequences are private derived data. This module neither
authorizes nor starts training; the caller owns GPU admission and plan checks.
"""

import json
import time
from pathlib import Path

from scripts import prepare_b3_training_schedule as prep
from scripts.hybrid_target_updates import validate_groups

SCHEDULE_SHA256 = {
    'sample-index.json': 'ef792c743f96ff1d96dd2ddca47a85322d4a6b9ffb700cff68b64685ba614aeb',
    'update-groups.json': 'caa852299dde7fbac76f2ec701074fd8a40500ae9a9cdff1327e2b978a12e8c4',
    'proposal.json': 'd798f92a2832bedb63477beb45f406a7b5f2f27631df1b85a5ba6467ffa69709',
}
MATERIALIZER_SHA256 = '39de0168b346b8ca7426321815447f2299b83fe6b6e1cb0c8c6be2b5b17a6e61'


def snapshot_schedule(root):
    """Read only the three registered files, rejecting symlinks and drift."""
    root = Path(root).resolve(strict=True)
    current = root
    for part in Path(prep.OUTPUT).parts:
        current = current / part
        prep.require(not current.is_symlink() and current.is_dir(), 'invalid prepared schedule directory')
    prep.require({p.name for p in current.iterdir()} == set(SCHEDULE_SHA256), 'prepared schedule coverage differs')
    values = {}
    for name, sha in SCHEDULE_SHA256.items():
        path = current / name
        prep.require(not path.is_symlink() and path.is_file(), 'invalid prepared schedule file: ' + name)
        content = path.read_bytes()
        prep.require(prep.digest_bytes(content) == sha, 'prepared schedule hash differs: ' + name)
        values[name] = content
    return values


def assemble_groups(rows, raw_rows, tokenizer, index, schedule):
    """Re-encode every original row and verify its identity before grouping."""
    prep.require(isinstance(rows, list) and rows and isinstance(raw_rows, list)
        and isinstance(index, list) and len(rows) == len(raw_rows) == len(index),
        'complete original row and index coverage required')
    prep.require(schedule == prep.build_schedule(index), 'registered schedule or row order differs')
    encoded_rows, seen_hashes = [], set()
    for number, (row, raw, entry) in enumerate(zip(rows, raw_rows, index, strict=True)):
        prep.require(isinstance(raw, bytes) and json.loads(raw) == row, 'raw row differs from parsed row')
        row_hash = prep.digest_bytes(raw)
        prep.require(row_hash not in seen_hashes and entry['source_index'] == number
            and entry['row_sha256'] == row_hash, 'duplicate, missing or changed original row')
        seen_hashes.add(row_hash)
        encoded = prep.encode_target(tokenizer, row['messages'], prep.MAX_TOKENS)
        prep.require(isinstance(encoded, dict) and set(encoded) == {'input_ids', 'attention_mask', 'labels'},
            'invalid regenerated sample')
        ids, labels = encoded['input_ids'], encoded['labels']
        supervised = sum(value != -100 for value in labels)
        prep.require(entry['input_ids_sha256'] == prep.token_hash(ids)
            and entry['labels_sha256'] == prep.token_hash(labels)
            and entry['input_tokens'] == len(ids) and entry['prompt_tokens'] == len(ids) - supervised
            and entry['supervised_tokens'] == supervised, 'regenerated sample hash or token count differs')
        encoded_rows.append(encoded)
    groups = [[encoded_rows[number] for number in group['sample_indices']] for group in schedule['groups']]
    totals = validate_groups(groups, max_length=prep.MAX_TOKENS)
    prep.require(totals == [g['supervised_tokens'] for g in schedule['groups']]
        and [sum(len(row['input_ids']) for row in group) for group in groups]
        == [g['input_tokens'] for g in schedule['groups']], 'encoded group token counts differ')
    return groups


def load_verified_groups(root, tokenizer_path):
    """Return all registered groups and a content-free summary after full checks.

    This may import torch as a dependency of the tokenizer. A GPU caller must
    acquire its ticket before calling it, but must not initialize CUDA here.
    """
    started = time.monotonic()
    root = Path(root).resolve(strict=True)
    prep.require(root == prep.ROOT.resolve(strict=True), 'registered repository root required')
    source = root / 'scripts/prepare_b3_training_schedule.py'
    prep.require(not source.is_symlink() and source.resolve(strict=True) == Path(prep.__file__).resolve(strict=True)
        and prep.digest(source) == MATERIALIZER_SHA256, 'registered materializer changed')
    snapshot = snapshot_schedule(root)
    recorded = json.loads(snapshot['proposal.json'])
    prep.require(recorded['sample_index_sha256'] == SCHEDULE_SHA256['sample-index.json']
        and recorded['update_groups_sha256'] == SCHEDULE_SHA256['update-groups.json']
        and recorded['materializer_sha256'] == MATERIALIZER_SHA256, 'proposal bindings differ')

    # Re-run the frozen materializer, including corpus, partition, serving
    # prefix, model/tokenizer and runtime checks, before consuming its rows.
    index, schedule, proposal = prep.prepare(root, tokenizer_path)
    prep.write_or_verify(root / prep.OUTPUT, index, schedule, proposal, True)
    plan, proof, token, rows, raw_rows, _, _ = prep.bound_corpus(root)
    files = prep.verify_tokenizer_files(tokenizer_path, prep.base.read(root / prep.DOWNLOAD), token)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    groups = assemble_groups(rows, raw_rows, tokenizer, index, schedule)
    summary = prep.summarize(index)
    prep.require(len(groups) == schedule['optimizer_updates'] == 86
        and all(len(group) == prep.GROUP_SIZE for group in groups)
        and summary['examples'] == prep.EXPECTED_ROWS
        and summary['input_tokens_one_epoch'] == prep.EXPECTED_INPUT
        and summary['supervised_tokens_one_epoch'] == prep.EXPECTED_TARGETS,
        'complete registered token workload required')
    prep.require(snapshot_schedule(root) == snapshot and prep.digest(source) == MATERIALIZER_SHA256
        and prep.bound_corpus(root)[:3] == (plan, proof, token)
        and prep.verify_tokenizer_files(tokenizer_path, prep.base.read(root / prep.DOWNLOAD), token) == files,
        'schedule, materializer, corpus or tokenizer changed during regeneration')
    return groups, {**summary, 'optimizer_updates': len(groups), 'microbatches_per_update': prep.GROUP_SIZE,
        'order_sha256': schedule['order_sha256'], 'schedule_files_sha256': dict(SCHEDULE_SHA256),
        'preparation_seconds': time.monotonic() - started,
        'scope': 'CPU token regeneration only; no CUDA, model load, training or GPU authorization.'}
