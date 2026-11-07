"""One terminal-only desktop fault source export, sealed on the data4 mounts.

No registration, tokenizer, model, queue or GPU operation occurs here. The
sealed source payload is private. Public output contains hashes and counts.
"""
import argparse
import os
from pathlib import Path

from scripts import codeact_within_writer_registration_001 as reg
from scripts.codeact_fault_source_export_001 import prepare_source_export
from scripts.native_appworld_worker_client import _encode, _exclusive
from scripts.remote_runtime_storage import REMOTE_TMP_DIR, verify_storage

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path('/home/hetaisheng/.local/share/workflow-learning-lab')


def _save_new(path, value):
    path = Path(path)
    descriptor = os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with _exclusive(descriptor,path.name) as file:
            raw = _encode(value) + b'\n'
            remaining = memoryview(raw)
            while remaining:
                size = file.write(remaining)
                reg._require(type(size) is int and size > 0)
                remaining = remaining[size:]
            file.flush()
            os.fsync(file.fileno())
        os.fsync(descriptor)
        return reg._hash(raw)
    finally:
        os.close(descriptor)


def prepare(report_sha256):
    storage = verify_storage(ROOT,REMOTE_TMP_DIR)
    exported = prepare_source_export(report_sha256)
    # Authentication and all twelve closed-source checks precede namespace writes.
    directory = PRIVATE / 'codeact-within-writer-preparation-001'
    directory.mkdir(mode=0o700)
    private = exported['private']
    manifest = private['preparation_manifest']
    registered = dict(preparation_manifest_sha256=reg.value_hash(manifest),
                      origin_ids=manifest['origin_ids'], selection=manifest['selection'])
    digest = _save_new(directory/'source-private.json',private)
    (PRIVATE/'codeact-within-writer-001').mkdir(mode=0o700)
    proof = dict(id='codeact-within-writer-inputs-CPU-001',
                 status='authenticated_sealed_writer_inputs_not_registered',
                 source_export=exported['public'], private_input_sha256=digest,
                 registered_inputs=registered, storage=storage,
                 model_calls=0, GPU_calls=0, writer_calls=0,
                 registered=False, utility_evaluated=False, training_authorized=False)
    _save_new(ROOT/reg.PREPARATION_PROOF,proof)
    return proof


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--source-report-sha256',required=True)
    args=parser.parse_args(argv)
    proof=prepare(args.source_report_sha256)
    # Deliberately output only a fixed public summary.
    print({'status':proof['status'],'private_input_sha256':proof['private_input_sha256'],
           'selected_source_count':proof['registered_inputs']['selection']['selected_source_count']})


if __name__ == '__main__':
    main()
