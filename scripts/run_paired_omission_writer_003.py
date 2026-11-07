"""Writer revision with the complete admitted-backend verifier binding."""

import hashlib
from pathlib import Path

from scripts.native_source_registration import public_read

MODEL_ARM = 'B3'
_REVISION_SOURCE = 'scripts/run_paired_omission_writer_002.py'
_REVISION_SHA = 'ac238a2017024d00bcfe3a3fd7a4662d07252edc46cba9d5171507b4b712eb83'


def _revision_code():
    root = Path(__file__).resolve().parents[1]
    raw = public_read(root, _REVISION_SOURCE)
    if hashlib.sha256(raw).hexdigest() != _REVISION_SHA:
        raise ValueError('paired_omission_revision_002_changed')
    source = raw.decode('utf-8')
    for old, new, count in (
        ('qwen38-paired-omission-writer-002', 'qwen38-paired-omission-writer-003', 2),
        ('run_paired_omission_writer_002', 'run_paired_omission_writer_003', 3),
        ('/paired-omission-writer-002', '/paired-omission-writer-003', 1),
        ('writer-operations-002.lock', 'writer-operations-003.lock', 1),
        ('paired-omission-writer-002-', 'paired-omission-writer-003-', 1),
        ('deployment-CPU-002.json', 'deployment-CPU-003.json', 1),
        ("PREVIOUS_PLAN = 'artifacts/qwen38-paired-omission-writer-001-plan.json'",
         "PREVIOUS_PLAN = 'artifacts/qwen38-paired-omission-writer-002-plan.json'", 1),
        ('50343891922f06cd3c7c21dd4e308d8000fe285b42e9cbd083d252edaf761be2',
         'a11f371cc371fd93002cc67ee4bb4a1ca9a0c8a71e68d6e64cd0aed76bd1bf6a', 1),
        ('writer-startup-001.json', 'writer-startup-002.json', 1),
        ('fba9161604af7662f22527375f25a5af38596ab0ef770d7cd870ad7709092f99',
         'dcdb9722969ac4a53c8ab4a336c3e9e18d7a1f80ce2e2e74ae37d0172043f13a', 1),
        ('model_loaded=False, GPU_operations=0,',
         'model_loaded=False, GPU_operations=0, ready_path_checked=True,', 1),
    ):
        if source.count(old) != count:
            raise ValueError('paired_omission_revision_003_transform_changed')
        source = source.replace(old, new)
    return compile(source, str(root / _REVISION_SOURCE), 'exec', dont_inherit=True)


# The only compiled bytes are the verified repository controller and fixed edits.
exec(_revision_code(), globals())
