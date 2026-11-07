"""Deployment-qualified writer revision after 001 stopped before GPU entry."""

import hashlib
import json
from pathlib import Path

from scripts.native_appworld_worker_client import _encode
from scripts.native_source_registration import public_read

ID = 'qwen38-paired-omission-writer-002'
MODULE = 'scripts.run_paired_omission_writer_002'
PREVIOUS_RUNNER = 'scripts/run_paired_omission_writer_001.py'
PREVIOUS_RUNNER_SHA = '889b7ff92361695924b4535171df9278fc39b56ba4b4d54f5df852deb31c13b4'
PREVIOUS_PLAN = 'artifacts/qwen38-paired-omission-writer-001-plan.json'
PREVIOUS_PLAN_SHA = '50343891922f06cd3c7c21dd4e308d8000fe285b42e9cbd083d252edaf761be2'
STARTUP_PROOF = 'artifacts/paired-omission-writer-startup-001.json'
STARTUP_PROOF_SHA = 'fba9161604af7662f22527375f25a5af38596ab0ef770d7cd870ad7709092f99'
SMOKE_PROOF = 'artifacts/paired-omission-writer-deployment-CPU-002.json'
DEPLOY = Path('/home/hts/workflow-learning-lab')


def verify_deployment(root, plan):
    for name, expected in ((PREVIOUS_PLAN, PREVIOUS_PLAN_SHA), (STARTUP_PROOF, STARTUP_PROOF_SHA)):
        if (plan['input_sha256'].get(name) != expected
                or hashlib.sha256(public_read(root, name)).hexdigest() != expected):
            raise ValueError('paired_omission_writer_predecessor_changed')
    raw = public_read(root, SMOKE_PROOF)
    expected = dict(id=ID, status='verified_deployment_CPU_import', deployment=str(DEPLOY),
                    exact_module=MODULE, writer_calls=0, model_loaded=False, GPU_operations=0,
                    source_map_sha256=hashlib.sha256(_encode(plan['source_sha256'])).hexdigest())
    if (hashlib.sha256(raw).hexdigest() != plan['input_sha256'].get(SMOKE_PROOF)
            or json.loads(raw) != expected):
        raise ValueError('paired_omission_writer_deployment_proof_changed')


def _previous_code():
    base = Path(__file__).resolve().parents[1]
    raw = public_read(base, PREVIOUS_RUNNER)
    if hashlib.sha256(raw).hexdigest() != PREVIOUS_RUNNER_SHA:
        raise ValueError('paired_omission_previous_writer_changed')
    source = raw.decode('utf-8')
    for old, new in (
        ("MODULE = 'scripts.run_paired_omission_writer_001'",
         "MODULE = 'scripts.run_paired_omission_writer_002'"),
        ("ID = 'qwen38-paired-omission-writer-001'", "ID = 'qwen38-paired-omission-writer-002'"),
        ("'/paired-omission-writer-001'", "'/paired-omission-writer-002'"),
        ("'paired-omission-writer-operations-001.lock'", "'paired-omission-writer-operations-002.lock'"),
        ("'wll-native-source-paired-omission-writer-'", "'wll-native-source-paired-omission-writer-002-'"),
        ("'scripts/paired_omission_writer_evidence_001.py',",
         "'scripts/paired_omission_writer_evidence_001.py', 'scripts/run_paired_omission_writer_002.py',"),
        ("    inputs = source_bindings()", "    verify_deployment(root, plan)\n    inputs = source_bindings()"),
    ):
        if source.count(old) != 1:
            raise ValueError('paired_omission_writer_revision_binding_changed')
        source = source.replace(old, new)
    return compile(source, str(base / PREVIOUS_RUNNER), 'exec', dont_inherit=True)


# The previous trusted controller is verified above, and no model text is compiled.
exec(_previous_code(), globals())
