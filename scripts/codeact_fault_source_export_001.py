"""Terminal-only controlled-fault export over the pinned natural-source reader.

Import reads only fixed human repository code. Real export remains a desktop
read-only operation after all twelve episodes pass their authenticated gate.
No Source, Task, actor file, writer, model or queue is accessed by import.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).with_name('codeact_within_source_export_001.py')
BASE_SHA256 = '2c68870d7791135b25a7651e54cf4a07d492334dbdc4392a3bbdb05f35b5e07f'
ID = 'codeact-fault-source-export-001'
SOURCE_ID = 'qwen38-codeact-fault-source-001'
SOURCE_PLAN = 'artifacts/qwen38-codeact-fault-source-plan-001.json'
SOURCE_PLAN_SHA256 = '76bc11bed6f1fb93ff5d91a69d33d231dc3033718d60c803fbf02acc61446876'
ERROR = 'codeact_fault_source_export_invalid'


def _require(value):
    if not value:
        raise ValueError(ERROR) from None


def _fault_plan(root, plan, registration):
    _require(plan.get('source_condition') == 'controlled_single_tool_refusal'
             and json.dumps(plan.get('fault_policy'), sort_keys=True, allow_nan=False)
             == json.dumps(registration.FAULT_POLICY, sort_keys=True, allow_nan=False))
    # The outer loader already checks these; retain the explicit proof boundary
    # when an externally supplied loader is used for a synthetic CPU fixture.
    registration._additional_validation(root, plan)


def _isolated(*, raw=None):
    _require(not BASE.is_symlink())
    raw = BASE.read_bytes() if raw is None else raw
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == BASE_SHA256)
    source = raw.decode('utf-8')
    changes = (
        ('codeact-within-source-export-001', ID),
        ('qwen38-codeact-within-source-001', SOURCE_ID),
        ('artifacts/qwen38-codeact-within-source-plan-001.json', SOURCE_PLAN),
        ('d97afc544dc53ae35af596776c2c5d2b15287677da32d241e711d9ee6849a41b', SOURCE_PLAN_SHA256),
        ('codeact_source_export_invalid', ERROR),
        ('codeact_within_registration_001', 'codeact_fault_registration_001'),
        ('run_codeact_within_source_001', 'run_codeact_fault_source_001'),
        ('run_codeact_within_source_study_001', 'run_codeact_fault_source_study_001'),
        ('from scripts.codeact_within_sources_001 import qualify_sources',
         'from scripts.codeact_fault_sources_001 import qualify_sources'),
        ('    reg.check_predecessor(root, plan)\n',
         '    _require(reg.check_predecessor(root, plan) is True)\n    _fault_plan(root, plan, reg)\n'),
    )
    for before, after in changes:
        _require(source.count(before) == 1)
        source = source.replace(before, after)
    spec = importlib.util.spec_from_file_location('scripts._codeact_fault_source_export_core', BASE)
    _require(spec is not None and spec.loader is not None)
    core = importlib.util.module_from_spec(spec)
    core.__dict__['_fault_plan'] = _fault_plan
    # SHA-bound human implementation only; model/actor text is never compiled.
    exec(compile(source, str(BASE), 'exec', dont_inherit=True), core.__dict__)
    return core


_CORE = _isolated()
prepare_source_export = _CORE.prepare_source_export
