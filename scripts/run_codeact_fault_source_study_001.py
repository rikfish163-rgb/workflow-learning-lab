"""Independent four-batch controlled-fault source parent, with no retries.

The original source parent is compiled in an isolated namespace only after its
SHA is verified. Fault metadata is retained as provenance; qualification still
uses the separately pinned twelve-episode/four-scene/twenty-target gate.
Import never registers or starts a task, model, queue or GPU.
"""

import hashlib
import importlib.util
from pathlib import Path

from scripts.codeact_fault_sources_001 import validate_fault

ROOT = Path(__file__).resolve().parents[1]
ID = 'qwen38-codeact-fault-source-001'
OUTPUT = ROOT / 'artifacts' / ID
PRIVATE = Path('/home/hetaisheng/.local/share/workflow-learning-lab/codeact-fault-source-001')
SOURCE = Path(__file__).with_name('run_codeact_within_source_study_001.py')
SOURCE_SHA256 = '129128bd787e4785934e977d98f3faf8256cc52f7d13840fc579b4332d6dd25c'
ERROR = 'codeact_fault_source_study_invalid'


def _require(condition):
    if not condition:
        raise ValueError(ERROR) from None


def _load(*, raw=None):
    _require(not SOURCE.is_symlink())
    raw = SOURCE.read_bytes() if raw is None else raw
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == SOURCE_SHA256)
    source = raw.decode('utf-8')
    changes = (
        ('qwen38-codeact-within-source-001', ID, 1),
        ('workflow-learning-lab/codeact-within-source-001',
         'workflow-learning-lab/codeact-fault-source-001', 1),
        ('codeact_source_study_invalid', ERROR, 1),
        ('codeact_within_registration_001', 'codeact_fault_registration_001', 1),
        ('run_codeact_within_source_001', 'run_codeact_fault_source_001', 1),
        ('codeact_within_sources_001', 'codeact_fault_sources_001', 1),
        ("    _require(c.verify_batch_report(stage, plan, index, sha) is True)\n",
         "    _, verified_report_sha = c._read(stage / 'run/report.json', maximum=128_000_000)\n"
         "    _require(c.verify_batch_report(stage, plan, index, sha) is True)\n", 1),
        ("    report, report_sha = read(stage / 'run/report.json')\n",
         "    report, report_sha = read(stage / 'run/report.json', expected=verified_report_sha)\n", 1),
        ("row['phase'] == 'source'", "row['phase'] == 'controlled_fault_source'", 1),
        ("phase='source'", "phase='controlled_fault_source'", 1),
        ("item['arm'] == 'first_attempt' and _same(item['intent'], intent['intent'])",
         "item['arm'] == intent.get('arm') == 'first_attempt' "
         "and intent.get('phase') == 'controlled_fault_source' and _same(item['intent'], intent['intent'])", 1),
        ("row.get('task_id') in plan['batches'][index] for n, row in enumerate(event_rows)",
         "row.get('task_id') in plan['batches'][index] and row.get('arm') == 'first_attempt' "
         "and row.get('phase') == 'controlled_fault_source' for n, row in enumerate(event_rows)", 1),
        ("        owner = row['owner']\n", "        fault = copy.deepcopy(row['fault_provenance'])\n"
         "        validate_fault(fault, outcome['counts']['api_request_dispatches'])\n"
         "        owner = row['owner']\n", 1),
        ("envelopes.append(dict(source_index=source_index, task_id=row['task_id'], **episode,",
         "envelopes.append(dict(source_index=source_index, task_id=row['task_id'], "
         "fault_provenance=copy.deepcopy(fault), **episode,", 1),
        ("phase_seconds=copy.deepcopy(owner['phase_seconds'])))",
         "phase_seconds=copy.deepcopy(owner['phase_seconds']), fault_provenance=copy.deepcopy(fault)))", 1),
        ('actor_wall_seconds=None, world_wall_seconds=None, phase_seconds=None)',
         'actor_wall_seconds=None, world_wall_seconds=None, phase_seconds=None, fault_provenance=None)', 1),
        ('    reg.check_predecessor(ROOT, plan)\n', '    _require(reg.check_predecessor(ROOT, plan) is True)\n', 1),
        ('            results, envelopes = [], []\n', '            results, envelopes, first_batches = [], [], []\n', 1),
        ("                    report['batches'][index] = batch\n",
         "                    report['batches'][index] = batch\n                    first_batches.append(copy.deepcopy(batch))\n", 1),
        ('                                       official_success=None, model_reported_complete=None)\n',
         '                                       official_success=None, model_reported_complete=None, fault_provenance=None)\n', 1),
        ("                        envelopes.extend(group)\n                        report['sources'][index * 3:index * 3 + 3], report['batches'][index] = public, batch",
         "                        _require(_same(batch, first_batches[index]))\n                        envelopes.extend(group)\n"
         "                        report['sources'][index * 3:index * 3 + 3], report['batches'][index] = public, batch", 1),
        ('                    qualified = deps.qualify_sources(envelopes)\n',
         "                    _require(_costs(report)['all_costs_known'] is True)\n"
         '                    qualified = deps.qualify_sources(envelopes)\n', 1),
    )
    for before, after, count in changes:
        _require(source.count(before) == count)
        source = source.replace(before, after)
    spec = importlib.util.spec_from_file_location('scripts._codeact_fault_source_study_core', SOURCE)
    _require(spec is not None and spec.loader is not None)
    core = importlib.util.module_from_spec(spec)
    core.__dict__['validate_fault'] = validate_fault
    # Only SHA-bound human-authored repository code is compiled, never actor code.
    exec(compile(source, str(SOURCE), 'exec', dont_inherit=True), core.__dict__)
    return core


_CORE = _load()
run_study, main = _CORE.run_study, _CORE.main


def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return getattr(_CORE, name)


if __name__ == '__main__':
    raise SystemExit(main())
