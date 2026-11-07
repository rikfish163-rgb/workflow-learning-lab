"""Pinned single-call writer for two exposed artificial-omission recoveries.

Only trusted, hash-checked repository source is compiled in an isolated module.
Importing this adapter performs no model, remote, world, or registration work.
"""

import hashlib
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).with_name('run_completion_lesson.py')
SOURCE_SHA256 = '388ca785e5e97d468bb1d60ccceae45106a2cb8639e12fe1dfbeed7f161aa05d'
SOURCE_KIND = 'two_induced_omissions_with_actor_semantics'
SOURCE_SCENARIOS = ('b7a9ee9', '2a163ab')
# Order: first CPU mutant, first model mutant, second CPU mutant, second model mutant.
SOURCE_TRACE_SHA256 = (
    '21e7611395ff0a109825352010ec0ee7e8886282520c74b9b12e4ccb177a172d',
    '826352f67a4d02cc4d27052256a51c8cde0d36ff9ec1588b6388eb9c5a9f49bf',
    '3fad2bd577075e551b0f803207ffa0795a2a1ada9067c93ec234262d7d388e82',
    'c771c1c145a70ba8ba7fe586df52627cb4986310e47c1093f68d4241c821ffe3',
)
NEW_REQUIRED_SOURCES = (
    'scripts/run_recovery_evidence_lesson.py', 'scripts/queue_recovery_evidence_lesson.py',
    'scripts/recovery_evidence_lesson.py', 'tests/test_recovery_evidence_lesson.py',
    'tests/test_recovery_evidence_lesson_runner.py', 'tests/test_recovery_evidence_lesson_queue.py',
    'artifacts/register_recovery_evidence_lesson_001.py',
    'tests/test_recovery_evidence_lesson_registration.py',
    'scripts/workflow_loop_omission.py', 'scripts/workflow_write_loop_omission.py',
    'scripts/write_partial_lesson.py', 'scripts/completion_boundary.py',
    'wll/experience.py', 'wll/runner.py', 'wll/model.py', 'wll/prompts.py',
)
REPLACEMENTS = (
    ('from scripts.completion_lesson import SYSTEM, parse_candidate',
     'from scripts.recovery_evidence_lesson import SYSTEM, parse_candidate', 1),
    ("ID = 'qwen38-completion-lesson-002'", "ID = 'qwen38-recovery-evidence-lesson-001'", 1),
    ("PLAN = 'artifacts/qwen38-completion-lesson-plan-002.json'",
     "PLAN = 'artifacts/qwen38-recovery-evidence-lesson-plan-001.json'", 1),
    ('from scripts.queue_completion_lesson import load_registration',
     'from scripts.queue_recovery_evidence_lesson import load_registration', 1),
    ('from scripts.queue_completion_lesson import validate_worker_entry',
     'from scripts.queue_recovery_evidence_lesson import validate_worker_entry', 1),
    ('REQUIRED_SOURCES = {',
     'REQUIRED_SOURCES = {\n    ' + ', '.join(repr(name) for name in NEW_REQUIRED_SOURCES) + ',', 1),
    ("'source_scenario': 'cf6abd2'",
     "'source_scenarios': list(SOURCE_SCENARIOS), 'source_kind': SOURCE_KIND, "
     "'automatic_utility': False", 1),
    ("and len(plan['source_trace_sha256']) == 2\n"
     "            and len(set(plan['source_trace_sha256'])) == 2\n"
     "            and all(_sha(value) for value in plan['source_trace_sha256'])",
     "and len(plan['source_trace_sha256']) == 4\n"
     "            and all(_sha(value) for value in plan['source_trace_sha256'])\n"
     "            and len(set(plan['source_trace_sha256'])) == 4\n"
     "            and plan['source_trace_sha256'] == list(SOURCE_TRACE_SHA256)", 1),
    ("'private_input_sha256': plan['private_input_sha256'], 'source_scenario': plan['source_scenario'],",
     "'private_input_sha256': plan['private_input_sha256'],\n"
     "        'source_scenarios': list(plan['source_scenarios']), 'source_kind': plan['source_kind'],\n"
     "        'automatic_utility': False,", 1),
    ("source_scenario=plan['source_scenario'], source_hashes=plan['source_trace_sha256']",
     "source_scenarios=plan['source_scenarios'], source_hashes=plan['source_trace_sha256']", 1),
)


def _isolated(*, raw=None):
    if raw is None:
        if not SOURCE.is_file() or SOURCE.is_symlink():
            raise ValueError('frozen_writer_source_changed')
        raw = SOURCE.read_bytes()
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('frozen_writer_source_changed')
    source = raw.decode('utf-8')
    for old, new, count in REPLACEMENTS:
        if source.count(old) != count:
            raise ValueError('frozen_writer_replacement_changed')
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._isolated_recovery_evidence_runner', SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError('frozen_writer_source_unavailable')
    module = importlib.util.module_from_spec(spec)
    module.SOURCE_KIND = SOURCE_KIND
    module.SOURCE_SCENARIOS = SOURCE_SCENARIOS
    module.SOURCE_TRACE_SHA256 = SOURCE_TRACE_SHA256
    # Source is trusted repository code verified above, never an actor response.
    exec(compile(source, str(SOURCE), 'exec', dont_inherit=True), module.__dict__)
    return module


_CORE = _isolated()


def __getattr__(name):
    return getattr(_CORE, name)


if __name__ == '__main__':
    raise SystemExit(_CORE.main())
