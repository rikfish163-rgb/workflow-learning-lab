"""Pinned single-call queue with a lazy, public-only registration gate."""

import hashlib
import importlib.util
import json
import os
import re
import stat
from pathlib import Path

SOURCE = Path(__file__).with_name('queue_completion_lesson.py')
SOURCE_SHA256 = 'e6e7ac7663881f8c2b62cf1892e140cf7818a3f436bb8eaa58cc55212e5b721d'
SOURCE_KIND = 'two_induced_omissions_with_actor_semantics'
SOURCE_SCENARIOS = ('b7a9ee9', '2a163ab')
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
REQUIRED_SOURCES = frozenset(NEW_REQUIRED_SOURCES) | {
    'config/gpu-policy.json', 'scripts/activation_offload.py', 'scripts/audit_b3_dev_batch.py',
    'scripts/completion_boundary.py', 'scripts/completion_lesson.py', 'scripts/completion_trace.py',
    'scripts/control_plane_diagnostics.py', 'scripts/documentation_feedback.py',
    'scripts/download_qwen38.py', 'scripts/generated_presence.py', 'scripts/gpu_idle_watch_b3.py',
    'scripts/load_hybrid_adapter.py', 'scripts/nf4_loader.py', 'scripts/origin_recovery_cohort.py',
    'scripts/origin_recovery_gpu_queue.py', 'scripts/owned_appworld_v3.py',
    'scripts/prepare_observed_lesson_inputs.py', 'scripts/prepare_recovery_inputs.py',
    'scripts/probe_allocator.py', 'scripts/probe_chunked_prefill.py',
    'scripts/probe_observation_visibility.py', 'scripts/probe_qwen38.py',
    'scripts/probe_qwen38_training.py', 'scripts/queue_completion_lesson.py',
    'scripts/qwen38_training.py', 'scripts/report_b3_dev_pairs.py',
    'scripts/report_observation_comparison.py', 'scripts/report_timed_tasks.py',
    'scripts/resumable_ranges.py', 'scripts/run_completion_lesson.py',
    'scripts/run_foundation_pilot.py', 'scripts/run_observed_lesson.py',
    'scripts/run_queued_decoder_completion.py', 'scripts/run_qwen38_b3_dev.py',
    'scripts/run_qwen38_capability.py', 'scripts/run_recovery_trial.py',
    'scripts/run_timed_tasks.py', 'scripts/run_timed_tasks_v2.py', 'scripts/run_timed_tasks_v3.py',
    'scripts/save_hybrid_adapter.py', 'scripts/serve_qwen38.py', 'scripts/serve_qwen38_b3.py',
    'scripts/serve_qwen38_idle180.py', 'scripts/serve_transformers_presence.py',
    'scripts/target_logits.py', 'scripts/text_chunked_prefill.py', 'scripts/training_numerics.py',
    'scripts/workflow_loop_omission.py', 'scripts/workflow_write_loop_omission.py',
    'scripts/write_partial_lesson.py', 'wll/appworld.py', 'wll/appworld_v2.py', 'wll/cli.py',
    'wll/experience.py', 'wll/gpu_idle.py', 'wll/gpu_policy.py', 'wll/model.py',
    'wll/prompts.py', 'wll/protocol.py', 'wll/recovery.py', 'wll/runner.py', 'wll/storage.py',
    'wll/training.py', 'wll/utility.py',
}
REPLACEMENTS = (
    ("ID = 'qwen38-completion-lesson-002'", "ID = 'qwen38-recovery-evidence-lesson-001'", 1),
    ("PLAN = 'artifacts/qwen38-completion-lesson-plan-002.json'",
     "PLAN = 'artifacts/qwen38-recovery-evidence-lesson-plan-001.json'", 1),
    ('/home/hts/.local/share/workflow-learning-lab/completion-lesson-002/output',
     '/home/hts/.local/share/workflow-learning-lab/recovery-evidence-lesson-001/output', 1),
    ('scripts.run_completion_lesson', 'scripts.run_recovery_evidence_lesson', 2),
    ('completion-lesson-enqueued-002.json', 'recovery-evidence-lesson-enqueued-001.json', 1),
    ('completion-lesson-cancelled-002.json', 'recovery-evidence-lesson-cancelled-001.json', 1),
    ('from scripts import queue_completion_lesson as q',
     'from scripts import queue_recovery_evidence_lesson as q', 1),
    ('from scripts.queue_completion_lesson import PYTHON,load_registration,validate_preflight',
     'from scripts.queue_recovery_evidence_lesson import PYTHON,load_registration,validate_preflight', 1),
    ("and plan.get('private_output') == str(PRIVATE_OUTPUT), 'writer registration differs')",
     "and plan.get('private_output') == str(PRIVATE_OUTPUT)\n"
     "            and plan.get('private_input') == str(PRIVATE_OUTPUT.parent / 'request.jsonl')\n"
     "            and type(plan.get('source_scenarios')) is list\n"
     "            and plan['source_scenarios'] == list(SOURCE_SCENARIOS)\n"
     "            and type(plan.get('source_trace_sha256')) is list\n"
     "            and plan['source_trace_sha256'] == list(SOURCE_TRACE_SHA256)\n"
     "            and plan.get('source_kind') == SOURCE_KIND\n"
     "            and plan.get('automatic_utility') is False\n"
     "            and plan.get('training_authorized') is False, 'writer registration differs')", 1),
    ("    require(required <= plan['source_sha256'].keys(), 'writer queue dependencies not bound')",
     "    required.update(REQUIRED_SOURCES)\n"
     "    require(required <= plan['source_sha256'].keys(), 'writer queue dependencies not bound')", 1),
    ("    require(plan['gpu_policy'] == read(root / 'config/gpu-policy.json')\n"
     "            and plan['gpu_policy_sha256'] == digest(root / 'config/gpu-policy.json')",
     "    policy_raw = public_read(root, 'config/gpu-policy.json')\n"
     "    require(plan['gpu_policy'] == json.loads(policy_raw)\n"
     "            and plan['gpu_policy_sha256'] == hashlib.sha256(policy_raw).hexdigest()", 1),
)


def _public_directory(path):
    """Open each component without following symlinks and return the directory fd."""
    path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open('/', flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _public_file_at(directory, name):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOCTTY
    fd = os.open(name, flags, dir_fd=directory)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError('regular_unlinked_public_file_required')
        chunks = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
        raw = b''.join(chunks)
        after = os.fstat(fd)
        def identity(value):
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if len(raw) != before.st_size or identity(before) != identity(after) or after.st_nlink != 1:
            raise ValueError('public_file_changed')
        return raw
    finally:
        os.close(fd)


def public_read(root, relative_name):
    """Read one project-relative regular file without following symlinks."""
    name = os.fspath(relative_name)
    if (type(name) is not str or not name or '\x00' in name or os.path.isabs(name)
            or '..' in Path(name).parts or os.path.normpath(name) != name):
        raise ValueError('invalid_public_path')
    path = Path(os.path.abspath(root)) / name
    directory = _public_directory(path.parent)
    try:
        return _public_file_at(directory, path.name)
    finally:
        os.close(directory)


def public_wll_fingerprint(root):
    """Use the frozen fingerprint algorithm on a descriptor-read WLL snapshot."""
    directory = _public_directory(Path(root) / 'wll')
    try:
        before = os.fstat(directory)
        names = [name for name in os.listdir(directory) if name.endswith('.py')]
        files = {name: _public_file_at(directory, name).decode('utf-8') for name in names}
        after = os.fstat(directory)
        def identity(value):
            return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError('public_wll_directory_changed')
        return _CORE.fingerprint(files)
    finally:
        os.close(directory)


def verify_public_frozen_plan(root, plan_path, plan_sha, *, expected_plan=None):
    """Apply the frozen plan checks using only descriptor-safe public reads."""
    if not isinstance(plan_sha, str) or not re.fullmatch('[a-f0-9]{64}', plan_sha):
        raise ValueError('original plan SHA256 required')
    root = Path(os.path.abspath(root))
    path = Path(os.path.abspath(plan_path))
    try:
        name = str(path.relative_to(root))
    except ValueError:
        raise ValueError('non-project plan') from None
    raw = public_read(root, name)
    if hashlib.sha256(raw).hexdigest() != plan_sha:
        raise ValueError('frozen plan changed')
    plan = json.loads(raw)
    if expected_plan is not None and plan != expected_plan:
        raise ValueError('in-memory plan changed')
    for entries in (plan['source_sha256'], plan['input_sha256']):
        for source, expected in entries.items():
            if hashlib.sha256(public_read(root, source)).hexdigest() != expected:
                raise ValueError('frozen public file changed')
    if plan['wll_code_fingerprint'] != public_wll_fingerprint(root):
        raise ValueError('core code changed')
    return plan


def _isolated(*, raw=None):
    if raw is None:
        raw = public_read(SOURCE.parent.parent, str(Path('scripts') / SOURCE.name))
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('frozen_writer_queue_source_changed')
    source = raw.decode('utf-8')
    for old, new, count in REPLACEMENTS:
        if source.count(old) != count:
            raise ValueError('frozen_writer_queue_replacement_changed')
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location('scripts._isolated_recovery_evidence_queue', SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError('frozen_writer_queue_source_unavailable')
    module = importlib.util.module_from_spec(spec)
    module.SOURCE_KIND = SOURCE_KIND
    module.SOURCE_SCENARIOS = SOURCE_SCENARIOS
    module.SOURCE_TRACE_SHA256 = SOURCE_TRACE_SHA256
    module.REQUIRED_SOURCES = REQUIRED_SOURCES
    module.public_read = public_read
    module.hashlib = hashlib
    # Source is trusted repository code verified above, never an actor response.
    exec(compile(source, str(SOURCE), 'exec', dont_inherit=True), module.__dict__)
    return module


_CORE = _isolated()
_BASE_LOAD_REGISTRATION = _CORE.load_registration
_CORE.verify_frozen_plan = verify_public_frozen_plan


def _public_load_registration(root, sha, *, native=False):
    _CORE.verify_frozen_plan(root, Path(root) / _CORE.PLAN, sha)
    plan = _BASE_LOAD_REGISTRATION(root, sha, native=native)
    # The registrar imports this queue for constants. Resolve its public gate
    # only at runtime, after both modules have finished importing.
    from artifacts.register_recovery_evidence_lesson_001 import verify_public_registration

    return verify_public_registration(Path(root), plan)


_CORE.load_registration = _public_load_registration


def __getattr__(name):
    return getattr(_CORE, name)


if __name__ == '__main__':
    raise SystemExit(_CORE.main())
