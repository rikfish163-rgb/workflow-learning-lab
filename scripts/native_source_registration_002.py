"""Independent V2 registration over authenticated, immutable V1 validator bytes.

The scientific method and initial-context proof are unchanged. Only execution
coordination is revised; predecessor metadata is provenance, never trial data.
Import does not register, write, load a task, reserve a queue or run a model.
"""

import hashlib
import importlib.util
import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_source_registration.py"
BASE_SHA256 = "4a3adc99ac8a07be9fd31c5b4b638f431344eba841162940f0a1e58910e07a06"
EPISODE_SHA256 = "6b034e2b0c53050832c43384b1e3c4bc5853f5c8e940bbe52c348a0d3ce36e1b"
ID = "qwen38-native-source-pilot-002"
PLAN = "artifacts/qwen38-native-source-pilot-plan-002.json"
PRIVATE_RUN_ROOT = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state/native-source-pilot-002"
LAUNCH_PROOF = "artifacts/native-source-launch-root-CPU-002.json"
PORT = 18991
SERVICE_PLAN = "artifacts/qwen38-b3-service-plan-001.json"
SERVICE_SHA = "d7f56521d101394c8559ae0604680dc75d1bd4e346a96032c1c1c94ec676bad5"
PYTHON = "/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python"
WAIT_SECONDS = 86400
BATCH_GPU_SECONDS = 7200
PREDECESSOR_PLAN = "artifacts/qwen38-native-source-pilot-plan-001.json"
PREDECESSOR_PLAN_SHA256 = "f8291cf6f62fbef3e302e58ab9a777b68c20ab26a6500e963207688f396bbbf4"
PREDECESSOR_TERMINAL = "artifacts/native-source-pilot-terminal-root-001.json"
PREDECESSOR_TERMINAL_SHA256 = "4bd7d6502ffee1838987a72e100024eeebcba6f3887c8a024ea5572eb2e441e6"
V2_SOURCES = frozenset(
    f"{directory}/{prefix}{name}_002.py"
    for name in ("native_source_registration", "native_source_gpu_queue", "native_source_episode", "run_native_source_pilot")
    for directory, prefix in (("scripts", ""), ("tests", "test_")))
_ERROR = "native_source_registration_002_source_invalid"
_CORE = None


def authenticated_source(name, digest):
    """Read one fixed public script without symlink traversal, then authenticate it."""
    if name not in {"native_source_registration.py", "native_source_episode.py"}:
        raise ValueError(_ERROR)
    from scripts.native_appworld_worker_client import _open_directory

    directory = _open_directory(ROOT / "scripts")
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1_000_000:
                raise ValueError(_ERROR)
            raw = file.read(1_000_001)
    finally:
        os.close(directory)
    if len(raw) > 1_000_000 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(_ERROR)
    return raw


def _isolated(*, raw=None):
    raw = authenticated_source(BASE_SOURCE.name, BASE_SHA256) if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError(_ERROR)
    spec = importlib.util.spec_from_file_location("scripts._native_source_registration_002_core", BASE_SOURCE)
    core = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), "exec", dont_inherit=True), core.__dict__)
    core.ID, core.PLAN, core.PRIVATE_RUN_ROOT, core.LAUNCH_PROOF = ID, PLAN, PRIVATE_RUN_ROOT, LAUNCH_PROOF
    core.NEW_SOURCES = core.NEW_SOURCES | V2_SOURCES
    core.REQUIRED_SOURCES = core.REQUIRED_SOURCES | V2_SOURCES
    core.FIXED_INPUT_SHA256 = {**core.FIXED_INPUT_SHA256, PREDECESSOR_PLAN: PREDECESSOR_PLAN_SHA256,
                             PREDECESSOR_TERMINAL: PREDECESSOR_TERMINAL_SHA256}
    core.FROZEN_SOURCE_SHA256 = {**core.FROZEN_SOURCE_SHA256,
        "scripts/native_source_registration.py": BASE_SHA256, "scripts/native_source_episode.py": EPISODE_SHA256}
    scientific_design = core.expected_design

    def expected_design(root):
        design = scientific_design(root)
        design["execution_revision"] = {
            "revision": 2, "cross_process_queue_serialization": True, "operation_guard_wait_seconds": 10,
            "completed_watcher_cycles_after_cancellation": 2, "release_seconds": 90,
            "shutdown_reserve_seconds": 150, "private_initiating_exception_capture": True,
            "independent_study": True, "automatic_resume": False, "outcomes_imported": False,
            "predecessor_id": "qwen38-native-source-pilot-001", "predecessor_policy_episodes": 0,
            "predecessor_plan": PREDECESSOR_PLAN, "predecessor_plan_sha256": PREDECESSOR_PLAN_SHA256,
            "predecessor_terminal": PREDECESSOR_TERMINAL,
            "predecessor_terminal_sha256": PREDECESSOR_TERMINAL_SHA256,
            "historical_experiment_name_prefix": "native-source-001"}
        return design

    core.expected_design = expected_design
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def load_registration(root, sha, *, private=False):
    return _get_core().load_registration(root, sha, private=private)


def validate_registration(root, plan, *, private=False):
    return _get_core().validate_registration(root, plan, private=private)


def register(root, **kwargs):
    return _get_core().register(root, **kwargs)


def expected_design(root):
    return _get_core().expected_design(root)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_get_core(), name)
