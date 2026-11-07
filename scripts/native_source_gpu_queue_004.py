"""V4 queue namespace over authenticated, unchanged V3 release-fence code.

Import and job construction do not touch the real queue, watcher or GPU. The
existing V2 guard, idle ticket and release fence run in isolated globals.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_source_gpu_queue_003.py"
BASE_SHA256 = "fb2cb553139e48e71dfb02c42feca0592ec3b9fdb5e593b87f6c42d1f4111e80"
ID = "qwen38-native-source-pilot-004"
PLAN = "artifacts/qwen38-native-source-pilot-plan-004.json"
PRIVATE_RUN_ROOT = "/home/data4/hts/workflow-learning-lab/runtime-remote-001/private-state/native-source-pilot-004"
PORT = 18991
REMOTE = Path("/home/hts/workflow-learning-lab")
GUARD_NAME = "native-source-operations-004.lock"
_CORE = None


def _isolated(*, raw=None):
    if raw is None:
        from scripts.native_source_registration import public_read

        raw = public_read(ROOT, "scripts/native_source_gpu_queue_003.py")
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError("native_source_gpu_queue_004_source_invalid")
    source = raw.decode("utf-8")
    for old, new in (
        ("'scripts.native_source_gpu_queue_003'", "'scripts.native_source_gpu_queue_004'"),
        ("'native_source_registration_003'", "'native_source_registration_004'"),
        ("'_native_source_gpu_queue_003_base'", "'_native_source_gpu_queue_004_base'"),
        ("'scripts._native_source_gpu_queue_003'", "'scripts._native_source_gpu_queue_004'"),
    ):
        if source.count(old) != 1:
            raise ValueError("native_source_gpu_queue_004_binding_invalid")
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location("scripts._native_source_gpu_queue_004_parent", BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError("native_source_gpu_queue_004_source_invalid")
    parent = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASE_SOURCE), "exec", dont_inherit=True), parent.__dict__)
    parent.ID, parent.PLAN, parent.GUARD_NAME = ID, PLAN, GUARD_NAME
    core = parent._isolated()
    load = core.load_registration

    def bound_registration(root, sha):
        plan = load(root, sha)
        if plan["source_sha256"].get("scripts/native_source_gpu_queue_003.py") != BASE_SHA256:
            raise ValueError("V3 queue source binding differs")
        return plan

    core.load_registration = bound_registration
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def make_job(index, owner, plan_sha):
    return _get_core().make_job(index, owner, plan_sha)


def load_registration(root, sha):
    return _get_core().load_registration(root, sha)


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    return _get_core().queue_operation(operation, job, root=root, known_pids=known_pids)


def record_gpu_entry(root, job, gpu):
    return _get_core().record_gpu_entry(root, job, gpu)


def fenced_release(runtime, job, known_pids):
    return _get_core().fenced_release(runtime, job, known_pids)


def serve(job, *, root=ROOT):
    return _get_core().serve(job, root=root)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_get_core(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.parse_args(argv)
    return _get_core().main(argv)


if __name__ == "__main__":
    main()
