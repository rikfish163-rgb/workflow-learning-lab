"""Fourteen serial utility leases over the authenticated source004 GPU fence.

This namespace reuses the project lock, per-device idle ticket and 90-second
release fence. Public registration is metadata only; private qualification is
the controller's mandatory prerequisite. Import makes no queue or GPU request.
"""

import argparse
import hashlib
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_source_gpu_queue_004.py"
BASE_SHA256 = "2e02ffd39f188608066990997b16515c9bece698a9668355d19107c5a8b8f995"
ID = "qwen38-native-experience-utility-001"
PLAN = "artifacts/qwen38-native-experience-utility-plan-001.json"
GUARD_NAME = "native-experience-utility-operations-001.lock"
REMOTE = Path("/home/hts/workflow-learning-lab")
PORT = 18991
PYTHON = "/home/data4/hts/workflow-learning-lab/venvs/qwen35-nf4-001/bin/python"
N_BATCHES = 14
_CORE = None


def _isolated(*, raw=None):
    from scripts.native_source_registration import public_read

    raw = public_read(ROOT, "scripts/native_source_gpu_queue_004.py") if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError("native_experience_utility_queue_invalid")
    source = raw.decode("utf-8")
    for old, new in (("scripts.native_source_gpu_queue_004", "scripts.native_experience_utility_gpu_queue_001"),
                     ("native_source_registration_004", "native_experience_utility_registration_001")):
        if source.count(old) != 1:
            raise ValueError("native_experience_utility_queue_invalid")
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location("scripts._native_experience_utility_queue_base", BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError("native_experience_utility_queue_invalid")
    parent = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASE_SOURCE), "exec", dont_inherit=True), parent.__dict__)
    parent.ID, parent.PLAN, parent.GUARD_NAME = ID, PLAN, GUARD_NAME
    core = parent._isolated()
    previous_load = core.load_registration

    def utility_job(index, owner, sha):
        if not (type(index) is int and 0 <= index < N_BATCHES
                and type(owner) is str and re.fullmatch(r"[a-f0-9]{32}", owner)
                and type(sha) is str and re.fullmatch(r"[a-f0-9]{64}", sha)):
            raise ValueError("native_experience_utility_queue_invalid")
        return {"id": f"{ID}-comparison-{index}", "kind": "comparison", "batch_index": index,
                "owner": owner, "plan_sha256": sha, "max_runtime_seconds": 7200,
                "argv": [PYTHON, "-B", "-m", "scripts.native_experience_utility_gpu_queue_001", "--serve",
                         "--plan-sha256", sha, "--batch", str(index), "--owner", owner]}

    def bound_registration(root, sha):
        plan = previous_load(root, sha)
        if plan["source_sha256"].get("scripts/native_source_gpu_queue_004.py") != BASE_SHA256:
            raise ValueError("native_experience_utility_queue_invalid")
        return plan

    core.make_job, core.load_registration = utility_job, bound_registration
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def make_job(index, owner, plan_sha):
    return _get_core().make_job(index, owner, plan_sha)


def _validate_job(job):
    if (type(job) is not dict or type(job.get("max_runtime_seconds")) is not int
            or job != make_job(job.get("batch_index"), job.get("owner"), job.get("plan_sha256"))):
        raise ValueError("native_experience_utility_queue_invalid")


def load_registration(root, sha):
    return _get_core().load_registration(root, sha)


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    _validate_job(job)
    return _get_core().queue_operation(operation, job, root=root, known_pids=known_pids)


def record_gpu_entry(root, job, gpu):
    _validate_job(job)
    return _get_core().record_gpu_entry(root, job, gpu)


def fenced_release(runtime, job, known_pids):
    _validate_job(job)
    return _get_core().fenced_release(runtime, job, known_pids)


def serve(job, *, root=ROOT):
    _validate_job(job)
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
    args = parser.parse_args(argv)
    serve(make_job(args.batch, args.owner, args.plan_sha256))


if __name__ == "__main__":
    main()
