"""Independent source006 singleton leases with authenticated source005 guards.

Import/job/help are inert. Enqueue authenticates actual predecessor and CPU proof
metadata inside the retained operation guard. GPU serve uses public validation;
full asset/directory preflight remains the coordinator's responsibility.
"""

import argparse
import hashlib
import importlib.util
import re
from pathlib import Path

from scripts import native_source_registration_006 as registration

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "scripts/native_source_gpu_queue_005.py"
BASE_SHA256 = "384cb576621de84e3cb2484a813f3f0e7d21f8a2fa13c51f834a96cfa4a7eb01"
ID, PLAN, PRIVATE_RUN_ROOT = registration.ID, registration.PLAN, registration.PRIVATE_RUN_ROOT
PORT, PYTHON = registration.PORT, registration.PYTHON
SERVICE_PLAN, SERVICE_SHA = registration.SERVICE_PLAN, registration.SERVICE_SHA
WAIT_SECONDS, BATCH_GPU_SECONDS = registration.WAIT_SECONDS, registration.BATCH_GPU_SECONDS
REMOTE = Path("/home/hts/workflow-learning-lab")
GUARD_NAME = "native-source-operations-006.lock"
_CORE = None


def make_job(index, owner, plan_sha):
    if (type(index) is not int or not 0 <= index < 12 or type(owner) is not str
            or re.fullmatch(r"[a-f0-9]{32}", owner) is None or type(plan_sha) is not str
            or re.fullmatch(r"[a-f0-9]{64}", plan_sha) is None):
        raise ValueError("native_source_gpu_queue_006_invalid") from None
    return {"id": f"{ID}-comparison-{index}", "kind": "comparison", "batch_index": index,
        "owner": owner, "plan_sha256": plan_sha, "max_runtime_seconds": BATCH_GPU_SECONDS,
        "argv": [PYTHON, "-B", "-m", "scripts.native_source_gpu_queue_006", "--serve",
                 "--plan-sha256", plan_sha, "--batch", str(index), "--owner", owner]}


def load_registration(root, sha):
    plan = registration.load_registration(root, sha, private=False)
    if (type(plan) is not dict or plan.get("id") != ID or type(plan.get("source_sha256")) is not dict
            or plan["source_sha256"].get("scripts/native_source_gpu_queue_005.py") != BASE_SHA256):
        raise ValueError("native_source_gpu_queue_006_invalid") from None
    return plan


def _isolated(*, raw=None):
    from scripts.native_source_registration import public_read

    raw = public_read(ROOT, "scripts/native_source_gpu_queue_005.py") if raw is None else raw
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError("native_source_gpu_queue_006_source_invalid") from None
    spec = importlib.util.spec_from_file_location("scripts._native_source_gpu_queue_006_parent", BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise ValueError("native_source_gpu_queue_006_source_invalid") from None
    parent = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE_SOURCE), "exec", dont_inherit=True), parent.__dict__)
    core = parent._isolated()
    if (core.PORT, core.PYTHON, core.SERVICE_PLAN, core.SERVICE_SHA, core.WAIT_SECONDS,
            core.BATCH_GPU_SECONDS, core.REMOTE) != (
            PORT, PYTHON, SERVICE_PLAN, SERVICE_SHA, WAIT_SECONDS, BATCH_GPU_SECONDS, REMOTE):
        raise ValueError("native_source_gpu_queue_006_binding_invalid") from None
    core.ID, core.PLAN, core.GUARD_NAME = ID, PLAN, GUARD_NAME
    core.make_job, core.load_registration = make_job, load_registration

    def guarded_operation(operation, job, *, root=REMOTE, known_pids=()):
        if operation not in {"enqueue", "pulse", "stop", "status"}:
            raise ValueError("native_source_gpu_queue_006_invalid") from None
        core._validate_job(job)
        with core._operation_guard(root):
            if operation == "enqueue":
                plan = load_registration(root, job["plan_sha256"])
                registration.check_predecessor(root, plan)
            return core._get_core().queue_operation(operation, job, root=root, known_pids=known_pids)

    core.queue_operation = guarded_operation
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


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
    args = parser.parse_args(argv)
    serve(make_job(args.batch, args.owner, args.plan_sha256))


if __name__ == "__main__":
    main()
