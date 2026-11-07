"""One serial lease per source episode, using the existing GPU watcher and fence."""

import argparse
import hashlib
import os
import re
import socket
from functools import cache
from pathlib import Path

from scripts import native_source_gpu_queue_004 as source4
from scripts import native_state_recovery_registration_001 as registration

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/home/hts/workflow-learning-lab")
ID, PLAN = registration.ID, registration.PLAN
GUARD_NAME = "native-state-recovery-source-operations-001.lock"
SOURCE = "scripts/native_source_gpu_queue_004.py"
SOURCE_SHA256 = "2e02ffd39f188608066990997b16515c9bece698a9668355d19107c5a8b8f995"


def make_job(index, owner, plan_sha):
    if (type(index) is not int or not 0 <= index < len(registration.TASK_IDS)
            or type(owner) is not str or re.fullmatch(r"[a-f0-9]{32}", owner) is None
            or type(plan_sha) is not str or re.fullmatch(r"[a-f0-9]{64}", plan_sha) is None):
        raise ValueError("native_state_recovery_queue_invalid")
    return {"id": f"{ID}-comparison-{index}", "kind": "comparison", "batch_index": index,
            "owner": owner, "plan_sha256": plan_sha, "max_runtime_seconds": 7200,
            "argv": [registration.PYTHON, "-B", "-m", "scripts.native_state_recovery_gpu_queue_001", "--serve",
                     "--plan-sha256", plan_sha, "--batch", str(index), "--owner", owner]}


@cache
def _core():
    raw = registration.public_read(ROOT, SOURCE)
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256 or Path(source4.__file__).resolve() != ROOT / SOURCE:
        raise ValueError("native_state_recovery_queue_source_invalid")
    core = source4._isolated(raw=registration.public_read(ROOT, "scripts/native_source_gpu_queue_003.py"))
    core.ID, core.PLAN, core.GUARD_NAME, core.PORT = ID, PLAN, GUARD_NAME, registration.PORT
    core.make_job, core.load_registration = make_job, registration.load_registration
    return core


def __getattr__(name):
    if name in {"ready_path", "entry_path", "queue_operation", "fenced_release", "runtime_state"}:
        return getattr(_core(), name)
    raise AttributeError(name)


def serve(job):
    from http.server import ThreadingHTTPServer

    from scripts.serve_qwen38 import make_handler
    from scripts.serve_qwen38_b3_runtime_relocation_001 import loaded_engine
    from wll.protocol import fingerprint

    core = _core()
    core._validate_job(job)
    if socket.gethostname() != "amax" or ROOT != REMOTE or Path.cwd().resolve() != REMOTE:
        raise ValueError("native_state_recovery_queue_host_invalid")
    plan = registration.load_registration(ROOT, job["plan_sha256"])
    core.check_admission_window(ROOT, job)
    runtime = core._get_core()
    with loaded_engine(plan["runtime_relocation"]["sha256"], plan["service_plan_sha256"], plan["model_id"],
                       admitted=lambda gpu: core.record_gpu_entry(ROOT, job, gpu)) as engine:
        if not registration._same(registration.static_backend(engine.backend), plan["expected_backend_static"]["B3"]):
            raise ValueError("native_state_recovery_backend_invalid")
        server = ThreadingHTTPServer(("127.0.0.1", registration.PORT), make_handler(engine))
        try:
            runtime.write_new(core.ready_path(ROOT / "artifacts", job), {
                "job_sha256": fingerprint(job), "ready_epoch": runtime.epoch(),
                "ready_monotonic": runtime.monotonic(), "backend_sha256": fingerprint(engine.backend),
                "pid": os.getpid(),
            })
            server.serve_forever()
        finally:
            server.server_close()
        if engine.failed:
            raise RuntimeError("native_state_recovery_backend_failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    serve(make_job(args.batch, args.owner, args.plan_sha256))


if __name__ == "__main__":
    main()
