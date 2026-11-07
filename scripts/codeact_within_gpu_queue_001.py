"""Four fixed CodeAct batches over the unchanged serial GPU admission core.

Import, job construction and help do not enqueue or load the model. Registration
and predecessor checks are lazy; admission uses the existing global queue lock,
one-shot idle tickets, device guards and process identity checks.
"""

import argparse
import re
from importlib import import_module

from scripts import coverage_contract_gpu_queue as previous

ID = "qwen38-codeact-within-source-001"
PLAN = "artifacts/qwen38-codeact-within-source-plan-001.json"
PORT, BATCH_COUNT, TASKS_PER_BATCH = 18992, 4, 3
WAIT_SECONDS, GPU_SECONDS, CLIENT_SECONDS = 86400, 7200, 6000
MODEL_START_SECONDS, CLEANUP_SECONDS = 900, 180
BASE_SHA256 = "928293f7318892657381cb9a379c132b443afd51635f30fa4a6ca2a9c08489e8"
ROOT, REMOTE, PYTHON = previous.ROOT, previous.REMOTE, previous.PYTHON
SERVICE_PLAN, SERVICE_SHA = previous.SERVICE_PLAN, previous.SERVICE_SHA
GATE, GATE_SHA = previous.GATE, previous.GATE_SHA
GATE_B3, GATE_B3_SHA = previous.GATE_B3, previous.GATE_B3_SHA
MODEL_IDS = dict(previous.MODEL_IDS)
_CORE = None
_ERROR = "codeact_within_gpu_queue_invalid"


def make_job(index, owner, plan_sha):
    if (type(index) is not int or not 0 <= index < BATCH_COUNT
            or type(owner) is not str or re.fullmatch(r"[a-f0-9]{32}", owner) is None
            or type(plan_sha) is not str or re.fullmatch(r"[a-f0-9]{64}", plan_sha) is None):
        raise ValueError(_ERROR) from None
    return {"id": f"{ID}-comparison-{index}", "kind": "comparison", "batch_index": index,
            "owner": owner, "plan_sha256": plan_sha, "max_runtime_seconds": GPU_SECONDS,
            "argv": [PYTHON, "-B", "-m", "scripts.codeact_within_gpu_queue_001", "--serve",
                     "--plan-sha256", plan_sha, "--batch", str(index), "--owner", owner]}


def _registration():
    return import_module("scripts.codeact_within_registration_001")


def load_registration(root, sha):
    return _registration().load_registration(root, sha)


def _isolated(*, raw=None):
    if previous.SOURCE_SHA256 != BASE_SHA256:
        raise ValueError(_ERROR) from None
    core = previous._isolated(raw=raw)
    core.ID, core.PLAN, core.PORT, core.WAIT_SECONDS = ID, PLAN, PORT, WAIT_SECONDS
    core.MODEL_IDS = dict(MODEL_IDS)
    core.make_job, core.load_registration = make_job, load_registration
    return core


def _get_core():
    global _CORE
    if _CORE is None:
        _CORE = _isolated()
    return _CORE


def queue_operation(operation, job, *, root=REMOTE, known_pids=()):
    if (type(operation) is not str or operation not in {"enqueue", "pulse", "stop", "status"}
            or type(job) is not dict
            or job != make_job(job.get("batch_index"), job.get("owner"), job.get("plan_sha256"))):
        raise ValueError(_ERROR) from None
    if operation == "enqueue":
        plan = load_registration(root, job["plan_sha256"])
        _registration().check_predecessor(root, plan)
    # The inherited operation repeats registration under its existing global lock.
    return _get_core().queue_operation(operation, job, root=root, known_pids=known_pids)


def record_gpu_entry(root, job, gpu):
    return _get_core().record_gpu_entry(root, job, gpu)


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
