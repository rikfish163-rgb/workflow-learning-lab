"""Verify the relocated service stopped at backend comparison before any actor."""

from scripts import native_appworld_scorer as scorer
from scripts import report_native_state_recovery_source_003 as release_audit
from scripts import run_native_state_recovery_authored_study_002 as study
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _same
from scripts.native_source_registration import public_read

PLAN_SHA = "cd7656e072a278c5cf79b50bc067b16b3838d91b1aaaba2cfc31c27e68d923af"
RESULT_SHA = "3499548b1550864ad3e315bbbf61583f8c167acabe4a4a1049be78b7fae6e6d3"


def report():
    core = study.bindings()
    plan, _bundle = core._preflight(WORKSPACE, PLAN_SHA)
    root = core.PRIVATE
    runtime = core.Runtime(WORKSPACE)
    try:
        launch = _decode(runtime.read(root / "parent-launch-intent.json"))
        raw = runtime.read(root / "study-result.json")
        result = _decode(raw)
        journal = core.frozen._core()._journal(runtime.read(root / "study-journal.jsonl"))
        core._require(launch["plan_sha256"] == PLAN_SHA and launch["unit"] == core._coordinator_unit(PLAN_SHA)
                      and scorer._main_pid(launch["unit"]) == 0 and core._hash(raw) == RESULT_SHA
                      and _same(journal[-1], {"event": "study_stop", "report": result})
                      and result["status"] == "unknown" and result["automatic_retry"] is False
                      and not result["executions"] and not result["costs"]["episodes"] and not result["costs"]["scorers"]
                      and result["utility"]["complete_pairs"] == 0 and result["utility"]["statistics"] is None
                      and all(not list((root / name).iterdir()) for name in ("configs", "runs", "scores"))
                      and not any(row["event"] in ("GPU_ready", "unit_launch_intent") for row in journal))
        item, = result["costs"]["GPU_jobs"]
        fence, = (row for row in journal if row["event"] == "release_fence_started")
        job = item["job"]
        queue = core._queue()
        core._require(_same(job, queue.make_job(0, job["owner"], PLAN_SHA)))
        released = release_audit._release(queue, plan, item, fence)
        art = queue.REMOTE / "artifacts"
        core._require(queue.entry_path(art, job).is_file() and not queue.ready_path(art, job).exists()
                      and public_read(queue.REMOTE, f"artifacts/gpu-idle-job-{job['id']}.log").splitlines()[-1]
                      == b"ValueError: native_state_recovery_backend_invalid")
        return {"scope": "authored_backend_comparison_failed_before_actor", "plan_sha256": PLAN_SHA,
                "result_sha256": RESULT_SHA, "backend_failure": "strict_comparison_before_json_normalization",
                "backend_ready": False, "executed_branches": 0, "model_calls": 0,
                "release_verified": True, "automatic_retry": False, **released}
    finally:
        runtime.close()


if __name__ == "__main__":
    print(_encode(report()).decode())
