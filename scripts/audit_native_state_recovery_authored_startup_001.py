"""Audit the frozen authored startup failure before creating another lineage."""

import os
from pathlib import Path

from scripts import native_appworld_scorer as scorer
from scripts import run_native_state_recovery_authored_study_001 as study
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _same
from scripts.native_source_registration import public_read

PLAN_SHA = "498080cf9576541fee2b2b20a8d9b528f55085b397a2c0ea41772774908595e4"
RESULT_SHA = "c87c75127ed89bb072ffb3081adc8c458785d1ec5b009c343ff3097e9b040636"


def report():
    study._require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and Path.cwd() == WORKSPACE)
    study.load_public(WORKSPACE, PLAN_SHA)
    runtime = study.Runtime(WORKSPACE)
    root = study.PRIVATE
    try:
        launch = _decode(runtime.read(root / "parent-launch-intent.json"))
        study._require(launch["plan_sha256"] == PLAN_SHA
                       and launch["unit"] == study._coordinator_unit(PLAN_SHA)
                       and scorer._main_pid(launch["unit"]) == 0)
        raw = runtime.read(root / "study-result.json")
        result = _decode(raw)
        journal = study.frozen._core()._journal(runtime.read(root / "study-journal.jsonl"))
        study._require(study._hash(raw) == RESULT_SHA and journal[-1]["event"] == "study_stop"
                       and _same(journal[-1]["report"], result) and result["status"] == "unknown"
                       and result["automatic_retry"] is False and not result["executions"]
                       and not result["costs"]["episodes"] and not result["costs"]["scorers"]
                       and result["utility"]["complete_pairs"] == 0 and result["utility"]["statistics"] is None
                       and all(not list((root / name).iterdir()) for name in ("configs", "runs", "scores"))
                       and not any(event["event"] in ("GPU_ready", "unit_launch_intent") for event in journal))
        item, = result["costs"]["GPU_jobs"]
        job, release = item["job"], item["release"]
        queued, = (row["job"] for row in journal if row["event"] == "GPU_enqueue_intent")
        study._require(_same(job, queued) and _same(job, study._job(0, job["owner"], PLAN_SHA)))
        state = release["state"]
        queue = study._queue()
        art = queue.REMOTE / "artifacts"
        name = job["id"]
        info = _decode(public_read(queue.REMOTE, f"artifacts/gpu-idle-job-{name}.json"))
        log = public_read(queue.REMOTE, f"artifacts/gpu-idle-job-{name}.log")
        ticket = info["ticket"]
        study._require(release["verified"] is True and release["runtime_overrun"] is False
                       and state["owned"] is state["admission_cancelled"] is state["released"] is True
                       and state["model_pids"] == [] and _same(state["job"], info)
                       and info["status"] == "finished" and info["exit_code"] == 1
                       and ticket["gpu_index"] == info["gpu_index"] == 8 and ticket["idle_seconds"] >= 180
                       and ticket["max_gap_seconds"] <= 35
                       and not any(path.exists() or path.is_symlink() for path in (
                           queue.entry_path(art, job), queue.ready_path(art, job),
                           art / f"gpu-idle-ticket-{name}.consumed.json",
                       ))
                       and log.splitlines()[-1].startswith(b"ValueError: service runtime changed: "))
        study._require(runtime.read(root / "study-result.json") == raw and scorer._main_pid(launch["unit"]) == 0)
        return {
            "scope": "authored_startup_failed_before_admission", "plan_sha256": PLAN_SHA,
            "result_sha256": RESULT_SHA, "backend_failure": "legacy_runtime_digest_mismatch",
            "physical_gpu_selected_by_watcher": 8, "observed_idle_seconds": ticket["idle_seconds"],
            "max_observation_gap_seconds": ticket["max_gap_seconds"], "ticket_consumed": False,
            "backend_ready": False, "executed_branches": 0, "model_calls": 0,
            "release_verified": True, "watcher_service_wall_seconds": release["GPU_seconds"],
            "automatic_retry": False,
        }
    finally:
        runtime.close()


if __name__ == "__main__":
    print(_encode(report()).decode())
