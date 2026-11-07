"""Collect the fixed source episodes and score closed worlds independently."""

import argparse
import ast
import hashlib
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

from scripts import native_state_recovery_episode_001 as episode
from scripts import native_state_recovery_evidence_001 as witness
from scripts import native_state_recovery_gpu_queue_001 as queue
from scripts import native_state_recovery_registration_001 as registration
from scripts import run_native_source_pilot_006 as pilot
from scripts.native_appworld_worker_client import WORKSPACE, _decode

ID = registration.ID
PILOT = "scripts/run_native_source_pilot.py"
PILOT_SHA256 = "c9cac394a244e8764d3bbd7143d3b44e0db00e20cc5f6016d84504d89bbee7f6"
PARENT_SECONDS = len(registration.TASK_IDS) * (86400 + 7200) + 600
_ERROR = "native_state_recovery_source_failed"


def _require(condition):
    if not condition:
        raise ValueError(_ERROR)


def _core():
    core = pilot._isolated()
    core.ID, core.PLAN = ID, registration.PLAN
    core.registration, core.queue = registration, queue
    core.episode = episode.bind(registration, queue).core
    core.select_prefix = lambda *_: None
    raw = registration.public_read(WORKSPACE, PILOT)
    _require(hashlib.sha256(raw).hexdigest() == PILOT_SHA256)
    source = raw.decode("utf-8")
    function, = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name == "_run_episode"]
    text = ast.get_source_segment(source, function)
    _require(text.count("native-source-001-") == 1 and not function.decorator_list)
    exec(compile(text.replace("native-source-001-", "native-state-recovery-001-"),
                 str(WORKSPACE / PILOT), "exec", dont_inherit=True), core.__dict__)
    unit_command = core.unit_command

    def command(kind, *args, **kwargs):
        values = unit_command(kind, *args, **kwargs)
        return [value.replace("scripts.native_source_episode_006", "scripts.run_native_state_recovery_source_001")
                .replace("scripts.run_native_source_pilot_006", "scripts.run_native_state_recovery_source_001")
                .replace(f"RuntimeMaxSec={pilot.PARENT_SECONDS}", f"RuntimeMaxSec={PARENT_SECONDS}")
                for value in values]

    core.unit_command = command
    return core


@contextmanager
def _lease(runtime, core, job, report):
    heartbeat, known_pids = None, []
    started = runtime.now()
    runtime.monitor(empty_port=True)
    runtime.record({"event": "GPU_enqueue_intent", "job": job})
    try:
        state = runtime.queue("enqueue", job)
        core._queue_state(state, runtime)
        heartbeat = runtime.heartbeat(job)
        while True:
            heartbeat.check()
            runtime.monitor()
            state = runtime.queue("status", job)
            core._queue_state(state, runtime)
            _require(state.get("released") is False)
            info = state.get("job") or {}
            if info.get("status") == "running" and runtime.epoch() - info["started_epoch"] >= 600:
                raise core._BudgetStop("model_ready_budget")
            service = runtime.ready(job, state)
            if service is not None:
                _require(runtime.epoch() - info["started_epoch"] < 600)
                known_pids = list(state["model_pids"])
                break
            if runtime.now() - started >= registration.WAIT_SECONDS:
                raise core._BudgetStop("admission_timeout")
            runtime.wait(core.PULSE_SECONDS)

        def guard():
            heartbeat.check()
            if runtime.epoch() - info["started_epoch"] >= registration.BATCH_GPU_SECONDS - 150:
                raise core._BudgetStop("GPU_cap")
            latest = runtime.queue("status", job)
            core._queue_state(latest, runtime)
            _require(latest.get("released") is False and latest.get("job", {}).get("status") == "running")

        runtime.record({"event": "GPU_ready", "job": job, "service": service})
        yield service, guard
    finally:
        try:
            if heartbeat is not None:
                heartbeat.stop()
        finally:
            release = {"verified": False, "GPU_seconds": None, "runtime_overrun": None}
            try:
                release = queue.fenced_release(runtime, job, known_pids)
            finally:
                report["costs"]["GPU_jobs"].append({"job": job, "release": release,
                    "admission_to_release_seconds": runtime.now() - started})
                runtime.record({"event": "GPU_released", "job": job, "release": release})
            _require(release["verified"] is True and release["runtime_overrun"] is False)


def gate(report):
    slots, costs = report["slots"], report["costs"]
    count = len(registration.TASK_IDS)
    known = (report["status"] == "complete" and len(slots) == count
             and [row["task_id"] for row in slots] == list(registration.TASK_IDS)
             and all(row["first_attempt"].get("status") == "known"
                     and row["witness"]["status"] != "unknown" for row in slots)
             and [row["source_index"] for row in costs["episodes"]] == list(range(count))
             and [row["source_index"] for row in costs["scorers"]] == list(range(count))
             and [row["job"]["batch_index"] for row in costs["GPU_jobs"]] == list(range(count))
             and all(row["counts_complete"] is True and row["unknown_cost"] is False for row in costs["episodes"])
             and all(row["cleanup_verified"] is True and row["result_validated"] is True
                     for row in costs["scorers"])
             and all(row["release"]["verified"] is True and row["release"]["runtime_overrun"] is False
                     for row in costs["GPU_jobs"]))
    qualified = {row["task_id"][:-2] for row in slots if row["witness"]["status"] == "qualified"}
    return {"status": ("passed" if len(qualified) >= 2 else "insufficient") if known else "not_evaluable",
            "witness_scenarios": len(qualified), "source_denominator": count}


def run_source(plan_sha, *, core=None, runtime=None):
    core = core or _core()
    runtime = runtime or core._Runtime(WORKSPACE)
    plan = runtime.load_plan(plan_sha)
    slots = [{"source_index": i, "task_id": task, "first_attempt": {"status": "unstarted"},
              "witness": {"status": "unstarted"}} for i, task in enumerate(registration.TASK_IDS)]
    report = {"id": ID, "plan_sha256": plan_sha, "status": "unknown", "slots": slots,
              "costs": {"episodes": [], "scorers": [], "GPU_jobs": []}, "automatic_retry": False,
              "writer_calls": 0, "training_jobs": 0, "utility_calls": 0}
    runtime.begin(plan, plan_sha, slots)
    slot = None
    try:
        rows = runtime.context_rows(plan)
        _require([row["task_id"] for row in rows] == list(registration.TASK_IDS))
        for index, (row, slot) in enumerate(zip(rows, slots, strict=True)):
            job = queue.make_job(index, runtime.owner(), plan_sha)
            with _lease(runtime, core, job, report) as (service, guard):
                slot["first_attempt"] = {"status": "running"}
                initial = {key: row[key] for key in ("metadata_sha256", "messages_sha256", "input_tokens")}
                first, _, _ = core._run_episode(runtime, plan, plan_sha, job, service, index, "first_attempt",
                                               initial, None, guard, report)
                path = Path(plan["private_run_root"]) / f"unit-{index:02d}-first_attempt-{job['owner']}"
                raw = runtime.read(path / episode.EVIDENCE_NAME)
                evidence = _decode(raw)
                _require(evidence["private"] is True and evidence["id"] == "native-state-recovery-closed-evidence-001")
                slot.update(first_attempt=first, witness=witness.public_projection(evidence["witness"]),
                            witness_sha256=hashlib.sha256(raw).hexdigest())
                runtime.record({"event": "source_finished", "slot": slot})
        _require(sum(row["release"]["GPU_seconds"] for row in report["costs"]["GPU_jobs"]) <= plan["total_gpu_seconds"])
        report["status"] = "complete"
        _require(gate(report)["status"] != "not_evaluable")
    except BaseException as error:
        report["status"] = "unknown"
        if slot is not None and slot["first_attempt"]["status"] == "running":
            slot.update(first_attempt={"status": "unknown"}, witness={"status": "unknown"})
        core._capture_initiating_exception(runtime, error)
    finally:
        report["source_gate"] = gate(report)
        try:
            runtime.record({"event": "study_stop", "report": report})
            runtime.write("study-result.json", report)
        finally:
            runtime.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    choice = parser.add_mutually_exclusive_group(required=False)
    choice.add_argument("--run", action="store_true")
    choice.add_argument("--launch", action="store_true")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--config")
    parser.add_argument("--config-sha256")
    args = parser.parse_args()
    if args.run or args.launch:
        _require(args.plan_sha256 and not args.config and not args.config_sha256)
        if args.launch:
            print(_core().launch_pilot(args.plan_sha256, root=WORKSPACE)["unit"])
            return 0

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        return int(run_source(args.plan_sha256)["status"] != "complete")
    _require(not args.plan_sha256 and args.config and args.config_sha256)
    receipt = episode.bind(registration, queue).run_episode(args.config, args.config_sha256)
    return int(receipt["status"] != "saved_and_closed")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        sys.stderr.write(_ERROR + "\n")
        raise SystemExit(2) from None
