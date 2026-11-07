"""Authenticate progress on one live V27 run without projecting interim effects."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

from scripts import native_appworld_scorer as scorer
from scripts import report_native_state_recovery_utility_001 as audit
from scripts import run_native_state_recovery_transition_utility_003 as utility
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _same
from scripts.native_source_registration import public_read
from wll.protocol import fingerprint


def _study_journal(read, path, parse):
    # A live fsync can change the size or timestamp during the private read.
    for attempt in range(3):
        try:
            return parse(read(path))
        except ValueError:
            if attempt == 2:
                raise
            time.sleep(0.2)


def _admission(plan, row, queue, boot_id):
    job, service = row["job"], row["service"]
    name = "artifacts/gpu-idle-ticket-" + job["id"] + ".consumed.json"
    ticket = _decode(public_read(queue.REMOTE, name))
    gpu = ticket["gpu_index"]
    audit._require(type(gpu) is int and gpu in (8, 9)
                   and service["backend"]["gpu_policy"]["physical_gpu_index"] == gpu
                   and ticket["job_id"] == job["id"] and ticket["policy_sha256"] == plan["gpu_policy_sha256"]
                   and ticket["gpu_uuid"] == plan["gpu_policy"]["allowed_gpus"][str(gpu)]
                   and ticket["boot_id"] == boot_id and type(ticket["sample_count"]) is int
                   and ticket["sample_count"] >= 2
                   and audit.source_report._finite(ticket["idle_seconds"]) and ticket["idle_seconds"] >= 180
                   and audit.source_report._finite(ticket["max_gap_seconds"]) and ticket["max_gap_seconds"] <= 35)
    info = _decode(public_read(queue.REMOTE, "artifacts/gpu-idle-job-" + job["id"] + ".json"))
    audit._require(info["job_id"] == job["id"] and info["gpu_index"] == gpu
                   and _same(info["ticket"], ticket) and audit.source_report._finite(info["started_epoch"]))
    return gpu, ticket["idle_seconds"], info


def _durable_finish(queue, job, info, fence, boot_id):
    if fence is None or info["status"] != "finished":
        return False
    cancel = _decode(public_read(queue.REMOTE, "artifacts/completion-admission-cancelled-" + job["id"] + ".json"))
    heartbeat = queue.REMOTE / "artifacts" / ("gpu-idle-heartbeat-" + job["id"])
    audit._require(cancel["reason"] == "controller_stop" and cancel["job_sha256"] == fingerprint(job)
                   and fence["boot_id"] == boot_id and audit.source_report._finite(cancel["epoch"])
                   and cancel["epoch"] <= fence["cancel_epoch"] and not os.path.lexists(heartbeat)
                   and audit.source_report._finite(info["finished_epoch"])
                   and 0 <= info["finished_epoch"] - info["started_epoch"] <= job["max_runtime_seconds"])
    return True


def report(expected_pid):
    audit._require(socket.gethostname() == "amax" and Path.cwd() == WORKSPACE
                   and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and type(expected_pid) is int and expected_pid > 0)
    reg, queue, runner = utility.bindings()
    reader = runner._services(WORKSPACE)[1]._read_private
    private = Path(reg.PRIVATE_RUN_ROOT)
    core = runner._core()

    def read(path):
        return reader(path, maximum=64_000_000)

    launch = _decode(read(private / "parent-launch-intent.json"))
    intent = _decode(read(private / "study-intent.json"))
    sha, unit = launch["plan_sha256"], launch["unit"]
    adapter = audit._adapter(utility.LINEAGE)
    adapter._verify_launch(launch, intent, sha)
    audit._require(intent["pid"] == expected_pid and scorer._main_pid(unit) == expected_pid)
    plan = reg.load_registration(WORKSPACE, sha, private=False)
    audit._require(_same(intent["summaries"], runner._summaries(plan)))
    bundle_raw = read(private / "reviewed-bundle.json")
    audit._require(audit._hash(bundle_raw) == intent["reviewed_bundle_sha256"])
    bundle = _decode(bundle_raw)
    runner._validate_bundle(plan, bundle)
    journal = _study_journal(read, private / "study-journal.jsonl", core._journal)
    enqueued, ready, fences = adapter._journal_jobs(journal, sha)
    units = {row["unit"] for row in journal if row["event"] == "unit_launch_return" and row["exit_code"] == 0}
    evidence = {"plan_sha256": sha, "costs": {"GPU_jobs": []}, "executions": [], "summaries": []}
    admissions, finishes = [], []
    for batch, row in ready.items():
        job = row["job"]
        gpu, idle, info = _admission(plan, row, queue, intent["boot_id"])
        if admissions:
            audit._require(finishes[-1] and admissions[-1][2]["finished_epoch"] <= info["started_epoch"])
        admissions.append((gpu, idle, info))
        finishes.append(_durable_finish(queue, job, info, fences.get(fingerprint(job)), intent["boot_id"]))
        evidence["costs"]["GPU_jobs"].append({"job": job})
        owner = job["owner"]
        for index in plan["batches"][batch]:
            metadata_hashes = []
            for arm in plan["branch_orders"][index]:
                scorer_unit = f"wll-native-source-utility-{index:02d}-score-{arm}-{owner}.service"
                score_path = private / f"score-{index:02d}-{arm}-{owner}/scoring-result.json"
                if scorer_unit not in units or not score_path.exists() or scorer._main_pid(scorer_unit) != 0:
                    continue
                directory = private / f"unit-{index:02d}-{arm}-{owner}"
                receipt_path = directory / "execution-receipt.json"
                receipt = read(receipt_path)
                controller = core._journal(read(directory / "controller.jsonl"))
                metadata = core._one(controller, "metadata_result")["reply"]["result"]
                metadata_sha = audit._hash(_encode(metadata))
                metadata_hashes.append(metadata_sha)
                evidence["executions"].append({"pair_index": index, "arm": arm,
                    "receipt_binding": {"path": str(receipt_path), "sha256": audit._hash(receipt)},
                    "metadata_sha256": metadata_sha, "actor_status": _decode(receipt)["actor_outcome"]["status"]})
                evidence["summaries"].append({"task_id": plan["expected_pairs"][index][0], "arm": arm,
                    "status": "known", "success": _decode(read(score_path))["success"]})
            audit._require(len(set(metadata_hashes)) <= 1)
    snapshots = {}
    verified = audit._audit_scores(plan, evidence, bundle, reader, snapshots, runner, queue)
    audit._require(len(verified) == len(evidence["executions"])
                   and all(audit._hash(read(path)) == digest for path, digest in snapshots.items())
                   and scorer._main_pid(unit) == expected_pid)
    return {"plan_sha256": sha, "coordinator_pid": expected_pid, "planned_pairs": len(plan["expected_pairs"]),
            "planned_branches": 2 * len(plan["expected_pairs"]), "audited_scored_branches": len(verified),
            "audited_complete_pairs": sum((index, "control") in verified and (index, "experience") in verified
                                          for index in range(len(plan["expected_pairs"]))),
            "gpu_jobs_enqueued": len(enqueued), "gpu_tickets_audited": len(admissions),
            "release_fences_started": sum(fingerprint(row["job"]) in fences for row in enqueued.values()),
            "durable_gpu_finishes_audited": sum(finishes),
            "physical_gpus_audited": sorted({gpu for gpu, _, _ in admissions}),
            "minimum_idle_seconds": min((seconds for _, seconds, _ in admissions), default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.pid), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("native_state_recovery_transition_progress_audit_unavailable") from None
