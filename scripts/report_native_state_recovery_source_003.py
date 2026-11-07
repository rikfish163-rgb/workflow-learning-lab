"""Reconcile scored source episodes and GPU releases without reading task answers."""

import argparse
import hashlib
import json
import math
import os
import re
import socket
import sys
from collections import Counter
from pathlib import Path

from scripts import native_state_recovery_writer_evidence_001 as evidence
from scripts import run_native_state_recovery_source_003 as source
from scripts.native_appworld_scorer import _main_pid
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _same
from scripts.native_source_registration import public_read
from wll.protocol import fingerprint

COUNTERS = ("generations", "prompt_tokens", "completion_tokens", "action_attempts", "api_invocations", "upstream_requests")
STOP_REASONS = {"supervisor_complete", "wall_seconds", "request_window_seconds", "max_generations", "max_actions",
                "max_api_requests", "max_input_tokens", "max_prompt_tokens_total", "max_completion_tokens_total"}
PLAN_SHA256 = evidence.SOURCE_PLAN_SHA256


def _require(condition):
    if not condition:
        raise ValueError("native_state_recovery_source_report_invalid")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _release(queue, plan, item, fence):
    job, release = item["job"], item["release"]
    state = release["state"]
    info = state["job"]
    ticket = info["ticket"]
    index = ticket["gpu_index"]
    name = job["id"]
    _require(release["verified"] is True and release["runtime_overrun"] is False
             and state["owned"] is state["admission_cancelled"] is state["released"] is True
             and state["model_pids"] == [] and info["status"] == "finished" and info["job_id"] == name
             and _finite(release["GPU_seconds"]) and release["GPU_seconds"] <= plan["gpu_seconds"]
             and release["GPU_seconds"] == release["service_wall_seconds"] == info["finished_epoch"] - info["started_epoch"]
             and type(index) is int and index in (8, 9) and info["gpu_index"] == index
             and ticket["job_id"] == name and ticket["policy_sha256"] == plan["gpu_policy_sha256"]
             and ticket["gpu_uuid"] == plan["gpu_policy"]["allowed_gpus"][str(index)]
             and _finite(ticket["idle_seconds"]) and ticket["idle_seconds"] >= 180
             and _finite(ticket["max_gap_seconds"]) and ticket["max_gap_seconds"] <= 35
             and type(ticket["sample_count"]) is int and ticket["sample_count"] >= 2
             and fence["job_sha256"] == fingerprint(job) and fence["boot_id"] == ticket["boot_id"]
             and _finite(fence["cancel_epoch"]) and fence["cancel_epoch"] <= state["epoch"])
    base = queue.REMOTE / "artifacts"
    _require(_same(info, _decode(public_read(base.parent, f"artifacts/gpu-idle-job-{name}.json")))
             and _same(ticket, _decode(public_read(base.parent, f"artifacts/gpu-idle-ticket-{name}.consumed.json")))
             and not os.path.lexists(base / f"gpu-idle-heartbeat-{name}"))
    return {"physical_GPU": index, "idle_seconds_at_admission": ticket["idle_seconds"],
            "lease_seconds": release["GPU_seconds"], "max_observation_gap_seconds": ticket["max_gap_seconds"]}


def _slot(episode, core, plan, private, slot, item, fence, queue):
    index, first = slot["source_index"], slot["first_attempt"]
    job = item["job"]
    _require(type(index) is int and 0 <= index < len(plan["task_ids"]) and slot["task_id"] == plan["task_ids"][index]
             and _same(job, queue.make_job(index, job["owner"], PLAN_SHA256)) and first["status"] == "known")
    gpu = _release(queue, plan, item, fence)
    owner = job["owner"]
    path = private / f"unit-{index:02d}-first_attempt-{owner}"
    score_path = private / f"score-{index:02d}-first_attempt-{owner}"
    row = core._context_row(WORKSPACE, plan, index)
    context = {key: row[key] for key in ("metadata_sha256", "messages_sha256", "input_tokens")}
    evidence._reconcile(episode, core, path, slot, context)
    receipt = _decode(core._read_private(path / "execution-receipt.json", maximum=episode.MAX_DOCUMENT))
    score = _decode(core._read_private(score_path / "scoring-result.json", maximum=4096))
    score_journal = episode._journal(core._read_private(score_path / "scoring-journal.jsonl", maximum=4096))
    manifest = receipt["snapshot"]["manifest_sha256"]
    parent = f"wll-native-source-{index:02d}-first_attempt-{owner}.service"
    score_unit = f"wll-native-source-{index:02d}-score-first_attempt-{owner}.service"
    _require(_same(score, {"status": "scored", "success": first["official_success"],
                          "receipt_sha256": first["receipt_sha256"], "manifest_sha256": manifest,
                          "is_policy_episode": True}) and type(first["official_success"]) is bool
             and type(first["model_reported_complete"]) is bool
             and first["model_reported_complete"] is (receipt["actor_outcome"]["status"] == "model_reported_complete")
             and _same(score_journal, [{"event": "scoring_intent", "receipt_sha256": first["receipt_sha256"],
                                         "parent_unit_name": parent},
                                        {"event": "validated_before_evaluation", "manifest_sha256": manifest},
                                        {"event": "scoring_stop", "status": "scored"}])
             and _main_pid(parent) == _main_pid(score_unit) == _main_pid(receipt["snapshot"]["unit_name"]) == 0)
    outcome = receipt["actor_outcome"]
    cost, stop_reason = outcome["cost"], outcome["stop_reason"]
    _require(all(type(cost.get(key)) is int and cost[key] >= 0 for key in COUNTERS))
    return {"index": index, "official_success": score["success"],
            "model_reported_complete": first["model_reported_complete"],
            "actor_stop_reason": stop_reason if stop_reason in STOP_REASONS else "other",
            "witness": slot["witness"], "cost": {key: cost[key] for key in COUNTERS}, "GPU": gpu}


def report(*, terminal_sha256=None):
    """Audit released, scored rows. A live prefix cannot qualify the cohort."""
    _require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
             and (terminal_sha256 is None or re.fullmatch(r"[a-f0-9]{64}", terminal_sha256)))
    reg, queue, episode, runner = source.bindings()
    plan = reg.load_registration(WORKSPACE, PLAN_SHA256, private=False)
    core = episode.bind(reg, queue).core
    private = Path(reg.PRIVATE_RUN_ROOT)
    launch = _decode(core._read_private(private / "parent-launch-intent.json", maximum=64_000))
    _require(launch["plan_sha256"] == PLAN_SHA256 and re.fullmatch(r"wll-native-source-pilot-[a-f0-9]{32}\.service", launch["unit"]))
    pid = _main_pid(launch["unit"])
    journal = episode._journal(core._read_private(private / "study-journal.jsonl", maximum=episode.MAX_DOCUMENT))
    finished = [row["slot"] for row in journal if row["event"] == "source_finished"]
    releases = [row for row in journal if row["event"] == "GPU_released"]
    fences = {row["job_sha256"]: row for row in journal if row["event"] == "release_fence_started"}
    _require([row["source_index"] for row in finished] == list(range(len(finished)))
             and [row["job"]["batch_index"] for row in releases] == list(range(len(releases)))
             and abs(len(releases) - len(finished)) <= 1)
    rows = [_slot(episode, core, plan, private, slot, item, fences[fingerprint(item["job"])], queue)
            for slot, item in zip(finished, releases, strict=False)]
    gate, terminal_status_counts = None, None
    if terminal_sha256 is not None:
        _require(pid == 0 and journal[-1]["event"] == "study_stop")
        raw = core._read_private(private / "study-result.json", maximum=episode.MAX_DOCUMENT)
        result = _decode(raw)
        _require(_hash(raw) == terminal_sha256 and _same(journal[-1]["report"], result)
                 and result["id"] == reg.ID and result["plan_sha256"] == PLAN_SHA256
                 and result["automatic_retry"] is False and result["slots"][:len(finished)] == finished
                 and _same(result["source_gate"], runner.gate(result)))
        gate = result["source_gate"]
        terminal_status_counts = Counter(slot["first_attempt"]["status"] for slot in result["slots"])
        _require(set(terminal_status_counts) <= {"known", "unknown", "unstarted"}
                 and sum(terminal_status_counts.values()) == len(plan["task_ids"]))
        if gate["status"] != "not_evaluable":
            _require(len(rows) == len(plan["task_ids"]))
    return {"plan_sha256": PLAN_SHA256, "terminal_result_sha256": terminal_sha256,
            "coordinator_running": pid > 0, "source_denominator": len(plan["task_ids"]),
            "audited_scored_and_released": len(rows), "gate": gate, "terminal_status_counts": terminal_status_counts,
            "audited_cost_subtotal": {key: sum(row["cost"][key] for row in rows) for key in COUNTERS},
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-sha256")
    args = parser.parse_args()
    print(json.dumps(report(terminal_sha256=args.terminal_sha256), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.stderr.write("native_state_recovery_source_report_invalid\n")
        raise SystemExit(2) from None
