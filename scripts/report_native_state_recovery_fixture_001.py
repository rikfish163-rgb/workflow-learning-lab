"""Audit authored workflow outcomes and derive only typed recovery facts."""

import argparse
import json
import os
import socket
import sys
from pathlib import Path

from scripts import native_state_recovery_fixture_episode_001 as fixture
from scripts import native_state_recovery_fixture_protocol_001 as protocol
from scripts import native_state_recovery_writer_evidence_001 as writer
from scripts import report_native_state_recovery_source_003 as natural
from scripts import run_native_state_recovery_fixture_001 as run
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _same
from wll.protocol import fingerprint


def _read_terminal(plan_sha, result_sha):
    require = run._require
    require(socket.gethostname() == "amax" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and run._sha(plan_sha) and run._sha(result_sha))
    reg, queue, episode, _runner = run.bindings()
    plan = reg.load_registration(WORKSPACE, plan_sha, private=False)
    run._preflight(plan)
    core = episode.bind(reg, queue).core
    root = Path(reg.PRIVATE_RUN_ROOT)

    def read(path):
        return core._read_private(path, maximum=episode.MAX_DOCUMENT)

    launch = _decode(read(root / "parent-launch-intent.json"))
    require(launch["plan_sha256"] == plan_sha and run._main_pid(launch["unit"]) == 0)
    raw = read(root / "study-result.json")
    result = _decode(raw)
    journal_raw = read(root / "study-journal.jsonl")
    journal = episode._journal(journal_raw)
    require(run._hash(raw) == result_sha and result["id"] == reg.ID and result["plan_sha256"] == plan_sha
            and result["automatic_retry"] is False and type(result["official_scorer_calls"]) is int
            and result["official_scorer_calls"] == 0 and len(result["slots"]) == len(reg.TASK_IDS) == 4
            and journal[-1]["event"] == "study_stop" and _same(journal[-1]["report"], result)
            and _same(result["source_gate"], run.gate(result, reg.TASK_IDS)))
    releases = {row["job"]["batch_index"]: row for row in result["costs"]["GPU_jobs"]}
    fences = {row["job_sha256"]: row for row in journal if row["event"] == "release_fence_started"}
    ended = {row["slot"]["source_index"]: row["slot"] for row in journal if row["event"] == "fixture_finished"}
    rows, cases, provenance = [], [], []
    for index, (slot, context_row) in enumerate(zip(result["slots"], plan["design"]["rows"], strict=True)):
        require(slot["source_index"] == index and slot["template_task_id"] == reg.TASK_IDS[index])
        if slot["status"] != "known":
            require(slot["status"] in {"unstarted", "unknown"})
            rows.append({"index": index, "status": slot["status"]})
            continue
        require(_same(ended[index], slot))
        item = releases[index]
        job = item["job"]
        owner = job["owner"]
        require(_same(job, queue.make_job(index, owner, plan_sha)))
        gpu = natural._release(queue, plan, item, fences[fingerprint(job)])
        path = root / f"unit-{index:02d}-first_attempt-{owner}"
        receipt_raw = read(path / "execution-receipt.json")
        receipt = _decode(receipt_raw)
        saved = read(path / episode.EVIDENCE_NAME)
        require(run._hash(receipt_raw) == slot["receipt_sha256"] and run._hash(saved) == slot["witness_sha256"]
                and receipt["is_policy_episode"] is True
                and run._main_pid(receipt["snapshot"]["unit_name"]) == 0
                and run._main_pid(f"wll-native-source-{index:02d}-first_attempt-{owner}.service") == 0)
        context = fixture._case_context(context_row, protocol.CASES[index])
        evidence = fixture.collect(episode, core, path, receipt, context)
        target = evidence["fixture"]
        require(_same(_decode(saved), evidence) and _same(target["witness"], slot["witness"])
                and target["fixture_target_verified"] is slot["fixture_target_verified"]
                and receipt["actor_outcome"]["stop_reason"] == slot["actor_stop_reason"])
        outcome = receipt["actor_outcome"]
        stop_reason = outcome["stop_reason"]
        rows.append({"index": index, "status": "known", "witness": target["witness"],
                     "fixture_target_verified": target["fixture_target_verified"], "GPU": gpu,
                     "actor_stop_reason": stop_reason if stop_reason in natural.STOP_REASONS else "other",
                     "cost": {key: outcome["cost"][key] for key in natural.COUNTERS}})
        if target["fixture_target_verified"]:
            cases.append(dict.fromkeys(writer._FACTS, True))
            provenance.append({"source_index": index, "witness_sha256": slot["witness_sha256"]})
    gate = result["source_gate"]
    if gate["status"] != "not_evaluable":
        require(all(row["status"] == "known" for row in rows) and len(cases) == gate["witness_scenarios"])
    require(read(root / "study-result.json") == raw and read(root / "study-journal.jsonl") == journal_raw
            and run._main_pid(launch["unit"]) == 0)
    public = {"id": "native-state-recovery-fixture-public-report-001", "plan_sha256": plan_sha,
              "terminal_result_sha256": result_sha, "source_denominator": 4, "scope": result["scope"],
              "official_scorer_calls": 0, "gate": gate, "rows": rows,
              "audited_closed_and_released": sum(row["status"] == "known" for row in rows)}
    return public, {"cases": cases, "private_provenance": provenance, "private_source_result_sha256": result_sha}


def report(plan_sha, result_sha):
    return _read_terminal(plan_sha, result_sha)[0]


def prepare(plan_sha, result_sha):
    """Expose only verified target facts after all four cases close."""
    public, evidence = _read_terminal(plan_sha, result_sha)
    run._require(public["gate"]["status"] == "passed" and public["audited_closed_and_released"] == 4)
    writer._validate_cases(evidence["cases"])
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--result-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.plan_sha256, args.result_sha256), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.stderr.write("native_state_recovery_fixture_report_invalid\n")
        raise SystemExit(2) from None
