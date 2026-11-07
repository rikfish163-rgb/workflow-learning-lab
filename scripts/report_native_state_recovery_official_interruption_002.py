"""Describe the stopped official run from its saved receipts and terminal audit."""

import os
from collections import Counter

from scripts import run_native_state_recovery_official_002 as entry
from scripts.native_appworld_scorer import _main_pid
from scripts.native_appworld_worker_client import _decode, _encode, _same
from scripts.native_source_registration import public_read


def report():
    study = entry.bindings()
    terminal = _decode(public_read(study.WORKSPACE, entry.REPORT))
    runtime = study.Runtime(study.WORKSPACE)
    try:
        raw = runtime.read(study.PRIVATE / "study-result.json")
        study._require(study._hash(raw) == terminal["result_sha256"])
        result = _decode(raw)
        sha = terminal["plan_sha256"]
        plan = study._load_private(study.WORKSPACE, sha)
        launch = _decode(runtime.read(study.PRIVATE / "parent-launch-intent.json"))
        study._require(result["plan_sha256"] == sha and launch["unit"] == study._coordinator_unit(sha)
                       and _main_pid(launch["unit"]) == 0 and terminal["parent_terminal"] == "failed"
                       and terminal["utility"]["statistics"] is None)
        unknown = [row for row in result["summaries"] if row["status"] == "unknown"]
        study._require(len(unknown) == 1)
        row = unknown[0]
        index = next(i for i, pair in enumerate(plan["expected_pairs"]) if pair[0] == row["task_id"])
        job = result["costs"]["GPU_jobs"][index // 2]["job"]
        arm, owner = row["arm"], job["owner"]
        run = study.PRIVATE / f"unit-{index:02d}-{arm}-{owner}"
        receipt = _decode(runtime.read(run / "execution-receipt.json"))
        journal = study.frozen._core()._journal(runtime.read(run / "controller.jsonl"))
        counts = Counter(event["event"] for event in journal)
        actor = receipt["actor_outcome"]
        study._require(receipt["task_id"] == row["task_id"] and receipt["status"] == "unknown"
                       and receipt["experiment_name"] == study._experiment(index, arm, owner)
                       and receipt["snapshot"] is None and actor["stop_reason"] == "generation_failure"
                       and actor["unknown_cost"] is True
                       and counts["generation_intent"] == counts["generation_result"] + 1
                       and counts["generation_failure"] == counts["controller_failure"] == 1)
        parent = f"wll-native-source-utility-{index:02d}-{arm}-{owner}.service"
        study._require(_main_pid(parent) == 0
                       and runtime.cleanup_worker(run, row["task_id"], receipt["experiment_name"])["verified"] is True)
        state = study._queue().queue_operation("status", job)
        study._require(state["released"] is state["admission_cancelled"] is True and state["model_pids"] == [])
        score_path = study.PRIVATE / f"score-{index:02d}-{arm}-{owner}"
        study._require(not score_path.exists())
        costs = {key: sum(item.get(key) or 0 for item in result["costs"]["episodes"])
                 for key in ("prompt_tokens", "completion_tokens", "generations", "action_attempts", "api_invocations")}
        costs["unknown_cost"] = True
        study._require(_same(result["summaries"], _decode(runtime.read(study.PRIVATE / "study-result.json"))["summaries"]))
        return {
            "scope": "saved_official_interruption_evidence", "plan_sha256": sha,
            "result_sha256": terminal["result_sha256"], "states": dict(Counter(r["status"] for r in result["summaries"])),
            "planned_pairs": 27, "complete_pairs": terminal["utility"]["complete_pairs"],
            "audited_scores": terminal["audited_scores"], "audited_finished_leases": terminal["audited_finished_leases"],
            "interrupted_pair_index": index, "interrupted_arm": arm,
            "generation_intents_in_interrupted_branch": counts["generation_intent"],
            "generation_returns_in_interrupted_branch": counts["generation_result"],
            "observed_stop_reason": actor["stop_reason"], "snapshot_saved": False,
            "independent_score_present": False, "cost_lower_bounds_including_interrupted_branch": costs,
            "project_gpu_release_verified": True, "effect_status": "not_evaluable",
            "model_calls_by_audit": 0, "scorer_calls_by_audit": 0,
        }
    finally:
        runtime.close()


if __name__ == "__main__":
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU_audit_required")
        print(_encode(report()).decode())
    except Exception:
        raise SystemExit("official_interruption_audit_failed") from None
