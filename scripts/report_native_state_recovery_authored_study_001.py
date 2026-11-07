"""Independently audit closed authored scores and releases without re-running them."""

import argparse
import re
from dataclasses import asdict
from types import SimpleNamespace

from scripts import native_state_recovery_authored_runtime_001 as assets
from scripts import report_native_state_recovery_source_003 as source_audit
from scripts import report_paired_omission_utility_001 as terminal_audit
from scripts import run_native_state_recovery_authored_study_001 as study
from scripts.native_appworld_worker_client import WORKSPACE, _decode, _encode, _same
from scripts.verification_token_counter import _message_payload
from wll.protocol import fingerprint

MAX_BYTES = 64_000_000


def _require(condition):
    if not condition:
        raise ValueError("native_authored_study_report_unverified") from None


def _launch_state(launch, intent, sha):
    unit = study._coordinator_unit(sha)
    _require(launch["unit"] == unit and launch["plan_sha256"] == intent["plan_sha256"] == sha
             and intent["automatic_retry"] is False and type(intent["pid"]) is int and intent["pid"] > 0)
    expected = study._unit_command("pilot", unit, ["--run", "--plan-sha256", sha],
                                   root=WORKSPACE, log_path=study.PRIVATE / "utility-parent.log")
    actual = launch["argv"]
    prefix = expected.index("--") + 3
    end_expected, end_actual = expected.index(study.PYTHON), actual.index(study.PYTHON)
    _require(expected[:prefix] == actual[:prefix] and expected[end_expected:] == actual[end_actual:])
    optional = {"PATH", "HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"}
    _require([v for v in actual[prefix:end_actual] if v.split("=", 1)[0] not in optional]
             == [v for v in expected[prefix:end_expected] if v.split("=", 1)[0] not in optional])
    facade = SimpleNamespace(_get_module=lambda: SimpleNamespace(PARENT_SECONDS=study.frozen.PARENT_SECONDS))
    terminal = study._bind(terminal_audit._terminal, runner=facade)(launch, intent)
    _require(terminal in ("success", "failed"))
    return terminal


def _key(row):
    return row["pair_index"], row["arm"]


def _scores(plan, bundle, result, jobs, read):
    root = study.PRIVATE
    initial = plan["draft"]["initial_state_manifest_sha256"]
    scorer = assets._module("native_appworld_scorer", assets._source("native_appworld_scorer"))
    scorer.client = assets._client(root, initial)
    core = study.frozen._core()
    expected = study._summaries(plan)
    summaries = {(row["task_id"], row["arm"]): row for row in result["summaries"]}
    _require(len(summaries) == len(result["summaries"]) == len(expected)
             and all((row["task_id"], row["arm"]) in summaries for row in expected))
    executions = {_key(item): item for item in result["executions"]}
    _require(len(executions) == len(result["executions"]))
    verified, paired_metadata = set(), {}
    counters = dict.fromkeys(source_audit.COUNTERS, 0)
    for (index, arm), execution in executions.items():
        _require(type(index) is int and 0 <= index < 27 and arm in ("control", "experience"))
        task, scene, seed = plan["expected_pairs"][index]
        row = summaries[task, arm]
        _require(row["status"] == "known" and type(row["success"]) is bool
                 and row["scenario_id"] == scene and row["seed"] == seed and row["split"] == "authored"
                 and _same(row["identity"], plan["identity"])
                 and row["experience_bundle_id"] == (bundle["experience_bundle_id"] if arm == "experience" else None))
        experiment = f"pair-{index:02d}-{arm}-{jobs[index // 2]['owner']}"
        run_path = root / "runs" / experiment
        score_path = root / "scores" / experiment
        parent = assets.parent_unit(root, experiment)
        score_unit = assets.parent_unit(root, "scorer-" + experiment).replace("-parent-", "-score-")
        binding = execution["receipt_binding"]
        _require(binding["path"] == str(run_path / "execution-receipt.json"))
        paths = {"receipt": "execution-receipt.json", "controller": "controller.jsonl",
                 "status": "worker/status.json", "actions": "worker/actions.jsonl",
                 "transport": "worker/transport.jsonl"}
        evidence = {key: read(run_path / filename) for key, filename in paths.items()}
        receipt, manifest, snapshot = scorer._validate_receipt_and_logs(evidence, run_path, binding["sha256"])
        scorer._require_stopped(parent, manifest["unit_name"])
        _require(scorer._main_pid(score_unit) == 0 and receipt["task_id"] == task
                 and receipt["experiment_name"] == experiment and receipt["is_policy_episode"] is True
                 and receipt["actor_outcome"]["status"] == execution["actor_status"])
        scorer._verify_db(snapshot, manifest["files"])
        journal = scorer._journal(evidence["controller"])
        _, intent = scorer._one(journal, "controller_intent")
        _, metadata = scorer._one(journal, "metadata_result")
        metadata = metadata["reply"]["result"]
        metadata_sha = study._hash(_encode(metadata))
        _require(metadata_sha == execution["metadata_sha256"] == paired_metadata.setdefault(index, metadata_sha)
                 and intent["seed"] == seed and intent["review_hint_present"] is (arm == "experience")
                 and _same(intent["limits"], asdict(study.frozen._limits(plan["utility"]))))
        hint = bundle["insertion_text"] if arm == "experience" else None
        messages = study.frozen.actor.build_actor_messages(metadata, review_hint=hint)
        core.actor_evidence(receipt, journal, expected_context={
            "metadata_sha256": metadata_sha, "messages_sha256": _message_payload(messages)[1],
        }, review_hint=hint)
        scored = _decode(read(score_path / "scoring-result.json"))
        score_journal = scorer._journal(read(score_path / "scoring-journal.jsonl"))
        _require(_same(scored, {"status": "scored", "success": row["success"],
                               "receipt_sha256": binding["sha256"], "manifest_sha256": manifest["manifest_sha256"],
                               "is_policy_episode": True})
                 and _same(score_journal, [
                     {"event": "scoring_intent", "receipt_sha256": binding["sha256"], "parent_unit_name": parent},
                     {"event": "validated_before_evaluation", "manifest_sha256": manifest["manifest_sha256"]},
                     {"event": "scoring_stop", "status": "scored"},
                 ]))
        verified.add((task, arm))
        for key in counters:
            counters[key] += receipt["actor_outcome"]["cost"][key]
    _require(verified == {key for key, row in summaries.items() if row["status"] == "known"})
    return len(verified), counters


def _outcome(plan, result, parent_terminal, known, released):
    computed = study._summary(plan, result["summaries"])
    _require(computed["known_branches"] == known)
    stored = result["utility"]
    ready = (parent_terminal == "success" and result["status"] == "complete"
             and known == 54 and released == 14 and computed["status"] == "complete"
             and stored["global_execution_and_costs_verified"] is True)
    if ready:
        _require(stored["status"] == "complete" and stored["passed"] is computed["passed"]
                 and _same(stored["statistics"], computed["statistics"]))
    else:
        _require(stored["statistics"] is None)
        computed.update(status="not_evaluable", passed=None, statistics=None)
    return computed


def report(sha, result_sha):
    _require(re.fullmatch(r"[a-f0-9]{64}", sha) is not None
             and re.fullmatch(r"[a-f0-9]{64}", result_sha) is not None)
    plan, bundle = study._preflight(WORKSPACE, sha)
    runtime = study.Runtime(WORKSPACE)
    root = study.PRIVATE
    snapshots = {}

    def read(path):
        data = runtime.reader._read_private(path, maximum=MAX_BYTES)
        digest = study._hash(data)
        _require(snapshots.setdefault(path, digest) == digest)
        return data

    try:
        launch = _decode(read(root / "parent-launch-intent.json"))
        intent = _decode(read(root / "study-intent.json"))
        parent_terminal = _launch_state(launch, intent, sha)
        raw = read(root / "study-result.json")
        journal = study.frozen._core()._journal(read(root / "study-journal.jsonl"))
        result = _decode(raw)
        _require(study._hash(raw) == result_sha and result["id"] == study.ID and result["plan_sha256"] == sha
                 and result["automatic_retry"] is False
                 and study._hash(read(root / "reviewed-bundle.json")) == intent["reviewed_bundle_sha256"]
                 and _same(_decode(read(root / "reviewed-bundle.json")), bundle)
                 and _same(journal[-1], {"event": "study_stop", "report": result}))
        queue = study._queue()
        leases = result["costs"]["GPU_jobs"]
        jobs, verified = {}, []
        fences = {event["job_sha256"]: event for event in journal if event["event"] == "release_fence_started"}
        for index, item in enumerate(leases):
            job, release = item["job"], item["release"]
            _require(index < 14 and _same(job, queue.make_job(index, job["owner"], sha)))
            jobs[index] = job
            if release["verified"] is True and release["state"]["job"] is not None:
                verified.append(source_audit._release(queue, plan, item, fences[fingerprint(job)]))
        known, counters = _scores(plan, bundle, result, jobs, read)
        outcome = _outcome(plan, result, parent_terminal, known, len(verified))
        for path, digest in tuple(snapshots.items()):
            _require(study._hash(read(path)) == digest)
        _require(_launch_state(launch, intent, sha) == parent_terminal)
        return {"study": study.ID, "plan_sha256": sha, "result_sha256": result_sha,
                "parent_terminal": parent_terminal, "audited_scores": known,
                "audited_finished_leases": len(verified),
                "audited_GPU_seconds": sum(item["lease_seconds"] for item in verified),
                "audited_actor_cost_subtotal": counters, "utility": outcome}
    finally:
        runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--result-sha256", required=True)
    args = parser.parse_args()
    try:
        print(_encode(report(args.plan_sha256, args.result_sha256)).decode())
    except Exception:
        raise SystemExit("native_authored_study_report_unverified") from None
