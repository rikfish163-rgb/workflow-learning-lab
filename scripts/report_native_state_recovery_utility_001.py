"""Export a terminal V27 result after checking the closed actor and scorer files."""

import hashlib
from dataclasses import asdict
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scripts import native_appworld_scorer as scorer
from scripts import report_native_state_recovery_source_003 as source_report
from scripts import run_native_state_recovery_utility_001 as utility
from scripts.native_appworld_worker_client import _decode, _encode, _same
from scripts.native_source_registration import public_read
from scripts.verification_token_counter import _message_payload

ROOT = Path(__file__).resolve().parents[1]
BASE = "scripts/report_paired_omission_utility_001.py"
BASE_SHA256 = "bb9e03effc0009fc6452a2b4701eb5839bb6b95943b7972eb29291839b9e0eae"
ERROR = "native_state_recovery_utility_report_invalid"


def _require(condition):
    if not condition:
        raise ValueError(ERROR)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _audit_scores(plan, report, bundle, reader, snapshots, runner, queue):
    memory = runner._validate_bundle(plan, bundle)
    private = Path(plan["private_run_root"])
    jobs = {row["job"]["batch_index"]: row["job"] for row in report["costs"]["GPU_jobs"]}
    executions = {(row["pair_index"], row["arm"]): row for row in report["executions"]}
    summaries = {(row["task_id"], row["arm"]): row for row in report["summaries"]}
    _require(len(executions) == len(report["executions"]) and len(summaries) == len(report["summaries"]))
    verified = set()

    def read(path):
        raw = reader(path, maximum=64_000_000)
        snapshots[path] = _hash(raw)
        return raw

    for (index, arm), execution in executions.items():
        _require(type(index) is int and 0 <= index < len(plan["expected_pairs"]) and arm in ("control", "experience"))
        task, _scene, seed = plan["expected_pairs"][index]
        summary = summaries[task, arm]
        if summary["status"] != "known":
            continue
        job = jobs[index // 2]
        owner = job["owner"]
        _require(_same(job, queue.make_job(index // 2, owner, report["plan_sha256"])))
        parent = f"wll-native-source-utility-{index:02d}-{arm}-{owner}.service"
        score_unit = f"wll-native-source-utility-{index:02d}-score-{arm}-{owner}.service"
        path = private / f"unit-{index:02d}-{arm}-{owner}"
        score_path = private / f"score-{index:02d}-{arm}-{owner}"
        files = {"receipt": "execution-receipt.json", "controller": "controller.jsonl", "status": "worker/status.json",
                 "actions": "worker/actions.jsonl", "transport": "worker/transport.jsonl"}
        raw = {key: read(path / name) for key, name in files.items()}
        binding = execution["receipt_binding"]
        _require(binding["path"] == str(path / files["receipt"]))
        receipt, manifest, _db_path = scorer._validate_receipt_and_logs(raw, path, binding["sha256"])
        journal = runner._core()._journal(raw["controller"])
        intent = runner._core()._one(journal, "controller_intent")
        metadata = runner._core()._one(journal, "metadata_result")["reply"]["result"]
        hint = memory if arm == "experience" else None
        messages = runner.actor.build_actor_messages(metadata, review_hint=hint)
        context = {"metadata_sha256": _hash(_encode(metadata)), "messages_sha256": _message_payload(messages)[1]}
        _require(receipt["task_id"] == task and receipt["experiment_name"] == runner._experiment(index, arm, owner)
                 and receipt["is_policy_episode"] is True and context["metadata_sha256"] == execution["metadata_sha256"]
                 and intent["seed"] == seed and intent["review_hint_present"] is (arm == "experience")
                 and _same(intent["limits"], asdict(runner._limits(plan["utility"]))) and receipt["replay_outcome"] is None
                 and receipt["actor_outcome"]["status"] == execution["actor_status"])
        runner._core().actor_evidence(receipt, journal, expected_context=context, review_hint=hint)
        scorer._require_stopped(parent, manifest["unit_name"])
        _require(scorer._main_pid(score_unit) == 0 and type(summary["success"]) is bool)
        scored = _decode(read(score_path / "scoring-result.json"))
        score_journal = scorer._journal(read(score_path / "scoring-journal.jsonl"))
        _require(_same(scored, {"status": "scored", "success": summary["success"], "receipt_sha256": binding["sha256"],
                               "manifest_sha256": manifest["manifest_sha256"], "is_policy_episode": True})
                 and _same(score_journal, [{"event": "scoring_intent", "receipt_sha256": binding["sha256"],
                                            "parent_unit_name": parent},
                                           {"event": "validated_before_evaluation", "manifest_sha256": manifest["manifest_sha256"]},
                                           {"event": "scoring_stop", "status": "scored"}]))
        verified.add((index, arm))
    return verified


@cache
def _adapter(lineage=utility.NATURAL):
    registration, queue, runner = utility.bindings(lineage)
    source_plan = utility._source_plan(ROOT, lineage)
    origins, pairs, _batches, _orders = utility._cohort(source_plan, lineage)
    raw = public_read(ROOT, BASE)
    _require(_hash(raw) == BASE_SHA256)
    text = raw.decode("utf-8")
    edits = (
        ("paired-omission-utility-public-report-001", lineage.slug.removesuffix("-001") + "-public-report-001"),
        ("Controller-recorded official scores; scorer receipts are not independently re-scored.",
         "Audited scores are reconciled with closed actor and scorer files. No task is re-scored."),
        ("    bundle_bound = False\n", "    bundle_bound = False\n    verified_scores = set()\n"),
        ("    terminal = _terminal(launch, intent)\n", "    terminal = _terminal(launch, intent)\n"
         "    if bundle_bound and terminal in {'success', 'failed'}:\n"
         "        verified_scores = _audit_scores(plan, report_value, bundle, reader, snapshots, runner._get_module(), queue)\n"),
        ("        and computed['status'] == 'complete' and complete_costs and type(saved) is dict",
         "        and execution_verified and computed['status'] == 'complete' and complete_costs and type(saved) is dict"),
        ("bundle_verified=True, boot_id=None):", "bundle_verified=True, boot_id=None, execution_verified=True):"),
        ("boot_id=intent['boot_id'])", "boot_id=intent['boot_id'], execution_verified=len(verified_scores) == 54)\n"
         "    result['audited_official_score_count'] = len(verified_scores)"),
    )
    for before, after in edits:
        _require(text.count(before) == 1)
        text = text.replace(before, after)
    module = ModuleType("scripts._native_state_recovery_utility_report_001")
    module.__file__ = str(ROOT / BASE)
    exec(compile(text, module.__file__, "exec", dont_inherit=True), vars(module))
    module.ERROR, module.queue = ERROR, queue
    module.runner = SimpleNamespace(_get_module=lambda: runner)
    module.registration = SimpleNamespace(**{**vars(registration), "SOURCE_SCENARIOS": tuple(origins),
        "UTILITY_TASK_IDS": tuple(row[0] for row in pairs), "BUNDLE_PREFIX": lineage.review.BUNDLE_PREFIX})
    module._audit_scores = _audit_scores
    module._verify_backend = lambda plan, backend: runner._services(ROOT)[1]._backend(plan, backend)
    original = module._durable_checker

    def durable(snapshots, plan):
        check = original(snapshots, plan)

        def verify(job, release, fence, service):
            source_report._release(queue, plan, {"job": job, "release": release}, fence)
            return check(job, release, fence, service)

        return verify

    module._durable_checker = durable
    return module


def report(plan_sha, result_sha, *, lineage=utility.NATURAL):
    return _adapter(lineage).report(plan_sha, result_sha)


if __name__ == "__main__":
    raise SystemExit(_adapter().main())
