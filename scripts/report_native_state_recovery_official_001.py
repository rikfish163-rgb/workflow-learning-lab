"""Audit official score receipts and release evidence without rerunning any task."""

import argparse
from types import SimpleNamespace

from scripts import report_native_state_recovery_authored_study_001 as paired_audit
from scripts import report_native_state_recovery_utility_001 as score_audit
from scripts import run_native_state_recovery_official_001 as study
from scripts.native_appworld_worker_client import _decode, _encode, _same


def _scores(plan, bundle, result, jobs, read):
    study._require(len(result["summaries"]) == 54)

    def read_evidence(path, *, maximum):
        raw = read(path)
        study._require(len(raw) <= maximum)
        return raw

    runner = SimpleNamespace(_validate_bundle=study._validate_bundle, _core=study.frozen._core,
                             actor=study.frozen.actor, _experiment=study._experiment, _limits=study.frozen._limits)
    verified = score_audit._audit_scores(plan, result, bundle, read_evidence, {}, runner, study._queue())
    indexes = {task: (index, scene, seed) for index, (task, scene, seed) in enumerate(plan["expected_pairs"])}
    known = set()
    for row in result["summaries"]:
        index, scene, seed = indexes[row["task_id"]]
        arm = row["arm"]
        study._require(arm in ("control", "experience") and row["split"] == "test_normal"
                       and row["scenario_id"] == scene and row["seed"] == seed
                       and _same(row["identity"], plan["identity"])
                       and row["experience_bundle_id"] == (bundle["experience_bundle_id"] if arm == "experience" else None))
        if row["status"] == "known":
            known.add((index, arm))
    study._require(known == verified and result["writer_calls"] == result["training_jobs"] == 0)
    counters = dict.fromkeys(paired_audit.source_audit.COUNTERS, 0)
    for index, arm in verified:
        owner = jobs[index // 2]["owner"]
        raw = read(study.PRIVATE / f"unit-{index:02d}-{arm}-{owner}" / "execution-receipt.json")
        observed = _decode(raw)["actor_outcome"]["cost"]
        for key in counters:
            counters[key] += observed[key]
    return len(verified), counters


def report(plan_sha, result_sha):
    facade = SimpleNamespace(**vars(study))
    facade._summary = study.authored._summary
    launch_state = study._bind(paired_audit._launch_state, study=facade)
    outcome = study._bind(paired_audit._outcome, study=facade)
    audit = study._bind(paired_audit.report, study=facade, _launch_state=launch_state, _scores=_scores, _outcome=outcome)
    return audit(plan_sha, result_sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--result-sha256", required=True)
    args = parser.parse_args()
    try:
        print(_encode(report(args.plan_sha256, args.result_sha256)).decode())
    except Exception:
        raise SystemExit("official_panel_audit_failed") from None
