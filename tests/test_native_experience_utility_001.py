"""Synthetic metadata only: no task instructions, model, worker or scorer."""

import copy
import random

import pytest

from scripts import native_experience_utility_001 as utility
from wll.protocol import fingerprint

IDENTITY = {"model_id": "model_b0", "scaffold_id": "scaffold_v1",
            "backend_id": "backend_v1", "budget_id": "budget_v1"}
BUNDLE = "experience_bundle_v1"
ORIGINS = ("source_scenario",)


def cohort(deltas, *, scenario_count=None):
    if scenario_count is None:
        scenario_count = len(deltas)
    expected, rows = [], []
    for index, delta in enumerate(deltas):
        triple = (f"task_{index:03}", f"scenario_{index % scenario_count:03}", 17)
        expected.append(triple)
        for arm, success in (("control", delta == -1), ("experience", delta == 1)):
            rows.append({"task_id": triple[0], "scenario_id": triple[1], "seed": triple[2],
                         "arm": arm, "split": "train", "status": "known", "closed": True,
                         "branch": "first", "replayed_steps": 0, "success": success,
                         "is_policy_episode": True, "worker_exit_code": 0, "score_status": "scored",
                         "outcome_unknown": False, "unknown_cost": False,
                         "identity": dict(IDENTITY),
                         "experience_bundle_id": BUNDLE if arm == "experience" else None})
    return expected, rows


def report(expected, rows, *, identity=IDENTITY, origins=ORIGINS, bundle=BUNDLE):
    return utility.evaluate_utility(expected, rows, identity=identity,
                                    source_scenario_ids=origins, experience_bundle_id=bundle)


def test_complete_positive_cohort_uses_both_prespecified_cluster_intervals():
    expected, rows = cohort([1] * 8)
    result = report(expected, rows)
    assert result["status"] == "complete" and result["passed"] is True
    assert result["planned_pairs"] == result["complete_pairs"] == 8
    stats = result["statistics"]
    assert stats["scenario_count"] == 8 and stats["n_pairs"] == 8
    assert stats["helped"] == 8 and stats["harmed"] == 0 and stats["ties"] == 0
    assert stats["delta"] == 1 and stats["cluster_bootstrap_ci95"] == [1, 1]
    assert stats["exact_cluster_ci95"] == [1, 1]
    assert stats["bootstrap_seed"] == 0 and result["thresholds"]["bootstrap_samples"] == 2000
    assert result["cohort_sha256"] == fingerprint(sorted(expected))
    assert result["source_scenarios_sha256"] == fingerprint(sorted(ORIGINS))
    assert all(result["checks"].values())


@pytest.mark.parametrize(("total", "passes"), [(20, True), (19, False)])
def test_observed_harm_ceiling_is_five_percent_inclusive(total, passes):
    expected, rows = cohort([1] * (total - 1) + [-1])
    result = report(expected, rows)
    assert result["statistics"]["harmed"] == 1
    assert result["checks"]["observed_harm_at_most_5_percent"] is passes
    assert result["passed"] is passes


def test_seeds_or_tasks_do_not_create_extra_independent_scenarios():
    expected, rows = cohort([1] * 14, scenario_count=7)
    result = report(expected, rows)
    assert result["statistics"]["n_pairs"] == 14
    assert result["statistics"]["scenario_count"] == 7
    assert result["passed"] is False and not result["checks"]["at_least_8_scenarios"]


def test_multiple_seeds_for_same_task_remain_one_scenario_cluster():
    expected, rows = cohort([1] * 8)
    task, scenario, _seed = expected[0]
    expected.append((task, scenario, 18))
    more = copy.deepcopy(rows[:2])
    for row in more:
        row["seed"] = 18
    result = report(expected, rows + more)
    assert result["status"] == "complete" and result["statistics"]["n_pairs"] == 9
    assert result["statistics"]["scenario_count"] == 8


def test_pair_and_row_input_order_do_not_change_statistics():
    expected, rows = cohort([1] * 4 + [0] * 5)
    before = report(expected, rows)
    random.Random(42).shuffle(expected)
    random.Random(19).shuffle(rows)
    assert report(expected, rows) == before


def test_exact_zero_lower_bound_does_not_pass():
    expected, rows = cohort([1] * 3 + [0] * 6)
    result = report(expected, rows)
    assert result["status"] == "complete" and result["statistics"]["exact_cluster_ci95"][0] == 0
    assert result["checks"]["exact_cluster_lower_positive"] is False
    assert result["passed"] is False


def test_missing_arm_keeps_planned_denominator_and_has_no_subset_estimate():
    expected, rows = cohort([1] * 8)
    result = report(expected, rows[:-1])
    assert result["status"] == "not_evaluable" and result["passed"] is None
    assert result["planned_pairs"] == 8 and result["planned_summaries"] == 16
    assert result["complete_pairs"] == 7 and result["missing_summaries"] == 1
    assert result["statistics"] is None and result["checks"] is None


def test_twenty_seven_pair_manifest_remains_the_denominator_if_one_arm_is_missing():
    expected, rows = cohort([1] * 12 + [0] * 15, scenario_count=9)
    result = report(expected, rows[:-1])
    assert result["status"] == "not_evaluable" and result["planned_pairs"] == 27
    assert result["planned_summaries"] == 54 and result["complete_pairs"] == 26
    assert result["missing_summaries"] == 1 and result["statistics"] is None


@pytest.mark.parametrize("status", ["unknown", "incomplete", "unstarted"])
def test_nonknown_result_never_becomes_zero_or_an_evaluable_subset(status):
    expected, rows = cohort([1] * 8)
    rows[0]["status"] = status
    rows[0]["success"] = False  # A stale bool does not erase unknown/incomplete status.
    result = report(expected, rows)
    assert result["status"] == "not_evaluable" and result["planned_pairs"] == 8
    assert result["complete_pairs"] == 7 and result["statistics"] is None
    assert result[status + "_summaries"] == 1


@pytest.mark.parametrize("field", ["outcome_unknown", "unknown_cost"])
def test_unknown_flags_block_even_an_observed_boolean_success(field):
    expected, rows = cohort([1] * 8)
    rows[0][field] = True
    result = report(expected, rows)
    assert result["status"] == "not_evaluable" and result["unknown_summaries"] == 1
    assert result["statistics"] is None


def test_unclosed_or_missing_success_is_incomplete():
    expected, rows = cohort([1] * 8)
    rows[0]["closed"] = False
    rows[2]["success"] = None
    result = report(expected, rows)
    assert result["status"] == "not_evaluable" and result["incomplete_summaries"] == 2
    assert result["planned_pairs"] == 8 and result["complete_pairs"] == 6


@pytest.mark.parametrize(("field", "value", "category"), [
    ("is_policy_episode", False, "incomplete"), ("worker_exit_code", 1, "incomplete"),
    ("worker_exit_code", None, "incomplete"), ("score_status", "not_scored", "incomplete"),
    ("score_status", "unknown", "unknown"),
])
def test_nongenuine_or_unscored_closed_rows_cannot_become_evaluable(field, value, category):
    expected, rows = cohort([1] * 8)
    rows[0][field] = value
    result = report(expected, rows)
    assert result["status"] == "not_evaluable" and result[category + "_summaries"] == 1
    assert result["statistics"] is None and result["planned_pairs"] == 8


@pytest.mark.parametrize("failure", ["duplicate_arm", "extra_task", "wrong_scenario", "wrong_seed",
    "experience_id", "control_has_experience", "model", "scaffold", "backend", "budget", "split",
    "recovery", "replay", "success_int", "flag_int", "worker_bool", "policy_string", "extra_payload"])
def test_invalid_matrix_or_identity_never_enters_statistics(failure):
    expected, rows = cohort([1] * 8)
    bad = copy.deepcopy(rows)
    if failure == "duplicate_arm":
        bad.append(copy.deepcopy(bad[0]))
    elif failure == "extra_task":
        bad[0]["task_id"] = "other_task"
    elif failure == "wrong_scenario":
        bad[0]["scenario_id"] = "other_scenario"
    elif failure == "wrong_seed":
        bad[0]["seed"] += 1
    elif failure == "experience_id":
        bad[1]["experience_bundle_id"] = "other_bundle"
    elif failure == "control_has_experience":
        bad[0]["experience_bundle_id"] = BUNDLE
    elif failure in {"model", "scaffold", "backend", "budget"}:
        bad[0]["identity"][failure + "_id"] = "other"
    elif failure == "split":
        bad[0]["split"] = "test"
    elif failure == "recovery":
        bad[0]["branch"] = "recovery"
    elif failure == "replay":
        bad[0]["replayed_steps"] = 1
    elif failure == "success_int":
        bad[0]["success"] = 0
    elif failure == "flag_int":
        bad[0]["outcome_unknown"] = 0
    elif failure == "worker_bool":
        bad[0]["worker_exit_code"] = False
    elif failure == "policy_string":
        bad[0]["is_policy_episode"] = "true"
    else:
        bad[0]["instruction"] = "private synthetic body"
    with pytest.raises(ValueError, match="native_experience_utility_invalid"):
        report(expected, bad)


def test_source_and_utility_scenarios_must_be_disjoint_even_with_missing_rows():
    expected, rows = cohort([1] * 8)
    with pytest.raises(ValueError, match="native_experience_utility_invalid"):
        report(expected, [], origins=(expected[0][1],))
    with pytest.raises(ValueError, match="native_experience_utility_invalid"):
        report(expected, rows, origins=())


@pytest.mark.parametrize("bad_manifest", [[], [("task", "scenario", True)],
    [("task", "scenario", 1), ("task", "scenario", 1)],
    [("task", "scenario_a", 1), ("task", "scenario_b", 1)]])
def test_frozen_manifest_is_nonempty_unique_and_task_scenario_unambiguous(bad_manifest):
    with pytest.raises(ValueError, match="native_experience_utility_invalid"):
        report(bad_manifest, [])


def test_exact_resampling_failure_is_not_a_pass_or_an_invented_interval(monkeypatch):
    expected, rows = cohort([1] * 8)

    def fail(_groups):
        raise ValueError("synthetic state limit")

    monkeypatch.setattr(utility, "exact_cluster_resampling", fail)
    result = report(expected, rows)
    assert result["status"] == "not_evaluable" and result["passed"] is None
    assert result["planned_pairs"] == result["complete_pairs"] == 8
    assert result["statistics"] is None and result["reason"] == "exact_resampling_unavailable"
