"""The public projection requires full independent terminal evidence."""

import pytest
from test_run_native_state_recovery_authored_study_001 import AuthoredRuntime

from scripts import report_native_state_recovery_authored_study_001 as audit
from scripts import run_native_state_recovery_authored_study_001 as study


def _terminal():
    plan = AuthoredRuntime().plan
    rows = study._summaries(plan)
    for row in rows:
        row.update(status="known", closed=True, is_policy_episode=True, worker_exit_code=0,
                   score_status="scored", success=row["arm"] == "experience")
    outcome = study._summary(plan, rows)
    result = {"status": "complete", "summaries": rows,
              "utility": {**outcome, "global_execution_and_costs_verified": True}}
    return plan, result


def test_complete_report_requires_exact_recorded_estimate():
    plan, result = _terminal()
    computed = audit._outcome(plan, result, "success", 54, 14)
    assert computed["complete_pairs"] == 27 and computed["passed"] is True
    result["utility"]["statistics"]["delta"] = .5
    with pytest.raises(ValueError):
        audit._outcome(plan, result, "success", 54, 14)


def test_without_last_release_all_scores_are_kept_without_an_effect():
    plan, result = _terminal()
    result["status"] = "unknown"
    result["utility"].update(status="not_evaluable", statistics=None, passed=None,
                             global_execution_and_costs_verified=False)
    computed = audit._outcome(plan, result, "failed", 54, 13)
    assert computed["status"] == "not_evaluable"
    assert computed["complete_pairs"] == 27 and computed["statistics"] is None


def test_known_score_count_cannot_exceed_independently_verified_receipts():
    plan, result = _terminal()
    with pytest.raises(ValueError):
        audit._outcome(plan, result, "success", 53, 14)
